






































































































































































































import math
import sys
import os
from os import path
import shutil
import random
import numpy as np
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.cuda.amp as amp
from PIL import Image
import gc
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from util.metrics import calculate_precision_at_k_and_iou_metrics
from util.utils import create_output_dir, create_checkpoint_dir, flatten_temporal_batch_dims
from datasets import build_dataset, Collator
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import misc as utils
from models.GroundingDINO.utils import compute_mask
from models import build_model
from rich.progress import track
import json
import torch.nn.functional as F
from misc import nested_tensor_from_videos_list

from datasets.ref_youtube_vos.ref_youtube_vos_dataset_dynamic import (
    ReferYTVOSDynamicSegmentsDataset, dynamic_collate_for_branch
)
from models.GroundingDINO.temporal_branch import TemporalBranch

from models.ot_prompt.losses import (
    pair_align_loss,
    mmd_rbf,
    temporal_tv,
    ot_cost_matrix_label_only,
    sinkhorn_cost_from_matrix,
    grade_center_contrast_loss,
    bidirectional_info_nce_loss,
    supervised_cross_modal_contrastive_loss,
)
IM_MEAN = [0.2663, 0.2736, 0.2876]
IM_STD  = [0.1378, 0.1416, 0.1483]
def _to_B1HW(x, name="tensor"):
    if x is None:
        return None
    if x.dim() == 5:
        if x.size(1) == 1 and x.size(2) == 1:
            x = x[:, 0, 0]
        elif x.size(2) == 1:
            x = x[:, -1]
        else:
            raise ValueError(f"{name} unexpected 5D shape: {tuple(x.shape)}")
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"{name} expect [B,1,H,W], got {tuple(x.shape)}")
    return x
def _cfg_val(key, default=None):
    v = tb_conf.get(key, default)
    if isinstance(v, dict) and 'value' in v:
        return v['value']
    try:
        if hasattr(v, 'value'):
            return v.value
    except Exception:
        pass
    return v
def _to_B1(x, name="tensor"):
    if x is None:
        return None
    if x.dim() == 3 and x.size(-1) == 1:
        x = x.squeeze(-1)
    if x.dim() != 2 or x.size(1) != 1:
        raise ValueError(f"{name} expect [B,1], got {tuple(x.shape)}")
    return x

def _all_finite(*xs):
    for x in xs:
        if isinstance(x, torch.Tensor) and not torch.isfinite(x).all():
            return False
    return True


def _gradients_finite(parameters):
    """Return False when any existing gradient contains NaN/Inf."""
    for p in parameters:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True

def _gather_last_valid_time(x, valid_time_mask):
    """Gather per-sample last valid temporal item from B,T,... tensors."""
    if valid_time_mask is None:
        return x[:, -1]
    valid_time_mask = valid_time_mask.to(device=x.device, dtype=torch.bool)
    last_idx = valid_time_mask.long().sum(dim=1).clamp(min=1) - 1
    gather_shape = [x.size(0), 1] + [1] * (x.dim() - 2)
    expand_shape = [x.size(0), 1] + list(x.shape[2:])
    idx = last_idx.view(*gather_shape).expand(*expand_shape)
    return x.gather(1, idx).squeeze(1)

class TextBank:
    def __init__(self, capacity=8192):
        self.capacity = capacity
        self.storage = None
        self.ptr = 0
        self.n = 0
    def add(self, x: torch.Tensor):
        if x is None:
            return
        x = x.detach().to('cpu')
        if x.numel() == 0:
            return
        B, d = x.shape
        if self.storage is None:
            self.storage = torch.zeros(self.capacity, d, dtype=x.dtype)
        if B >= self.capacity:
            self.storage[...] = x[-self.capacity:]
            self.n = self.capacity
            self.ptr = 0
            return
        end = self.ptr + B
        if end <= self.capacity:
            self.storage[self.ptr:end] = x
        else:
            first = self.capacity - self.ptr
            self.storage[self.ptr:] = x[:first]
            self.storage[:end - self.capacity] = x[first:]
        self.ptr = end % self.capacity
        self.n = min(self.n + B, self.capacity)
    def sample(self, k: int, device: torch.device):
        if self.storage is None or self.n == 0:
            return None
        k = min(k, self.n)
        idx = torch.randint(0, self.n, (k,))
        return self.storage[idx].to(device)


class Trainer:
    def __init__(self, config, process_id, device_id, num_processes):
        self.config = config

        self.world_size = num_processes
        self.distributed = num_processes > 1
        self.process_id = process_id
        self.is_main_process = process_id == 0
        self.device = init_process_group_and_set_device(num_processes, process_id, device_id, config)
        self.accum_iter = int(getattr(config, 'accumulate_iter', 1))
        seed = config.seed + config.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model, criterion, postprocessor = build_model(config)

        model_without_ddp = model.module if hasattr(model, "module") else model
        print(f"Model classification output shape: {model_without_ddp.class_embed[0].num_classes if hasattr(model_without_ddp.class_embed[0],'num_classes') else 'unknown'} (should be 4)")

        criterion.to(self.device)
        model.to(self.device)

        if config.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device_id],
                output_device=device_id,
                find_unused_parameters=True,
                static_graph=False
            )
            model_without_ddp = model.module
        else:
            model_without_ddp = model

        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessor

        use_simple_embed = getattr(config, 'use_simple_embed', False)
        if use_simple_embed:
            if self.is_main_process:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("!!! WARNING: Training in Simple Embedding Baseline Mode !!!")
                print("!!! OT Module is DISABLED. Only using Grade Embedding. !!!")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            if hasattr(self.model, 'module'):
                self.model.module.use_simple_embed = True
            else:
                self.model.use_simple_embed = True


        self.d_model = getattr(model_without_ddp, 'hidden_dim', None)
        self.t_dim = getattr(model_without_ddp, 't_dim', self.d_model)
        if self.t_dim is None:
            self.t_dim = self.d_model
        if self.d_model != self.t_dim:
            self.proj_text = torch.nn.Linear(self.d_model, self.t_dim, bias=False).to(self.device)
            proj_params = list(self.proj_text.parameters())
        else:
            self.proj_text = torch.nn.Identity().to(self.device)
            proj_params = []

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        self.dataset_name = config.dataset_name

        if self.dataset_name in ('a2d_sentences', 'jhmdb_sentences'):
            self.evaluate = self.evaluate_a2d_sentences
        else:
            self.evaluate = None

        dataset_train = build_dataset(image_set='train', **vars(config))
        if self.distributed:
            self.sampler_train = DistributedSampler(dataset_train, num_replicas=config.world_size, rank=config.rank,
                                                    shuffle=True, seed=config.seed, drop_last=False)
        else:
            self.sampler_train = None
        self.data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=self.sampler_train,
                                            collate_fn=Collator(), num_workers=config.num_workers,
                                            pin_memory=True, shuffle=self.sampler_train is None, drop_last=True)

        if self.evaluate is not None:
            dataset_val = build_dataset(image_set='test',** vars(config))
            if self.distributed:
                sampler_val = DistributedSampler(dataset_val, num_replicas=config.world_size, rank=config.rank, shuffle=False)
            else:
                sampler_val = None
            eval_batch_size = config.eval_batch_size
            self.data_loader_val = DataLoader(dataset_val, batch_size=eval_batch_size, sampler=sampler_val, drop_last=False,
                                              collate_fn=Collator(), num_workers=config.num_workers,
                                              pin_memory=True)

        self.enable_branch = bool(getattr(config, 'TemporalBranch', {}).get('enable', False))
        self.branch = None
        self.branch_iter = None
        self._branch_step = 0
        self.branch_print_freq = 0
        self.branch_steps_per_main = 0
        self.branch_mem_cache = {}
        branch_params = []
        if self.enable_branch:
            tb_conf = config.TemporalBranch
            def _cfg_val(key, default=None):
                v = tb_conf.get(key, default)
                if isinstance(v, dict) and 'value' in v:
                    return v['value']
                try:
                    if hasattr(v, 'value'):
                        return v.value
                except Exception:
                    pass
                return v

            branch_dataset = ReferYTVOSDynamicSegmentsDataset(
                subset_type='train',
                dataset_path=config.dataset_path,
                supervise_start=bool(_cfg_val('supervise_start', False)),
                max_gap=_cfg_val('max_gap', None),
                corrupt_prob=float(_cfg_val('corrupt_prob', 0.0)),
            )


            if self.distributed:
                self.sampler_branch = DistributedSampler(branch_dataset, num_replicas=config.world_size, rank=config.rank,
                                                         shuffle=True, seed=config.seed, drop_last=False)
            else:
                self.sampler_branch = None
            self.data_loader_branch = DataLoader(
                branch_dataset,
                batch_size=tb_conf.get('batch_size', config.batch_size),
                sampler=self.sampler_branch,
                collate_fn=dynamic_collate_for_branch,
                num_workers=config.num_workers,
                pin_memory=True,
                shuffle=self.sampler_branch is None,
                drop_last=True
            )
            if self.is_main_process:
                try:
                    print(f"[TemporalBranch] dynamic segments (total): {len(branch_dataset)}")
                    print(f"[TemporalBranch] steps per epoch (branch loader): {len(self.data_loader_branch)}")
                except Exception as e:
                    print(f"[TemporalBranch] dataset size print error: {e}")

            self.branch = TemporalBranch(
                feat_ch=int(_cfg_val('feat_ch', 256)),
                hidden_ch=int(_cfg_val('gru_hidden_dim', 256)),
                use_exist_head=bool(_cfg_val('use_exist_head', False)),
                shared_encoder=None,
                freeze_encoder=bool(_cfg_val('freeze_encoder', True)),
                return_logits=True,
                iterative_guidance=bool(_cfg_val('iterative_guidance', True)),
                gn_groups=int(_cfg_val('gn_groups', 32)),

                learnable_alpha=bool(_cfg_val('learnable_alpha', True)),
                alpha_channelwise=bool(_cfg_val('alpha_channelwise', True)),
                alpha_init=float(_cfg_val('alpha_init', 1.0)),
                use_guide_blend=bool(_cfg_val('use_guide_blend', True)),
                learnable_beta=bool(_cfg_val('learnable_beta', True)),
                beta_init=float(_cfg_val('beta_init', 0.5)),

                use_decoder_aspp=bool(_cfg_val('use_decoder_aspp', True)),
                use_decoder_depthwise=bool(_cfg_val('use_decoder_depthwise', True)),
                use_decoder_skip=bool(_cfg_val('use_decoder_skip', True)),

                dual_scale=bool(_cfg_val('dual_scale', True)),
                hidden_ch_low=int(_cfg_val('hidden_ch_low', 128)) if _cfg_val('hidden_ch_low', None) is not None else None,
                gamma_init=float(_cfg_val('gamma_init', 0.5)),

                decode_head_first=bool(_cfg_val('decode_head_first', False)),
                gate_init=0.5
            ).to(self.device)

            self.branch_steps_per_main = int(_cfg_val('steps_per_main', 1))
            self.branch_print_freq = int(_cfg_val('print_freq', 50))
            self.branch_corrupt_prob = float(_cfg_val('corrupt_prob', 0.0))
            self.branch_per_channel_norm_guide = bool(_cfg_val('per_channel_norm_guide', False))
            self.branch_head_sup_weight = float(_cfg_val('head_sup_weight', 0.0))
            self.branch_head_sup_epochs = int(_cfg_val('head_sup_epochs', 0))

            branch_params = list(p for p in self.branch.parameters() if p.requires_grad)
            print(f'[TemporalBranch] params: {sum(p.numel() for p in branch_params)}')

        main_params = [p for n, p in model_without_ddp.named_parameters() if 'class_proj' not in n]
        cls_params = [p for n, p in model_without_ddp.named_parameters() if 'class_proj' in n]

        main_lr = config.lr
        cls_lr = getattr(config, 'cls_lr', main_lr)

        param_dicts = main_params + proj_params

        if self.enable_branch and len(branch_params) > 0:
            br_lr_factor = float(self.config.TemporalBranch.get('lr_factor', 1.0))
            self.optimizer = torch.optim.AdamW([
                {'params': param_dicts, 'lr': main_lr, 'weight_decay': config.weight_decay},
                {'params': cls_params, 'lr': cls_lr, 'weight_decay': config.weight_decay},
                {'params': branch_params, 'lr': main_lr * br_lr_factor, 'weight_decay': config.weight_decay},
            ])
        else:
            self.optimizer = torch.optim.AdamW([
                {'params': param_dicts, 'lr': main_lr, 'weight_decay': config.weight_decay},
                {'params': cls_params, 'lr': cls_lr, 'weight_decay': config.weight_decay},
            ])
        for i, group in enumerate(self.optimizer.param_groups):
            print(f"Param group {i}: lr={group['lr']}, num_params={len(group['params'])}")




        self.num_batches_per_epoch = len(self.data_loader_train)
        if self.dataset_name == 'a2d_sentences':
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=config.lr_drop, gamma=0.2, verbose=True)
        else:
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=config.lr_drop, gamma=0.1)
        self.grad_scaler = amp.GradScaler(enabled=config.enable_amp)
        self.max_norm = config.clip_max_norm

        self.w_ot   = getattr(config, 'w_ot', 1.0)
        self.align_objective = str(getattr(config, 'align_objective', 'ot')).lower()
        self.w_contrast = float(getattr(config, 'w_contrast', 0.0))
        self.contrast_temperature = float(getattr(config, 'contrast_temperature', 0.07))
        self.w_pair = getattr(config, 'w_pair', 0.2)
        self.w_mmd  = getattr(config, 'w_mmd', 0.5)
        self.w_tv   = getattr(config, 'w_tv', 0.0)
        self.w_grade_contrast = getattr(config, 'w_grade_contrast', 0.0)
        self.grade_contrast_temperature = getattr(config, 'grade_contrast_temperature', 0.2)
        self.ot_label_cost = float(getattr(config, 'ot_label_cost', 0.5))
        self.ot_eps   = getattr(config, 'ot_eps', 0.5)
        self.ot_iters = getattr(config, 'ot_iters', 50)
        self.mmd_sigma    = getattr(config, 'mmd_sigma', 1.0)
        self.mmd_ref_num  = getattr(config, 'mmd_ref_num', 512)
        self.mmd_bank_size= getattr(config, 'mmd_bank_size', 8192)
        self.text_bank = TextBank(capacity=self.mmd_bank_size)

        self.branch_lambda_sup  = float(getattr(config.TemporalBranch, 'lambda_sup', 1.0)) if self.enable_branch else 0.0
        self.branch_lambda_dist = float(getattr(config.TemporalBranch, 'lambda_dist', 0.0)) if self.enable_branch else 0.0
        self.branch_dist_type   = str(getattr(config.TemporalBranch, 'dist_type', 'mse')).lower().strip()

        self.teacher_topk       = int(getattr(config.TemporalBranch, 'teacher_topk', 3))
        self.mem_override_after = int(getattr(config.TemporalBranch, 'mem_override_after', 2))
        self.mem_override_conf  = float(getattr(config.TemporalBranch, 'mem_override_conf', 0.7))

        self.dist_start_epoch   = int(getattr(config.TemporalBranch, 'dist_start_epoch', 10))
        self.dist_warmup_epochs = int(getattr(config.TemporalBranch, 'dist_warmup_epochs', 2))

        if self.is_main_process:
            self.output_dir_path = create_output_dir(config)
            self.checkpoint_dir_path = create_checkpoint_dir(self.output_dir_path)
            if config.wandb_mode == 'online':
                is_resume = config.running_mode == 'resume_train'
                wandb.init(project='RefVOS', config=config, mode=config.wandb_mode, name=config.version, resume=is_resume)
            print(config)
        else:
            self.output_dir_path = ''
        if self.distributed:
            output_dir_sync_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(output_dir_sync_list, self.output_dir_path)
            self.output_dir_path = output_dir_sync_list[0]

        self.total_epochs = config.epochs
        self.epoch = 0
        self.iteration = 0
        self.best_mAP = 0
        self.best_loss = math.inf

        if self.config.pretrained_weights is not None:
            print("============================================>")
            print(f"Load pretrained weights from {self.config.pretrained_weights} ...")
            checkpoint = torch.load(self.config.pretrained_weights, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("============================================>")

        self.embedding_cache = {
            'hat_T': [],
            'E_T': [],
            'hat_T_type': [],
            'E_T_type': [],
            'hat_T_class': [],
            'E_T_class': [],
        }

        self.embed_sample_max = 1000
        self.embed_save_interval = 1
        self.embed_save_dir = os.path.join(self.output_dir_path, 'embedding_data')
        if self.is_main_process:
            os.makedirs(self.embed_save_dir, exist_ok=True)

    def check_for_nan_or_inf(self, tensor, name="tensor"):
        print(f"Checking for NaN or Inf in {name}...")
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}: {tensor}")
        if torch.isinf(tensor).any():
            print(f"Inf detected in {name}: {tensor}")
        print(f"{name} stats -> min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
        print(f"Shape of {name}: {tensor.shape}")

    @staticmethod
    def _sanitize_prob(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x):
            x = x.float()
        x = torch.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
        return x.clamp_(0.0, 1.0)

    @staticmethod
    def _safe_resize_prob(p: torch.Tensor, size_hw: tuple) -> torch.Tensor:
        p = Trainer._sanitize_prob(p)
        if p.shape[-2:] != size_hw:
            p = torch.nn.functional.interpolate(p, size=size_hw, mode='bilinear', align_corners=False)
        return Trainer._sanitize_prob(p)

    @torch.no_grad()
    def _backbone_predict_mask_on_imgs(self, imgs_b3hw: torch.Tensor, topk: int = 3, return_logit: bool = False):
        self.model.eval()
        B, _, H, W = imgs_b3hw.shape
        vids = imgs_b3hw.unsqueeze(1)
        samples = nested_tensor_from_videos_list(vids, size_divisibility=32).to(self.device)
        captions = None
        targets_stub = [{"size": torch.as_tensor([H, W], device=self.device)} for _ in range(B)]
        outputs = self.model(samples, captions, targets_stub)
        self.model.train()

        pred_logits = outputs["pred_logits"]
        pred_masks  = outputs["pred_masks"]
        scores = pred_logits[:, 0, :, 0].sigmoid()
        k = min(max(1, topk), scores.shape[1])
        vals, idxs = torch.topk(scores, k=k, dim=1)

        probs = []
        for b in range(B):
            mb = pred_masks[b, 0, idxs[b]]
            pb = torch.sigmoid(mb)
            wb = vals[b] / (vals[b].sum() + 1e-6)
            probs.append((pb * wb[:, None, None]).sum(dim=0))
        prob = torch.stack(probs, dim=0)[:, None, ...]

        prob_up = torch.nn.functional.interpolate(prob, size=(H, W), mode='bilinear', align_corners=False)
        prob_up = self._sanitize_prob(prob_up)

        if return_logit:
            eps = 1e-6
            logit = torch.log(prob_up.clamp(eps, 1 - eps) / (1 - prob_up.clamp(eps, 1 - eps)))
            return logit.detach()
        return prob_up.detach()

    @staticmethod
    def _entropy_confidence(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = Trainer._sanitize_prob(p)
        p = p.clamp(eps, 1 - eps)
        h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        h = h.mean(dim=(1, 2, 3), keepdim=True)
        return 1 - h

    @staticmethod
    def _fuse_probs(b_main: torch.Tensor, b_branch: torch.Tensor) -> torch.Tensor:
        b_main = Trainer._sanitize_prob(b_main)
        b_branch = Trainer._sanitize_prob(b_branch)
        c_m = Trainer._entropy_confidence(b_main)
        c_t = Trainer._entropy_confidence(b_branch)
        alpha = c_m / (c_m + c_t + 1e-6)
        out = alpha * b_main + (1 - alpha) * b_branch
        return Trainer._sanitize_prob(out)

    def _dist_effective_weight(self) -> float:
        if not self.enable_branch or self.branch_lambda_dist <= 0:
            return 0.0
        if self.epoch < self.dist_start_epoch:
            return 0.0
        if self.dist_warmup_epochs <= 0:
            return float(self.branch_lambda_dist)
        k = (self.epoch - self.dist_start_epoch + 1) / float(self.dist_warmup_epochs)
        k = max(0.0, min(1.0, k))
        return float(self.branch_lambda_dist * k)

    def save_embedding_data(self, epoch):
        if len(self.embedding_cache['hat_T']) == 0:
            return
        hat_T_all = np.vstack(self.embedding_cache['hat_T'])
        E_T_all   = np.vstack(self.embedding_cache['E_T'])
        hat_T_type = np.array(self.embedding_cache['hat_T_type'])
        E_T_type   = np.array(self.embedding_cache['E_T_type'])
        hat_T_class = np.concatenate(self.embedding_cache['hat_T_class'])
        E_T_class   = np.concatenate(self.embedding_cache['E_T_class'])

        save_path = os.path.join(self.embed_save_dir, f'epoch_{epoch}.npz')
        np.savez_compressed(
            save_path,
            hat_T=hat_T_all,
            E_T=E_T_all,
            hat_T_type=hat_T_type,
            E_T_type=E_T_type,
            hat_T_class=hat_T_class,
            E_T_class=E_T_class
        )
        print(f"[embedding] saved epoch {epoch} data to {save_path}")

        self.embedding_cache = {
            'hat_T': [], 'E_T': [],
            'hat_T_type': [], 'E_T_type': [],
            'hat_T_class': [], 'E_T_class': []
        }

    def train(self):
        print("Training started...")
        batch_ob, pg = utils.get_batch_observer(self.epoch, self.total_epochs, len(self.data_loader_train),
                                                disable=(not self.is_main_process) or self.config.debug)
        if self.enable_branch:
            self.branch.train()
            self.branch_iter = iter(self.data_loader_branch)
            self._branch_step = 0

        for self.epoch in range(self.epoch, self.total_epochs):
            self.model.train()
            self.criterion.train()
            if self.enable_branch:
                self.branch.train()
                self._branch_step = 0
            metric_logger = utils.MetricLogger(delimiter="  ")
            if self.enable_branch and self.branch_head_sup_weight > 0.0:
                metric_logger.meters.setdefault('head_init_logit_mean', utils.SmoothedValue(window_size=20, fmt='{value:.6f}'))
                metric_logger.meters.setdefault('head_init_prob_mean', utils.SmoothedValue(window_size=20, fmt='{value:.6f}'))
                metric_logger.meters.setdefault('head_sup_loss', utils.SmoothedValue(window_size=20, fmt='{value:.6f}'))

            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            if self.distributed:
                self.sampler_train.set_epoch(self.epoch)
                if self.enable_branch:
                    self.sampler_branch.set_epoch(self.epoch)
            total_epoch_loss = 0
            loss_sums_dict = {k: 0 for k in self.criterion.weight_dict.keys()}
            extra_keys = ['loss_ot', 'loss_contrast', 'loss_pair', 'loss_mmd', 'loss_tv', 'loss_grade_contrast', 'loss_branch_sup', 'loss_branch_dist']
            extra_sums = {k: 0.0 for k in extra_keys}

            batch_ob.start()
            for i, batch_dict in enumerate(self.data_loader_train):
                samples = batch_dict['samples'].to(self.device)
                targets = self.to_device(batch_dict['targets'], self.device)
                text_queries = batch_dict['text_queries']

                with amp.autocast(enabled=self.config.enable_amp):
                    outputs = self.model(samples, text_queries, targets)

                    loss_dict = self.criterion(outputs, targets)
                    weight_dict = self.criterion.weight_dict
                    skip_prefixes = ('loss_pair', 'loss_mmd', 'loss_tv')
                    base_keys = [k for k in loss_dict.keys()
                                 if (k in weight_dict) and (not k.startswith(skip_prefixes))]
                    base_losses = sum(loss_dict[k] * weight_dict[k] for k in base_keys)

                    hat_T = outputs.get('hat_T_global', None)
                    text_cls = outputs.get('text_cls', None)
                    hat_T_seq = outputs.get('hat_T_seq', None)

                    loss_ot = torch.tensor(0.0, device=self.device)
                    loss_contrast = torch.tensor(0.0, device=self.device)
                    loss_pair = torch.tensor(0.0, device=self.device)
                    loss_mmd = torch.tensor(0.0, device=self.device)
                    loss_tv = torch.tensor(0.0, device=self.device)
                    loss_grade_contrast = torch.tensor(0.0, device=self.device)

                    embedding_finite = isinstance(hat_T, torch.Tensor) and _all_finite(hat_T)
                    if text_cls is not None:
                        embedding_finite = embedding_finite and isinstance(text_cls, torch.Tensor) and _all_finite(text_cls)
                    E_T = None
                    if hat_T is not None and embedding_finite:
                        hat_T = F.normalize(torch.nan_to_num(hat_T.float(), nan=0.0, posinf=1.0, neginf=-1.0), dim=-1, eps=1e-6)
                        if text_cls is not None:
                            E_T = self.proj_text(text_cls).float()
                            E_T = F.normalize(torch.nan_to_num(E_T, nan=0.0, posinf=1.0, neginf=-1.0), dim=-1, eps=1e-6)

                        if E_T is not None and self.align_objective == 'ot' and self.w_ot > 0:
                            if isinstance(targets, dict) and "classes" in targets:
                                classes = targets["classes"].to(hat_T.device)
                            elif isinstance(targets, list):
                                classes = torch.stack([t['classes'][0] for t in targets]).to(hat_T.device)
                            else:
                                raise RuntimeError("Targets must contain class labels")
                            model_for_flags = self.model.module if hasattr(self.model, "module") else self.model
                            lambda_g = 0.0 if getattr(model_for_flags, "grade_free_prompt", False) else self.ot_label_cost
                            cost_matrix = ot_cost_matrix_label_only(hat_T, E_T, classes, lambda_g=lambda_g)

                            loss_ot = sinkhorn_cost_from_matrix(
                                cost_matrix, eps=self.ot_eps, iters=self.ot_iters
                            )

                        if E_T is not None and self.align_objective in ('info_nce', 'infonce') and self.w_contrast > 0:
                            loss_contrast = bidirectional_info_nce_loss(
                                hat_T,
                                E_T,
                                temperature=self.contrast_temperature,
                            )

                        if E_T is not None and self.align_objective in ('supcon', 'supervised_contrastive') and self.w_contrast > 0:
                            if isinstance(targets, dict) and "classes" in targets:
                                classes_for_contrast = targets["classes"].to(hat_T.device).long()
                            elif isinstance(targets, list):
                                classes_for_contrast = torch.stack([t['classes'][0] for t in targets]).to(hat_T.device).long()
                            else:
                                raise RuntimeError("Targets must contain class labels")
                            loss_contrast = supervised_cross_modal_contrastive_loss(
                                hat_T,
                                E_T,
                                classes_for_contrast,
                                temperature=self.contrast_temperature,
                            )

                        if E_T is not None and self.w_pair > 0:
                            loss_pair = pair_align_loss(hat_T, E_T, w_cos=1.0, w_l2=0.0)
                        if E_T is not None:
                            self.text_bank.add(E_T)

                        if self.is_main_process and E_T is not None:
                            current_total = len(self.embedding_cache['hat_T']) * self.config.batch_size
                            if current_total < self.embed_sample_max:
                                batch_hat_T = hat_T.detach().cpu().numpy()
                                batch_E_T = E_T.detach().cpu().numpy()
                                batch_size = batch_hat_T.shape[0]
                                if isinstance(targets, dict) and "classes" in targets:
                                    batch_labels = targets["classes"].detach().cpu().numpy()
                                elif isinstance(targets, list):
                                    batch_labels = torch.stack([t['classes'][0] for t in targets]).detach().cpu().numpy()
                                else:
                                    raise RuntimeError("Targets must contain class labels")
                                self.embedding_cache['hat_T'].append(batch_hat_T)
                                self.embedding_cache['E_T'].append(batch_E_T)
                                self.embedding_cache.setdefault('hat_T_class', []).append(batch_labels)
                                self.embedding_cache.setdefault('E_T_class', []).append(batch_labels)
                                self.embedding_cache.setdefault('hat_T_type', []).extend([0]*batch_size)
                                self.embedding_cache.setdefault('E_T_type', []).extend([1]*batch_size)

                    if hat_T is not None and self.w_mmd > 0 and embedding_finite:
                        ref = self.text_bank.sample(self.mmd_ref_num, self.device)
                        if ref is not None:
                            loss_mmd = mmd_rbf(hat_T, ref, sigma=self.mmd_sigma)

                    if hat_T_seq is not None and self.w_tv > 0 and _all_finite(hat_T_seq):
                        loss_tv = temporal_tv(hat_T_seq, flow_mask=None)

                    if hat_T is not None and E_T is not None and self.w_grade_contrast > 0 and embedding_finite:
                        if isinstance(targets, dict) and "classes" in targets:
                            classes_for_grade = targets["classes"].to(hat_T.device).long()
                        elif isinstance(targets, list):
                            classes_for_grade = torch.stack([t['classes'][0] for t in targets]).to(hat_T.device).long()
                        else:
                            raise RuntimeError("Targets must contain class labels")
                        centers = []
                        text_for_centers = E_T.detach()
                        fallback_center = text_for_centers.mean(dim=0)
                        for grade_id in range(4):
                            grade_mask = classes_for_grade == grade_id
                            if grade_mask.any():
                                centers.append(text_for_centers[grade_mask].mean(dim=0))
                            else:
                                centers.append(fallback_center)
                        centers = torch.stack(centers, dim=0)
                        loss_grade_contrast = grade_center_contrast_loss(
                            hat_T,
                            centers,
                            classes_for_grade,
                            temperature=self.grade_contrast_temperature,
                        )

                    loss_branch_sup = torch.tensor(0.0, device=self.device)
                    loss_branch_dist = torch.tensor(0.0, device=self.device)
                    effective_lambda_dist = self._dist_effective_weight()

                    if self.enable_branch and (self.branch_lambda_sup > 0 or effective_lambda_dist > 0):
                        acc_sup = torch.tensor(0.0, device=self.device)
                        acc_dist = torch.tensor(0.0, device=self.device)
                        valid_branch_steps = 0
                        steps_this_main = max(1, self.branch_steps_per_main)
                        for _ in range(steps_this_main):
                            try:
                                branch_batch = next(self.branch_iter)
                            except StopIteration:
                                self.branch_iter = iter(self.data_loader_branch)
                                branch_batch = next(self.branch_iter)

                            mem_img_raw = branch_batch["mem_img"].to(self.device)
                            frames      = branch_batch["frames"].to(self.device)
                            valid_time_mask = branch_batch.get("valid_time_mask", None)
                            if valid_time_mask is not None:
                                valid_time_mask = valid_time_mask.to(self.device)
                            mask_start  = branch_batch["mask_start"].to(self.device)
                            mask_end    = branch_batch["mask_end"].to(self.device)
                            vids        = branch_batch["video_id"]
                            start_ids   = branch_batch["start_fid"]
                            end_ids     = branch_batch["end_fid"]
                            frames_end  = _gather_last_valid_time(frames, valid_time_mask)
                            text_queries = branch_batch["text_queries"]
                            classes =   branch_batch["classes"].to(self.device)


                            from models.dino_util.misc import nested_tensor_from_tensor_list
                            mem_img_raw_seq = [img for img in mem_img_raw]
                            frames_end_seq = [img for img in frames_end]
                            samples_start = nested_tensor_from_tensor_list(mem_img_raw_seq)
                            samples_end = nested_tensor_from_tensor_list(frames_end_seq)

                            with torch.no_grad():
                                outputs_start = self.model(samples_start, text_queries, targets={"classes": classes})
                                outputs_end = self.model(samples_end, text_queries, targets={"classes": classes})

                            scores_start = outputs_start["pred_logits"][:, 0, :, 0].sigmoid()
                            vals_start, idxs_start = torch.topk(scores_start, k=1, dim=1)
                            main_pred_start = outputs_start["pred_masks"][:, 0]
                            main_pred_start = main_pred_start.gather(
                                1, idxs_start.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, main_pred_start.shape[2], main_pred_start.shape[3])
                            )
                            main_pred_start_prob = torch.sigmoid(main_pred_start).clamp(0, 1)
                            main_pred_start_prob = F.interpolate(main_pred_start_prob, size=mem_img_raw.shape[2:], mode='bilinear', align_corners=False)
                            main_pred_start_prob = main_pred_start_prob.float().detach().clone()
                            del outputs_start, samples_start, main_pred_start

                            scores_end = outputs_end["pred_logits"][:, 0, :, 0].sigmoid()
                            vals_end, idxs_end = torch.topk(scores_end, k=1, dim=1)
                            main_pred_end = outputs_end["pred_masks"][:, 0]
                            main_pred_end = main_pred_end.gather(
                                1, idxs_end.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, main_pred_end.shape[2], main_pred_end.shape[3])
                            )
                            main_pred_end = F.interpolate(main_pred_end, size=mem_img_raw.shape[2:], mode='bilinear', align_corners=False)
                            main_pred_end = main_pred_end.float().detach().clone()
                            main_pred_end_prob = torch.sigmoid(main_pred_end).clamp(0, 1)
                            del outputs_end, samples_end

                            mask_start_orig = mask_start.clone()

                            if main_pred_start_prob.dtype != torch.float32:
                                main_pred_start_prob = main_pred_start_prob.float()
                            if main_pred_start_prob.max() > 1.0:
                                main_pred_start_prob = main_pred_start_prob / 255.0
                            main_pred_start_prob = main_pred_start_prob.clamp(0.0, 1.0)

                            if (self.branch_corrupt_prob > 0.0) and (random.random() < 1.0):
                                try:
                                    mask_start_train = self.branch.corrupt_mask_for_training(main_pred_start_prob, prob_erase=self.branch_corrupt_prob)
                                except Exception:
                                    mask_start_train = _corrupt_mask_tensor(main_pred_start_prob[0], prob_erase=self.branch_corrupt_prob)
                                    mask_start_train = mask_start_train.unsqueeze(0)
                            else:
                                mask_start_train = main_pred_start_prob

                            if (self.epoch >= self.mem_override_after) and (len(vids) == mem_img_raw.size(0)):
                                mem_masks_over = mask_start_train.clone()
                                Hc, Wc = mem_masks_over.shape[-2:]
                                for bi, (vid, s_id) in enumerate(zip(vids, start_ids)):
                                    key = (vid, s_id)
                                    if key in self.branch_mem_cache:
                                        fused_mem = self.branch_mem_cache[key]
                                        if isinstance(fused_mem, torch.Tensor):
                                            fm = fused_mem.to(self.device)
                                            conf = self._entropy_confidence(fm).item()
                                            if conf >= self.mem_override_conf:
                                                fm = self._safe_resize_prob(fm, (Hc, Wc))
                                                mem_masks_over[bi:bi+1] = fm
                                mask_start_train = mem_masks_over


                            head_sup_loss = torch.tensor(0.0, device=self.device)
                            do_head_sup = (self.branch_head_sup_weight > 0.0) and (self.epoch < self.branch_head_sup_epochs)
                            gt_mask_for_head = mask_start_orig.float().to(self.device)
                            if do_head_sup:
                                with torch.no_grad():
                                    mem_feat_for_head = self.branch.encoder(mem_img_raw)
                                guide0 = self.branch._build_guide(mem_feat_for_head, mask_start_train, per_channel_norm=self.branch_per_channel_norm_guide)
                                h_init = self.branch.h_init(guide0)
                                if getattr(self.branch, "use_decoder_skip", False):
                                    out_init = self.branch.decoder(h_init, mem_feat_for_head)
                                else:
                                    out_init = self.branch.decoder(h_init)
                                head_sup_loss = torch.nn.functional.binary_cross_entropy_with_logits(out_init.float(), gt_mask_for_head)
                                head_sup_loss = head_sup_loss * float(self.branch_head_sup_weight)
                                try:
                                    with torch.no_grad():
                                        prob_init = torch.sigmoid(out_init).clamp(0,1)
                                        mean_logit = float(out_init.mean().item())
                                        mean_prob = float(prob_init.mean().item())
                                    metric_logger.update(head_init_logit_mean=mean_logit, head_init_prob_mean=mean_prob, head_sup_loss=float(head_sup_loss))
                                    if self.is_main_process and getattr(wandb, "log", None) and (self.config.wandb_mode == 'online'):
                                        wandb.log({"head_init/head_sup_loss": float(head_sup_loss),
                                                    "head_init/logit_mean": mean_logit,
                                                    "head_init/prob_mean": mean_prob,
                                                    "epoch": self.epoch}, commit=False)
                                except Exception:
                                    pass


                            try:
                                preds_all, _ = self.branch.forward_segment(
                                    mem_img_raw, mask_start_train, frames, valid_time_mask=valid_time_mask,
                                    last_only=False, detach_bptt=True,
                                    per_channel_norm_guide=self.branch_per_channel_norm_guide,
                                    return_endpoints=True,
                                )
                            except torch.cuda.OutOfMemoryError:
                                if self.is_main_process:
                                    bad_desc = ", ".join(f"{v}:{s}->{e}" for v, s, e in zip(vids, start_ids, end_ids))
                                    print(f"[branch][skip-oom] epoch={self.epoch} step={i} samples={bad_desc}")
                                torch.cuda.empty_cache()
                                continue
                            except RuntimeError as exc:
                                if "out of memory" not in str(exc).lower():
                                    raise
                                if self.is_main_process:
                                    bad_desc = ", ".join(f"{v}:{s}->{e}" for v, s, e in zip(vids, start_ids, end_ids))
                                    print(f"[branch][skip-oom] epoch={self.epoch} step={i} samples={bad_desc}")
                                torch.cuda.empty_cache()
                                continue


                            logits_start = _to_B1HW(preds_all[:, 0], "preds_start")
                            logits_end   = _to_B1HW(preds_all[:, -1], "preds_end")


                            mask_end_gt    = _to_B1HW(mask_end, "mask_end").float().clamp(0., 1.)

                            logits_end = F.interpolate(logits_end, size=mem_img_raw.shape[2:], mode='bilinear', align_corners=False)
                            prob_start = torch.sigmoid(logits_start).clamp(0, 1)
                            prob_end   = torch.sigmoid(logits_end).clamp(0, 1)

                            gate = torch.sigmoid(self.branch.gate)
                            fused_pred_end = gate * main_pred_end_prob + (1 - gate) * prob_end
                            fused_logits_end = gate * main_pred_end + (1 - gate) * logits_end

                            loss_dice_start = self._dice_loss(prob_start, gt_mask_for_head)
                            loss_bce_start  = torch.nn.functional.binary_cross_entropy_with_logits(logits_start.float(), gt_mask_for_head.float())
                            loss_dice_end_branch = self._dice_loss(prob_end, mask_end_gt)
                            loss_bce_end_branch = torch.nn.functional.binary_cross_entropy_with_logits(
                                logits_end.float(), mask_end_gt.float()
                            )
                            loss_dice_end_fused = self._dice_loss(fused_pred_end, mask_end_gt)
                            loss_bce_end_fused = torch.nn.functional.binary_cross_entropy_with_logits(
                                fused_logits_end.float(), mask_end_gt.float()
                            )
                            loss_dice_end = 0.75 * loss_dice_end_branch + 0.25 * loss_dice_end_fused
                            loss_bce_end = 0.75 * loss_bce_end_branch + 0.25 * loss_bce_end_fused
                            step_sup = (loss_dice_start + 0.5*loss_bce_start + loss_dice_end + 0.5*loss_bce_end) + 0.1*head_sup_loss
                            if not _all_finite(step_sup, loss_dice_start, loss_bce_start,
                                               loss_dice_end_branch, loss_bce_end_branch,
                                               loss_dice_end_fused, loss_bce_end_fused,
                                               head_sup_loss):
                                if self.is_main_process:
                                    bad_desc = ", ".join(f"{v}:{s}->{e}" for v, s, e in zip(vids, start_ids, end_ids))
                                    print(f"[branch][skip-nonfinite] epoch={self.epoch} step={i} samples={bad_desc}")
                                    print(f"  loss_dice_start={loss_dice_start.detach()} loss_bce_start={loss_bce_start.detach()} "
                                          f"loss_dice_end={loss_dice_end.detach()} loss_bce_end={loss_bce_end.detach()} "
                                          f"head_sup_loss={head_sup_loss.detach()} gate={gate.detach()}")
                                continue
                            acc_sup = acc_sup + step_sup
                            valid_branch_steps += 1
                            print(f"loss_dice_start: {loss_dice_start.item():.4f}, loss_bce_start: {loss_bce_start.item():.4f}, "
                                    f"loss_dice_end: {loss_dice_end.item():.4f}, loss_bce_end: {loss_bce_end.item():.4f}, "
                                    f"head_sup_loss: {head_sup_loss.item():.4f}, gate: {gate.item():.4f}")

                            self._branch_step += 1

                        loss_branch_sup = acc_sup / float(max(1, valid_branch_steps))

                        if effective_lambda_dist > 0:
                            loss_branch_dist = acc_dist / float(steps_this_main)
                        else:
                            loss_branch_dist = torch.tensor(0.0, device=self.device)

                    extra_loss = (self.w_ot * loss_ot
                                  + self.w_contrast * loss_contrast
                                  + self.w_pair * loss_pair
                                  + self.w_mmd * loss_mmd
                                  + self.w_tv * loss_tv
                                  + self.w_grade_contrast * loss_grade_contrast
                                  + self.branch_lambda_sup * loss_branch_sup
                                  + effective_lambda_dist * loss_branch_dist)

                    losses = base_losses + extra_loss

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items()
                                            if (k in weight_dict) and (not k.startswith(skip_prefixes))}
                base_total_reduced = sum(loss_dict_reduced_scaled.values()).item()

                extra_dict = {
                    'loss_ot': self.w_ot * loss_ot.detach(),
                    'loss_contrast': self.w_contrast * loss_contrast.detach(),
                    'loss_pair': self.w_pair * loss_pair.detach(),
                    'loss_mmd': self.w_mmd * loss_mmd.detach(),
                    'loss_tv': self.w_tv * loss_tv.detach(),
                    'loss_grade_contrast': self.w_grade_contrast * loss_grade_contrast.detach(),
                    'loss_branch_sup': self.branch_lambda_sup * loss_branch_sup.detach(),
                    'loss_branch_dist': (effective_lambda_dist * loss_branch_dist.detach()),
                }
                extra_dict_reduced = utils.reduce_dict(extra_dict)
                extra_total_reduced = sum(v for v in extra_dict_reduced.values()).item()

                total_loss_reduced = base_total_reduced + extra_total_reduced
                if not math.isfinite(total_loss_reduced):
                    if self.is_main_process:
                        print("[finite-guard] Loss is {}; skipping epoch={} step={}".format(
                            total_loss_reduced, self.epoch, i))
                        print(loss_dict_reduced, extra_dict_reduced)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.grad_scaler.scale(losses).backward()
                if ((i + 1) % self.accum_iter) == 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    grad_finite = _gradients_finite(
                        list(self.model.parameters()) +
                        (list(self.branch.parameters()) if self.enable_branch and self.branch is not None else [])
                    )
                    bad_grad = torch.tensor(0 if grad_finite else 1, device=self.device, dtype=torch.int32)
                    if self.distributed:
                        dist.all_reduce(bad_grad, op=dist.ReduceOp.MAX)
                    if int(bad_grad.item()) != 0:
                        if self.is_main_process:
                            print(f"[finite-guard] non-finite gradient; skipping epoch={self.epoch} step={i}")
                        self.optimizer.zero_grad(set_to_none=True)
                        current_scale = self.grad_scaler.get_scale()
                        self.grad_scaler.update(new_scale=max(current_scale / 2.0, 1.0))
                        continue
                    if self.max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, error_if_nonfinite=False)
                        if self.enable_branch and (self.branch is not None):
                            torch.nn.utils.clip_grad_norm_(self.branch.parameters(), self.max_norm, error_if_nonfinite=False)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad()

                metric_logger.update(loss=total_loss_reduced,** loss_dict_reduced_scaled)
                for k in extra_keys:
                    metric_logger.meters.setdefault(k, utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
                    metric_logger.update(**{k: extra_dict_reduced.get(k, torch.tensor(0.0)).item()})

                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
                batch_ob.update(pg, advance=1, epoch=self.epoch,
                                loss=metric_logger.meters['loss'].global_avg,
                                cls=metric_logger.meters.get('loss_ce', utils.SmoothedValue()).global_avg,
                                bbox=metric_logger.meters.get('loss_bbox', utils.SmoothedValue()).global_avg,
                                giou=metric_logger.meters.get('loss_giou', utils.SmoothedValue()).global_avg,
                                mask=metric_logger.meters.get('loss_mask', utils.SmoothedValue()).global_avg,
                                dice=metric_logger.meters.get('loss_dice', utils.SmoothedValue()).global_avg,
                                proj=metric_logger.meters.get('loss_proj', utils.SmoothedValue()).global_avg,
                                video_cls=metric_logger.meters.get('loss_video_cls', utils.SmoothedValue()).global_avg,
                                loss_ot=metric_logger.meters['loss_ot'].global_avg,
                                loss_pair=metric_logger.meters['loss_pair'].global_avg,
                                loss_mmd=metric_logger.meters['loss_mmd'].global_avg,
                                loss_tv=metric_logger.meters['loss_tv'].global_avg,
                                loss_branch_sup=metric_logger.meters['loss_branch_sup'].global_avg,
                                )

                if self.is_main_process and self.enable_branch and (self.branch_lambda_sup > 0 or effective_lambda_dist > 0) and self.branch_print_freq > 0:
                    if i % self.branch_print_freq == 0:
                        planned_branch_steps = len(self.data_loader_train) * max(1, self.branch_steps_per_main)
                        print(f"[branch] step={self._branch_step}/{planned_branch_steps}  "
                              f"loss_branch_sup={ (self.branch_lambda_sup * loss_branch_sup).detach():.4f}  "
                              f"loss_branch_dist={ (effective_lambda_dist * loss_branch_dist).detach():.4f}")

                self.iteration += 1
                total_epoch_loss += total_loss_reduced
                for k in loss_sums_dict.keys():
                    if (k in loss_dict_reduced_scaled):
                        loss_sums_dict[k] += loss_dict_reduced_scaled.get(k, torch.zeros(1)).item()
                for k in extra_keys:
                    extra_sums[k] += extra_dict_reduced.get(k, torch.tensor(0.0)).item()

            batch_ob.reset(pg)
            batch_ob.stop()
            metric_logger.synchronize_between_processes()
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': self.epoch}

            self.lr_scheduler.step()

            if self.is_main_process and (self.epoch + 1) % self.embed_save_interval == 0:
                self.save_embedding_data(self.epoch + 1)

            self.clear_memory()
            if self.epoch >= 0 and self.evaluate is not None:
                eval_metrics = self.evaluate()
                self.model.train()
                for key, value in eval_metrics.items():
                    log_stats['evaluate' + key] = value

            if self.is_main_process:
                epoch_score = total_epoch_loss / max(1, len(self.data_loader_train))
                if self.dataset_name == 'a2d_sentences':
                    mAP_score = eval_metrics.get('mAP 0.5:0.95') if self.evaluate is not None else None
                    if mAP_score is not None:
                        self.save_checkpoint(mAP_score)
                else:
                    self.save_checkpoint(epoch_score)

                with open(os.path.join(self.output_dir_path, 'log.txt'), 'a')as f:
                    f.write(json.dumps(log_stats) + "\n")

            self.clear_memory()
            if self.distributed:
                dist.barrier()

    def _dice_loss(self, pred, target, eps: float = 1e-6):
        if pred.dim() == 5:
            if pred.size(1) == 1 and pred.size(2) == 1:
                pred = pred[:, 0, 0]
            elif pred.size(2) == 1:
                pred = pred.mean(dim=1)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)

        if target.dim() == 5:
            if target.size(1) == 1 and target.size(2) == 1:
                target = target[:, 0, 0]
            elif target.size(2) == 1:
                target = target.mean(dim=1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        inter = (pred * target).sum(dim=(2, 3))
        den   = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
        dice  = 1.0 - (2.0 * inter + eps) / den
        return dice.mean()

    @torch.no_grad()
    def evaluate_a2d_sentences(self):
        self.model.eval()
        predictions = []
        for batch_dict in track(self.data_loader_val, description="Evaluating",
                                disable=(not self.is_main_process) or self.config.debug):
            samples = batch_dict['samples'].to(self.device)
            targets = self.to_device(batch_dict['targets'], self.device)
            text_queries = batch_dict['text_queries']

            force_grade = None
            mode = getattr(self.config, 'force_label_mode', 'none')

            if mode != 'none':
                gt_labels = torch.stack([t['classes'][0] for t in targets]).to(self.device)
                num_classes = 4

                if mode == 'gt':
                    force_grade = gt_labels
                elif mode == 'wrong':
                    force_grade = (gt_labels + 1) % num_classes
                elif mode == 'random':
                    force_grade = torch.randint(0, num_classes, gt_labels.shape, device=self.device)

            model_ptr = self.model.module if self.distributed else self.model

            outputs = model_ptr.infer(samples, captions=text_queries, targets=targets, force_grade=force_grade)

            outputs.pop('aux_outputs', None)

            processed_outputs = self.postprocessor(outputs, resized_padded_sample_size=samples.tensors.shape[-2:],
                                                   resized_sample_sizes=[t['size'] for t in targets],
                                                   orig_sample_sizes=[t['orig_size'] for t in targets])
            image_ids = [t['image_id'] for t in targets]

            for p, image_id in zip(processed_outputs, image_ids):
                for s, m in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,
                                        'segmentation': m,
                                        'score': s.item()})

        if self.distributed:
            gathered_pred_lists = utils.all_gather(predictions)
            predictions = [p for p_list in gathered_pred_lists for p in p_list]
        eval_metrics = {}
        if self.is_main_process:
            coco_gt = COCO(self.config.dataset_coco_gt_format_path)
            coco_pred = coco_gt.loadRes(predictions)
            coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
            coco_eval.params.useCats = 0
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
            ap_metrics = coco_eval.stats
            eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
            if self.config.calculate_precision_and_iou_metrics:
                precision_at_k, overall_iou, mean_iou = (
                    calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred))
                eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
                eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
            print(eval_metrics)
        if self.distributed:
            dist.barrier()
        return eval_metrics

    def to_device(self, sample, device=None):
        device = device or self.device
        if isinstance(sample, torch.Tensor):
            sample = sample.to(device)
        elif isinstance(sample, tuple) or isinstance(sample, list):
            sample = [self.to_device(s, device) for s in sample]
        elif isinstance(sample, dict):
            sample = {k: self.to_device(v, device) for k, v in sample.items()}
        return sample

    def load_checkpoint(self, checkpoint_path, total_epoch=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint['epoch'] + 1
        if total_epoch is None:
            self.total_epochs = checkpoint['total_epochs']
        else:
            self.total_epochs = total_epoch
        if self.dataset_name == 'a2d_sentences':
            self.best_mAP = checkpoint['best_mAP']
        else:
            self.best_loss = checkpoint['best_loss']

        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model

        ckpt_state = checkpoint['model_state_dict']
        model_state = model_without_ddp.state_dict()
        filtered_state = {k: v for k, v in ckpt_state.items() if k in model_state and v.shape == model_state[k].shape}

        skipped = [k for k in ckpt_state if k not in filtered_state]
        if self.process_id == 0 and skipped:
            print(f"[Resume] Skipped keys due to shape mismatch: {skipped}")

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(filtered_state, strict=False)
        if self.process_id == 0 and missing_keys:
            print(f"[Resume] Missing keys: {missing_keys}")
        if self.process_id == 0 and unexpected_keys:
            print(f"[Resume] Unexpected keys: {unexpected_keys}")

        if self.enable_branch and ('branch_state_dict' in checkpoint):
            if self.branch is None:
                self.branch = TemporalBranch(return_logits=True).to(self.device)
            self.branch.load_state_dict(checkpoint['branch_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    def save_checkpoint(self, epoch_score):
        if not self.is_main_process:
            return
        is_best = False
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint_dict = {
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'train_loss': float(epoch_score),
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }
        if self.enable_branch and (self.branch is not None):
            checkpoint_dict['branch_state_dict'] = self.branch.state_dict()

        if self.dataset_name == 'a2d_sentences':
            is_best_mAP = epoch_score > self.best_mAP
            if is_best_mAP:
                self.best_mAP = epoch_score
                is_best = True
            checkpoint_dict['best_mAP'] = self.best_mAP
        else:
            is_best_loss = epoch_score < self.best_loss
            if is_best_loss:
                self.best_loss = epoch_score
                is_best = True
            checkpoint_dict['best_loss'] = self.best_loss
        filename = self.get_checkpoint_filename()
        tmp_filename = filename + f".tmp.{os.getpid()}"
        try:
            torch.save(checkpoint_dict, tmp_filename)
            os.replace(tmp_filename, filename)
        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
        print(f'saved checkpoint: {filename}')
        if is_best:
            best_filename = self.get_checkpoint_filename(is_best=True)
            best_tmp = best_filename + f".tmp.{os.getpid()}"
            try:
                shutil.copyfile(filename, best_tmp)
                os.replace(best_tmp, best_filename)
            finally:
                if os.path.exists(best_tmp):
                    os.remove(best_tmp)

    def get_checkpoint_filename(self, is_best=False):
        if is_best:
            basename = 'best_mAP' if self.dataset_name == 'a2d_sentences' else 'best_train_loss'
        else:
            basename = f'{self.epoch:02d}'
        return os.path.join(self.checkpoint_dir_path, f'{basename}.pth.tar')

    def remove_extra_checkpoints(self):
        filenames = sorted(os.listdir(self.checkpoint_dir_path))
        max_num_checkpoints = 5
        num_files_to_remove = max(0, len(filenames) - max_num_checkpoints)
        for filename in filenames[:num_files_to_remove]:
            os.remove(os.path.join(self.checkpoint_dir_path, filename))

    def clear_memory(self):
        compute_mask.cache_clear()
        gc.collect()
        torch.cuda.empty_cache()


def pre_trained_model_to_finetune(checkpoint, args):
    checkpoint = checkpoint['model_state_dict']
    num_layers = args.DeformTransformer['dec_layers'] + 1 if args.DeformTransformer['two_stage'] else args.DeformTransformer['dec_layers']
    for l in range(num_layers):
        del checkpoint["class_embed.{}.weight".format(l)]
        del checkpoint["class_embed.{}.bias".format(l)]
    return checkpoint


def init_process_group_and_set_device(world_size, process_id, device_id, config):
    config.world_size = world_size
    config.rank = process_id
    if device_id != 'cpu':
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    config.device = device
    if world_size > 1:
        config.distributed = True
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        utils.setup_for_distributed(config.rank == 0)
    else:
        config.distributed = False
    return device


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def _corrupt_mask_tensor(mask, prob_erase=0.2):
    mask = mask.squeeze().cpu().numpy()
    H, W = mask.shape
    for _ in range(int(H * W * prob_erase)):
        h = np.random.randint(0, H)
        w = np.random.randint(0, W)
        mask[h, w] = 0.0
    return torch.tensor(mask, dtype=torch.float32)
