import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
from PIL import Image
from tqdm import tqdm
import shutil
import multiprocessing as mp
import threading
import warnings
warnings.filterwarnings("ignore")

from ruamel.yaml import YAML
from easydict import EasyDict

import misc as utils
from models.GroundingDINO.temporal_branch import TemporalBranch

IM_MEAN = [0.2663, 0.2736, 0.2876]
IM_STD  = [0.1378, 0.1416, 0.1483]
transform = T.Compose([
    T.Resize(256),
    T.ToTensor(),
    T.Normalize(IM_MEAN, IM_STD),
])

def _cfg_val(tb_conf, key, default=None):
    v = tb_conf.get(key, default)
    if isinstance(v, dict) and 'value' in v:
        return v['value']
    try:
        if hasattr(v, 'value'):
            return v.value
    except Exception:
        pass
    return v

def load_mask_as_tensor(path: str, target_hw: tuple):
    if not path or (not os.path.exists(path)):
        return None
    m = Image.open(path).convert('L')
    m = m.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)
    arr = np.array(m).astype(np.float32)
    if arr.max() > 1.0:
        arr = (arr > 127.5).astype(np.float32)
    return torch.from_numpy(arr)[None, None, ...]

def mask_has_fg(path: str, min_pixels: int = 1):
    if not path or (not os.path.exists(path)):
        return False
    m = Image.open(path).convert('L')
    arr = np.asarray(m)
    return (arr > 127).sum() >= min_pixels

def get_gt_path(gt_root: str, video: str, exp_id: str, frame_name: str, exts, gt_layout="auto", debug=False):
    def try_patterns(layout: str):
        paths = []
        if layout == "per_exp":
            for ext in exts:
                paths.append(os.path.join(gt_root, video, exp_id, f"{frame_name}.{ext}"))
        elif layout == "per_video":
            for ext in exts:
                paths.append(os.path.join(gt_root, video, f"{frame_name}.{ext}"))
        else:
            raise ValueError(f"Unknown gt_layout={layout}")
        return paths
    layouts = ["per_exp", "per_video"] if gt_layout == "auto" else [gt_layout]
    for layout in layouts:
        for p in try_patterns(layout):
            if os.path.exists(p):
                if debug:
                    print(f"[DBG][GT] matched layout={layout} path={p}")
                return p
    if debug:
        print(f"[DBG][GT] not found for {video}/{exp_id}/{frame_name}")
    return None

def normalize_valid_frames(valid_frames, frames_all, debug=False, vid="", exp=""):
    name_to_idx = {name: i for i, name in enumerate(frames_all)}
    if isinstance(valid_frames, str):
        s = valid_frames.strip()
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('(') and s.endswith(')')):
            try:
                import json as _json
                valid_frames = _json.loads(s.replace("'", '"'))
            except Exception:
                valid_frames = [s]
        else:
            valid_frames = [s]
    if not isinstance(valid_frames, (list, tuple)):
        if debug:
            print(f"[DBG][{vid}/{exp}] valid_frames not list/tuple after normalize: {valid_frames}")
        return []
    def strip_ext(s):
        s = s.strip()
        base, ext = os.path.splitext(s)
        return base if base != "" else s
    idxs = []
    for v in valid_frames:
        if isinstance(v, int):
            if 0 <= v < len(frames_all):
                idxs.append(v)
        elif isinstance(v, str):
            vv = strip_ext(v)
            if vv in name_to_idx:
                idxs.append(name_to_idx[vv])
            else:
                try:
                    vi = int(vv)
                    if 0 <= vi < len(frames_all):
                        idxs.append(vi)
                except Exception:
                    if debug:
                        print(f"[DBG] unknown token {v}")
    return sorted(set(idxs))

def build_seg_starts_from_scan(frames_all, video_name, exp_id, gt_folder, gt_exts, gt_layout, require_nonempty=True, debug=False):
    seg = []
    for i, fname in enumerate(frames_all):
        p = get_gt_path(gt_folder, video_name, exp_id, fname, gt_exts, gt_layout, debug=False)
        if p is None:
            continue
        if require_nonempty:
            if mask_has_fg(p):
                seg.append(i)
        else:
            seg.append(i)
    if debug:
        print(f"[DBG][{video_name}/{exp_id}] scan found {len(seg)} starts")
    return seg

def forward_roll_with_burn_in(branch: TemporalBranch, mem_img: torch.Tensor, mem_mask: torch.Tensor, frames: torch.Tensor,
                              burn_in=3, use_soft_mem=True, smooth_k=7, per_channel_norm_guide: bool = False,
                              debug=False, video_name=None, exp_id=None):
    branch.eval()
    B, Tp = frames.shape[:2]
    device = mem_img.device
    dtype = mem_img.dtype

    mem_mask = mem_mask.to(device=device, dtype=dtype)
    if mem_mask.dtype != torch.float32:
        mem_mask = mem_mask.float()
    if mem_mask.max() > 1.0:
        mem_mask = mem_mask / 255.0
    mem_mask = mem_mask.clamp(0.0, 1.0)

    if use_soft_mem:
        mem_mask = F.avg_pool2d(mem_mask, kernel_size=smooth_k, stride=1, padding=smooth_k//2).clamp(0.0, 1.0)

    with torch.no_grad():
        mem_feat_high = branch.encoder(mem_img)
        prev_mask = mem_mask

        guide0 = branch._build_guide(mem_feat_high if not branch.use_guide_blend else mem_feat_high, prev_mask, per_channel_norm=per_channel_norm_guide)
        h_high = branch.h_init(guide0)
        h_low = None
        if getattr(branch, "dual_scale", False):
            guide0_low = F.avg_pool2d(guide0, 2)
            h_low = branch.h_init_low(guide0_low)

        preds = []
        for t in range(Tp):
            feat_t_high = branch.encoder(frames[:, t])
            use_prev = prev_mask if (t >= burn_in) else mem_mask
            if branch.use_guide_blend:
                beta = branch._beta()
                feat_for_guide = beta * mem_feat_high + (1.0 - beta) * feat_t_high
            else:
                feat_for_guide = mem_feat_high

            guide_t = branch._build_guide(feat_for_guide, use_prev, per_channel_norm=per_channel_norm_guide)

            if getattr(branch, "dual_scale", False):
                feat_t_low = F.avg_pool2d(feat_t_high, 2)
                guide_t_low = F.avg_pool2d(guide_t, 2)
                x_low = feat_t_low + branch._alpha() * guide_t_low
                h_low = branch.gru_low(x_low, h_low)

            h_high = branch._step_update(feat_t_high, guide_t, h_high, h_low if getattr(branch, "dual_scale", False) else None)

            skip_feat = feat_t_high if branch.use_decoder_skip else None
            out = branch.decoder(h_high, skip_feat) if skip_feat is not None else branch.decoder(h_high)
            prob = torch.sigmoid(out) if branch.return_logits else out
            prob = prob.clamp(0.0, 1.0)
            preds.append(prob.detach())
            if branch.iterative_guidance:
                prev_mask = prob

            if debug and (video_name is not None) and t < 6:
                print(f"[burn_in] vid={video_name} exp={exp_id} t={t} mean_prob={float(prob.mean().item()):.4f}")

        probs = torch.stack(preds, dim=1)
    return probs

def sub_processor(lock, pid, config, data, save_path_prefix, img_folder, gt_folder, video_list):
    text = f'processor {pid}'
    with lock:
        progress = tqdm(total=len(video_list), position=pid, desc=text, ncols=0)
    try:
        torch.cuda.set_device(pid)
    except Exception:
        pass
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    debug = bool(config.get('debug', False))

    tb_conf = config.get('TemporalBranch', {}) or {}
    branch = TemporalBranch(
        feat_ch=int(_cfg_val(tb_conf, 'feat_ch', 256)),
        hidden_ch=int(_cfg_val(tb_conf, 'gru_hidden_dim', 256)),
        use_exist_head=bool(_cfg_val(tb_conf, 'use_exist_head', False)),
        shared_encoder=None,
        freeze_encoder=bool(_cfg_val(tb_conf, 'freeze_encoder', True)),
        return_logits=True,
        iterative_guidance=bool(_cfg_val(tb_conf, 'iterative_guidance', True)),
        gn_groups=int(_cfg_val(tb_conf, 'gn_groups', 32)),
        learnable_alpha=bool(_cfg_val(tb_conf, 'learnable_alpha', False)),
        alpha_channelwise=bool(_cfg_val(tb_conf, 'alpha_channelwise', False)),
        alpha_init=float(_cfg_val(tb_conf, 'alpha_init', 1.0)),
        use_guide_blend=bool(_cfg_val(tb_conf, 'use_guide_blend', False)),
        learnable_beta=bool(_cfg_val(tb_conf, 'learnable_beta', True)),
        beta_init=float(_cfg_val(tb_conf, 'beta_init', 0.5)),
        use_decoder_aspp=bool(_cfg_val(tb_conf, 'use_decoder_aspp', False)),
        use_decoder_depthwise=bool(_cfg_val(tb_conf, 'use_decoder_depthwise', False)),
        use_decoder_skip=bool(_cfg_val(tb_conf, 'use_decoder_skip', False)),
        dual_scale=bool(_cfg_val(tb_conf, 'dual_scale', False)),
        hidden_ch_low=int(_cfg_val(tb_conf, 'hidden_ch_low', 128)) if _cfg_val(tb_conf, 'hidden_ch_low', None) is not None else None,
        gamma_init=float(_cfg_val(tb_conf, 'gamma_init', 0.5)),
        decode_head_first=bool(_cfg_val(tb_conf, 'decode_head_first', False)),
    ).to(device)
    branch.eval()

    if config.get('checkpoint_path', None) is None:
        raise ValueError("Please specify --checkpoint_path")
    ckpt = torch.load(config.checkpoint_path, map_location='cpu')
    if 'branch_state_dict' not in ckpt:
        raise ValueError("Checkpoint missing 'branch_state_dict'")
    branch.load_state_dict(ckpt['branch_state_dict'], strict=False)
    branch.to(device)
    if pid == 0:
        print(f"[Branch] loaded checkpoint into device {device}")
    gate = torch.sigmoid(getattr(branch, "gate", torch.tensor(0.5))).item()
    if pid == 0:
        print(f"[INFO] branch.gate (raw): {branch.gate.data}")
        print(f"[INFO] branch.gate (sigmoid): {gate}")

    gt_exts = [e.strip() for e in str(config.get('gt_exts', 'png,PNG,jpg,JPG')).split(',') if e.strip()]
    use_json_valid = bool(config.get('use_json_valid_frames', False))
    fill_before_first = config.get('fill_before_first', 'skip')
    gt_layout = config.get('gt_layout', 'auto')

    for video in video_list:
        torch.cuda.empty_cache()
        video_path = os.path.join(img_folder, video)
        frames_all = sorted(f[:-4] for f in os.listdir(video_path) if f.endswith('.jpg'))
        T = len(frames_all)
        if T == 0:
            with lock:
                progress.update(1)
            continue

        if debug:
            print(f"[DBG][{video}] T={T}")

        imgs = []
        origin_w = origin_h = None
        for fname in frames_all:
            img = Image.open(os.path.join(video_path, f"{fname}.jpg")).convert('RGB')
            origin_w, origin_h = img.size
            imgs.append(transform(img))
        imgs = torch.stack(imgs, dim=0).to(device)
        _, _, Ht, Wt = imgs.shape

        expressions = data[video]["expressions"]
        video_level_valid = data[video].get("valid_frames", None)
        objects = data[video].get("objects", {})

        metas = []
        for exp_id, exp_dict in expressions.items():
            metas.append({
                "video": video,
                "exp_id": exp_id,
                "frames": frames_all,
                "valid_frames_json": exp_dict.get("valid_frames", None),
                "obj_id": str(exp_dict.get("obj_id", "")) if "obj_id" in exp_dict else ""
            })

        for meta in metas:
            video_name = meta["video"]
            exp_id = meta["exp_id"]
            obj_id = meta.get("obj_id", "") or ""
            save_dir = os.path.join(save_path_prefix, video_name, exp_id)
            os.makedirs(save_dir, exist_ok=True)

            seg_starts = []
            seg_src = "scan"
            if use_json_valid:
                vf = meta.get("valid_frames_json", None)
                if vf:
                    seg_starts = normalize_valid_frames(vf, frames_all, debug=debug, vid=video_name, exp=f"{exp_id}:exp")
                    seg_src = "json(exp)"
                if not seg_starts and obj_id and (obj_id in objects):
                    vf_obj = objects.get(obj_id, {}).get("valid_frames", None)
                    if vf_obj:
                        seg_starts = normalize_valid_frames(vf_obj, frames_all, debug=debug, vid=video_name, exp=f"{exp_id}:obj")
                        seg_src = "json(obj)"
                if not seg_starts and video_level_valid:
                    seg_starts = normalize_valid_frames(video_level_valid, frames_all, debug=debug, vid=video_name, exp=f"{exp_id}:video")
                    seg_src = "json(video)"

            if not seg_starts:
                seg_starts = build_seg_starts_from_scan(frames_all, video_name, exp_id, gt_folder, gt_exts, gt_layout, require_nonempty=True, debug=debug)
                seg_src = "scan"

            if len(seg_starts) == 0:
                warnings.warn(f"[{video_name}/{exp_id}] No valid segment starts (src={seg_src}). Skip.")
                continue

            seg_starts = sorted(set([i for i in seg_starts if 0 <= i < T]))
            seg_ends = seg_starts[1:] + [T]

            first_start = seg_starts[0]
            if first_start > 0 and fill_before_first == 'zeros':
                zeros_img = np.zeros((origin_h, origin_w), dtype=np.uint8)
                for t_abs in range(0, first_start):
                    Image.fromarray(zeros_img).save(os.path.join(save_dir, f"{frames_all[t_abs]}.png"))

            total_written = 0
            for s, e in zip(seg_starts, seg_ends):
                fname_s = frames_all[s]
                gt_path = get_gt_path(gt_folder, video_name, exp_id, fname_s, gt_exts, gt_layout, debug=debug)
                if gt_path is None:
                    warnings.warn(f"[{video_name}/{exp_id}] missing GT for head {fname_s}. skip segment.")
                    continue

                mem_img = imgs[s:s+1]
                mem_mask = load_mask_as_tensor(gt_path, (Ht, Wt))
                if mem_mask is None:
                    warnings.warn(f"[{video_name}/{exp_id}] failed to load GT at {gt_path}. skip.")
                    continue
                mem_mask = mem_mask.to(device).clamp(0.0, 1.0)

                with torch.no_grad():
                    branch_key_logits = branch.forward_initial(
                        mem_img, mem_mask,
                        per_channel_norm_guide=bool(_cfg_val(tb_conf, 'per_channel_norm_guide', False))
                    )
                    branch_key_mask = torch.sigmoid(branch_key_logits).squeeze(0).squeeze(0)
                main_mask_bin = mem_mask.squeeze(0).squeeze(0)
                fused_mask = gate * main_mask_bin + (1 - gate) * branch_key_mask
                if fused_mask.dim() == 2:
                    fused_mask_up = F.interpolate(fused_mask.unsqueeze(0).unsqueeze(0), size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                elif fused_mask.dim() == 3:
                    fused_mask_up = F.interpolate(fused_mask.unsqueeze(0), size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                elif fused_mask.dim() == 4:
                    fused_mask_up = F.interpolate(fused_mask, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                else:
                    raise ValueError(f"Unexpected fused_mask shape: {fused_mask.shape}")
                fused_mask_bin = (fused_mask_up.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                Image.fromarray(fused_mask_bin).save(os.path.join(save_dir, f"{fname_s}.png"))
                total_written += 1

                if e - (s + 1) > 0:
                    frames_roll = imgs[s+1:e].unsqueeze(0)
                    per_ch_norm = bool(_cfg_val(tb_conf, 'per_channel_norm_guide', False)) if 'tb_conf' in locals() else False
                    probs = forward_roll_with_burn_in(branch, mem_img, mem_mask, frames_roll,
                                                      burn_in=0, use_soft_mem=False, smooth_k=1,
                                                      per_channel_norm_guide=per_ch_norm,
                                                      debug=debug, video_name=video_name, exp_id=exp_id)
                    probs = probs.squeeze(0).squeeze(1)
                    for t_local, t_abs in enumerate(range(s+1, e)):
                        fname = frames_all[t_abs]
                        p = probs[t_local]
                        p_up = F.interpolate(p.unsqueeze(0).unsqueeze(0), size=(origin_h, origin_w), mode='bilinear', align_corners=False)
                        m_bin = (p_up.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                        Image.fromarray(m_bin).save(os.path.join(save_dir, f"{fname}.png"))
                        total_written += 1

            if debug:
                print(f"[DBG][{video_name}/{exp_id}] written_masks={total_written}")

        with lock:
            progress.update(1)

    with lock:
        progress.close()

def main(config):
    print("Branch fusion inference starting")
    print("Config keys:", {k: config.get(k) for k in ['dataset_root','split','num_gpus','version', 'inference_result_dir'] if k in config})

    seed = config.get('seed', 42) + (utils.get_rank() if hasattr(utils, 'get_rank') else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_dir = os.path.join(config.output_dir, config.dataset_name, config.version)
    output_dir = os.path.join(save_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    if config.get('config_path', None):
        try:
            shutil.copyfile(src=config.config_path, dst=os.path.join(save_dir, 'config.yaml'))
        except Exception:
            pass

    root = Path(config.dataset_root)
    split = config.split
    img_folder = os.path.join(root, split, config.img_dir_name)

    gt_folder = config.get('inference_result_dir', None)

    if gt_folder is None:
        print("[Warning] No --inference_result_dir provided. Trying to guess from output_dir...")
        gt_folder = os.path.join(config.output_dir, config.dataset_name, config.version, "Annotations")

    print(f"===> Using inference source (Backbone Masks) from: {gt_folder}")
    if not os.path.exists(gt_folder):
        raise ValueError(f"Inference result directory does not exist: {gt_folder}")

    meta_file = os.path.join(root, split, "cleaned_output.json")

    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = sorted(data.keys())

    thread_num = int(config.num_gpus)
    processes = []
    lock = threading.Lock()
    per_thread_video_num = len(video_list) // thread_num if thread_num > 0 else len(video_list)

    start_time = time.time()
    print('Start branch fusion inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_list = video_list[i * per_thread_video_num:]
        else:
            sub_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, config, data, output_dir, img_folder, gt_folder, sub_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.4f}s")
    zip_file_path = os.path.join(save_dir, config.version)
    shutil.make_archive(zip_file_path, 'zip', root_dir=save_dir, base_dir='Annotations')
    print('Saved zip:', zip_file_path + ".zip")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Temporal Branch Fusion: backbone+branch mask fusion inference')
    parser.add_argument('--config_path', '-c', required=True, help='path to configuration file')
    parser.add_argument("--checkpoint_path", '-ckpt', required=True, help="checkpoint path (must contain branch_state_dict)")
    parser.add_argument("--version", default="branch_gt_roll_dynamic", help="output version subfolder")
    parser.add_argument('--num_gpus', '-ng', type=int, required=True, help='number of CUDA gpus to run on.')
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--img_dir_name', type=str, default='JPEGImages')
    parser.add_argument('--gt_dir_name', type=str, default='Annotations')
    parser.add_argument('--use_json_valid_frames', action='store_true')
    parser.add_argument('--gt_exts', type=str, default='png,PNG,jpg,JPG')
    parser.add_argument('--gt_layout', type=str, default='auto', choices=['auto','per_exp','per_video'])
    parser.add_argument('--fill_before_first', type=str, default='skip', choices=['skip','zeros'])
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--inference_result_dir', type=str, default=None,
                        help="Path to the 'Annotations' folder containing backbone inference results (masks) to be used as guidance.")

    args_cli = parser.parse_args()
    with open(args_cli.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: (v['value'] if isinstance(v, dict) and 'value' in v else getattr(v, 'value', v)) for k, v in config.items()}
    cfg = {**config, **vars(args_cli)}
    cfg = EasyDict(cfg)
    cfg.checkpoint_path = cfg.get('checkpoint_path', cfg.get('ckpt', cfg.get('checkpoint_path', None)))
    cfg.output_dir = cfg.get('output_dir', cfg.get('output_dir', './output'))

    main(cfg)
