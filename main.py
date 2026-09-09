
























import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from trainer import Trainer
from pretrainer import PreTrainer

import os
import wandb

from ruamel.yaml import YAML
from easydict import EasyDict
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run(process_id, args):
    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        raw = yaml.load(f)
    cfg = {k: v['value'] for k, v in raw.items()}

    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    cfg = {**cfg, **args_dict}

    if 'GroundingDINO' not in cfg:
        cfg['GroundingDINO'] = {}
    gd = cfg['GroundingDINO']

    def _override_gd(key, val):
        if val is not None:
            gd[key] = val

    _override_gd('use_ot_prompt', cfg.get('use_ot_prompt', None))
    _override_gd('textfree_infer', cfg.get('textfree_infer', None))
    _override_gd('t_dim', cfg.get('t_dim', None))
    _override_gd('num_prompt_tokens', cfg.get('num_prompt_tokens', None))
    _override_gd('ot_hidden', cfg.get('ot_hidden', None))
    _override_gd('ot_pool', cfg.get('ot_pool', None))
    _override_gd('strong_label_prompt', cfg.get('strong_label_prompt', None))
    _override_gd('label_residual_min_scale', cfg.get('label_residual_min_scale', None))
    _override_gd('label_prompt_film', cfg.get('label_prompt_film', None))
    _override_gd('label_prompt_film_scale', cfg.get('label_prompt_film_scale', None))
    _override_gd('grade_free_prompt', cfg.get('grade_free_prompt', None))

    _override_gd('use_simple_embed', cfg.get('use_simple_embed', None))

    if cfg.get('textless_prob', None) is None:
        cfg['textless_prob'] = raw.get('textless_prob', {}).get('value', 0.0)

    def _pull_top(name, default=None):
        if cfg.get(name, None) is None:
            cfg[name] = raw.get(name, {}).get('value', default)

    for k, dv in [
        ('w_ot', 1.0), ('w_pair', 0.2), ('w_mmd', 0.5), ('w_tv', 0.0),
        ('align_objective', 'ot'), ('w_contrast', 0.0), ('contrast_temperature', 0.07),
        ('w_grade_contrast', 0.0), ('grade_contrast_temperature', 0.2),
        ('ot_label_cost', 0.5),
        ('ot_eps', 0.1), ('ot_iters', 50),
        ('mmd_sigma', 1.0), ('mmd_ref_num', 512), ('mmd_bank_size', 8192),
        ('lr_scale', 1.0),
    ]:
        _pull_top(k, dv)

    cfg.setdefault('checkpoint_path', None)
    cfg.setdefault('pretrained_weights', None)
    cfg.setdefault('eval', False)
    cfg.setdefault('eval_off', False)
    cfg.setdefault('debug', False)

    config = EasyDict(cfg)
    bs = args.batch_size if args.batch_size is not None else cfg.get('batch_size', 1)
    config['lr'] *= bs * args.num_devices / 2 * float(config.lr_scale)

    if config.running_mode == 'pretrain':
        trainer = PreTrainer(config, process_id, device_id=args.device_ids[process_id], num_processes=args.num_devices)
    else:
        trainer = Trainer(config, process_id, device_id=args.device_ids[process_id], num_processes=args.num_devices)

    if config.eval:
        print("Load checkpoint from {} ...".format(config.checkpoint_path))
        if config.checkpoint_path is None:
             raise ValueError("Eval mode requires --checkpoint_path / -ckpt!")

        model_state_dict = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
        if 'model_state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['model_state_dict']

        from torch.nn.parallel import DistributedDataParallel as DDP
        model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model

        res = model_without_ddp.load_state_dict(model_state_dict, strict=False)
        print(f"Load result: {res}")

        trainer.evaluate()
    else:
        if config.checkpoint_path is not None:
            trainer.load_checkpoint(config.checkpoint_path, total_epoch=config.epochs)
        trainer.train()
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOS DINO: Training')
    parser.add_argument('--config_path', '-c', required=True,
                        help='path to configuration file')

    parser.add_argument('--running_mode', '-rm', choices=['train', 'pretrain'], required=True)

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epochs', type=int, default=30,
                        help="the total epochs")

    parser.add_argument("--checkpoint_path", '-ckpt', default=None,
                        help="The checkpoint path"
                        )

    parser.add_argument("--pretrained_weights", '-pw', default=None,
                        help='The pretrained weights path'
                        )

    parser.add_argument("--version", default="refer_dino",
                        help="the saved ckpt and output version")
    parser.add_argument("--lr_drop", default=[8], type=int, nargs='+')
    parser.add_argument('--lr_scale', type=float, default=None,
                        help='additional multiplier after automatic batch/GPU LR scaling')
    parser.add_argument('--enable_amp', type=lambda x: str(x).lower() == 'true', default=None,
                        help='enable automatic mixed precision; false is recommended for stable full training')

    parser.add_argument('--batch_size', '-bs', type=int, default=None,
                        help='training batch size per device')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--eval_off', action="store_true")

    parser.add_argument('--use_ot_prompt', type=lambda x: str(x).lower()=='true', default=None,
                        help='override GroundingDINO.use_ot_prompt')
    parser.add_argument('--textfree_infer', type=lambda x: str(x).lower()=='true', default=None,
                        help='override GroundingDINO.textfree_infer')
    parser.add_argument('--t_dim', type=int, default=None)
    parser.add_argument('--num_prompt_tokens', type=int, default=None)
    parser.add_argument('--ot_hidden', type=int, default=None)
    parser.add_argument('--ot_pool', type=str, choices=['mean','max'], default=None)
    parser.add_argument('--strong_label_prompt', type=lambda x: str(x).lower()=='true', default=None)
    parser.add_argument('--label_residual_min_scale', type=float, default=None)
    parser.add_argument('--label_prompt_film', type=lambda x: str(x).lower()=='true', default=None)
    parser.add_argument('--label_prompt_film_scale', type=float, default=None)
    parser.add_argument('--grade_free_prompt', type=lambda x: str(x).lower()=='true', default=None)

    parser.add_argument('--textless_prob', type=float, default=None)

    parser.add_argument('--use_simple_embed', type=lambda x: str(x).lower()=='true', default=None,
                        help='Enable Simple Embedding Baseline (Disable OT module)')

    parser.add_argument('--force_label_mode', type=str, choices=['pred', 'gt', 'noise'], default='pred',
                        help='Control the labels used for inference sensitivity analysis. (pred/gt/noise)')

    parser.add_argument('--w_ot', type=float, default=None)
    parser.add_argument('--align_objective', type=str, choices=['ot', 'info_nce', 'infonce', 'supcon', 'supervised_contrastive'], default=None)
    parser.add_argument('--w_contrast', type=float, default=None)
    parser.add_argument('--contrast_temperature', type=float, default=None)
    parser.add_argument('--w_pair', type=float, default=None)
    parser.add_argument('--w_mmd', type=float, default=None)
    parser.add_argument('--w_tv', type=float, default=None)
    parser.add_argument('--w_grade_contrast', type=float, default=None)
    parser.add_argument('--grade_contrast_temperature', type=float, default=None)
    parser.add_argument('--ot_label_cost', type=float, default=None)
    parser.add_argument('--ot_eps', type=float, default=None)
    parser.add_argument('--ot_iters', type=int, default=None)
    parser.add_argument('--mmd_sigma', type=float, default=None)
    parser.add_argument('--mmd_ref_num', type=int, default=None)
    parser.add_argument('--mmd_bank_size', type=int, default=None)

    gpu_params_group = parser.add_mutually_exclusive_group(required=True)
    gpu_params_group.add_argument('--num_gpus', '-ng', type=int, default=argparse.SUPPRESS)
    gpu_params_group.add_argument('--gpu_ids', '-gids', type=int, nargs='+', default=argparse.SUPPRESS)
    gpu_params_group.add_argument('--cpu', '-cpu', action='store_true', default=argparse.SUPPRESS)

    args = parser.parse_args()

    if hasattr(args, 'num_gpus'):
        args.num_devices = max(min(args.num_gpus, torch.cuda.device_count()), 1)
        args.device_ids = list(range(args.num_gpus))
    elif hasattr(args, 'gpu_ids'):
        for gpu_id in args.gpu_ids:
            assert 0 <= gpu_id <= torch.cuda.device_count() - 1, \
                f'error: gpu ids must be between 0 and {torch.cuda.device_count() - 1}'
        args.num_devices = len(args.gpu_ids)
        args.device_ids = args.gpu_ids
    else:
        args.device_ids = ['cpu']
        args.num_devices = 1

    if args.num_devices > 1:
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        if 'MASTER_PORT' not in os.environ:
            import random
            os.environ['MASTER_PORT'] = str(random.randint(20000, 30000))

        torch.multiprocessing.spawn(run, nprocs=args.num_devices, args=(args,))
    else:
        run(process_id=0, args=args)
