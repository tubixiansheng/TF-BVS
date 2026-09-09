







































































































































import argparse
import csv
import hashlib
import json
import queue
import random
import time
from pathlib import Path

import numpy as np
import torch

import misc as utils
from models import build_model
import torchvision.transforms as T
import os
from PIL import Image, ImageDraw
import torch.nn.functional as F

from tqdm import tqdm
import shutil
import re
import multiprocessing as mp
import threading
import warnings
warnings.filterwarnings("ignore")

from ruamel.yaml import YAML
from easydict import EasyDict
from torch.cuda.amp import autocast
from misc import nested_tensor_from_videos_list

color_list = utils.colormap()
color_list = color_list.astype('uint8').tolist()

transform = T.Compose([
    T.Resize(256),
    T.ToTensor(),
    T.Normalize([0.2663, 0.2736, 0.2876], [0.1378, 0.1416, 0.1483])
])

def extract_grade_from_video_name(video_name):
    """Extract the BI-RADS grade encoded in a video name."""
    match = re.search(r'L([2-5])', video_name)
    if match:
        return int(match.group(1)) - 2
    else:
        raise ValueError(f"Strict Check Failed: Cannot find grade label (L2-L5) in video name: {video_name}")

def build_forced_grade_label(gt_val, mode, video_name, num_classes=4):
    """Construct deterministic labels for grade-sensitivity experiments."""
    if mode == 'gt':
        return gt_val
    if mode == 'wrong':
        return (gt_val + 1) % num_classes
    if mode == 'high_low_swap':
        return {0: 3, 1: 2, 2: 1, 3: 0}[gt_val]
    if mode == 'adjacent_swap':
        return {0: 1, 1: 0, 2: 3, 3: 2}[gt_val]
    if mode == 'random':
        candidates = [i for i in range(num_classes) if i != gt_val]
        digest = hashlib.md5(f"{video_name}:random_grade".encode("utf-8")).hexdigest()
        return candidates[int(digest, 16) % len(candidates)]
    raise ValueError(f"Unsupported force_label_mode: {mode}")

def main(config):
    print("Inference only supports for batch size = 1")
    print(config)

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = getattr(config, "split", None) or "test"
    save_dir = os.path.join(args.output_dir, args.dataset_name, args.version)
    output_dir = os.path.join(save_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src=args.config_path, dst=os.path.join(save_dir, 'config.yaml'))

    save_visualize_path_prefix = os.path.join(save_dir, split + '_images')
    if config.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    root = Path(args.dataset_path)
    img_folder = os.path.join(root, split, "JPEGImages")

    meta_file = os.path.join(root, split, "cleaned_output.json")
    if not os.path.exists(meta_file):
        meta_file_alt = os.path.join(root, split, "meta.json")
        if os.path.exists(meta_file_alt):
            meta_file = meta_file_alt

    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = sorted(data.keys())
    max_videos = int(getattr(config, 'max_videos', 0) or 0)
    if max_videos > 0:
        video_list = video_list[:max_videos]

    if getattr(config, 'force_textfree', False):
        if 'GroundingDINO' in config and isinstance(config.GroundingDINO, dict):
            config.GroundingDINO['textfree_infer'] = True
            config.GroundingDINO['use_ot_prompt'] = True

    thread_num = config.num_gpus
    processes = []
    lock = threading.Lock()
    progress_queue = mp.Queue()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num if thread_num > 0 else video_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, config, data,
                                                   output_dir, save_visualize_path_prefix,
                                                   img_folder, sub_video_list,
                                                   progress_queue))
        p.start()
        processes.append(p)

    completed_videos = 0
    with tqdm(total=video_num, desc='inference', ncols=0) as progress:
        while completed_videos < video_num:
            try:
                item = progress_queue.get(timeout=5)
            except queue.Empty:
                if all(not p.is_alive() for p in processes):
                    break
                continue
            if item == 'done':
                completed_videos += 1
                progress.update(1)
            elif isinstance(item, tuple) and item[0] == 'error':
                progress.write(f"[ERROR][processor {item[1]}] {item[2]}")

    for p in processes:
        p.join()
        if p.exitcode not in (0, None):
            raise RuntimeError(f"Inference worker exited with code {p.exitcode}")

    debug_csv = getattr(args, 'debug_label_csv', None)
    if debug_csv:
        debug_parts = [Path(f"{debug_csv}.worker{pid}.csv") for pid in range(thread_num)]
        debug_rows = []
        for part in debug_parts:
            if not part.exists():
                continue
            with part.open("r", newline="") as f:
                debug_rows.extend(csv.DictReader(f))
        if debug_rows:
            debug_path = Path(debug_csv)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            fields = list(debug_rows[0].keys())
            with debug_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(debug_rows)
            print(f"[OK] label debug CSV: {debug_path}")

    end_time = time.time()
    total_time = end_time - start_time

    print("Total inference time: %.4f s" % (total_time))

    print('creating a zip file with the predictions...')
    zip_file_path = os.path.join(save_dir, args.version)
    shutil.make_archive(zip_file_path, 'zip', root_dir=save_dir, base_dir='Annotations')
    print('a zip file was successfully created.')
    shutil.rmtree(output_dir)

    print("\nSave results at: {}".format(zip_file_path + ".zip"))

def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, video_list,
                  progress_queue=None):
    text = 'processor %d' % pid
    show_worker_progress = getattr(args, 'worker_progress', False)
    if show_worker_progress:
        with lock:
            progress = tqdm(total=len(video_list), position=pid, desc=text, ncols=0)
    else:
        progress = None
    torch.cuda.set_device(pid)

    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    model_without_ddp = model

    use_simple_embed = getattr(args, 'use_simple_embed', False)
    if use_simple_embed:
        if hasattr(model_without_ddp, 'use_simple_embed'):
            model_without_ddp.use_simple_embed = True
        else:
            setattr(model_without_ddp, 'use_simple_embed', True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)
        if use_simple_embed:
            print("===> Inference Mode: Simple Embedding Baseline ENABLED")

        mode = getattr(args, 'force_label_mode', 'none')
        print(f"===> Inference Label Mode: {mode.upper()}")
        if mode == 'none':
            print("     (Using model predictions)")
        elif mode == 'gt':
            print("     (Using Ground Truth labels - Upper Bound Analysis)")
        elif mode == 'wrong':
            print("     (Using deterministic cyclic wrong labels - Sensitivity Analysis)")
        elif mode == 'random':
            print("     (Using deterministic random non-GT labels - Sensitivity Analysis)")
        elif mode == 'high_low_swap':
            print("     (Swapping L2<->L5 and L3<->L4 labels - Sensitivity Analysis)")
        elif mode == 'adjacent_swap':
            print("     (Swapping L2<->L3 and L4<->L5 labels - Sensitivity Analysis)")

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        raw_state = checkpoint.get("model_state_dict", checkpoint)

        model_state = model_without_ddp.state_dict()
        state_dict = {k: v for k, v in raw_state.items()
                      if k in model_state and v.shape == model_state[k].shape}

        skipped = [k for k in raw_state.keys() if k not in state_dict]
        if skipped and pid == 0:
            print("??  skipped mismatch keys:", skipped)

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

        if len(missing_keys) > 0 and pid == 0:
            print('Missing Keys: {}'.format(missing_keys))
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    num_all_frames = 0
    model.eval()
    debug_rows = []

    for video in video_list:
        torch.cuda.empty_cache()

        video_path = os.path.join(img_folder, video)
        frames_all = sorted(f[:-4] for f in os.listdir(video_path) if f.endswith('.jpg'))
        video_len = len(frames_all)

        k = max(1, int(getattr(args, 'frame_step', 1)))
        sel_idx = [t for t in range(video_len) if t % k == 0]
        if len(sel_idx) == 0:
            continue
        frames = [frames_all[t] for t in sel_idx]

        imgs = []
        origin_w, origin_h = None, None
        for frame in frames:
            img_path = os.path.join(video_path, f"{frame}.jpg")
            img = Image.open(img_path).convert('RGB')
            origin_w, origin_h = img.size
            imgs.append(transform(img))
        imgs = torch.stack(imgs, dim=0).to(args.device)

        samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=getattr(args, 'size_divisibility', 1))
        img_h, img_w = imgs.shape[-2:]
        size   = torch.as_tensor([img_h, img_w], device=args.device)

        target = {"size": size, "frames_idx": sel_idx}

        cli_mode = getattr(args, 'force_label_mode', 'none')
        try:
            gt_val = extract_grade_from_video_name(video)
            used_val = gt_val if cli_mode == 'none' else build_forced_grade_label(gt_val, cli_mode, video)
            target["classes"] = torch.tensor([used_val], device=args.device)
            target["gt_classes"] = torch.tensor([gt_val], device=args.device)
        except ValueError as e:
            if cli_mode != 'none':
                print(f"\n[ERROR] Processing video: {video}")
                print(f"[ERROR] Mode is '{cli_mode}', but failed to extract grade from filename.")
                raise e

        model_grade_mode = 'pred'
        if cli_mode != 'none':
            model_grade_mode = 'gt'

        expressions = data[video]["expressions"]
        metas = []
        for exp_id, exp_dict in expressions.items():
            metas.append({
                "video":  video,
                "exp":    exp_dict["exp"],
                "exp_id": exp_id,
                "frames": frames
            })
        num_expressions = len(metas)

        for i in range(num_expressions):
            video_name = metas[i]["video"]
            exp = metas[i]["exp"]
            exp_id = metas[i]["exp_id"]
            frames_i = metas[i]["frames"]

            captions = None if getattr(args, 'force_textfree', False) else [exp]

            with torch.no_grad():
                with autocast(args.enable_amp):
                    force_grade = target.get("classes", None) if cli_mode != 'none' else None
                    outputs = model.infer(
                        samples,
                        captions,
                        [target],
                        grade_mode=model_grade_mode,
                        force_grade=force_grade,
                        return_prompt_debug=bool(getattr(args, 'debug_label_csv', None)),
                    )
                    if getattr(args, 'debug_label_csv', None):
                        for prompt_row in outputs.get("prompt_debug", []):
                            debug_rows.append({
                                "pid": pid,
                                "video": video_name,
                                "exp_id": exp_id,
                                "cli_mode": cli_mode,
                                "model_grade_mode": model_grade_mode,
                                "gt_label_from_name": target.get("gt_classes", torch.tensor([-1], device=args.device)).detach().cpu().view(-1)[0].item(),
                                "used_label_in_target": target.get("classes", torch.tensor([-1], device=args.device)).detach().cpu().view(-1)[0].item(),
                                **prompt_row,
                            })

            pred_logits = outputs["pred_logits"][0]
            pred_masks  = outputs["pred_masks"][0]
            pred_boxes  = outputs.get("pred_boxes", None)
            if pred_boxes is not None:
                pred_boxes = pred_boxes[0]

            pred_scores = pred_logits.sigmoid().mean(0)
            max_scores, _ = pred_scores.max(-1)
            max_ind = max_scores.argmax().item()

            pred_masks_sel = pred_masks[:, max_ind, ...].unsqueeze(0)
            if pred_boxes is not None:
                pred_boxes_sel = pred_boxes[:, max_ind]
            else:
                pred_boxes_sel = None

            pred_masks_sel = pred_masks_sel[:, :, :img_h, :img_w]
            pred_masks_sel = F.interpolate(pred_masks_sel, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            pred_masks_sel = (pred_masks_sel.sigmoid() > 0.5).squeeze(0).detach().cpu().numpy()

            if args.visualize:
                for t_local, frame in enumerate(frames_i):
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path).convert('RGBA')
                    if pred_boxes_sel is not None:
                        draw = ImageDraw.Draw(source_img)
                        box = pred_boxes_sel[t_local].unsqueeze(0).detach()
                        draw_boxes = rescale_bboxes(box, (origin_w, origin_h)).tolist()
                        xmin, ymin, xmax, ymax = draw_boxes[0]
                        draw.rectangle(((xmin, ymin), (xmax, ymax)),
                                       outline=tuple(color_list[i % len(color_list)]), width=2)
                    source_img = vis_add_mask(source_img, pred_masks_sel[t_local], color_list[i % len(color_list)])
                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                    os.makedirs(save_visualize_path_dir, exist_ok=True)
                    save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                    source_img.save(save_visualize_path)

            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            os.makedirs(save_path, exist_ok=True)
            for t_local, frame in enumerate(frames_i):
                mask = pred_masks_sel[t_local].astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame + ".png")
                mask.save(save_file)

        if progress_queue is not None:
            progress_queue.put('done')
        if progress is not None:
            with lock:
                progress.update(1)
    if progress is not None:
        with lock:
            progress.close()
    debug_csv = getattr(args, 'debug_label_csv', None)
    if debug_csv and debug_rows:
        part_path = Path(f"{debug_csv}.worker{pid}.csv")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(debug_rows[0].keys())
        with part_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(debug_rows)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color, dtype=np.float32)
    mask = mask.astype('uint8') > 0.5
    origin_img = origin_img.astype(np.float32)
    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = origin_img.astype(np.uint8)
    return Image.fromarray(origin_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOS DINO: Inference')
    parser.add_argument('--config_path', '-c', required=True, help='path to configuration file')
    parser.add_argument("--checkpoint_path", '-ckpt', required=True, help="The checkpoint path")
    parser.add_argument("--version", default="refer_dino", help="the saved ckpt and output version")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument('--num_gpus', '-ng', type=int, required=True, help='number of CUDA gpus to run on.')
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--force_textfree', action='store_true', help='captions=None (requires model.infer to support)')
    parser.add_argument('--size_divisibility', type=int, default=1, help='padding divisibility (train often 32)')
    parser.add_argument('--frame_step', type=int, default=1, help='process only frames t where t % k == 0')
    parser.add_argument('--dataset_path', default=None,
                        help='override dataset root from the YAML config')
    parser.add_argument('--split', default=None,
                        help='override dataset split from the YAML config')
    parser.add_argument('--worker_progress', action='store_true', help='show per-worker progress bars')
    parser.add_argument('--max_videos', type=int, default=0, help='debug only: limit the number of videos')
    parser.add_argument('--debug_label_csv', type=str, default=None, help='write label/prompt debug information to CSV')

    parser.add_argument('--use_simple_embed', action='store_true', help='Enable Simple Embedding Baseline')
    parser.add_argument('--force_label_mode', type=str,
                        choices=['none', 'gt', 'wrong', 'random', 'high_low_swap', 'adjacent_swap'],
                        default='none',
                        help='Control labels for sensitivity analysis')
    parser.add_argument('--strong_label_prompt', type=lambda x: str(x).lower() == 'true', default=None,
                        help='Override GroundingDINO.strong_label_prompt (true/false)')
    parser.add_argument('--label_residual_min_scale', type=float, default=None,
                        help='Override GroundingDINO.label_residual_min_scale')
    parser.add_argument('--label_prompt_film', type=lambda x: str(x).lower() == 'true', default=None,
                        help='Override GroundingDINO.label_prompt_film (true/false)')
    parser.add_argument('--label_prompt_film_scale', type=float, default=None,
                        help='Override GroundingDINO.label_prompt_film_scale')
    parser.add_argument('--grade_free_prompt', type=lambda x: str(x).lower() == 'true', default=None,
                        help='Override GroundingDINO.grade_free_prompt (true/false)')

    args_cli = parser.parse_args()

    with open(args_cli.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}

    cli_args = vars(args_cli).copy()
    if cli_args.get('dataset_path') is None:
        cli_args.pop('dataset_path', None)
    if cli_args.get('split') is None:
        cli_args.pop('split', None)
    args = {**config, **cli_args}
    args = EasyDict(args)

    if hasattr(args, "GroundingDINO") and isinstance(args.GroundingDINO, dict):
        for key in [
            "strong_label_prompt",
            "label_residual_min_scale",
            "label_prompt_film",
            "label_prompt_film_scale",
            "grade_free_prompt",
        ]:
            val = getattr(args_cli, key, None)
            if val is not None:
                args.GroundingDINO[key] = val

    main(args)
