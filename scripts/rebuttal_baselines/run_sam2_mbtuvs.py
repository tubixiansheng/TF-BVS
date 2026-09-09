#!/usr/bin/env python3
"""Run SAM2 zero-shot / prompt-based VOS on MBTUVS.

This script keeps SAM2 as a promptable, single-modal video segmentation
baseline. It supports three fair prompt protocols:

  first_mask: use the first non-empty GT mask as a semi-supervised VOS prompt.
  first_box:  use the tight GT box on the first non-empty mask as a light prompt.
  first_point: use one positive point from the first non-empty mask.

Predictions are saved in the same layout consumed by
scripts/rebuttal/aggregate_metrics.py:

  <output_dir>/Annotations/<video>/exp0/<frame>.png
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def add_repo_to_path(repo: Path):
    sys.path.insert(0, str(repo))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def save_mask(path: Path, mask: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def mask_box(mask: np.ndarray, pad: int = 0) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("cannot build a box from an empty mask")
    h, w = mask.shape
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad, w - 1)
    y2 = min(int(ys.max()) + pad, h - 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def mask_center_point(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("cannot build a point from an empty mask")
    mid = len(xs) // 2
    return np.array([[xs[mid], ys[mid]]], dtype=np.float32)


def read_videos(dataset_root: Path, split: str, max_videos: int = 0):
    meta = dataset_root / split / "cleaned_output.json"
    if meta.exists():
        with meta.open("r") as f:
            videos = sorted(json.load(f)["videos"].keys())
    else:
        videos = sorted(p.name for p in (dataset_root / split / "JPEGImages").iterdir() if p.is_dir())
    return videos[:max_videos] if max_videos > 0 else videos


def frame_names(video_dir: Path):
    return sorted(p.stem for p in video_dir.glob("*.jpg"))


def first_nonempty_prompt(gt_dir: Path, frames):
    for idx, name in enumerate(frames):
        mask_path = gt_dir / f"{name}.png"
        if not mask_path.exists():
            continue
        mask = load_mask(mask_path)
        if mask.any():
            return idx, name, mask
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-repo", default="method/sam2-main")
    parser.add_argument("--config", default="configs/sam2/sam2_hiera_l.yaml")
    parser.add_argument("--checkpoint", default="method/sam2-main/checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--dataset-root", default="data/processed_breast_ultrasound_5frames")
    parser.add_argument("--split", default="test")
    parser.add_argument("--prompt-mode", choices=["first_mask", "first_box", "first_point"], default="first_box")
    parser.add_argument("--box-pad", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/rebuttal_baselines/sam2_first_box")
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    repo = Path(args.sam2_repo).resolve()
    add_repo_to_path(repo)
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
        hydra_overrides_extra=["++model.non_overlap_masks=true"],
    )

    dataset_root = Path(args.dataset_root)
    img_root = dataset_root / args.split / "JPEGImages"
    gt_root = dataset_root / args.split / "Annotations"
    ann_root = Path(args.output_dir) / "Annotations"
    ann_root.mkdir(parents=True, exist_ok=True)

    videos = read_videos(dataset_root, args.split, args.max_videos)
    skipped = []
    use_cuda_amp = str(args.device).startswith("cuda")

    for video in tqdm(videos, desc=f"SAM2 {args.prompt_mode}"):
        video_dir = img_root / video
        gt_dir = gt_root / video
        frames = frame_names(video_dir)
        if not frames:
            skipped.append((video, "no frames"))
            continue
        start_idx, _, prompt_mask = first_nonempty_prompt(gt_dir, frames)
        if prompt_mask is None:
            skipped.append((video, "no non-empty prompt mask"))
            continue

        out_dir = ann_root / video / "exp0"
        out_dir.mkdir(parents=True, exist_ok=True)

        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda_amp else torch.no_grad()
        with torch.inference_mode(), ctx:
            state = predictor.init_state(
                video_path=str(video_dir),
                async_loading_frames=False,
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
            if args.prompt_mode == "first_mask":
                predictor.add_new_mask(state, frame_idx=start_idx, obj_id=1, mask=prompt_mask)
            elif args.prompt_mode == "first_box":
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=start_idx,
                    obj_id=1,
                    box=mask_box(prompt_mask, pad=args.box_pad),
                )
            else:
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=start_idx,
                    obj_id=1,
                    points=mask_center_point(prompt_mask),
                    labels=np.array([1], dtype=np.int32),
                )

            h, w = prompt_mask.shape
            zero = np.zeros((h, w), dtype=bool)

            seen = set()
            for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
                state, start_frame_idx=start_idx, reverse=False
            ):
                if len(out_obj_ids) == 0:
                    pred = zero
                else:
                    pred = (out_logits[0] > args.score_thresh).detach().cpu().numpy().squeeze()
                save_mask(out_dir / f"{frames[out_idx]}.png", pred)
                seen.add(out_idx)

            for idx, name in enumerate(frames):
                if idx not in seen and idx >= start_idx:
                    save_mask(out_dir / f"{name}.png", zero)

        if hasattr(predictor, "reset_state"):
            try:
                predictor.reset_state(state)
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if skipped:
        skip_path = Path(args.output_dir) / "skipped.txt"
        with skip_path.open("w") as f:
            for video, reason in skipped:
                f.write(f"{video}\t{reason}\n")
        print(f"[WARN] skipped {len(skipped)} videos; see {skip_path}")

    if args.zip:
        zip_base = Path(args.output_dir) / Path(args.output_dir).name
        shutil.make_archive(str(zip_base), "zip", root_dir=args.output_dir, base_dir="Annotations")
        print(f"[OK] zip: {zip_base}.zip")
    print(f"[OK] predictions: {ann_root}")


if __name__ == "__main__":
    main()
