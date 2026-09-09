#!/usr/bin/env python3
"""Run MemSAM inference on MBTUVS and save masks for metric aggregation."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image
from tqdm import tqdm


def add_repo_to_path(repo: Path):
    sys.path.insert(0, str(repo))


def load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)


def load_mask(path: Path, size: int) -> np.ndarray:
    if not path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.uint8)


def original_size(path: Path):
    img = Image.open(path)
    return img.size[1], img.size[0]


def fixed_click(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([[mask.shape[1] // 2, mask.shape[0] // 2]], dtype=np.float32), np.array([0], dtype=np.int64)
    cx, cy = xs.mean(), ys.mean()
    center_idx = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
    return np.array([[xs[center_idx], ys[center_idx]]], dtype=np.float32), np.array([1], dtype=np.int64)


def list_videos(dataset_root: Path, split: str, max_videos: int = 0):
    meta = dataset_root / split / "cleaned_output.json"
    if meta.exists():
        with meta.open("r") as f:
            videos = sorted(json.load(f)["videos"].keys())
    else:
        videos = sorted(p.name for p in (dataset_root / split / "JPEGImages").iterdir() if p.is_dir())
    return videos[:max_videos] if max_videos > 0 else videos


def first_nonempty(gt_dir: Path, frames, size: int):
    for idx, name in enumerate(frames):
        mask = load_mask(gt_dir / f"{name}.png", size)
        if mask.any():
            return idx, mask
    return 0, np.zeros((size, size), dtype=np.uint8)


def build_memsam(args):
    add_repo_to_path(Path(args.memsam_repo).resolve())
    from models.model_dict import get_model

    model_args = EasyDict(
        modelname="MemSAM",
        encoder_input_size=args.image_size,
        low_image_size=args.image_size,
        batch_size=1,
        n_gpu=1,
        sam_ckpt=args.sam_ckpt,
        vit_name="vit_b",
        enable_memory=not args.disable_memory,
        reinforce=args.reinforce,
        disable_point_prompt=False,
    )
    return get_model("MemSAM", args=model_args, opt=None)


def save_pred(path: Path, pred_256: np.ndarray, size_hw):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(pred_256.astype(np.uint8) * 255).resize((size_hw[1], size_hw[0]), Image.NEAREST)
    img.save(path)


@torch.no_grad()
def infer_clip(model, imgs_np, prompt_mask, device, threshold=0.5):
    t = len(imgs_np)
    imgs = torch.from_numpy(np.stack(imgs_np)).float().unsqueeze(0).to(device)
    pt, label = fixed_click(prompt_mask)
    pts = torch.from_numpy(np.repeat(pt[None, None, ...], t, axis=1)).float().to(device)
    labels = torch.from_numpy(label[None, ...]).long().to(device)
    logits = model(imgs, (pts, labels), None)
    return (torch.sigmoid(logits[0, :, 0]) > threshold).detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memsam-repo", default="method/MemSAM-main")
    parser.add_argument("--sam-ckpt", default="method/MemSAM-main/sam_vit_b_01ec64.pth")
    parser.add_argument("--checkpoint", required=True, help="MemSAM checkpoint from train_memsam_mbtuvs.py")
    parser.add_argument("--dataset-root", default="data/processed_breast_ultrasound_5frames")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/rebuttal_baselines/memsam_mbtuvs_eval")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames-per-pass", type=int, default=0, help="0 means full video in one pass; >0 uses a new GT first-point per chunk.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_memsam(args).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()

    root = Path(args.dataset_root)
    img_root = root / args.split / "JPEGImages"
    gt_root = root / args.split / "Annotations"
    ann_root = Path(args.output_dir) / "Annotations"
    ann_root.mkdir(parents=True, exist_ok=True)

    videos = list_videos(root, args.split, args.max_videos)
    for video in tqdm(videos, desc="MemSAM inference"):
        img_dir = img_root / video
        gt_dir = gt_root / video
        frames = sorted(p.stem for p in img_dir.glob("*.jpg"))
        if not frames:
            continue
        out_dir = ann_root / video / "exp0"
        out_dir.mkdir(parents=True, exist_ok=True)
        start_idx, prompt_mask = first_nonempty(gt_dir, frames, args.image_size)


        if args.max_frames_per_pass and args.max_frames_per_pass > 0:
            chunk_starts = list(range(start_idx, len(frames), args.max_frames_per_pass))
        else:
            chunk_starts = [start_idx]

        for chunk_start in chunk_starts:
            chunk_end = len(frames) if not args.max_frames_per_pass else min(len(frames), chunk_start + args.max_frames_per_pass)
            names = frames[chunk_start:chunk_end]
            imgs = [load_image(img_dir / f"{name}.jpg", args.image_size) for name in names]
            chunk_prompt = prompt_mask if chunk_start == start_idx else load_mask(gt_dir / f"{frames[chunk_start]}.png", args.image_size)
            preds = infer_clip(model, imgs, chunk_prompt, device, threshold=args.threshold)
            for name, pred in zip(names, preds):
                size_hw = original_size(img_dir / f"{name}.jpg")
                save_pred(out_dir / f"{name}.png", pred, size_hw)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.zip:
        zip_base = Path(args.output_dir) / Path(args.output_dir).name
        shutil.make_archive(str(zip_base), "zip", root_dir=args.output_dir, base_dir="Annotations")
        print(f"[OK] zip: {zip_base}.zip")
    print(f"[OK] predictions: {ann_root}")


if __name__ == "__main__":
    main()
