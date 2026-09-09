#!/usr/bin/env python3
"""Fine-tune MemSAM on MBTUVS as a single-modal medical video baseline.

MemSAM is not a language/multimodal model in the official implementation. This
script adapts its video + point-prompt training loop to MBTUVS without injecting
diagnostic reports, so the rebuttal can report it as a strong single-modal
medical video reference.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def add_repo_to_path(repo: Path):
    sys.path.insert(0, str(repo))


def load_mask(path: Path, size: int) -> np.ndarray:
    if not path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.uint8)


def load_image(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


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


class MBTUVSClipDataset(Dataset):
    def __init__(self, dataset_root, split, clip_length=10, image_size=256, max_videos=0, train=True):
        self.root = Path(dataset_root)
        self.split = split
        self.clip_length = int(clip_length)
        self.image_size = int(image_size)
        self.train = train
        self.videos = list_videos(self.root, split, max_videos)

    def __len__(self):
        return len(self.videos)

    def _frames(self, video):
        return sorted(p.stem for p in (self.root / self.split / "JPEGImages" / video).glob("*.jpg"))

    def _nonempty_indices(self, video, frames):
        gt_dir = self.root / self.split / "Annotations" / video
        out = []
        for i, name in enumerate(frames):
            mask_path = gt_dir / f"{name}.png"
            if mask_path.exists() and np.asarray(Image.open(mask_path).convert("L")).max() > 127:
                out.append(i)
        return out

    def __getitem__(self, index):
        video = self.videos[index]
        frames = self._frames(video)
        nonempty = self._nonempty_indices(video, frames)
        if not frames:
            raise RuntimeError(f"No frames found for {video}")
        if nonempty:
            start = random.choice(nonempty) if self.train else nonempty[0]
        else:
            start = 0
        idxs = list(range(start, min(start + self.clip_length, len(frames))))
        while len(idxs) < self.clip_length:
            idxs.append(idxs[-1])

        img_dir = self.root / self.split / "JPEGImages" / video
        gt_dir = self.root / self.split / "Annotations" / video
        imgs, masks = [], []
        for i in idxs:
            name = frames[i]
            imgs.append(load_image(img_dir / f"{name}.jpg", self.image_size).transpose(2, 0, 1))
            masks.append(load_mask(gt_dir / f"{name}.png", self.image_size))

        imgs = torch.from_numpy(np.stack(imgs)).float()
        masks_np = np.stack(masks).astype(np.float32)
        masks = torch.from_numpy(masks_np)
        valid = torch.from_numpy((masks_np.reshape(self.clip_length, -1).max(axis=1) > 0).astype(np.float32))
        pt, p_label = fixed_click(masks_np[0])

        pts = torch.from_numpy(np.repeat(pt[None, ...], self.clip_length, axis=0)).float()
        labels = torch.from_numpy(p_label).long()
        return {
            "image": imgs,
            "label": masks,
            "valid": valid,
            "pt": pts,
            "p_label": labels,
            "video": video,
            "start_frame": frames[idxs[0]],
        }


def dice_bce_loss(logits, target, valid):
    target = target.unsqueeze(2).float()
    valid = valid.bool()
    if not valid.any():
        valid = torch.ones_like(valid, dtype=torch.bool)

    bce_frame = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(2, 3, 4))
    bce = bce_frame[valid].mean()
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3, 4))
    denom = prob.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice_frame = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0))
    dice = dice_frame[valid].mean()
    return bce + dice


def build_memsam(args):
    add_repo_to_path(Path(args.memsam_repo).resolve())
    from models.model_dict import get_model

    model_args = EasyDict(
        modelname="MemSAM",
        encoder_input_size=args.image_size,
        low_image_size=args.image_size,
        batch_size=args.batch_size,
        n_gpu=1,
        sam_ckpt=args.sam_ckpt,
        vit_name="vit_b",
        enable_memory=not args.disable_memory,
        reinforce=args.reinforce,
        disable_point_prompt=False,
    )
    return get_model("MemSAM", args=model_args, opt=None)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    dices, ious, n_frames = [], [], 0
    for batch in tqdm(loader, desc="val", leave=False):
        imgs = batch["image"].to(device)
        masks = batch["label"].to(device)
        valid = batch["valid"].to(device).bool()
        pt = (batch["pt"].to(device), batch["p_label"].to(device))
        logits = model(imgs, pt, None)
        pred = torch.sigmoid(logits[:, :, 0]) > threshold
        gt = masks > 0.5
        pred = pred.flatten(0, 1)
        gt = gt.flatten(0, 1)
        valid = valid.flatten()
        if not valid.any():
            continue
        pred = pred[valid]
        gt = gt[valid]
        inter = (pred & gt).sum(dim=(1, 2)).float()
        union = (pred | gt).sum(dim=(1, 2)).float()
        denom = pred.sum(dim=(1, 2)).float() + gt.sum(dim=(1, 2)).float()
        ious.extend(((inter + 1.0) / (union + 1.0)).detach().cpu().tolist())
        dices.extend(((2 * inter + 1.0) / (denom + 1.0)).detach().cpu().tolist())
        n_frames += int(valid.sum().item())
    if not dices:
        return 0.0, 0.0, 0
    return float(np.mean(dices)), float(np.mean(ious)), n_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memsam-repo", default="method/MemSAM-main")
    parser.add_argument("--sam-ckpt", default="method/MemSAM-main/sam_vit_b_01ec64.pth")
    parser.add_argument("--dataset-root", default="data/processed_breast_ultrasound_5frames")
    parser.add_argument("--output-dir", default="outputs/rebuttal_baselines/memsam_mbtuvs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep 1: MemSAM upstream point-label path assumes bs=1.")
    parser.add_argument("--clip-length", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-videos", type=int, default=0)
    parser.add_argument("--max-val-videos", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("Please keep --batch-size 1; upstream MemSAM uses the first batch label in its prompt path.")

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_memsam(args).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=False)

    train_set = MBTUVSClipDataset(args.dataset_root, "train", args.clip_length, args.image_size, args.max_train_videos, train=True)
    val_set = MBTUVSClipDataset(args.dataset_root, "valid", args.clip_length, args.image_size, args.max_val_videos, train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_dice = -1.0
    log_path = out_dir / "train_log.csv"
    with log_path.open("w") as f:
        f.write("epoch,train_loss,val_dice,val_iou,val_frames\n")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            imgs = batch["image"].to(device)
            masks = batch["label"].to(device)
            valid = batch["valid"].to(device)
            pt = (batch["pt"].to(device), batch["p_label"].to(device))
            logits = model(imgs, pt, None)
            loss = dice_bce_loss(logits, masks, valid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_dice, val_iou, val_frames = evaluate(model, val_loader, device, threshold=args.eval_threshold)
        train_loss = float(np.mean(losses))
        with log_path.open("a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_dice:.6f},{val_iou:.6f},{val_frames}\n")
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_dice={val_dice:.6f} val_iou={val_iou:.6f} val_frames={val_frames}")

        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "val_dice": val_dice, "args": vars(args)}
        torch.save(state, ckpt_dir / "last.pth")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(state, ckpt_dir / "best.pth")

    print(f"[OK] best checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
