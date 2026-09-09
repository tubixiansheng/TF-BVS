#!/usr/bin/env python3
"""Aggregate segmentation metrics for rebuttal experiments.

The script compares a prediction directory (or zip) with the MBTUVS ground-truth
mask directory and reports overall/per-grade metrics. It supports the common
prediction layout:

  Annotations/<video>/<exp_id>/<frame>.png

and the ground-truth layout:

  <dataset_root>/<split>/Annotations/<video>/<frame>.png
"""

import argparse
import csv
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


GRADE_RE = re.compile(r"-L([2-5])-")


def parse_grade(video_name: str) -> str:
    match = GRADE_RE.search(video_name)
    return f"L{match.group(1)}" if match else "unknown"


def as_annotations_root(path: Path) -> Path:
    if path.name == "Annotations":
        return path
    child = path / "Annotations"
    return child if child.exists() else path


def load_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 127


def boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask ^ eroded


def boundary_f(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    pred_b = boundary(pred)
    gt_b = boundary(gt)
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0

    pred_match = ndimage.binary_dilation(gt_b, iterations=tol) & pred_b
    gt_match = ndimage.binary_dilation(pred_b, iterations=tol) & gt_b
    precision = pred_match.sum() / max(pred_b.sum(), 1)
    recall = gt_match.sum() / max(gt_b.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_b = boundary(pred)
    gt_b = boundary(gt)
    if not pred_b.any() and not gt_b.any():
        return np.array([0.0], dtype=np.float64)
    diag = float(np.hypot(*pred.shape))
    if not pred_b.any() or not gt_b.any():
        return np.array([diag], dtype=np.float64)

    dt_gt = ndimage.distance_transform_edt(~gt_b)
    dt_pred = ndimage.distance_transform_edt(~pred_b)
    return np.concatenate([dt_gt[pred_b], dt_pred[gt_b]]).astype(np.float64)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, fast: bool = False) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if union == 0:
        iou = 1.0
    else:
        iou = inter / union
    if pred_sum + gt_sum == 0:
        dice = 1.0
    else:
        dice = 2 * inter / (pred_sum + gt_sum)

    if fast:
        hd95 = float("nan")
        asd = float("nan")
    else:
        dists = surface_distances(pred, gt)
        hd95 = float(np.percentile(dists, 95))
        asd = float(dists.mean())
    return {
        "IoU": float(iou),
        "F-measure": boundary_f(pred, gt),
        "Dice": float(dice),
        "HD95": hd95,
        "ASD": asd,
    }


def iter_prediction_files(pred_root: Path):
    for video_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        exp_dirs = sorted(p for p in video_dir.iterdir() if p.is_dir())
        if exp_dirs:
            for exp_dir in exp_dirs:
                for pred_file in sorted(exp_dir.glob("*.png")):
                    yield video_dir.name, exp_dir.name, pred_file
        else:
            for pred_file in sorted(video_dir.glob("*.png")):
                yield video_dir.name, "exp0", pred_file


def mean_row(rows, key=None, value=None):
    subset = rows if key is None else [r for r in rows if r[key] == value]
    if not subset:
        return None
    out = {"group": value if value is not None else "overall", "n": len(subset)}
    for metric in ["IoU", "F-measure", "Dice", "HD95", "ASD"]:
        out[metric] = float(np.mean([float(r[metric]) for r in subset]))
    return out


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Prediction Annotations directory or zip file")
    parser.add_argument("--dataset-root", default="data/processed_breast_ultrasound_5frames")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="outputs/rebuttal_metrics")
    parser.add_argument("--name", default=None)
    parser.add_argument("--fast", action="store_true", help="Skip HD95/ASD distance transforms")
    args = parser.parse_args()

    pred_input = Path(args.pred)
    tmp_dir = None
    if pred_input.suffix == ".zip":
        tmp_dir = Path(tempfile.mkdtemp(prefix="rebuttal_pred_"))
        with zipfile.ZipFile(pred_input) as zf:
            zf.extractall(tmp_dir)
        pred_root = as_annotations_root(tmp_dir)
    else:
        pred_root = as_annotations_root(pred_input)

    gt_root = Path(args.dataset_root) / args.split / "Annotations"
    name = args.name or pred_input.stem
    output_dir = Path(args.output_dir)
    raw_rows = []
    missing = 0

    try:
        for video, exp_id, pred_file in iter_prediction_files(pred_root):
            gt_file = gt_root / video / pred_file.name
            if not gt_file.exists():
                missing += 1
                continue
            pred = load_mask(pred_file)
            gt = load_mask(gt_file)
            if pred.shape != gt.shape:
                pred_img = Image.fromarray(pred.astype(np.uint8) * 255).resize(
                    (gt.shape[1], gt.shape[0]), resample=Image.NEAREST
                )
                pred = np.asarray(pred_img) > 127
            metrics = compute_metrics(pred, gt, fast=args.fast)
            raw_rows.append({
                "video": video,
                "grade": parse_grade(video),
                "exp_id": exp_id,
                "frame": pred_file.name,
                **metrics,
            })
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not raw_rows:
        raise RuntimeError(f"No matched prediction/GT masks found. pred={pred_input}, gt={gt_root}")

    raw_path = output_dir / f"{name}_per_frame.csv"
    summary_path = output_dir / f"{name}_summary.csv"
    fields = ["video", "grade", "exp_id", "frame", "IoU", "F-measure", "Dice", "HD95", "ASD"]
    write_csv(raw_path, raw_rows, fields)

    summary = [mean_row(raw_rows)]
    for grade in ["L2", "L3", "L4", "L5", "unknown"]:
        row = mean_row(raw_rows, "grade", grade)
        if row is not None:
            summary.append(row)
    write_csv(summary_path, summary, ["group", "n", "IoU", "F-measure", "Dice", "HD95", "ASD"])

    print(f"[OK] matched frames: {len(raw_rows)}, missing predictions skipped: {missing}")
    print(f"[OK] per-frame CSV: {raw_path}")
    print(f"[OK] summary CSV: {summary_path}")
    for row in summary:
        print(
            f"{row['group']:>8s} n={row['n']:5d} "
            f"J={row['IoU']:.4f} F={row['F-measure']:.4f} "
            f"Dice={row['Dice']:.4f} HD95={row['HD95']:.4f} ASD={row['ASD']:.4f}"
        )


if __name__ == "__main__":
    main()
