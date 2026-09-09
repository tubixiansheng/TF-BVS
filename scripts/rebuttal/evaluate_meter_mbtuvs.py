#!/usr/bin/env python3
"""Evaluate MBTUVS predictions with the paper's meter.py metric definitions.

This wrapper reuses metric functions from the repository-level `meter.py`:
Jaccard/IoU, DAVIS-style boundary F-measure, Dice, HD95 and ASD. It adds
argument parsing, zip support, common prediction layouts, and per-grade summary
for rebuttal experiments.
"""

import argparse
import csv
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from meter import db_eval_boundary, db_eval_iou, dice_score, is_non_empty_mask
from medpy.metric.binary import assd, hd95


GRADE_RE = re.compile(r"-L([2-5])-")


def parse_grade(video_name: str) -> str:
    match = GRADE_RE.search(video_name)
    return f"L{match.group(1)}" if match else "unknown"


def as_annotations_root(path: Path) -> Path:
    if path.name == "Annotations":
        return path
    child = path / "Annotations"
    return child if child.exists() else path


def pred_path_for(pred_root: Path, video: str, frame_name: str, exp_id: str):
    candidates = [
        pred_root / video / exp_id / frame_name,
        pred_root / video / "exp0" / frame_name,
        pred_root / video / frame_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def mean_row(rows, group="overall"):
    if not rows:
        return None
    out = {"group": group, "n": len(rows)}
    for key in ["IoU", "F-measure", "Dice", "HD95", "ASD"]:
        out[key] = float(np.nanmean([r[key] for r in rows]))
    return out


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_gt_frames(gt_root: Path):
    items = []
    for video in sorted(os.listdir(gt_root)):
        gt_dir = gt_root / video
        if not gt_dir.is_dir():
            continue
        for gt_path in sorted(gt_dir.glob("*.png")):
            items.append((video, gt_path))
    return items


def evaluate(gt_root: Path, pred_root: Path, exp_id: str, skip_empty_gt: bool, show_progress: bool = True):
    records = []
    missing = 0
    gt_items = iter_gt_frames(gt_root)
    iterator = tqdm(gt_items, desc="metrics", ncols=0, disable=not show_progress)
    for video, gt_path in iterator:
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        if skip_empty_gt and not is_non_empty_mask(gt):
            continue
        pred_path = pred_path_for(pred_root, video, gt_path.name, exp_id)
        if pred_path is None:
            missing += 1
            continue
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_bin = (gt > 0).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)

        try:
            haus95 = hd95(pred_bin, gt_bin)
        except Exception:
            haus95 = np.nan
        try:
            surface = assd(pred_bin, gt_bin)
        except Exception:
            surface = np.nan

        records.append({
            "video": video,
            "grade": parse_grade(video),
            "frame": gt_path.name,
            "IoU": db_eval_iou(gt_bin, pred_bin),
            "F-measure": db_eval_boundary(gt_bin, pred_bin),
            "Dice": dice_score(gt_bin, pred_bin),
            "HD95": haus95,
            "ASD": surface,
        })
    return records, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Prediction Annotations directory or zip file")
    parser.add_argument("--dataset-root", default="data/processed_breast_ultrasound_5frames")
    parser.add_argument("--split", default="test")
    parser.add_argument("--exp-id", default="exp0")
    parser.add_argument("--output-dir", default="outputs/rebuttal_metrics")
    parser.add_argument("--name", default=None)
    parser.add_argument("--include-empty-gt", action="store_true", help="meter.py skips empty GT by default; this disables that behavior.")
    parser.add_argument("--no-progress", action="store_true", help="Disable zip extraction and metric progress bars.")
    args = parser.parse_args()

    pred_input = Path(args.pred)
    tmp_dir = None
    if pred_input.suffix == ".zip":
        tmp_dir = Path(tempfile.mkdtemp(prefix="meter_pred_"))
        with zipfile.ZipFile(pred_input) as zf:
            members = zf.infolist()
            for member in tqdm(members, desc="extract", ncols=0, disable=args.no_progress):
                zf.extract(member, tmp_dir)
        pred_root = as_annotations_root(tmp_dir)
    else:
        pred_root = as_annotations_root(pred_input)

    gt_root = Path(args.dataset_root) / args.split / "Annotations"
    name = args.name or pred_input.stem
    output_dir = Path(args.output_dir)

    try:
        rows, missing = evaluate(
            gt_root,
            pred_root,
            args.exp_id,
            skip_empty_gt=not args.include_empty_gt,
            show_progress=not args.no_progress,
        )
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not rows:
        raise RuntimeError(f"No matched non-empty GT frames evaluated. pred={pred_input}, gt={gt_root}")

    per_frame_path = output_dir / f"{name}_meter_per_frame.csv"
    summary_path = output_dir / f"{name}_meter_summary.csv"
    fields = ["video", "grade", "frame", "IoU", "F-measure", "Dice", "HD95", "ASD"]
    write_csv(per_frame_path, rows, fields)

    summary = [mean_row(rows, "overall")]
    for grade in ["L2", "L3", "L4", "L5", "unknown"]:
        subset = [r for r in rows if r["grade"] == grade]
        row = mean_row(subset, grade)
        if row is not None:
            summary.append(row)
    write_csv(summary_path, summary, ["group", "n", "IoU", "F-measure", "Dice", "HD95", "ASD"])

    pd.DataFrame(rows).to_csv(per_frame_path, index=False)
    print(f"[OK] meter-compatible evaluated frames: {len(rows)}, missing predictions skipped: {missing}")
    print(f"[OK] per-frame CSV: {per_frame_path}")
    print(f"[OK] summary CSV: {summary_path}")
    for row in summary:
        print(
            f"{row['group']:>8s} n={row['n']:5d} "
            f"J={row['IoU']:.4f} F={row['F-measure']:.4f} Dice={row['Dice']:.4f} "
            f"HD95={row['HD95']:.4f} ASD={row['ASD']:.4f}"
        )


if __name__ == "__main__":
    main()
