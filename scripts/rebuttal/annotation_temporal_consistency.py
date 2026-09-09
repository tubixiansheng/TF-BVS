#!/usr/bin/env python3
"""Compute temporal consistency of final consensus annotations.

This script is a fallback quality analysis when repeated masks from multiple
annotators are not available. It does not compute true inter-annotator
agreement. Instead, it selects videos with stable foreground masks and reports
adjacent annotated-frame Dice/IoU plus ICC on mask-derived scalar area.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/rebuttal_annotation_quality/temporal_consistency")
    parser.add_argument("--mask-exts", default="png,jpg,jpeg")
    return parser.parse_args()


def frame_index(path):
    try:
        return int(path.stem)
    except ValueError:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else 0


def load_mask(path):
    mask = np.asarray(Image.open(path).convert("L"))
    return mask > 0


def mask_stats(mask):
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    if area == 0:
        return area, math.nan, math.nan
    return area, float(xs.mean()), float(ys.mean())


def dice_score(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def iou_score(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def icc_absolute_agreement(data):
    """ICC(A,1), two-way random absolute agreement, for an n x k matrix."""
    x = np.asarray(data, dtype=np.float64)
    x = x[np.all(np.isfinite(x), axis=1)]
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return math.nan

    mean_rows = x.mean(axis=1, keepdims=True)
    mean_cols = x.mean(axis=0, keepdims=True)
    grand = x.mean()

    ms_rows = k * np.sum((mean_rows - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((mean_cols - grand) ** 2) / (k - 1)
    residual = x - mean_rows - mean_cols + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-12:
        return math.nan
    return float((ms_rows - ms_error) / denom)


def icc_consistency(data):
    """ICC(C,1), two-way mixed consistency, for an n x k matrix."""
    x = np.asarray(data, dtype=np.float64)
    x = x[np.all(np.isfinite(x), axis=1)]
    n, k = x.shape if x.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return math.nan

    mean_rows = x.mean(axis=1, keepdims=True)
    mean_cols = x.mean(axis=0, keepdims=True)
    grand = x.mean()

    ms_rows = k * np.sum((mean_rows - grand) ** 2) / (n - 1)
    residual = x - mean_rows - mean_cols + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error
    if abs(denom) < 1e-12:
        return math.nan
    return float((ms_rows - ms_error) / denom)


def collect_video_pairs(split, video_dir, exts):
    mask_paths = []
    for ext in exts:
        mask_paths.extend(video_dir.glob(f"*.{ext}"))
    mask_paths = sorted(mask_paths, key=frame_index)

    nonempty = []
    for path in mask_paths:
        mask = load_mask(path)
        area, cx, cy = mask_stats(mask)
        if area > 0:
            nonempty.append((path, mask, area, cx, cy))

    rows = []
    diag = None
    for left, right in zip(nonempty[:-1], nonempty[1:]):
        path_a, mask_a, area_a, cx_a, cy_a = left
        path_b, mask_b, area_b, cx_b, cy_b = right
        if diag is None:
            h, w = mask_a.shape
            diag = math.sqrt(h * h + w * w)
        center_dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) / max(diag, 1e-6)
        area_rel_diff = abs(area_a - area_b) / max((area_a + area_b) / 2.0, 1.0)
        rows.append(
            {
                "split": split,
                "video": video_dir.name,
                "frame_a": path_a.stem,
                "frame_b": path_b.stem,
                "dice": dice_score(mask_a, mask_b),
                "iou": iou_score(mask_a, mask_b),
                "area_a": area_a,
                "area_b": area_b,
                "area_rel_diff": area_rel_diff,
                "center_dist_norm": center_dist,
            }
        )
    return rows


def mean_or_nan(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else math.nan


def std_or_nan(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.std(ddof=1)) if arr.size > 1 else math.nan


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    exts = [x.strip().lower() for x in args.mask_exts.split(",") if x.strip()]

    video_summaries = []
    all_pairs_by_video = {}

    for split in args.splits:
        ann_root = root / split / "Annotations"
        for video_dir in sorted(p for p in ann_root.iterdir() if p.is_dir()):
            pairs = collect_video_pairs(split, video_dir, exts)
            if len(pairs) < args.min_pairs:
                continue
            dice_mean = mean_or_nan([r["dice"] for r in pairs])
            area_cv = np.std([r["area_a"] for r in pairs] + [r["area_b"] for r in pairs]) / max(
                np.mean([r["area_a"] for r in pairs] + [r["area_b"] for r in pairs]), 1e-6
            )
            center_mean = mean_or_nan([r["center_dist_norm"] for r in pairs])
            stability_score = dice_mean - 0.25 * area_cv - 0.50 * center_mean
            all_pairs_by_video[(split, video_dir.name)] = pairs
            video_summaries.append(
                {
                    "split": split,
                    "video": video_dir.name,
                    "n_pairs": len(pairs),
                    "mean_dice": dice_mean,
                    "mean_iou": mean_or_nan([r["iou"] for r in pairs]),
                    "mean_area_rel_diff": mean_or_nan([r["area_rel_diff"] for r in pairs]),
                    "mean_center_dist_norm": center_mean,
                    "area_cv": float(area_cv),
                    "stability_score": float(stability_score),
                }
            )

    selected = sorted(video_summaries, key=lambda r: r["stability_score"], reverse=True)[: args.top_k]
    selected_keys = {(r["split"], r["video"]) for r in selected}
    selected_pairs = []
    for key in selected_keys:
        selected_pairs.extend(all_pairs_by_video[key])
    selected_pairs = sorted(selected_pairs, key=lambda r: (r["split"], r["video"], frame_index(Path(r["frame_a"]))))

    area_pairs = np.asarray([[r["area_a"], r["area_b"]] for r in selected_pairs], dtype=np.float64)
    summary = [
        {
            "scope": "selected_pairs",
            "n_videos": len(selected),
            "n_pairs": len(selected_pairs),
            "mean_dice": mean_or_nan([r["dice"] for r in selected_pairs]),
            "std_dice": std_or_nan([r["dice"] for r in selected_pairs]),
            "mean_iou": mean_or_nan([r["iou"] for r in selected_pairs]),
            "std_iou": std_or_nan([r["iou"] for r in selected_pairs]),
            "mean_area_rel_diff": mean_or_nan([r["area_rel_diff"] for r in selected_pairs]),
            "mean_center_dist_norm": mean_or_nan([r["center_dist_norm"] for r in selected_pairs]),
            "area_icc_absolute": icc_absolute_agreement(area_pairs),
            "area_icc_consistency": icc_consistency(area_pairs),
        }
    ]

    write_csv(
        output_dir / "selected_videos.csv",
        selected,
        [
            "split",
            "video",
            "n_pairs",
            "mean_dice",
            "mean_iou",
            "mean_area_rel_diff",
            "mean_center_dist_norm",
            "area_cv",
            "stability_score",
        ],
    )
    write_csv(
        output_dir / "selected_pairs.csv",
        selected_pairs,
        [
            "split",
            "video",
            "frame_a",
            "frame_b",
            "dice",
            "iou",
            "area_a",
            "area_b",
            "area_rel_diff",
            "center_dist_norm",
        ],
    )
    write_csv(
        output_dir / "summary.csv",
        summary,
        [
            "scope",
            "n_videos",
            "n_pairs",
            "mean_dice",
            "std_dice",
            "mean_iou",
            "std_iou",
            "mean_area_rel_diff",
            "mean_center_dist_norm",
            "area_icc_absolute",
            "area_icc_consistency",
        ],
    )

    print(f"[OK] selected videos: {len(selected)}")
    print(f"[OK] selected pairs: {len(selected_pairs)}")
    print(f"[OK] output dir: {output_dir}")
    for key, value in summary[0].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
