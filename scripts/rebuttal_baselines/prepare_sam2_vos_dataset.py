#!/usr/bin/env python3
"""Prepare a DAVIS/MOSE-style VOS view of MBTUVS for official SAM2 tools.

This is optional. The zero-shot prompt baseline can be run directly with
run_sam2_mbtuvs.py. This converter is useful if we later decide to use SAM2's
official `tools/vos_inference.py` or training configs.
"""

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm


def videos_from_meta(dataset_root: Path, split: str, max_videos: int = 0):
    meta = dataset_root / split / "cleaned_output.json"
    if meta.exists():
        with meta.open("r") as f:
            videos = sorted(json.load(f)["videos"].keys())
    else:
        videos = sorted(p.name for p in (dataset_root / split / "JPEGImages").iterdir() if p.is_dir())
    return videos[:max_videos] if max_videos > 0 else videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/processed_breast_ultrasound_5frames")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--output-root", default="outputs/rebuttal_baselines/mbtuvs_sam2_vos")
    parser.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.output_root)
    img_out = out_root / "JPEGImages"
    ann_out = out_root / "Annotations"
    lists_out = out_root / "ImageSets"
    lists_out.mkdir(parents=True, exist_ok=True)

    linker = shutil.copy2 if args.copy else lambda src, dst: Path(dst).symlink_to(Path(src).resolve())

    for split in args.splits:
        videos = videos_from_meta(dataset_root, split, args.max_videos)
        with (lists_out / f"{split}.txt").open("w") as f:
            f.write("\n".join(videos) + "\n")

        for video in tqdm(videos, desc=f"prepare {split}"):
            src_img_dir = dataset_root / split / "JPEGImages" / video
            src_ann_dir = dataset_root / split / "Annotations" / video
            dst_img_dir = img_out / split / video
            dst_ann_dir = ann_out / split / video
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_ann_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(src_img_dir.glob("*.jpg")):
                dst = dst_img_dir / src.name
                if not dst.exists():
                    linker(src, dst)
            for src in sorted(src_ann_dir.glob("*.png")):
                dst = dst_ann_dir / src.name
                if not dst.exists():
                    linker(src, dst)

    print(f"[OK] SAM2 VOS view: {out_root}")
    print(f"[OK] lists: {lists_out}")


if __name__ == "__main__":
    main()
