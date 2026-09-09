#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def list_masks(root: Path):
    return sorted(p for p in root.rglob("*.png") if p.is_file())


def mask_array(path: Path, threshold: int):
    return np.asarray(Image.open(path).convert("L")) > threshold


def main():
    parser = argparse.ArgumentParser("Compare two prediction mask folders pixel-wise.")
    parser.add_argument("--a", required=True, type=Path, help="First Annotations directory.")
    parser.add_argument("--b", required=True, type=Path, help="Second Annotations directory.")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--threshold", type=int, default=127)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    if not args.a.exists():
        raise FileNotFoundError(args.a)
    if not args.b.exists():
        raise FileNotFoundError(args.b)

    files = list_masks(args.a)
    if args.max_files > 0:
        files = files[: args.max_files]

    common = 0
    missing = 0
    diff_files = 0
    total_pixels = 0
    xor_pixels = 0
    area_a = 0
    area_b = 0
    example_rows = []

    for path_a in tqdm(files, desc="compare", ncols=0):
        rel = path_a.relative_to(args.a)
        path_b = args.b / rel
        if not path_b.exists():
            missing += 1
            continue

        mask_a = mask_array(path_a, args.threshold)
        mask_b = mask_array(path_b, args.threshold)
        if mask_a.shape != mask_b.shape:
            raise ValueError(f"Shape mismatch for {rel}: {mask_a.shape} vs {mask_b.shape}")

        diff = np.logical_xor(mask_a, mask_b).sum()
        common += 1
        total_pixels += mask_a.size
        xor_pixels += int(diff)
        area_a += int(mask_a.sum())
        area_b += int(mask_b.sum())
        if diff > 0:
            diff_files += 1
            if len(example_rows) < args.examples:
                example_rows.append((rel, int(diff), int(mask_a.sum()), int(mask_b.sum())))

    xor_rate = xor_pixels / max(total_pixels, 1)
    print(f"{args.name_a} vs {args.name_b}")
    print(f"common_files={common}")
    print(f"missing_in_b={missing}")
    print(f"diff_files={diff_files}")
    print(f"xor_pixels={xor_pixels}")
    print(f"xor_rate={xor_rate:.8f}")
    print(f"area_{args.name_a}={area_a}")
    print(f"area_{args.name_b}={area_b}")
    if example_rows:
        print("examples:")
        for rel, diff, sum_a, sum_b in example_rows:
            print(f"  {rel} diff={diff} area_a={sum_a} area_b={sum_b}")


if __name__ == "__main__":
    main()
