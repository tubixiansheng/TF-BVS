#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def run(cmd):
    print("\n[RUN]", " ".join(cmd), flush=True)
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    subprocess.run(cmd, check=True, env=env)


def unzip_annotations(zip_path: Path, output_root: Path):
    annotations = output_root / "Annotations"
    if annotations.exists():
        shutil.rmtree(annotations)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_root)
    if not annotations.exists():
        raise RuntimeError(f"Annotations not found after unzip: {zip_path}")
    return annotations


def main():
    parser = argparse.ArgumentParser("Run backbone inference, unzip masks, branch fusion, and optional metric evaluation.")
    parser.add_argument("-c", "--config", default="configs/ytvos_swinb.yaml")
    parser.add_argument("-ckpt", "--checkpoint", required=True)
    parser.add_argument("-ng", "--num-gpus", type=int, default=4)
    parser.add_argument("--dataset-root", default="data/processed_breast_ultrasound_5frames")
    parser.add_argument("--dataset-path", default=None,
                        help="dataset root passed to backbone inference; defaults to YAML")
    parser.add_argument("--split", default="test")
    parser.add_argument("--backbone-version", required=True)
    parser.add_argument("--fusion-version", required=True)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--force-label-mode", default="none",
                        choices=["none", "gt", "wrong", "random", "high_low_swap", "adjacent_swap"])
    parser.add_argument("--strong-label-prompt", choices=["true", "false"], default=None)
    parser.add_argument("--label-residual-min-scale", type=float, default=None)
    parser.add_argument("--label-prompt-film", choices=["true", "false"], default=None)
    parser.add_argument("--label-prompt-film-scale", type=float, default=None)
    parser.add_argument("--grade-free-prompt", choices=["true", "false"], default=None)
    parser.add_argument("--use-simple-embed", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--metrics-name", default=None)
    parser.add_argument("--output-dir", default="outputs/rebuttal_metrics")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    backbone_cmd = [
        args.python, "eval/inference_backbone_only.py",
        "-c", args.config,
        "-ng", str(args.num_gpus),
        "-ckpt", args.checkpoint,
        "--version", args.backbone_version,
        "--force_textfree",
        "--frame_step", str(args.frame_step),
        "--split", args.split,
    ]
    if args.dataset_path:
        backbone_cmd.extend(["--dataset_path", args.dataset_path])
    if args.force_label_mode != "none":
        backbone_cmd.extend(["--force_label_mode", args.force_label_mode])
    if args.use_simple_embed:
        backbone_cmd.append("--use_simple_embed")
    if args.strong_label_prompt is not None:
        backbone_cmd.extend(["--strong_label_prompt", args.strong_label_prompt])
    if args.label_residual_min_scale is not None:
        backbone_cmd.extend(["--label_residual_min_scale", str(args.label_residual_min_scale)])
    if args.label_prompt_film is not None:
        backbone_cmd.extend(["--label_prompt_film", args.label_prompt_film])
    if args.label_prompt_film_scale is not None:
        backbone_cmd.extend(["--label_prompt_film_scale", str(args.label_prompt_film_scale)])
    if args.grade_free_prompt is not None:
        backbone_cmd.extend(["--grade_free_prompt", args.grade_free_prompt])
    run(backbone_cmd)

    backbone_root = Path("output") / "ref_youtube_vos" / args.backbone_version
    backbone_zip = backbone_root / f"{args.backbone_version}.zip"
    annotations = unzip_annotations(backbone_zip, backbone_root)

    fusion_cmd = [
        args.python, "eval/fusion.py",
        "-c", args.config,
        "-ng", str(args.num_gpus),
        "-ckpt", args.checkpoint,
        "--version", args.fusion_version,
        "--dataset_root", args.dataset_root,
        "--split", args.split,
        "--fill_before_first", "skip",
        "--gt_exts", "png,jpg",
        "--inference_result_dir", str(annotations),
    ]
    run(fusion_cmd)

    if not args.skip_eval:
        metrics_name = args.metrics_name or args.fusion_version
        fusion_zip = Path("output") / "ref_youtube_vos" / args.fusion_version / f"{args.fusion_version}.zip"
        eval_cmd = [
            args.python, "scripts/rebuttal/evaluate_meter_mbtuvs.py",
            "--pred", str(fusion_zip),
            "--dataset-root", args.dataset_root,
            "--split", args.split,
            "--name", metrics_name,
            "--output-dir", args.output_dir,
        ]
        run(eval_cmd)


if __name__ == "__main__":
    main()
