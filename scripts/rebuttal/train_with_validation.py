#!/usr/bin/env python3
"""Train one epoch at a time and select the checkpoint on ``valid`` online.

The normal DDP trainer owns all GPUs while it is running, so launching the
full backbone->fusion validation pipeline from inside a DDP rank would either
deadlock or compete for the same GPU memory.  This wrapper therefore creates
an explicit boundary after every epoch:

    train one epoch -> process exits -> validate -> update best_val -> next epoch

The training command is passed after ``--`` and is otherwise kept unchanged.
The wrapper only rewrites ``--epochs`` and the resume checkpoint, so the
optimizer/scheduler state is restored exactly between stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from select_best_validation_checkpoint import (
    REPO_ROOT,
    atomic_copy,
    is_better,
    read_overall_summary,
    score,
    write_csv,
)


CHECKPOINT_EPOCH_RE = re.compile(r"(\d+)\.pth\.tar$")


def find_option(args: list[str], names: set[str]) -> str | None:
    for index, token in enumerate(args):
        if token in names and index + 1 < len(args):
            return args[index + 1]
        for name in names:
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
    return None


def strip_options(args: list[str], names: set[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in names:
            index += 2
            continue
        if any(token.startswith(name + "=") for name in names):
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def checkpoint_epoch(path: Path) -> int:
    match = CHECKPOINT_EPOCH_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer epoch from checkpoint name: {path}")
    return int(match.group(1))


def run(command: list[str]) -> None:
    print("\n[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("epoch", "n"):
                if row.get(key, "") != "":
                    row[key] = int(row[key])
            for key in ("train_loss", "J", "F", "Dice", "HD95", "ASD", "val_score"):
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train epoch-by-epoch and select best_val.pth.tar on valid after every epoch."
    )
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--resume-from", default=None,
                        help="Resume checkpoint for start-epoch; omit for a fresh run")
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0,
                        help="Validation epochs without improvement before stopping; 0 disables early stopping")
    parser.add_argument("--validation-dataset-path", required=True)
    parser.add_argument("--validation-dataset-root", required=True)
    parser.add_argument("--validation-split", default="valid")
    parser.add_argument("--validation-num-gpus", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--score-weights", type=float, nargs=3, default=(0.5, 0.3, 0.2), metavar=("J", "F", "DICE"))
    parser.add_argument("--score-tie-tol", type=float, default=0.001)
    parser.add_argument("--skip-existing-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the per-epoch commands without training or validation")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("train_args", nargs=argparse.REMAINDER,
                        help="Arguments for main.py, after a literal --")
    args = parser.parse_args()

    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    if not args.train_args:
        raise ValueError("Pass the main.py arguments after --")
    if args.total_epochs < 1 or not (0 <= args.start_epoch < args.total_epochs):
        raise ValueError("Require 0 <= start-epoch < total-epochs")
    if args.validate_every < 1:
        raise ValueError("--validate-every must be >= 1")
    if args.validation_split.lower() in {"test", "test_new"}:
        raise ValueError("Training-time selection must use valid, never test/test_new")

    weights = tuple(float(value) for value in args.score_weights)
    if any(value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("--score-weights must be non-negative and not all zero")
    total_weight = sum(weights)
    weights = tuple(value / total_weight for value in weights)

    train_args = list(args.train_args)
    version = find_option(train_args, {"--version"})
    if not version:
        raise ValueError("The main.py command must contain --version")
    config = find_option(train_args, {"-c", "--config_path"}) or "configs/ytvos_swinb.yaml"
    train_gpu_count = find_option(train_args, {"-ng", "--num_gpus"})
    validation_num_gpus = args.validation_num_gpus or int(train_gpu_count or 1)

    if args.resume_from:
        resume_checkpoint = Path(args.resume_from).resolve()
        inferred_start = checkpoint_epoch(resume_checkpoint) + 1
        if args.start_epoch == 0:
            args.start_epoch = inferred_start
        elif args.start_epoch != inferred_start:
            raise ValueError("--start-epoch does not match --resume-from")
    else:
        resume_checkpoint = None

    run_dir = REPO_ROOT / "output" / "ref_youtube_vos" / version
    validation_dir = run_dir / "validation" / args.validation_split
    metric_dir = validation_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    summary_path = validation_dir / "validation_summary.csv"
    rows = load_rows(summary_path)
    best = None
    for existing in sorted(rows, key=lambda row: int(row["epoch"])):
        if is_better(existing, best, args.score_tie_tol):
            best = existing

    if args.start_epoch == 0 and not args.resume_from and not args.dry_run:
        existing_numbered = list((run_dir / "checkpoints").glob("[0-9][0-9].pth.tar"))
        if existing_numbered:
            raise RuntimeError(
                f"{run_dir} already contains numbered checkpoints. "
                "Choose a new --version or pass --resume-from."
            )

    for epoch in range(args.start_epoch, args.total_epochs):
        checkpoint_dir = run_dir / "checkpoints"
        current_checkpoint = checkpoint_dir / f"{epoch:02d}.pth.tar"
        previous_checkpoint = resume_checkpoint if epoch == args.start_epoch and resume_checkpoint else checkpoint_dir / f"{epoch - 1:02d}.pth.tar"

        stage_args = strip_options(train_args, {"--epochs"})
        stage_args = strip_options(stage_args, {"-ckpt", "--checkpoint_path"})
        if epoch > 0 or resume_checkpoint is not None:
            stage_args = strip_options(stage_args, {"-pw", "--pretrained_weights"})
            stage_args.extend(["-ckpt", str(previous_checkpoint)])
        stage_args.extend(["--epochs", str(epoch + 1)])
        train_command = [args.python, "main.py", *stage_args]
        if args.dry_run:
            print(f"[DRY-RUN] epoch {epoch}: ", " ".join(train_command), flush=True)
        else:
            run(train_command)

        if not args.dry_run and not current_checkpoint.exists():
            raise FileNotFoundError(f"Training finished without {current_checkpoint}")

        should_validate = ((epoch - args.start_epoch) % args.validate_every == 0) or epoch == args.total_epochs - 1
        if not should_validate:
            continue

        tag = f"{version}_online_valid_e{epoch:02d}"
        fusion_version = f"{tag}_fusion"
        summary_csv = metric_dir / f"{fusion_version}_meter_summary.csv"
        if not (args.skip_existing_validation and summary_csv.exists()):
            validation_command = [
                args.python, "scripts/rebuttal/run_full_inference_pipeline.py",
                "-c", config,
                "-ckpt", str(current_checkpoint),
                "-ng", str(validation_num_gpus),
                "--dataset-path", str(Path(args.validation_dataset_path).resolve()),
                "--dataset-root", str(Path(args.validation_dataset_root).resolve()),
                "--split", args.validation_split,
                "--backbone-version", f"{tag}_backbone",
                "--fusion-version", fusion_version,
                "--frame-step", str(args.frame_step),
                "--metrics-name", fusion_version,
                "--output-dir", str(metric_dir),
            ]
            if args.dry_run:
                print(f"[DRY-RUN] valid epoch {epoch}: ", " ".join(validation_command), flush=True)
                continue
            run(validation_command)

        if args.dry_run:
            continue

        metrics = read_overall_summary(summary_csv)
        row = {
            "epoch": epoch,
            "checkpoint": str(current_checkpoint),
            "train_loss": "",
            **metrics,
            "val_score": score(metrics, weights),
            "summary_csv": str(summary_csv),
        }
        rows = [old for old in rows if int(old["epoch"]) != epoch]
        rows.append(row)
        rows.sort(key=lambda old: int(old["epoch"]))
        write_csv(summary_path, rows)
        if is_better(row, best, args.score_tie_tol):
            best = row
            atomic_copy(current_checkpoint, checkpoint_dir / "best_val.pth.tar")
            (validation_dir / "best_val.json").write_text(json.dumps({
                "selection": "online_validation",
                "best_epoch": epoch,
                "best_checkpoint": str(current_checkpoint),
                "best_val_checkpoint": str(checkpoint_dir / "best_val.pth.tar"),
                "dataset_root": str(Path(args.validation_dataset_root).resolve()),
                "split": args.validation_split,
                "score_weights": {"J": weights[0], "F": weights[1], "Dice": weights[2]},
                "score_tie_tol": args.score_tie_tol,
                "metrics": {key: row[key] for key in ("n", "J", "F", "Dice", "HD95", "ASD", "val_score")},
            }, indent=2, ensure_ascii=False) + "\n")
            print(f"[BEST-VAL] epoch={epoch} score={row['val_score']:.6f}", flush=True)
        else:
            print(f"[VAL] epoch={epoch} score={row['val_score']:.6f} (best={best['epoch'] if best else 'none'})", flush=True)

        if args.patience > 0 and best is not None:
            validated_epochs = sorted(int(old["epoch"]) for old in rows)
            best_epoch = int(best["epoch"])
            if len([value for value in validated_epochs if value > best_epoch]) >= args.patience:
                print(f"[EARLY-STOP] no validation improvement for {args.patience} validation epochs", flush=True)
                break


if __name__ == "__main__":
    main()
