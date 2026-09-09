#!/usr/bin/env python3
"""Select a checkpoint using the fixed validation split.

This is deliberately an offline selector.  The validation path reuses the
same backbone -> temporal-fusion -> meter pipeline used for the final test
report, but runs after the DDP training process has exited.  That keeps the
training GPUs and distributed process group out of the validation subprocess
and avoids selecting a checkpoint with test-set information.

The default selection score is

    0.5 * J + 0.3 * F + 0.2 * Dice

where all three overlap metrics are in [0, 1].  Scores within
``--score-tie-tol`` are resolved by lower HD95, then lower ASD, then the
earlier epoch.  The script writes ``checkpoints/best_val.pth.tar`` and a
complete CSV/JSON audit trail under ``<run-dir>/validation/<split>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EPOCH_RE = re.compile(r"^(\d+)\.pth\.tar$")


def run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def epoch_from_path(path: Path) -> int | None:
    match = EPOCH_RE.match(path.name)
    return int(match.group(1)) if match else None


def checkpoint_candidates(checkpoint_dir: Path, first: int, last: int, every: int) -> list[Path]:
    paths = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        epoch = epoch_from_path(path)
        if epoch is None or epoch < first or epoch > last or (epoch - first) % every != 0:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: epoch_from_path(p))


def read_train_losses(log_path: Path) -> dict[int, float]:
    losses: dict[int, float] = {}
    if not log_path.exists():
        return losses
    for line in log_path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        epoch = item.get("epoch")
        loss = item.get("train_loss")
        if isinstance(epoch, int) and isinstance(loss, (float, int)):
            losses[epoch] = float(loss)
    return losses


def read_overall_summary(summary_path: Path) -> dict[str, float | int]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Validation summary not found: {summary_path}")
    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    overall = next((row for row in rows if row.get("group") == "overall"), None)
    if overall is None:
        raise RuntimeError(f"No overall row in validation summary: {summary_path}")
    return {
        "n": int(overall["n"]),
        "J": float(overall["IoU"]),
        "F": float(overall["F-measure"]),
        "Dice": float(overall["Dice"]),
        "HD95": float(overall["HD95"]),
        "ASD": float(overall["ASD"]),
    }


def score(metrics: dict[str, float | int], weights: tuple[float, float, float]) -> float:
    return (
        weights[0] * float(metrics["J"])
        + weights[1] * float(metrics["F"])
        + weights[2] * float(metrics["Dice"])
    )


def is_better(candidate: dict, incumbent: dict | None, tie_tol: float) -> bool:
    if incumbent is None:
        return True
    delta = float(candidate["val_score"]) - float(incumbent["val_score"])
    if delta > tie_tol:
        return True
    if delta < -tie_tol:
        return False
    for key in ("HD95", "ASD"):
        c_value = float(candidate[key])
        i_value = float(incumbent[key])
        if c_value < i_value - 1e-12:
            return True
        if c_value > i_value + 1e-12:
            return False
    return int(candidate["epoch"]) < int(incumbent["epoch"])


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".tmp.", dir=destination.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "checkpoint", "train_loss", "n", "J", "F", "Dice", "HD95", "ASD", "val_score",
        "summary_csv",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best checkpoint on a non-test validation split.")
    parser.add_argument("--run-dir", required=True, nargs="+",
                        help="One or more training output directories containing checkpoints/ and log.txt")
    parser.add_argument("--dataset-root", required=True, help="Dataset root used by fusion and meter evaluation")
    parser.add_argument("--dataset-path", default=None, help="Dataset root passed to backbone inference; defaults to --dataset-root")
    parser.add_argument("--split", default="valid", help="Validation split; test/test_new are rejected by default")
    parser.add_argument("-c", "--config", default="configs/ytvos_swinb.yaml")
    parser.add_argument("-ng", "--num-gpus", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--from-epoch", type=int, default=0)
    parser.add_argument("--to-epoch", type=int, default=10**9)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--score-weights", type=float, nargs=3, default=(0.5, 0.3, 0.2), metavar=("J", "F", "DICE"))
    parser.add_argument("--score-tie-tol", type=float, default=0.001,
                        help="Score difference treated as a tie; default is 0.1 percentage point")
    parser.add_argument("--output-dir", default=None,
                        help="Metric output directory; defaults to <run-dir>/validation/<split>/metrics")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse an existing summary CSV when present")
    parser.add_argument("--dry-run", action="store_true", help="Print validation commands without running them")
    parser.add_argument("--allow-test-selection", action="store_true",
                        help="Explicitly allow test/test_new; do not use this for formal model selection")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.every < 1:
        raise ValueError("--every must be >= 1")
    weights = tuple(float(x) for x in args.score_weights)
    if any(x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError("--score-weights must be non-negative and not all zero")
    weight_sum = sum(weights)
    weights = tuple(x / weight_sum for x in weights)

    split = args.split.lower()
    if split in {"test", "test_new"} and not args.allow_test_selection:
        raise ValueError("Refusing to select a checkpoint on the test split; use --allow-test-selection only for diagnostics")

    run_dirs = [Path(path).resolve() for path in args.run_dir]
    for run_dir in run_dirs:
        if not (run_dir / "checkpoints").is_dir():
            raise FileNotFoundError(f"Missing checkpoint directory: {run_dir / 'checkpoints'}")
    run_dir = run_dirs[0]
    checkpoint_dir = run_dir / "checkpoints"
    dataset_root = Path(args.dataset_root).resolve()
    dataset_path = Path(args.dataset_path or args.dataset_root).resolve()
    validation_dir = run_dir / "validation" / split
    metric_dir = Path(args.output_dir).resolve() if args.output_dir else validation_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_with_logs = []
    train_losses: dict[int, float] = {}
    for source_dir in run_dirs:
        source_checkpoints = checkpoint_candidates(
            source_dir / "checkpoints", args.from_epoch, args.to_epoch, args.every
        )
        source_losses = read_train_losses(source_dir / "log.txt")
        train_losses.update(source_losses)
        checkpoints_with_logs.extend((path, source_losses) for path in source_checkpoints)
    checkpoints_with_logs.sort(key=lambda item: (epoch_from_path(item[0]), str(item[0])))
    checkpoints = [path for path, _ in checkpoints_with_logs]
    if not checkpoints:
        raise RuntimeError(f"No numeric checkpoints found in {[str(p / 'checkpoints') for p in run_dirs]}")
    run_name = "_".join(path.name.replace("/", "_") for path in run_dirs)
    rows: list[dict] = []
    best: dict | None = None

    print(f"[INFO] validation split: {dataset_root / split}")
    print(f"[INFO] checkpoints: {[epoch_from_path(p) for p in checkpoints]}")
    print(f"[INFO] score weights: J={weights[0]:.4f}, F={weights[1]:.4f}, Dice={weights[2]:.4f}")

    for checkpoint in checkpoints:
        epoch = epoch_from_path(checkpoint)
        tag = f"{run_name}_valid_e{epoch:02d}"
        backbone_version = f"{tag}_backbone"
        fusion_version = f"{tag}_fusion"
        metrics_name = fusion_version
        summary_path = metric_dir / f"{metrics_name}_meter_summary.csv"

        if not (args.skip_existing and summary_path.exists()):
            command = [
                args.python, "scripts/rebuttal/run_full_inference_pipeline.py",
                "-c", args.config,
                "-ckpt", str(checkpoint),
                "-ng", str(args.num_gpus),
                "--dataset-path", str(dataset_path),
                "--dataset-root", str(dataset_root),
                "--split", split,
                "--backbone-version", backbone_version,
                "--fusion-version", fusion_version,
                "--frame-step", str(args.frame_step),
                "--metrics-name", metrics_name,
                "--output-dir", str(metric_dir),
            ]
            if args.dry_run:
                print("[DRY-RUN]", " ".join(command))
                continue
            run(command)

        metrics = read_overall_summary(summary_path)
        row = {
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "train_loss": train_losses.get(epoch, ""),
            **metrics,
            "val_score": score(metrics, weights),
            "summary_csv": str(summary_path),
        }
        rows.append(row)
        if is_better(row, best, args.score_tie_tol):
            best = row
        write_csv(validation_dir / "validation_summary.csv", rows)
        print(
            f"[VAL] epoch={epoch} J={row['J']:.4f} F={row['F']:.4f} "
            f"Dice={row['Dice']:.4f} HD95={row['HD95']:.4f} ASD={row['ASD']:.4f} "
            f"score={row['val_score']:.6f}"
        )

    if args.dry_run:
        return
    if best is None:
        raise RuntimeError("No validation result was produced")

    best_checkpoint = Path(best["checkpoint"])
    best_path = checkpoint_dir / "best_val.pth.tar"
    atomic_copy(best_checkpoint, best_path)
    manifest = {
        "selection": "validation",
        "best_epoch": int(best["epoch"]),
        "best_checkpoint": str(best_checkpoint),
        "best_val_checkpoint": str(best_path),
        "dataset_root": str(dataset_root),
        "split": split,
        "score_formula": "0.5*J + 0.3*F + 0.2*Dice after weight normalization",
        "score_weights": {"J": weights[0], "F": weights[1], "Dice": weights[2]},
        "score_tie_tol": args.score_tie_tol,
        "metrics": {k: best[k] for k in ("n", "J", "F", "Dice", "HD95", "ASD", "val_score")},
        "validation_summary": str(validation_dir / "validation_summary.csv"),
    }
    (validation_dir / "best_val.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[BEST] epoch={best['epoch']} -> {best_path}")
    print(f"[BEST] audit CSV: {validation_dir / 'validation_summary.csv'}")


if __name__ == "__main__":
    main()
