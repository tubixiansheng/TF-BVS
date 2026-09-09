#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = ["overall", "L2", "L3", "L4", "L5"]


def read_summary(path):
    rows = {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            rows[row["group"]] = {
                "n": int(row["n"]),
                "J": float(row["IoU"]),
                "Dice": float(row["Dice"]),
                "HD95": float(row["HD95"]),
                "ASD": float(row["ASD"]),
            }
    return rows


def plot_delta(gt, high_low, out_path):
    groups = [g for g in GROUP_ORDER if g in gt and g in high_low]
    delta_j = np.array([(high_low[g]["J"] - gt[g]["J"]) * 100 for g in groups])
    delta_dice = np.array([(high_low[g]["Dice"] - gt[g]["Dice"]) * 100 for g in groups])
    ns = [gt[g]["n"] for g in groups]

    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    c_j = "#3b6ea8"
    c_d = "#c94f4f"
    bars_j = ax.bar(x - width / 2, delta_j, width, label="Delta J", color=c_j, alpha=0.88)
    bars_d = ax.bar(x + width / 2, delta_dice, width, label="Delta Dice", color=c_d, alpha=0.88)

    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(groups, ns)], fontsize=9)
    ax.set_ylabel("Performance change after high-low reversal (points)", fontsize=10)
    ax.set_title("High-low risk reversal mainly affects the L5 subset", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars, vals in [(bars_j, delta_j), (bars_d, delta_dice)]:
        for bar, val in zip(bars, vals):
            y = bar.get_height()
            va = "top" if y < 0 else "bottom"
            offset = -0.12 if y < 0 else 0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + offset,
                f"{val:+.2f}",
                ha="center",
                va=va,
                fontsize=8,
            )

    ax.annotate(
        "L5: -7.08 J / -7.51 Dice",
        xy=(x[-1], min(delta_j[-1], delta_dice[-1])),
        xytext=(x[-1] - 1.45, min(delta_j[-1], delta_dice[-1]) + 2.2),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.9),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_l5_absolute(gt, high_low, out_path):
    metrics = ["J", "Dice", "HD95", "ASD"]
    gt_vals = [gt["L5"][m] if m in ["HD95", "ASD"] else gt["L5"][m] * 100 for m in metrics]
    hl_vals = [high_low["L5"][m] if m in ["HD95", "ASD"] else high_low["L5"][m] * 100 for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), dpi=300)
    colors = ["#4d7fb8", "#d65f5f"]
    for ax, metric_names, title in [
        (axes[0], ["J", "Dice"], "Higher is better"),
        (axes[1], ["HD95", "ASD"], "Lower is better"),
    ]:
        idx = [metrics.index(m) for m in metric_names]
        x = np.arange(len(metric_names))
        width = 0.36
        a = [gt_vals[i] for i in idx]
        b = [hl_vals[i] for i in idx]
        bars1 = ax.bar(x - width / 2, a, width, label="GT grade", color=colors[0], alpha=0.9)
        bars2 = ax.bar(x + width / 2, b, width, label="High-low reversal", color=colors[1], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    axes[0].set_ylabel("L5 score")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("L5 performance under high-low risk reversal", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="outputs/rebuttal_metrics/rebuttal_labelstrong_B_stable005_s0.20_gt_fusion_meter_summary.csv")
    parser.add_argument("--high-low", default="outputs/rebuttal_metrics/rebuttal_labelstrong_B_stable005_s0.20_high_low_swap_fusion_meter_summary.csv")
    parser.add_argument("--output-dir", default="outputs/rebuttal_figures")
    args = parser.parse_args()

    gt = read_summary(args.gt)
    high_low = read_summary(args.high_low)
    out_dir = Path(args.output_dir)
    delta_path = out_dir / "label_sensitivity_high_low_delta_by_grade.png"
    l5_path = out_dir / "label_sensitivity_l5_absolute_metrics.png"
    plot_delta(gt, high_low, delta_path)
    plot_l5_absolute(gt, high_low, l5_path)
    print(f"[OK] delta figure: {delta_path}")
    print(f"[OK] L5 figure: {l5_path}")


if __name__ == "__main__":
    main()
