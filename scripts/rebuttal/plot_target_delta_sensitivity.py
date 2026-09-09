#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = ["overall", "L2", "L3", "L4", "L5"]
NS = {"overall": 10192, "L2": 823, "L3": 6668, "L4": 2441, "L5": 260}
DELTA_J = {"overall": -2.26, "L2": -2.86, "L3": -1.64, "L4": -3.22, "L5": -7.08}
DELTA_DICE = {"overall": -2.57, "L2": -2.91, "L3": -2.03, "L4": -3.39, "L5": -7.51}


def main():
    out_dir = Path("outputs/rebuttal_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "target_label_sensitivity_delta_by_grade.png"

    x = np.arange(len(GROUPS))
    width = 0.36
    delta_j = np.array([DELTA_J[g] for g in GROUPS])
    delta_dice = np.array([DELTA_DICE[g] for g in GROUPS])

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    bars_j = ax.bar(x - width / 2, delta_j, width, label="Delta J", color="#3b6ea8", alpha=0.9)
    bars_d = ax.bar(x + width / 2, delta_dice, width, label="Delta Dice", color="#c94f4f", alpha=0.9)

    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(n={NS[g]})" for g in GROUPS], fontsize=9)
    ax.set_ylabel("Performance change after high-low reversal (points)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bars, vals in [(bars_j, delta_j), (bars_d, delta_dice)]:
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.08,
                f"{val:.2f}",
                ha="center",
                va="top",
                fontsize=8,
            )


    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] figure: {out_path}")


if __name__ == "__main__":
    main()
