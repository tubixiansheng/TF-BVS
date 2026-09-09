#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize


GRADE_NAMES = {0: "L2", 1: "L3", 2: "L4", 3: "L5"}
GRADE_IDS = {v: k for k, v in GRADE_NAMES.items()}
COLORS = {"L2": "#2f6fb0", "L3": "#5aa469", "L4": "#d9893d", "L5": "#c3423f"}


def parse_pair(text):
    left, right = text.upper().split("-", 1)
    return GRADE_IDS[left], GRADE_IDS[right]


def covariance_ellipse(points, n_std=1.6):
    if len(points) < 3:
        return None
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-8))
    return width, height, angle


def plot_pair(z, labels, pair_name, out_path):
    plt.figure(figsize=(5.2, 4.2), dpi=300)
    for grade_id in sorted(set(labels.tolist())):
        grade_name = GRADE_NAMES[int(grade_id)]
        mask = labels == grade_id
        color = COLORS[grade_name]
        pts = z[mask]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=22,
            alpha=0.78,
            c=color,
            label=f"{grade_name} real text (n={len(pts)})",
            linewidths=0,
        )
        center = pts.mean(axis=0)
        plt.scatter(center[0], center[1], s=70, c=color, marker="x", linewidths=1.6)
        ellipse = covariance_ellipse(pts)
        if ellipse is not None:
            from matplotlib.patches import Ellipse
            width, height, angle = ellipse
            patch = Ellipse(
                center,
                width,
                height,
                angle=angle,
                facecolor=color,
                edgecolor=color,
                alpha=0.13,
                linewidth=1.0,
            )
            plt.gca().add_patch(patch)
    plt.title(f"Real clinical text embeddings: {pair_name}", fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=7, frameon=False, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def fit_umap(x, seed, n_neighbors, min_dist):
    try:
        import numba
        orig_njit = numba.njit
        orig_jit = numba.jit
        orig_vectorize = numba.vectorize

        def no_cache(decorator):
            def wrapper(*args, **kwargs):
                kwargs["cache"] = False
                return decorator(*args, **kwargs)
            return wrapper

        def njit_no_cache(*args, **kwargs):
            kwargs["cache"] = False
            return orig_njit(*args, **kwargs)

        numba.jit = no_cache(orig_jit)
        numba.njit = njit_no_cache
        numba.vectorize = no_cache(orig_vectorize)
    except Exception:
        pass
    import umap
    return umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(x) - 1),
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="output/ref_youtube_vos/rebuttal_lv2tot_label_residual_swinb/embedding_data/epoch_35.npz")
    parser.add_argument("--output-dir", default="outputs/rebuttal_embedding_analysis/real_text_grade_umap_response")
    parser.add_argument("--pairs", nargs="+", default=["L2-L5", "L2-L3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-grade", type=int, default=0, help="0 keeps all samples; >0 down-samples each grade for balanced visualization.")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.35)
    args = parser.parse_args()

    data = np.load(args.embedding)
    x_real = data["E_T"].astype(np.float32)
    y_real = data["E_T_class"].astype(int)
    x_real = normalize(StandardScaler().fit_transform(x_real))

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    rows = []
    for left_id, right_id in [parse_pair(pair) for pair in args.pairs]:
        pair_name = f"{GRADE_NAMES[left_id]} vs {GRADE_NAMES[right_id]}"
        indices = []
        for grade_id in [left_id, right_id]:
            grade_indices = np.where(y_real == grade_id)[0]
            if args.max_per_grade > 0 and len(grade_indices) > args.max_per_grade:
                grade_indices = rng.choice(grade_indices, size=args.max_per_grade, replace=False)
            indices.append(grade_indices)
        indices = np.concatenate(indices)
        x_pair = x_real[indices]
        y_pair = y_real[indices]
        z = fit_umap(x_pair, args.seed, args.n_neighbors, args.min_dist)
        out_path = out_dir / f"epoch_35_real_{GRADE_NAMES[left_id]}_vs_{GRADE_NAMES[right_id]}_umap_response.png"
        plot_pair(z, y_pair, pair_name, out_path)

        left = x_pair[y_pair == left_id]
        right = x_pair[y_pair == right_id]
        dist = pairwise_distances(left, right, metric="cosine")
        rows.append({
            "pair": f"{GRADE_NAMES[left_id]}-{GRADE_NAMES[right_id]}",
            "left_n": len(left),
            "right_n": len(right),
            "centroid_l2": float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0))),
            "mean_cross_cosine_distance": float(dist.mean()),
            "silhouette_cosine": float(silhouette_score(x_pair, y_pair, metric="cosine")) if len(set(y_pair.tolist())) == 2 else np.nan,
            "figure": str(out_path),
        })
        print(f"[OK] {pair_name}: {out_path}")

    metrics_path = out_dir / "epoch_35_real_text_pair_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
