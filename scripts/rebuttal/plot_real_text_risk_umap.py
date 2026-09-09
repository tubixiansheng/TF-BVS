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
COLORS = {
    "L2": "#2f6fb0",
    "L4+L5": "#c3423f",
}


def covariance_ellipse(points, n_std=1.45):
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


def fit_umap(x, seed, n_neighbors, min_dist, y=None, target_weight=0.0):
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
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(x) - 1),
        min_dist=min_dist,
        metric="cosine",
        target_metric="categorical",
        target_weight=target_weight,
        random_state=seed,
    )
    if y is not None and target_weight > 0:
        return reducer.fit_transform(x, y=y)
    return reducer.fit_transform(x)


def balanced_low_high_indices(y_real, seed):
    rng = np.random.default_rng(seed)
    l2_idx = np.where(y_real == 0)[0]
    l4_idx = np.where(y_real == 2)[0]
    l5_idx = np.where(y_real == 3)[0]
    high_idx = np.concatenate([l4_idx, l5_idx])

    n = min(len(l2_idx), len(high_idx))
    if len(l2_idx) > n:
        l2_idx = rng.choice(l2_idx, size=n, replace=False)

    if len(high_idx) > n:
        if len(l5_idx) >= n:
            high_balanced = rng.choice(l5_idx, size=n, replace=False)
        else:
            l4_need = n - len(l5_idx)
            l4_take = rng.choice(l4_idx, size=l4_need, replace=False)
            high_balanced = np.concatenate([l5_idx, l4_take])
    else:
        high_balanced = high_idx

    return np.sort(l2_idx), np.sort(high_balanced), len(l4_idx), len(l5_idx)


def plot_low_high(z, group_labels, high_grade_labels, out_path, target_weight=0.0):
    plt.figure(figsize=(5.4, 4.25), dpi=300)
    for group_id, group_name in [(0, "L2"), (1, "L4+L5")]:
        pts = z[group_labels == group_id]
        color = COLORS[group_name]
        if group_name == "L4+L5":
            l4_n = int((high_grade_labels == 2).sum())
            l5_n = int((high_grade_labels == 3).sum())
            label = f"{group_name} real text (n={len(pts)}; L4={l4_n}, L5={l5_n})"
        else:
            label = f"{group_name} real text (n={len(pts)})"
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            s=24,
            alpha=0.78,
            c=color,
            label=label,
            linewidths=0,
        )
        center = pts.mean(axis=0)
        plt.scatter(center[0], center[1], s=78, c=color, marker="x", linewidths=1.7)
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

    if target_weight <= 0:
        title_prefix = "Unsupervised UMAP of real clinical text embeddings"
    elif target_weight <= 0.1:
        title_prefix = "Weakly supervised UMAP of real clinical text embeddings"
    else:
        title_prefix = "Supervised UMAP of real clinical text embeddings"
    plt.title(f"{title_prefix}: L2 vs L4+L5", fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=7, frameon=False, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="output/ref_youtube_vos/rebuttal_lv2tot_label_residual_swinb/embedding_data/epoch_35.npz")
    parser.add_argument("--output-dir", default="outputs/rebuttal_embedding_analysis/real_text_risk_umap_response")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.28)
    parser.add_argument(
        "--preprocess",
        choices=["standardize_normalize", "standardize", "normalize", "raw"],
        default="standardize_normalize",
    )
    parser.add_argument(
        "--topk-risk-dims",
        type=int,
        default=0,
        help="0 uses all embedding dimensions; >0 selects dimensions with the largest low/high Cohen's d before UMAP.",
    )
    parser.add_argument("--target-weight", type=float, default=0.0, help=">0 enables supervised UMAP layout with risk-group labels.")
    args = parser.parse_args()

    data = np.load(args.embedding)
    x_real = data["E_T"].astype(np.float32)
    y_real = data["E_T_class"].astype(int)
    if args.preprocess == "standardize_normalize":
        x_real = normalize(StandardScaler().fit_transform(x_real))
    elif args.preprocess == "standardize":
        x_real = StandardScaler().fit_transform(x_real)
    elif args.preprocess == "normalize":
        x_real = normalize(x_real)

    l2_idx, high_idx, total_l4, total_l5 = balanced_low_high_indices(y_real, args.seed)
    indices = np.concatenate([l2_idx, high_idx])
    group_labels = np.concatenate([
        np.zeros(len(l2_idx), dtype=int),
        np.ones(len(high_idx), dtype=int),
    ])
    high_grade_labels = y_real[high_idx]
    x_pair = x_real[indices]
    selected_dims = []
    if args.topk_risk_dims > 0 and args.topk_risk_dims < x_pair.shape[1]:
        low_x = x_pair[group_labels == 0]
        high_x = x_pair[group_labels == 1]
        pooled_std = np.sqrt(0.5 * (low_x.var(axis=0) + high_x.var(axis=0))) + 1e-6
        cohen_d = np.abs(low_x.mean(axis=0) - high_x.mean(axis=0)) / pooled_std
        selected_dims = np.argsort(cohen_d)[-args.topk_risk_dims:][::-1].tolist()
        x_pair = x_pair[:, selected_dims]

    z = fit_umap(
        x_pair,
        args.seed,
        args.n_neighbors,
        args.min_dist,
        y=group_labels,
        target_weight=args.target_weight,
    )
    out_dir = Path(args.output_dir)
    dim_tag = f"risk_top{args.topk_risk_dims}" if args.topk_risk_dims > 0 else "all_dims"
    sup_tag = f"supervised_tw{args.target_weight:g}" if args.target_weight > 0 else "unsupervised"
    out_path = out_dir / f"epoch_35_real_L2_vs_L4L5_balanced_{args.preprocess}_{dim_tag}_umap_response.png"
    out_path = out_dir / f"epoch_35_real_L2_vs_L4L5_balanced_{args.preprocess}_{dim_tag}_{sup_tag}_umap_response.png"
    plot_low_high(z, group_labels, high_grade_labels, out_path, target_weight=args.target_weight)

    low = x_pair[group_labels == 0]
    high = x_pair[group_labels == 1]
    dist = pairwise_distances(low, high, metric="cosine")
    rows = [{
        "comparison": "L2-vs-L4+L5",
        "l2_n": len(low),
        "high_n": len(high),
        "sampled_l4_n": int((high_grade_labels == 2).sum()),
        "sampled_l5_n": int((high_grade_labels == 3).sum()),
        "total_l4_available": total_l4,
        "total_l5_available": total_l5,
        "centroid_l2": float(np.linalg.norm(low.mean(axis=0) - high.mean(axis=0))),
        "mean_cross_cosine_distance": float(dist.mean()),
        "silhouette_cosine": float(silhouette_score(x_pair, group_labels, metric="cosine")),
        "umap_seed": args.seed,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "preprocess": args.preprocess,
        "topk_risk_dims": args.topk_risk_dims,
        "target_weight": args.target_weight,
        "selected_dims": " ".join(map(str, selected_dims[:32])),
        "figure": str(out_path),
    }]
    metrics_path = out_dir / f"epoch_35_real_L2_vs_L4L5_balanced_{args.preprocess}_{dim_tag}_{sup_tag}_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] figure: {out_path}")
    print(f"[OK] metrics: {metrics_path}")
    print(rows[0])


if __name__ == "__main__":
    main()
