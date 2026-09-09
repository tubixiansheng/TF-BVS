#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


GRADE_NAMES = {0: "L2", 1: "L3", 2: "L4", 3: "L5"}
GRADE_IDS = {v: k for k, v in GRADE_NAMES.items()}


def load_npz(path: Path, max_per_type: int = 0):
    data = np.load(path)
    hat_t = data["hat_T"]
    e_t = data["E_T"]
    hat_c = data["hat_T_class"].astype(int)
    e_c = data["E_T_class"].astype(int)

    if max_per_type > 0:
        hat_t = hat_t[:max_per_type]
        e_t = e_t[:max_per_type]
        hat_c = hat_c[:max_per_type]
        e_c = e_c[:max_per_type]

    x = np.concatenate([hat_t, e_t], axis=0)
    domain = np.array(["pseudo"] * len(hat_t) + ["real"] * len(e_t))
    grade = np.concatenate([hat_c, e_c], axis=0)
    return x, domain, grade


def rbf_mmd(x, y, sigma=1.0):
    xx = pairwise_distances(x, x, metric="sqeuclidean")
    yy = pairwise_distances(y, y, metric="sqeuclidean")
    xy = pairwise_distances(x, y, metric="sqeuclidean")
    return float(np.exp(-xx / (2 * sigma * sigma)).mean() + np.exp(-yy / (2 * sigma * sigma)).mean() - 2 * np.exp(-xy / (2 * sigma * sigma)).mean())


def centroid_distance(x, domain, grade):
    rows = []
    for g in sorted(set(grade.tolist())):
        px = x[(domain == "pseudo") & (grade == g)]
        rx = x[(domain == "real") & (grade == g)]
        if len(px) == 0 or len(rx) == 0:
            continue
        rows.append({
            "grade": GRADE_NAMES.get(int(g), str(g)),
            "pseudo_n": len(px),
            "real_n": len(rx),
            "centroid_l2": float(np.linalg.norm(px.mean(axis=0) - rx.mean(axis=0))),
            "mmd_rbf": rbf_mmd(px, rx),
        })
    return rows


def knn_domain_accuracy(x, domain, n_neighbors=5):
    if len(set(domain.tolist())) < 2:
        return np.nan
    clf = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(x)))
    clf.fit(x, domain)
    return float(clf.score(x, domain))


def plot_embedding(z, domain, grade, title, path: Path):
    colors = {"pseudo": "#1f77b4", "real": "#d62728"}
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}
    plt.figure(figsize=(8, 6), dpi=180)
    for d in ["pseudo", "real"]:
        for g in sorted(set(grade.tolist())):
            mask = (domain == d) & (grade == g)
            if not mask.any():
                continue
            plt.scatter(
                z[mask, 0],
                z[mask, 1],
                s=16,
                alpha=0.72,
                c=colors[d],
                marker=markers.get(int(g), "o"),
                label=f"{d}-{GRADE_NAMES.get(int(g), str(g))}",
                linewidths=0,
            )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_group_embedding(z, labels, title, path: Path):
    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#7f7f7f",
    ]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]
    groups = sorted(set(labels.tolist()))
    plt.figure(figsize=(8, 6), dpi=180)
    for i, group in enumerate(groups):
        mask = labels == group
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=18,
            alpha=0.74,
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=str(group),
            linewidths=0,
        )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def fit_tsne(x_scaled, seed):
    if len(x_scaled) < 3:
        return None
    perplexity = max(2, min(30, (len(x_scaled) - 1) // 3))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x_scaled)


def fit_umap(x_scaled, seed):
    if len(x_scaled) < 3:
        return None
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
    n_neighbors = max(2, min(15, len(x_scaled) - 1))
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed).fit_transform(x_scaled)


def plot_reductions(x_scaled, domain, grade, emb_stem, out_dir: Path, seed: int, enable_umap: bool = True):
    outputs = []

    z_tsne = fit_tsne(x_scaled, seed)
    if z_tsne is not None:
        path = out_dir / f"{emb_stem}_tsne.png"
        plot_embedding(z_tsne, domain, grade, f"LV2TOT pseudo vs real text t-SNE ({emb_stem})", path)
        outputs.append(("t-SNE", "overall", path))

    if enable_umap:
        try:
            z_umap = fit_umap(x_scaled, seed)
            if z_umap is not None:
                path = out_dir / f"{emb_stem}_umap.png"
                plot_embedding(z_umap, domain, grade, f"LV2TOT pseudo vs real text UMAP ({emb_stem})", path)
                outputs.append(("UMAP", "overall", path))
        except Exception as exc:
            print(f"[WARN] UMAP overall skipped: {exc}")

    for g in sorted(set(grade.tolist())):
        mask = grade == g
        if not mask.any():
            continue
        grade_name = GRADE_NAMES.get(int(g), str(g))
        grade_x = x_scaled[mask]
        grade_domain = domain[mask]
        grade_labels = grade[mask]
        if len(set(grade_domain.tolist())) < 2:
            print(f"[WARN] {grade_name} skipped: missing pseudo or real samples")
            continue

        z_tsne_g = fit_tsne(grade_x, seed)
        if z_tsne_g is not None:
            path = out_dir / f"{emb_stem}_{grade_name}_tsne.png"
            plot_embedding(
                z_tsne_g,
                grade_domain,
                grade_labels,
                f"LV2TOT {grade_name} pseudo vs real text t-SNE ({emb_stem})",
                path,
            )
            outputs.append(("t-SNE", grade_name, path))

        if enable_umap:
            try:
                z_umap_g = fit_umap(grade_x, seed)
                if z_umap_g is not None:
                    path = out_dir / f"{emb_stem}_{grade_name}_umap.png"
                    plot_embedding(
                        z_umap_g,
                        grade_domain,
                        grade_labels,
                        f"LV2TOT {grade_name} pseudo vs real text UMAP ({emb_stem})",
                        path,
                    )
                    outputs.append(("UMAP", grade_name, path))
            except Exception as exc:
                print(f"[WARN] UMAP {grade_name} skipped: {exc}")

    return outputs


def parse_grade_pair(pair_text):
    if "-" not in pair_text:
        raise ValueError(f"Bad grade pair: {pair_text}. Expected format like L2-L4")
    left, right = pair_text.split("-", 1)
    left = left.strip().upper()
    right = right.strip().upper()
    if left not in GRADE_IDS or right not in GRADE_IDS:
        raise ValueError(f"Unknown grade pair: {pair_text}")
    return GRADE_IDS[left], GRADE_IDS[right]


def subset_by_specs(x_scaled, domain, grade, specs):
    xs, labels = [], []
    for d, g, label in specs:
        mask = (domain == d) & (grade == g)
        if not mask.any():
            continue
        xs.append(x_scaled[mask])
        labels.extend([label] * int(mask.sum()))
    if not xs:
        return None, None
    return np.concatenate(xs, axis=0), np.asarray(labels)


def plot_contrast_outputs(x_scaled, domain, grade, emb_stem, out_dir: Path, seed: int, pairs, enable_umap: bool = True):
    outputs = []
    for ga, gb in pairs:
        name_a = GRADE_NAMES[ga]
        name_b = GRADE_NAMES[gb]
        pair_name = f"{name_a}_vs_{name_b}"

        contrast_specs = [
            [("real", ga, f"real-{name_a}"), ("real", gb, f"real-{name_b}")],
            [("pseudo", ga, f"pseudo-{name_a}"), ("real", gb, f"real-{name_b}")],
            [("pseudo", gb, f"pseudo-{name_b}"), ("real", ga, f"real-{name_a}")],
            [
                ("pseudo", ga, f"pseudo-{name_a}"),
                ("real", ga, f"real-{name_a}"),
                ("pseudo", gb, f"pseudo-{name_b}"),
                ("real", gb, f"real-{name_b}"),
            ],
        ]
        contrast_names = [
            f"real_{pair_name}",
            f"pseudo_{name_a}_vs_real_{name_b}",
            f"pseudo_{name_b}_vs_real_{name_a}",
            f"fourway_{pair_name}",
        ]

        for specs, contrast_name in zip(contrast_specs, contrast_names):
            sub_x, labels = subset_by_specs(x_scaled, domain, grade, specs)
            if sub_x is None or len(set(labels.tolist())) < 2:
                print(f"[WARN] contrast {contrast_name} skipped: missing samples")
                continue
            z_tsne = fit_tsne(sub_x, seed)
            if z_tsne is not None:
                path = out_dir / f"{emb_stem}_{contrast_name}_tsne.png"
                plot_group_embedding(z_tsne, labels, f"LV2TOT {contrast_name} t-SNE ({emb_stem})", path)
                outputs.append(("t-SNE", contrast_name, path))
            if enable_umap:
                try:
                    z_umap = fit_umap(sub_x, seed)
                    if z_umap is not None:
                        path = out_dir / f"{emb_stem}_{contrast_name}_umap.png"
                        plot_group_embedding(z_umap, labels, f"LV2TOT {contrast_name} UMAP ({emb_stem})", path)
                        outputs.append(("UMAP", contrast_name, path))
                except Exception as exc:
                    print(f"[WARN] UMAP contrast {contrast_name} skipped: {exc}")
    return outputs


def contrast_metrics(x_scaled, domain, grade, pairs):
    rows = []
    for ga, gb in pairs:
        px_a = x_scaled[(domain == "pseudo") & (grade == ga)]
        px_b = x_scaled[(domain == "pseudo") & (grade == gb)]
        rx_a = x_scaled[(domain == "real") & (grade == ga)]
        rx_b = x_scaled[(domain == "real") & (grade == gb)]
        if min(len(px_a), len(px_b), len(rx_a), len(rx_b)) == 0:
            continue
        d_pa_ra = float(np.linalg.norm(px_a.mean(axis=0) - rx_a.mean(axis=0)))
        d_pa_rb = float(np.linalg.norm(px_a.mean(axis=0) - rx_b.mean(axis=0)))
        d_pb_rb = float(np.linalg.norm(px_b.mean(axis=0) - rx_b.mean(axis=0)))
        d_pb_ra = float(np.linalg.norm(px_b.mean(axis=0) - rx_a.mean(axis=0)))
        rows.append({
            "pair": f"{GRADE_NAMES[ga]}-{GRADE_NAMES[gb]}",
            "real_real_centroid_l2": float(np.linalg.norm(rx_a.mean(axis=0) - rx_b.mean(axis=0))),
            "pseudo_pseudo_centroid_l2": float(np.linalg.norm(px_a.mean(axis=0) - px_b.mean(axis=0))),
            "pseudo_a_real_a_l2": d_pa_ra,
            "pseudo_a_real_b_l2": d_pa_rb,
            "pseudo_a_margin_l2": d_pa_rb - d_pa_ra,
            "pseudo_b_real_b_l2": d_pb_rb,
            "pseudo_b_real_a_l2": d_pb_ra,
            "pseudo_b_margin_l2": d_pb_ra - d_pb_rb,
            "mmd_pseudo_a_real_a": rbf_mmd(px_a, rx_a),
            "mmd_pseudo_a_real_b": rbf_mmd(px_a, rx_b),
            "mmd_pseudo_b_real_b": rbf_mmd(px_b, rx_b),
            "mmd_pseudo_b_real_a": rbf_mmd(px_b, rx_a),
            "pseudo_a_n": len(px_a),
            "pseudo_b_n": len(px_b),
            "real_a_n": len(rx_a),
            "real_b_n": len(rx_b),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--epoch", type=int, default=-1, help="-1 uses the latest epoch_*.npz")
    parser.add_argument("--output-dir", default="outputs/rebuttal_embedding_analysis")
    parser.add_argument("--max-per-type", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-umap", action="store_true")
    parser.add_argument(
        "--contrast-pairs",
        nargs="*",
        default=["L2-L3", "L4-L5", "L2-L4", "L2-L5", "L3-L4", "L3-L5"],
        help="Grade pairs for real-vs-real and cross-grade pseudo/real contrast plots.",
    )
    parser.add_argument("--no-contrast", action="store_true")
    args = parser.parse_args()

    emb_dir = Path(args.embedding_dir)
    if args.epoch > 0:
        emb_path = emb_dir / f"epoch_{args.epoch}.npz"
    else:
        candidates = sorted(emb_dir.glob("epoch_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
        if not candidates:
            raise FileNotFoundError(f"No epoch_*.npz in {emb_dir}")
        emb_path = candidates[-1]
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)

    out_dir = Path(args.output_dir)
    x, domain, grade = load_npz(emb_path, args.max_per_type)
    x_scaled = StandardScaler().fit_transform(x)

    reduction_outputs = plot_reductions(
        x_scaled,
        domain,
        grade,
        emb_path.stem,
        out_dir,
        seed=args.seed,
        enable_umap=not args.no_umap,
    )
    contrast_outputs = []
    contrast_rows = []
    contrast_pairs = [parse_grade_pair(p) for p in args.contrast_pairs]
    if not args.no_contrast:
        contrast_outputs = plot_contrast_outputs(
            x_scaled,
            domain,
            grade,
            emb_path.stem,
            out_dir,
            seed=args.seed,
            pairs=contrast_pairs,
            enable_umap=not args.no_umap,
        )
        contrast_rows = contrast_metrics(x_scaled, domain, grade, contrast_pairs)

    rows = centroid_distance(x_scaled, domain, grade)
    pseudo = x_scaled[domain == "pseudo"]
    real = x_scaled[domain == "real"]
    summary = [{
        "epoch_file": emb_path.name,
        "n_total": len(x_scaled),
        "n_pseudo": len(pseudo),
        "n_real": len(real),
        "overall_centroid_l2": float(np.linalg.norm(pseudo.mean(axis=0) - real.mean(axis=0))),
        "overall_mmd_rbf": rbf_mmd(pseudo, real),
        "domain_knn_acc": knn_domain_accuracy(x_scaled, domain),
    }]

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{emb_path.stem}_per_grade_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["grade", "pseudo_n", "real_n", "centroid_l2", "mmd_rbf"])
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{emb_path.stem}_summary_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    if contrast_rows:
        with (out_dir / f"{emb_path.stem}_grade_contrast_metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(contrast_rows[0].keys()))
            writer.writeheader()
            writer.writerows(contrast_rows)

    print(f"[OK] embedding file: {emb_path}")
    for method, group, path in reduction_outputs + contrast_outputs:
        print(f"[OK] {method} {group}: {path}")
    print(f"[OK] summary: {out_dir / f'{emb_path.stem}_summary_metrics.csv'}")
    print(f"[OK] per-grade: {out_dir / f'{emb_path.stem}_per_grade_metrics.csv'}")
    if contrast_rows:
        print(f"[OK] grade-contrast: {out_dir / f'{emb_path.stem}_grade_contrast_metrics.csv'}")


if __name__ == "__main__":
    main()
