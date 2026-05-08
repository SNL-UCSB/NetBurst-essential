#!/usr/bin/env python3
"""
After cluster_analysis + per-cluster CKA, build global TSFresh metrics and per-cluster local
feature importance outputs.

Primary outputs:
- Global metrics for all TSFresh features: Fisher ratio + global CKA to embedding.
- CDF plots: global Fisher, global CKA, and cluster importance.
- Globally important features: Fisher threshold filter (default: >= 0.1).
- Per-cluster local feature importance tables:
    - full filtered set
    - top-K by local importance = local CKA * abs(Cohen's d)

Legacy compatibility outputs are still written for existing dual-overlay plots:
- overlay_cka_top{K}.csv
- overlay_cohen_d_top{K}.csv
- global_filtered_local_overlay_rankings.csv

Global feature set (``--global-filter-mode``):
- ``intersection``: Fisher top-K ∩ global CKA top-K (legacy behavior).
- ``fisher_min``: all features with Fisher ratio >= ``--global-fisher-min``.
- ``--global-filtered-features-csv``: bypass global Fisher/CKA computation and use an
    existing feature list directly.

Requires existing cluster_*/selected_features_cka.csv under --parent-out.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

_analysis_dir = Path(__file__).resolve().parent
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

from cluster_analysis import coerce_numeric_feature_block, read_table
from cka_core import cka_embedding_vs_each_tsfresh_column
from cluster_fs import safe_cluster_dir_name
from cluster_variance_stats import (
    cohens_d_from_agg_df,
    feature_mean_var_pairs_from_variance_df,
    population_stats_from_variance_df,
    mean_across_clusters_matrix,
)
from local_cka_cohen_utils import collect_cluster_means_from_member_csvs


def _apply_ip_suffix(df: pd.DataFrame, id_ip: str, suffix: str | None) -> pd.DataFrame:
    if suffix is None or not str(suffix).strip():
        return df
    s = str(suffix).strip()
    if id_ip not in df.columns:
        raise ValueError(f"ip_suffix_filter requires column {id_ip!r}")
    mask = df[id_ip].astype(str).str.endswith(s)
    return df.loc[mask].reset_index(drop=True)


def load_global_cka_features(path: str, col: str = "tsfresh_feature", top_k: int = 100) -> List[str]:
    df = pd.read_csv(path, low_memory=False)
    if col not in df.columns:
        raise KeyError(f"Global CKA CSV missing {col!r}; columns={list(df.columns)}")
    names = df[col].astype(str).head(int(top_k)).tolist()
    return names


def _cdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs = np.sort(v)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def _plot_cdf(values: np.ndarray, out_png: Path, *, xlabel: str, title: str) -> None:
    xs, ys = _cdf_xy(values)
    if xs.size == 0:
        print(f"[global_filtered_overlay_prep] skip CDF (no finite values): {out_png}", flush=True)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=160)
    ax.step(xs, ys, where="post", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"Wrote {out_png}", flush=True)


def _resolve_global_cka_all(
    *,
    merged: pd.DataFrame,
    feature_cols: List[str],
    repr_prefix: str,
    global_cka_all_csv: str | None,
    cka_feature_col: str,
    cka_value_col: str,
) -> pd.DataFrame:
    """
    Return DataFrame: [tsfresh_feature, global_cka_to_embedding].
    Uses provided all-features CSV when present, otherwise computes from merged data.
    """
    if global_cka_all_csv:
        p = Path(os.path.expanduser(os.path.expandvars(global_cka_all_csv)))
        if p.is_file():
            df = pd.read_csv(p, low_memory=False)
            if cka_feature_col not in df.columns or cka_value_col not in df.columns:
                raise KeyError(
                    f"global CKA CSV missing required columns {cka_feature_col!r}/{cka_value_col!r}; "
                    f"columns={list(df.columns)}"
                )
            out = df[[cka_feature_col, cka_value_col]].copy()
            out.columns = ["tsfresh_feature", "global_cka_to_embedding"]
            out["tsfresh_feature"] = out["tsfresh_feature"].astype(str)
            out["global_cka_to_embedding"] = pd.to_numeric(
                out["global_cka_to_embedding"], errors="coerce"
            )
            out = out.dropna(subset=["global_cka_to_embedding"])
            # If incomplete, fall through to recompute.
            covered = set(out["tsfresh_feature"].tolist())
            if len(covered.intersection(set(feature_cols))) == len(set(feature_cols)):
                return out
            print(
                "[global_filtered_overlay_prep] global CKA CSV is partial; recomputing full all-features global CKA.",
                flush=True,
            )

    repr_cols = [c for c in merged.columns if str(c).startswith(str(repr_prefix))]
    if not repr_cols:
        raise ValueError(
            f"No representation columns with prefix {repr_prefix!r} found in merged table; "
            "cannot compute full global CKA. Provide --global-cka-all-csv with complete coverage."
        )
    X = merged[repr_cols].to_numpy(dtype=float)
    Y = merged[feature_cols].to_numpy(dtype=float)
    df_cka = cka_embedding_vs_each_tsfresh_column(X, Y, feature_cols)
    out = df_cka[["tsfresh_feature", "cka_to_embedding"]].copy()
    out.columns = ["tsfresh_feature", "global_cka_to_embedding"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clustering-csv", required=True)
    ap.add_argument("--tsfresh-csv", required=True)
    ap.add_argument(
        "--global-cka-top-csv",
        default=None,
        help="Required for --global-filter-mode intersection (e.g. top100_by_embedding_cka.csv).",
    )
    ap.add_argument("--parent-out", required=True, help="K_* directory containing cluster_* subdirs")
    ap.add_argument("--label-col", default="label", help="Must match cluster_analysis --label-col")
    ap.add_argument("--id-ip", default="ip")
    ap.add_argument("--id-source-file", default="source_file")
    ap.add_argument("--ip-suffix-filter", default=None)
    ap.add_argument(
        "--global-filter-mode",
        choices=("intersection", "fisher_min"),
        default="intersection",
        help="How to build global_filtered_features.csv (default: intersection).",
    )
    ap.add_argument(
        "--global-filtered-features-csv",
        default=None,
        help=(
            "Optional precomputed global feature list CSV. If set, this is used directly and "
            "global Fisher/CKA metrics are not recomputed. Must contain 'tsfresh_feature' or 'feature'."
        ),
    )
    ap.add_argument(
        "--global-fisher-min",
        type=float,
        default=0.1,
        help="Used with --global-filter-mode fisher_min: keep features with Fisher ratio >= this.",
    )
    ap.add_argument("--global-cka-top-k", type=int, default=100)
    ap.add_argument("--global-cka-all-csv", default=None, help="Optional full global CKA CSV for all TSFresh features.")
    ap.add_argument("--global-cka-all-feature-col", default="tsfresh_feature")
    ap.add_argument("--global-cka-all-value-col", default="cka_to_embedding")
    ap.add_argument("--repr-prefix", default="repr_dim")
    ap.add_argument(
        "--all-features-csv",
        default="all_features_global_cka_fisher.csv",
        help="All TSFresh features with global Fisher ratio and global CKA, sorted by CKA desc. "
             "No filter applied. Written under --parent-out.",
    )
    ap.add_argument("--global-metrics-csv", default="global_feature_metrics_fisher_cka.csv")
    ap.add_argument("--global-fisher-cdf-png", default="cdf_global_fisher_ratio.png")
    ap.add_argument("--global-cka-cdf-png", default="cdf_global_cka_to_embedding.png")
    ap.add_argument("--cluster-importance-csv", default="cluster_importance_topk_sum.csv")
    ap.add_argument("--cluster-importance-cdf-png", default="cdf_cluster_importance_topk_sum.png")
    ap.add_argument(
        "--local-importance-all-filename",
        default="local_feature_importance_all_global_filtered.csv",
        help="Per-cluster full table on globally filtered features.",
    )
    ap.add_argument(
        "--local-importance-top-filename",
        default="local_feature_importance_topk.csv",
        help="Per-cluster top-K table by local importance.",
    )
    ap.add_argument(
        "--local-overlay-top-k",
        type=int,
        default=10,
        help="Per-cluster top features after global filter (CKA and |Cohen d| orderings; default 10).",
    )
    ap.add_argument(
        "--aggregate-rankings-csv",
        default="global_filtered_local_overlay_rankings.csv",
        help="Written under --parent-out: all clusters, CKA and cohen_d rows with feature + values.",
    )
    ap.add_argument(
        "--all-clusters-top-k-features-csv",
        default="all_clusters_top_k_features_with_means.csv",
        help="Parent-level concat of top-K rows per cluster with cluster_mean from feature_mean_var_cohens_d.csv.",
    )
    ap.add_argument("--cka-feature-col", default="tsfresh_feature")
    ap.add_argument(
            "--cohens-d-tanh-d0",
            default=None,
        help=(
                "Apply tanh rescaling to |Cohen's d| before computing feature_importance: "
                "tanh(|d| / d0). Pass a float (e.g. 1.0) or 'median' to auto-set d0 from "
                "the median of all finite |d| values across features and clusters. "
                "Raw cohens_d_mean values in output CSVs are unchanged."
        ),
    )
    ap.add_argument(
        "--cohens-d-max-norm",
        action="store_true",
        default=False,
        help=(
            "Apply max-norm rescaling to |Cohen's d| before computing feature_importance: "
            "for each feature, divide by max(|d|) across clusters → [0, 1] per feature. "
            "This prevents explosion when within-cluster variance is near-zero. "
            "Raw cohens_d_mean values in output CSVs are unchanged."
        ),
    )
    ap.add_argument(
        "--enable-cdf-plots",
        action="store_true",
        help="Write Fisher/CKA/cluster-importance CDF PNGs (default: CSV outputs only).",
    )
    args = ap.parse_args()

    id_cols = [args.id_ip, args.id_source_file]
    parent = Path(os.path.expanduser(os.path.expandvars(str(args.parent_out)))).resolve()
    parent.mkdir(parents=True, exist_ok=True)

    mode = str(args.global_filter_mode).strip().lower()
    if mode == "intersection" and not args.global_filtered_features_csv:
        if not args.global_cka_top_csv:
            raise SystemExit("--global-cka-top-csv is required for --global-filter-mode intersection")
        top_cka_names = load_global_cka_features(
            str(Path(args.global_cka_top_csv).expanduser()),
            col=args.cka_feature_col,
            top_k=int(args.global_cka_top_k),
        )
    else:
        top_cka_names = []

    clus = read_table(args.clustering_csv)
    for c in id_cols + [args.label_col]:
        if c not in clus.columns:
            raise KeyError(f"clustering CSV missing {c!r}")
    clus_small = clus[id_cols + [args.label_col]].copy()
    for c in id_cols:
        clus_small[c] = clus_small[c].astype(str)
    clus_small = _apply_ip_suffix(clus_small, args.id_ip, args.ip_suffix_filter)

    ts = read_table(args.tsfresh_csv)
    ts_num, feature_cols = coerce_numeric_feature_block(ts, id_cols=id_cols)
    for c in id_cols:
        if c in ts_num.columns:
            ts_num[c] = ts_num[c].astype(str)
    ts_num = _apply_ip_suffix(ts_num, args.id_ip, args.ip_suffix_filter)

    merged = clus_small.merge(ts_num, on=id_cols, how="inner")
    if merged.shape[0] == 0:
        raise ValueError("Join clustering + TSFresh produced 0 rows.")
    feat_in_merged = set(feature_cols)

    def _var(x):
        d = x.dropna()
        return d.var(ddof=0) if d.size >= 2 else np.nan

    if args.global_filtered_features_csv:
        pre_path = Path(os.path.expanduser(os.path.expandvars(str(args.global_filtered_features_csv))))
        if not pre_path.is_file():
            raise FileNotFoundError(f"Precomputed global-filtered CSV not found: {pre_path}")
        pre_df = pd.read_csv(pre_path, low_memory=False)
        if "tsfresh_feature" in pre_df.columns:
            feat_col = "tsfresh_feature"
        elif "feature" in pre_df.columns:
            feat_col = "feature"
        else:
            raise KeyError(
                "--global-filtered-features-csv must contain column 'tsfresh_feature' or 'feature'"
            )

        pre_df = pre_df.copy()
        pre_df[feat_col] = pre_df[feat_col].astype(str)
        seen_pre: Set[str] = set()
        global_filtered: List[str] = []
        for feat_name in pre_df[feat_col].tolist():
            if feat_name in seen_pre:
                continue
            seen_pre.add(feat_name)
            if feat_name in feat_in_merged:
                global_filtered.append(feat_name)

        meta = pre_df[pre_df[feat_col].isin(set(global_filtered))].copy()
        if feat_col != "tsfresh_feature":
            meta = meta.rename(columns={feat_col: "tsfresh_feature"})
        if "tsfresh_feature" in meta.columns:
            meta = meta.drop_duplicates(subset=["tsfresh_feature"], keep="first")
            order_map = {f: i for i, f in enumerate(global_filtered)}
            meta["__order"] = meta["tsfresh_feature"].map(order_map)
            meta = meta.sort_values("__order", kind="mergesort").drop(columns=["__order"]).reset_index(drop=True)

        print(
            f"[global_filtered_overlay_prep] using precomputed global-filtered features from {pre_path} "
            f"(n={len(global_filtered)} after TSFresh-join filtering).",
            flush=True,
        )
    else:
        # Full feature agg for Fisher (all numeric TSFresh columns)
        sub_all = merged[[args.label_col] + feature_cols].copy()

        agg_all = sub_all.groupby(args.label_col, sort=False)[feature_cols].agg(
            [("mean", "mean"), ("var", _var)]
        )
        agg_all.columns = [f"{c[0]}_{c[1]}" for c in agg_all.columns]
        agg_all.index.name = "cluster_id"
        agg_all["n_members"] = sub_all.groupby(args.label_col, sort=False).size()

        df_var = agg_all.reset_index()
        if df_var.columns[0] != "cluster_id":
            df_var = df_var.rename(columns={df_var.columns[0]: "cluster_id"})

        hm = population_stats_from_variance_df(
            df_var,
            cluster_id_col="cluster_id",
            fisher_reorder=False,
        )
        bases = list(hm["feature_bases"])
        fr = np.asarray(hm["fisher_ratio_vis"], dtype=float)

        df_fisher = pd.DataFrame(
            {
                "tsfresh_feature": [str(x) for x in bases],
                "fisher_ratio": fr,
            }
        )

        df_global_cka = _resolve_global_cka_all(
            merged=merged,
            feature_cols=feature_cols,
            repr_prefix=str(args.repr_prefix),
            global_cka_all_csv=args.global_cka_all_csv,
            cka_feature_col=str(args.global_cka_all_feature_col),
            cka_value_col=str(args.global_cka_all_value_col),
        )

        global_metrics = df_fisher.merge(df_global_cka, on="tsfresh_feature", how="left")
        global_metrics = global_metrics[global_metrics["tsfresh_feature"].isin(feature_cols)].copy()
        global_metrics["global_cka_to_embedding"] = pd.to_numeric(
            global_metrics["global_cka_to_embedding"], errors="coerce"
        )

        out_global_metrics = parent / str(args.global_metrics_csv).strip()
        global_metrics.to_csv(out_global_metrics, index=False)
        print(f"Wrote {out_global_metrics} (n={len(global_metrics)})", flush=True)

        # All-features CSV: every TSFresh feature, Fisher + CKA, sorted by CKA descending (no filter)
        all_feats_csv_name = str(args.all_features_csv).strip()
        if all_feats_csv_name:
            out_all = parent / all_feats_csv_name
            df_all = global_metrics.copy()
            df_all = df_all.sort_values(
                "global_cka_to_embedding", ascending=False, kind="mergesort"
            ).reset_index(drop=True)
            df_all.to_csv(out_all, index=False)
            print(f"Wrote {out_all} (n={len(df_all)}, all features, sorted by CKA desc)", flush=True)

        if args.enable_cdf_plots:
            _plot_cdf(
                global_metrics["fisher_ratio"].to_numpy(dtype=float),
                parent / str(args.global_fisher_cdf_png).strip(),
                xlabel="Fisher ratio",
                title="CDF of global Fisher ratio",
            )
            _plot_cdf(
                global_metrics["global_cka_to_embedding"].to_numpy(dtype=float),
                parent / str(args.global_cka_cdf_png).strip(),
                xlabel="Global CKA to embedding",
                title="CDF of global CKA to embedding",
            )

        if mode == "intersection":
            fisher_set: Set[str] = set()
            thr = float(args.global_fisher_min)
            for j, name in enumerate(bases):
                nm = str(name)
                if nm not in feat_in_merged:
                    continue
                fv = float(fr[j])
                if np.isfinite(fv) and fv >= thr:
                    fisher_set.add(nm)
            global_filtered = [f for f in top_cka_names if f in fisher_set and f in feat_in_merged]
            meta = global_metrics[global_metrics["tsfresh_feature"].isin(global_filtered)].copy()
            meta = meta.set_index("tsfresh_feature").loc[global_filtered].reset_index()
            meta["rank_in_cka_top"] = np.arange(1, len(meta) + 1)
        else:
            # fisher_min: Fisher ratio >= threshold, descending Fisher order
            thr = float(args.global_fisher_min)
            pairs: List[Tuple[str, float]] = []
            for j, name in enumerate(bases):
                nm = str(name)
                if nm not in feat_in_merged:
                    continue
                fv = float(fr[j])
                if np.isfinite(fv) and fv >= thr:
                    pairs.append((nm, fv))
            pairs.sort(key=lambda t: -t[1])
            global_filtered = [p[0] for p in pairs]
            meta = global_metrics[global_metrics["tsfresh_feature"].isin(global_filtered)].copy()
            meta = meta.sort_values("fisher_ratio", ascending=False, kind="mergesort").reset_index(drop=True)
            meta["rank_by_fisher_desc"] = np.arange(1, len(meta) + 1)

    out_glob = parent / "global_filtered_features.csv"
    meta.to_csv(out_glob, index=False)
    print(f"Wrote {out_glob} (n={len(global_filtered)})", flush=True)

    summary = parent / "global_filtered_intersection_summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        if args.global_filtered_features_csv:
            f.write(
                f"mode=precomputed_csv source={args.global_filtered_features_csv} "
                f"n_features={len(global_filtered)}\n"
            )
        elif mode == "intersection":
            f.write(
                f"mode=intersection n_cka_top={len(top_cka_names)} "
                f"n_fisher_min={len(fisher_set)} global_fisher_min={args.global_fisher_min} "
                f"n_intersection={len(global_filtered)}\n"
            )
        else:
            f.write(
                f"mode=fisher_min global_fisher_min={args.global_fisher_min} "
                f"n_features={len(global_filtered)}\n"
            )
    print(f"Wrote {summary}", flush=True)

    gf_set = set(global_filtered)
    if not gf_set:
        print("[global_filtered_overlay_prep] empty global filter set; skip per-cluster overlays.", flush=True)
        return

    # Cohen's d for global_filtered only.
    sub_gf = merged[[args.label_col] + global_filtered].copy()
    agg_gf = sub_gf.groupby(args.label_col, sort=False)[global_filtered].agg(
        [("mean", "mean"), ("var", _var)]
    )
    agg_gf.columns = [f"{c[0]}_{c[1]}" for c in agg_gf.columns]
    agg_gf.index.name = "cluster_id"
    agg_gf["n_members"] = sub_gf.groupby(args.label_col, sort=False).size()
    df_gf = agg_gf.reset_index()
    if df_gf.columns[0] != "cluster_id":
        df_gf = df_gf.rename(columns={df_gf.columns[0]: "cluster_id"})

    cohens_d_df = cohens_d_from_agg_df(agg_gf, eps=1e-3)
    cohens_d_df.columns = cohens_d_df.columns.map(str)
    cohens_d_df.index = cohens_d_df.index.map(str)

    # Per-feature max-normalise |d| across clusters with epsilon floor.
    # Tanh rescaling of |d|: tanh(|d| / d0) → [0, 1) when --cohens-d-tanh-d0 is set.
    if args.cohens_d_max_norm:
        _abs_d = cohens_d_df.abs()
        # cohens_d_df shape: (n_features, n_clusters) — rows=features, cols=clusters.
        # GLOBAL-SCALAR normalization: divide every value by the single largest |d|
        # across ALL features × ALL clusters.  This preserves absolute calibration —
        # clusters where no feature is genuinely discriminative get low scores everywhere.
        _global_max = float(_abs_d.to_numpy(dtype=float).ravel().max())
        _global_max_safe = max(_global_max, 1e-3)
        _max_per_feature = _abs_d.max(axis=1)   # one value per feature row  (for audit)
        _max_per_cluster = _abs_d.max(axis=0)   # one value per cluster col  (for audit)
        print(
            f"[NORM-AUDIT global_filtered_overlay_prep] cohens_d_df shape={cohens_d_df.shape} "
            f"(rows=features, cols=clusters)\n"
            f"  global scalar max |d| = {_global_max:.6f}  ← used for normalization\n"
            f"  per-feature max range = [{float(_max_per_feature.min()):.4f}, "
            f"{float(_max_per_feature.max()):.4f}]  (NOT used)\n"
            f"  per-cluster max range = [{float(_max_per_cluster.min()):.4f}, "
            f"{float(_max_per_cluster.max()):.4f}]  (NOT used)\n"
            f"  n_features with max|d|<0.01: {int((_max_per_feature < 0.01).sum())} / {len(_max_per_feature)}\n"
            f"  n_clusters with max|d|<0.01: {int((_max_per_cluster < 0.01).sum())} / {len(_max_per_cluster)}",
            flush=True,
        )
        d_norm_df: pd.DataFrame | None = (_abs_d / _global_max_safe).fillna(0.0)
        print(
            f"[global_filtered_overlay_prep] global-scalar max-norm (scalar={_global_max_safe:.6f}): "
            f"|d| range [{float(_abs_d.min().min()):.4f}, {float(_abs_d.max().max()):.4f}] "
            f"→ normalized range [{float(d_norm_df.min().min()):.4f}, {float(d_norm_df.max().max()):.4f}].",
            flush=True,
        )
    elif args.cohens_d_tanh_d0 is not None:
        _abs_d = cohens_d_df.abs()
        _d0_str = str(args.cohens_d_tanh_d0).strip().lower()
        if _d0_str == "median":
            _finite_vals = _abs_d.to_numpy(dtype=float).ravel()
            _finite_vals = _finite_vals[np.isfinite(_finite_vals)]
            _d0 = float(np.median(_finite_vals)) if _finite_vals.size > 0 else 1.0
            print(
                f"[global_filtered_overlay_prep] --cohens-d-tanh-d0=median: "
                f"d0={_d0:.4f} (median of {_finite_vals.size} finite |d| values).",
                flush=True,
            )
        else:
            _d0 = float(_d0_str)
        _d0 = max(_d0, 1e-6)  # guard against zero
        d_norm_df = np.tanh(_abs_d / _d0)
        print(
            f"[global_filtered_overlay_prep] tanh(|d| / {_d0:.4f}): "
            f"|d| range [{float(_abs_d.min().min()):.3f}, {float(_abs_d.max().max()):.3f}] "
            f"→ tanh range [{float(d_norm_df.min().min()):.4f}, {float(d_norm_df.max().max()):.4f}].",
            flush=True,
        )
    else:
        d_norm_df = None

    _global_max_abs_cohen = float(cohens_d_df.abs().to_numpy(dtype=float).max())
    _global_max_abs_cohen = max(_global_max_abs_cohen, 1e-12)

    cluster_ids = [str(x) for x in agg_gf.index.tolist()]
    local_k = int(args.local_overlay_top_k)
    cka_fname = f"overlay_cka_top{local_k}.csv"
    d_fname = f"overlay_cohen_d_top{local_k}.csv"
    aggregate_rows: list[dict[str, object]] = []

    cluster_importance_rows: List[Dict[str, Any]] = []
    all_top_parts: List[pd.DataFrame] = []

    _cka_vals: List[float] = []
    for _cid in cluster_ids:
        _subdir = parent / safe_cluster_dir_name(_cid)
        _ckp = _subdir / "selected_features_cka.csv"
        if not _ckp.is_file():
            continue
        _df = pd.read_csv(_ckp, low_memory=False)
        if _df.empty or "cka_to_embedding" not in _df.columns:
            continue
        _w = _df[_df["feature"].astype(str).isin(gf_set)].copy()
        _w["cka_to_embedding"] = pd.to_numeric(_w["cka_to_embedding"], errors="coerce")
        _cka_vals.extend(_w["cka_to_embedding"].dropna().tolist())
    global_max_cka = float(np.nanmax(_cka_vals)) if _cka_vals else 1.0
    global_max_cka = max(global_max_cka, 1e-12)
    print(
        f"[global_filtered_overlay_prep] global_max_cka (norm_cka divisor)={global_max_cka:.6f}",
        flush=True,
    )

    for cid in cluster_ids:
        subdir = parent / safe_cluster_dir_name(cid)
        if not subdir.is_dir():
            continue
        cka_path = subdir / "selected_features_cka.csv"
        if not cka_path.is_file():
            print(f"[global_filtered_overlay_prep] skip {subdir.name}: missing selected_features_cka.csv", flush=True)
            continue
        df_cka = pd.read_csv(cka_path, low_memory=False)
        if df_cka.empty or "feature" not in df_cka.columns or "cka_to_embedding" not in df_cka.columns:
            continue
        work = df_cka[df_cka["feature"].astype(str).isin(gf_set)].copy()
        work["cka_to_embedding"] = pd.to_numeric(work["cka_to_embedding"], errors="coerce")
        work = work.sort_values("cka_to_embedding", ascending=False, kind="mergesort").head(local_k)
        work = work.reset_index(drop=True)
        work["rank"] = np.arange(1, len(work) + 1)
        out_cka = work[["feature", "cka_to_embedding", "rank"]].copy()
        out_cka.to_csv(subdir / cka_fname, index=False)
        for _, r in out_cka.iterrows():
            aggregate_rows.append(
                {
                    "cluster_id": cid,
                    "order_type": "cka",
                    "rank": int(r["rank"]),
                    "feature": str(r["feature"]),
                    "cka_to_embedding": float(r["cka_to_embedding"]),
                    "cohen_d": np.nan,
                    "abs_cohen_d": np.nan,
                }
            )

        d_col = cid
        if d_col not in cohens_d_df.columns:
            for c in cohens_d_df.columns:
                if str(c) == str(cid):
                    d_col = str(c)
                    break
        if d_col not in cohens_d_df.columns:
            print(f"[global_filtered_overlay_prep] no Cohen d column for cluster {cid!r}", flush=True)
            continue

        d_rows = []
        for feat in global_filtered:
            if feat not in cohens_d_df.index:
                continue
            dv = float(pd.to_numeric(cohens_d_df.loc[feat, d_col], errors="coerce"))
            if not np.isfinite(dv):
                continue
            d_rows.append((feat, dv, abs(dv)))
        d_rows.sort(key=lambda t: -t[2])
        d_take = d_rows[:local_k]
        if d_take:
            df_d = pd.DataFrame(
                [{"feature": f, "cohen_d": d, "abs_cohen_d": ad} for f, d, ad in d_take]
            )
            df_d = df_d.reset_index(drop=True)
            df_d["rank"] = np.arange(1, len(df_d) + 1)
            df_d = df_d[["feature", "cohen_d", "abs_cohen_d", "rank"]]
        else:
            df_d = pd.DataFrame(columns=["feature", "cohen_d", "abs_cohen_d", "rank"])
        df_d.to_csv(subdir / d_fname, index=False)
        for _, r in df_d.iterrows():
            aggregate_rows.append(
                {
                    "cluster_id": cid,
                    "order_type": "cohen_d",
                    "rank": int(r["rank"]),
                    "feature": str(r["feature"]),
                    "cka_to_embedding": np.nan,
                    "cohen_d": float(r["cohen_d"]),
                    "abs_cohen_d": float(r["abs_cohen_d"]),
                }
            )

        # Local importance: norm_cka × norm_cohen (aligned with compute_normalized_cka_cohen_importance)
        cka_full = df_cka[["feature", "cka_to_embedding"]].copy()
        cka_full["feature"] = cka_full["feature"].astype(str)
        cka_full["cka_to_embedding"] = pd.to_numeric(cka_full["cka_to_embedding"], errors="coerce")
        cka_full = cka_full[cka_full["feature"].isin(gf_set)].copy()

        local = cka_full.copy()
        local["norm_cka"] = local["cka_to_embedding"] / global_max_cka
        local["feature"] = local["feature"].astype(str)
        local["cohens_d_mean"] = local["feature"].map(cohens_d_df[d_col]).fillna(np.nan)
        if d_norm_df is not None and d_col in d_norm_df.columns:
            norm_cohens = d_norm_df[d_col].copy()
            norm_cohens.index = norm_cohens.index.astype(str)
            local["norm_cohen"] = local["feature"].map(norm_cohens).fillna(0.0)
        else:
            local["norm_cohen"] = (
                local["feature"].map(cohens_d_df[d_col].abs()).fillna(0.0) / _global_max_abs_cohen
            )

        local["feature_importance"] = local["norm_cka"] * local["norm_cohen"]
        local = local[np.isfinite(local["feature_importance"])].copy()
        local.insert(0, "cluster_id", str(cid))

        local = local.sort_values(
            ["feature_importance", "norm_cka", "norm_cohen", "feature"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        local_all_path = subdir / str(args.local_importance_all_filename).strip()
        local.to_csv(local_all_path, index=False)

        local_top = local.head(local_k).copy()
        local_top.insert(1, "rank", np.arange(1, len(local_top) + 1))
        local_top_path = subdir / str(args.local_importance_top_filename).strip()
        local_top.to_csv(local_top_path, index=False)

        cluster_importance_rows.append(
            {
                "cluster_id": str(cid),
                "cluster_importance_topk_sum": float(local_top["feature_importance"].sum()),
                "n_top_features_used": int(len(local_top)),
            }
        )
        all_top_parts.append(local_top.copy())

    agg_path = parent / str(args.aggregate_rankings_csv).strip()
    if aggregate_rows:
        pd.DataFrame(aggregate_rows).to_csv(agg_path, index=False)
        print(f"Wrote {agg_path} (n_rows={len(aggregate_rows)})", flush=True)
    else:
        print("[global_filtered_overlay_prep] no aggregate rows (empty global set or missing cluster dirs).", flush=True)

    acsv = str(args.all_clusters_top_k_features_csv).strip()
    if acsv and all_top_parts:
        big_top = pd.concat(all_top_parts, ignore_index=True)
        means_df = collect_cluster_means_from_member_csvs(str(parent), big_top)
        merged_top = big_top.merge(
            means_df,
            on=["cluster_id", "feature"],
            how="left",
        )
        out_ac = parent / acsv
        merged_top.to_csv(out_ac, index=False)
        print(f"Wrote {out_ac} (n_rows={len(merged_top)})", flush=True)

    if cluster_importance_rows:
        df_ci = pd.DataFrame(cluster_importance_rows)
        out_ci = parent / str(args.cluster_importance_csv).strip()
        df_ci.to_csv(out_ci, index=False)
        print(f"Wrote {out_ci} (n_clusters={len(df_ci)})", flush=True)
        if args.enable_cdf_plots:
            _plot_cdf(
                df_ci["cluster_importance_topk_sum"].to_numpy(dtype=float),
                parent / str(args.cluster_importance_cdf_png).strip(),
                xlabel="Cluster importance (sum of top-K feature importance)",
                title="CDF of cluster importance",
            )
    else:
        print("[global_filtered_overlay_prep] no cluster-importance rows were written.", flush=True)

    print("[global_filtered_overlay_prep] done.", flush=True)


if __name__ == "__main__":
    main()
