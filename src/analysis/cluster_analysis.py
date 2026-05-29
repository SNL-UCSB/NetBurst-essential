#!/usr/bin/env python3
"""
Find the most *invariant* TSFresh features within clusters.

Inputs:
  1) clustering results file with columns: ip, source_file, label, repr_dim_*
  2) TSFresh features file with columns: ip, source_file, <many feature columns>

Output:
  - A ranked list of TSFresh features that are most stable within clusters.
  - Also reports between-cluster separability (optional but useful).

Invariance metrics (computed per feature):
  - within_var_mean: mean variance within clusters (lower => more invariant)
  - within_cv_mean : mean coefficient of variation within clusters (lower => more invariant)
  - w_over_t       : within_var / total_var (lower => more cluster-stable)
  - eta2           : between_var / total_var (higher => better separation)

Global metrics (one value per run; written to cluster_quality_ch_db.json by default):
  - Calinski–Harabasz: higher => better separation in TSFresh Euclidean space
  - Davies–Bouldin: lower => tighter, better-separated clusters (Euclidean)

You can sort per-feature scores by any metric.
"""

import argparse
import hashlib
import json
import os
import time
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from cluster_fs import safe_cluster_dir_name
from cluster_variance_stats import (
    cohens_d_from_agg_df,
    fisher_feature_order_dataframe,
    population_stats_from_variance_df,
)
from per_cluster_cohens_d_cka import write_per_cluster_cohens_d_and_cka


def _resolve_path_after_staging(path: str) -> str:
    """
    If bash expanded ${STAGE_DIR} to empty, argv can be /clustering_j.csv (root). Rejoin using
    STAGE_DIR / NETBURST_STAGE_DIR (set by the shell or by --netburst-stage-dir for this process).
    Do not infer paths from SLURM_JOB_ID: that env can be stale across sessions.
    """
    path = os.path.expanduser(os.path.expandvars(path))
    stage = os.environ.get("STAGE_DIR") or os.environ.get("NETBURST_STAGE_DIR")
    base = os.path.basename(path)
    # Empty STAGE_DIR in "${STAGE_DIR}/name.csv" becomes "/name.csv" at filesystem root.
    if path != "/" + base:
        return path
    if not stage:
        return path
    if (base.startswith("clustering_") and base.endswith(".csv")) or base == "tsfresh.csv":
        return os.path.join(stage, base)
    return path


def read_table(path: str) -> pd.DataFrame:
    """Read CSV/Parquet into a DataFrame."""
    path = _resolve_path_after_staging(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    if ext in [".csv"]:
        # engine inference tends to be fine; set low_memory False to reduce dtype surprises
        return pd.read_csv(path, low_memory=False)
    if ext in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file extension: {ext}. Use .csv / .parquet / .feather")


def coerce_numeric_feature_block(df: pd.DataFrame, id_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keeps id_cols + numeric feature cols.
    If TSFresh features are strings, tries to coerce to numeric.
    """
    keep = df.copy()

    # Ensure id cols exist
    missing = [c for c in id_cols if c not in keep.columns]
    if missing:
        raise KeyError(f"Missing required id columns in TSFresh file: {missing}")

    feature_cols = [c for c in keep.columns if c not in id_cols]
    # Try to coerce to numeric for all features; non-coercible -> NaN
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(keep[c]):
            keep[c] = pd.to_numeric(keep[c], errors="coerce")

    # Keep only numeric features (drop all-NaN, too)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(keep[c])]
    # Drop columns that became entirely NaN after coercion
    numeric_cols = [c for c in numeric_cols if not keep[c].isna().all()]

    out = keep[id_cols + numeric_cols]
    return out, numeric_cols


def _rng_for_cluster(global_seed: int, cluster_id: Any) -> np.random.Generator:
    payload = f"{global_seed}\0{cluster_id!s}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    child_seed = int.from_bytes(digest[:8], "little", signed=False) % (2**63)
    return np.random.default_rng(child_seed)


def write_cluster_member_samples(
    merged: pd.DataFrame,
    *,
    label_col: str,
    id_ip: str,
    id_source_file: str,
    parent_dir: str,
    n_sample: int,
    global_seed: int,
) -> None:
    """
    For each cluster in merged, write members.json with up to n_sample randomly chosen rows
    (after join). Clusters with fewer than n_sample rows write all rows. Reproducible via
    global_seed and a per-cluster derived RNG.
    """
    parent = os.path.expanduser(parent_dir)
    os.makedirs(parent, exist_ok=True)
    id_cols = [id_ip, id_source_file]
    for cid, g in merged.groupby(label_col, sort=False):
        rng = _rng_for_cluster(global_seed, cid)
        k_take = min(n_sample, len(g))
        if k_take <= 0:
            continue
        idx = rng.choice(len(g), size=k_take, replace=False)
        sampled = g.iloc[idx][id_cols]
        records = sampled.to_dict(orient="records")
        sub = os.path.join(parent, safe_cluster_dir_name(cid))
        os.makedirs(sub, exist_ok=True)
        out_path = os.path.join(sub, "members.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    print(
        f"Wrote members.json (up to {n_sample} ids per cluster) under {parent}",
        flush=True,
    )


def write_per_cluster_member_tsfresh_csv(
    merged: pd.DataFrame,
    *,
    label_col: str,
    id_cols: List[str],
    feature_cols: List[str],
    parent_dir: str,
) -> None:
    """
    For each cluster, write ``cluster_*/members_tsfresh.csv``: one row per joined member
    (same rows as the clustering+TSFresh inner join) with ids, cluster label, and all
    numeric TSFresh columns (no embedding / repr columns).
    """
    parent = os.path.expanduser(parent_dir)
    os.makedirs(parent, exist_ok=True)
    base_cols = [c for c in id_cols + [label_col] if c in merged.columns]
    use_feats = sorted([c for c in feature_cols if c in merged.columns])
    if not use_feats:
        print(
            "[cluster_analysis] skip members_tsfresh.csv: no TSFresh columns in merged.",
            flush=True,
        )
        return
    out_cols = base_cols + use_feats
    n_clusters = 0
    for cid, g in merged.groupby(label_col, sort=False):
        sub = g[out_cols].copy()
        subdir = os.path.join(parent, safe_cluster_dir_name(cid))
        os.makedirs(subdir, exist_ok=True)
        out_path = os.path.join(subdir, "members_tsfresh.csv")
        sub.to_csv(out_path, index=False)
        n_clusters += 1
    print(
        f"Wrote members_tsfresh.csv ({len(use_feats)} TSFresh cols) for {n_clusters} clusters under {parent}",
        flush=True,
    )


def maybe_write_per_cluster_member_tsfresh_csv(
    args: Any,
    merged: pd.DataFrame,
    id_cols: List[str],
    feature_cols: List[str],
) -> None:
    """Honor ``--no-write-per-cluster-member-tsfresh-csv`` and parent-dir presence."""
    if getattr(args, "no_write_per_cluster_member_tsfresh_csv", False):
        return
    parent = args.per_cluster_thresholds_dir or args.cluster_members_parent_dir
    if not parent:
        return
    write_per_cluster_member_tsfresh_csv(
        merged,
        label_col=args.label_col,
        id_cols=id_cols,
        feature_cols=feature_cols,
        parent_dir=parent,
    )


def compute_and_write_global_cluster_metrics(
    merged: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    min_cluster_size: int,
    out_json_path: str,
) -> None:
    """
    Calinski–Harabasz (CH) and Davies–Bouldin (DB) on the full numeric TSFresh block
    (same columns as per-feature invariance). Rows: clusters with size >= min_cluster_size,
    then drop rows with non-finite values in any feature. Euclidean geometry in sklearn.
    """
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    out_json_path = os.path.abspath(os.path.expanduser(out_json_path))
    _dir = os.path.dirname(out_json_path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)

    record: dict[str, Any] = {
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "status": "pending",
        "min_cluster_size": int(min_cluster_size),
        "n_tsfresh_features": len(feature_cols),
        "notes": (
            "CH/DB use all numeric TSFresh features in merged, same row filter as "
            "per-feature scores (min_cluster_size + finite rows). Interpretation is global "
            "cluster quality in this high-dimensional Euclidean space, not per-feature."
        ),
    }

    if not feature_cols:
        record["status"] = "skipped"
        record["reason"] = "no_numeric_tsfresh_features"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Saved cluster quality (skipped) -> {out_json_path}", flush=True)
        return

    if label_col not in merged.columns:
        record["status"] = "error"
        record["error"] = f"missing label column {label_col!r}"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Saved cluster quality (error) -> {out_json_path}", flush=True)
        return

    cluster_sizes = merged[label_col].value_counts(dropna=False)
    valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
    record["n_clusters_passing_threshold"] = len(valid_clusters)

    if len(valid_clusters) < 2:
        record["status"] = "skipped"
        record["reason"] = "need_at_least_two_clusters_with_min_size"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Saved cluster quality (skipped: {record['reason']}) -> {out_json_path}", flush=True)
        return

    dfv = merged[merged[label_col].isin(valid_clusters)].copy()
    X = dfv[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = dfv[label_col].to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    record["n_samples_used"] = int(X.shape[0])

    if X.shape[0] < 2:
        record["status"] = "skipped"
        record["reason"] = "too_few_finite_rows"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Saved cluster quality (skipped: {record['reason']}) -> {out_json_path}", flush=True)
        return

    uniq = np.unique(y)
    record["n_distinct_labels_in_rows"] = int(len(uniq))
    if len(uniq) < 2:
        record["status"] = "skipped"
        record["reason"] = "need_at_least_two_distinct_labels_after_filtering"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"Saved cluster quality (skipped: {record['reason']}) -> {out_json_path}", flush=True)
        return

    try:
        record["calinski_harabasz"] = float(calinski_harabasz_score(X, y))
        record["davies_bouldin"] = float(davies_bouldin_score(X, y))
        record["status"] = "ok"
    except Exception as e:
        record["status"] = "error"
        record["error"] = repr(e)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(
        f"Saved global cluster quality (Calinski–Harabasz, Davies–Bouldin) -> {out_json_path}",
        flush=True,
    )


def compute_invariance_scores(
    merged: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    min_cluster_size: int = 5,
    progress_every: int = 200,
) -> pd.DataFrame:
    """
    Computes invariance/separability metrics per feature.

    For each cluster k:
      - var_k(feature) computed on non-NaN values
      - cv_k(feature) = std/|mean| (robustified with eps)

    Aggregation across clusters:
      - within_var_mean: mean(var_k) over clusters with size >= min_cluster_size
      - within_cv_mean : mean(cv_k)  over clusters with size >= min_cluster_size

    Global:
      - total_var: variance over all samples (non-NaN)
      - between_var: variance of cluster means (used for Fisher-style separation)
      - w_over_t = within_var_weighted / total_var
      - eta2 = between_var / total_var
      - fisher_ratio = between_var / within_var_mean (higher => better cluster separation)
    """
    if label_col not in merged.columns:
        raise KeyError(f"'{label_col}' not found in merged dataframe. Columns: {list(merged.columns)[:20]} ...")

    # Cluster sizes
    cluster_sizes = merged[label_col].value_counts(dropna=False)
    valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()

    if len(valid_clusters) == 0:
        raise ValueError(
            f"No clusters have size >= {min_cluster_size}. "
            f"Try lowering --min-cluster-size. Size summary:\n{cluster_sizes.head(20)}"
        )

    dfv = merged[merged[label_col].isin(valid_clusters)].copy()

    eps = 1e-12
    rows = []

    total_feats = len(feature_cols)
    computed_feats = 0
    start_t = time.time()
    print(
        f"Scoring invariance/separation for {total_feats:,} TSFresh features "
        f"(min_cluster_size={min_cluster_size}, progress_every={progress_every})...",
        flush=True,
    )

    # Precompute weights for weighted within variance
    total_n = len(dfv)
    weights = (dfv[label_col].value_counts() / total_n).to_dict()

    # Group once
    grouped = dfv.groupby(label_col, sort=False)

    # For performance: pull underlying array per feature
    for i, feat in enumerate(feature_cols, start=1):
        # Global stats (on valid rows only)
        x_all = dfv[feat]
        x_all = x_all.dropna()
        if x_all.shape[0] < 2:
            if progress_every > 0 and i % progress_every == 0:
                print(
                    f"Progress: considered {i:,}/{total_feats:,} features, "
                    f"computed {computed_feats:,} so far, elapsed={time.time() - start_t:.1f}s",
                    flush=True,
                )
            continue

        total_var = float(x_all.var(ddof=0))
        total_mean = float(x_all.mean())

        within_vars = []
        within_cvs = []
        cluster_means = []
        within_var_weighted = 0.0

        for k, g in grouped:
            x = g[feat].dropna()
            if x.shape[0] < 2:
                continue

            var_k = float(x.var(ddof=0))
            mean_k = float(x.mean())
            std_k = float(np.sqrt(var_k))
            cv_k = float(std_k / (abs(mean_k) + eps))

            within_vars.append(var_k)
            within_cvs.append(cv_k)
            cluster_means.append(mean_k)

            within_var_weighted += weights.get(k, 0.0) * var_k

        if len(within_vars) == 0:
            if progress_every > 0 and i % progress_every == 0:
                print(
                    f"Progress: considered {i:,}/{total_feats:,} features, "
                    f"computed {computed_feats:,} so far, elapsed={time.time() - start_t:.1f}s",
                    flush=True,
                )
            continue

        # Fisher/LDA-style: use variance of cluster means (equal cluster weighting)
        between_var = float(np.var(cluster_means, ddof=0))
        within_var_mean = float(np.mean(within_vars))
        w_over_t = within_var_weighted / (total_var + eps)
        eta2 = between_var / (total_var + eps)
        fisher_ratio = between_var / (within_var_mean + eps)  # higher => better separation

        rows.append(
            {
                "feature": feat,
                "total_mean": total_mean,
                "total_var": total_var,
                "within_var_mean": float(np.mean(within_vars)),
                "within_cv_mean": float(np.mean(within_cvs)),
                "within_var_weighted": within_var_weighted,
                "between_var": between_var,
                "w_over_t": w_over_t,   # lower => more invariant within clusters
                "eta2": eta2,           # higher => more between-cluster separation
                "fisher_ratio": fisher_ratio,  # between / within variance
                "clusters_used": len(valid_clusters),
            }
        )
        computed_feats += 1

        if progress_every > 0 and i % progress_every == 0:
            print(
                f"Progress: considered {i:,}/{total_feats:,} features, "
                f"computed {computed_feats:,} so far, elapsed={time.time() - start_t:.1f}s",
                flush=True,
            )

    out = pd.DataFrame(rows)
    print(
        f"Done scoring features: computed {computed_feats:,}/{total_feats:,} "
        f"(output rows={len(out):,}, elapsed={time.time() - start_t:.1f}s).",
        flush=True,
    )
    return out


def write_cohen_population_artifacts(
    agg_df: pd.DataFrame,
    *,
    variance_out_path: str,
    parent_for_per_cluster: Optional[str],
    fisher_reorder: bool,
) -> None:
    """Write Cohen's d and Fisher-order artifacts, plus optional per-cluster long CSVs."""
    df_in = agg_df.reset_index()
    if "cluster_id" not in df_in.columns and agg_df.index.name:
        df_in = df_in.rename(columns={df_in.columns[0]: str(agg_df.index.name)})
    hm = population_stats_from_variance_df(
        df_in,
        cluster_id_col="cluster_id",
        fisher_reorder=fisher_reorder,
    )
    cohens_d_df = cohens_d_from_agg_df(agg_df)
    cohens_d_df.index = cohens_d_df.index.map(str)
    cohens_d_df.columns = cohens_d_df.columns.map(str)
    cohens_d_df = cohens_d_df.loc[hm["feature_bases"], hm["cluster_ids"]]
    out_dir = os.path.dirname(os.path.abspath(os.path.expanduser(variance_out_path)))
    os.makedirs(out_dir, exist_ok=True)
    cohens_d_df.to_csv(os.path.join(out_dir, "cohens_d_across_clusters.csv"))
    fisher_feature_order_dataframe(hm).to_csv(os.path.join(out_dir, "feature_order_fisher.csv"))
    print(
        f"Saved Cohen's d (wide) and Fisher feature order -> {out_dir}",
        flush=True,
    )
    if not parent_for_per_cluster:
        return
    parent = os.path.expanduser(parent_for_per_cluster)
    for i, cid in enumerate(hm["cluster_ids"]):
        subdir = os.path.join(parent, safe_cluster_dir_name(cid))
        os.makedirs(subdir, exist_ok=True)
        rows = []
        for j, feat in enumerate(hm["feature_bases"]):
            rows.append(
                {
                    "feature": feat,
                    "mean_within": float(hm["M"][i, j]),
                    "var_within": float(hm["V"][i, j]),
                    "cohens_d_mean": float(cohens_d_df.loc[feat, str(cid)]),
                }
            )
        pd.DataFrame(rows).to_csv(os.path.join(subdir, "feature_mean_var_cohens_d.csv"), index=False)
    print(
        f"Saved per-cluster feature_mean_var_cohens_d.csv under {parent} ({len(hm['cluster_ids'])} clusters).",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustering", required=True, help="Path to clustering results (.csv/.parquet/.feather)")
    ap.add_argument("--tsfresh", required=True, help="Path to TSFresh features (.csv/.parquet/.feather)")
    ap.add_argument("--out", default="tsfresh_invariance_scores.csv", help="Output CSV for scores")
    ap.add_argument(
        "--per-cluster-out",
        default=None,
        help=(
            "Optional output CSV for per-cluster TSFresh value summaries of the selected "
            "invariant features. Writes long-form rows: label, feature, n, mean, std, min, max."
        ),
    )
    ap.add_argument("--min-cluster-size", type=int, default=5, help="Ignore clusters smaller than this")
    ap.add_argument("--id-ip", default="ip", help="Column name for IP id")
    ap.add_argument("--id-source-file", default="source_file", help="Column name for source_file id")
    ap.add_argument("--label-col", default="label", help="Column name for cluster label")
    ap.add_argument(
        "--global-fisher-min",
        type=float,
        default=0.1,
        help="Keep TSFresh features with fisher_ratio >= this threshold (Fisher-min selection).",
    )
    ap.add_argument(
        "--cluster-variance-out",
        default=None,
        dest="cluster_variance_out",
        help="Per-cluster mean/variance CSV for Fisher-selected features "
        "(not written with --only-per-cluster-cka). Required for full pipeline runs.",
    )
    ap.add_argument(
        "--cluster-feature-rank-out",
        default=None,
        dest="cluster_feature_rank_out",
        help="Per-cluster feature ranks by variance (rank 1 = lowest variance).",
    )
    ap.add_argument(
        "--write-cluster-member-samples",
        action="store_true",
        help="Write per-cluster cluster_*/members.json with sampled id rows under --cluster-members-parent-dir.",
    )
    ap.add_argument(
        "--cluster-members-parent-dir",
        default=None,
        help="Parent directory for cluster_*/members.json (required with --write-cluster-member-samples).",
    )
    ap.add_argument(
        "--member-sample-n",
        type=int,
        default=5,
        help="Random sample size per cluster for members.json (default 5); smaller clusters write all rows.",
    )
    ap.add_argument(
        "--member-sample-seed",
        type=int,
        default=0,
        help="Base RNG seed; each cluster uses a deterministic derived seed (default 0).",
    )
    ap.add_argument(
        "--no-write-per-cluster-member-tsfresh-csv",
        action="store_true",
        help=(
            "Do not write cluster_*/members_tsfresh.csv (ids, label, all numeric TSFresh columns) "
            "when --cluster-members-parent-dir or --per-cluster-thresholds-dir is set."
        ),
    )
    ap.add_argument(
        "--write-per-cluster-cohens-d-cka",
        action="store_true",
        help=(
            "Write per-cluster selected_features_cohens_d.csv (all Fisher-selected features ranked by |Cohen's d|), "
            "selected_features_cka.csv, and all_clusters_selected_features_cka_long.csv under parent "
            "(requires repr columns in clustering CSV). "
            "Also population CSVs from all TSFresh features: selected_features_highest_var_pop.csv, "
            "per_cluster_min_cohens_d_all_features.csv, per_cluster_max_cohens_d_all_features.csv, "
            "per_cluster_min_var_within_all_features.csv, "
            "per_cluster_cohens_d_and_variance_summary.csv (wide one-row-per-cluster summary)."
        ),
    )
    ap.add_argument(
        "--repr-prefix",
        default="repr_dim",
        help="Prefix for representation columns in clustering file (default repr_dim).",
    )
    ap.add_argument(
        "--per-cluster-thresholds-dir",
        default=None,
        help="Parent directory for cluster_*/ Cohen's d and CKA CSVs. Defaults to --cluster-members-parent-dir.",
    )
    ap.add_argument(
        "--per-cluster-cka-workers",
        type=int,
        default=None,
        help=(
            "Parallel threads for per-cluster Cohen's d/CKA (default: sequential). "
            "Values >= 2 enable a thread pool over clusters; BLAS is forced to 1 thread "
            "for the duration to limit oversubscription."
        ),
    )
    ap.add_argument(
        "--netburst-stage-dir",
        default=None,
        help=(
            "Absolute path to the input staging directory for this job (e.g. /dev/shm/netburst_invariance_<job>_<task>). "
            "Set from the Slurm script as --netburst-stage-dir \"${STAGE_DIR}\" so it always matches the live job; "
            "used to repair /clustering_*.csv argv if STAGE_DIR was empty at expansion time."
        ),
    )
    ap.add_argument(
        "--skip-population-cohen-artifacts",
        action="store_true",
        help="Do not write cohens_d_across_clusters.csv, feature_order_fisher.csv, or per-cluster feature_mean_var_cohens_d.csv.",
    )
    ap.add_argument(
        "--skip-heatmap-population-stats",
        action="store_true",
        help="Deprecated alias for --skip-population-cohen-artifacts.",
    )
    ap.add_argument(
        "--no-heatmap-fisher-reorder",
        action="store_true",
        help="Keep feature column order as in the variance CSV (default: reorder columns by Fisher ratio like the notebook heatmap).",
    )
    ap.add_argument(
        "--ip-suffix-filter",
        default=None,
        help=(
            "If set, keep only rows whose --id-ip value ends with this suffix before joining "
            "(e.g. /32 for single-host CIDRs). Applied to both clustering and TSFresh tables."
        ),
    )
    ap.add_argument(
        "--cluster-quality-out",
        default=None,
        help=(
            "JSON path for Calinski–Harabasz and Davies–Bouldin scores on the TSFresh block "
            "(default: <directory of --out>/cluster_quality_ch_db.json)."
        ),
    )
    ap.add_argument(
        "--skip-cluster-quality-metrics",
        action="store_true",
        help="Do not write cluster_quality_ch_db.json (global CH/DB on TSFresh features).",
    )
    ap.add_argument(
        "--only-per-cluster-cka",
        action="store_true",
        help=(
            "Join clustering + TSFresh, compute per-cluster Cohen/CKA artifacts only. "
            "Still computes invariance scores for Fisher-min selection but does not write --out, "
            "--cluster-variance-out, or --cluster-feature-rank-out. Implies --write-per-cluster-cohens-d-cka; use with "
            "--skip-cluster-quality-metrics and --skip-heatmap-population-stats (default forced on). "
            "Does not write members.json unless you also pass --write-cluster-member-samples."
        ),
    )
    args = ap.parse_args()

    args.skip_population_cohen_artifacts = bool(
        getattr(args, "skip_population_cohen_artifacts", False)
        or getattr(args, "skip_heatmap_population_stats", False)
    )

    if getattr(args, "only_per_cluster_cka", False):
        args.write_per_cluster_cohens_d_cka = True
        args.skip_cluster_quality_metrics = True
        args.skip_population_cohen_artifacts = True
    elif not args.cluster_variance_out:
        raise SystemExit("error: --cluster-variance-out is required unless --only-per-cluster-cka")

    # Ignore whitespace-only --netburst-stage-dir; otherwise treat like unset and use env in read_table.
    if args.netburst_stage_dir and str(args.netburst_stage_dir).strip():
        stage_abs = os.path.realpath(os.path.expanduser(os.path.expandvars(args.netburst_stage_dir)))
        os.environ["STAGE_DIR"] = stage_abs
        os.environ["NETBURST_STAGE_DIR"] = stage_abs

    id_cols = [args.id_ip, args.id_source_file]

    # Read clustering and keep only join cols + label
    clus = read_table(args.clustering)
    print(f"Read clustering file: {args.clustering}")
    missing = [c for c in id_cols + [args.label_col] if c not in clus.columns]
    if missing:
        raise KeyError(f"Missing required columns in clustering file: {missing}")

    repr_cols = [c for c in clus.columns if str(c).startswith(str(args.repr_prefix))]
    if args.write_per_cluster_cohens_d_cka and not repr_cols:
        raise ValueError(
            f"--write-per-cluster-cohens-d-cka requires representation columns with prefix "
            f"{args.repr_prefix!r} in the clustering file."
        )

    if args.write_per_cluster_cohens_d_cka:
        clus_small = clus[id_cols + [args.label_col] + repr_cols].copy()
    else:
        clus_small = clus[id_cols + [args.label_col]].copy()

    # Read tsfresh and coerce numeric
    ts = read_table(args.tsfresh)
    print(f"Read tsfresh file: {args.tsfresh}")
    ts_num, feature_cols = coerce_numeric_feature_block(ts, id_cols=id_cols)

    # Normalize join key dtypes to avoid pandas merge errors like:
    # "merge on int64 and object columns for key 'source_file'".
    for c in id_cols:
        if c in clus_small.columns:
            clus_small[c] = clus_small[c].astype(str)
        if c in ts_num.columns:
            ts_num[c] = ts_num[c].astype(str)

    if args.ip_suffix_filter:
        suf = str(args.ip_suffix_filter)
        ip_col = args.id_ip
        n0c, n0t = len(clus_small), len(ts_num)
        mask_c = clus_small[ip_col].astype(str).str.endswith(suf)
        mask_t = ts_num[ip_col].astype(str).str.endswith(suf)
        clus_small = clus_small.loc[mask_c].reset_index(drop=True)
        ts_num = ts_num.loc[mask_t].reset_index(drop=True)
        print(
            f"ip_suffix_filter={suf!r}: clustering {len(clus_small)}/{n0c} rows, "
            f"tsfresh {len(ts_num)}/{n0t} rows",
            flush=True,
        )

    # If representation columns share names with TSFresh features (e.g. TSFresh-as-repr),
    # keep TSFresh columns unsuffixed and suffix clustering-side overlaps with "_repr".
    repr_feature_overlap = set(repr_cols).intersection(feature_cols)

    # Inner join (only items that have both clustering + tsfresh)
    merged = clus_small.merge(ts_num, on=id_cols, how="inner", suffixes=("_repr", ""))

    repr_cols_merged: List[str] = []
    if args.write_per_cluster_cohens_d_cka:
        for c in repr_cols:
            if c in repr_feature_overlap and f"{c}_repr" in merged.columns:
                repr_cols_merged.append(f"{c}_repr")
            else:
                repr_cols_merged.append(c)

        missing_repr = [c for c in repr_cols_merged if c not in merged.columns]
        if missing_repr:
            raise KeyError(
                "Missing representation columns in merged dataframe after join: "
                f"{missing_repr[:10]}"
            )

    if merged.shape[0] == 0:
        raise ValueError(
            "Join produced 0 rows. Check that ip/source_file match exactly "
            "(same dtype/format/path prefixes)."
        )

    if not args.skip_cluster_quality_metrics:
        _out_dir = os.path.dirname(os.path.abspath(args.out))
        if not _out_dir:
            _out_dir = os.getcwd()
        cq_path = args.cluster_quality_out or os.path.join(
            _out_dir,
            "cluster_quality_ch_db.json",
        )
        compute_and_write_global_cluster_metrics(
            merged,
            feature_cols,
            args.label_col,
            args.min_cluster_size,
            cq_path,
        )

    if args.write_cluster_member_samples:
        if not args.cluster_members_parent_dir:
            raise ValueError(
                "--cluster-members-parent-dir is required when --write-cluster-member-samples is set."
            )
        write_cluster_member_samples(
            merged,
            label_col=args.label_col,
            id_ip=args.id_ip,
            id_source_file=args.id_source_file,
            parent_dir=args.cluster_members_parent_dir,
            n_sample=int(args.member_sample_n),
            global_seed=int(args.member_sample_seed),
        )

    scores = compute_invariance_scores(
        merged,
        feature_cols=feature_cols,
        label_col=args.label_col,
        min_cluster_size=args.min_cluster_size,
    )
    if scores.empty:
        raise ValueError("No features produced scores (likely too many NaNs or too few samples).")

    fisher_min = float(args.global_fisher_min)
    selected_features = (
        scores.loc[scores["fisher_ratio"] >= fisher_min]
        .sort_values("fisher_ratio", ascending=False)["feature"]
        .astype(str)
        .tolist()
    )
    selected_features = [f for f in selected_features if f in feature_cols]
    if not selected_features:
        raise ValueError(
            f"No features passed fisher_ratio >= {fisher_min}. "
            "Lower --global-fisher-min or inspect scores."
        )
    print(
        f"Fisher-min selection: {len(selected_features)} features with fisher_ratio >= {fisher_min}.",
        flush=True,
    )

    # Compute mean and variance per cluster for the selected feature universe.
    sub = merged[[args.label_col] + selected_features].copy()

    def _var(x):
        d = x.dropna()
        return d.var(ddof=0) if d.size >= 2 else np.nan

    agg_df = sub.groupby(args.label_col, sort=False)[selected_features].agg(
        [("mean", "mean"), ("var", _var)]
    )
    agg_df.columns = [f"{c[0]}_{c[1]}" for c in agg_df.columns]
    agg_df.index.name = "cluster_id"
    agg_df["n_members"] = sub.groupby(args.label_col, sort=False).size()

    # Full TSFresh block for population var_pop extremes (all numeric features).
    agg_df_population: Optional[pd.DataFrame] = None
    if args.only_per_cluster_cka or args.write_per_cluster_cohens_d_cka:
        sub_all = merged[[args.label_col] + feature_cols].copy()
        agg_df_population = sub_all.groupby(args.label_col, sort=False)[feature_cols].agg(
            [("mean", "mean"), ("var", _var)]
        )
        agg_df_population.columns = [f"{c[0]}_{c[1]}" for c in agg_df_population.columns]
        agg_df_population.index.name = "cluster_id"
        agg_df_population["n_members"] = sub_all.groupby(args.label_col, sort=False).size()

    if args.only_per_cluster_cka:
        thresholds_parent = args.per_cluster_thresholds_dir or args.cluster_members_parent_dir
        if not thresholds_parent:
            raise ValueError(
                "--only-per-cluster-cka requires --per-cluster-thresholds-dir or --cluster-members-parent-dir."
            )
        write_per_cluster_cohens_d_and_cka(
            merged,
            label_col=args.label_col,
            repr_cols=repr_cols_merged,
            selected_features=selected_features,
            agg_df=agg_df,
            agg_df_population=agg_df_population,
            parent_dir=thresholds_parent,
            max_cluster_workers=args.per_cluster_cka_workers,
        )
        maybe_write_per_cluster_member_tsfresh_csv(args, merged, id_cols, feature_cols)
        print("--only-per-cluster-cka: finished (no invariant_scores.csv or cluster variance/rank CSVs).", flush=True)
        return

    agg_df.to_csv(args.cluster_variance_out)
    print(
        f"Saved per-cluster mean and variance for {len(selected_features)} selected features -> {args.cluster_variance_out}"
    )

    if not args.skip_population_cohen_artifacts:
        thresholds_parent_early = args.per_cluster_thresholds_dir or args.cluster_members_parent_dir
        write_cohen_population_artifacts(
            agg_df,
            variance_out_path=args.cluster_variance_out,
            parent_for_per_cluster=thresholds_parent_early,
            fisher_reorder=not args.no_heatmap_fisher_reorder,
        )

    # Rank features by variance per cluster (rank 1 = lowest variance = most invariant)
    var_cols = [c for c in agg_df.columns if c.endswith("_var")]
    rank_df = agg_df[var_cols].rank(axis=1, method="average", na_option="bottom")
    rank_df.columns = [c.replace("_var", "") for c in var_cols]
    rank_df.index.name = "cluster_id"
    if args.cluster_feature_rank_out:
        rank_df.to_csv(args.cluster_feature_rank_out)
        print(f"Saved per-cluster feature ranks (by variance) -> {args.cluster_feature_rank_out}")

    thresholds_parent = args.per_cluster_thresholds_dir or args.cluster_members_parent_dir
    if args.write_per_cluster_cohens_d_cka:
        if not thresholds_parent:
            raise ValueError(
                "--write-per-cluster-cohens-d-cka requires --per-cluster-thresholds-dir or "
                "--cluster-members-parent-dir."
            )
        write_per_cluster_cohens_d_and_cka(
            merged,
            label_col=args.label_col,
            repr_cols=repr_cols_merged,
            selected_features=selected_features,
            agg_df=agg_df,
            agg_df_population=agg_df_population,
            parent_dir=thresholds_parent,
            max_cluster_workers=args.per_cluster_cka_workers,
        )

    maybe_write_per_cluster_member_tsfresh_csv(args, merged, id_cols, feature_cols)

    # Global score listing is ordered by Fisher ratio across all clusters.
    scores = scores.sort_values("fisher_ratio", ascending=False).reset_index(drop=True)

    # Save full table
    scores.to_csv(args.out, index=False)

    # Optional: per-cluster value summaries for Fisher-selected features
    if args.per_cluster_out:
        top_features = selected_features

        # Build a small frame for aggregation: label + selected features
        sub = merged[[args.label_col] + top_features].copy()

        agg = (
            sub.groupby(args.label_col, sort=False)[top_features]
            .agg(["count", "mean", "std", "min", "max"])
        )

        # Convert to long form: one row per (label, feature)
        long_df = agg.stack(level=0).reset_index()
        long_df = long_df.rename(
            columns={
                args.label_col: "label",
                "level_1": "feature",
                "count": "n",
            }
        )
        # Ensure consistent column order
        keep_cols = ["label", "feature", "n", "mean", "std", "min", "max"]
        long_df = long_df[keep_cols]
        long_df.to_csv(args.per_cluster_out, index=False)

    # Display only Fisher-selected features
    show_cols = ["feature", "within_var_mean", "within_cv_mean", "w_over_t", "eta2", "fisher_ratio", "total_var"]
    display_df = scores[scores["feature"].isin(selected_features)].sort_values(
        "fisher_ratio", ascending=False
    ).reset_index(drop=True)
    print(f"\nMerged rows: {merged.shape[0]:,}")
    print(f"TSFresh numeric features scored: {len(scores):,}")
    print(f"Saved scores -> {args.out}")
    print(f"Selected features ({len(display_df)} rows):\n")
    print(display_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
