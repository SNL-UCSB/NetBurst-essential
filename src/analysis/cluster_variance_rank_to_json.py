#!/usr/bin/env python3
"""
Read the per-cluster mean/variance CSV (from cluster_analysis.py --cluster-variance-out),
rank features by variance within each cluster, and write a JSON with rank, mean, and variance.

Usage:
  python cluster_variance_rank_to_json.py --variance-csv path/to/cluster_variance.csv --out path/to/out.json

JSON structure:
  {
    "<cluster_id>": {
      "n_members": 42,   // if present in variance CSV
      "<feature_name>": { "rank": 1, "mean": 0.5, "var": 0.01 },
      ...
    },
    ...
  }
Rank 1 = lowest variance (most invariant). NaN values are emitted as null.
"""

import argparse
import json
import os
from typing import Optional

import pandas as pd

from cluster_variance_stats import (
    feature_mean_var_pairs_from_variance_df,
    mean_across_clusters_matrix,
    variance_across_clusters_matrix,
)


def read_variance_csv(path: str) -> pd.DataFrame:
    """Load the variance CSV (cluster_id, feat1_mean, feat1_var, ...)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    if ext in [".parquet"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported extension: {path}. Use .csv or .parquet")


def scalar_for_json(x) -> Optional[float]:
    """Convert a scalar to a JSON-serializable value; NaN -> null."""
    if pd.isna(x):
        return None
    return float(x)


def main():
    ap = argparse.ArgumentParser(
        description="Rank features by variance per cluster and output JSON (rank, mean, var)."
    )
    ap.add_argument(
        "--variance-csv",
        required=True,
        help="Path to the per-cluster mean/variance CSV (columns: cluster_id, *_mean, *_var).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: same base name as --variance-csv with .json extension.",
    )
    ap.add_argument(
        "--cluster-id-col",
        default="cluster_id",
        help="Column name for cluster ID (default: cluster_id).",
    )
    ap.add_argument(
        "--variance-by-cluster-csv",
        default=None,
        help="Optional: CSV with variance across clusters. Rows = features, columns = cluster_id.",
    )
    ap.add_argument(
        "--mean-by-cluster-csv",
        default=None,
        help="Optional: CSV with mean across clusters. Rows = features, columns = cluster_id.",
    )
    args = ap.parse_args()

    df = read_variance_csv(args.variance_csv)

    if args.cluster_id_col not in df.columns:
        raise KeyError(
            f"Cluster ID column '{args.cluster_id_col}' not found. Columns: {list(df.columns)[:15]}..."
        )

    feature_pairs, _ = feature_mean_var_pairs_from_variance_df(
        df, cluster_id_col=args.cluster_id_col
    )

    # Per cluster: rank by variance (ascending -> rank 1 = lowest variance)
    result = {}
    cluster_ids = df[args.cluster_id_col].astype(str).tolist()

    for i, cid in enumerate(cluster_ids):
        row = df.iloc[i]
        var_vals = {
            feat: row[var_col]
            for feat, (_, var_col) in feature_pairs.items()
        }
        # Rank by variance ascending: lower variance -> lower rank (rank 1 = lowest variance)
        var_series = pd.Series(var_vals)
        ranks = var_series.rank(method="average", ascending=True, na_option="bottom").astype(int)

        result[cid] = {}
        if "n_members" in df.columns:
            result[cid]["n_members"] = int(row["n_members"]) if pd.notna(row["n_members"]) else None
        for feat, (mean_col, var_col) in feature_pairs.items():
            result[cid][feat] = {
                "rank": int(ranks[feat]),
                "mean": scalar_for_json(row[mean_col]),
                "var": scalar_for_json(row[var_col]),
            }

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(args.variance_csv)[0]
        out_path = base + ".json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote rank/mean/var for {len(result)} clusters, {len(feature_pairs)} features -> {out_path}")

    # Variance across clusters: one row per feature, one column per cluster
    if args.variance_by_cluster_csv:
        var_across = variance_across_clusters_matrix(
            df, feature_pairs, cluster_id_col=args.cluster_id_col
        )
        var_across.to_csv(args.variance_by_cluster_csv)
        print(f"Wrote variance across clusters ({var_across.shape[0]} features x {var_across.shape[1]} clusters) -> {args.variance_by_cluster_csv}")

    # Mean across clusters: one row per feature, one column per cluster
    mean_across = None
    if args.mean_by_cluster_csv:
        mean_across = mean_across_clusters_matrix(
            df, feature_pairs, cluster_id_col=args.cluster_id_col
        )
        mean_across.to_csv(args.mean_by_cluster_csv)
        print(f"Wrote mean across clusters ({mean_across.shape[0]} features x {mean_across.shape[1]} clusters) -> {args.mean_by_cluster_csv}")

if __name__ == "__main__":
    main()
