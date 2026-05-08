#!/usr/bin/env python3
"""
Shared variance / mean across clusters from cluster_analysis variance CSV.
Used by cluster_variance_rank_to_json.py and cluster_analysis.py (per-cluster thresholds).

``population_stats_from_variance_df`` computes Fisher ratios, per-cluster M/V, and
population mean/variance for reordering — no notebook heatmap matrix dependencies.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def feature_mean_var_pairs_from_variance_df(
    df: pd.DataFrame,
    *,
    cluster_id_col: str = "cluster_id",
) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
    """
    Build feature_name -> (mean_col, var_col) from per-cluster variance CSV.
    """
    mean_cols = [
        c
        for c in df.columns
        if c not in (cluster_id_col, "n_members") and c.endswith("_mean")
    ]
    var_suffix = "_var"
    mean_suffix = "_mean"
    feature_pairs: Dict[str, Tuple[str, str]] = {}
    for c in mean_cols:
        base = c[: -len(mean_suffix)]
        if base + var_suffix in df.columns:
            feature_pairs[base] = (c, base + var_suffix)
    if not feature_pairs:
        raise ValueError("No *_mean / *_var column pairs found in variance CSV.")
    return feature_pairs, list(feature_pairs.keys())


def mean_across_clusters_matrix(
    df: pd.DataFrame,
    feature_pairs: Dict[str, Tuple[str, str]],
    *,
    cluster_id_col: str = "cluster_id",
) -> pd.DataFrame:
    """Rows = features, columns = cluster_id (mean per cluster for each feature)."""
    mean_across = df.set_index(cluster_id_col)[[p[0] for p in feature_pairs.values()]].T
    mean_across.index = list(feature_pairs.keys())
    mean_across.index.name = "feature"
    return mean_across


def cohens_d_from_agg_df(
    agg_df: pd.DataFrame,
    *,
    eps: float = 1e-3,
) -> pd.DataFrame:
    """
    Cohen's d effect size per (feature, cluster): rows = features, columns = cluster_id.

    For each cluster c and feature f:

        d_{c,f} = (M_c - M_rest_c) / denom

    where:
        M_rest_c = (sum_{i≠c} n_i * M_i) / (N - n_c)  — size-weighted complement mean
        V_c      = within-cluster variance of cluster c on feature f (ddof=0)
        denom    = max(sqrt(V_c), eps * (1 + |M_rest_c|))

    The denominator floor eps * (1 + |M_rest_c|) is scale-adaptive: for large-scale
    features (e.g. byte counts where M_rest ~ 1e6) the floor is ~1e3, preventing
    blowup; for ratio features (M_rest ~ 1) the floor is merely ~1e-3. This avoids
    inflating scores for features whose raw values are tiny relative to their spread.

    Returns a DataFrame with rows = feature names, columns = cluster_id.
    """
    var_df = agg_df.reset_index()
    cluster_id_col = var_df.columns[0]
    feature_pairs, bases = feature_mean_var_pairs_from_variance_df(
        var_df, cluster_id_col=cluster_id_col
    )

    n_c = len(var_df)
    n_f = len(bases)
    cluster_ids = var_df[cluster_id_col].tolist()

    M = np.zeros((n_c, n_f), dtype=float)
    V = np.zeros((n_c, n_f), dtype=float)
    for j, base in enumerate(bases):
        mcol, vcol = feature_pairs[base]
        M[:, j] = pd.to_numeric(var_df[mcol], errors="coerce").to_numpy()
        V[:, j] = pd.to_numeric(var_df[vcol], errors="coerce").to_numpy()

    # Cluster sizes
    if "n_members" in var_df.columns:
        n_members = pd.to_numeric(var_df["n_members"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        n_members = np.clip(n_members, 1.0, None)
    else:
        n_members = np.ones(n_c, dtype=float)

    N_total = n_members.sum()  # scalar
    V_clipped = np.clip(V, 0.0, None)

    # Size-weighted complement mean per (cluster, feature)
    sum_nM = (n_members[:, None] * M).sum(axis=0)        # (n_f,)
    n_c_vec = n_members[:, None]                          # (n_c, 1)
    n_rest = N_total - n_c_vec
    M_rest = (sum_nM[None, :] - n_c_vec * M) / np.maximum(n_rest, 1.0)  # (n_c, n_f)

    # Denominator: per-cluster within-group std, floored at eps*(1+|M_rest|)
    # so scale differences across features (bytes vs. ratios) don't cause blowup.
    std_c = np.sqrt(V_clipped)                            # (n_c, n_f)
    scale_floor = eps * (1.0 + np.abs(M_rest))            # (n_c, n_f)
    denom = np.maximum(std_c, scale_floor)                # (n_c, n_f)

    D = (M - M_rest) / denom
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    n_floored = int(np.sum(std_c < scale_floor))
    print(
        f"[cohens_d_from_agg_df] Cohen's d  eps={eps:.0e}  "
        f"shape=({n_c} clusters, {n_f} features)\n"
        f"  formula: (M_c - M_rest_c) / max(sqrt(V_c), eps*(1+|M_rest_c|))  [scale-adaptive floor]\n"
        f"  n_(cluster,feat) pairs where std was floored: {n_floored} / {n_c * n_f} "
        f"({100.0 * n_floored / max(n_c * n_f, 1):.1f}%)\n"
        f"  |d| range: [{float(np.abs(D).min()):.4f}, {float(np.abs(D).max()):.4f}]",
        flush=True,
    )

    # Return as DataFrame: rows = features, columns = cluster_id
    out = pd.DataFrame(D.T, index=bases, columns=cluster_ids)
    out.index.name = "feature"
    return out


def variance_across_clusters_matrix(
    df: pd.DataFrame,
    feature_pairs: Dict[str, Tuple[str, str]],
    *,
    cluster_id_col: str = "cluster_id",
) -> pd.DataFrame:
    var_across = df.set_index(cluster_id_col)[[p[1] for p in feature_pairs.values()]].T
    var_across.index = list(feature_pairs.keys())
    var_across.index.name = "feature"
    return var_across


def population_stats_from_variance_df(
    df: pd.DataFrame,
    *,
    cluster_id_col: str = "cluster_id",
    fisher_reorder: bool = True,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Weighted population mean/variance and Fisher-style ratios for feature ordering.

    Returns dict with:
      cluster_ids, feature_bases, M, V, mean_pop, std_pop,
      weights, Wsum, fisher_ratio_vis (per column in final order).
    """
    feature_pairs, bases_in_file_order = feature_mean_var_pairs_from_variance_df(
        df, cluster_id_col=cluster_id_col
    )
    cluster_ids = df[cluster_id_col].astype(str).tolist()
    n_c = len(cluster_ids)
    n_f = len(bases_in_file_order)

    n_members_col = "n_members" if "n_members" in df.columns else None
    if n_members_col:
        weights = pd.to_numeric(df[n_members_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    else:
        weights = np.ones(n_c, dtype=float)
    weights = np.clip(weights, 0.0, None)
    wsum = float(weights.sum())
    if wsum <= 0:
        raise ValueError("Sum of n_members/weights is <= 0; cannot compute population normalization.")

    M = np.zeros((n_c, n_f), dtype=float)
    V = np.zeros((n_c, n_f), dtype=float)
    for j, base in enumerate(bases_in_file_order):
        mcol, vcol = feature_pairs[base]
        M[:, j] = pd.to_numeric(df[mcol], errors="coerce").to_numpy()
        V[:, j] = pd.to_numeric(df[vcol], errors="coerce").to_numpy()

    within_var_mean = np.nanmean(V, axis=0)
    between_var = np.nanvar(M, axis=0, ddof=0)
    fisher_ratio_raw = between_var / (within_var_mean + eps)
    if fisher_reorder:
        order = np.argsort(fisher_ratio_raw)[::-1]
        fisher_ratio_vis = fisher_ratio_raw[order]
        bases_ordered = [bases_in_file_order[i] for i in order]
        M = M[:, order]
        V = V[:, order]
    else:
        fisher_ratio_vis = fisher_ratio_raw
        bases_ordered = list(bases_in_file_order)

    W = weights[:, None]
    mean_pop = np.nansum(W * M, axis=0) / wsum
    var_pop = np.nansum(W * (V + (M - mean_pop[None, :]) ** 2), axis=0) / wsum
    std_pop = np.sqrt(var_pop)

    return {
        "cluster_ids": cluster_ids,
        "feature_bases": bases_ordered,
        "M": M,
        "V": V,
        "mean_pop": mean_pop,
        "std_pop": std_pop,
        "weights": weights,
        "Wsum": wsum,
        "fisher_ratio_vis": fisher_ratio_vis,
    }


# Deprecated alias (removed Z/alpha notebook heatmap matrices).
heatmap_population_z_alpha_from_variance_df = population_stats_from_variance_df


def fisher_feature_order_dataframe(heatmap_result: Dict[str, Any]) -> pd.DataFrame:
    bases = heatmap_result["feature_bases"]
    fr = heatmap_result["fisher_ratio_vis"]
    return pd.DataFrame(
        {
            "feature_base": bases,
            "heatmap_col_index": np.arange(len(bases), dtype=int),
            "fisher_ratio_vis": fr,
        }
    )
