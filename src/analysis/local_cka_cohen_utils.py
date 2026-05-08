"""
Shared local CKA × Cohen's d feature importance (max-normalized globally, then product).

Single source of truth for normalization × inner join × top-k used by
``compute_normalized_cka_cohen_importance.py`` and ``global_filtered_overlay_prep.py``
(so Slurm and standalone runs cannot drift).
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from cluster_fs import safe_cluster_dir_name


def cohens_d_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide (feature × cluster_id) -> long with cohens_d, abs_cohens_d."""
    if "feature" in df_wide.columns:
        df_long = df_wide.melt(id_vars="feature", var_name="cluster_id", value_name="cohens_d")
    else:
        df2 = df_wide.reset_index()
        name0 = df2.columns[0]
        df_long = df2.melt(id_vars=name0, var_name="cluster_id", value_name="cohens_d")
        if name0 != "feature":
            df_long = df_long.rename(columns={name0: "feature"})
    df_long["cluster_id"] = df_long["cluster_id"].astype(int)
    df_long["abs_cohens_d"] = df_long["cohens_d"].abs()
    return df_long


def compute_feature_importance_long(
    cka_long: pd.DataFrame,
    cohen_long: pd.DataFrame,
    *,
    cka_max_eps: float = 1e-12,
    cohen_max_eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Global max-norm of cka_local and |cohens_d|, inner join, product.
    cka_long: cluster_id, feature, cka_local
    cohen_long: cluster_id, feature, cohens_d, abs_cohens_d
    """
    g_cka = float(cka_long["cka_local"].max())
    g_co = float(cohen_long["abs_cohens_d"].max())
    g_cka = max(g_cka, cka_max_eps)
    g_co = max(g_co, cohen_max_eps)

    a = cka_long.copy()
    b = cohen_long.copy()
    a["norm_cka"] = a["cka_local"] / g_cka
    b["norm_cohen"] = b["abs_cohens_d"] / g_co

    merged = pd.merge(
        a[["cluster_id", "feature", "cka_local", "norm_cka"]],
        b[["cluster_id", "feature", "cohens_d", "abs_cohens_d", "norm_cohen"]],
        on=["cluster_id", "feature"],
        how="inner",
    )
    merged["feature_importance"] = merged["norm_cka"] * merged["norm_cohen"]
    merged = merged.sort_values(["cluster_id", "feature_importance"], ascending=[True, False])
    return merged


def top_k_per_cluster(
    importance_df: pd.DataFrame,
    k: int,
    *,
    cluster_col: str = "cluster_id",
    importance_col: str = "feature_importance",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top-k rows per cluster; cluster_importance = sum of top-k importance."""
    top = (
        importance_df.groupby(cluster_col, sort=False)
        .apply(lambda g: g.nlargest(int(k), importance_col), include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )
    cluster_imp = (
        top.groupby(cluster_col)[importance_col]
        .agg(cluster_importance_topk_sum="sum", n_top_features_used="count")
        .reset_index()
        .sort_values("cluster_importance_topk_sum", ascending=False)
    )
    return top, cluster_imp


def collect_cluster_means_from_member_csvs(
    parent: str,
    top_df: pd.DataFrame,
    *,
    fname: str = "feature_mean_var_cohens_d.csv",
    mean_key: str = "mean_within",
) -> pd.DataFrame:
    """Build long table of cluster_mean from each cluster_*/feature_mean_var_cohens_d.csv."""
    rows = []
    for _, r in top_df.iterrows():
        cid = r["cluster_id"]
        feat = str(r["feature"])
        sub = os.path.join(parent, safe_cluster_dir_name(cid))
        path = os.path.join(sub, fname)
        if not os.path.isfile(path):
            rows.append({"cluster_id": cid, "feature": feat, "cluster_mean": np.nan})
            continue
        subdf = pd.read_csv(path, low_memory=False)
        if subdf.empty or "feature" not in subdf.columns:
            rows.append({"cluster_id": cid, "feature": feat, "cluster_mean": np.nan})
            continue
        m = subdf.loc[subdf["feature"].astype(str) == feat, mean_key]
        val = float(m.iloc[0]) if len(m) else np.nan
        rows.append({"cluster_id": cid, "feature": feat, "cluster_mean": val})
    return pd.DataFrame(rows)
