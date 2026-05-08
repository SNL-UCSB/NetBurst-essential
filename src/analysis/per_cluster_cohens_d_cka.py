#!/usr/bin/env python3
"""
Per-cluster Cohen's d filtering and embedding-vs-TSFresh linear CKA.

Uses shared math from cluster_variance_stats.py and cka_core.py.

``selected_features_cohens_d.csv`` lists Fisher-selected features ranked by ``|Cohen's d|``.
For the per-cluster population-normalized artifacts used by cluster importance,
see ``feature_mean_var_cohens_d.csv`` from cluster_analysis.py and
``cohens_d_across_clusters.csv`` in the same directory as ``--cluster-variance-out``.

CKA: for each cluster, linear CKA between the full embedding matrix (all ``repr_dim*`` columns)
and each TSFresh column (1D), on cluster rows only (needs >=2 members).

CKA is computed for Fisher-selected features only; results are written to
``selected_features_cka.csv`` and ``all_clusters_selected_features_cka_long.csv``.
``all_clusters_selected_features_cka_long.csv`` aggregates cluster × Fisher-selected feature CKA.

Population-normalized artifacts use **all numeric TSFresh features** (not only the Fisher-selected list),
reusing the same weighted population means as the heatmap:

  - ``selected_features_highest_var_pop.csv`` — global top-N features by marginal ``var_pop``.
    - ``per_cluster_min_cohens_d_all_features.csv`` — per cluster, the feature with **minimum** Cohen's d.
    - ``per_cluster_max_cohens_d_all_features.csv`` — per cluster, the feature with **maximum** Cohen's d.
  - ``per_cluster_min_var_within_all_features.csv`` — per cluster, the feature with **minimum**
        raw within-cluster variance ``V`` (same ``var_within`` as heatmap / ``feature_mean_var_cohens_d``).
    - ``per_cluster_cohens_d_and_variance_summary.csv`` — **one row per cluster** with
        Cohen's d min/max (across all features), marginal ``var_pop`` at those features, and
    min/max raw ``var_within`` across features (see module docstring on that function).
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cka_core import linear_cka, safe_feature_stats
from cluster_fs import safe_cluster_dir_name
from cluster_variance_stats import (
    cohens_d_from_agg_df,
    feature_mean_var_pairs_from_variance_df,
    population_stats_from_variance_df,
)


def _agg_df_to_cluster_id_frame(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Reset index so first column is ``cluster_id`` (for heatmap / population Z)."""
    df_in = agg_df.reset_index()
    if df_in.columns[0] != "cluster_id":
        df_in = df_in.rename(columns={df_in.columns[0]: "cluster_id"})
    return df_in


def _nan_argmin_idx(vals: np.ndarray) -> int:
    """Index of minimum among finite values; ``-1`` if none."""
    v = np.asarray(vals, dtype=float)
    if not np.any(np.isfinite(v)):
        return -1
    return int(np.nanargmin(v))


def _nan_argmax_idx(vals: np.ndarray) -> int:
    """Index of maximum among finite values; ``-1`` if none."""
    v = np.asarray(vals, dtype=float)
    if not np.any(np.isfinite(v)):
        return -1
    return int(np.nanargmax(v))


def write_population_all_features_csvs(
    agg_df_all: pd.DataFrame,
    parent: str,
    *,
    top_n_global: int = 50,
    fisher_reorder: bool = True,
) -> None:
    """
    Population heatmap on **all** TSFresh features: global top-N by ``var_pop``, and per-cluster
    extrema of Cohen's d and of raw within-cluster variance ``V``.
    """
    df_in = _agg_df_to_cluster_id_frame(agg_df_all)
    hm = population_stats_from_variance_df(
        df_in,
        cluster_id_col="cluster_id",
        fisher_reorder=fisher_reorder,
    )
    bases: List[str] = list(hm["feature_bases"])
    std_pop = np.asarray(hm["std_pop"], dtype=float)
    var_pop = np.square(std_pop)
    fisher_vis = np.asarray(hm["fisher_ratio_vis"], dtype=float)

    order_high = np.argsort(-var_pop, kind="mergesort")[: int(top_n_global)]
    rows_top: List[Dict[str, Any]] = []
    for r, j in enumerate(order_high, start=1):
        jj = int(j)
        rows_top.append(
            {
                "rank": r,
                "feature": bases[jj],
                "var_pop": float(var_pop[jj]),
                "std_pop": float(std_pop[jj]),
                "fisher_ratio_vis": float(fisher_vis[jj]),
            }
        )
    out_top = os.path.join(parent, "selected_features_highest_var_pop.csv")
    pd.DataFrame(rows_top).to_csv(out_top, index=False)
    print(
        f"Wrote top-{top_n_global} features by population variance (all TSFresh features) -> {out_top}",
        flush=True,
    )

    cohens_d_df = cohens_d_from_agg_df(agg_df_all)
    cohens_d_df.index = cohens_d_df.index.map(str)
    cohens_d_df.columns = cohens_d_df.columns.map(str)
    cluster_ids = [str(x) for x in hm["cluster_ids"]]
    D = cohens_d_df.loc[bases, cluster_ids].to_numpy(dtype=float).T
    V = np.asarray(hm["V"], dtype=float)

    rows_dmin: List[Dict[str, Any]] = []
    rows_dmax: List[Dict[str, Any]] = []
    rows_vmin: List[Dict[str, Any]] = []
    for i, cid in enumerate(cluster_ids):
        drow = D[i, :]
        vrow = V[i, :]
        jdmin = _nan_argmin_idx(drow)
        jdmax = _nan_argmax_idx(drow)
        jvmin = _nan_argmin_idx(vrow)

        if jdmin < 0:
            rows_dmin.append(
                {
                    "cluster_id": cid,
                    "feature": "",
                    "cohens_d": float("nan"),
                    "var_within": float("nan"),
                    "var_pop": float("nan"),
                }
            )
        else:
            rows_dmin.append(
                {
                    "cluster_id": cid,
                    "feature": bases[jdmin],
                    "cohens_d": float(drow[jdmin]),
                    "var_within": float(vrow[jdmin]),
                    "var_pop": float(var_pop[jdmin]),
                }
            )

        if jdmax < 0:
            rows_dmax.append(
                {
                    "cluster_id": cid,
                    "feature": "",
                    "cohens_d": float("nan"),
                    "var_within": float("nan"),
                    "var_pop": float("nan"),
                }
            )
        else:
            rows_dmax.append(
                {
                    "cluster_id": cid,
                    "feature": bases[jdmax],
                    "cohens_d": float(drow[jdmax]),
                    "var_within": float(vrow[jdmax]),
                    "var_pop": float(var_pop[jdmax]),
                }
            )

        if jvmin < 0:
            rows_vmin.append(
                {
                    "cluster_id": cid,
                    "feature": "",
                    "var_within": float("nan"),
                    "z_pop": float("nan"),
                    "var_pop": float("nan"),
                }
            )
        else:
            rows_vmin.append(
                {
                    "cluster_id": cid,
                    "feature": bases[jvmin],
                    "var_within": float(vrow[jvmin]),
                    "cohens_d": float(drow[jvmin]),
                    "var_pop": float(var_pop[jvmin]),
                }
            )

    out_dmin = os.path.join(parent, "per_cluster_min_cohens_d_all_features.csv")
    out_dmax = os.path.join(parent, "per_cluster_max_cohens_d_all_features.csv")
    out_vmin = os.path.join(parent, "per_cluster_min_var_within_all_features.csv")
    pd.DataFrame(rows_dmin).to_csv(out_dmin, index=False)
    pd.DataFrame(rows_dmax).to_csv(out_dmax, index=False)
    pd.DataFrame(rows_vmin).to_csv(out_vmin, index=False)
    print(f"Wrote per-cluster min Cohen's d (all features) -> {out_dmin}", flush=True)
    print(f"Wrote per-cluster max Cohen's d (all features) -> {out_dmax}", flush=True)
    print(f"Wrote per-cluster min within-cluster variance (all features) -> {out_vmin}", flush=True)

    _write_per_cluster_population_summary_csv(
        parent=parent,
        cluster_ids=cluster_ids,
        bases=bases,
        var_pop=var_pop,
        D=D,
        V=V,
        agg_df_all=agg_df_all,
    )


def _write_per_cluster_population_summary_csv(
    *,
    parent: str,
    cluster_ids: List[str],
    bases: List[str],
    var_pop: np.ndarray,
    D: np.ndarray,
    V: np.ndarray,
    agg_df_all: pd.DataFrame,
) -> None:
    """
    Single wide CSV: per cluster, min/max Cohen's d, marginal ``var_pop`` at the
    argmin/argmax features, within-cluster ``V`` at those features, and min/max ``V`` across
    all features in the cluster.

    ``D`` uses Cohen's d with weighted population means and local within-cluster variance.
    ``var_pop[j]`` is the **global marginal** population variance for feature ``j`` (weighted
    law of total variance); it does not vary by cluster for a fixed feature.
    """
    n_by_c: Dict[str, Any] = {}
    df_idx = _agg_df_to_cluster_id_frame(agg_df_all)
    if "n_members" in df_idx.columns:
        for _, row in df_idx.iterrows():
            n_by_c[str(row["cluster_id"])] = row["n_members"]

    summary_rows: List[Dict[str, Any]] = []
    for i, cid in enumerate(cluster_ids):
        drow = D[i, :]
        vrow = V[i, :]
        jdmin = _nan_argmin_idx(drow)
        jdmax = _nan_argmax_idx(drow)
        jvmin = _nan_argmin_idx(vrow)
        jvmax = _nan_argmax_idx(vrow)

        row: Dict[str, Any] = {"cluster_id": cid}
        row["n_members"] = n_by_c.get(cid, "")

        if not np.any(np.isfinite(drow)):
            row.update(
                {
                    "cohens_d_min": float("nan"),
                    "cohens_d_max": float("nan"),
                    "feature_at_cohens_d_min": "",
                    "feature_at_cohens_d_max": "",
                    "var_pop_marginal_at_cohens_d_min_feature": float("nan"),
                    "var_pop_marginal_at_cohens_d_max_feature": float("nan"),
                    "var_within_at_cohens_d_min_feature": float("nan"),
                    "var_within_at_cohens_d_max_feature": float("nan"),
                    "var_within_min_across_features": float("nan"),
                    "var_within_max_across_features": float("nan"),
                    "feature_at_var_within_min": "",
                    "feature_at_var_within_max": "",
                }
            )
            summary_rows.append(row)
            continue

        dmin = float(np.nanmin(drow))
        dmax = float(np.nanmax(drow))

        def _feat(j: int) -> str:
            return bases[j] if j >= 0 else ""

        def _vp(j: int) -> float:
            return float(var_pop[j]) if j >= 0 else float("nan")

        def _vw(j: int) -> float:
            return float(vrow[j]) if j >= 0 else float("nan")

        vw_min = float(np.nanmin(vrow)) if jvmin >= 0 else float("nan")
        vw_max = float(np.nanmax(vrow)) if jvmax >= 0 else float("nan")

        row.update(
            {
                "cohens_d_min": dmin,
                "cohens_d_max": dmax,
                "feature_at_cohens_d_min": _feat(jdmin),
                "feature_at_cohens_d_max": _feat(jdmax),
                "var_pop_marginal_at_cohens_d_min_feature": _vp(jdmin),
                "var_pop_marginal_at_cohens_d_max_feature": _vp(jdmax),
                "var_within_at_cohens_d_min_feature": _vw(jdmin),
                "var_within_at_cohens_d_max_feature": _vw(jdmax),
                "var_within_min_across_features": vw_min,
                "var_within_max_across_features": vw_max,
                "feature_at_var_within_min": _feat(jvmin),
                "feature_at_var_within_max": _feat(jvmax),
            }
        )
        summary_rows.append(row)

    out_sum = os.path.join(parent, "per_cluster_cohens_d_and_variance_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_sum, index=False)
    print(
        f"Wrote per-cluster Cohen's d / var summary (1 row per cluster) -> {out_sum}",
        flush=True,
    )


def cohens_d_matrix_from_agg_variance_df(agg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cohen's d effect size matrix from per-cluster variance dataframe.
    rows = features, columns = cluster_id.
    d_{c,f} = (M_{c,f} - mu_pop_f) / sqrt(V_{c,f} + eps)
    """
    return cohens_d_from_agg_df(agg_df)


def _resolve_cluster_col(wide_df: pd.DataFrame, cid: Any) -> str:
    """Find cluster column label (handles int/float/str mismatches)."""
    for c in wide_df.columns:
        if str(c) == str(cid):
            return str(c)
    raise KeyError(f"No cluster column for cluster id {cid!r}. Columns: {list(wide_df.columns)[:10]}")


def _cka_embedding_vs_tsfresh_columns(
    sub: pd.DataFrame,
    repr_cols: List[str],
    features: List[str],
    var_eps: float,
) -> pd.DataFrame:
    """One row per feature: linear CKA(embedding, feature) on rows in ``sub``."""
    X = sub[repr_cols].to_numpy(dtype=float)
    rows: List[Tuple[Any, float]] = []
    for feat in features:
        if feat not in sub.columns:
            continue
        y = sub[feat].to_numpy(dtype=float).reshape(-1, 1)
        _, _, is_const = safe_feature_stats(y.ravel(), eps=var_eps)
        if is_const:
            cka = 0.0
        else:
            try:
                cka = linear_cka(X, y)
            except Exception:
                cka = np.nan
        rows.append((feat, float(cka) if np.isfinite(cka) else np.nan))
    return pd.DataFrame(rows, columns=["feature", "cka_to_embedding"])


def _per_cluster_cohens_d_cka_one(
    cid: Any,
    merged: pd.DataFrame,
    label_col: str,
    cohens_d_wide_df: pd.DataFrame,
    selected_features: List[str],
    parent: str,
    repr_cols: List[str],
    cka_compute_cols: List[str],
    var_eps: float,
    selected_set: Any,
    empty_two: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """
    Write Cohen's d + CKA CSVs for one cluster; return long_rows_selected.
    Safe to run concurrently when each cluster uses a distinct subdirectory.
    """
    c_col = _resolve_cluster_col(cohens_d_wide_df, cid)
    sub = merged[merged[label_col].astype(str) == str(cid)]
    subdir = os.path.join(parent, safe_cluster_dir_name(cid))
    os.makedirs(subdir, exist_ok=True)

    cohens_rows = []
    for feat in selected_features:
        fk = str(feat)
        if fk not in cohens_d_wide_df.index:
            continue
        d = float(cohens_d_wide_df.loc[fk, c_col])
        if not np.isfinite(d):
            continue
        cohens_rows.append({"feature": feat, "cohens_d": d, "abs_cohens_d": abs(d)})
    df_cohens = pd.DataFrame(cohens_rows)
    if not df_cohens.empty:
        df_cohens = df_cohens.sort_values("abs_cohens_d", ascending=False)
    df_cohens.to_csv(os.path.join(subdir, "selected_features_cohens_d.csv"), index=False)
    cohens_map = {str(r["feature"]): float(r["cohens_d"]) for _, r in df_cohens.iterrows()}

    long_rows: List[Dict[str, Any]] = []

    if len(sub) < 2:
        empty_two.to_csv(os.path.join(subdir, "selected_features_cka.csv"), index=False)
        return long_rows

    full_cka = _cka_embedding_vs_tsfresh_columns(sub, repr_cols, cka_compute_cols, var_eps)

    all_cka = full_cka[full_cka["feature"].isin(selected_set)].copy()
    all_cka["cohens_d"] = all_cka["feature"].astype(str).map(cohens_map)
    all_cka["abs_cohens_d"] = all_cka["cohens_d"].abs()

    cka_max = float(np.nanmax(all_cka["cka_to_embedding"].to_numpy(dtype=float))) if len(all_cka) else 0.0
    cka_max = cka_max if np.isfinite(cka_max) and cka_max > 0 else 1.0
    coh_max = float(np.nanmax(all_cka["abs_cohens_d"].to_numpy(dtype=float))) if len(all_cka) else 0.0
    coh_max = coh_max if np.isfinite(coh_max) and coh_max > 0 else 1.0

    all_cka["norm_cka"] = all_cka["cka_to_embedding"] / cka_max
    all_cka["norm_cohen"] = all_cka["abs_cohens_d"] / coh_max
    all_cka["feature_importance"] = all_cka["norm_cka"] * all_cka["norm_cohen"]
    all_cka = all_cka.sort_values(
        ["feature_importance", "norm_cka", "norm_cohen", "feature"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    all_cka["rank"] = np.arange(1, len(all_cka) + 1)
    all_cka.to_csv(os.path.join(subdir, "selected_features_cka.csv"), index=False)

    cid_str = str(cid)
    for _, row in all_cka.iterrows():
        long_rows.append(
            {
                "cluster_id": cid_str,
                "feature": row["feature"],
                "cka_to_embedding": row["cka_to_embedding"],
                "rank": int(row["rank"]),
            }
        )

    return long_rows


def _blas_thread_env_sandbox():
    """Force single-threaded BLAS in this process while the context is active (nested-safe)."""
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    prev: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}

    class _Ctx:
        def __enter__(self):
            for k in keys:
                os.environ[k] = "1"
            return self

        def __exit__(self, *exc):
            for k in keys:
                v = prev[k]
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return False

    return _Ctx()


def write_per_cluster_cohens_d_and_cka(
    merged: pd.DataFrame,
    *,
    label_col: str,
    repr_cols: List[str],
    selected_features: List[str],
    agg_df: pd.DataFrame,
    parent_dir: str,
    var_eps: float = 1e-12,
    max_cluster_workers: Optional[int] = None,
    agg_df_population: Optional[pd.DataFrame] = None,
) -> None:
    """
    Under parent_dir/cluster_<id>/ write:
      - selected_features_cohens_d.csv (all Fisher-selected features ranked by |Cohen's d|)
      - selected_features_cka.csv (all Fisher-selected features with CKA for that cluster)
    Also writes parent_dir/all_clusters_selected_features_cka_long.csv.

    If ``agg_df_population`` is set (all TSFresh columns, same layout as ``agg_df``), writes
    population CSVs: ``selected_features_highest_var_pop.csv``,
    ``per_cluster_min_cohens_d_all_features.csv``, ``per_cluster_max_cohens_d_all_features.csv``,
    ``per_cluster_min_var_within_all_features.csv``,
    ``per_cluster_cohens_d_and_variance_summary.csv`` (see module docstring).

    If max_cluster_workers >= 2, clusters are processed with a thread pool (shared ``merged``
    frame; one BLAS thread per worker via env for the pool lifetime). None or 1 keeps the
    original sequential loop.
    """
    missing_repr = [c for c in repr_cols if c not in merged.columns]
    if missing_repr:
        raise KeyError(f"merged missing repr columns: {missing_repr[:5]}")

    cohens_d_wide_df = cohens_d_matrix_from_agg_variance_df(agg_df)
    cohens_d_wide_df.columns = cohens_d_wide_df.columns.map(str)
    cohens_d_wide_df.index = cohens_d_wide_df.index.map(str)

    parent = os.path.expanduser(parent_dir)
    os.makedirs(parent, exist_ok=True)

    if agg_df_population is not None:
        write_population_all_features_csvs(agg_df_population, parent)

    cka_compute_cols = list(selected_features)
    cluster_ids = list(agg_df.index)
    long_rows: List[Dict[str, Any]] = []
    empty_two = pd.DataFrame(columns=["feature", "cka_to_embedding"])
    selected_set = set(selected_features)

    n_workers = int(max_cluster_workers) if max_cluster_workers is not None else 1
    use_threads = n_workers >= 2 and len(cluster_ids) >= 2
    if use_threads:
        w = min(n_workers, len(cluster_ids))
        with _blas_thread_env_sandbox():
            with ThreadPoolExecutor(max_workers=w) as ex:
                futures = {
                    ex.submit(
                        _per_cluster_cohens_d_cka_one,
                        cid,
                        merged,
                        label_col,
                        cohens_d_wide_df,
                        selected_features,
                        parent,
                        repr_cols,
                        cka_compute_cols,
                        var_eps,
                        selected_set,
                        empty_two,
                    ): cid
                    for cid in cluster_ids
                }
                by_cid: Dict[Any, List[Dict[str, Any]]] = {}
                for fut in as_completed(futures):
                    cid = futures[fut]
                    by_cid[cid] = fut.result()
        for cid in cluster_ids:
            long_rows.extend(by_cid[cid])
    else:
        for cid in cluster_ids:
            lr = _per_cluster_cohens_d_cka_one(
                cid,
                merged,
                label_col,
                cohens_d_wide_df,
                selected_features,
                parent,
                repr_cols,
                cka_compute_cols,
                var_eps,
                selected_set,
                empty_two,
            )
            long_rows.extend(lr)

    long_path = os.path.join(parent, "all_clusters_selected_features_cka_long.csv")
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    print(f"Wrote all-clusters CKA long table -> {long_path}", flush=True)

    print(
        f"Wrote per-cluster Cohen's d / CKA artifacts under {parent}"
        + (f"; cluster_workers={n_workers}" if use_threads else "")
        + ".",
        flush=True,
    )
