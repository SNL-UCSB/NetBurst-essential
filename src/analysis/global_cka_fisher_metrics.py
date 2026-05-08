#!/usr/bin/env python3
"""
Compute global per-feature CKA (embedding block vs each TSFresh column).

With --clustering_csv, also computes Fisher ratio per feature across cluster labels.

Outputs under --out_dir:
- embedding_per_tsfresh_cka.csv (always)
- fisher_ratio_per_tsfresh_feature.csv, global_feature_metrics_fisher_cka.csv,
  all_features_global_cka_fisher.csv — only when --clustering_csv is set
- CDF PNGs — only with --write-cdf-pngs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from cka_core import cka_embedding_vs_each_tsfresh_column


def _validate_keys(df: pd.DataFrame, id_cols: List[str]) -> None:
    missing = [c for c in id_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing key columns in CSV: {missing}")


def _apply_ip_suffix_filter(df: pd.DataFrame, id_ip: str, suffix: Optional[str]) -> pd.DataFrame:
    if suffix is None:
        return df
    s = str(suffix).strip()
    if not s:
        return df
    if id_ip not in df.columns:
        raise ValueError(f"filter_ip_suffix requires column {id_ip!r}.")
    mask = df[id_ip].astype(str).str.endswith(s)
    return df.loc[mask].copy()


def _make_key(df: pd.DataFrame, id_cols: List[str], key_name: str = "key") -> pd.DataFrame:
    sep = "|"
    out = df.copy()
    out[key_name] = out[id_cols[0]].astype(str)
    for c in id_cols[1:]:
        out[key_name] = out[key_name] + sep + out[c].astype(str)
    return out


def _select_repr_columns(df: pd.DataFrame, repr_prefix: str) -> List[str]:
    cols = [c for c in df.columns if str(c).startswith(repr_prefix)]
    if not cols:
        raise ValueError(f"No representation columns found with prefix {repr_prefix!r}.")
    return cols


def _select_tsfresh_feature_columns(df: pd.DataFrame, id_cols: List[str]) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in numeric_cols if c not in id_cols]
    if not cols:
        raise ValueError("No numeric TSFresh feature columns found after excluding id_cols.")
    return cols


def load_merged(
    *,
    clustering_csv: Optional[str],
    reps_csv: str,
    tsfresh_csv: str,
    id_cols: List[str],
    id_ip: str,
    label_col: str,
    repr_prefix: str,
    filter_ip_suffix: Optional[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    reps_df = pd.read_csv(reps_csv, low_memory=False)
    ts_df = pd.read_csv(tsfresh_csv, low_memory=False)

    _validate_keys(reps_df, id_cols)
    _validate_keys(ts_df, id_cols)

    reps_df = _apply_ip_suffix_filter(reps_df, id_ip=id_ip, suffix=filter_ip_suffix)
    ts_df = _apply_ip_suffix_filter(ts_df, id_ip=id_ip, suffix=filter_ip_suffix)

    reps_df = _make_key(reps_df, id_cols=id_cols, key_name="key")
    ts_df = _make_key(ts_df, id_cols=id_cols, key_name="key")

    reps_df = reps_df.drop_duplicates(subset=["key"]).copy()
    ts_df = ts_df.drop_duplicates(subset=["key"]).copy()

    repr_cols = _select_repr_columns(reps_df, repr_prefix=repr_prefix)
    ts_cols = _select_tsfresh_feature_columns(ts_df, id_cols=id_cols + ["key"])

    if clustering_csv:
        clus_df = pd.read_csv(clustering_csv, low_memory=False)
        _validate_keys(clus_df, id_cols)
        if label_col not in clus_df.columns:
            raise ValueError(f"label_col {label_col!r} not found in clustering CSV.")
        clus_df = _apply_ip_suffix_filter(clus_df, id_ip=id_ip, suffix=filter_ip_suffix)
        clus_df = _make_key(clus_df, id_cols=id_cols, key_name="key")
        clus_df = clus_df.drop_duplicates(subset=["key"]).copy()
        merged = clus_df[["key", label_col]].merge(
            reps_df[["key"] + repr_cols],
            on=["key"],
            how="inner",
        )
        merged = merged.merge(
            ts_df[["key"] + id_cols + ts_cols],
            on=["key"],
            how="inner",
        )
    else:
        merged = reps_df[["key"] + repr_cols].merge(
            ts_df[["key"] + id_cols + ts_cols],
            on=["key"],
            how="inner",
        )
    if merged.shape[0] == 0:
        raise ValueError("No overlapping keys between reps and TSFresh after filtering.")
    return merged, repr_cols, ts_cols


def fisher_ratio_per_feature(
    merged: pd.DataFrame,
    *,
    feature_cols: List[str],
    label_col: str,
    eps: float = 1e-12,
) -> pd.DataFrame:
    if label_col not in merged.columns:
        raise ValueError(f"label_col {label_col!r} not found in merged data.")

    rows = []
    gb = merged.groupby(label_col, sort=False)
    for feat in feature_cols:
        means = gb[feat].mean()
        between_var = float(np.var(means.to_numpy(dtype=float), ddof=0)) if len(means) else float("nan")

        vars_per_group = gb[feat].var(ddof=0)
        vars_np = vars_per_group.to_numpy(dtype=float)
        finite_vars = vars_np[np.isfinite(vars_np)]
        within_var_mean = float(np.mean(finite_vars)) if finite_vars.size else float("nan")

        if np.isfinite(between_var) and np.isfinite(within_var_mean):
            fisher_ratio = float(between_var / (within_var_mean + eps))
        else:
            fisher_ratio = float("nan")

        rows.append(
            (
                feat,
                fisher_ratio,
                between_var,
                within_var_mean,
                int(len(means)),
                int(np.isfinite(vars_np).sum()),
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "tsfresh_feature",
            "fisher_ratio",
            "between_var",
            "within_var_mean",
            "n_groups",
            "n_groups_with_var",
        ],
    )


def _cdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs = np.sort(v)
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def plot_cdf(values: np.ndarray, out_png: Path, *, xlabel: str, title: str) -> None:
    xs, ys = _cdf_xy(values)
    if xs.size == 0:
        print(f"[global_cka_fisher_metrics] skip CDF (no finite values): {out_png}", flush=True)
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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clustering_csv",
        default=None,
        help="Optional CSV with cluster labels (label_col). If omitted, only CKA (+ CKA-sorted all-features) CSVs.",
    )
    ap.add_argument("--reps_csv", required=True)
    ap.add_argument("--tsfresh_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_cols", nargs="*", default=["ip", "source_file"])
    ap.add_argument("--id_ip", default="ip")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--repr_prefix", default="repr_dim")
    ap.add_argument("--filter_ip_suffix", default=None)

    ap.add_argument("--cka_csv_name", default="embedding_per_tsfresh_cka.csv")
    ap.add_argument("--fisher_csv_name", default="fisher_ratio_per_tsfresh_feature.csv")
    ap.add_argument("--metrics_csv_name", default="global_feature_metrics_fisher_cka.csv")
    ap.add_argument(
        "--all_features_csv_name",
        default="all_features_global_cka_fisher.csv",
        help="With clustering: CKA+Fisher join sorted by CKA. Without clustering: CKA-only sorted by CKA.",
    )
    ap.add_argument("--fisher_cdf_png", default="cdf_global_fisher_ratio.png")
    ap.add_argument("--cka_cdf_png", default="cdf_global_cka_to_embedding.png")
    ap.add_argument(
        "--write-cdf-pngs",
        action="store_true",
        help="Write Fisher and CKA CDF plots (requires matplotlib).",
    )
    return ap


def run_global_cka(args: argparse.Namespace) -> None:
    out_dir = Path(os.path.expanduser(str(args.out_dir))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    id_cols = [str(c) for c in args.id_cols]
    clustering_csv = str(args.clustering_csv).strip() if args.clustering_csv else None
    if clustering_csv == "":
        clustering_csv = None

    merged, repr_cols, ts_cols = load_merged(
        clustering_csv=clustering_csv,
        reps_csv=str(args.reps_csv),
        tsfresh_csv=str(args.tsfresh_csv),
        id_cols=id_cols,
        id_ip=str(args.id_ip),
        label_col=str(args.label_col),
        repr_prefix=str(args.repr_prefix),
        filter_ip_suffix=args.filter_ip_suffix,
    )

    X = merged[repr_cols].to_numpy(dtype=float)
    Y = merged[ts_cols].to_numpy(dtype=float)

    df_cka = cka_embedding_vs_each_tsfresh_column(X, Y, ts_cols, var_eps=1e-12)
    cka_csv = out_dir / str(args.cka_csv_name)
    df_cka.to_csv(cka_csv, index=False)
    print(f"Wrote {cka_csv} (n={len(df_cka)})", flush=True)

    if clustering_csv:
        df_fisher = fisher_ratio_per_feature(
            merged,
            feature_cols=ts_cols,
            label_col=str(args.label_col),
        )

        df_metrics = df_fisher.merge(df_cka, on="tsfresh_feature", how="outer")
        df_metrics = df_metrics.sort_values("fisher_ratio", ascending=False, kind="mergesort").reset_index(drop=True)

        df_all = df_fisher.merge(df_cka, on="tsfresh_feature", how="outer")
        df_all = df_all.sort_values("cka_to_embedding", ascending=False, kind="mergesort").reset_index(drop=True)

        fisher_csv = out_dir / str(args.fisher_csv_name)
        metrics_csv = out_dir / str(args.metrics_csv_name)
        all_features_csv = out_dir / str(args.all_features_csv_name)

        df_fisher.to_csv(fisher_csv, index=False)
        df_metrics.to_csv(metrics_csv, index=False)
        df_all.to_csv(all_features_csv, index=False)

        print(f"Wrote {fisher_csv} (n={len(df_fisher)})", flush=True)
        print(f"Wrote {metrics_csv} (n={len(df_metrics)})", flush=True)
        print(f"Wrote {all_features_csv} (n={len(df_all)}, all features, sorted by CKA desc)", flush=True)

        if args.write_cdf_pngs:
            plot_cdf(
                df_metrics["fisher_ratio"].to_numpy(dtype=float),
                out_dir / str(args.fisher_cdf_png),
                xlabel="Fisher ratio",
                title="CDF of global Fisher ratio",
            )
            plot_cdf(
                df_metrics["cka_to_embedding"].to_numpy(dtype=float),
                out_dir / str(args.cka_cdf_png),
                xlabel="Global CKA to embedding",
                title="CDF of global CKA to embedding",
            )
    else:
        df_all_cka = df_cka.sort_values("cka_to_embedding", ascending=False, kind="mergesort").reset_index(drop=True)
        all_features_csv = out_dir / str(args.all_features_csv_name)
        df_all_cka.to_csv(all_features_csv, index=False)
        print(
            f"Wrote {all_features_csv} (n={len(df_all_cka)}, CKA only, sorted by CKA desc; no clustering_csv)",
            flush=True,
        )


def run_from_config_dict(cfg: dict) -> None:
    """Run from a configuration dict (keys = argparse ``dest`` names)."""
    ap = build_arg_parser()
    ap.set_defaults(**cfg)
    args = ap.parse_args([])
    run_global_cka(args)


def run_from_config(config_path: str) -> None:
    """Load flat JSON (argument names as keys, matching argparse ``dest`` names) and run."""
    path = os.path.expanduser(str(config_path))
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    run_from_config_dict(cfg)


def main() -> None:
    args = build_arg_parser().parse_args()
    run_global_cka(args)


if __name__ == "__main__":
    main()
