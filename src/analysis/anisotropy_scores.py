#!/usr/bin/env python3
"""
Compute anisotropy scores for model representations (no TSFresh required).

This script expects a CSV containing representation vectors with columns:
- Keys: typically 'ip' and 'source_file' (configurable via --id_cols)
- Feature columns: prefixed with 'repr_dim' (configurable via --repr_prefix)

Anisotropy metrics provided:
- mean_pairwise_cosine: Average cosine similarity across all pairs of unit-normalized rows
  Efficiently computed as (||sum u_i||^2 - n) / (n * (n - 1)). Higher = more anisotropic.
- top_ev_ratio: Explained variance ratio of the first principal component (PCA) on centered features
  Higher = more anisotropic (variance concentrated in one direction).
- effective_rank: exp(Entropy of normalized singular value spectrum); lower indicates anisotropy.
- effective_rank_norm: effective_rank normalized by min(n_samples, n_dims) in [0, 1].
- mcc_top_dim_abs: Mean absolute cosine contribution of the most dominant coordinate (and its index).

Usage:
    python anisotropy_scores.py \
        --reps_csv /path/to/representations.csv \
        --out /path/to/output.csv \
        [--group_by_source] [--repr_prefix repr_dim]

Notes:
- Features are column-centered prior to PCA-based metrics.
- Rows with any NaNs in features are dropped.
- If --group_by_source is provided, computes metrics per source_file.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _validate_keys(df: pd.DataFrame, id_cols: List[str]) -> None:
    missing = [c for c in id_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing key columns in CSV: {missing}")


def _apply_ip_suffix_filter(df: pd.DataFrame, suffix: Optional[str]) -> pd.DataFrame:
    """Keep rows where the ``ip`` column string ends with ``suffix`` (e.g. ``/32``)."""
    if suffix is None:
        return df
    s = str(suffix).strip()
    if not s:
        return df
    if "ip" not in df.columns:
        raise ValueError("filter_ip_suffix requires an 'ip' column in the CSV.")
    mask = df["ip"].astype(str).str.endswith(s)
    return df.loc[mask].copy()


def _select_repr_features(
    reps_df: pd.DataFrame, repr_prefix: str = "repr_dim", id_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    if repr_prefix == "*":
        excluded = set(id_cols or ["ip", "source_file"])
        feature_cols = [c for c in reps_df.columns if c not in excluded]
    else:
        feature_cols = [c for c in reps_df.columns if c.startswith(repr_prefix)]
    if not feature_cols:
        raise ValueError(
            f"No representation feature columns found for repr_prefix='{repr_prefix}'."
        )
    return reps_df[feature_cols]


def _center_columns(X: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(X, axis=0, keepdims=True)
    return X - col_means


def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero: keep rows with norm > 0
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return X / safe_norms


def mean_pairwise_cosine(X: np.ndarray) -> float:
    """Average cosine similarity over all pairs of unit-normalized rows.

    Efficient formula: Let u_i be unit vectors of rows. Then
        mean_{i!=j}(u_i · u_j) = (||sum_i u_i||^2 - n) / (n * (n - 1))
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    n = X.shape[0]
    if n < 2:
        return np.nan

    # Drop rows with NaNs
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    n = X.shape[0]
    if n < 2:
        return np.nan

    U = _normalize_rows(X)
    S = np.sum(U, axis=0)  # shape (d,)
    num = float(np.dot(S, S) - n)
    den = float(n * (n - 1))
    return num / den


def _svd_spectrum(Xc: np.ndarray) -> np.ndarray:
    """Return singular values of centered matrix Xc.

    Uses full SVD with `compute_uv=False` to get singular values efficiently.
    """
    # If any NaNs remain, drop rows
    mask = ~np.isnan(Xc).any(axis=1)
    Xc = Xc[mask]
    if Xc.shape[0] == 0:
        raise ValueError("No valid rows after NaN filtering for SVD.")
    # SVD on centered data
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    return s  # singular values


def top_explained_variance_ratio(X: np.ndarray) -> float:
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] < 2:
        return np.nan
    Xc = _center_columns(X)
    s = _svd_spectrum(Xc)
    # Eigenvalues of covariance ~ s^2 (up to scaling).
    lam = s ** 2
    total = float(np.sum(lam))
    if total == 0.0:
        return np.nan
    return float(lam[0] / total)


def effective_rank(X: np.ndarray) -> Tuple[float, float]:
    """Compute effective rank and its normalized version in [0, 1].

    effective_rank = exp(H), where H is the Shannon entropy of the
    normalized spectrum p_i = lam_i / sum(lam).
    Returns (effective_rank, effective_rank_norm).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] < 2:
        return (np.nan, np.nan)
    Xc = _center_columns(X)
    s = _svd_spectrum(Xc)
    lam = s ** 2
    total = float(np.sum(lam))
    if total == 0.0:
        return (np.nan, np.nan)
    p = lam / total
    # Avoid log(0): clip small values
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    H = -float(np.sum(p * np.log(p)))
    erank = float(np.exp(H))
    k = float(min(X.shape[0], X.shape[1]))
    erank_norm = erank / k
    return (erank, erank_norm)


def mcc_top_dimension(X: np.ndarray, use_abs: bool = True) -> Tuple[float, int]:
    """Mean Cosine Contribution per coordinate; returns (max_value, top_index).
    Rows are unit-normalized first. For each coordinate k:
      - use_abs=True:  contribution_k = mean(|u_{ik}|)
      - use_abs=False: contribution_k = mean(u_{ik}^2)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] == 0:
        return (np.nan, -1)

    # Drop rows with NaNs
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    if X.shape[0] == 0:
        return (np.nan, -1)

    U = _normalize_rows(X)
    if use_abs:
        contrib = np.mean(np.abs(U), axis=0)
    else:
        contrib = np.mean(U ** 2, axis=0)

    top_idx = int(np.argmax(contrib))
    return float(contrib[top_idx]), top_idx


def _pc1_loadings_and_scores(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return (pc1_loadings, pc1_scores_per_row, top_ev_ratio).
    Drops rows with any NaNs; features are column-centered.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    # Drop rows with any NaNs in features
    row_mask = ~np.isnan(X).any(axis=1)
    X = X[row_mask]
    if X.shape[0] < 2:
        return np.full((X.shape[1],), np.nan), np.full((X.shape[0],), np.nan), np.nan
    Xc = _center_columns(X)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Vt[0]               # loadings in feature space (d,)
    scores = Xc @ pc1         # sample scores along PC1 (n,)
    lam = s ** 2
    total = float(np.sum(lam))
    ev_ratio = float(lam[0] / total) if total > 0.0 else np.nan
    return pc1, scores, ev_ratio


def _corr_with_pc1_scores(X: np.ndarray, features: pd.DataFrame) -> Dict[str, Tuple[float, int]]:
    """
    Compute absolute Pearson correlation between PC1 sample scores and each feature column.
    Returns dict: feature_name -> (abs_corr, n_used)
    """
    # Drop rows with NaNs in X and align features
    row_mask = ~np.isnan(X).any(axis=1)
    X = X[row_mask]
    features = features.loc[features.index[row_mask]]
    if X.shape[0] < 2:
        return {}
    _, scores, _ = _pc1_loadings_and_scores(X)
    out: Dict[str, Tuple[float, int]] = {}
    for col in features.columns:
        f = features[col].to_numpy()
        mask = np.isfinite(scores) & np.isfinite(f)
        n = int(np.count_nonzero(mask))
        if n < 3:
            out[col] = (np.nan, n)
            continue
        c = float(np.corrcoef(scores[mask], f[mask])[0, 1])
        out[col] = (abs(c), n)
    return out


def _top_k_corrs(corr_map: Dict[str, Tuple[float, int]], k: int) -> List[Tuple[str, float, int]]:
    """
    From feature->(abs_corr, n_used), return top-k list of (feature, abs_corr, n_used)
    sorted by abs_corr desc, then n_used desc, then feature name asc.
    """
    items: List[Tuple[str, float, int]] = []
    for feat, (c, n) in corr_map.items():
        if c is None or not np.isfinite(c):
            continue
        items.append((feat, float(c), int(n)))
    items.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return items[: max(0, int(k))]


def _filter_corrs(corr_map: Dict[str, Tuple[float, int]], min_corr: float) -> List[Tuple[str, float, int]]:
    """
    From feature->(abs_corr, n_used), return all (feature, abs_corr, n_used)
    with abs_corr >= min_corr, sorted by abs_corr desc, then n_used desc, then feature name asc.
    """
    items: List[Tuple[str, float, int]] = []
    thr = float(min_corr)
    for feat, (c, n) in corr_map.items():
        if c is None or not np.isfinite(c):
            continue
        if float(c) >= thr:
            items.append((feat, float(c), int(n)))
    items.sort(key=lambda t: (-t[1], -t[2], t[0]))
    return items


def compute_anisotropy_metrics(X: np.ndarray) -> Dict[str, float]:
    mpc = mean_pairwise_cosine(X)
    ter = top_explained_variance_ratio(X)
    erank, erank_norm = effective_rank(X)
    mcc_abs, mcc_idx = mcc_top_dimension(X, use_abs=True)
    return {
        "mean_pairwise_cosine": mpc,
        "top_ev_ratio": ter,
        "effective_rank": erank,
        "effective_rank_norm": erank_norm,
        "mcc_top_dim_abs": mcc_abs,
        "mcc_top_dim_index": mcc_idx,
    }


def compute_anisotropy_for_csv(
    reps_csv: str,
    id_cols: List[str] = ["ip", "source_file"],
    repr_prefix: str = "repr_dim",
    group_by_source: bool = False,
    filter_ip_suffix: Optional[str] = None,
) -> pd.DataFrame:
    reps_df = pd.read_csv(reps_csv)
    _validate_keys(reps_df, id_cols)
    reps_df = _apply_ip_suffix_filter(reps_df, filter_ip_suffix)

    # Select features
    reps_feats = _select_repr_features(reps_df, repr_prefix=repr_prefix, id_cols=id_cols)

    # Drop rows with all-NaN in features
    valid_row_mask = ~np.isnan(reps_feats).all(axis=1)
    reps_df = reps_df[valid_row_mask].copy()
    reps_feats = reps_feats[valid_row_mask].copy()

    if not group_by_source:
        X = reps_feats.to_numpy()
        metrics = compute_anisotropy_metrics(X)
        result = {
            **metrics,
            "n_samples": int(X.shape[0]),
            "n_dims": int(X.shape[1]),
        }
        return pd.DataFrame([result])

    # Group by source_file
    if "source_file" not in reps_df.columns:
        raise ValueError("source_file column required for --group_by_source.")

    rows = []
    feature_cols = reps_feats.columns.tolist()
    df = pd.concat([reps_df[["source_file"]], reps_feats], axis=1)
    for sf, grp in df.groupby("source_file"):
        X_sf = grp[feature_cols].to_numpy()
        if X_sf.shape[0] < 2:
            rows.append(
                {
                    "source_file": sf,
                    "mean_pairwise_cosine": np.nan,
                    "top_ev_ratio": np.nan,
                    "effective_rank": np.nan,
                    "effective_rank_norm": np.nan,
                    "mcc_top_dim_abs": np.nan,
                    "mcc_top_dim_index": -1,
                    "n_samples": int(X_sf.shape[0]),
                    "n_dims": int(X_sf.shape[1]) if X_sf.shape[0] > 0 else np.nan,
                }
            )
            continue
        metrics = compute_anisotropy_metrics(X_sf)
        rows.append({
            "source_file": sf,
            **metrics,
            "n_samples": int(X_sf.shape[0]),
            "n_dims": int(X_sf.shape[1]),
        })
    return pd.DataFrame(rows).sort_values(["source_file"]).reset_index(drop=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute anisotropy metrics for representation CSV.")
    parser.add_argument("--reps_csv", required=False, help="Path to representations CSV")
    parser.add_argument("--out", required=False, help="Output CSV path to save results")
    parser.add_argument("--group_by_source", action="store_true", help="Compute metrics per source_file")
    parser.add_argument("--id_cols", nargs="*", default=["ip", "source_file"], help="Key columns present in reps CSV")
    parser.add_argument(
        "--repr_prefix",
        default="repr_dim",
        help="Prefix for representation feature columns; use '*' to select all columns except --id_cols.",
    )
    parser.add_argument(
        "--filter_ip_suffix",
        default=None,
        help="Keep only rows whose ip column ends with this suffix (e.g. /32 for host CIDRs)",
    )
    parser.add_argument("--demo", action="store_true", help="Run a tiny demo with synthetic data")
    parser.add_argument(
        "--align_pc1_to_tsfresh",
        action="store_true",
        help="Find TSFresh feature most aligned with PC1 scores via abs Pearson correlation",
    )
    parser.add_argument(
        "--tsfresh_csv",
        required=False,
        help="Path to TSFresh features CSV (if not in reps_csv)",
    )
    parser.add_argument(
        "--ts_cols",
        nargs="*",
        default=[],
        help="Explicit TSFresh feature columns to consider; if empty, uses all numeric non-repr columns",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top TSFresh features to display by absolute correlation (default: 10)",
    )
    parser.add_argument(
        "--min_corr",
        type=float,
        default=None,
        help="If set, list all TSFresh features with absolute correlation >= this threshold (e.g., 0.9)",
    )
    return parser


def run_anisotropy(args: argparse.Namespace) -> None:
    if args.demo:
        rng = np.random.default_rng(0)
        n, d = 200, 64
        # Create anisotropic data by adding a strong common direction
        base = rng.normal(size=(n, d))
        direction = rng.normal(size=(d,))
        direction /= np.linalg.norm(direction) + 1e-12
        X_aniso = base + 3.0 * rng.normal(size=(n, 1)) * direction  # strong first PC
        X_iso = rng.normal(size=(n, d))  # approximately isotropic

        m_aniso = compute_anisotropy_metrics(X_aniso)
        m_iso = compute_anisotropy_metrics(X_iso)
        print("Anisotropic sample:")
        for k, v in m_aniso.items():
            print(f"  {k}: {v:.6f}")
        print("Isotropic sample:")
        for k, v in m_iso.items():
            print(f"  {k}: {v:.6f}")
        return

    if not args.reps_csv:
        print("Error: --reps_csv is required unless --demo is used.", file=sys.stderr)
        sys.exit(2)

    # Optional: align PC1 to TSFresh features
    if args.align_pc1_to_tsfresh:
        reps_df = pd.read_csv(args.reps_csv)
        _validate_keys(reps_df, args.id_cols)
        reps_df = _apply_ip_suffix_filter(reps_df, args.filter_ip_suffix)
        reps_feats = _select_repr_features(reps_df, repr_prefix=args.repr_prefix, id_cols=args.id_cols)

        # Drop rows with all-NaN in representation features
        valid_row_mask = ~np.isnan(reps_feats).all(axis=1)
        reps_df = reps_df[valid_row_mask].copy()
        reps_feats = reps_feats[valid_row_mask].copy()

        # Build TSFresh feature dataframe
        if args.tsfresh_csv:
            ts_df = pd.read_csv(args.tsfresh_csv)
            _validate_keys(ts_df, args.id_cols)
            # Join on id_cols to align row order with reps_df
            join_df = pd.merge(reps_df[args.id_cols], ts_df, on=args.id_cols, how="left", validate="m:1")
            ts_df = join_df.drop(columns=args.id_cols)
        else:
            # Assume TSFresh features are in reps_csv alongside representation columns.
            ts_df = reps_df.drop(columns=list(reps_feats.columns), errors="ignore")

        # Select TSFresh columns
        if args.ts_cols:
            missing = [c for c in args.ts_cols if c not in ts_df.columns]
            if missing:
                print(f"TSFresh columns not found: {missing}", file=sys.stderr)
                sys.exit(1)
            ts_sel = ts_df[args.ts_cols].copy()
        else:
            # Use all numeric columns
            ts_sel = ts_df.select_dtypes(include=[np.number]).copy()
            if ts_sel.shape[1] == 0:
                print("No numeric TSFresh columns available for alignment.", file=sys.stderr)
                sys.exit(1)

        if args.group_by_source:
            if "source_file" not in reps_df.columns:
                print("source_file column required for --group_by_source.", file=sys.stderr)
                sys.exit(1)
            feature_cols = reps_feats.columns.tolist()
            df_sf = pd.concat([reps_df[["source_file"]], reps_feats, ts_sel], axis=1)
            for sf, grp in df_sf.groupby("source_file"):
                X_sf = grp[feature_cols].to_numpy()
                corr_map = _corr_with_pc1_scores(X_sf, grp[ts_sel.columns])
                if not corr_map:
                    print(f"{sf}: best_feature=NA | corr=nan | n=0")
                    continue
                # Also report EV ratio for context
                _, _, ev_ratio = _pc1_loadings_and_scores(X_sf)
                ev_str = "nan" if np.isnan(ev_ratio) else f"{ev_ratio:.6f}"
                if args.min_corr is not None:
                    filt = _filter_corrs(corr_map, args.min_corr)
                    print(f"{sf}: top_ev_ratio={ev_str} | features_abs_corr_ge_{args.min_corr} (feature|abs_corr|n): count={len(filt)}")
                    for rank, (feat, corr, n_used) in enumerate(filt, start=1):
                        print(f"  {rank}. {feat} | {corr:.6f} | {n_used}")
                else:
                    top_list = _top_k_corrs(corr_map, args.top_k)
                    print(f"{sf}: top_ev_ratio={ev_str} | top_{len(top_list)}_tsfresh_features (feature|abs_corr|n):")
                    for rank, (feat, corr, n_used) in enumerate(top_list, start=1):
                        print(f"  {rank}. {feat} | {corr:.6f} | {n_used}")
        else:
            X = reps_feats.to_numpy()
            corr_map = _corr_with_pc1_scores(X, ts_sel)
            if not corr_map:
                print("best_tsfresh_feature=NA | corr=nan | n=0")
            else:
                _, _, ev_ratio = _pc1_loadings_and_scores(X)
                ev_str = "nan" if np.isnan(ev_ratio) else f"{ev_ratio:.6f}"
                if args.min_corr is not None:
                    filt = _filter_corrs(corr_map, args.min_corr)
                    print(f"top_ev_ratio={ev_str} | features_abs_corr_ge_{args.min_corr} (feature|abs_corr|n): count={len(filt)}")
                    for rank, (feat, corr, n_used) in enumerate(filt, start=1):
                        print(f"  {rank}. {feat} | {corr:.6f} | {n_used}")
                else:
                    top_list = _top_k_corrs(corr_map, args.top_k)
                    print(f"top_ev_ratio={ev_str} | top_{len(top_list)}_tsfresh_features (feature|abs_corr|n):")
                    for rank, (feat, corr, n_used) in enumerate(top_list, start=1):
                        print(f"  {rank}. {feat} | {corr:.6f} | {n_used}")

    try:
        df = compute_anisotropy_for_csv(
            reps_csv=args.reps_csv,
            id_cols=args.id_cols,
            repr_prefix=args.repr_prefix,
            group_by_source=args.group_by_source,
            filter_ip_suffix=args.filter_ip_suffix,
        )
    except Exception as e:
        print(f"Failed to compute anisotropy: {e}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved results to {args.out}")
    else:
        print(df.to_string(index=False))


def run_from_config_dict(cfg: dict) -> None:
    """Run from a configuration dict (keys = argparse ``dest`` names)."""
    ap = build_arg_parser()
    ap.set_defaults(**cfg)
    args = ap.parse_args([])
    run_anisotropy(args)


def run_from_config(config_path: str) -> None:
    """Load flat JSON (argument names as keys, matching argparse ``dest`` names) and run."""
    path = os.path.expanduser(str(config_path))
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    run_from_config_dict(cfg)


def main():
    args = build_arg_parser().parse_args()
    run_anisotropy(args)


if __name__ == "__main__":
    main()
