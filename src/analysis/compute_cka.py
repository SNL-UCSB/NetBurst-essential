#!/usr/bin/env python3
"""
Compute linear CKA similarity between two feature sets (representations vs TSFresh)
aligned by a concatenated key of `ip|source_file`.

Usage:
    python compute_cka.py \
        --reps_csv /path/to/representations.csv \
        --tsfresh_csv /path/to/tsfresh.csv \
        --out /path/to/output.csv \
        [--group_by_source]

Notes:
- The script expects both CSVs to have columns: 'ip' and 'source_file' as keys.
- In reps CSV, only columns with prefix 'repr_dim' are used as features.
- In TSFresh CSV, all numeric columns (excluding keys) are used as features.
- Features are column-centered (mean=0) prior to linear CKA.
- Linear CKA is computed as: ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F).

Output:
- If --group_by_source is provided, saves per-source_file CKA scores to CSV.
- Otherwise, saves a single row with the overall CKA score.
- All outputs include n_merged (rows after inner join on key) and n_samples (rows used in CKA
  after dropping rows with NaNs in either feature matrix).

"""
import argparse
import sys
from typing import List, Optional

import numpy as np
import pandas as pd


def _validate_keys(df: pd.DataFrame, id_cols: List[str]) -> None:
    missing = [c for c in id_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing key columns in CSV: {missing}")


def _apply_ip_suffix_filter(df: pd.DataFrame, suffix: Optional[str]) -> pd.DataFrame:
    """Keep rows whose ``ip`` column string ends with ``suffix`` (e.g. ``/32``)."""
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
    # repr_prefix='*' selects all numeric columns except ID columns.
    if repr_prefix == "*":
        excluded = set(id_cols or ["ip", "source_file"])
        numeric_cols = reps_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in excluded]
    else:
        prefixed = [c for c in reps_df.columns if c.startswith(repr_prefix)]
        feature_cols = [c for c in prefixed if pd.api.types.is_numeric_dtype(reps_df[c])]
    if not feature_cols:
        raise ValueError(
            f"No numeric representation feature columns found for repr_prefix '{repr_prefix}'."
        )
    return reps_df[feature_cols]


def _select_tsfresh_features(ts_df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    # Use all numeric columns except id columns
    numeric_cols = ts_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in id_cols]
    if not feature_cols:
        raise ValueError("No numeric TSFresh feature columns found after excluding keys.")
    return ts_df[feature_cols]


def _make_key(df: pd.DataFrame, id_cols: List[str], key_name: str = "key") -> pd.DataFrame:
    # Concatenate id columns to a single string key: ip|source_file
    if len(id_cols) != 2 or id_cols != ["ip", "source_file"]:
        # We still support arbitrary id cols, but default is ['ip', 'source_file']
        sep = "|"
        df[key_name] = df[id_cols[0]].astype(str)
        for c in id_cols[1:]:
            df[key_name] = df[key_name] + sep + df[c].astype(str)
        return df
    df[key_name] = df["ip"].astype(str) + "|" + df["source_file"].astype(str)
    return df


def _center_columns(X: np.ndarray) -> np.ndarray:
    # Center each column to have mean 0 across samples
    col_means = np.nanmean(X, axis=0, keepdims=True)
    return X - col_means


def linear_cka(X: np.ndarray, Y: np.ndarray) -> tuple[float, int]:
    """Compute linear CKA between two centered feature matrices.

    X: shape (n_samples, d_x)
    Y: shape (n_samples, d_y)
    Returns: (CKA in [0, 1], number of rows used after NaN filtering)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")

    # Center columns
    Xc = _center_columns(X)
    Yc = _center_columns(Y)

    # If any rows have NaNs, drop those rows synchronously
    valid_mask = (~np.isnan(Xc).any(axis=1)) & (~np.isnan(Yc).any(axis=1))
    Xc = Xc[valid_mask]
    Yc = Yc[valid_mask]

    n_samples = int(Xc.shape[0])
    if n_samples == 0:
        raise ValueError("No valid rows remaining after NaN filtering.")

    # Compute cross-covariance Frobenius norm
    XtY = Xc.T @ Yc  # (d_x x d_y)
    num = np.linalg.norm(XtY, ord="fro") ** 2

    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    den = np.linalg.norm(XtX, ord="fro") * np.linalg.norm(YtY, ord="fro")

    if den == 0:
        raise ValueError("Denominator is zero; features may be constant.")

    return float(num / den), n_samples


def compute_cka_for_csvs(
    reps_csv: str,
    tsfresh_csv: str,
    id_cols: List[str] = ["ip", "source_file"],
    repr_prefix: str = "repr_dim",
    group_by_source: bool = False,
    filter_ip_suffix: Optional[str] = None,
) -> pd.DataFrame:
    reps_df = pd.read_csv(reps_csv)
    ts_df = pd.read_csv(tsfresh_csv)

    _validate_keys(reps_df, id_cols)
    _validate_keys(ts_df, id_cols)
    reps_df = _apply_ip_suffix_filter(reps_df, filter_ip_suffix)
    ts_df = _apply_ip_suffix_filter(ts_df, filter_ip_suffix)

    # Create concatenated key in both DataFrames
    reps_df = _make_key(reps_df, id_cols, key_name="key")
    ts_df = _make_key(ts_df, id_cols, key_name="key")

    # Deduplicate by key to avoid Cartesian product on merge
    reps_df = reps_df.drop_duplicates(subset=["key"])\
                     .copy()
    ts_df = ts_df.drop_duplicates(subset=["key"])\
                 .copy()

    # Select features
    reps_feats = _select_repr_features(reps_df, repr_prefix=repr_prefix, id_cols=id_cols)
    ts_feats = _select_tsfresh_features(ts_df, id_cols + ["key"])  # exclude id + key

    # Keep stable column namespaces through merge to avoid collisions.
    reps_feats = reps_feats.rename(columns={c: f"reps__{c}" for c in reps_feats.columns})
    ts_feats = ts_feats.rename(columns={c: f"ts__{c}" for c in ts_feats.columns})

    # Add key back to feature frames for merge
    reps_feats = pd.concat([reps_df[["key", "source_file"]], reps_feats], axis=1)
    ts_feats = pd.concat([ts_df[["key", "source_file"]], ts_feats], axis=1)

    # Inner merge on key to align rows present in both
    merged = reps_feats.merge(ts_feats, on=["key"], how="inner", suffixes=("_reps", "_ts"))

    if merged.shape[0] == 0:
        raise ValueError("No overlapping keys between reps and TSFresh after merge.")

    # Build matrices X and Y
    reps_cols = [c for c in merged.columns if c.startswith("reps__")]
    ts_cols = [c for c in merged.columns if c.startswith("ts__")]

    X = merged[reps_cols].to_numpy()
    Y = merged[ts_cols].to_numpy()

    n_merged_total = int(merged.shape[0])

    if not group_by_source:
        cka_val, n_samples = linear_cka(X, Y)
        result = pd.DataFrame(
            {"cka": [cka_val], "n_merged": [n_merged_total], "n_samples": [n_samples]}
        )
        return result

    # Choose a source_file column (they should be identical after merge). Prefer reps side.
    if "source_file_reps" in merged.columns:
        sf_col = "source_file_reps"
    elif "source_file" in merged.columns:
        sf_col = "source_file"
    else:
        # Fallback to ts side
        sf_col = "source_file_ts" if "source_file_ts" in merged.columns else None

    if sf_col is None:
        raise ValueError("source_file column not found after merge for grouping.")

    rows = []
    for sf, grp in merged.groupby(sf_col):
        X_sf = grp[reps_cols].to_numpy()
        Y_sf = grp[ts_cols].to_numpy()
        n_merged_sf = int(X_sf.shape[0])
        if X_sf.shape[0] < 2:
            rows.append(
                {
                    "source_file": sf,
                    "cka": np.nan,
                    "n_merged": n_merged_sf,
                    "n_samples": n_merged_sf,
                }
            )
            continue
        try:
            cka_val, n_samples = linear_cka(X_sf, Y_sf)
        except Exception:
            cka_val = np.nan
            n_samples = np.nan
        rows.append(
            {
                "source_file": sf,
                "cka": cka_val,
                "n_merged": n_merged_sf,
                "n_samples": n_samples,
            }
        )

    return pd.DataFrame(rows).sort_values(["source_file"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Compute linear CKA between reps and TSFresh features.")
    parser.add_argument("--reps_csv", required=False, help="Path to representations CSV")
    parser.add_argument("--tsfresh_csv", required=False, help="Path to TSFresh CSV")
    parser.add_argument("--out", required=False, help="Output CSV path to save results")
    parser.add_argument("--group_by_source", action="store_true", help="Compute CKA per source_file")
    parser.add_argument("--id_cols", nargs="*", default=["ip", "source_file"], help="Key columns for alignment")
    parser.add_argument(
        "--repr_prefix",
        default="repr_dim",
        help="Prefix for representation feature columns; use '*' to select all columns except --id_cols.",
    )
    parser.add_argument(
        "--filter_ip_suffix",
        default=None,
        help="Keep only rows whose ip column ends with this suffix (e.g. /32); applied to both CSVs before merge",
    )
    parser.add_argument("--demo", action="store_true", help="Run a tiny demo with synthetic data")
    args = parser.parse_args()

    if args.demo:
        # Synthetic demo
        rng = np.random.default_rng(0)
        n = 100
        # Create correlated synthetic features
        X = rng.normal(size=(n, 32))
        Y = X @ rng.normal(size=(32, 48)) + 0.1 * rng.normal(size=(n, 48))
        demo_cka, demo_n = linear_cka(X, Y)
        print(f"Demo CKA: {demo_cka:.6f} (n_samples={demo_n})")
        return

    if not args.reps_csv or not args.tsfresh_csv:
        print("Error: --reps_csv and --tsfresh_csv are required unless --demo is used.", file=sys.stderr)
        sys.exit(2)

    try:
        df = compute_cka_for_csvs(
            reps_csv=args.reps_csv,
            tsfresh_csv=args.tsfresh_csv,
            id_cols=args.id_cols,
            repr_prefix=args.repr_prefix,
            group_by_source=args.group_by_source,
            filter_ip_suffix=args.filter_ip_suffix,
        )
    except Exception as e:
        print(f"Failed to compute CKA: {e}", file=sys.stderr)
        sys.exit(1)

    # Save or print
    if args.out:
        # Choose CSV output
        out_path = args.out
        if args.group_by_source:
            df.to_csv(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")
    else:
        # Print to stdout
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
