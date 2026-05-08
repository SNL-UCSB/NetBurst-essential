#!/usr/bin/env python3
"""
Shared linear CKA and helpers for embedding vs TSFresh analyses.
Used by cka_embedding_per_feature.py and per-cluster CKA in cluster_analysis.py.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def center_columns(X: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(X, axis=0, keepdims=True)
    return X - col_means


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two centered feature matrices. X: (n, d_x), Y: (n, d_y)."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    Xc = center_columns(X)
    Yc = center_columns(Y)
    valid_mask = (~np.isnan(Xc).any(axis=1)) & (~np.isnan(Yc).any(axis=1))
    Xc = Xc[valid_mask]
    Yc = Yc[valid_mask]
    if Xc.shape[0] == 0:
        raise ValueError("No valid rows after NaN filtering.")
    XtY = Xc.T @ Yc
    num = np.linalg.norm(XtY, ord="fro") ** 2
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    den = np.linalg.norm(XtX, ord="fro") * np.linalg.norm(YtY, ord="fro")
    if den == 0:
        return 0.0
    return float(num / den)


def safe_feature_stats(y: np.ndarray, eps: float = 1e-12) -> Tuple[float, float, bool]:
    """Variance, frac_zero, is_constant for a single feature vector."""
    y1 = y.astype(float, copy=False)
    mask = ~np.isnan(y1)
    if not np.any(mask):
        return float("nan"), float("nan"), True
    v = float(np.nanvar(y1, ddof=0))
    nz = y1[mask]
    frac_zero = float(np.mean(nz == 0.0)) if nz.size else float("nan")
    is_constant = (not np.isfinite(v)) or (v <= eps)
    return v, frac_zero, is_constant


def cka_embedding_vs_each_tsfresh_column(
    X: np.ndarray,
    Y: np.ndarray,
    tsfresh_names: List[str],
    *,
    var_eps: float = 1e-12,
) -> pd.DataFrame:
    """
    X: (n, d_repr), Y: (n, n_tsfresh), same row order.
    Returns DataFrame with tsfresh_feature, cka_to_embedding, tsfresh_var, tsfresh_frac_zero, tsfresh_is_constant.
    """
    rows = []
    for j, name in enumerate(tsfresh_names):
        y_j = Y[:, j : j + 1]
        var_j, frac0_j, is_const = safe_feature_stats(Y[:, j], eps=var_eps)
        if is_const:
            cka = 0.0
        else:
            try:
                cka = linear_cka(X, y_j)
            except Exception:
                cka = 0.0
        rows.append((name, float(cka), var_j, frac0_j, bool(is_const)))
    return pd.DataFrame(
        rows,
        columns=["tsfresh_feature", "cka_to_embedding", "tsfresh_var", "tsfresh_frac_zero", "tsfresh_is_constant"],
    )
