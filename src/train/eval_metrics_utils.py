"""
Shared evaluation metrics for Chronos, DeepAR, and N-BEATS training scripts.
MAPE and Wasserstein distance are computed the same way across all of them.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance as wasserstein_distance_scipy


def mae_nonzero_gt(
    forecast: np.ndarray,
    actual: np.ndarray,
    thr: float = 0.0,
) -> float:
    """
    Mean Absolute Error using only indices where actual > thr.
    Returns np.nan if no such indices.
    """
    if forecast is None or actual is None:
        return np.nan
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    n = min(len(f), len(a))
    if n == 0:
        return np.nan
    f, a = f[:n], a[:n]
    mask = a > thr
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs(f[mask] - a[mask])))


def mape_nonzero_gt(
    forecast: np.ndarray,
    actual: np.ndarray,
    nz_thresh: float = 0.0,
    scale: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    MAPE over indices where actual > nz_thresh (threshold in original units, e.g. bytes).
    All values are scaled by scale for the MAPE computation (e.g. scale=1e6 for bytes -> MB),
    so MAPE is mean(|f/scale - a/scale| / (a/scale + eps)) in scaled space.
    Returns np.nan if no indices with actual > nz_thresh.
    """
    if forecast is None or actual is None:
        return np.nan
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    n = min(len(f), len(a))
    if n == 0:
        return np.nan
    f, a = f[:n], a[:n]
    mask = a > nz_thresh
    if not np.any(mask):
        return np.nan
    # Scale all values (e.g. bytes -> MB); MAPE in scaled units
    f_s = f[mask] / scale
    a_s = a[mask] / scale
    denom = a_s + eps
    return float(np.mean(np.abs(f_s - a_s) / denom))


def wasserstein_distance_tails(
    gt_tail: np.ndarray,
    pred_tail: np.ndarray,
    zero_thresh: float = 0.0,
    eps: float = 1e-2,
) -> float:
    """
    Wasserstein distance between the two tails, treating time indices as positions
    and (normalized) values as probability weights.

    Each time step i is a position (np.arange(n)); the normalized value at that
    step is its probability weight.  Values <= zero_thresh are zeroed using the
    GT mask before normalization so small/zero bins don't contribute mass.
    eps is added before L1-normalization to avoid division-by-zero.

    The result is measured in units of time steps: a WD of k means the predicted
    mass distribution is shifted by ~k time steps relative to ground truth,
    preserving timing (imin) information.

    Returns np.nan if either array is empty after trimming.
    """
    if gt_tail is None or pred_tail is None or len(gt_tail) == 0:
        return np.nan
    gt = np.asarray(gt_tail, dtype=float)
    pr = np.asarray(pred_tail, dtype=float) if pred_tail is not None and len(pred_tail) > 0 else np.array([], dtype=float)
    n = len(gt)
    if n == 0:
        return np.nan
    gt = gt.copy()
    if len(pr) < n:
        pr = np.concatenate([pr, np.zeros(n - len(pr))])
    pr = pr[:n].copy()
    mask_small = gt <= zero_thresh
    gt = np.where(mask_small, 0.0, gt) + eps
    # Clip pred to non-negative before using as probability weights; models like
    # DeepAR/N-BEATS can produce negative forecasts which are invalid as weights.
    pr = np.where(mask_small, 0.0, np.clip(pr, 0.0, None)) + eps
    gt_weights = gt / np.sum(gt)
    pr_weights = pr / np.sum(pr)
    positions = np.arange(n, dtype=float)
    return float(wasserstein_distance_scipy(positions, positions, gt_weights, pr_weights))
