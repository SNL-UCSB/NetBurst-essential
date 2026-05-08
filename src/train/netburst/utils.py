"""Utilities shared across NetBurst training and inference."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch


def fano_factor_tensor(vals: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-12) -> float:
    """Fano factor = variance/mean over valid elements. Returns nan if mean <= 0 or no valid."""
    if mask is not None:
        flat = vals[mask].float().flatten()
    else:
        flat = vals.float().flatten()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 2:
        return float("nan")
    mu = flat.mean().item()
    if not np.isfinite(mu) or mu <= eps:
        return float("nan")
    var = flat.var().item()
    if not np.isfinite(var):
        return float("nan")
    return float(var / mu)


def fano_factor_numpy(x: Union[List[float], np.ndarray], eps: float = 1e-12) -> float:
    """Fano factor for lists/arrays (e.g. inference forecasts). Returns nan if mean <= 0 or len < 2."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mu = float(np.mean(arr))
    if not np.isfinite(mu) or mu <= eps:
        return float("nan")
    var = float(np.var(arr))
    if not np.isfinite(var):
        return float("nan")
    return float(var / mu)


def strip_prefix_if_present(state_dict, prefixes):
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if k.startswith(p):
                new_k = k[len(p) :]
                break
        out[new_k] = v
    return out
