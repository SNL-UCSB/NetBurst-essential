#!/usr/bin/env python3
"""Filesystem-safe names for cluster artifacts (shared by cluster_analysis and per-cluster artifact writers)."""

import hashlib
from typing import Any

import numpy as np
import pandas as pd


def safe_cluster_dir_name(label: Any) -> str:
    """Filesystem-safe subdirectory name for a cluster label."""
    if pd.isna(label):
        return "cluster_nan"
    if isinstance(label, (bool, np.bool_)):
        return f"cluster_{str(label).lower()}"
    if isinstance(label, (int, np.integer)):
        return f"cluster_{int(label)}"
    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        return f"cluster_{int(label)}"
    s = str(label)
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in s)
    if not safe.strip("_"):
        digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]
        safe = f"id_{digest}"
    return f"cluster_{safe}"[:200]
