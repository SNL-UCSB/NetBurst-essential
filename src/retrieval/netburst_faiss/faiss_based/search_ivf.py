"""Single-query IVF search with optional ID subset (cluster filter)."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

import faiss


def search_one(
    index: faiss.Index,
    query_vec: np.ndarray,
    allowed_ids: Optional[np.ndarray],
    topk: int,
    nprobe: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run one query against IndexIVFFlat (inner product on L2-normalized vectors).

    Parameters
    ----------
    query_vec
        Shape (D,) float32, L2-normalized (cosine / IP equivalence).
    allowed_ids
        If None, search the full index (--no-cluster-filter).
        If length-0 array, return -1 indices and NaN scores (no candidates).
        Otherwise IDSelectorBatch(milvus_id list for this cluster).

    Returns
    -------
    indices, scores, elapsed_ms
        FAISS indices (milvus_id); scores are inner product (higher = closer for unit vectors).
    """
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    params = faiss.SearchParametersIVF()
    params.nprobe = int(nprobe)

    if allowed_ids is not None:
        if allowed_ids.size == 0:
            return (
                np.full(topk, -1, dtype=np.int64),
                np.full(topk, np.nan, dtype=np.float32),
                0.0,
            )
        params.sel = faiss.IDSelectorBatch(allowed_ids)

    t0 = time.perf_counter()
    distances, indices = index.search(q, int(topk), params=params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return indices[0], distances[0], elapsed_ms
