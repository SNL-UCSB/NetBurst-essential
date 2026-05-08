"""Build cluster_id -> train milvus_id sets from a split manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_cluster_to_milvus_ids(split_manifest_path: str) -> dict[int, np.ndarray]:
    """
    For each cluster_id, return int64 array of milvus_id values for train rows.

    Used with FAISS IDSelectorBatch to mirror Milvus scalar filter cluster_id == label.
    """
    manifest = pd.read_csv(split_manifest_path)
    train = manifest[manifest["split"] == "train"]
    out: dict[int, np.ndarray] = {}
    for cid, grp in train.groupby("cluster_id"):
        out[int(cid)] = grp["milvus_id"].to_numpy(dtype=np.int64)
    return out
