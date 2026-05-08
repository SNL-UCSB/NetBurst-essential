"""
Reproducibility helpers for NSDI-style Milvus IVF benchmarks (Milvus Lite).

- Package versions (pymilvus, milvus-lite)
- benchmark_config.json emission
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore


def package_version(dist_name: str) -> str:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def collect_runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pymilvus": package_version("pymilvus"),
        "milvus_lite": package_version("milvus-lite"),
        "faiss_cpu": package_version("faiss-cpu"),
    }


def write_benchmark_config(
    output_dir: str,
    payload: dict[str, Any],
    filename: str = "benchmark_config.json",
) -> str:
    """Write a frozen JSON snapshot for paper appendices / reproducibility."""
    os.makedirs(output_dir, exist_ok=True)
    out = dict(payload)
    out.setdefault("written_at_utc", datetime.now(timezone.utc).isoformat())
    out.setdefault("runtime_versions", collect_runtime_versions())
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[benchmark_repro] Wrote {path}")
    return path


def pipeline_faiss_single_model_payload(
    *,
    model: str,
    cluster_csv: str,
    output_dir: str,
    faiss_index_path: str,
    parquet_path: str,
    split_manifest: Optional[str],
    seed: int,
    n_eval: int,
    nlist: int,
    nprobe: int,
    topk: int,
    eval_fraction: Optional[float] = None,
    query_mode: str = "filtered_cluster_ivf",
) -> dict[str, Any]:
    """Standard fields for run_pipeline_faiss (FAISS IVF-Flat, inner product on unit vectors)."""
    return {
        "benchmark": "netvecdb_faiss_ivf_flat",
        "engine": "faiss_cpu",
        "model": model,
        "index": {
            "index_type": "IVF_FLAT",
            "metric_type": "METRIC_INNER_PRODUCT",
            "nlist": nlist,
            "nprobe": nprobe,
            "topk": topk,
        },
        "query_mode": query_mode,
        "notes": (
            "L2-normalized train vectors; IP = cosine similarity. "
            "With cluster filter: IDSelectorBatch on train milvus_ids in same cluster_id."
        ),
        "inputs": {
            "cluster_csv": os.path.abspath(cluster_csv),
            "parquet_path": os.path.abspath(parquet_path),
            "faiss_index": os.path.abspath(faiss_index_path),
            "split_manifest": os.path.abspath(split_manifest) if split_manifest else None,
        },
        "split": {
            "seed": seed,
            "n_eval": n_eval,
            **({"eval_fraction": eval_fraction} if eval_fraction is not None else {}),
        },
    }
