#!/usr/bin/env python3
"""
Compare top-1 retrieved cluster changes between two FAISS nprobe values.

Typical use case:
  - Query each eval vector once with nprobe=1.
  - Query each eval vector once with nprobe=10.
  - Count how many eval queries change their retrieved neighbor cluster.

Outputs:
  {model}_nprobe_{a}_vs_{b}_cluster_changes.csv
  {model}_nprobe_{a}_vs_{b}_cluster_changes_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import faiss
import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from faiss_based.cluster_allowed_ids import build_cluster_to_milvus_ids
from faiss_based.search_ivf import search_one


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare top-1 FAISS retrieval cluster assignments between two nprobe values."
        )
    )
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--eval-vecs", required=True)
    p.add_argument("--faiss-index", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--nprobe-a", type=int, default=1, help="First nprobe value (default: 1)")
    p.add_argument(
        "--nprobe-b", type=int, default=10, help="Second nprobe value (default: 10)"
    )
    p.add_argument(
        "--use-cluster-filter",
        action="store_true",
        default=False,
        help=(
            "Use per-query cluster filter (IDSelectorBatch by eval cluster_id). "
            "Default is global IVF search (no cluster filter)."
        ),
    )
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def _cluster_of_neighbor(neighbor_milvus_id: int, train_cluster_by_milvus: dict[int, int]) -> int:
    if neighbor_milvus_id < 0:
        return -1
    return int(train_cluster_by_milvus.get(int(neighbor_milvus_id), -1))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = pd.read_csv(args.split_manifest)
    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    eval_rows = manifest[manifest["split"] == "eval"].reset_index(drop=True)

    eval_vecs = np.load(args.eval_vecs)
    if len(eval_vecs) != len(eval_rows):
        raise ValueError(
            f"Eval vector count ({len(eval_vecs)}) != manifest eval rows ({len(eval_rows)})"
        )

    index = faiss.read_index(args.faiss_index)
    if not isinstance(index, faiss.IndexIVFFlat):
        print(f"  Warning: expected IndexIVFFlat, got {type(index)}")

    train_cluster_by_milvus = {
        int(r.milvus_id): int(r.cluster_id) for r in train_rows.itertuples(index=False)
    }

    cluster_map: dict[int, np.ndarray] | None = None
    if args.use_cluster_filter:
        cluster_map = build_cluster_to_milvus_ids(args.split_manifest)
        print(
            f"[cluster-change] Using cluster filter with {len(cluster_map)} cluster keys.",
            flush=True,
        )
    else:
        print("[cluster-change] Using global IVF search (no cluster filter).", flush=True)

    rows: list[dict[str, object]] = []
    n_eval = len(eval_rows)
    pe = max(1, int(args.progress_every))
    t0 = time.perf_counter()

    for i, eval_row in enumerate(eval_rows.itertuples(index=False)):
        eval_cluster_id = int(eval_row.cluster_id)
        vec = np.asarray(eval_vecs[i], dtype=np.float32)

        if args.use_cluster_filter:
            assert cluster_map is not None
            allowed_ids = cluster_map.get(eval_cluster_id, np.array([], dtype=np.int64))
        else:
            allowed_ids = None

        idxs_a, _scores_a, q_ms_a = search_one(
            index=index,
            query_vec=vec,
            allowed_ids=allowed_ids,
            topk=1,
            nprobe=args.nprobe_a,
        )
        idxs_b, _scores_b, q_ms_b = search_one(
            index=index,
            query_vec=vec,
            allowed_ids=allowed_ids,
            topk=1,
            nprobe=args.nprobe_b,
        )

        nbr_a = int(idxs_a[0]) if len(idxs_a) else -1
        nbr_b = int(idxs_b[0]) if len(idxs_b) else -1

        cluster_a = _cluster_of_neighbor(nbr_a, train_cluster_by_milvus)
        cluster_b = _cluster_of_neighbor(nbr_b, train_cluster_by_milvus)

        rows.append(
            {
                "model": args.model,
                "test_point_id": eval_row.point_id,
                "eval_cluster_id": eval_cluster_id,
                "nprobe_a": int(args.nprobe_a),
                "nprobe_b": int(args.nprobe_b),
                "neighbor_milvus_id_nprobe_a": nbr_a,
                "neighbor_milvus_id_nprobe_b": nbr_b,
                "retrieved_cluster_nprobe_a": cluster_a,
                "retrieved_cluster_nprobe_b": cluster_b,
                "cluster_changed": bool(cluster_a != cluster_b),
                "neighbor_changed": bool(nbr_a != nbr_b),
                "query_time_ms_nprobe_a": round(float(q_ms_a), 4),
                "query_time_ms_nprobe_b": round(float(q_ms_b), 4),
            }
        )

        n_done = i + 1
        if n_done % pe == 0 or n_done == n_eval:
            elapsed_s = time.perf_counter() - t0
            print(
                f"  [{args.model}] progress {n_done}/{n_eval} "
                f"(elapsed={elapsed_s:.1f}s)",
                flush=True,
            )

    out_per_query = os.path.join(
        args.output_dir,
        f"{args.model}_nprobe_{args.nprobe_a}_vs_{args.nprobe_b}_cluster_changes.csv",
    )
    out_summary = os.path.join(
        args.output_dir,
        f"{args.model}_nprobe_{args.nprobe_a}_vs_{args.nprobe_b}_cluster_changes_summary.csv",
    )

    per_query_df = pd.DataFrame(rows)
    per_query_df.to_csv(out_per_query, index=False)

    n_total = int(len(per_query_df))
    n_cluster_changed = int(per_query_df["cluster_changed"].sum()) if n_total else 0
    n_neighbor_changed = int(per_query_df["neighbor_changed"].sum()) if n_total else 0
    pct_cluster_changed = (100.0 * n_cluster_changed / n_total) if n_total else 0.0
    pct_neighbor_changed = (100.0 * n_neighbor_changed / n_total) if n_total else 0.0

    summary_df = pd.DataFrame(
        [
            {
                "model": args.model,
                "nprobe_a": int(args.nprobe_a),
                "nprobe_b": int(args.nprobe_b),
                "total_eval_queries": n_total,
                "cluster_changed_queries": n_cluster_changed,
                "cluster_changed_pct": round(pct_cluster_changed, 4),
                "neighbor_changed_queries": n_neighbor_changed,
                "neighbor_changed_pct": round(pct_neighbor_changed, 4),
                "cluster_filter_mode": "on" if args.use_cluster_filter else "off",
            }
        ]
    )
    summary_df.to_csv(out_summary, index=False)

    print("\n=== Cluster change summary ===")
    print(summary_df.to_string(index=False))
    print(f"Saved per-query: {out_per_query}")
    print(f"Saved summary:   {out_summary}")


if __name__ == "__main__":
    main()
