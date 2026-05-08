#!/usr/bin/env python3
"""
Collect top-1 retrieved neighbor IDs and cluster IDs across many nprobe values.

Outputs:
  - Per-query wide CSV with neighbor/cluster columns for each nprobe
  - Summary CSV with change counts vs a reference nprobe
  - Train mapping CSV (milvus_id -> cluster_id + point metadata)
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
        description="Collect top-1 retrieved cluster IDs per eval query across nprobe values."
    )
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--eval-vecs", required=True)
    p.add_argument("--faiss-index", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--nprobes",
        type=int,
        nargs="+",
        required=True,
        help="List of nprobe values to evaluate (e.g. 1 2 4 8 16).",
    )
    p.add_argument(
        "--nprobe-ref",
        type=int,
        default=None,
        help="Reference nprobe for change summaries (default: first value in --nprobes).",
    )
    p.add_argument(
        "--use-cluster-filter",
        action="store_true",
        default=False,
        help=(
            "Use per-query cluster filter (IDSelectorBatch by eval cluster_id). "
            "Default is global IVF search."
        ),
    )
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def _sanitize_nprobes(nprobes: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for v in nprobes:
        vv = int(v)
        if vv < 1:
            continue
        if vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out


def _cluster_of_neighbor(neighbor_milvus_id: int, train_cluster_by_milvus: dict[int, int]) -> int:
    if neighbor_milvus_id < 0:
        return -1
    return int(train_cluster_by_milvus.get(int(neighbor_milvus_id), -1))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    nprobes = _sanitize_nprobes(args.nprobes)
    if not nprobes:
        raise ValueError("No valid nprobe values after sanitization.")
    nprobe_ref = int(args.nprobe_ref) if args.nprobe_ref is not None else int(nprobes[0])
    if nprobe_ref not in nprobes:
        raise ValueError(f"nprobe_ref={nprobe_ref} must be in --nprobes list {nprobes}")

    manifest = pd.read_csv(args.split_manifest)
    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    eval_rows = manifest[manifest["split"] == "eval"].reset_index(drop=True)

    eval_vecs = np.load(args.eval_vecs)
    if len(eval_vecs) != len(eval_rows):
        raise ValueError(
            f"Eval vector count ({len(eval_vecs)}) != eval rows ({len(eval_rows)})"
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
            f"[nprobe-grid] {args.model}: cluster filter ON ({len(cluster_map)} clusters).",
            flush=True,
        )
    else:
        print(f"[nprobe-grid] {args.model}: cluster filter OFF (global IVF).", flush=True)

    map_csv = os.path.join(args.output_dir, f"{args.model}_train_milvus_id_to_cluster_map.csv")
    train_rows[
        ["milvus_id", "cluster_id", "point_id", "ip", "source_file"]
    ].to_csv(map_csv, index=False)

    n_eval = len(eval_rows)
    pe = max(1, int(args.progress_every))
    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []

    for i, eval_row in enumerate(eval_rows.itertuples(index=False)):
        eval_cluster_id = int(eval_row.cluster_id)
        vec = np.asarray(eval_vecs[i], dtype=np.float32)

        if args.use_cluster_filter:
            assert cluster_map is not None
            allowed_ids = cluster_map.get(eval_cluster_id, np.array([], dtype=np.int64))
        else:
            allowed_ids = None

        row: dict[str, object] = {
            "model": args.model,
            "test_point_id": eval_row.point_id,
            "eval_cluster_id": eval_cluster_id,
        }

        for npb in nprobes:
            idxs, _scores, q_ms = search_one(
                index=index,
                query_vec=vec,
                allowed_ids=allowed_ids,
                topk=1,
                nprobe=npb,
            )
            nbr = int(idxs[0]) if len(idxs) else -1
            clus = _cluster_of_neighbor(nbr, train_cluster_by_milvus)

            row[f"neighbor_milvus_id_nprobe_{npb}"] = nbr
            row[f"retrieved_cluster_nprobe_{npb}"] = clus
            row[f"query_time_ms_nprobe_{npb}"] = round(float(q_ms), 4)

        rows.append(row)

        n_done = i + 1
        if n_done % pe == 0 or n_done == n_eval:
            elapsed_s = time.perf_counter() - t0
            print(
                f"  [{args.model}] progress {n_done}/{n_eval} (elapsed={elapsed_s:.1f}s)",
                flush=True,
            )

    per_query_df = pd.DataFrame(rows)
    per_query_csv = os.path.join(
        args.output_dir, f"{args.model}_nprobe_grid_cluster_assignments.csv"
    )
    per_query_df.to_csv(per_query_csv, index=False)

    summary_rows = []
    ref_cluster_col = f"retrieved_cluster_nprobe_{nprobe_ref}"
    ref_neighbor_col = f"neighbor_milvus_id_nprobe_{nprobe_ref}"
    n_total = int(len(per_query_df))

    for npb in nprobes:
        cur_cluster_col = f"retrieved_cluster_nprobe_{npb}"
        cur_neighbor_col = f"neighbor_milvus_id_nprobe_{npb}"
        cluster_changed = (per_query_df[cur_cluster_col] != per_query_df[ref_cluster_col]).sum()
        neighbor_changed = (per_query_df[cur_neighbor_col] != per_query_df[ref_neighbor_col]).sum()
        summary_rows.append(
            {
                "model": args.model,
                "nprobe_ref": nprobe_ref,
                "nprobe": int(npb),
                "total_eval_queries": n_total,
                "cluster_changed_queries_vs_ref": int(cluster_changed),
                "cluster_changed_pct_vs_ref": round(
                    100.0 * float(cluster_changed) / n_total if n_total else 0.0, 4
                ),
                "neighbor_changed_queries_vs_ref": int(neighbor_changed),
                "neighbor_changed_pct_vs_ref": round(
                    100.0 * float(neighbor_changed) / n_total if n_total else 0.0, 4
                ),
                "cluster_filter_mode": "on" if args.use_cluster_filter else "off",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(
        args.output_dir, f"{args.model}_nprobe_grid_change_summary_vs_{nprobe_ref}.csv"
    )
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== nprobe grid summary ===")
    print(summary_df.to_string(index=False))
    print(f"Saved per-query assignments: {per_query_csv}")
    print(f"Saved change summary:        {summary_csv}")
    print(f"Saved train id map:          {map_csv}")


if __name__ == "__main__":
    main()
