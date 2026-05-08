#!/usr/bin/env python3
"""
Query eval points using FAISS IVF-Flat (same outputs as query_eval.py).

Reuses Wasserstein/DTW/parquet helpers from ../query_eval.py.

Environment: module load conda && module load pytorch/2.6.0, then pip install -r faiss_based/requirements-faiss.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import faiss
import query_eval as qe

from faiss_based.cluster_allowed_ids import build_cluster_to_milvus_ids
from faiss_based.search_ivf import search_one


PARQUET_DEFAULT = qe.PARQUET_DEFAULT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FAISS IVF query → Wasserstein + DTW (same as query_eval.py)"
    )
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--eval-vecs", required=True)
    p.add_argument("--faiss-index", required=True, help="Path to .faiss index from ingest_faiss.py")
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--parquet-path", default=PARQUET_DEFAULT)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--debug-plot", action="store_true", default=False)
    p.add_argument("--debug-plot-max", type=int, default=100)
    p.add_argument("--debug-plot-dir", default=None)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument(
        "--no-cluster-filter",
        action="store_true",
        default=False,
        help="Search full index without IDSelector (same semantics as Milvus --no-cluster-filter).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = pd.read_csv(args.split_manifest)
    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    eval_rows = manifest[manifest["split"] == "eval"].reset_index(drop=True)
    print(f"[query_eval_faiss] Train: {len(train_rows):,}  |  Eval: {len(eval_rows):,}")

    id_to_info = {
        int(r.milvus_id): (r.point_id, r.ip, r.source_file)
        for r in train_rows.itertuples(index=False)
    }

    eval_vecs = np.load(args.eval_vecs)
    if len(eval_vecs) != len(eval_rows):
        raise ValueError(
            f"Eval vector count ({len(eval_vecs)}) != manifest eval rows ({len(eval_rows)})"
        )

    index = faiss.read_index(args.faiss_index)
    if not isinstance(index, faiss.IndexIVFFlat):
        print(f"  Warning: expected IndexIVFFlat, got {type(index)}")

    cluster_map: dict[int, np.ndarray] | None = None
    if not args.no_cluster_filter:
        cluster_map = build_cluster_to_milvus_ids(args.split_manifest)
        print(f"  Cluster filter: {len(cluster_map)} cluster keys (train-only milvus ids)")

    ts_index = qe.load_parquet_index(args.parquet_path)

    distance_rows = []
    timing_rows = []
    ts_missing_eval = 0
    ts_missing_nbr = 0
    n_padded_slots_total = 0
    debug_plot_count = 0
    debug_plot_dir = args.debug_plot_dir or os.path.join(args.output_dir, f"{args.model}_debug_plots")

    n_eval_total = len(eval_rows)
    pe = max(1, int(args.progress_every))
    loop_t0 = time.perf_counter()
    cumulative_search_ms = 0.0
    print(
        f"[query_eval_faiss] Starting eval loop: {n_eval_total:,} points, progress every {pe}",
        flush=True,
    )

    for i, eval_row in enumerate(eval_rows.itertuples(index=False)):
        test_point_id = eval_row.point_id
        cluster_id = int(eval_row.cluster_id)
        vec = np.asarray(eval_vecs[i], dtype=np.float32)

        if args.no_cluster_filter:
            allowed_ids = None
        else:
            assert cluster_map is not None
            allowed_ids = cluster_map.get(cluster_id, np.array([], dtype=np.int64))

        idxs, _scores, query_time_ms = search_one(
            index,
            vec,
            allowed_ids,
            args.topk,
            args.nprobe,
        )
        cumulative_search_ms += query_time_ms

        timing_rows.append(
            {
                "model": args.model,
                "test_point_id": test_point_id,
                "query_time_ms": round(query_time_ms, 4),
            }
        )

        neighbor_count = int(np.sum(idxs >= 0))
        n_padded_slots_total += int(np.sum(idxs < 0))

        eval_key = (eval_row.ip, str(eval_row.source_file))
        eval_inbound = ts_index.get(eval_key)
        if eval_inbound is None:
            ts_missing_eval += 1

        plot_neighbors: list[tuple[str, np.ndarray]] = []

        for rank in range(1, args.topk + 1):
            nbr_milvus_id = int(idxs[rank - 1]) if (rank - 1) < len(idxs) else -1
            if nbr_milvus_id < 0:
                wd_n = wd_u = dtw_n = dtw_u = float("nan")
                neighbor_id = "__insufficient_neighbors__"
            else:
                info = id_to_info.get(nbr_milvus_id)
                if info is None:
                    wd_n = wd_u = dtw_n = dtw_u = float("nan")
                    neighbor_id = str(nbr_milvus_id)
                else:
                    neighbor_id, nbr_ip, nbr_sf = info
                    nbr_inbound = ts_index.get((nbr_ip, str(nbr_sf)))
                    if eval_inbound is None:
                        wd_n = wd_u = dtw_n = dtw_u = float("nan")
                    elif nbr_inbound is None:
                        ts_missing_nbr += 1
                        wd_n = wd_u = dtw_n = dtw_u = float("nan")
                    else:
                        wd_n = qe.compute_wasserstein(eval_inbound, nbr_inbound, mode="normalized")
                        wd_u = qe.compute_wasserstein(eval_inbound, nbr_inbound, mode="unnormalized")
                        dtw_n = qe.compute_dtw(eval_inbound, nbr_inbound, mode="normalized")
                        dtw_u = qe.compute_dtw(eval_inbound, nbr_inbound, mode="unnormalized")
                        if args.debug_plot and debug_plot_count < args.debug_plot_max:
                            plot_neighbors.append(
                                (neighbor_id, np.asarray(nbr_inbound, dtype=np.float64))
                            )

            distance_rows.append(
                {
                    "model": args.model,
                    "test_point_id": test_point_id,
                    "neighbor_id": neighbor_id,
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "wasserstein_distance": wd_n,
                    "wasserstein_distance_normalized": wd_n,
                    "wasserstein_distance_unnormalized": wd_u,
                    "dtw_distance_normalized": dtw_n,
                    "dtw_distance_unnormalized": dtw_u,
                    "neighbor_count": neighbor_count,
                }
            )

        if (
            args.debug_plot
            and debug_plot_count < args.debug_plot_max
            and plot_neighbors
            and eval_inbound is not None
        ):
            qe.save_debug_plot(
                out_dir=debug_plot_dir,
                model=args.model,
                test_point_id=test_point_id,
                cluster_id=cluster_id,
                eval_ts_raw=np.asarray(eval_inbound, dtype=np.float64),
                neighbors=plot_neighbors,
            )
            qe.save_debug_timeseries_csv(
                out_dir=debug_plot_dir,
                test_point_id=test_point_id,
                cluster_id=cluster_id,
                eval_ts_raw=np.asarray(eval_inbound, dtype=np.float64),
                neighbors=plot_neighbors,
            )
            debug_plot_count += 1

        n_done = i + 1
        if n_done % pe == 0 or n_done == n_eval_total:
            wall_s = time.perf_counter() - loop_t0
            avg_ann = cumulative_search_ms / n_done
            print(
                f"  [{args.model}] progress {n_done}/{n_eval_total} | "
                f"wall_elapsed={wall_s:.1f}s | ann_sum_ms={cumulative_search_ms:.1f} | "
                f"avg_ann_ms={avg_ann:.4f} | last_ann_ms={query_time_ms:.4f}",
                flush=True,
            )

    wall_total_s = time.perf_counter() - loop_t0
    print(
        f"[query_eval_faiss] Eval loop done: {n_eval_total:,} points in {wall_total_s:.1f}s wall, "
        f"ANN search total {cumulative_search_ms:.1f} ms "
        f"(avg {cumulative_search_ms / n_eval_total:.4f} ms/query)",
        flush=True,
    )

    dist_path = os.path.join(args.output_dir, f"{args.model}_distances.csv")
    time_path = os.path.join(args.output_dir, f"{args.model}_query_times.csv")
    pd.DataFrame(distance_rows).to_csv(dist_path, index=False)
    pd.DataFrame(timing_rows).to_csv(time_path, index=False)

    n_finite_n = sum(
        1 for r in distance_rows if np.isfinite(r["wasserstein_distance_normalized"])
    )
    n_finite_u = sum(
        1 for r in distance_rows if np.isfinite(r["wasserstein_distance_unnormalized"])
    )
    n_finite_dtw_n = sum(1 for r in distance_rows if np.isfinite(r["dtw_distance_normalized"]))
    n_finite_dtw_u = sum(1 for r in distance_rows if np.isfinite(r["dtw_distance_unnormalized"]))
    print(f"\n=== [{args.model}] Sanity check (FAISS) ===")
    print(f"  Eval points queried:   {len(timing_rows):,}")
    print(
        f"  Distance rows total:   {len(distance_rows):,}  "
        f"(finite WD: normalized={n_finite_n}, unnormalized={n_finite_u}; "
        f"finite DTW: normalized={n_finite_dtw_n}, unnormalized={n_finite_dtw_u})"
    )
    print(f"  Eval points missing inbound ts:     {ts_missing_eval}")
    print(f"  Neighbor entries missing inbound ts: {ts_missing_nbr}")
    print(f"  Padded neighbor slots (cluster had < topk train): {n_padded_slots_total:,}")
    if args.no_cluster_filter:
        print("  Cluster filter: DISABLED (global IVF search)")
    else:
        print("  Cluster filter: FAISS IDSelectorBatch(train ids in same cluster_id)")
    print(f"  Saved distances:   {dist_path}")
    print(f"  Saved query times: {time_path}")
    if args.debug_plot:
        print(f"  Debug plots saved: {debug_plot_count} in {debug_plot_dir}")
    print("[query_eval_faiss] Done.\n")


if __name__ == "__main__":
    main()
