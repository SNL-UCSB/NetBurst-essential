#!/usr/bin/env python3
"""
Query each eval point against its assigned cluster in Milvus and compute
1D Wasserstein distances and DTW (dynamic time warping) to the top-k neighbors
using inbound time series from the netreplica parquet. Each eval-neighbor pair
records PDF-weighted WD for ``normalized`` and empirical WD on raw values for
``unnormalized`` (no mass weights-SciPy ``wasserstein_distance`` on the two aligned
arrays). DTW uses z-normalized series for ``normalized`` and 1e6-scaled nonnegative
values for ``unnormalized``. DTW uses L1 per-step cost and the standard O(nm) DP with
O(m) memory.

Inputs (from ingest.py):
  {model}_split_manifest.csv
  {model}_eval_vecs.npy

Outputs in --output-dir:
  {model}_distances.csv    -- Exactly topk rows per eval point. neighbor_id
                               __insufficient_neighbors__ if the cluster had fewer
                               than topk train points; NaN distances when series
                               are missing or unusable.
  {model}_query_times.csv  -- model, test_point_id, query_time_ms

Usage:
  python query_eval.py \
    --split-manifest /out/{model}_split_manifest.csv \
    --eval-vecs     /out/{model}_eval_vecs.npy \
    --model         chronos2 \
    --output-dir    /out \
    [--parquet-path /path/to/parquet_dir] \
    [--db-path /path/to/milvus_lite.db] \
    [--topk 3] [--nprobe 32]
"""

import argparse
import os
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pymilvus import Collection, connections
from scipy.stats import wasserstein_distance as scipy_wd

PARQUET_DEFAULT = (
    "<external-netburst-root>/output/"
    "netreplica_converted_6Mbps_100ms_Sparse_ip"
)
DB_DEFAULT = "<data-root>/vector_db/milvus_lite.db"


def parse_args():
    p = argparse.ArgumentParser(
        description="Query eval points → Wasserstein + DTW distances"
    )
    p.add_argument("--split-manifest", required=True,
                   help="{model}_split_manifest.csv from ingest.py")
    p.add_argument("--eval-vecs", required=True,
                   help="{model}_eval_vecs.npy from ingest.py")
    p.add_argument("--model", required=True,
                   help="Model label matching the ingested collection")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--parquet-path", default=PARQUET_DEFAULT,
                   help="Parquet directory with ip, source_file, inbound columns")
    p.add_argument("--db-path", default=DB_DEFAULT)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument(
        "--debug-plot",
        action="store_true",
        help="If set, save raw inbound time-series plots and CSV dumps for eval point + neighbors",
    )
    p.add_argument(
        "--debug-plot-max",
        type=int,
        default=100,
        help="Maximum number of eval points to plot when --debug-plot is enabled",
    )
    p.add_argument(
        "--debug-plot-dir",
        default=None,
        help="Directory for debug plots (default: <output-dir>/<model>_debug_plots)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N eval points (wall time + cumulative ANN ms). Use 1 for noisy per-row logs.",
    )
    p.add_argument(
        "--no-cluster-filter",
        action="store_true",
        default=False,
        help=(
            "Disable the cluster_id scalar filter so Milvus searches across ALL train vectors. "
            "Use with IVF_FLAT (nlist=K) to let the IVF index itself act as the partitioning."
        ),
    )
    return p.parse_args()


def collection_name(model: str) -> str:
    return f"net_repr_{model.replace('-', '_')}"


def load_parquet_index(parquet_path: str) -> dict:
    """Return {(ip, source_file): inbound_ndarray} for fast lookup."""
    print(f"[query_eval] Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=["ip", "source_file", "inbound"])
    idx = {}
    for row in df.itertuples(index=False):
        idx[(row.ip, row.source_file)] = np.asarray(row.inbound, dtype=np.float64)
    print(f"  Loaded {len(idx):,} (ip, source_file) time-series entries")
    return idx


def compute_wasserstein(ts_a: np.ndarray, ts_b: np.ndarray, mode: str = "normalized") -> float:
    """
    1D Wasserstein distance between two inbound time series (trimmed to common length).

    **Normalized:** discrete distributions on time indices 0, 1, ..., n-1 with nonnegative
    weights (mass per bin): ``wasserstein_distance(x, x, u_weights=a, v_weights=b)``
    with ``x = np.arange(n)``. Matches NetBurst-style preprocessing: zero both series
    where eval <= 0, add eps, normalize weights to sum to 1.

    **Unnormalized:** empirical 1D Wasserstein on the raw aligned values only-
    ``wasserstein_distance(raw_a, raw_b)`` with SciPy's default equal weight per
    sample (no custom mass weights, no 1e6 scaling).
    """
    if ts_a is None or ts_b is None:
        return float("nan")

    raw_a = np.asarray(ts_a, dtype=np.float64)
    raw_b = np.asarray(ts_b, dtype=np.float64)
    n = min(len(raw_a), len(raw_b))
    if n == 0:
        return float("nan")

    raw_a = raw_a[:n].copy()
    raw_b = raw_b[:n].copy()
    x = np.arange(n, dtype=np.float64)

    if mode == "normalized":
        mask_small = raw_a <= 0.0
        raw_a = np.where(mask_small, 0.0, raw_a)
        raw_b = np.where(mask_small, 0.0, raw_b)
        eps = 1e-20
        pdf_a = raw_a + eps
        pdf_b = raw_b + eps
        pdf_a = pdf_a / np.sum(pdf_a)
        pdf_b = pdf_b / np.sum(pdf_b)
        return float(scipy_wd(x, x, u_weights=pdf_a, v_weights=pdf_b))

    return float(scipy_wd(raw_a, raw_b))


def _prepare_dtw_series_normalized(ts_a: np.ndarray, ts_b: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Joint mask as WD normalized; z-normalize each series for scale-invariant DTW."""
    raw_a = np.asarray(ts_a, dtype=np.float64)
    raw_b = np.asarray(ts_b, dtype=np.float64)
    n = min(len(raw_a), len(raw_b))
    if n == 0:
        return None
    raw_a = raw_a[:n].copy()
    raw_b = raw_b[:n].copy()
    mask_small = raw_a <= 0.0
    raw_a = np.where(mask_small, 0.0, raw_a)
    raw_b = np.where(mask_small, 0.0, raw_b)

    def znorm(x: np.ndarray) -> np.ndarray:
        mu = float(np.mean(x))
        std = float(np.std(x))
        if std < 1e-12:
            return np.zeros_like(x)
        return (x - mu) / std

    return znorm(raw_a), znorm(raw_b)


def _prepare_dtw_series_unnormalized(ts_a: np.ndarray, ts_b: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """DTW only: divide by 1e6, clip negatives to 0 (independent of WD unnormalized)."""
    scale = 1e6
    raw_a = np.asarray(ts_a, dtype=np.float64)
    raw_b = np.asarray(ts_b, dtype=np.float64)
    n = min(len(raw_a), len(raw_b))
    if n == 0:
        return None
    s1 = np.maximum(raw_a[:n] / scale, 0.0)
    s2 = np.maximum(raw_b[:n] / scale, 0.0)
    return s1, s2


def dtw_l1_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Classic DTW with L1 local cost; O(n*m) time, O(m) memory."""
    n, m = int(len(s1)), int(len(s2))
    if n == 0 or m == 0:
        return float("nan")
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        a = float(s1[i - 1])
        for j in range(1, m + 1):
            cost = abs(a - float(s2[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m])


def compute_dtw(ts_a: np.ndarray, ts_b: np.ndarray, mode: str = "normalized") -> float:
    if ts_a is None or ts_b is None:
        return float("nan")
    if mode == "normalized":
        prep = _prepare_dtw_series_normalized(ts_a, ts_b)
    else:
        prep = _prepare_dtw_series_unnormalized(ts_a, ts_b)
    if prep is None:
        return float("nan")
    s1, s2 = prep
    return dtw_l1_distance(s1, s2)


def safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text))


def save_debug_plot(
    out_dir: str,
    model: str,
    test_point_id: str,
    cluster_id: int,
    eval_ts_raw: np.ndarray,
    neighbors: list[tuple[str, np.ndarray]],
) -> None:
    # Lazy import so plotting dependency is only required when debug plotting is enabled.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(eval_ts_raw, label="eval", linewidth=2.5)
    for idx, (nbr_id, nbr_ts_raw) in enumerate(neighbors, start=1):
        plt.plot(nbr_ts_raw, label=f"nbr{idx}: {nbr_id}", alpha=0.8)

    plt.title(f"{model} | cluster {cluster_id} | {test_point_id}")
    plt.xlabel("time index")
    plt.ylabel("inbound")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{safe_slug(test_point_id)}__cluster_{cluster_id}.png"
    plt.savefig(os.path.join(out_dir, fname), dpi=140)
    plt.close()


def save_debug_timeseries_csv(
    out_dir: str,
    test_point_id: str,
    cluster_id: int,
    eval_ts_raw: np.ndarray,
    neighbors: list[tuple[str, np.ndarray]],
) -> None:
    max_len = max([len(eval_ts_raw)] + [len(series) for _, series in neighbors])
    data = {
        "time_index": np.arange(max_len, dtype=np.int64),
        "eval": np.full(max_len, np.nan, dtype=np.float64),
    }
    data["eval"][: len(eval_ts_raw)] = eval_ts_raw

    for idx, (nbr_id, nbr_ts_raw) in enumerate(neighbors, start=1):
        col_name = f"neighbor_{idx}__{safe_slug(nbr_id)}"
        data[col_name] = np.full(max_len, np.nan, dtype=np.float64)
        data[col_name][: len(nbr_ts_raw)] = nbr_ts_raw

    df = pd.DataFrame(data)
    fname = f"{safe_slug(test_point_id)}__cluster_{cluster_id}.csv"
    df.to_csv(os.path.join(out_dir, fname), index=False)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load split manifest -------------------------------------------------
    manifest = pd.read_csv(args.split_manifest)
    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    eval_rows = manifest[manifest["split"] == "eval"].reset_index(drop=True)
    print(f"[query_eval] Train: {len(train_rows):,}  |  Eval: {len(eval_rows):,}")

    # milvus_id -> (point_id, ip, source_file) for neighbor lookup
    id_to_info = {
        int(r.milvus_id): (r.point_id, r.ip, r.source_file)
        for r in train_rows.itertuples(index=False)
    }

    # -- Load eval vectors ---------------------------------------------------
    eval_vecs = np.load(args.eval_vecs)  # shape (n_eval, D), already L2-normalized
    if len(eval_vecs) != len(eval_rows):
        raise ValueError(
            f"Eval vector count ({len(eval_vecs)}) != manifest eval rows ({len(eval_rows)})"
        )

    # -- Load parquet time-series index --------------------------------------
    ts_index = load_parquet_index(args.parquet_path)

    # -- Connect to Milvus ---------------------------------------------------
    connections.connect(alias="default", uri=args.db_path)
    cname = collection_name(args.model)
    col = Collection(cname)
    col.load()
    print(f"  Connected to collection '{cname}'")

    search_params = {"metric_type": "COSINE", "params": {"nprobe": args.nprobe}}

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
        f"[query_eval] Starting eval loop: {n_eval_total:,} points, progress every {pe} "
        f"(wall_elapsed + cumulative ANN search ms; stuck = wall grows but count stops)",
        flush=True,
    )

    # -- Evaluate each eval point --------------------------------------------
    for i, eval_row in enumerate(eval_rows.itertuples(index=False)):
        test_point_id = eval_row.point_id
        cluster_id = int(eval_row.cluster_id)
        vec = eval_vecs[i].tolist()

        # -- Timed ANN query (only this call is timed) -----------------------
        expr = None if args.no_cluster_filter else f"cluster_id == {cluster_id}"
        t0 = time.perf_counter()
        res = col.search(
            data=[vec],
            anns_field="embedding",
            param=search_params,
            limit=args.topk,
            expr=expr,
            output_fields=["id", "cluster_id"],
        )
        query_time_ms = (time.perf_counter() - t0) * 1_000.0
        cumulative_search_ms += query_time_ms

        timing_rows.append({
            "model":         args.model,
            "test_point_id": test_point_id,
            "query_time_ms": round(query_time_ms, 4),
        })

        hits = res[0]
        neighbor_count = len(hits)
        n_padded_slots_total += max(0, args.topk - neighbor_count)

        # -- Inbound series for eval point -----------------------------------
        eval_key = (eval_row.ip, str(eval_row.source_file))
        eval_inbound = ts_index.get(eval_key)
        if eval_inbound is None:
            ts_missing_eval += 1

        plot_neighbors: list[tuple[str, np.ndarray]] = []

        # -- Exactly topk distance rows per eval (NaN when unusable or padded) -
        for rank in range(1, args.topk + 1):
            if rank <= neighbor_count:
                hit = hits[rank - 1]
                nbr_milvus_id = int(hit.id)
                info = id_to_info.get(nbr_milvus_id)
                if info is None:
                    wd_n = float("nan")
                    wd_u = float("nan")
                    dtw_n = float("nan")
                    dtw_u = float("nan")
                    neighbor_id = str(nbr_milvus_id)
                else:
                    neighbor_id, nbr_ip, nbr_sf = info
                    nbr_inbound = ts_index.get((nbr_ip, str(nbr_sf)))
                    if eval_inbound is None:
                        wd_n = float("nan")
                        wd_u = float("nan")
                        dtw_n = float("nan")
                        dtw_u = float("nan")
                    elif nbr_inbound is None:
                        ts_missing_nbr += 1
                        wd_n = float("nan")
                        wd_u = float("nan")
                        dtw_n = float("nan")
                        dtw_u = float("nan")
                    else:
                        wd_n = compute_wasserstein(eval_inbound, nbr_inbound, mode="normalized")
                        wd_u = compute_wasserstein(eval_inbound, nbr_inbound, mode="unnormalized")
                        dtw_n = compute_dtw(eval_inbound, nbr_inbound, mode="normalized")
                        dtw_u = compute_dtw(eval_inbound, nbr_inbound, mode="unnormalized")
                        if args.debug_plot and debug_plot_count < args.debug_plot_max:
                            plot_neighbors.append((neighbor_id, np.asarray(nbr_inbound, dtype=np.float64)))
            else:
                wd_n = float("nan")
                wd_u = float("nan")
                dtw_n = float("nan")
                dtw_u = float("nan")
                neighbor_id = "__insufficient_neighbors__"

            distance_rows.append({
                "model":                           args.model,
                "test_point_id":                   test_point_id,
                "neighbor_id":                     neighbor_id,
                "cluster_id":                      cluster_id,
                "rank":                            rank,
                "wasserstein_distance":            wd_n,
                "wasserstein_distance_normalized": wd_n,
                "wasserstein_distance_unnormalized": wd_u,
                "dtw_distance_normalized":         dtw_n,
                "dtw_distance_unnormalized":       dtw_u,
                "neighbor_count":                neighbor_count,
            })

        if args.debug_plot and debug_plot_count < args.debug_plot_max and plot_neighbors and eval_inbound is not None:
            save_debug_plot(
                out_dir=debug_plot_dir,
                model=args.model,
                test_point_id=test_point_id,
                cluster_id=cluster_id,
                eval_ts_raw=np.asarray(eval_inbound, dtype=np.float64),
                neighbors=plot_neighbors,
            )
            save_debug_timeseries_csv(
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
        f"[query_eval] Eval loop done: {n_eval_total:,} points in {wall_total_s:.1f}s wall, "
        f"ANN search total {cumulative_search_ms:.1f} ms (avg {cumulative_search_ms / n_eval_total:.4f} ms/query)",
        flush=True,
    )

    # -- Write output CSVs ---------------------------------------------------
    dist_path = os.path.join(args.output_dir, f"{args.model}_distances.csv")
    time_path = os.path.join(args.output_dir, f"{args.model}_query_times.csv")
    pd.DataFrame(distance_rows).to_csv(dist_path, index=False)
    pd.DataFrame(timing_rows).to_csv(time_path, index=False)

    # -- Sanity report -------------------------------------------------------
    n_finite_n = sum(
        1 for r in distance_rows if np.isfinite(r["wasserstein_distance_normalized"])
    )
    n_finite_u = sum(
        1 for r in distance_rows if np.isfinite(r["wasserstein_distance_unnormalized"])
    )
    n_finite_dtw_n = sum(1 for r in distance_rows if np.isfinite(r["dtw_distance_normalized"]))
    n_finite_dtw_u = sum(1 for r in distance_rows if np.isfinite(r["dtw_distance_unnormalized"]))
    print(f"\n=== [{args.model}] Sanity check ===")
    print(f"  Eval points queried:   {len(timing_rows):,}")
    print(f"  Distance rows total:   {len(distance_rows):,}  "
          f"(finite WD: normalized={n_finite_n}, unnormalized={n_finite_u}; "
          f"finite DTW: normalized={n_finite_dtw_n}, unnormalized={n_finite_dtw_u})")
    print(f"  Eval points missing inbound ts:     {ts_missing_eval}")
    print(f"  Neighbor entries missing inbound ts: {ts_missing_nbr}")
    print(f"  Padded neighbor slots (cluster had < topk train): {n_padded_slots_total:,}")
    if args.no_cluster_filter:
        print("  Cluster filter: DISABLED (IVF partitioning only, no scalar filter)")
    else:
        print(f"  Cluster filter: enforced by Milvus scalar filter (expr='cluster_id == <id>')")
    print(
        "  Wasserstein: normalized = weights on time indices (joint mask); "
        "unnormalized = empirical WD on raw aligned values (no weights)"
    )
    print(
        "  DTW: L1 cost, same length after preprocessing as WD; "
        "normalized=z-score after joint mask; unnormalized=1e6 scale + clip"
    )
    if args.debug_plot:
        print(f"  Debug plots saved: {debug_plot_count} in {debug_plot_dir}")
    print(f"  Saved distances:   {dist_path}")
    print(f"  Saved query times: {time_path}")
    print(f"[query_eval] Done.\n")


if __name__ == "__main__":
    main()

