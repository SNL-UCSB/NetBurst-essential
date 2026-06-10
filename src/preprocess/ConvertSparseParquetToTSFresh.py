#!/usr/bin/env python3
"""
Extract real TSFresh features from sparse inbound parquet datasets.

Expected parquet columns:
  - ip
  - source_file
  - inbound

Output CSV schema:
  ip, source_file, <tsfresh feature columns...>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import (
        ComprehensiveFCParameters,
        EfficientFCParameters,
        MinimalFCParameters,
    )
    from tsfresh.utilities.dataframe_functions import impute
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import tsfresh. Install it with `pip install tsfresh`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TSFresh features from sparse inbound parquet."
    )
    parser.add_argument(
        "--parquet_root",
        type=str,
        required=True,
        help="Root directory of sparse parquet dataset (task_* layout allowed).",
    )
    out = parser.add_mutually_exclusive_group(required=True)
    out.add_argument("--output_file", type=str, help="Single output CSV path.")
    out.add_argument(
        "--output_dir",
        type=str,
        help="Distributed output base directory (writes task_$SLURM_PROCID subdir).",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="efficient",
        choices=["minimal", "efficient", "comprehensive"],
        help="TSFresh feature parameter preset.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=0,
        help="Parallel workers for tsfresh (0 means all cores).",
    )
    parser.add_argument(
        "--time_scale_ms",
        type=int,
        default=1000,
        help="Time step for long-form tsfresh time column.",
    )
    parser.add_argument(
        "--fixed_len",
        type=int,
        default=600,
        help="Pad/truncate inbound arrays to fixed length.",
    )
    parser.add_argument(
        "--chunk_rows",
        type=int,
        default=500,
        help="Rows per extraction chunk.",
    )
    parser.add_argument(
        "--partition_strategy",
        type=str,
        default="hash",
        choices=["hash", "row_mod"],
        help="When running with SLURM tasks, partition strategy across tasks.",
    )
    return parser.parse_args()


def get_fc_parameters(name: str):
    if name == "minimal":
        return MinimalFCParameters()
    if name == "comprehensive":
        return ComprehensiveFCParameters()
    return EfficientFCParameters()


def _stable_int_hash(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _pad_trunc(arr: List[float], fixed_len: int) -> np.ndarray:
    if arr is None:
        return np.zeros((fixed_len,), dtype=np.float64)
    values = np.array([0.0 if v is None else float(v) for v in arr], dtype=np.float64)
    # Sparse generators may use -1 as padding; treat as missing.
    values[values < 0] = 0.0
    if values.size >= fixed_len:
        return values[:fixed_len]
    out = np.zeros((fixed_len,), dtype=np.float64)
    out[: values.size] = values
    return out


def _build_long(values_2d: np.ndarray, time_step_ms: int) -> pd.DataFrame:
    batch_size, timesteps = values_2d.shape
    ids = np.repeat(np.arange(batch_size, dtype=np.int64), timesteps)
    times = np.tile(np.arange(timesteps, dtype=np.int64) * int(time_step_ms), batch_size)
    values = values_2d.reshape(batch_size * timesteps)
    return pd.DataFrame({"id": ids, "time": times, "value": values})


def _resolve_dataset_sources(parquet_root: str) -> list[str]:
    root = os.path.expanduser(parquet_root)
    has_glob = any(ch in root for ch in ["*", "?", "["])
    matches = sorted(glob.glob(root)) if has_glob else [root]
    if not matches:
        raise FileNotFoundError(f"No paths matched parquet_root pattern: {parquet_root}")

    parquet_files: list[str] = []
    for path_str in matches:
        path = Path(path_str)
        if path.is_file():
            if path.suffix == ".parquet":
                parquet_files.append(str(path))
            continue
        if path.is_dir():
            parquet_files.extend(
                sorted(str(p) for p in path.rglob("*.parquet") if p.is_file())
            )

    if parquet_files:
        # Keep deterministic order and avoid duplicate entries.
        return sorted(set(parquet_files))

    return [str(Path(root))]


def main() -> None:
    args = parse_args()

    task_rank = int(os.getenv("SLURM_PROCID", "0"))
    num_tasks = int(os.getenv("SLURM_NTASKS", "1"))
    distributed = args.output_dir is not None

    if distributed:
        out_base = os.path.abspath(args.output_dir)
        out_dir = os.path.join(out_base, f"task_{task_rank}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "tsfresh_features.csv")
    else:
        out_path = os.path.abspath(args.output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Always start fresh for this run; append mode below is only for chunk flushes.
    if os.path.exists(out_path):
        os.remove(out_path)

    print(
        "Running ConvertSparseParquetToTSFresh with:",
        f"parquet_root={args.parquet_root}",
        f"out={out_path}",
        f"feature_set={args.feature_set}",
        f"fixed_len={args.fixed_len}",
        f"chunk_rows={args.chunk_rows}",
        f"rank={task_rank} ntasks={num_tasks} strategy={args.partition_strategy}",
        sep="\n  ",
    )

    dataset_sources = _resolve_dataset_sources(args.parquet_root)
    dataset = ds.dataset(dataset_sources, format="parquet")
    scanner = dataset.scanner(columns=["ip", "source_file", "inbound"])
    fc_params = get_fc_parameters(args.feature_set)

    feature_chunks = []
    total_rows_seen = 0
    total_rows_used = 0

    start_time = time.perf_counter()
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        total_rows_seen += len(frame)

        if num_tasks > 1:
            if args.partition_strategy == "row_mod":
                frame = frame.reset_index(drop=True)
                frame = frame[frame.index % num_tasks == task_rank]
            else:
                keys = (
                    frame["ip"].astype(str) + "|" + frame["source_file"].astype(str)
                ).tolist()
                assigned = [(_stable_int_hash(k) % num_tasks) for k in keys]
                frame = frame[np.array(assigned, dtype=np.int64) == task_rank]

        if frame.empty:
            continue

        frame = frame.reset_index(drop=True)
        for start in range(0, len(frame), int(args.chunk_rows)):
            chunk = frame.iloc[start : start + int(args.chunk_rows)].reset_index(drop=True)
            if chunk.empty:
                continue

            values = np.vstack(
                [_pad_trunc(v, int(args.fixed_len)) for v in chunk["inbound"].tolist()]
            )
            long_df = _build_long(values, time_step_ms=int(args.time_scale_ms))

            id_map = chunk[["ip", "source_file"]].copy()
            id_map["__id__"] = np.arange(len(chunk), dtype=np.int64)

            feats = extract_features(
                long_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=fc_params,
                n_jobs=int(args.n_jobs),
                disable_progressbar=True,
            )
            impute(feats)

            feats = feats.reset_index().rename(columns={"index": "__id__"})
            out = id_map.merge(feats, on="__id__", how="inner").drop(columns=["__id__"])
            feature_chunks.append(out)
            total_rows_used += len(out)

            if len(feature_chunks) >= 10:
                partial = pd.concat(feature_chunks, ignore_index=True)
                header = not os.path.exists(out_path)
                partial.to_csv(out_path, mode="a", index=False, header=header)
                feature_chunks = []
                print(f"Flushed chunk output. total_rows_used={total_rows_used}")

    if feature_chunks:
        partial = pd.concat(feature_chunks, ignore_index=True)
        header = not os.path.exists(out_path)
        partial.to_csv(out_path, mode="a", index=False, header=header)

    elapsed = time.perf_counter() - start_time
    rate = (total_rows_used / elapsed) if elapsed > 0 else float("nan")
    print(
        f"TSFresh extraction wall time: {elapsed:.3f}s; "
        f"input rows seen={total_rows_seen}, rows processed={total_rows_used}, "
        f"rate={rate:.2f} rows/s"
    )
    print(
        f"Done. total_rows_seen={total_rows_seen} "
        f"total_rows_used={total_rows_used} wrote={out_path}"
    )


if __name__ == "__main__":
    main()
