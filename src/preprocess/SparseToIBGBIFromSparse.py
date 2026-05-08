#!/usr/bin/env python3
"""
Convert sparse dense timeseries parquet to IBG/BI parquet.

This script is designed for sparse outputs produced by NonGraphSparseTimeseries.py,
where each row contains keys + source_file + inbound/outbound arrays.

It intentionally avoids assumptions from IBGAndBIProcessing.py about packet-level
aggregate schemas (e.g. src_ip/dst_ip columns in base aggregates).
"""

import argparse
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


ENTITY_KEYS = {
    "ip": ["ip"],
    "ip_service": ["ip", "service_port"],
    "subnet": ["subnet"],
    "ip2ip": ["src_ip", "dst_ip"],
}

ENTITY_SUFFIX = {
    "ip": "ip",
    "ip_service": "service",
    "subnet": "subnet",
    "ip2ip": "ip2ip",
}


def _resolve_input_path(base_or_dir: str, entity_type: str) -> str:
    """
    Accept either:
    1) direct entity path (e.g. /.../sparse_1s_ip), or
    2) base path (e.g. /.../sparse_1s) and append suffix (_ip/_service/_subnet/_ip2ip).
    """
    if os.path.isdir(base_or_dir):
        return base_or_dir

    suffix = ENTITY_SUFFIX[entity_type]
    candidate = f"{base_or_dir}_{suffix}"
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not resolve sparse input directory from '{base_or_dir}'. "
        f"Checked '{base_or_dir}' and '{candidate}'."
    )


def _validate_args(args):
    if args.min_seq_len <= 0:
        raise ValueError(f"--min_seq_len must be > 0, got {args.min_seq_len}")
    if args.max_seq_len <= 0:
        raise ValueError(f"--max_seq_len must be > 0, got {args.max_seq_len}")
    if args.bin_ms <= 0:
        raise ValueError(f"--bin_ms must be > 0, got {args.bin_ms}")


def main():
    parser = argparse.ArgumentParser(description="Convert sparse timeseries parquet to IBG/BI")
    parser.add_argument("--sparse_input", type=str, required=True,
                        help="Sparse input base path or direct entity directory")
    parser.add_argument("--entity_type", type=str, required=True, choices=list(ENTITY_KEYS.keys()),
                        help="Entity type to process")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for IBG/BI parquet")
    parser.add_argument("--burst_column", type=str, default="inbound", choices=["inbound", "outbound"],
                        help="Sparse array column used to define bursts")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Burst threshold on burst_column")
    parser.add_argument("--min_seq_len", type=int, default=10,
                        help="Minimum number of bursts to keep")
    parser.add_argument("--max_seq_len", type=int, default=9000,
                        help="Maximum number of bursts to keep")
    parser.add_argument("--ibg_in_ms", action="store_true",
                        help="Store IBG values in milliseconds (else bin units)")
    parser.add_argument("--bin_ms", type=int, default=1000,
                        help="Bin size in ms (only used when --ibg_in_ms is set)")
    parser.add_argument("--store_active_bins", action="store_true",
                        help="Store active_bins array in output")
    parser.add_argument("--disable_slurm_shard", action="store_true",
                        help="Do not shard rows by SLURM task; process full input on this process")
    args = parser.parse_args()
    _validate_args(args)

    task_rank = int(os.getenv("SLURM_PROCID", "0"))
    num_tasks = int(os.getenv("SLURM_NTASKS", "1"))

    spark = (
        SparkSession.builder
        .appName(f"Sparse->IBGBI {args.entity_type} task {task_rank}")
        .config("spark.driver.memory", "64g")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )

    key_cols = ENTITY_KEYS[args.entity_type]
    input_dir = _resolve_input_path(args.sparse_input, args.entity_type)

    # Sparse datasets are typically laid out as task_*/part-*.parquet.
    # Read via task_* glob so Spark sees parquet files under nested task directories.
    input_glob = os.path.join(input_dir.rstrip("/"), "task_*")
    df = spark.read.parquet(input_glob)

    required_cols = set(key_cols + ["source_file", args.burst_column])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in sparse input {input_dir}: {missing}")

    # Keep only relevant columns and optionally shard across SLURM tasks.
    df = df.select(*(key_cols + ["source_file", args.burst_column]))
    if not args.disable_slurm_shard:
        shard_hash = F.pmod(
            F.xxhash64(*[F.col(c).cast("string") for c in key_cols + ["source_file"]]),
            F.lit(num_tasks),
        )
        df = df.withColumn("_task_hash", shard_hash).filter(F.col("_task_hash") == F.lit(task_rank)).drop("_task_hash")

    burst_col = args.burst_column
    threshold = float(args.threshold)

    # Normalize null arrays and safely build active bins.
    # Spark's sequence(0, -1) is invalid for our use (can lead to element_at index 0),
    # so we guard empty arrays explicitly.
    df = df.withColumn("_burst_arr", F.coalesce(F.col(burst_col), F.array()))

    # active_bins: 0-based positions where selected sparse value >= threshold
    df = df.withColumn(
        "active_bins",
        F.expr(
            f"IF(size(_burst_arr) = 0, array(), filter(sequence(0, size(_burst_arr) - 1), i -> cast(element_at(_burst_arr, i + 1) as double) >= {threshold}))"
        ),
    )

    # bi: sparse values at active bins
    df = df.withColumn(
        "bi",
        F.expr("transform(active_bins, i -> cast(element_at(_burst_arr, i + 1) as long))"),
    )

    # ibg in bins: first element is 0, then differences between consecutive active bins
    df = df.withColumn(
        "ibg",
        F.expr(
            """
            transform(
              sequence(1, size(active_bins)),
              k -> IF(
                k = 1,
                CAST(0 AS BIGINT),
                CAST(element_at(active_bins, k) - element_at(active_bins, k - 1) AS BIGINT)
              )
            )
            """
        ),
    )

    if args.ibg_in_ms:
        df = df.withColumn("ibg", F.expr(f"transform(ibg, x -> cast(x * {int(args.bin_ms)} as bigint))"))

    df = df.withColumn("_burst_len", F.size(F.col("bi")))
    df = df.filter(F.col("_burst_len") >= F.lit(args.min_seq_len))

    df = (
        df.withColumn("active_bins", F.expr(f"slice(active_bins, 1, {int(args.max_seq_len)})"))
          .withColumn("bi", F.expr(f"slice(bi, 1, {int(args.max_seq_len)})"))
          .withColumn("ibg", F.expr(f"slice(ibg, 1, {int(args.max_seq_len)})"))
          .withColumn("first_bin", F.element_at(F.col("active_bins"), 1))
            .drop("_burst_len", "_burst_arr")
    )

    out_cols = key_cols + ["source_file", "first_bin", "bi", "ibg"]
    if args.store_active_bins:
        out_cols.append("active_bins")
    out_df = df.select(*out_cols)

    out_path = args.output_dir
    out_df.write.mode("overwrite").option("compression", "zstd").parquet(f"{out_path}/task_{task_rank}")

    print(
        f"[Task {task_rank}/{num_tasks}] Wrote sparse->IBGBI for entity={args.entity_type} "
        f"from {input_glob} to {out_path}/task_{task_rank} "
        f"(burst_column={args.burst_column}, threshold={threshold})"
    )


if __name__ == "__main__":
    main()
