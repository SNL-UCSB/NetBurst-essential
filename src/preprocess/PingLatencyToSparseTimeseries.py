#!/usr/bin/env python3
import os
import argparse

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


# ----------------------------
# MPI/Slurm setup
# ----------------------------
task_rank = int(os.getenv("SLURM_PROCID", "0"))
num_tasks = int(os.getenv("SLURM_NTASKS", "1"))
print(f"----------------TASK RANK {task_rank}")


def build_spark(app_name: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(f"{app_name} - Task {task_rank}")
        .config("spark.driver.memory", "200g")
        .config("spark.executor.memory", "200g")
        .config("spark.driver.maxResultSize", "200g")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )
    return spark


def normalize_columns(df):
    out = df
    for col_name in df.columns:
        norm = col_name.strip().lower().replace(" ", "_")
        if norm != col_name:
            out = out.withColumnRenamed(col_name, norm)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV files (or globs expanded by shell)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory base for sparse timeseries parquet",
    )
    parser.add_argument(
        "--threshold_ms",
        type=float,
        required=True,
        help="Keep a bin value iff avg latency (ms) >= threshold_ms, else 0",
    )
    parser.add_argument(
        "--bin_sec",
        type=int,
        default=180,
        help="Window size in seconds; default 180 (3 minutes)",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=10,
        help="Keep entities with at least this many non-zero bins",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=9000,
        help="Cap dense timeseries to this many elements",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=480,
        help="Fixed output sequence length per example; longer sequences are split into multiple examples",
    )
    parser.add_argument(
        "--store_bins",
        action="store_true",
        help="Store dense bin-index array in output",
    )
    parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=None,
        help="Override spark.sql.shuffle.partitions",
    )
    parser.add_argument(
        "--compute_quantiles",
        action="store_true",
        help="Compute and store reference quantiles over sparse inbound values (including zeros)",
    )
    parser.add_argument(
        "--quantile_levels",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated quantile levels for reference (default: 10 evenly spaced)",
    )
    args = parser.parse_args()

    if args.bin_sec <= 0:
        raise ValueError(f"bin_sec must be > 0; got {args.bin_sec}")
    if args.min_seq_len <= 0:
        raise ValueError(f"min_seq_len must be > 0; got {args.min_seq_len}")
    if args.max_seq_len < args.min_seq_len:
        raise ValueError(
            f"max_seq_len must be >= min_seq_len; got {args.max_seq_len} < {args.min_seq_len}"
        )
    if args.seq_len <= 0:
        raise ValueError(f"seq_len must be > 0; got {args.seq_len}")

    spark = build_spark("Ping Latency to Sparse Timeseries")

    default_par = spark.sparkContext.defaultParallelism or (os.cpu_count() or 1)
    if args.shuffle_partitions is None:
        shuffle_parts = max(default_par * 2, 8)
    else:
        shuffle_parts = int(args.shuffle_partitions)

    spark.conf.set("spark.sql.shuffle.partitions", shuffle_parts)
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    print(f"Input paths: {args.input_paths}")
    print(f"Output dir base: {args.output_dir}")
    print(f"Threshold ms: {args.threshold_ms}")
    print(f"Bin seconds: {args.bin_sec}")
    print(f"Min nonzero bins/group: {args.min_seq_len}; Max elements: {args.max_seq_len}")
    print(f"Fixed output sequence length: {args.seq_len}")
    print(f"Shuffle partitions: {shuffle_parts} (defaultParallelism={default_par})")

    raw_df = (
        spark.read.option("header", "true")
        .option("multiLine", "false")
        .csv(*args.input_paths)
    )
    raw_df = normalize_columns(raw_df)

    required = {"label", "time", "ping"}
    missing = [c for c in required if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {raw_df.columns}"
        )

    df = raw_df.select(
        F.col("label").cast("string").alias("label"),
        F.col("time").cast("string").alias("time_str"),
        F.col("ping").cast("string").alias("ping_raw"),
        F.input_file_name().alias("input_file"),
    )

    df = df.withColumn(
        "source_file",
        F.regexp_extract(F.col("input_file"), r"^(.*)/[^/]+$", 1),
    )

    df = df.withColumn(
        "ts",
        F.coalesce(
            F.to_timestamp(F.col("time_str"), "yyyy-MM-dd HH:mm:ss.SSSSSS"),
            F.to_timestamp(F.col("time_str"), "yyyy-MM-dd HH:mm:ss.SSS"),
            F.to_timestamp(F.col("time_str")),
        ),
    )

    ping_avg_pattern = r"rtt\s+min/avg/max/mdev\s*=\s*[^/]+/([^/]+)/[^/]+/[^\s]+\s*ms"
    df = df.withColumn(
        "latency_ms",
        F.regexp_extract(F.col("ping_raw"), ping_avg_pattern, 1).cast("double"),
    )

    df = df.filter(
        F.col("label").isNotNull()
        & F.col("source_file").isNotNull()
        & F.col("ts").isNotNull()
        & F.col("latency_ms").isNotNull()
    )

    # Distribute entities across tasks
    df = df.withColumn(
        "entity_hash",
        F.pmod(
            F.xxhash64(F.concat(F.col("label"), F.lit("_"), F.col("source_file"))),
            F.lit(num_tasks),
        ),
    )
    df = df.filter(F.col("entity_hash") == F.lit(task_rank)).drop("entity_hash")

    # Average latency per 3-minute window (or configured bin_sec)
    df_bins = (
        df.withColumn("bin", F.floor(F.unix_timestamp("ts") / F.lit(args.bin_sec)).cast("long"))
        .groupBy("label", "source_file", "bin")
        .agg(F.avg("latency_ms").alias("avg_latency_ms"))
    )

    # Build dense bin range per entity, clamped to max_seq_len from first observed bin
    bounds = (
        df_bins.groupBy("label", "source_file")
        .agg(
            F.min("bin").alias("min_bin"),
            F.max("bin").alias("max_bin"),
        )
        .withColumn("end_bin", F.least(F.col("max_bin"), F.col("min_bin") + F.lit(args.max_seq_len - 1)))
        .withColumn("dense_bins", F.sequence(F.col("min_bin"), F.col("end_bin")))
        .select("label", "source_file", F.explode("dense_bins").alias("bin"))
    )

    dense = (
        bounds.join(df_bins, on=["label", "source_file", "bin"], how="left")
        .withColumn("avg_latency_ms", F.coalesce(F.col("avg_latency_ms"), F.lit(0.0)))
        .withColumn(
            "timeseries_value",
            F.when(
                F.col("avg_latency_ms") >= F.lit(float(args.threshold_ms)),
                F.col("avg_latency_ms").cast("float"),
            ).otherwise(F.lit(0.0).cast("float")),
        )
    )

    row_struct = F.struct(
        F.col("bin").alias("bin"),
        F.col("timeseries_value").alias("v"),
    )

    out = (
        dense.select("label", "source_file", row_struct.alias("row"))
        .groupBy("label", "source_file")
        .agg(F.array_sort(F.collect_list("row")).alias("rows"))
        .select(
            F.col("label").alias("ip"),
            "source_file",
            F.transform(F.col("rows"), lambda x: x["bin"]).alias("bins"),
            F.transform(F.col("rows"), lambda x: x["v"]).alias("inbound"),
        )
    )

    out = out.withColumn(
        "nonzero_len",
        F.size(F.filter(F.col("inbound"), lambda x: x > F.lit(0.0))),
    )
    out = out.filter(F.col("nonzero_len") >= F.lit(args.min_seq_len))

    out = out.withColumn("active_len", F.size("inbound"))
    out = out.filter(F.col("active_len") >= F.lit(args.seq_len))
    out = out.withColumn("num_chunks", F.floor(F.col("active_len") / F.lit(args.seq_len)).cast("int"))
    out = out.filter(F.col("num_chunks") > 0)
    out = out.withColumn("chunk_idx", F.explode(F.sequence(F.lit(0), F.col("num_chunks") - F.lit(1))))
    out = out.withColumn("slice_start", F.col("chunk_idx") * F.lit(args.seq_len) + F.lit(1))

    out = (
        out.withColumn("bins", F.slice(F.col("bins"), F.col("slice_start"), F.lit(args.seq_len)))
        .withColumn("inbound", F.slice(F.col("inbound"), F.col("slice_start"), F.lit(args.seq_len)))
        .drop("slice_start", "active_len", "num_chunks", "nonzero_len")
    )

    out = out.withColumn("first_bin", F.element_at(F.col("bins"), 1))
    out = out.withColumn("example_id", F.concat_ws("_", F.col("source_file"), F.col("ip"), F.col("chunk_idx")))
    if not args.store_bins:
        out = out.drop("bins")

    out = out.repartition("ip", "source_file", "chunk_idx")

    out.write.mode("overwrite").option("compression", "zstd").parquet(
        f"{args.output_dir}_label/task_{task_rank}"
    )

    out_count = out.count()
    print(
        f"[Task {task_rank}] Wrote sparse latency timeseries for labels. "
        f"Output groups={out_count}. Threshold(ms)>={args.threshold_ms}."
    )

    # Optional: compute and store quantiles over sparse inbound (task 0 only)
    if args.compute_quantiles and task_rank == 0:
        import sys
        _preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        if _preprocess_dir not in sys.path:
            sys.path.insert(0, _preprocess_dir)
        from quantiles_utils import (
            compute_quantiles_from_array_column,
            write_quantiles_parquet,
            DEFAULT_QUANTILE_LEVELS,
        )
        levels_str = args.quantile_levels.strip()
        levels_list = [float(x.strip()) for x in levels_str.split(",") if x.strip()]
        if not levels_list:
            levels_list = list(DEFAULT_QUANTILE_LEVELS)
        df_all = spark.read.parquet(f"{args.output_dir}_label")
        levels, quantiles = compute_quantiles_from_array_column(
            df_all, "inbound", levels=levels_list
        )
        write_quantiles_parquet(
            spark, "ping_latency", levels, quantiles, f"{args.output_dir}_quantiles"
        )
        print(f"[Task 0] Wrote quantiles for ping_latency to {args.output_dir}_quantiles")

    spark.stop()


if __name__ == "__main__":
    main()
