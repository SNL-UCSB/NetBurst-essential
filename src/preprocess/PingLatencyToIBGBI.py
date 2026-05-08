#!/usr/bin/env python3
import os
import sys
import argparse

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window


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
    for col in df.columns:
        norm = col.strip().lower().replace(" ", "_")
        if norm != col:
            out = out.withColumnRenamed(col, norm)
    return out


def build_ibgbi_from_sparse_examples(df, threshold_ms: float, args, num_tasks: int, task_rank: int):
    entity_col = "label" if "label" in df.columns else "ip" if "ip" in df.columns else None
    if entity_col is None:
        raise ValueError(f"Sparse parquet must have 'label' or 'ip' column; got {df.columns}")

    required_cols = {entity_col, "source_file", "inbound"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Sparse parquet must include {sorted(required_cols)}; missing {missing}. Found columns: {df.columns}"
        )

    optional_cols = [col for col in ["chunk_idx", "example_id", "first_bin", "bins"] if col in df.columns]
    df = df.select(
        F.col(entity_col).cast("string").alias("label"),
        F.col("source_file").cast("string").alias("source_file"),
        F.col("inbound").alias("inbound"),
        *[F.col(col) for col in optional_cols],
    )

    if "chunk_idx" in df.columns:
        df = df.withColumn("chunk_idx", F.col("chunk_idx").cast("int"))
    else:
        df = df.withColumn("chunk_idx", F.lit(0).cast("int"))

    if "example_id" not in df.columns:
        df = df.withColumn("example_id", F.lit(None).cast("string"))

    if "first_bin" in df.columns:
        df = df.withColumn("first_bin", F.col("first_bin").cast("long"))

    # Task sharding is handled by reading task-specific input directories,
    # so no per-row filtering is needed here.
    df = df.withColumn("inbound", F.coalesce(F.col("inbound"), F.array()))

    if "bins" in df.columns:
        df = df.withColumn("bins", F.coalesce(F.col("bins"), F.array()))

    active_pos_expr = (
        "IF(size(inbound) = 0, array(), "
        f"filter(sequence(0, size(inbound) - 1), i -> cast(element_at(inbound, i + 1) as double) >= {float(threshold_ms)}))"
    )
    df = df.withColumn("active_pos", F.expr(active_pos_expr))
    df = df.withColumn(
        "bi",
        F.expr("transform(active_pos, i -> cast(element_at(inbound, i + 1) as double))"),
    )

    if "bins" in df.columns:
        active_bins_expr = "transform(active_pos, i -> cast(element_at(bins, i + 1) as long))"
    elif "first_bin" in df.columns:
        active_bins_expr = "transform(active_pos, i -> cast(i + first_bin as long))"
    else:
        active_bins_expr = "transform(active_pos, i -> cast(i as long))"
    df = df.withColumn("active_bins", F.expr(active_bins_expr))

    ibg_expr = """
        transform(
          sequence(1, size(active_pos)),
          k -> IF(
            k = 1,
            CAST(0 AS BIGINT),
            CAST(element_at(active_pos, k) - element_at(active_pos, k - 1) AS BIGINT)
          )
        )
    """
    df = df.withColumn("ibg", F.expr(ibg_expr))
    if args.ibg_in_seconds:
        df = df.withColumn(
            "ibg",
            F.expr(f"transform(ibg, x -> cast(x * {int(args.bin_sec)} as bigint))"),
        )

    if "first_bin" in df.columns:
        df = df.withColumn(
            "first_bin",
            F.coalesce(F.col("first_bin"), F.element_at(F.col("active_bins"), 1)),
        )
    else:
        df = df.withColumn("first_bin", F.element_at(F.col("active_bins"), 1))

    df = df.withColumn("active_len", F.size(F.col("bi")))
    df = df.filter(F.col("active_len") >= F.lit(args.min_seq_len))
    df = (
        df.withColumn("active_bins", F.slice(F.col("active_bins"), F.lit(1), F.lit(args.max_seq_len)))
        .withColumn("bi", F.slice(F.col("bi"), F.lit(1), F.lit(args.max_seq_len)))
        .withColumn("ibg", F.slice(F.col("ibg"), F.lit(1), F.lit(args.max_seq_len)))
        .drop("active_pos", "active_len", "inbound")
    )

    df = df.withColumn(
        "example_id",
        F.coalesce(
            F.col("example_id"),
            F.concat_ws("_", F.col("source_file"), F.col("label"), F.col("chunk_idx")),
        ),
    )

    out_cols = ["label", "source_file", "chunk_idx", "example_id", "first_bin", "bi", "ibg"]
    if args.store_active_bins:
        out_cols.append("active_bins")
    else:
        df = df.drop("active_bins")

    return df.select(*out_cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="*",
        default=None,
        help="One or more CSV files (or globs expanded by shell); not required when --sparse_input_dir is set",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory base for processed data",
    )
    parser.add_argument(
        "--threshold_ms",
        type=float,
        default=None,
        help="A bin is active if average latency (ms) >= threshold_ms; optional if --threshold_quantiles_path etc. are set",
    )
    parser.add_argument(
        "--threshold_quantiles_path",
        type=str,
        default=None,
        help="Path to quantiles Parquet; use with --threshold_quantile to set threshold from quantiles",
    )
    parser.add_argument(
        "--threshold_quantile",
        type=float,
        default=None,
        help="Quantile in [0,1] for latency threshold (e.g. 0.9); used with --threshold_quantiles_path",
    )
    parser.add_argument(
        "--threshold_entity_type",
        type=str,
        default="ping_latency",
        help="Entity type in quantiles (default: ping_latency); used with --threshold_quantiles_path",
    )
    parser.add_argument(
        "--sparse_input_dir",
        type=str,
        default=None,
        help="Optional: base path of existing sparse ping Parquet (OUT_label); when set, build IBG/BI from sparse instead of raw CSV",
    )
    parser.add_argument(
        "--bin_sec",
        type=int,
        default=180,
        help="Window size in seconds; default 180 (3 minutes)",
    )
    parser.add_argument(
        "--expected_sampling_sec",
        type=int,
        default=180,
        help="Expected raw ping interval in seconds for sampling check",
    )
    parser.add_argument(
        "--strict_sampling_check",
        action="store_true",
        help="Fail if any entity has non-expected sampling interval",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=10,
        help="Keep entities with at least this many active bins",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=9000,
        help="Cap arrays to this many elements",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=480,
        help="Fixed output sequence length per example; longer sequences are split into multiple examples",
    )
    parser.add_argument(
        "--ibg_in_seconds",
        action="store_true",
        help="If set, scale IBG from bins to seconds",
    )
    parser.add_argument(
        "--store_active_bins",
        action="store_true",
        help="Store active_bins array; disable for compact output",
    )
    parser.add_argument(
        "--shuffle_partitions",
        type=int,
        default=None,
        help="Override spark.sql.shuffle.partitions",
    )
    args = parser.parse_args()

    if args.bin_sec <= 0:
        raise ValueError(f"bin_sec must be > 0; got {args.bin_sec}")
    if args.expected_sampling_sec <= 0:
        raise ValueError(
            f"expected_sampling_sec must be > 0; got {args.expected_sampling_sec}"
        )
    if args.min_seq_len <= 0:
        raise ValueError(f"min_seq_len must be > 0; got {args.min_seq_len}")
    if args.max_seq_len < args.min_seq_len:
        raise ValueError(
            f"max_seq_len must be >= min_seq_len; got {args.max_seq_len} < {args.min_seq_len}"
        )
    if args.seq_len <= 0:
        raise ValueError(f"seq_len must be > 0; got {args.seq_len}")

    # Resolve threshold_ms: from quantiles (via select_bi_threshold.py) or from --threshold_ms
    if args.threshold_quantiles_path and args.threshold_quantile is not None:
        _preprocess_dir = os.path.dirname(os.path.abspath(__file__))
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_preprocess_dir, "select_bi_threshold.py"),
                "--quantiles_path", args.threshold_quantiles_path,
                "--entity_type", args.threshold_entity_type,
                "--quantile", str(args.threshold_quantile),
            ],
            capture_output=True,
            text=True,
            cwd=_preprocess_dir,
        )
        if result.returncode != 0:
            raise RuntimeError(f"select_bi_threshold failed: {result.stderr or result.stdout}")
        threshold_ms = float(result.stdout.strip())
        print(f"Threshold from quantiles: {threshold_ms} ms (entity_type={args.threshold_entity_type}, quantile={args.threshold_quantile})")
    elif args.threshold_ms is not None:
        threshold_ms = float(args.threshold_ms)
    else:
        raise ValueError("Provide either --threshold_ms or (--threshold_quantiles_path and --threshold_quantile)")

    use_sparse = bool(args.sparse_input_dir)
    if not use_sparse and (not args.input_paths or len(args.input_paths) == 0):
        raise ValueError("Provide --input_paths (CSV files) or --sparse_input_dir (existing sparse Parquet)")

    spark = build_spark("Ping Latency to IBG/BI")

    default_par = spark.sparkContext.defaultParallelism or (os.cpu_count() or 1)
    if args.shuffle_partitions is None:
        shuffle_parts = max(default_par * 2, 8)
    else:
        shuffle_parts = int(args.shuffle_partitions)

    spark.conf.set("spark.sql.shuffle.partitions", shuffle_parts)
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB")

    print(f"Input paths: {args.input_paths}")
    print(f"Output dir base: {args.output_dir}")
    print(f"Threshold ms: {threshold_ms}")
    print(f"Bin seconds: {args.bin_sec}")
    print(f"Expected sampling seconds: {args.expected_sampling_sec}")
    print(f"Min active bins/group: {args.min_seq_len}; Max elements: {args.max_seq_len}")
    print(f"Fixed output sequence length: {args.seq_len}")
    print(f"Shuffle partitions: {shuffle_parts} (defaultParallelism={default_par})")

    if use_sparse:
        # In sparse-input mode, each task reads from its corresponding task_${task_rank} directory
        # to avoid all tasks reading all data and then filtering.
        task_specific_dir = f"{args.sparse_input_dir}/task_{task_rank}"
        try:
            sparse_df = spark.read.parquet(task_specific_dir)
        except Exception:
            # Fallback: if task-specific directory doesn't exist, read all and filter by hash
            sparse_df = spark.read.parquet(args.sparse_input_dir)
        out = build_ibgbi_from_sparse_examples(
            sparse_df,
            threshold_ms=threshold_ms,
            args=args,
            num_tasks=num_tasks,
            task_rank=task_rank,
        )
    else:
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

        df = df.filter(F.col("label").isNotNull() & F.col("ts").isNotNull() & F.col("latency_ms").isNotNull())

        # Distribute entities across tasks
        df = df.withColumn(
            "entity_hash",
            F.pmod(F.xxhash64(F.concat(F.col("label"), F.lit("_"), F.col("source_file"))), F.lit(num_tasks)),
        )
        df = df.filter(F.col("entity_hash") == F.lit(task_rank)).drop("entity_hash")

        # Sampling interval check (expected every 3 minutes by default)
        w_time = Window.partitionBy("label", "source_file").orderBy("ts")
        df_check = (
            df.withColumn("prev_ts", F.lag("ts").over(w_time))
            .withColumn(
                "interval_sec",
                F.when(
                    F.col("prev_ts").isNotNull(),
                    F.unix_timestamp("ts") - F.unix_timestamp("prev_ts"),
                ),
            )
        )

        sampling_stats = (
            df_check.filter(F.col("interval_sec").isNotNull())
            .groupBy("label", "source_file")
            .agg(
                F.count("*").alias("interval_count"),
                F.sum(F.when(F.col("interval_sec") != F.lit(args.expected_sampling_sec), 1).otherwise(0)).alias("non_expected_count"),
                F.min("interval_sec").alias("min_interval_sec"),
                F.max("interval_sec").alias("max_interval_sec"),
                F.avg("interval_sec").alias("avg_interval_sec"),
            )
        )

        sampling_stats.write.mode("overwrite").option("compression", "zstd").parquet(
            f"{args.output_dir}_sampling_check/task_{task_rank}"
        )

        summary = sampling_stats.agg(
            F.count("*").alias("entities_with_intervals"),
            F.sum("interval_count").alias("total_intervals"),
            F.sum("non_expected_count").alias("non_expected_total"),
            F.sum(F.when(F.col("non_expected_count") > 0, 1).otherwise(0)).alias("entities_with_non_expected"),
        ).collect()[0]

        entities_with_intervals = int(summary["entities_with_intervals"] or 0)
        total_intervals = int(summary["total_intervals"] or 0)
        non_expected_total = int(summary["non_expected_total"] or 0)
        entities_with_non_expected = int(summary["entities_with_non_expected"] or 0)

        print(
            "Sampling check: "
            f"entities_with_intervals={entities_with_intervals}, "
            f"total_intervals={total_intervals}, "
            f"non_expected_total={non_expected_total}, "
            f"entities_with_non_expected={entities_with_non_expected}, "
            f"expected_sec={args.expected_sampling_sec}"
        )

        if args.strict_sampling_check and non_expected_total > 0:
            raise ValueError(
                "strict_sampling_check enabled and non-expected sampling intervals were found. "
                f"See {args.output_dir}_sampling_check for details."
            )

        df_bins = (
            df.withColumn("bin", F.floor(F.unix_timestamp("ts") / F.lit(args.bin_sec)).cast("long"))
            .groupBy("label", "source_file", "bin")
            .agg(F.avg("latency_ms").alias("avg_latency_ms"))
        )

        # Active windows by threshold
        df_active = df_bins.filter(F.col("avg_latency_ms") >= F.lit(float(threshold_ms)))

        # IBG over active bins
        w_bin = Window.partitionBy("label", "source_file").orderBy("bin")
        df_with_ibg = df_active.withColumn("ibg_raw", F.col("bin") - F.lag("bin").over(w_bin))
        ibg_filled = F.coalesce(F.col("ibg_raw"), F.lit(0))
        ibg_final = ibg_filled * F.lit(args.bin_sec) if args.ibg_in_seconds else ibg_filled

        row_struct = F.struct(
            F.col("bin").alias("bin"),
            F.col("avg_latency_ms").cast("double").alias("bi"),
            ibg_final.cast("long").alias("ibg"),
        )

        grouped = (
            df_with_ibg.select("label", "source_file", row_struct.alias("row"))
            .groupBy("label", "source_file")
            .agg(F.array_sort(F.collect_list("row")).alias("rows"))
        )

        out = grouped.select(
            "label",
            "source_file",
            F.transform(F.col("rows"), lambda x: x["bin"]).alias("active_bins"),
            F.transform(F.col("rows"), lambda x: x["bi"]).alias("bi"),
            F.transform(F.col("rows"), lambda x: x["ibg"]).alias("ibg"),
        )

        out = out.withColumn("active_len", F.size("active_bins"))
        out = out.filter(F.col("active_len") >= F.lit(args.seq_len))
        out = out.withColumn("num_chunks", F.floor(F.col("active_len") / F.lit(args.seq_len)).cast("int"))
        out = out.filter(F.col("num_chunks") > 0)
        out = out.withColumn("chunk_idx", F.explode(F.sequence(F.lit(0), F.col("num_chunks") - F.lit(1))))
        out = out.withColumn("slice_start", F.col("chunk_idx") * F.lit(args.seq_len) + F.lit(1))
        out = (
            out.withColumn("active_bins", F.slice(F.col("active_bins"), F.col("slice_start"), F.lit(args.seq_len)))
            .withColumn("bi", F.slice(F.col("bi"), F.col("slice_start"), F.lit(args.seq_len)))
            .withColumn("ibg", F.slice(F.col("ibg"), F.col("slice_start"), F.lit(args.seq_len)))
            .drop("slice_start", "active_len", "num_chunks")
        )

        out = out.withColumn("first_bin", F.element_at(F.col("active_bins"), 1))
        out = out.withColumn("example_id", F.concat_ws("_", F.col("source_file"), F.col("label"), F.col("chunk_idx")))
        if not args.store_active_bins:
            out = out.drop("active_bins")

    out = out.repartition("label", "source_file", "chunk_idx")

    out.write.mode("overwrite").option("compression", "zstd").parquet(
        f"{args.output_dir}_label/task_{task_rank}"
    )

    out_count = out.count()
    print(
        f"[Task {task_rank}] Wrote latency BI/IBG datasets for labels. "
        f"Output groups={out_count}. Threshold(ms)>={threshold_ms}."
    )

    spark.stop()


if __name__ == "__main__":
    main()
