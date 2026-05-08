#!/usr/bin/env python3
"""
Create train/test (70:30 random) and context/forecast (70:30 contiguous) splits from IBG/BI.

Supports two sparse slicing modes when --sparse_dir is provided:
1) index: split sparse arrays by burst counts (legacy behavior)
2) burst_timestamp: slice sparse arrays by IBG/BI burst timestamps so sparse forecast
   contains the same forecast bursts as IBG/BI.
"""

import argparse
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


DEFAULT_SEED = 42
ENTITY_KEYS = {
    "ip": ["ip"],
    "ip_service": ["ip", "service_port"],
    "subnet": ["subnet"],
    "ip2ip": ["src_ip", "dst_ip"],
    # Ping latency (3-min bins): IBG/BI has `label`; labeled sparse may expose the same value as `ip`.
    "ping_latency": ["label", "chunk_idx"],
}


def _validate_mirror_sparse_args(args):
    if not args.sparse_dir:
        raise ValueError("--mirror_sparse_only requires --sparse_dir")
    if args.alignment_mode == "burst_timestamp" and args.burst_threshold is None:
        raise ValueError(
            "--burst_threshold is required for --alignment_mode burst_timestamp when mirroring sparse"
        )


def _validate_args(args):
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError(f"--train_ratio must be in (0,1), got {args.train_ratio}")
    if not (0.0 < args.context_ratio < 1.0):
        raise ValueError(f"--context_ratio must be in (0,1), got {args.context_ratio}")
    if args.alignment_mode == "burst_timestamp" and args.sparse_dir and args.burst_threshold is None:
        raise ValueError(
            "--burst_threshold is required for --alignment_mode burst_timestamp when --sparse_dir is set"
        )
    if args.bin_ms <= 0:
        raise ValueError(f"--bin_ms must be > 0, got {args.bin_ms}")
    if args.min_seq_len is not None and int(args.min_seq_len) < 1:
        raise ValueError(f"--min_seq_len must be >= 1 if set, got {args.min_seq_len}")
    if args.max_seq_len is not None and int(args.max_seq_len) < 1:
        raise ValueError(f"--max_seq_len must be >= 1 if set, got {args.max_seq_len}")
    if args.min_seq_len is not None and args.max_seq_len is not None:
        if int(args.min_seq_len) > int(args.max_seq_len):
            raise ValueError(
                f"--min_seq_len ({args.min_seq_len}) cannot exceed --max_seq_len ({args.max_seq_len})"
            )


def _build_burst_bins(df, ibg_col="ibg", first_bin_col="first_bin", out_col="burst_bins"):
    """
    Reconstruct per-burst bin indices from first_bin + cumulative ibg sequence.
    IBG sequence is expected to have same length as BI (first element usually 0).
    """
    return df.withColumn(
        out_col,
        F.expr(
            f"""
            aggregate(
              {ibg_col},
                            named_struct('acc', cast({first_bin_col} as bigint), 'arr', cast(array() as array<bigint>)),
              (s, x) -> named_struct('acc', s.acc + cast(x as bigint), 'arr', concat(s.arr, array(s.acc + cast(x as bigint)))),
              s -> s.arr
            )
            """
        ),
    )


def _resolve_parquet_input_path(path: str) -> str:
    """
    Resolve parquet input path for datasets stored either directly as parquet
    or under task_* subfolders.
    """
    # If user already passed a glob, keep it as-is.
    if "*" in path:
        return path

    task_glob = os.path.join(path.rstrip("/"), "task_*")
    if os.path.isdir(path) and os.path.isdir(os.path.join(path, "task_0")):
        return task_glob
    return path


def _read_parquet_maybe_recursive(spark, path: str):
    """
    Read parquet from either explicit globs or directory roots.
    If path is a plain directory (no glob), enable recursive file lookup.
    """
    reader = spark.read
    if "*" not in path:
        reader = reader.option("recursiveFileLookup", "true")
    return reader.parquet(path)


def _mirror_sparse_from_meta(spark, meta, args):
    """
    Write sparse_train/sparse_test by joining sparse series to an existing splits_meta
    (same train/test keys and context/forecast geometry as a prior split run).
    """
    keys = ENTITY_KEYS[args.entity_type] + ["source_file"]
    required = set(keys + ["split", "context_len", "forecast_len"])
    if args.alignment_mode == "burst_timestamp":
        required |= {"forecast_start_bin", "forecast_end_bin"}
    missing = sorted(c for c in required if c not in meta.columns)
    if missing:
        raise ValueError(f"splits_meta is missing required columns: {missing}")

    meta_keep = list(keys) + ["split", "context_len", "forecast_len"]
    if args.alignment_mode == "burst_timestamp":
        meta_keep += ["forecast_start_bin", "forecast_end_bin"]
    meta_slim = meta.select(*meta_keep)

    sparse_read_path = _resolve_parquet_input_path(args.sparse_dir)
    sparse_df = _read_parquet_maybe_recursive(spark, sparse_read_path)
    if args.entity_type == "ping_latency" and "label" not in sparse_df.columns and "ip" in sparse_df.columns:
        sparse_df = sparse_df.withColumn("label", F.col("ip"))
    joined = sparse_df.join(meta_slim, on=keys, how="inner")
    sparse_len_col = F.size(F.col("inbound"))

    if args.alignment_mode == "burst_timestamp":
        joined = joined.withColumn("sparse_context_len", F.least(sparse_len_col, F.col("forecast_start_bin")))
        joined = joined.withColumn(
            "sparse_forecast_len",
            F.when(
                F.col("forecast_end_bin").isNull(),
                F.lit(0),
            ).otherwise(
                F.greatest(
                    F.lit(0),
                    F.least(
                        sparse_len_col - F.col("forecast_start_bin"),
                        F.col("forecast_end_bin") - F.col("forecast_start_bin") + F.lit(1),
                    ),
                )
            ),
        )
        joined = joined.withColumn(
            "context_inbound",
            F.slice(F.col("inbound"), 1, F.col("sparse_context_len")),
        ).withColumn(
            "forecast_inbound",
            F.slice(F.col("inbound"), F.col("forecast_start_bin") + 1, F.col("sparse_forecast_len")),
        )
        if "outbound" in sparse_df.columns:
            joined = joined.withColumn(
                "context_outbound",
                F.slice(F.col("outbound"), 1, F.col("sparse_context_len")),
            ).withColumn(
                "forecast_outbound",
                F.slice(F.col("outbound"), F.col("forecast_start_bin") + 1, F.col("sparse_forecast_len")),
            )

        sparse_forecast_col = f"forecast_{args.burst_column}"
        joined = joined.withColumn(
            "sparse_forecast_burst_count",
            F.size(F.expr(f"filter({sparse_forecast_col}, x -> x >= {float(args.burst_threshold)})")),
        )
        joined = joined.withColumn("ibgbi_forecast_burst_count", F.col("forecast_len"))
    else:
        joined = joined.withColumn("sparse_context_len", F.col("context_len"))
        joined = joined.withColumn("sparse_forecast_len", F.col("forecast_len"))
        joined = joined.withColumn(
            "context_inbound",
            F.slice(F.col("inbound"), 1, F.col("context_len")),
        ).withColumn(
            "forecast_inbound",
            F.slice(F.col("inbound"), F.col("context_len") + 1, F.col("forecast_len")),
        )
        if "outbound" in sparse_df.columns:
            joined = joined.withColumn(
                "context_outbound",
                F.slice(F.col("outbound"), 1, F.col("context_len")),
            ).withColumn(
                "forecast_outbound",
                F.slice(F.col("outbound"), F.col("context_len") + 1, F.col("forecast_len")),
            )

    train_sparse = joined.filter(F.col("split") == "train")
    test_sparse = joined.filter(F.col("split") == "test")
    train_sparse.write.mode("overwrite").option("compression", "zstd").parquet(
        os.path.join(args.output_dir, "sparse_train")
    )
    test_sparse.write.mode("overwrite").option("compression", "zstd").parquet(
        os.path.join(args.output_dir, "sparse_test")
    )
    print(f"Wrote sparse_train and sparse_test (mirror_sparse_only) with alignment_mode={args.alignment_mode}")

    if args.alignment_mode == "burst_timestamp":
        mismatches = test_sparse.filter(
            F.col("sparse_forecast_burst_count") != F.col("ibgbi_forecast_burst_count")
        ).count()
        total_test = test_sparse.count()
        print(f"Burst parity on test: mismatches={mismatches} / {total_test}")
        # ping_latency: forecast_len counts IBG-BI slots; sparse >=threshold counts differ — do not enforce.
        if mismatches > 0 and args.entity_type != "ping_latency":
            raise RuntimeError(
                f"Found {mismatches} burst-count mismatches in test split under burst_timestamp mode"
            )


def main():
    parser = argparse.ArgumentParser(description="Create train/test and context/forecast splits from IBG/BI")
    parser.add_argument("--ibgbi_dir", type=str, default=None,
                        help="Path to IBG/BI Parquet dir (not used with --mirror_sparse_only)")
    parser.add_argument("--entity_type", type=str, required=True, choices=list(ENTITY_KEYS),
                        help="Entity type: ip, ip_service, subnet, ip2ip, ping_latency")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output base dir for splits")
    parser.add_argument("--min_bursts", type=int, default=10,
                        help="Keep sequences with at least this many bursts (BI > 0); default 10")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Fraction of examples for train; default 0.7")
    parser.add_argument("--context_ratio", type=float, default=0.7,
                        help="Fraction of each sequence for context; default 0.7")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for train/test split")
    parser.add_argument("--sparse_dir", type=str, default=None,
                        help="Optional sparse parquet dir to mirror splits")
    parser.add_argument("--alignment_mode", type=str, default="index", choices=["index", "burst_timestamp"],
                        help="Sparse split mode: index (legacy) or burst_timestamp (recommended)")
    parser.add_argument("--burst_column", type=str, default="inbound", choices=["inbound", "outbound"],
                        help="Sparse column used to count bursts for parity checks")
    parser.add_argument("--burst_threshold", type=float, default=None,
                        help="Threshold used to define sparse bursts (required for burst_timestamp mode)")
    parser.add_argument("--bin_ms", type=int, default=1000,
                        help="Bin size in ms (needed when --ibg_in_ms is set)")
    parser.add_argument("--ibg_in_ms", action="store_true",
                        help="Set if IBG values in input are expressed in milliseconds")
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=None,
        help="If set, drop rows with len(bi) < this (after --min_bursts filter)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="If set, slice bi/ibg/active_bins to at most this length (like SparseToIBGBIFromSparse)",
    )
    parser.add_argument(
        "--mirror_sparse_only",
        action="store_true",
        help=(
            "Only write sparse_train/sparse_test using existing splits_meta under --output_dir "
            "(or --splits_meta_dir). Does not overwrite splits_meta or ibgbi_train/ibgbi_test."
        ),
    )
    parser.add_argument(
        "--splits_meta_dir",
        type=str,
        default=None,
        help="Path to splits_meta parquet (default: output_dir/splits_meta). Used with --mirror_sparse_only.",
    )
    args = parser.parse_args()

    if args.mirror_sparse_only:
        _validate_mirror_sparse_args(args)
        meta_path = args.splits_meta_dir or os.path.join(args.output_dir, "splits_meta")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No splits_meta at {meta_path!r}. Run a full split first (without --mirror_sparse_only) "
                "so splits_meta and ibgbi_train/ibgbi_test exist, then re-run with --mirror_sparse_only."
            )
        spark = (
            SparkSession.builder.appName("Mirror sparse splits from existing splits_meta")
            .config("spark.driver.memory", "16g")
            .config("spark.sql.parquet.compression.codec", "zstd")
            .getOrCreate()
        )
        meta = _read_parquet_maybe_recursive(spark, meta_path)
        _mirror_sparse_from_meta(spark, meta, args)
        spark.stop()
        return

    if not args.ibgbi_dir:
        raise ValueError("--ibgbi_dir is required unless --mirror_sparse_only is set")
    _validate_args(args)

    spark = (
        SparkSession.builder
        .appName("Create train/test splits from IBG/BI")
        .config("spark.driver.memory", "16g")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )

    keys = ENTITY_KEYS[args.entity_type] + ["source_file"]
    ibgbi_read_path = _resolve_parquet_input_path(args.ibgbi_dir)
    ibgbi_df = _read_parquet_maybe_recursive(spark, ibgbi_read_path)

    # Filter and lengths on burst-domain sequences.
    ibgbi_df = ibgbi_df.withColumn("_burst_count", F.size(F.expr("filter(bi, x -> x > 0)")))
    ibgbi_df = ibgbi_df.filter(F.col("_burst_count") >= F.lit(args.min_bursts))
    ibgbi_df = ibgbi_df.withColumn("_seq_len", F.size(F.col("bi")))
    if args.min_seq_len is not None:
        ibgbi_df = ibgbi_df.filter(F.col("_seq_len") >= F.lit(int(args.min_seq_len)))
    if args.max_seq_len is not None:
        m = int(args.max_seq_len)
        ibgbi_df = ibgbi_df.withColumn(
            "bi",
            F.when(F.col("_seq_len") > F.lit(m), F.slice(F.col("bi"), F.lit(1), F.lit(m))).otherwise(F.col("bi")),
        )
        ibgbi_df = ibgbi_df.withColumn(
            "ibg",
            F.when(F.col("_seq_len") > F.lit(m), F.slice(F.col("ibg"), F.lit(1), F.lit(m))).otherwise(F.col("ibg")),
        )
        if "active_bins" in ibgbi_df.columns:
            ibgbi_df = ibgbi_df.withColumn(
                "active_bins",
                F.when(F.col("_seq_len") > F.lit(m), F.slice(F.col("active_bins"), F.lit(1), F.lit(m))).otherwise(
                    F.col("active_bins")
                ),
            )
        ibgbi_df = ibgbi_df.withColumn("_seq_len", F.size(F.col("bi")))

    context_len = (F.col("_seq_len") * F.lit(args.context_ratio)).cast("int")
    ibgbi_df = ibgbi_df.withColumn("context_len", context_len)
    ibgbi_df = ibgbi_df.withColumn("forecast_len", F.col("_seq_len") - F.col("context_len"))

    ibgbi_df = ibgbi_df.withColumn("_rand", F.rand(args.seed))
    ibgbi_df = ibgbi_df.withColumn(
        "split",
        F.when(F.col("_rand") < F.lit(args.train_ratio), F.lit("train")).otherwise(F.lit("test")),
    )
    ibgbi_df = ibgbi_df.withColumn("example_id", F.concat_ws("_", *[F.col(c).cast("string") for c in keys]))

    # Build per-example burst bins. If active_bins exists, prefer it.
    if "active_bins" in ibgbi_df.columns:
        burst_bins_df = ibgbi_df.withColumn("burst_bins", F.col("active_bins"))
    else:
        burst_bins_df = ibgbi_df
        if args.ibg_in_ms:
            burst_bins_df = burst_bins_df.withColumn(
                "_ibg_bins",
                F.expr(f"transform(ibg, x -> cast(cast(x as double) / {args.bin_ms} as bigint))"),
            )
        else:
            burst_bins_df = burst_bins_df.withColumn("_ibg_bins", F.col("ibg"))
        burst_bins_df = _build_burst_bins(burst_bins_df, ibg_col="_ibg_bins", first_bin_col="first_bin", out_col="burst_bins")

    burst_bins_df = burst_bins_df.withColumn(
        "context_end_bin",
        F.when(F.col("context_len") > 0, F.element_at(F.col("burst_bins"), F.col("context_len"))),
    )
    burst_bins_df = burst_bins_df.withColumn(
        "forecast_start_bin",
        F.when(F.col("forecast_len") > 0, F.element_at(F.col("burst_bins"), F.col("context_len") + F.lit(1))),
    )
    burst_bins_df = burst_bins_df.withColumn(
        "forecast_end_bin",
        F.when(
            F.col("forecast_len") > 0,
            F.element_at(F.col("burst_bins"), F.col("context_len") + F.col("forecast_len")),
        ),
    )

    # Add IBG/BI context+forecast arrays.
    ibgbi_with_slices = burst_bins_df.withColumn("context_bi", F.slice(F.col("bi"), 1, F.col("context_len")))
    ibgbi_with_slices = ibgbi_with_slices.withColumn(
        "forecast_bi", F.slice(F.col("bi"), F.col("context_len") + 1, F.col("forecast_len"))
    )
    ibgbi_with_slices = ibgbi_with_slices.withColumn("context_ibg", F.slice(F.col("ibg"), 1, F.col("context_len")))
    ibgbi_with_slices = ibgbi_with_slices.withColumn(
        "forecast_ibg", F.slice(F.col("ibg"), F.col("context_len") + 1, F.col("forecast_len"))
    )

    # Persist metadata for auditing + sparse mirroring.
    meta_cols = [
        "example_id", *keys,
        "first_bin",
        "_burst_count",
        "_seq_len",
        "context_len",
        "forecast_len",
        "context_end_bin",
        "forecast_start_bin",
        "forecast_end_bin",
        "split",
    ]
    meta = ibgbi_with_slices.select(*meta_cols)
    meta.write.mode("overwrite").option("compression", "zstd").parquet(os.path.join(args.output_dir, "splits_meta"))

    # Write IBG/BI train and test datasets with context/forecast arrays.
    ibgbi_train = ibgbi_with_slices.filter(F.col("split") == "train")
    ibgbi_test = ibgbi_with_slices.filter(F.col("split") == "test")
    ibgbi_train.write.mode("overwrite").option("compression", "zstd").parquet(os.path.join(args.output_dir, "ibgbi_train"))
    ibgbi_test.write.mode("overwrite").option("compression", "zstd").parquet(os.path.join(args.output_dir, "ibgbi_test"))

    n_total = meta.count()
    n_train = meta.filter(F.col("split") == "train").count()
    n_test = meta.filter(F.col("split") == "test").count()
    print(f"Wrote splits_meta and ibgbi_train/ibgbi_test: total={n_total}, train={n_train}, test={n_test}")

    if args.sparse_dir:
        _mirror_sparse_from_meta(spark, meta, args)

    spark.stop()


if __name__ == "__main__":
    main()
