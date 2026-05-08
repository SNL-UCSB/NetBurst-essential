import argparse
import os
from itertools import groupby

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, LongType, StringType, StructField, StructType


def _struct_field_for_group_col(name: str) -> StructField:
    if name == "session_id":
        return StructField(name, LongType(), True)
    return StructField(name, StringType(), True)


def pad_or_truncate(arr, target_len):
    cur_len = len(arr)
    if cur_len > target_len:
        return arr[:target_len]
    if cur_len < target_len:
        return np.pad(arr, (0, target_len - cur_len), constant_values=-1)
    return arr


def arrays_from_sum_by_bin_dense(sum_by_bin, threshold, max_len):
    if not sum_by_bin:
        return None, 0

    bins = list(sum_by_bin.keys())
    min_b = min(bins)
    max_b = max(bins)

    end_b = min(max_b, min_b + max_len - 1)
    span = end_b - min_b + 1

    inbound = np.zeros(span, dtype=float)

    for b, value in sum_by_bin.items():
        if b < min_b or b > end_b:
            continue
        idx = b - min_b
        if value >= threshold:
            inbound[idx] = value

    kept = int((inbound > 0.0).sum())
    return inbound, kept


def build_series_record(group_dict, rows, threshold, min_seq_len, max_seq_len, copy_inbound_to_outbound):
    sum_by_bin = {}
    for r in rows:
        if r.bin_idx is None or r.value is None:
            continue
        b = int(r.bin_idx)
        v = float(r.value)
        sum_by_bin[b] = sum_by_bin.get(b, 0.0) + v

    inbound, kept = arrays_from_sum_by_bin_dense(sum_by_bin, threshold, max_seq_len)
    if inbound is None or kept < min_seq_len:
        return None

    inbound = pad_or_truncate(inbound, min(kept, max_seq_len))

    if copy_inbound_to_outbound:
        outbound = inbound.copy()
    else:
        outbound = np.zeros(len(inbound), dtype=float)

    out = dict(group_dict)
    out["inbound"] = inbound.tolist()
    out["outbound"] = outbound.tolist()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert ConvertToTimeseries output into NonGraphSparseTimeseries-style arrays."
    )
    parser.add_argument("--input_parquet", type=str, required=True, help="Input parquet from ConvertToTimeseries")
    parser.add_argument("--output_parquet", type=str, required=True, help="Output parquet path")
    parser.add_argument("--bin_seconds", type=int, default=10, help="Bin size in seconds (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Keep inbound bin if value >= threshold")
    parser.add_argument("--min_seq_len", type=int, default=10, help="Minimum kept bins required to keep a sequence")
    parser.add_argument("--max_seq_len", type=int, default=9000, help="Maximum sequence length")
    parser.add_argument('--num_tasks', type=int, default=1, help='Total number of hash shards (default: 1)')
    parser.add_argument('--task_id', type=int, default=0, help='Shard id for this task (0..num_tasks-1)')
    parser.add_argument(
        "--copy_inbound_to_outbound",
        action="store_true",
        help="If set, copies inbound array to outbound instead of writing zeros",
    )
    parser.add_argument(
        "--group_cols",
        type=str,
        default="meta_five_tuple_id,session_id",
        help="Comma-separated identity columns (e.g. meta_five_tuple_id,session_id or meta_src_ip,meta_dst_ip,session_id). "
        "If source_file exists in the input, it is appended automatically when not listed.",
    )
    args = parser.parse_args()

    if args.bin_seconds <= 0:
        raise ValueError(f"bin_seconds must be > 0, got {args.bin_seconds}")

    spark = SparkSession.builder \
        .appName("ConvertToNonGraphSparseFormat") \
        .config("spark.driver.memory", "200g") \
        .config("spark.executor.memory", "200g") \
        .config("spark.driver.maxResultSize", "200g") \
        .getOrCreate()

    df = spark.read.parquet(args.input_parquet)

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    if not group_cols:
        raise ValueError("--group_cols must list at least one column")

    if "source_file" in df.columns and "source_file" not in group_cols:
        group_cols.append("source_file")

    required = set(group_cols) | {"window_start", "sum_num_bits"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base = (
        df.where(F.col("sum_num_bits").isNotNull())
          .select(
              *[F.col(c) for c in group_cols],
              F.floor(F.col("window_start").cast("long") / F.lit(args.bin_seconds)).cast("long").alias("bin_idx"),
              F.col("sum_num_bits").cast("double").alias("value"),
          )
    )

    # Optional sharding by hash over group_cols (and source_file is appended automatically above if present)
    if args.num_tasks > 1:
        h = F.pmod(F.abs(F.xxhash64(*[F.col(c) for c in group_cols])), F.lit(args.num_tasks))
        base = base.withColumn("_shard", h).filter(F.col("_shard") == F.lit(args.task_id)).drop("_shard")

    normalized = (
        base.groupBy(*group_cols, "bin_idx")
            .agg(F.sum("value").alias("value"))
            .repartition(*group_cols)
            .sortWithinPartitions(*group_cols, "bin_idx")
    )

    def partition_builder(iter_rows):
        def key_fn(row):
            return tuple(getattr(row, c) for c in group_cols)

        for key, grp in groupby(iter_rows, key=key_fn):
            group_dict = {c: key[i] for i, c in enumerate(group_cols)}
            rec = build_series_record(
                group_dict=group_dict,
                rows=grp,
                threshold=args.threshold,
                min_seq_len=args.min_seq_len,
                max_seq_len=args.max_seq_len,
                copy_inbound_to_outbound=args.copy_inbound_to_outbound,
            )
            if rec is not None:
                yield rec

    out_rdd = normalized.rdd.mapPartitions(partition_builder)

    schema_fields = [_struct_field_for_group_col(c) for c in group_cols]

    schema_fields.extend([
        StructField("inbound", ArrayType(FloatType()), True),
        StructField("outbound", ArrayType(FloatType()), True),
    ])

    out_schema = StructType(schema_fields)
    out_df = spark.createDataFrame(out_rdd, out_schema)

    out_dir = os.path.join(args.output_parquet, f"task_{args.task_id}")
    out_df.write.mode("overwrite").parquet(out_dir)
    spark.stop()


if __name__ == "__main__":
    main()
