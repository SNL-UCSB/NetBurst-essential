import os
import numpy as np
from itertools import groupby

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, ArrayType, FloatType, LongType, StringType
)
from pyspark.sql.functions import regexp_extract, col, split, concat_ws

# ----------------------------
# MPI/Spark setup
# ----------------------------
task_rank = int(os.getenv("SLURM_PROCID", "0"))
num_tasks = int(os.getenv("SLURM_NTASKS", "1"))
print(f"----------------TASK RANK {task_rank}")

spark = SparkSession.builder \
    .appName(f"PySpark Timeseries Processing - Task {task_rank}") \
    .config("spark.driver.memory", "200g") \
    .config("spark.executor.memory", "200g") \
    .config("spark.driver.maxResultSize", "200g") \
    .getOrCreate()

# ----------------------------
# Config (defaults; MIN_SEQ_LEN set after argparse)
# ----------------------------
FINETUNING  = False
MAX_SEQ_LEN = 9000

BIN_MS = 1000

# ----------------------------
# Args
# ----------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bin_ms", type=int, default=1000,
                    help="Bin size in milliseconds (>=100 and multiple of 100)")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for processed data")
parser.add_argument("--finetuning", action="store_true",
                    help="Set if processing finetuning dataset")
parser.add_argument("--threshold", type=float, default=0.0,
                    help="Keep a bin in the inbound or outbound series iff inbound >= this threshold")
parser.add_argument("--max_examples", type=int, default=1000000,
    help="Number of candidate keys to keep per dataset (per task)")
parser.add_argument(
    "--min_seq_len",
    type=int,
    default=3,
    help="Minimum active bins (max of inbound/outbound kept counts) required to emit a series",
)
parser.add_argument(
    "--min_bursts",
    type=int,
    default=3,
    help="Alias floor with --min_seq_len; series must satisfy max(in,out) >= max(min_seq_len, min_bursts)",
)
parser.add_argument(
    "--input_parquet",
    type=str,
    default=None,
    help=(
        "Override Parquet root for 100ms aggregates (dirs or hive-style table). "
        "If unset, uses PINOT_INPUT_PARQUET env, else defaults: main vs MAWI from --finetuning."
    ),
)
args = parser.parse_args()

MIN_SEQ_LEN = max(int(args.min_seq_len), int(args.min_bursts))

BIN_MS            = args.bin_ms
OUTPUT_DIR        = args.output_dir
FINETUNING        = args.finetuning
THRESH_IN_BYTES   = float(args.threshold)
THRESH_OUT_BYTES  = float(args.threshold)

if BIN_MS < 100 or BIN_MS % 100 != 0:
    raise ValueError(f"BIN_MS must be >=100 and a multiple of 100; got {BIN_MS}")

AGG_FACTOR = BIN_MS // 100
print(f"Aggregating 100ms ticks into {BIN_MS}ms bins (factor={AGG_FACTOR}).")
print(f"Finetuning: {FINETUNING}")
print(f"MIN_SEQ_LEN (effective): {MIN_SEQ_LEN} (from --min_seq_len {args.min_seq_len}, --min_bursts {args.min_bursts})")
print(f"Inbound threshold:  {THRESH_IN_BYTES} bytes/bin")
print(f"Outbound threshold: {THRESH_OUT_BYTES} bytes/bin")

# ----------------------------
# Load
# ----------------------------
_default_main = "<data-root>/service_100ms_aggregate_13dfs.parquet"
_default_mawi = "<data-root>/mawi_aggregate_100ms"
_input = args.input_parquet or os.environ.get("PINOT_INPUT_PARQUET") or ""
if _input.strip():
    _parquet_path = _input.strip()
elif not FINETUNING:
    _parquet_path = _default_main
else:
    _parquet_path = _default_mawi
print(f"Reading 100ms aggregates from: {_parquet_path}")
aggregated_df = spark.read.parquet(_parquet_path)

# Clean source_file
aggregated_df = aggregated_df.withColumn(
    "source_file",
    regexp_extract(col("source_file"), r'^(.*)/[^/]+$', 1)
)

# Filter out localhost
aggregated_df = aggregated_df.filter(~col("ip").startswith("127."))

# Add /24 subnet column
octets = split(col("ip"), "\\.")
aggregated_df = aggregated_df.withColumn(
    "subnet", concat_ws(".", octets.getItem(0), octets.getItem(1), octets.getItem(2))
)

# Distribute by task (IP-based)
aggregated_df = aggregated_df.withColumn("last_octet", octets.getItem(3).cast("int"))
aggregated_df = aggregated_df.withColumn("ip_hash", col("last_octet") % num_tasks)
df_ip_part = aggregated_df.filter(col("ip_hash") == task_rank)

# Distribute by task (Subnet-based) — use stable hash of subnet
df_subnet_part = aggregated_df.withColumn(
    "subnet_hash", F.pmod(F.xxhash64("subnet"), F.lit(num_tasks))
).filter(col("subnet_hash") == task_rank)

MAX_KEYS = int(args.max_examples)

# ---- IP: first K (ip, source_file) ----
ip_keys = (
    df_ip_part
    .select("ip", "source_file")
    .dropDuplicates(["ip", "source_file"])
    .orderBy("ip", "source_file")     # deterministic "first"
    .limit(MAX_KEYS)
)
df_ip_part_sel = df_ip_part.join(ip_keys, ["ip", "source_file"], "inner")

# ---- IP+Service: first K (ip, service_port, source_file) ----
ip_service_keys = (
    df_ip_part
    .select("ip", "service_port", "source_file")
    .dropDuplicates(["ip", "service_port", "source_file"])
    .orderBy("ip", "service_port", "source_file")
    .limit(MAX_KEYS)
)
df_ip_service_part_sel = df_ip_part.join(
    ip_service_keys, ["ip", "service_port", "source_file"], "inner"
)

# ---- Subnet: first K (subnet, source_file) ----
subnet_keys = (
    df_subnet_part
    .select("subnet", "source_file")
    .dropDuplicates(["subnet", "source_file"])
    .orderBy("subnet", "source_file")
    .limit(MAX_KEYS)
)
df_subnet_part_sel = df_subnet_part.join(
    subnet_keys, ["subnet", "source_file"], "inner"
)

spark.conf.set("spark.sql.shuffle.partitions", num_tasks * 4)

df_ip_part_sel            = df_ip_part_sel.repartition("ip", "source_file")
df_ip_service_part_sel    = df_ip_service_part_sel.repartition("ip", "service_port", "source_file")
df_subnet_part_sel        = df_subnet_part_sel.repartition("subnet", "source_file")
# ----------------------------
# Utils
# ----------------------------
def pad_or_truncate(arr, target_len):
    cur_len = len(arr)
    if cur_len > target_len:
        return arr[:target_len]
    if cur_len < target_len:
        return np.pad(arr, (0, target_len - cur_len), constant_values=-1)
    return arr

def arrays_from_sum_by_window_dir_dense(sum_by_window, thresh_in, thresh_out, max_len):
    """
    Build dense inbound/outbound arrays with zeros in missing bins,
    but clamp the span to at most `max_len` bins starting at the first active bin.
    This avoids iterating over huge inactive ranges while preserving dense semantics.
    Returns: in_arr, out_arr, kept_in_count, kept_out_count
    """
    if not sum_by_window:
        return None, None, 0, 0

    # active range by observed bins
    bins = list(sum_by_window.keys())
    min_b = min(bins)
    max_b = max(bins)

    # clamp to at most max_len bins (we used to build full span then truncate; this saves work)
    end_b = min(max_b, min_b + max_len - 1)
    span = end_b - min_b + 1

    in_arr  = np.zeros(span, dtype=float)
    out_arr = np.zeros(span, dtype=float)

    # only touch bins we actually observed and that fall into the clamped range
    for b, (ti, to) in sum_by_window.items():
        if b < min_b or b > end_b:
            continue
        idx = b - min_b
        if ti >= thresh_in:
            in_arr[idx] = ti  # else stays 0.0
        # outbound independent
        if to >= thresh_out:
            out_arr[idx] = to  # else stays 0.0

    kept_in_count  = int((in_arr  > 0.0).sum())
    kept_out_count = int((out_arr > 0.0).sum())
    return in_arr, out_arr, kept_in_count, kept_out_count

# ----------------------------
# Builders (IP-level)
# ----------------------------
def build_ip_timeseries(ip, src, rows):
    # Sum across ALL service ports per *aggregated* bin
    sum_by_window = {}
    for r in rows:
        ti = float(r.total_inbound_bytes)
        to = float(r.total_outbound_bytes)
        # keep 100ms entries even if one direction 0; aggregation happens at bin level
        w = int(r.window_start)          # 100ms ticks
        b = w // AGG_FACTOR             # aggregated bin index
        if b not in sum_by_window:
            sum_by_window[b] = [0.0, 0.0]
        sum_by_window[b][0] += ti
        sum_by_window[b][1] += to

    in_arr, out_arr, L_in, L_out = arrays_from_sum_by_window_dir_dense(
        sum_by_window, THRESH_IN_BYTES, THRESH_OUT_BYTES, MAX_SEQ_LEN
    )

    # Drop only if neither direction has enough data
    if max(L_in, L_out) < MIN_SEQ_LEN:
        return None

    in_arr  = pad_or_truncate(in_arr,  min(L_in,  MAX_SEQ_LEN))
    out_arr = pad_or_truncate(out_arr, min(L_out, MAX_SEQ_LEN))

    return {
        "ip": ip,
        "source_file": src,
        "inbound":  in_arr.tolist(),
        "outbound": out_arr.tolist(),
    }

def build_ip_service_timeseries(ip, sp, src, rows):
    # Sum per (ip,service) within *aggregated* bins
    sum_by_window = {}
    for r in rows:
        ti = float(r.total_inbound_bytes)
        to = float(r.total_outbound_bytes)
        w  = int(r.window_start)
        b  = w // AGG_FACTOR
        if b not in sum_by_window:
            sum_by_window[b] = [0.0, 0.0]
        sum_by_window[b][0] += ti
        sum_by_window[b][1] += to

    in_arr, out_arr, L_in, L_out = arrays_from_sum_by_window_dir_dense(
        sum_by_window, THRESH_IN_BYTES, THRESH_OUT_BYTES, MAX_SEQ_LEN
    )

    if max(L_in, L_out) < MIN_SEQ_LEN:
        return None

    in_arr  = pad_or_truncate(in_arr,  min(L_in,  MAX_SEQ_LEN))
    out_arr = pad_or_truncate(out_arr, min(L_out, MAX_SEQ_LEN))

    return {
        "ip": ip,
        "service_port": sp,
        "source_file": src,
        "inbound":  in_arr.tolist(),
        "outbound": out_arr.tolist(),
    }

# ----------------------------
# Builders (Subnet-level, /24)
# ----------------------------
def build_subnet_timeseries(subnet, src, rows):
    # Sum across ALL IPs & ports within *aggregated* bins
    sum_by_window = {}
    for r in rows:
        ti = float(r.total_inbound_bytes)
        to = float(r.total_outbound_bytes)
        w  = int(r.window_start)
        b  = w // AGG_FACTOR
        if b not in sum_by_window:
            sum_by_window[b] = [0.0, 0.0]
        sum_by_window[b][0] += ti
        sum_by_window[b][1] += to

    in_arr, out_arr, L_in, L_out = arrays_from_sum_by_window_dir_dense(
        sum_by_window, THRESH_IN_BYTES, THRESH_OUT_BYTES, MAX_SEQ_LEN
    )

    if max(L_in, L_out) < MIN_SEQ_LEN:
        return None

    in_arr  = pad_or_truncate(in_arr,  min(L_in,  MAX_SEQ_LEN))
    out_arr = pad_or_truncate(out_arr, min(L_out, MAX_SEQ_LEN))

    return {
        "subnet": subnet,
        "source_file": src,
        "inbound":  in_arr.tolist(),
        "outbound": out_arr.tolist(),
    }

# ----------------------------
# RDDs (IP)
# ----------------------------
def ip_partitions(iter_rows):
    rows = list(iter_rows)
    if not rows:
        return iter(())
    rows.sort(key=lambda r: (r.ip, r.source_file, r.window_start, r.service_port))
    for (ip, src), grp in groupby(rows, key=lambda r: (r.ip, r.source_file)):
        res = build_ip_timeseries(ip, src, list(grp))
        if res is not None:
            yield res

def ip_service_partitions(iter_rows):
    rows = list(iter_rows)
    if not rows:
        return iter(())
    rows.sort(key=lambda r: (r.ip, r.service_port, r.source_file, r.window_start))
    for (ip, sp, src), grp in groupby(rows, key=lambda r: (r.ip, r.service_port, r.source_file)):
        res = build_ip_service_timeseries(ip, sp, src, list(grp))
        if res is not None:
            yield res

# ----------------------------
# RDDs (Subnet)
# ----------------------------
def subnet_partitions(iter_rows):
    rows = list(iter_rows)
    if not rows:
        return iter(())
    rows.sort(key=lambda r: (r.subnet, r.source_file, r.window_start, r.service_port, r.ip))
    for (subnet, src), grp in groupby(rows, key=lambda r: (r.subnet, r.source_file)):
        res = build_subnet_timeseries(subnet, src, list(grp))
        if res is not None:
            yield res

ip_rdd         = df_ip_part_sel.rdd.mapPartitions(ip_partitions)
ip_service_rdd = df_ip_service_part_sel.rdd.mapPartitions(ip_service_partitions)
subnet_rdd     = df_subnet_part_sel.rdd.mapPartitions(subnet_partitions)

# ----------------------------
# Schemas & Save
# ----------------------------
ip_schema = StructType([
    StructField("ip", StringType(), True),
    StructField("source_file", StringType(), True),
    StructField("inbound",  ArrayType(FloatType()), True),
    StructField("outbound", ArrayType(FloatType()), True),
])

ip_service_schema = StructType([
    StructField("ip", StringType(), True),
    StructField("service_port", LongType(), True),
    StructField("source_file", StringType(), True),
    StructField("inbound",  ArrayType(FloatType()), True),
    StructField("outbound", ArrayType(FloatType()), True),
])

subnet_schema = StructType([
    StructField("subnet", StringType(), True),
    StructField("source_file", StringType(), True),
    StructField("inbound",  ArrayType(FloatType()), True),
    StructField("outbound", ArrayType(FloatType()), True),
])

ip_df         = spark.createDataFrame(ip_rdd, ip_schema)
ip_service_df = spark.createDataFrame(ip_service_rdd, ip_service_schema)
subnet_df     = spark.createDataFrame(subnet_rdd, subnet_schema)

ip_df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}_ip/task_{task_rank}")
ip_service_df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}_service/task_{task_rank}")
subnet_df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}_subnet/task_{task_rank}")

print(f"[Task {task_rank}] Wrote IP, IP+Service, Subnet(/24) datasets.")
