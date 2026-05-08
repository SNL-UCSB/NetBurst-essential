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
# Config
# ----------------------------
FINETUNING  = False
MIN_SEQ_LEN = 10
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
parser.add_argument("--compute_quantiles", action="store_true",
    help="Compute and store 10 (or --quantile_levels) reference quantiles per entity type over sparse values (including zeros)")
parser.add_argument("--quantile_levels", type=str,
    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    help="Comma-separated quantile levels for reference (default: 10 evenly spaced)")
parser.add_argument("--pcap_parquet_dir", type=str, default=None,
    help="Comma-separated list of directories; each is a PcapToDf_multi_node output. Read from these and build 100ms aggregate. Each folder is treated as a separate source (same IP in different folders is not merged). Ignores finetuning path.")
parser.add_argument("--quantiles_only", action="store_true",
    help="Skip sparse creation; only compute quantiles from existing sparse output (--output_dir must point to existing sparse dir with _ip, _service, _subnet, _ip2ip). Use when sparse was already written but quantile step failed or was skipped.")
args = parser.parse_args()

BIN_MS            = args.bin_ms
OUTPUT_DIR        = args.output_dir
FINETUNING        = args.finetuning
THRESH_IN_BYTES   = float(args.threshold)
THRESH_OUT_BYTES  = float(args.threshold)

if BIN_MS < 100 or BIN_MS % 100 != 0:
    raise ValueError(f"BIN_MS must be >=100 and a multiple of 100; got {BIN_MS}")

# ----------------------------
# Quantiles-only mode: compute quantiles from existing sparse output then exit
# ----------------------------
if getattr(args, "quantiles_only", False):
    _task_rank = int(os.getenv("SLURM_PROCID", "0"))
    _spark = SparkSession.builder \
        .appName("PySpark Quantiles Only") \
        .config("spark.driver.memory", "32g") \
        .getOrCreate()
    _out = args.output_dir
    import sys
    _preprocess_dir = os.path.dirname(os.path.abspath(__file__))
    if _preprocess_dir not in sys.path:
        sys.path.insert(0, _preprocess_dir)
    from quantiles_utils import (
        compute_quantiles_from_array_column,
        write_quantiles_parquet_multi,
        DEFAULT_QUANTILE_LEVELS,
    )
    _levels_str = getattr(args, "quantile_levels", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    _levels_list = [float(x.strip()) for x in _levels_str.split(",") if x.strip()]
    if not _levels_list:
        _levels_list = list(DEFAULT_QUANTILE_LEVELS)
    _rows = []
    for _entity_type, _path_suffix in [
        ("ip", f"{_out}_ip"),
        ("ip_service", f"{_out}_service"),
        ("subnet", f"{_out}_subnet"),
        ("ip2ip", f"{_out}_ip2ip"),
    ]:
        _path_glob = _path_suffix.rstrip("/") + "/task_*"
        _df_all = _spark.read.parquet(_path_glob)
        _levels, _quantiles = compute_quantiles_from_array_column(
            _df_all, "inbound", levels=_levels_list
        )
        _rows.append((_entity_type, _levels, _quantiles))
    write_quantiles_parquet_multi(_spark, _rows, f"{_out}_quantiles")
    print(f"[Quantiles-only] Wrote quantiles for ip, ip_service, subnet, ip2ip to {_out}_quantiles")
    sys.exit(0)

AGG_FACTOR = BIN_MS // 100
print(f"Aggregating 100ms ticks into {BIN_MS}ms bins (factor={AGG_FACTOR}).")
print(f"Finetuning: {FINETUNING}")
print(f"Inbound threshold:  {THRESH_IN_BYTES} bytes/bin")
print(f"Outbound threshold: {THRESH_OUT_BYTES} bytes/bin")

# ----------------------------
# Load
# ----------------------------
if args.pcap_parquet_dir:
    # List of folders (each = PcapToDf_multi_node output). Treat each folder as a separate source so IPs are not merged across folders.
    pcap_dirs = [d.strip() for d in args.pcap_parquet_dir.split(",") if d.strip()]
    if not pcap_dirs:
        raise ValueError("--pcap_parquet_dir must contain at least one directory path")
    # Read each folder separately and tag with that folder path as source_file (no cross-folder mixing)
    packet_dfs = []
    for folder in pcap_dirs:
        df = spark.read.parquet(folder)
        # Use folder path as source_file so timeseries are per-folder; overwrite if Parquet had no or different source_file
        df = df.withColumn("source_file", F.lit(folder))
        packet_dfs.append(df)
    packets = packet_dfs[0]
    for df in packet_dfs[1:]:
        packets = packets.unionByName(df)
    packets = packets.filter(~col("src_ip").startswith("127.")).filter(~col("dst_ip").startswith("127."))

    # 100ms tick index (timestamp in seconds epoch)
    packets = packets.withColumn("window_start", (F.col("timestamp") * 10).cast("long"))
    src_port_ = F.coalesce(col("src_port"), F.lit(0))
    dst_port_ = F.coalesce(col("dst_port"), F.lit(0))
    packets = packets.withColumn("service_port", F.least(src_port_, dst_port_))
    packets = packets.withColumn(
        "outbound",
        (F.coalesce(col("dst_port"), F.lit(-1)) == col("service_port"))
    )
    packets = packets.withColumn(
        "inbound_bytes",
        F.when(~col("outbound"), F.col("size").cast("long")).otherwise(0)
    )
    packets = packets.withColumn(
        "outbound_bytes",
        F.when(col("outbound"), F.col("size").cast("long")).otherwise(0)
    )

    # Per-IP aggregate (union src and dst views, then group by ip)
    src_df = packets.select(
        "window_start",
        F.col("src_ip").alias("ip"),
        "service_port",
        "inbound_bytes",
        "outbound_bytes",
        "source_file",
    )
    dst_df = packets.select(
        "window_start",
        F.col("dst_ip").alias("ip"),
        "service_port",
        F.col("outbound_bytes").alias("inbound_bytes"),
        F.col("inbound_bytes").alias("outbound_bytes"),
        "source_file",
    )
    combined = src_df.union(dst_df)
    agg_ip = combined.groupBy("window_start", "ip", "service_port", "source_file").agg(
        F.sum("inbound_bytes").alias("total_inbound_bytes"),
        F.sum("outbound_bytes").alias("total_outbound_bytes"),
    )
    octets_ip = split(col("ip"), "\\.")
    agg_ip = agg_ip.withColumn(
        "subnet", concat_ws(".", octets_ip.getItem(0), octets_ip.getItem(1), octets_ip.getItem(2))
    )
    agg_ip = agg_ip.withColumn("src_ip", F.lit(None).cast("string"))
    agg_ip = agg_ip.withColumn("dst_ip", F.lit(None).cast("string"))

    # IP-to-IP aggregate (one row per window_start, src_ip, dst_ip, source_file)
    agg_ip2ip = packets.groupBy("window_start", "src_ip", "dst_ip", "source_file").agg(
        F.sum("size").cast("long").alias("total_inbound_bytes"),
        F.lit(0).cast("long").alias("total_outbound_bytes"),
    )
    agg_ip2ip = agg_ip2ip.withColumn("ip", F.lit(None).cast("string"))
    agg_ip2ip = agg_ip2ip.withColumn("service_port", F.lit(None).cast("long"))
    agg_ip2ip = agg_ip2ip.withColumn("subnet", F.lit(None).cast("string"))

    # Union so downstream sees one table with both row types (same column set on both sides)
    aggregated_df = agg_ip.unionByName(agg_ip2ip)

    # Hashes for distribution (only where key is present)
    aggregated_df = aggregated_df.withColumn(
        "last_octet",
        F.when(col("ip").isNotNull(), split(col("ip"), "\\.").getItem(3).cast("int"))
    )
    aggregated_df = aggregated_df.withColumn(
        "ip_hash",
        F.when(col("ip").isNotNull(), col("last_octet") % num_tasks)
    )
    aggregated_df = aggregated_df.withColumn(
        "subnet_hash",
        F.when(col("subnet").isNotNull(), F.pmod(F.xxhash64("subnet"), F.lit(num_tasks)))
    )
    aggregated_df = aggregated_df.withColumn(
        "src_ip_last_octet",
        F.when(col("src_ip").isNotNull(), split(col("src_ip"), "\\.").getItem(3).cast("int"))
    )
    aggregated_df = aggregated_df.withColumn(
        "src_ip_hash",
        F.when(col("src_ip").isNotNull(), col("src_ip_last_octet") % num_tasks)
    )

    df_ip_part = aggregated_df.filter(col("ip_hash") == task_rank)
    df_subnet_part = aggregated_df.filter(col("subnet_hash") == task_rank)
    df_ip2ip_part = aggregated_df.filter(col("src_ip_hash") == task_rank)
    print(f"[Task {task_rank}] Built 100ms aggregate from PcapToDf output ({len(pcap_dirs)} folder(s)): {pcap_dirs}")
else:
    if not FINETUNING:
        aggregated_df = spark.read.parquet("<data-root>/service_100ms_aggregate_13dfs.parquet")
    else:
        aggregated_df = spark.read.parquet("<data-root>/mawi_aggregate_100ms")

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

    # Distribute by task (IP-to-IP, shard by src_ip)
    src_octets = split(col("src_ip"), "\\.")
    aggregated_df = aggregated_df.withColumn("src_ip_last_octet", src_octets.getItem(3).cast("int"))
    aggregated_df = aggregated_df.withColumn("src_ip_hash", col("src_ip_last_octet") % num_tasks)
    df_ip2ip_part = aggregated_df.filter(col("src_ip_hash") == task_rank)

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

# ---- IP-to-IP: first K (src_ip, dst_ip, source_file) ----
ip2ip_keys = (
    df_ip2ip_part
    .select("src_ip", "dst_ip", "source_file")
    .dropDuplicates(["src_ip", "dst_ip", "source_file"])
    .orderBy("src_ip", "dst_ip", "source_file")
    .limit(MAX_KEYS)
)
df_ip2ip_part_sel = df_ip2ip_part.join(
    ip2ip_keys, ["src_ip", "dst_ip", "source_file"], "inner"
)

spark.conf.set("spark.sql.shuffle.partitions", num_tasks * 4)

df_ip_part_sel            = df_ip_part_sel.repartition("ip", "source_file")
df_ip_service_part_sel    = df_ip_service_part_sel.repartition("ip", "service_port", "source_file")
df_subnet_part_sel        = df_subnet_part_sel.repartition("subnet", "source_file")
df_ip2ip_part_sel         = df_ip2ip_part_sel.repartition("src_ip", "dst_ip", "source_file")
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
# Builders (IP-to-IP, srcIP -> dstIP)
# ----------------------------
def build_ip2ip_timeseries(src_ip, dst_ip, src, rows):
    # Sum per (src_ip, dst_ip) within *aggregated* bins
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
        "src_ip": src_ip,
        "dst_ip": dst_ip,
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

def ip2ip_partitions(iter_rows):
    rows = list(iter_rows)
    if not rows:
        return iter(())
    rows.sort(key=lambda r: (r.src_ip, r.dst_ip, r.source_file, r.window_start, r.service_port))
    for (src_ip, dst_ip, src), grp in groupby(rows, key=lambda r: (r.src_ip, r.dst_ip, r.source_file)):
        res = build_ip2ip_timeseries(src_ip, dst_ip, src, list(grp))
        if res is not None:
            yield res

ip_rdd         = df_ip_part_sel.rdd.mapPartitions(ip_partitions)
ip_service_rdd = df_ip_service_part_sel.rdd.mapPartitions(ip_service_partitions)
subnet_rdd     = df_subnet_part_sel.rdd.mapPartitions(subnet_partitions)
ip2ip_rdd      = df_ip2ip_part_sel.rdd.mapPartitions(ip2ip_partitions)

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

ip2ip_schema = StructType([
    StructField("src_ip", StringType(), True),
    StructField("dst_ip", StringType(), True),
    StructField("source_file", StringType(), True),
    StructField("inbound",  ArrayType(FloatType()), True),
    StructField("outbound", ArrayType(FloatType()), True),
])

ip_df         = spark.createDataFrame(ip_rdd, ip_schema)
ip_service_df = spark.createDataFrame(ip_service_rdd, ip_service_schema)
subnet_df     = spark.createDataFrame(subnet_rdd, subnet_schema)
ip2ip_df      = spark.createDataFrame(ip2ip_rdd, ip2ip_schema)

ip_df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}_ip/task_{task_rank}")
ip_service_df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}_service/task_{task_rank}")
subnet_df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}_subnet/task_{task_rank}")
ip2ip_df.write.mode("overwrite").parquet(f"{OUTPUT_DIR}_ip2ip/task_{task_rank}")

print(f"[Task {task_rank}] Wrote IP, IP+Service, Subnet(/24), and IP-to-IP datasets.")

# ----------------------------
# Optional: compute and store quantiles per entity type (task 0 only)
# ----------------------------
if getattr(args, "compute_quantiles", False) and task_rank == 0:
    import sys
    import os as _os
    _preprocess_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _preprocess_dir not in sys.path:
        sys.path.insert(0, _preprocess_dir)
    from quantiles_utils import (
        compute_quantiles_from_array_column,
        write_quantiles_parquet_multi,
        DEFAULT_QUANTILE_LEVELS,
    )
    quantile_levels_str = getattr(args, "quantile_levels", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    levels_list = [float(x.strip()) for x in quantile_levels_str.split(",") if x.strip()]
    if not levels_list:
        levels_list = list(DEFAULT_QUANTILE_LEVELS)
    rows = []
    for entity_type, path_suffix in [
        ("ip", f"{OUTPUT_DIR}_ip"),
        ("ip_service", f"{OUTPUT_DIR}_service"),
        ("subnet", f"{OUTPUT_DIR}_subnet"),
        ("ip2ip", f"{OUTPUT_DIR}_ip2ip"),
    ]:
        # Recursive read: parquet files live under task_0, task_1, ... subdirs
        path_glob = path_suffix.rstrip("/") + "/task_*"
        df_all = spark.read.parquet(path_glob)
        levels, quantiles = compute_quantiles_from_array_column(
            df_all, "inbound", levels=levels_list
        )
        rows.append((entity_type, levels, quantiles))
    write_quantiles_parquet_multi(spark, rows, f"{OUTPUT_DIR}_quantiles")
    print(f"[Task 0] Wrote quantiles for entity types ip, ip_service, subnet, ip2ip to {OUTPUT_DIR}_quantiles")
