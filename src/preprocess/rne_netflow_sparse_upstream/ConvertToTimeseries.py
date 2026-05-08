
import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window




# Parse input arguments
parser = argparse.ArgumentParser(description="Convert flow edge hits to timeseries.")
parser.add_argument('--flow_edge_hits_file', type=str, required=True, help='Path to the flow edge hits parquet file')
parser.add_argument('--output_parquet', type=str, required=True, help='Output directory root (writes task_{id}/)')
parser.add_argument('--num_tasks', type=int, default=1, help='Total number of hash shards (default: 1)')
parser.add_argument('--task_id', type=int, default=0, help='Shard id for this task (0..num_tasks-1)')
parser.add_argument('--shuffle_partitions', type=int, default=2000, help='spark.sql.shuffle.partitions')
args = parser.parse_args()


# 1. Create your Spark session
spark = SparkSession.builder \
    .appName(f"Convert to timeseries") \
    .config("spark.driver.memory", "200g") \
    .config("spark.executor.memory", "200g") \
    .config("spark.driver.maxResultSize", "200g") \
  .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions)) \
    .getOrCreate()

# Helper to normalize column names (replace dots and @ symbols)
def normalize_name(name: str) -> str:
    return name.replace(".", "_").replace("@", "")

# 2. Read the Parquet file
df = spark.read.parquet(args.flow_edge_hits_file)

# Normalize all column names (replace dots with underscores, @ with nothing)
for old in df.columns:
    new = normalize_name(old)
    if new != old:
        df = df.withColumnRenamed(old, new)

# Normalize timestamp text for Spark 3+ parser compatibility.
# Handles values like:
# - 2026-03-23T22:18:56.000338Z
# - 2026-03-23T22:18:56.679874+00:00
df = df.withColumn(
  "_ts_norm",
  F.regexp_replace(
    F.col("timestamp").cast("string"),
    r"(\.\d{6})\d+(Z|[+-]\d\d:\d\d)$",
    r"$1$2",
  ),
)

df = df.withColumn(
  "ingest_ts",
  F.coalesce(
    F.to_timestamp(F.col("_ts_norm"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX"),
    F.to_timestamp(F.col("_ts_norm"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSSX"),
    F.to_timestamp(F.col("_ts_norm")),
  ),
)

df = df.drop("_ts_norm")

print(f"[DEBUG] Input data: {df.count()} rows")
print(f"[DEBUG] Columns: {df.columns}")
print(f"[DEBUG] meta_five_tuple_id nulls: {df.filter(F.col('meta_five_tuple_id').isNull()).count()}")
print(f"[DEBUG] values_num_bits nulls: {df.filter(F.col('values_num_bits').isNull()).count()}")
print(f"[DEBUG] ingest_ts nulls (parse failures): {df.filter(F.col('ingest_ts').isNull()).count()}")

# Ensure values_num_bits exists and is numeric for aggregation
if "values_num_bits" not in df.columns:
  raise ValueError("Missing required column: values_num_bits")

df = df.withColumn("values_num_bits_num", F.col("values_num_bits").cast("double"))
print(f"[DEBUG] values_num_bits_num nulls after cast: {df.filter(F.col('values_num_bits_num').isNull()).count()}")

# 3. Service-level grouping key: destination IP + service port.
# Service port is defined as min(src_port, dst_port).
required_service_cols = ["meta_dst_ip", "meta_src_port", "meta_dst_port"]
missing_service_cols = [c for c in required_service_cols if c not in df.columns]
if missing_service_cols:
  raise ValueError(f"Missing required columns for service grouping: {missing_service_cols}")

df = df.withColumn(
  "service_port",
  F.when(
    F.col("meta_src_port").isNull() & F.col("meta_dst_port").isNull(),
    F.lit(None).cast("long"),
  ).otherwise(
    F.least(F.col("meta_src_port").cast("long"), F.col("meta_dst_port").cast("long"))
  ),
)

df = df.filter(
  F.col("meta_dst_ip").isNotNull()
  & F.col("service_port").isNotNull()
  & F.col("ingest_ts").isNotNull()
)

# Optional sharding by hash(meta_dst_ip, service_port, source_file?)
hash_cols = [F.col("meta_dst_ip"), F.col("service_port")]
if "source_file" in df.columns:
  hash_cols.append(F.col("source_file"))
if args.num_tasks > 1:
  shard_col = F.pmod(F.abs(F.xxhash64(*hash_cols)), F.lit(args.num_tasks)).alias("_shard")
  df = df.withColumn("_shard", shard_col).filter(F.col("_shard") == F.lit(args.task_id)).drop("_shard")

group_keys = ["meta_dst_ip", "service_port"]
print(f"[DEBUG] Using group_keys: {group_keys}")
print(f"[DEBUG] Rows after filters (+shard): {df.count()}")

df = df.repartition(*[F.col(k) for k in group_keys])

# Compute gaps and session ids using windows directly on the base rows
w = Window.partitionBy(*group_keys).orderBy(F.col("ingest_ts"))
with_gap = df.withColumn("prev_ts", F.lag("ingest_ts").over(w)) \
            .withColumn("gap_secs", (F.col("ingest_ts").cast("long") - F.col("prev_ts").cast("long")))

flagged = with_gap.withColumn("is_new_session", F.when(F.col("gap_secs") > 300, 1).otherwise(0))

w2 = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
sessioned = flagged.withColumn("session_id", F.sum("is_new_session").over(w2))

timeseries = (
    sessioned
      .groupBy(
        *group_keys,
        "session_id",
        F.window("ingest_ts", "60 seconds")
      )
      .agg(
        F.sum(F.coalesce(F.col("values_num_bits_num").cast("double"), F.lit(0.0))).alias("sum_num_bits")
      )
      .select(
        *[F.col(k) for k in group_keys],
        "session_id",
        F.col("window.start").alias("window_start"),
        "sum_num_bits"
      )
)
print(f"[DEBUG] Timeseries (before final filter): {timeseries.count()} windows")

# 2) Count how many windows each session has
session_window_counts = (
    timeseries
  .groupBy(*group_keys, "session_id")
      .agg(F.count("*").alias("num_windows"))
      .filter(F.col("num_windows") >= 5)    # keep only sessions with >=5 windows
  .select(*group_keys, "session_id")
)
print(f"[DEBUG] Sessions with >= 5 windows: {session_window_counts.count()}")

# 3) Inner-join back to keep only those sessions' timeseries
final_ts = timeseries.join(
    session_window_counts,
  on=group_keys + ["session_id"],
    how="inner"
)
print(f"[DEBUG] Final timeseries (after join): {final_ts.count()} rows")

# 4) Persist to Parquet
out_dir = os.path.join(args.output_parquet, f"task_{args.task_id}")
final_ts.write.mode("overwrite").parquet(out_dir)

spark.stop()
