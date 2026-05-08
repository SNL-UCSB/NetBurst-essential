
import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.window import Window


# Parse input arguments
parser = argparse.ArgumentParser(
  description="Convert flow edge hits to timeseries grouped by destination IP (meta_dst_ip)."
)
parser.add_argument("--flow_edge_hits_file", type=str, required=True, help="Path to the flow edge hits parquet file")
parser.add_argument("--output_parquet", type=str, required=True, help="Output directory root (writes task_{id}/)")
parser.add_argument('--num_tasks', type=int, default=1, help='Total number of hash shards (default: 1)')
parser.add_argument('--task_id', type=int, default=0, help='Shard id for this task (0..num_tasks-1)')
parser.add_argument('--shuffle_partitions', type=int, default=2000, help='spark.sql.shuffle.partitions')
args = parser.parse_args()

group_keys = ["meta_dst_ip"]

# 1. Create your Spark session
spark = SparkSession.builder \
    .appName("Convert to timeseries (IP pair)") \
    .config("spark.driver.memory", "200g") \
    .config("spark.executor.memory", "200g") \
    .config("spark.driver.maxResultSize", "200g") \
  .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions)) \
    .getOrCreate()

# 2. Read the Parquet file
df = spark.read.parquet(args.flow_edge_hits_file)

# Starting from your original df
df = reduce(
    lambda current_df, col_name: current_df.withColumnRenamed(
        col_name,
        col_name.replace(".", "_")
    ),
    df.columns,
    df
)

df = df.withColumnRenamed("@timestamp", "ingest_ts")

# Some rows include 7-9 fractional digits (nanoseconds). Spark timestamp parser is stricter,
# so trim any fraction beyond 6 digits before parsing.
df = df.withColumn(
  "ingest_ts_norm",
  F.regexp_replace(F.col("ingest_ts"), r"(\.\d{6})\d+(Z)$", r"$1$2"),
)

df = df.withColumn(
  "ingest_ts",
  F.coalesce(
    F.to_timestamp("ingest_ts_norm", "yyyy-MM-dd'T'HH:mm:ss.SSSSSS'Z'"),
    F.to_timestamp("ingest_ts_norm", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"),
    F.to_timestamp("ingest_ts_norm", "yyyy-MM-dd'T'HH:mm:ss'Z'"),
  ),
).drop("ingest_ts_norm")

missing_ip = [c for c in group_keys if c not in df.columns]
if missing_ip:
    raise ValueError(f"Missing required columns for IP-level grouping: {missing_ip}")

df = df.filter(F.col("meta_dst_ip").isNotNull() & F.col("ingest_ts").isNotNull())

# Optional sharding by hash(meta_dst_ip, source_file?)
hash_cols = [F.col("meta_dst_ip")]
if "source_file" in df.columns:
  hash_cols.append(F.col("source_file"))
if args.num_tasks > 1:
  df = df.withColumn("_shard", F.pmod(F.abs(F.xxhash64(*hash_cols)), F.lit(args.num_tasks))) \
       .filter(F.col("_shard") == F.lit(args.task_id)).drop("_shard")

# 4. Build a list of collect_list expressions for every other column
other_cols = [c for c in df.columns if c not in group_keys and c != "ingest_ts"]

# Build a struct with timestamp first, then the rest of the fields
struct_cols = [F.col("ingest_ts")] + [F.col(c) for c in other_cols]


# 5. Group and aggregate
# Direct windowing without collect_list/posexplode
df = df.repartition(*[F.col(k) for k in group_keys])

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
        F.sum(F.coalesce(F.col("values_num_bits").cast("double"), F.lit(0.0))).alias("sum_num_bits")
      )
      .select(
        *group_keys,
        "session_id",
        F.col("window.start").alias("window_start"),
        "sum_num_bits"
      )
)

session_window_counts = (
    timeseries
      .groupBy(*group_keys, "session_id")
      .agg(F.count("*").alias("num_windows"))
      .filter(F.col("num_windows") >= 10)
      .select(*group_keys, "session_id")
)

final_ts = timeseries.join(
    session_window_counts,
    on=group_keys + ["session_id"],
    how="inner"
)

out_dir = os.path.join(args.output_parquet, f"task_{args.task_id}")
final_ts.write.mode("overwrite").parquet(out_dir)

spark.stop()
