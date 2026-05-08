"""
Flow-level Parquet -> 100ms-binned aggregates for NonGraphSparseTimeseries.py.

Input schema (per row): src_ip, dst_ip, src_port, dst_port, timestamp, size
  timestamp: Unix seconds (float or long).

Output schema: window_start, ip, service_port, total_inbound_bytes,
  total_outbound_bytes, source_file

window_start is floor(timestamp_ms / window_ms), i.e. bin index in units of window_ms
(NonGraphSparseTimeseries treats these as 100ms ticks when window_ms=100).

By default, service_port is replaced by dense_rank within each IP (legacy behavior).
Use --no-rank-service-ports to keep the numeric service port.
"""

import argparse
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, dense_rank, input_file_name, when


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate flow Parquet to windowed bytes per ip/service.")
    p.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Parquet path(s), comma-separated (dirs or files).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for aggregated Parquet (Spark write).",
    )
    p.add_argument(
        "--window_ms",
        type=int,
        default=100,
        help="Window size in milliseconds (100 => same ticks as NonGraphSparseTimeseries).",
    )
    p.add_argument(
        "--no-rank-service-ports",
        action="store_true",
        help="Keep literal service port. Default: rank ports by total bytes per IP (legacy).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank_service_ports = not args.no_rank_service_ports
    window_ms = int(args.window_ms)
    if window_ms < 1:
        raise ValueError("--window_ms must be >= 1")

    input_dirs = [x.strip() for x in str(args.input_dir).split(",") if x.strip()]

    spark = (
        SparkSession.builder.appName("AggregateToWindowSize")
        .config("spark.driver.memory", "100g")
        .config("spark.driver.maxResultSize", "16g")
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "200"))
        .getOrCreate()
    )

    df = spark.read.parquet(*input_dirs)
    df = df.withColumn("source_file", input_file_name())
    df = df.withColumn(
        "service_port",
        when(col("src_port") < col("dst_port"), col("src_port")).otherwise(col("dst_port")),
    )
    df = df.withColumn("outbound", col("service_port") == col("dst_port"))

    windowed_df = df.withColumn(
        "timestamp_ms",
        (F.col("timestamp").cast("double") * F.lit(1000.0)).cast("long"),
    ).withColumn(
        "window_start",
        F.floor(F.col("timestamp_ms") / F.lit(float(window_ms))).cast("long"),
    )

    bytes_df = windowed_df.withColumn(
        "inbound_bytes",
        F.when(F.col("outbound") == F.lit(False), F.col("size")).otherwise(F.lit(0)),
    ).withColumn(
        "outbound_bytes",
        F.when(F.col("outbound") == F.lit(True), F.col("size")).otherwise(F.lit(0)),
    )

    src_df = bytes_df.select(
        "window_start",
        F.col("src_ip").alias("ip"),
        "service_port",
        "inbound_bytes",
        "outbound_bytes",
        "source_file",
    )
    dst_df = bytes_df.select(
        "window_start",
        F.col("dst_ip").alias("ip"),
        "service_port",
        F.col("outbound_bytes").alias("inbound_bytes"),
        F.col("inbound_bytes").alias("outbound_bytes"),
        "source_file",
    )
    combined_df = src_df.union(dst_df)

    aggregated_df = combined_df.groupBy(
        "window_start", "ip", "service_port", "source_file"
    ).agg(
        F.sum("inbound_bytes").alias("total_inbound_bytes"),
        F.sum("outbound_bytes").alias("total_outbound_bytes"),
    )

    if rank_service_ports:
        df_with_total_bytes = aggregated_df.withColumn(
            "total_bytes",
            col("total_inbound_bytes") + col("total_outbound_bytes"),
        )
        window_spec = Window.partitionBy("ip").orderBy(col("total_bytes").desc())
        df_with_rank = df_with_total_bytes.withColumn(
            "service_port_rank",
            dense_rank().over(window_spec),
        )
        df_final = (
            df_with_rank.drop("service_port", "total_bytes")
            .withColumnRenamed("service_port_rank", "service_port")
        )
    else:
        df_final = aggregated_df

    df_final = df_final.select(
        "window_start",
        "ip",
        "service_port",
        "total_inbound_bytes",
        "total_outbound_bytes",
        "source_file",
    )

    out = args.output_dir
    df_final.write.mode("overwrite").parquet(out)
    print(f"Wrote aggregated Parquet to {out} (window_ms={window_ms}, rank_ports={rank_service_ports})")


if __name__ == "__main__":
    main()
