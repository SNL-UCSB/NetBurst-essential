"""
Prepare RnE-NetFlow sparse parquet for SparseToIBGBIFromSparse.py.

Renames meta_dst_ip -> ip (to match NetBurst ENTITY_KEYS convention) and adds a
constant source_file column (required by SparseToIBGBIFromSparse and
create_train_test_splits). Reads from task_* shards under the input dir and writes
to task_* shards under the output dir, preserving the sharding layout.

Usage:
    python3 PrepareRnENetFlowSparse.py \
        --input_dir  /path/to/sparse_out \
        --output_dir /path/to/sparse_out_prepared \
        --source_file_value rne_netflow
"""
import argparse
import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def main():
    parser = argparse.ArgumentParser(description="Prepare RnE-NetFlow sparse parquet for NetBurst IBG/BI pipeline")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input sparse parquet dir (contains task_* subdirs)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir; task_* subdirs will be written here")
    parser.add_argument("--source_file_value", type=str, default="rne_netflow",
                        help="Constant value to use for the source_file column (default: rne_netflow)")
    parser.add_argument("--ip_col", type=str, default="meta_dst_ip",
                        help="Name of the IP column in the input parquet (default: meta_dst_ip)")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("PrepareRnENetFlowSparse")
        .config("spark.driver.memory", "64g")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )

    input_glob = os.path.join(args.input_dir.rstrip("/"), "task_*")
    df = spark.read.parquet(input_glob)

    if args.ip_col not in df.columns:
        raise ValueError(f"Column '{args.ip_col}' not found in {args.input_dir}. Columns: {df.columns}")

    df = df.withColumnRenamed(args.ip_col, "ip")

    if "source_file" not in df.columns:
        if "session_id" in df.columns:
            df = df.withColumn("source_file", F.col("session_id").cast("string"))
        else:
            df = df.withColumn("source_file", F.lit(args.source_file_value))

    task_rank = int(os.getenv("SLURM_PROCID", "0"))
    num_tasks = int(os.getenv("SLURM_NTASKS", "1"))

    if num_tasks > 1:
        key_cols = [c for c in df.columns if c not in ("inbound", "outbound", "source_file")]
        h = F.pmod(F.xxhash64(*[F.col(c).cast("string") for c in key_cols + ["source_file"]]),
                   F.lit(num_tasks))
        df = df.withColumn("_shard", h).filter(F.col("_shard") == F.lit(task_rank)).drop("_shard")

    out_path = os.path.join(args.output_dir, f"task_{task_rank}")
    df.write.mode("overwrite").option("compression", "zstd").parquet(out_path)
    print(f"[Task {task_rank}/{num_tasks}] Wrote prepared sparse to {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()
