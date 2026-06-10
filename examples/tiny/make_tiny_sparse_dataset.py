#!/usr/bin/env python3
"""
Create a tiny sparse parquet dataset for smoke tests.

Input is expected to be a sparse dataset laid out under task_* directories.
The output keeps the same task_0 layout so preprocess scripts can read it.
"""

from __future__ import annotations

import argparse
import json
import os

from pyspark.sql import SparkSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny sparse parquet subset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="<data-root>/sparse_timeseries_1s_ip",
        help="Source sparse dataset directory (expects task_* parquet shards)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/tiny/data/sparse_timeseries_1s_ip_tiny",
        help="Output directory for tiny sparse dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of rows to keep in the tiny dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.builder.appName("netburst_tiny_sparse_builder")
        .config("spark.driver.memory", "16g")
        .config("spark.sql.parquet.compression.codec", "zstd")
        .getOrCreate()
    )

    input_glob = os.path.join(args.input_dir.rstrip("/"), "task_*")
    df = spark.read.parquet(input_glob)
    total_rows = df.count()
    tiny_rows = min(int(args.num_samples), total_rows)

    tiny_df = df.limit(tiny_rows)
    tiny_task_dir = os.path.join(args.output_dir, "task_0")
    tiny_df.write.mode("overwrite").option("compression", "zstd").parquet(tiny_task_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "tiny_dataset_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_input_dir": args.input_dir,
                "source_input_glob": input_glob,
                "requested_num_samples": int(args.num_samples),
                "written_num_samples": int(tiny_rows),
                "source_total_rows": int(total_rows),
                "tiny_task_dir": tiny_task_dir,
            },
            f,
            indent=2,
        )

    print(
        f"Wrote tiny sparse dataset with {tiny_rows} rows to {tiny_task_dir} "
        f"(source rows={total_rows})."
    )
    print(f"Wrote metadata to {metadata_path}")
    spark.stop()


if __name__ == "__main__":
    main()
