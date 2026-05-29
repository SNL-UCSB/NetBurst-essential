#!/usr/bin/env python3
"""
Create a tiny TSFresh-like numeric feature table for smoke tests.

The output schema is:
  ip, source_file, <numeric feature columns...>

Features are simple statistics derived from sparse inbound/outbound arrays so
analysis/interpretability pipeline wiring can be validated end-to-end.
"""

from __future__ import annotations

import argparse
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiny TSFresh fixture CSV")
    parser.add_argument(
        "--sparse_dir",
        type=str,
        default="examples/tiny/data/sparse_timeseries_1s_ip_tiny",
        help="Tiny sparse parquet directory (expects task_* layout)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="examples/tiny/outputs/full_pipeline/analysis/tiny_tsfresh_features.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def _safe_mean(expr: str) -> F.Column:
    return F.when(F.size(F.col(expr)) > 0, F.expr(f"aggregate({expr}, cast(0.0 as double), (acc, x) -> acc + cast(x as double)) / size({expr})")).otherwise(F.lit(0.0))


def _safe_std(expr: str) -> F.Column:
    # std = sqrt(E[x^2] - E[x]^2) with guards for empty arrays.
    mean_col = _safe_mean(expr)
    mean_sq_col = F.when(
        F.size(F.col(expr)) > 0,
        F.expr(
            f"aggregate({expr}, cast(0.0 as double), (acc, x) -> acc + cast(x as double) * cast(x as double)) / size({expr})"
        ),
    ).otherwise(F.lit(0.0))
    return F.sqrt(F.greatest(F.lit(0.0), mean_sq_col - mean_col * mean_col))


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.builder.appName("netburst_tiny_tsfresh_fixture")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    sparse_glob = os.path.join(args.sparse_dir.rstrip("/"), "task_*")
    df = spark.read.parquet(sparse_glob)

    required = {"ip", "source_file", "inbound"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required sparse columns: {missing}")

    if "outbound" not in df.columns:
        df = df.withColumn("outbound", F.array())

    out = (
        df.select("ip", "source_file", "inbound", "outbound")
        .withColumn("feat_inbound_len", F.size(F.col("inbound")).cast("double"))
        .withColumn("feat_outbound_len", F.size(F.col("outbound")).cast("double"))
        .withColumn("feat_inbound_sum", F.expr("aggregate(inbound, cast(0.0 as double), (acc, x) -> acc + cast(x as double))"))
        .withColumn("feat_outbound_sum", F.expr("aggregate(outbound, cast(0.0 as double), (acc, x) -> acc + cast(x as double))"))
        .withColumn("feat_inbound_mean", _safe_mean("inbound"))
        .withColumn("feat_outbound_mean", _safe_mean("outbound"))
        .withColumn("feat_inbound_std", _safe_std("inbound"))
        .withColumn("feat_outbound_std", _safe_std("outbound"))
        .withColumn("feat_inbound_max", F.when(F.size(F.col("inbound")) > 0, F.array_max(F.col("inbound")).cast("double")).otherwise(F.lit(0.0)))
        .withColumn("feat_outbound_max", F.when(F.size(F.col("outbound")) > 0, F.array_max(F.col("outbound")).cast("double")).otherwise(F.lit(0.0)))
        .withColumn("feat_inbound_nonzero", F.expr("size(filter(inbound, x -> cast(x as double) > 0.0))").cast("double"))
        .withColumn("feat_outbound_nonzero", F.expr("size(filter(outbound, x -> cast(x as double) > 0.0))").cast("double"))
        .withColumn("feat_inbound_first", F.when(F.size(F.col("inbound")) > 0, F.element_at(F.col("inbound"), 1).cast("double")).otherwise(F.lit(0.0)))
        .withColumn("feat_inbound_last", F.when(F.size(F.col("inbound")) > 0, F.element_at(F.col("inbound"), -1).cast("double")).otherwise(F.lit(0.0)))
        .withColumn("feat_outbound_first", F.when(F.size(F.col("outbound")) > 0, F.element_at(F.col("outbound"), 1).cast("double")).otherwise(F.lit(0.0)))
        .withColumn("feat_outbound_last", F.when(F.size(F.col("outbound")) > 0, F.element_at(F.col("outbound"), -1).cast("double")).otherwise(F.lit(0.0)))
        .drop("inbound", "outbound")
    )

    output_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out.toPandas().to_csv(args.output_csv, index=False)

    print(f"Wrote tiny TSFresh fixture CSV to {args.output_csv}")
    print(f"Rows: {out.count()}")
    spark.stop()


if __name__ == "__main__":
    main()
