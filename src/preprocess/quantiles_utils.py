#!/usr/bin/env python3
"""
Shared utilities to compute, store, and read quantiles over sparse time-series arrays.
Quantiles are computed over all values (including zeros) so sparsity is preserved in the distribution.
"""

from typing import List, Optional, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    FloatType,
    StringType,
)
import pyspark.sql.functions as F


# Default 10 evenly spaced quantile levels for reference
DEFAULT_QUANTILE_LEVELS = [0.1 * (i + 1) for i in range(10)]  # [0.1, 0.2, ..., 1.0]


def compute_quantiles_from_array_column(
    df: DataFrame,
    array_column: str,
    levels: Optional[List[float]] = None,
    relative_error: float = 0.01,
) -> Tuple[List[float], List[float]]:
    """
    Compute approximate quantiles over all elements of an array column (including zeros).
    Explodes the array so every value contributes to the distribution (sparsity preserved).

    Args:
        df: Spark DataFrame with at least one array column.
        array_column: Name of the array column (e.g. "inbound", "outbound").
        levels: Quantile levels in [0, 1], e.g. [0.1, 0.2, ..., 1.0]. Default: 10 evenly spaced.
        relative_error: Relative error for approxQuantile (0 = exact, higher = faster).

    Returns:
        (levels, quantiles): lists of same length.
    """
    if levels is None:
        levels = list(DEFAULT_QUANTILE_LEVELS)
    levels = sorted(levels)

    exploded = df.select(F.explode(F.col(array_column)).alias("_value"))
    quantiles = exploded.stat.approxQuantile("_value", levels, relative_error)
    return levels, quantiles


def compute_quantiles_from_array_column_multitask(
    spark: SparkSession,
    df: DataFrame,
    array_column: str,
    levels: Optional[List[float]] = None,
    relative_error: float = 0.01,
) -> Tuple[List[float], List[float]]:
    """
    Same as compute_quantiles_from_array_column but safe when df is built from
    multiple task outputs (e.g. multiple parquet dirs). Use when reading from
    OUTPUT_DIR_ip/task_* etc.
    """
    return compute_quantiles_from_array_column(df, array_column, levels, relative_error)


_QUANTILES_SCHEMA = StructType([
    StructField("entity_type", StringType(), False),
    StructField("levels", ArrayType(FloatType(), False), False),
    StructField("quantiles", ArrayType(FloatType(), False), False),
])


def write_quantiles_parquet(
    spark: SparkSession,
    entity_type: str,
    levels: List[float],
    quantiles: List[float],
    path: str,
) -> None:
    """
    Write a single entity type's quantiles to Parquet.
    Schema: entity_type (string), levels (array<float>), quantiles (array<float>).
    """
    levels_f = [float(x) for x in levels]
    quantiles_f = [float(x) for x in quantiles]
    row = [(entity_type, levels_f, quantiles_f)]
    out_df = spark.createDataFrame(row, _QUANTILES_SCHEMA)
    out_df.write.mode("overwrite").parquet(path)


def write_quantiles_parquet_multi(
    spark: SparkSession,
    rows: List[Tuple[str, List[float], List[float]]],
    path: str,
) -> None:
    """
    Write multiple entity types' quantiles to a single Parquet.
    rows: list of (entity_type, levels, quantiles).
    """
    data = [
        (et, [float(x) for x in lv], [float(x) for x in qv])
        for et, lv, qv in rows
    ]
    out_df = spark.createDataFrame(data, _QUANTILES_SCHEMA)
    out_df.write.mode("overwrite").parquet(path)




def read_quantiles_parquet(path: str, spark: SparkSession) -> DataFrame:
    """Read a quantiles Parquet (entity_type, levels, quantiles)."""
    return spark.read.parquet(path)


def get_threshold_for_quantile(
    levels: List[float],
    quantiles: List[float],
    q: float,
) -> float:
    """
    Interpolate threshold for a requested quantile q from stored (levels, quantiles).
    If q exactly equals a level, return the corresponding quantile; otherwise interpolate.
    """
    if not levels or not quantiles or len(levels) != len(quantiles):
        raise ValueError("levels and quantiles must be non-empty and same length")
    if q <= levels[0]:
        return quantiles[0]
    if q >= levels[-1]:
        return quantiles[-1]
    for i in range(len(levels) - 1):
        if levels[i] <= q <= levels[i + 1]:
            if levels[i + 1] == levels[i]:
                return quantiles[i]
            t = (q - levels[i]) / (levels[i + 1] - levels[i])
            return quantiles[i] + t * (quantiles[i + 1] - quantiles[i])
    return quantiles[-1]
