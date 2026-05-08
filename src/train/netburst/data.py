"""Datasets and Spark-backed parquet loaders for NetBurst."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from torch.utils.data import Dataset

class SeriesDataset(Dataset):
    def __init__(self, series_list):
        # each series is a Python list of floats (no -1)
        self.data = series_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = np.array(self.data[idx], dtype=np.float32)
        return torch.from_numpy(arr)

class PairSeriesDataset(Dataset):
    """
    Dataset that returns pairs of sequences (bi, ibg), each as 1D float tensors.
    """
    def __init__(self, pair_list):
        self.data = pair_list  # list of tuples (bi_list, ibg_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bi, ibg = self.data[idx]
        bi_t  = torch.tensor(bi, dtype=torch.float32)
        ibg_t = torch.tensor(ibg, dtype=torch.float32)
        return bi_t, ibg_t

def load_cleaned_series(parquet_root, min_len=10, max_len = None, limit=None, nonHierarchical=False, outbound_only = False):
    spark = SparkSession.builder \
        .appName("IEI‐Spark‐Load") \
        .config("spark.driver.memory","100g") \
        .config("spark.driver.maxResultSize","50g") \
        .getOrCreate()

    schema = StructType([
        StructField("ip", StringType(), False),
        StructField("node_features",
            ArrayType(ArrayType(FloatType()), False), False),
        StructField("edge_indices",
            ArrayType(ArrayType(LongType()), False), True),
    ])
    if not nonHierarchical:
        df = (spark.read
                .option("recursiveFileLookup","true")
                .schema(schema)
                .parquet(parquet_root)
                .select("node_features"))

        # pull just the port=0 lists, remove -1s, filter by length
        vals = (df
            .select(F.explode("node_features").alias("feat"))
            .filter("feat[0]=0.0")
            .selectExpr("slice(feat,2,size(feat)-1) as vals")
            .rdd
            .map(lambda r: [v for v in r[0] if v!=-1.0 and v != 0.0])
        )
        if limit:
            vals = vals.zipWithIndex().filter(lambda x_i: x_i[1]<limit).keys()
        # filter by min_len and max_len, if specified
        if max_len is None:
            series = vals.filter(lambda arr: len(arr)>=min_len).collect()
        else:
            series = vals.filter(lambda arr: len(arr)>=min_len).map(lambda arr: arr[:max_len]).collect()
    else:
        df = spark.read.option("recursiveFileLookup", "true").parquet(parquet_root)
        # Pick one inbound-like and one outbound-like column
        in_cols  = [c for c in df.columns if "inbound"  in c.lower()]
        out_cols = [c for c in df.columns if "outbound" in c.lower()]
        assert in_cols and out_cols, f"Need at least one inbound and one outbound column, got: {df.columns}"

        in_col  = sorted(in_cols,  key=len)[0]   # prefer the simplest name (e.g., 'inbound' over 'iei_inbound')
        out_col = sorted(out_cols, key=len)[0]   # prefer 'outbound' over 'iei_outbound'

        # Remove values <= 0.0 (this also drops -1 paddings)
        print(f"count of df before cleaning: {df.count()}")
        df_clean = df.select(
            F.when(F.col(in_col).isNotNull(),  F.expr(f"filter({in_col},  x -> x > 0.0)"))
            .otherwise(F.array().cast("array<double>")).alias("in_clean"),
            F.when(F.col(out_col).isNotNull(), F.expr(f"filter({out_col}, x -> x > 0.0)"))
            .otherwise(F.array().cast("array<double>")).alias("out_clean"),
        )

        # Keep only rows with enough remaining positives
        df_clean = df_clean.filter(F.size("in_clean")  >= min_len) 
        # Optional: cap length
        if max_len is not None:
            df_clean = df_clean.select(
                F.expr(f"slice(in_clean,  1, {max_len})").alias("in_clean"),
                F.expr(f"slice(out_clean, 1, {max_len})").alias("out_clean"),
            )

        # Back to your RDD pair shape
        pairs = df_clean.select("in_clean", "out_clean").rdd.map(lambda r: (r["in_clean"], r["out_clean"]))

        # Collect and choose which side you want
        inbound, outbound = zip(*pairs.collect())
        inbound, outbound = list(inbound), list(outbound)
        series = outbound if outbound_only else inbound
        #print number of series found
        print(f"Number of series found: {len(series)}")
    #if any of the series have 0 print "0 series found"
    return series

def collate_as_list(batch):
    """
    Custom collate function to convert a batch of tensors into a list.
    """
    # Convert each tensor in the batch to a list
    return batch


def iterative_quantile_bins(list_of_lists, n_bins):
    """
    Build exactly n_bins+1 edges by:
    1. Flattening and sorting the data.
    2. Repeatedly computing all rem_bins quantile cut-points.
    3. If any quantile value repeats, adding all unique smaller cuts,
       truncating data to > repeating value, and looping.
    4. If no repeats, taking all remaining cuts at once.
    5. Finally appending the global maximum.
    """
    flat = np.sort(np.concatenate(list_of_lists))
    if flat.size == 0:
        return np.array([])

    edges = []
    data = flat.copy()
    remaining_bins = n_bins

    while remaining_bins > 1 and data.size > 0:
        # 1) quantile positions including 100%
        percents = np.arange(1, remaining_bins) * (100.0 / remaining_bins)
        cuts = np.percentile(data, percents, method='lower')

        # 2) detect first repeating cut
        seen = set()
        first_rep = None
        print(f"remaining_bins : {remaining_bins}")
        for v in cuts:
            if v in seen:
                first_rep = v
                break
            seen.add(v)

        if first_rep is None:
            # no repeats: accept all and finish
            edges.extend(cuts.tolist())
            remaining_bins = 0
            break
        else:
            # take all cuts smaller than the plateau
            smaller_cuts = [v for v in cuts if v < first_rep]
            edges.extend(smaller_cuts)
            edges.append(first_rep)
            remaining_bins -= (len(smaller_cuts)+1)
            # truncate data to values strictly greater than the plateau
            data = data[data > first_rep]

    # append the global max
    edges.append(flat[-1])
    if len(edges) > n_bins + 1:
        # if we have too many edges, truncate to n_bins + 1
        edges = edges[:n_bins + 1]
    elif len(edges) < n_bins + 1:
        # from left to right, find the non consecutive values, for each non consecutive value pair, take their mid point and add the midpoint to the list. repeat this till length of edges is n_bins + 1
        while len(edges) < n_bins + 1:
            new_edges = []
            for i in range(len(edges) - 1):
                if edges[i + 1] == edges[i] + 1:
                    # non consecutive
                    continue
                mid_point = (edges[i] + edges[i + 1]) / 2
                new_edges.append(mid_point)
                if len(edges) + len(new_edges) >= n_bins + 1:
                    break
            edges.extend(new_edges)
            print(len(edges))
            #sort the edges list
            edges = sorted(edges)
    return np.array(edges)

def load_ibg_bi_series(parquet_root: str, min_len: int = 10, max_len: Optional[int] = None, limit: Optional[int] = None):
    """
    Read parquet generated by IBGAndBIProcessing.py and return list of (bi_list, ibg_list),
    ensuring both arrays are the same length per record.
    """
    spark = SparkSession.builder \
        .appName("IBG-BI-Load") \
        .config("spark.driver.memory", "100g") \
        .config("spark.driver.maxResultSize", "50g") \
        .getOrCreate()

    df = spark.read.option("recursiveFileLookup", "true").parquet(parquet_root)

    # Some outputs may not include active_bins; we only need bi and ibg
    # Filter for min length and equal lengths
    df2 = (
        df.select(
            F.when(F.col("bi").isNotNull(), F.col("bi")).otherwise(F.array().cast("array<double>")).alias("bi"),
            F.when(F.col("ibg").isNotNull(), F.col("ibg")).otherwise(F.array().cast("array<double>")).alias("ibg"),
        )
        .filter((F.size("bi") >= F.lit(min_len)) & (F.size("ibg") >= F.lit(min_len)))
        .filter(F.size("bi") == F.size("ibg"))
    )

    if max_len is not None:
        df2 = df2.select(
            F.expr(f"slice(bi, 1, {max_len})").alias("bi"),
            F.expr(f"slice(ibg, 1, {max_len})").alias("ibg"),
        )

    rdd = df2.rdd.map(lambda r: (list(map(float, r["bi"])), list(map(float, r["ibg"]))))
    if limit is not None:
        rdd = rdd.zipWithIndex().filter(lambda x_i: x_i[1] < limit).keys()

    series = rdd.collect()
    print(f"Loaded {len(series)} (bi, ibg) pairs from parquet")
    spark.stop()
    return series

def load_fires_ibg_bi(parquet_root: str, min_len: int = 10, max_len: Optional[int] = None, limit: Optional[int] = None):
    """
    Load (bi, ibg) pairs from a Fires parquet preprocessed by
    src/preprocess/ConvertFiresToIBGBI.py. Expects columns:
      - bi:  array<double>
      - ibg: array<long/double>
    Returns: list of (bi_list, ibg_list) with equal lengths per row.
    """
    spark = SparkSession.builder \
        .appName("Fires-IBG-BI-Load") \
        .config("spark.driver.memory", "100g") \
        .config("spark.driver.maxResultSize", "50g") \
        .getOrCreate()

    df = spark.read.option("recursiveFileLookup", "true").parquet(parquet_root)

    # Ensure required columns exist
    if "bi" not in df.columns or "ibg" not in df.columns:
        spark.stop()
        raise ValueError(
            f"Fires parquet at {parquet_root} must contain 'bi' and 'ibg' columns. "
            "Please preprocess with ConvertFiresToIBGBI.py."
        )

    df2 = (
        df.select(
            F.when(F.col("bi").isNotNull(),  F.col("bi")).otherwise(F.array().cast("array<double>")).alias("bi"),
            # Cast ibg to double array if needed to keep numeric consistency downstream
            F.when(F.col("ibg").isNotNull(), F.col("ibg")).otherwise(F.array().cast("array<double>")).alias("ibg"),
        )
        .filter((F.size("bi") >= F.lit(min_len)) & (F.size("ibg") >= F.lit(min_len)))
        .filter(F.size("bi") == F.size("ibg"))
    )

    if max_len is not None:
        df2 = df2.select(
            F.expr(f"slice(bi, 1, {max_len})").alias("bi"),
            F.expr(f"slice(ibg, 1, {max_len})").alias("ibg"),
        )

    rdd = df2.rdd.map(lambda r: (list(map(float, r["bi"])), list(map(float, r["ibg"])) ))
    if limit is not None:
        rdd = rdd.zipWithIndex().filter(lambda x_i: x_i[1] < limit).keys()

    series = rdd.collect()
    print(f"Loaded {len(series)} (bi, ibg) pairs from Fires parquet")
    spark.stop()
    return series

def load_filtered_ibg_bi(parquet_root: str, ips_csv: str, min_len: int = 10, max_len: Optional[int] = None, limit: Optional[int] = None):
    """
    Load (bi, ibg) pairs from parquet but restrict rows to allowed keys provided in a CSV.
    The CSV may have columns: 'ip', or 'ip,service_port', or 'subnet'.
    Returns a list of (bi_list, ibg_list) with equal lengths per row.
    """
    spark = SparkSession.builder \
        .appName("IBG-BI-Filtered-Load") \
        .config("spark.driver.memory", "100g") \
        .config("spark.driver.maxResultSize", "50g") \
        .getOrCreate()

    # Read allowed keys
    allowed_pd = pd.read_csv(ips_csv)
    allowed_pd.columns = [c.strip().lower() for c in allowed_pd.columns]

    if {"ip", "service_port"}.issubset(allowed_pd.columns):
        key_cols = ["ip", "service_port"]
        allowed_pd["service_port"] = allowed_pd["service_port"].astype("int64")
    elif "ip" in allowed_pd.columns:
        key_cols = ["ip"]
    elif "subnet" in allowed_pd.columns:
        key_cols = ["subnet"]
    else:
        spark.stop()
        raise ValueError("CSV must have header 'ip', or 'ip,service_port', or 'subnet'.")

    allowed_sdf = spark.createDataFrame(allowed_pd[key_cols]).dropDuplicates()

    # Read parquet and normalize keys
    df = spark.read.option("recursiveFileLookup", "true").parquet(parquet_root)

    if "service_port" in key_cols and "service_port" in df.columns:
        df = df.withColumn("service_port", F.col("service_port").cast("int"))

    if key_cols == ["subnet"] and "subnet" not in df.columns and "ip" in df.columns:
        df = df.withColumn("subnet", F.regexp_extract("ip", r"^(\\d+\\.\\d+\\.\\d+)\\.\\d+$", 1))

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        spark.stop()
        raise RuntimeError(f"Parquet is missing key columns: {missing}")

    # Restrict to allowed keys
    df = df.join(allowed_sdf, on=key_cols, how="inner")

    # Ensure bi and ibg columns exist
    if "bi" not in df.columns or "ibg" not in df.columns:
        spark.stop()
        raise RuntimeError("Parquet must contain 'bi' and 'ibg' columns for filtered IBG/BI inference.")

    # Keep key columns with the arrays
    select_cols = key_cols + [
        F.when(F.col("bi").isNotNull(), F.col("bi")).otherwise(F.array().cast("array<double>")).alias("bi"),
        F.when(F.col("ibg").isNotNull(), F.col("ibg")).otherwise(F.array().cast("array<double>")).alias("ibg"),
    ]
    df2 = df.select(*select_cols) \
           .filter((F.size("bi") >= F.lit(min_len)) & (F.size("ibg") >= F.lit(min_len))) \
           .filter(F.size("bi") == F.size("ibg"))

    if max_len is not None:
        df2 = df2.select(*key_cols,
            F.expr(f"slice(bi, 1, {max_len})").alias("bi"),
            F.expr(f"slice(ibg, 1, {max_len})").alias("ibg"),
        )

    def pack_key_row(r):
        if len(key_cols) == 1:
            return r[key_cols[0]]
        return tuple(r[c] for c in key_cols)

    rdd = df2.rdd.map(lambda r: (pack_key_row(r), list(map(float, r["bi"])), list(map(float, r["ibg"]))))
    if limit is not None:
        rdd = rdd.zipWithIndex().filter(lambda x_i: x_i[1] < limit).keys()

    series = rdd.collect()  # list of (key, bi, ibg)
    print(f"Loaded {len(series)} filtered (key, bi, ibg) triplets from parquet")
    spark.stop()
    return series

def load_precomputed_ibgbi_context_forecast(parquet_root: str, min_len: int = 20, limit: Optional[int] = None):
    """
    Load rows that already include context/forecast slices from create_train_test_splits.py output.
    Expected columns: context_bi, forecast_bi, context_ibg, forecast_ibg.
    Optional key columns include ip/subnet/service_port/source_file.
    Returns list of (key, context_bi, forecast_bi, context_ibg, forecast_ibg).

    Rows must have at least one context step and one forecast step. ``min_len`` is a lower bound on
    total length (len(context)+len(forecast)); it is floored at 2 so (1+1) is always allowed when
    ``min_len`` <= 2.
    """
    spark = SparkSession.builder \
        .appName("IBGBI-Precomputed-Context-Forecast-Load") \
        .config("spark.driver.memory", "100g") \
        .config("spark.driver.maxResultSize", "50g") \
        .getOrCreate()

    df = spark.read.option("recursiveFileLookup", "true").parquet(parquet_root)
    req = {"context_bi", "forecast_bi", "context_ibg", "forecast_ibg"}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        spark.stop()
        raise RuntimeError(
            f"Parquet at {parquet_root} is missing required precomputed columns: {missing}"
        )

    key_cols = [
        c
        for c in ("ip", "service_port", "subnet", "label", "chunk_idx", "source_file", "example_id")
        if c in df.columns
    ]
    df2 = (
        df.select(
            *key_cols,
            F.when(F.col("context_bi").isNotNull(), F.col("context_bi")).otherwise(F.array().cast("array<double>")).alias("context_bi"),
            F.when(F.col("forecast_bi").isNotNull(), F.col("forecast_bi")).otherwise(F.array().cast("array<double>")).alias("forecast_bi"),
            F.when(F.col("context_ibg").isNotNull(), F.col("context_ibg")).otherwise(F.array().cast("array<double>")).alias("context_ibg"),
            F.when(F.col("forecast_ibg").isNotNull(), F.col("forecast_ibg")).otherwise(F.array().cast("array<double>")).alias("forecast_ibg"),
        )
        .withColumn("_total_len", F.size("context_bi") + F.size("forecast_bi"))
        .filter(F.size("context_bi") >= 1)
        .filter(F.size("forecast_bi") >= 1)
        .filter(F.col("_total_len") >= F.lit(max(2, int(min_len))))
        .filter((F.size("context_bi") + F.size("forecast_bi")) == (F.size("context_ibg") + F.size("forecast_ibg")))
    )

    def _pack_row(r):
        if key_cols:
            key = tuple(r[c] for c in key_cols) if len(key_cols) > 1 else r[key_cols[0]]
        else:
            key = None
        return (
            key,
            list(map(float, r["context_bi"])),
            list(map(float, r["forecast_bi"])),
            list(map(float, r["context_ibg"])),
            list(map(float, r["forecast_ibg"])),
        )

    rdd = df2.rdd.map(_pack_row)
    if limit is not None:
        rdd = rdd.zipWithIndex().filter(lambda x_i: x_i[1] < limit).keys()
    rows = rdd.collect()
    print(f"Loaded {len(rows)} precomputed (key, context/forecast bi/ibg) rows from parquet")
    spark.stop()
    return rows

def load_series_eval(parquet_root, allowed_ips, min_len=10):
    """
    Load only records whose `ip` is in allowed_ips, then
    explode/clean exactly as before and return (ip, series) pairs.
    
    :param parquet_root: path to your Parquet files
    :param allowed_ips:  Python list or set of IP‐strings to keep
    :param min_len:     minimum length of the cleaned series
    """
    spark = SparkSession.builder \
        .appName("IEI‐Spark‐Load") \
        .config("spark.driver.memory","200g") \
        .config("spark.driver.maxResultSize","200g") \
        .getOrCreate()

    schema = StructType([
        StructField("ip", StringType(), False),
        StructField("node_features",
            ArrayType(ArrayType(FloatType()), False), False),
        StructField("edge_indices",
            ArrayType(ArrayType(LongType()), False), True),
    ])

    # read only the IPs you care about
    df = (spark.read
             .option("recursiveFileLookup","true")
             .schema(schema)
             .parquet(parquet_root)
             .select("ip", "node_features")
             # DataFrame‐level filter is more efficient:
             .filter(F.col("ip").isin(list(allowed_ips)))
         )

    # explode & clean just like before
    rdd = (
        df
        .select("ip", F.explode("node_features").alias("feat"))
        .filter("feat[0] = 0.0")
        .select(
            "ip",
            F.expr("slice(feat, 2, size(feat)-1)").alias("vals")
        )
        .rdd
        .map(lambda row: (
            row["ip"],
            [v for v in row["vals"] if v != -1.0]
        ))
    )

    # filter by min length and collect
    series = (
        rdd
        .filter(lambda ip_series: len(ip_series[1]) >= min_len)
        .collect()
    )

    spark.stop()
    return series

# -----------------------------------
# Your original function, upgraded
# -----------------------------------

def pick_series_col_by_substring(df, override=None):
    if override is not None:
        if override not in df.columns:
            raise RuntimeError(f"--series_col={override} not in parquet columns {sorted(df.columns)}")
        return override

    # collect candidates that contain 'inbound' (avoid columns we create later like *_filtered)
    lowered = {c.lower(): c for c in df.columns}
    candidates = [orig for low, orig in lowered.items()
                  if "inbound" in low and not low.endswith("_filtered")]

    if not candidates:
        raise RuntimeError(f"No column containing 'inbound' found in {sorted(df.columns)}")

    # priority: exact 'inbound' -> exact 'iei_inbound' -> shortest name
    if "inbound" in lowered:
        return lowered["inbound"]
    if "iei_inbound" in lowered:
        return lowered["iei_inbound"]
    return min(candidates, key=len)


def load_ibg_bi_with_ip(
    parquet_root: str,
    allowed_ips: Optional[Set[str]] = None,
    min_len: int = 10,
    ip_suffix_filter: Optional[str] = None,
):
    """Load (bi, ibg) pairs preserving IP and source_file columns."""
    spark = (
        SparkSession.builder.appName("IP-Representations-Load")
        .config("spark.driver.memory", "200g")
        .config("spark.driver.maxResultSize", "200g")
        .getOrCreate()
    )

    df = spark.read.option("recursiveFileLookup", "true").parquet(parquet_root)
    print("Available columns:", df.columns)

    if allowed_ips and "ip" in df.columns:
        df = df.filter(F.col("ip").isin(list(allowed_ips)))

    if ip_suffix_filter:
        if "ip" not in df.columns:
            print("Warning: --ip_suffix_filter set but parquet has no 'ip' column; suffix filter skipped.")
        else:
            df = df.filter(F.col("ip").endswith(ip_suffix_filter))
            print(f"Applied ip suffix filter: endswith({ip_suffix_filter!r})")

    if "bi" not in df.columns or "ibg" not in df.columns:
        spark.stop()
        raise ValueError("Parquet must contain 'bi' and 'ibg' columns for TwinHeadChronosPredictor")

    df_clean = df.select(
        F.col("ip") if "ip" in df.columns else F.lit("unknown").alias("ip"),
        F.col("source_file") if "source_file" in df.columns else F.lit("unknown").alias("source_file"),
        F.when(F.col("bi").isNotNull(), F.col("bi")).otherwise(F.array().cast("array<double>")).alias("bi"),
        F.when(F.col("ibg").isNotNull(), F.col("ibg")).otherwise(F.array().cast("array<double>")).alias("ibg"),
    ).filter(
        (F.size("bi") >= F.lit(min_len))
        & (F.size("ibg") >= F.lit(min_len))
        & (F.size("bi") == F.size("ibg"))
    )

    results = df_clean.select("ip", "source_file", "bi", "ibg").collect()
    spark.stop()

    return [
        (row["ip"], row["source_file"], (list(map(float, row["bi"])), list(map(float, row["ibg"]))))
        for row in results
    ]
