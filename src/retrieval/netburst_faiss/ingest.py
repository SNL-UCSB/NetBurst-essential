#!/usr/bin/env python3
"""
Ingest a pre-clustered representation CSV into a Milvus Lite collection.

Expected CSV columns: ip, source_file, label, repr_dim_0, repr_dim_1, ...

Produces in --output-dir:
  {model}_split_manifest.csv  -- point_id, ip, source_file, cluster_id, split, milvus_id
  {model}_eval_vecs.npy       -- (n_eval, D) L2-normalized float32 eval vectors,
                                  same row order as eval rows in the manifest
  {model}_split.pkl           -- pickle with keys 'train_df' and 'eval_df' (raw DataFrames
                                  before vector normalization, including all original columns)

Usage:
  python ingest.py \
    --cluster-csv /path/to/clusters.csv \
    --model chronos2 \
    --output-dir /path/to/outputs \
    [--db-path /path/to/milvus_lite.db] \
    [--seed 42] [--eval-fraction 0.3] [--n-eval N] [--nlist 200] \
    [--index-type FLAT]

Default split when no --split-manifest: 70%% train (ingested) / 30%% eval
(--eval-fraction 0.3). Use --n-eval for an exact eval count (overrides fraction).
Default index type is FLAT (exact brute-force). Use --index-type IVF_FLAT to
enable approximate search with --nlist centroids.
"""

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

REPR_PREFIX = "repr_dim_"
TSFRESH_PREFIX = "value__"


def parse_args():
    p = argparse.ArgumentParser(description="Ingest cluster CSV into Milvus Lite")
    p.add_argument("--cluster-csv", required=True,
                   help="Cluster CSV: ip, source_file, label, repr_dim_*")
    p.add_argument("--model", required=True,
                   help="Model label, e.g. 'chronos2' or 'netburst'")
    p.add_argument("--db-path",
                   default="<data-root>/vector_db/milvus_lite.db",
                   help="Milvus Lite .db file path (created if absent)")
    p.add_argument("--output-dir", required=True,
                   help="Directory for split manifest and eval vectors")
    p.add_argument("--split-manifest", default=None,
                   help="Pre-aligned split manifest (if provided, uses this instead of computing split)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.3,
        help="Fraction of rows for eval when --n-eval is not set (default 0.3 => 70%% ingested / 30%% eval).",
    )
    p.add_argument(
        "--n-eval",
        type=int,
        default=None,
        help="Exact eval count (overrides --eval-fraction). Ignored if --split-manifest provided.",
    )
    p.add_argument("--nlist", type=int, default=200,
                   help="IVF_FLAT nlist parameter (only used when --index-type IVF_FLAT)")
    p.add_argument("--index-type", default="FLAT",
                   choices=["FLAT", "IVF_FLAT"],
                   help="Milvus index type: FLAT (exact brute-force, default) or IVF_FLAT (approximate)")
    return p.parse_args()


def collection_name(model: str) -> str:
    return f"net_repr_{model.replace('-', '_')}"


def embedding_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns used as the vector embedding: repr_dim_* (Neural) or value__* (TSFresh)."""
    repr_cols = sorted(
        [c for c in df.columns if c.startswith(REPR_PREFIX)],
        key=lambda c: int(c[len(REPR_PREFIX) :]),
    )
    if repr_cols:
        return repr_cols
    tsfresh_cols = sorted(c for c in df.columns if c.startswith(TSFRESH_PREFIX))
    if tsfresh_cols:
        return tsfresh_cols
    meta = {"ip", "source_file", "label", "point_id"}
    numeric = [
        c
        for c in df.columns
        if c not in meta and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric:
        raise ValueError(
            f"No embedding columns: expected '{REPR_PREFIX}*', '{TSFRESH_PREFIX}*', "
            "or other numeric columns besides ip/source_file/label."
        )
    return sorted(numeric)


def load_and_validate(csv_path: str):
    print(f"[ingest] Loading cluster CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"ip", "source_file", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cluster CSV missing required columns: {missing}")
    repr_cols = embedding_feature_columns(df)
    print(f"  Rows: {len(df):,}  |  embedding dim: {len(repr_cols)}")
    return df, repr_cols


def build_point_ids(df: pd.DataFrame) -> pd.Series:
    ids = df["ip"].astype(str) + "::" + df["source_file"].astype(str)
    # Deduplicate: append row-index suffix on duplicates
    dup_mask = ids.duplicated(keep=False)
    if dup_mask.any():
        print(f"  WARNING: {dup_mask.sum()} rows share a point_id — appending row-index suffix")
        ids[dup_mask] = ids[dup_mask] + "::" + df.index[dup_mask].astype(str)
    return ids


def split_deterministic(df: pd.DataFrame, n_eval: int, seed: int):
    if n_eval >= len(df):
        raise ValueError(f"n_eval={n_eval} >= dataset size={len(df)}")
    rng = np.random.default_rng(seed)
    eval_indices = rng.choice(len(df), size=n_eval, replace=False)
    eval_mask = np.zeros(len(df), dtype=bool)
    eval_mask[eval_indices] = True
    train_df = df[~eval_mask].reset_index(drop=True)
    eval_df = df[eval_mask].reset_index(drop=True)
    print(f"  Split (seed={seed}): train={len(train_df):,}  eval={len(eval_df):,}")
    return train_df, eval_df


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def insert_in_batches(col, ids_np, cluster_ids_np, vecs_np, rows_per_batch=10_000):
    N = len(ids_np)
    for s in range(0, N, rows_per_batch):
        e = min(s + rows_per_batch, N)
        col.insert([
            ids_np[s:e].tolist(),
            cluster_ids_np[s:e].tolist(),
            vecs_np[s:e].tolist(),
        ])
    col.flush()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load & validate -----------------------------------------------------
    df, repr_cols = load_and_validate(args.cluster_csv)
    df = df.copy()
    df["point_id"] = build_point_ids(df)

    # -- Deterministic split -------------------------------------------------
    if args.split_manifest:
        print(f"[ingest] Using pre-aligned split manifest: {args.split_manifest}")
        split_df = pd.read_csv(args.split_manifest)
        # Filter to rows in this cluster CSV
        df_with_split = df.merge(split_df[["point_id", "split"]], on="point_id", how="inner")
        train_df = df_with_split[df_with_split["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        eval_df = df_with_split[df_with_split["split"] == "eval"].drop(columns=["split"]).reset_index(drop=True)
    else:
        if args.n_eval is not None:
            n_eval = args.n_eval
        else:
            n_eval = int(round(len(df) * float(args.eval_fraction)))
            n_eval = max(1, min(n_eval, len(df) - 1))
        print(f"  Using n_eval={n_eval:,} (eval_fraction={args.eval_fraction}, n_eval override={args.n_eval is not None})")
        train_df, eval_df = split_deterministic(df, n_eval, args.seed)

    # Assign sequential surrogate Milvus IDs to train rows
    train_df = train_df.copy()
    train_df["milvus_id"] = np.arange(len(train_df), dtype=np.int64)

    # -- Save split manifest -------------------------------------------------
    manifest_train = train_df[["point_id", "ip", "source_file", "milvus_id"]].copy()
    manifest_train["cluster_id"] = train_df["label"].astype(np.int32)
    manifest_train["split"] = "train"

    manifest_eval = eval_df[["point_id", "ip", "source_file"]].copy()
    manifest_eval["cluster_id"] = eval_df["label"].astype(np.int32)
    manifest_eval["milvus_id"] = np.int64(-1)
    manifest_eval["split"] = "eval"

    manifest = pd.concat([manifest_train, manifest_eval], ignore_index=True)
    manifest_path = os.path.join(args.output_dir, f"{args.model}_split_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"  Saved split manifest: {manifest_path}")

    # -- Save eval vectors ---------------------------------------------------
    eval_vecs = np.nan_to_num(
        eval_df[repr_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    eval_vecs_norm = l2_normalize(eval_vecs)
    eval_vecs_path = os.path.join(args.output_dir, f"{args.model}_eval_vecs.npy")
    np.save(eval_vecs_path, eval_vecs_norm)
    print(f"  Saved eval vectors:   {eval_vecs_path}  shape={eval_vecs_norm.shape}")

    # -- Save train/eval split as pickle ------------------------------------
    pickle_path = os.path.join(args.output_dir, f"{args.model}_split.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({"train_df": train_df, "eval_df": eval_df}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved split pickle:   {pickle_path}  (train={len(train_df):,}, eval={len(eval_df):,})")

    # -- Prepare train vectors -----------------------------------------------
    train_vecs = np.nan_to_num(
        train_df[repr_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    train_vecs_norm = l2_normalize(train_vecs)
    DIM = train_vecs_norm.shape[1]

    # -- Connect to Milvus Lite ----------------------------------------------
    connections.connect(alias="default", uri=args.db_path)
    cname = collection_name(args.model)
    if utility.has_collection(cname):
        utility.drop_collection(cname)
        print(f"  Dropped existing collection '{cname}'")

    # -- Create collection schema --------------------------------------------
    fields = [
        FieldSchema(name="id",         dtype=DataType.INT64,         is_primary=True, auto_id=False),
        FieldSchema(name="cluster_id", dtype=DataType.INT32),
        FieldSchema(name="embedding",  dtype=DataType.FLOAT_VECTOR,  dim=DIM),
    ]
    schema = CollectionSchema(fields=fields, description=f"{args.model} representations")
    col = Collection(name=cname, schema=schema)

    # -- Ingest train vectors first (IVF coarse quantizer trains on inserted data) --
    rows_per_batch = max(4_000, 400_000 // DIM)
    t0 = time.perf_counter()
    insert_in_batches(
        col,
        train_df["milvus_id"].to_numpy(dtype=np.int64),
        train_df["label"].to_numpy(dtype=np.int32),
        train_vecs_norm,
        rows_per_batch=rows_per_batch,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Ingested {len(train_df):,} train vectors into '{cname}' "
          f"(dim={DIM}) in {elapsed:.1f}s")

    if args.index_type == "IVF_FLAT":
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": args.nlist},
        }
        print(f"  Building IVF_FLAT index (nlist={args.nlist}) ...")
    else:
        index_params = {
            "metric_type": "COSINE",
            "index_type": "FLAT",
            "params": {},
        }
        print("  Building FLAT index (exact brute-force) ...")

    col.create_index(field_name="embedding", index_params=index_params)
    utility.wait_for_index_building_complete(cname, index_name="embedding")
    col.load()
    print("[ingest] Done.\n")


if __name__ == "__main__":
    main()
