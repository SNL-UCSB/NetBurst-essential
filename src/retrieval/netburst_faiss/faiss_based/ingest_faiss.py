#!/usr/bin/env python3
"""
Ingest cluster CSV into FAISS IndexIVFFlat (inner product = cosine on L2-normalized rows).

Writes the same artifacts as ingest.py (manifest, eval_vecs.npy, split.pkl) plus:
  {model}_ivf_flat.faiss       -- FAISS index
  {model}_ivf_flat.faiss.meta.json -- dim, nlist, nprobe default, metric

Does not use Milvus or pymilvus.

Environment: module load conda && module load pytorch/2.6.0, then pip install -r faiss_based/requirements-faiss.txt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ingest import (  # noqa: E402
    build_point_ids,
    load_and_validate,
    l2_normalize,
    split_deterministic,
)

import faiss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAISS IVF-Flat ingest (no Milvus)")
    p.add_argument("--cluster-csv", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--index-out",
        default=None,
        help="Path for .faiss index file (default: <output-dir>/{model}_ivf_flat.faiss)",
    )
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=0.3,
        help="Eval fraction when --n-eval is not set.",
    )
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--nlist", type=int, default=200, help="IVF coarse clusters")
    p.add_argument(
        "--nprobe-default",
        type=int,
        default=32,
        help="Stored in metadata as default nprobe (query_eval_faiss overrides via CLI).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, repr_cols = load_and_validate(args.cluster_csv)
    df = df.copy()
    df["point_id"] = build_point_ids(df)

    if args.split_manifest:
        print(f"[ingest_faiss] Using pre-aligned split manifest: {args.split_manifest}")
        split_df = pd.read_csv(args.split_manifest)
        df_with_split = df.merge(split_df[["point_id", "split"]], on="point_id", how="inner")
        train_df = df_with_split[df_with_split["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        eval_df = df_with_split[df_with_split["split"] == "eval"].drop(columns=["split"]).reset_index(drop=True)
    else:
        if args.n_eval is not None:
            n_eval = args.n_eval
        else:
            n_eval = int(round(len(df) * float(args.eval_fraction)))
            n_eval = max(1, min(n_eval, len(df) - 1))
        print(
            f"  Using n_eval={n_eval:,} (eval_fraction={args.eval_fraction}, "
            f"n_eval override={args.n_eval is not None})"
        )
        train_df, eval_df = split_deterministic(df, n_eval, args.seed)

    train_df = train_df.copy()
    train_df["milvus_id"] = np.arange(len(train_df), dtype=np.int64)

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

    pickle_path = os.path.join(args.output_dir, f"{args.model}_split.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({"train_df": train_df, "eval_df": eval_df}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved split pickle:   {pickle_path}")

    train_vecs = np.nan_to_num(
        train_df[repr_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    train_vecs_norm = l2_normalize(train_vecs).astype(np.float32)
    n_train, dim = train_vecs_norm.shape

    nlist = int(args.nlist)
    if nlist < 1:
        raise ValueError(f"nlist must be >= 1, got {nlist}")
    if n_train < 2:
        raise ValueError(f"Need at least 2 train vectors for IVF, got {n_train}")
    if nlist > n_train:
        print(f"  [ingest_faiss] Warning: nlist={nlist} > n_train={n_train}; clamping nlist to n_train")
        nlist = n_train

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = int(args.nprobe_default)

    t0 = time.perf_counter()
    index.train(train_vecs_norm)
    index.add(train_vecs_norm)
    elapsed = time.perf_counter() - t0
    print(f"  FAISS IVF_FLAT trained+added {n_train:,} vectors (dim={dim}, nlist={nlist}) in {elapsed:.1f}s")

    index_out = args.index_out or os.path.join(args.output_dir, f"{args.model}_ivf_flat.faiss")
    faiss.write_index(index, index_out)
    print(f"  Wrote FAISS index: {index_out}")

    meta = {
        "model": args.model,
        "dim": dim,
        "nlist": nlist,
        "nprobe_default": int(args.nprobe_default),
        "metric": "METRIC_INNER_PRODUCT",
        "note": "Vectors are L2-normalized; IP equals cosine similarity ranking.",
        "index_path": os.path.abspath(index_out),
        "n_train": n_train,
    }
    meta_path = index_out + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote metadata: {meta_path}")
    print("[ingest_faiss] Done.\n")


if __name__ == "__main__":
    main()
