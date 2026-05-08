#!/usr/bin/env python3
"""
End-to-end for one model: ingest_faiss.py -> query_eval_faiss.py (no Milvus).

Environment (NERSC / similar):
  module load conda && module load pytorch/2.6.0
  pip install -r faiss_based/requirements-faiss.txt

Example:
  python run_pipeline_faiss.py \
    --cluster-csv /path/to/clusters.csv \
    --model chronos2 \
    --output-dir /path/to/out \
    --nlist 200 --nprobe 32
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd

from benchmark_repro import pipeline_faiss_single_model_payload, write_benchmark_config

PARQUET_DEFAULT = (
    "<external-netburst-root>/output/"
    "netreplica_converted_6Mbps_100ms_Sparse_ip"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAISS ingest + query (IVF-Flat, faiss-cpu)")
    p.add_argument("--cluster-csv", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--index-out",
        default=None,
        help="FAISS index path (default: <output-dir>/{model}_ivf_flat.faiss)",
    )
    p.add_argument("--split-manifest", default=None)
    p.add_argument("--parquet-path", default=PARQUET_DEFAULT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-fraction", type=float, default=0.3)
    p.add_argument("--n-eval", type=int, default=None)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--nlist", type=int, default=200)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument(
        "--no-cluster-filter",
        action="store_true",
        default=False,
    )
    p.add_argument("--debug-plot", action="store_true")
    p.add_argument("--debug-plot-max", type=int, default=100)
    p.add_argument("--debug-plot-dir", default=None)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    pretty = " \\\n    ".join(cmd)
    print(f"\n>>> {pretty}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    src_dir = os.path.dirname(os.path.abspath(__file__))
    manifest = os.path.join(args.output_dir, f"{args.model}_split_manifest.csv")
    eval_vecs = os.path.join(args.output_dir, f"{args.model}_eval_vecs.npy")
    index_path = args.index_out or os.path.join(args.output_dir, f"{args.model}_ivf_flat.faiss")

    ingest_cmd = [
        sys.executable,
        os.path.join(src_dir, "faiss_based", "ingest_faiss.py"),
        "--cluster-csv",
        args.cluster_csv,
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
        "--index-out",
        index_path,
        "--seed",
        str(args.seed),
        "--nlist",
        str(args.nlist),
        "--nprobe-default",
        str(args.nprobe),
    ]
    if args.n_eval is not None:
        ingest_cmd.extend(["--n-eval", str(args.n_eval)])
    else:
        ingest_cmd.extend(["--eval-fraction", str(args.eval_fraction)])
    if args.split_manifest:
        ingest_cmd.extend(["--split-manifest", args.split_manifest])
    run(ingest_cmd)

    query_cmd = [
        sys.executable,
        os.path.join(src_dir, "faiss_based", "query_eval_faiss.py"),
        "--split-manifest",
        manifest,
        "--eval-vecs",
        eval_vecs,
        "--faiss-index",
        index_path,
        "--model",
        args.model,
        "--output-dir",
        args.output_dir,
        "--parquet-path",
        args.parquet_path,
        "--topk",
        str(args.topk),
        "--nprobe",
        str(args.nprobe),
    ]
    if args.no_cluster_filter:
        query_cmd.append("--no-cluster-filter")
    if args.debug_plot:
        query_cmd.append("--debug-plot")
        query_cmd.extend(["--debug-plot-max", str(args.debug_plot_max)])
        if args.debug_plot_dir:
            query_cmd.extend(["--debug-plot-dir", args.debug_plot_dir])
    run(query_cmd)

    manifest_df = pd.read_csv(manifest)
    n_eval_actual = int(manifest_df["split"].eq("eval").sum())

    cfg = pipeline_faiss_single_model_payload(
        model=args.model,
        cluster_csv=args.cluster_csv,
        output_dir=os.path.abspath(args.output_dir),
        faiss_index_path=index_path,
        parquet_path=args.parquet_path,
        split_manifest=args.split_manifest,
        seed=args.seed,
        n_eval=n_eval_actual,
        nlist=args.nlist,
        nprobe=args.nprobe,
        topk=args.topk,
        eval_fraction=args.eval_fraction if args.n_eval is None and not args.split_manifest else None,
        query_mode="global_ivf" if args.no_cluster_filter else "filtered_cluster_ivf",
    )
    write_benchmark_config(
        args.output_dir,
        cfg,
        filename=f"benchmark_config_{args.model}.json",
    )

    print(f"\n{'='*60}")
    print(f"FAISS pipeline complete for model '{args.model}'")
    print(f"Output directory: {args.output_dir}")
    print(f"  {args.model}_ivf_flat.faiss")
    print(f"  {args.model}_split_manifest.csv")
    print(f"  {args.model}_distances.csv")
    print(f"  {args.model}_query_times.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
