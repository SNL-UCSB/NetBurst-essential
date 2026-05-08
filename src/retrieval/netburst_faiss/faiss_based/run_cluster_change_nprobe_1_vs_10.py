#!/usr/bin/env python3
"""
Run nprobe cluster-change comparison for multiple FAISS models.

Expected build artifacts in --build-dir (from prior FAISS ingest):
  {model}_split_manifest.csv
  {model}_eval_vecs.npy
  {model}_ivf_flat.faiss
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run top-1 cluster change comparison between nprobe=1 and nprobe=10."
    )
    p.add_argument("--build-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--models",
        nargs="+",
        default=["chronos2", "tsfresh", "netburst"],
        help="Models to evaluate (default: chronos2 tsfresh netburst).",
    )
    p.add_argument("--nprobe-a", type=int, default=1)
    p.add_argument("--nprobe-b", type=int, default=10)
    p.add_argument("--use-cluster-filter", action="store_true", default=False)
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    pretty = " \\\n+    ".join(cmd)
    print(f"\n>>> {pretty}\n", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    compare_script = os.path.join(this_dir, "compare_cluster_change_nprobe.py")

    summaries = []
    for model in args.models:
        split_manifest = os.path.join(args.build_dir, f"{model}_split_manifest.csv")
        eval_vecs = os.path.join(args.build_dir, f"{model}_eval_vecs.npy")
        faiss_index = os.path.join(args.build_dir, f"{model}_ivf_flat.faiss")

        missing = [
            p for p in [split_manifest, eval_vecs, faiss_index] if not os.path.exists(p)
        ]
        if missing:
            print(f"[SKIP] {model}: missing required files: {missing}", flush=True)
            continue

        cmd = [
            sys.executable,
            compare_script,
            "--split-manifest",
            split_manifest,
            "--eval-vecs",
            eval_vecs,
            "--faiss-index",
            faiss_index,
            "--model",
            model,
            "--output-dir",
            args.output_dir,
            "--nprobe-a",
            str(args.nprobe_a),
            "--nprobe-b",
            str(args.nprobe_b),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.use_cluster_filter:
            cmd.append("--use-cluster-filter")
        run(cmd)

        summary_csv = os.path.join(
            args.output_dir,
            f"{model}_nprobe_{args.nprobe_a}_vs_{args.nprobe_b}_cluster_changes_summary.csv",
        )
        if os.path.exists(summary_csv):
            summaries.append(pd.read_csv(summary_csv))

    if not summaries:
        print("\nNo model summaries were generated.", flush=True)
        return

    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.sort_values("model").reset_index(drop=True)
    combined_out = os.path.join(
        args.output_dir,
        f"all_models_nprobe_{args.nprobe_a}_vs_{args.nprobe_b}_cluster_changes_summary.csv",
    )
    combined.to_csv(combined_out, index=False)
    print("\n=== Combined summary ===")
    print(combined.to_string(index=False))
    print(f"Saved combined summary: {combined_out}", flush=True)


if __name__ == "__main__":
    main()
