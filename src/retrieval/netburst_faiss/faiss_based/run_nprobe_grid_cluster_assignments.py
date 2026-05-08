#!/usr/bin/env python3
"""
Run per-query nprobe grid cluster assignment collection for multiple models.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run nprobe grid retrieval collection for chronos2/tsfresh/netburst."
    )
    p.add_argument("--build-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--models",
        nargs="+",
        default=["chronos2", "tsfresh", "netburst"],
    )
    p.add_argument(
        "--nprobes",
        type=int,
        nargs="+",
        required=True,
    )
    p.add_argument("--nprobe-ref", type=int, default=None)
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
    collector = os.path.join(this_dir, "collect_cluster_ids_nprobe_grid.py")

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
            collector,
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
            "--nprobes",
            *[str(v) for v in args.nprobes],
            "--progress-every",
            str(args.progress_every),
        ]
        if args.nprobe_ref is not None:
            cmd.extend(["--nprobe-ref", str(args.nprobe_ref)])
        if args.use_cluster_filter:
            cmd.append("--use-cluster-filter")

        run(cmd)

        ref = args.nprobe_ref if args.nprobe_ref is not None else args.nprobes[0]
        summary_csv = os.path.join(
            args.output_dir, f"{model}_nprobe_grid_change_summary_vs_{ref}.csv"
        )
        if os.path.exists(summary_csv):
            summaries.append(pd.read_csv(summary_csv))

    if not summaries:
        print("\nNo model summaries generated.", flush=True)
        return

    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.sort_values(["model", "nprobe"]).reset_index(drop=True)
    ref = args.nprobe_ref if args.nprobe_ref is not None else args.nprobes[0]
    combined_out = os.path.join(
        args.output_dir, f"all_models_nprobe_grid_change_summary_vs_{ref}.csv"
    )
    combined.to_csv(combined_out, index=False)
    print("\n=== Combined summary ===")
    print(combined.to_string(index=False))
    print(f"Saved combined summary: {combined_out}", flush=True)


if __name__ == "__main__":
    main()
