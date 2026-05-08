#!/usr/bin/env python3
"""
One Slurm task: read clustering_then_invariance_multi_node_per_run_parallel_k_netburst_20260322.json
(or another sweep JSON with the same keys), choose K via SLURM_PROCID (same ordering as k_values:
proc 0 → first K, e.g. 10; 1 → 100; 2 → 500; 3 → 1000 for the default netburst JSON).

Runs cluster_analysis.py with --write-per-cluster-cohens-d-cka and the same path templates as the
sweep (run_1, seed from --seed, default 42). Adds --skip-cluster-quality-metrics and
--skip-population-cohen-artifacts so CH/DB and the population-level Cohen / feature_order
artifacts are not rewritten. Optional sweep JSON key ``per_cluster_cka_workers`` (int >= 2) forwards to
``--per-cluster-cka-workers`` for thread-parallel per-cluster CKA.

Still reloads clustering + TSFresh and recomputes invariant_scores.csv, the cluster variance CSV,
and cluster feature-rank CSV, then writes under each cluster_*/:
selected_features_cohens_d.csv, selected_features_cka.csv, plus at the seed directory root
all_clusters_selected_features_cka_long.csv.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json",
        required=True,
        help="clustering_then_invariance sweep JSON (paths via templates).",
    )
    ap.add_argument(
        "--proc-id",
        type=int,
        default=None,
        help="Index into k_values (default: SLURM_PROCID, else 0).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    proc = args.proc_id if args.proc_id is not None else int(os.environ.get("SLURM_PROCID", "0"))

    cfg_path = Path(args.json).expanduser()
    with cfg_path.open() as f:
        cfg = json.load(f)

    k_list = cfg["k_values"]
    run = cfg["run_values"][0]
    seed = args.seed

    if proc < 0 or proc >= len(k_list):
        print(
            f"[run_per_cluster_cka] proc_id={proc} out of range for k_values={k_list}",
            file=sys.stderr,
        )
        sys.exit(1)

    k = k_list[proc]

    def fmt(tpl: str) -> str:
        return tpl.format(k=k, run=run, seed=seed)

    workdir = Path(cfg["workdir"]).expanduser()
    analysis_py = workdir / "cluster_analysis.py"
    if not analysis_py.is_file():
        sys.exit(f"Missing {analysis_py}")

    clustering = fmt(cfg["clustering_labels_template"])
    if not Path(clustering).is_file():
        sys.exit(f"Clustering labels not found: {clustering}")

    tsfresh = str(Path(cfg["tsfresh_csv"]).expanduser())
    out_csv = fmt(cfg["out_csv_template"])
    variance_out = fmt(cfg["cluster_variance_out_csv_template"])
    rank_out = fmt(cfg["cluster_feature_rank_out_csv_template"])
    parent_dir = fmt(cfg["cluster_members_parent_dir_template"])

    min_cs = int(cfg.get("min_cluster_size", 10))

    cmd: list[str] = [
        sys.executable,
        str(analysis_py),
        "--clustering",
        clustering,
        "--tsfresh",
        tsfresh,
        "--out",
        out_csv,
        "--min-cluster-size",
        str(min_cs),
        "--cluster-variance-out",
        variance_out,
        "--cluster-feature-rank-out",
        rank_out,
        "--id-ip",
        str(cfg.get("id_ip", "ip")),
        "--id-source-file",
        str(cfg.get("id_source_file", "source_file")),
        "--write-per-cluster-cohens-d-cka",
        "--per-cluster-thresholds-dir",
        parent_dir,
        "--repr-prefix",
        str(cfg.get("repr_prefix", "repr_dim")),
        "--skip-cluster-quality-metrics",
        "--skip-population-cohen-artifacts",
    ]
    pcw = cfg.get("per_cluster_cka_workers")
    if pcw is not None and str(pcw).strip() != "":
        cmd += ["--per-cluster-cka-workers", str(int(pcw))]
    suf = cfg.get("ip_suffix_filter")
    if suf is not None and str(suf).strip():
        cmd += ["--ip-suffix-filter", str(suf).strip()]

    print(
        f"[run_per_cluster_cka] proc={proc} k={k} run={run} seed={seed}",
        flush=True,
    )
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return

    os.chdir(workdir)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
