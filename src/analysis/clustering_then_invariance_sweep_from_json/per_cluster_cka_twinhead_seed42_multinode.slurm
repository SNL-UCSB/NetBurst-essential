#!/bin/bash
# Per-cluster Cohen + linear CKA for existing clustering_then_invariance_cosine_netburst_20260322
# outputs (no re-clustering). Each task: full cluster_analysis reload (invariant_scores,
# cluster variance CSV, cluster feature-rank CSV) plus per-cluster CSVs;
# skips CH/DB and heatmap / feature_mean_var_z_pop rewrites.
#
# JSON: clustering_then_invariance_multi_node_per_run_parallel_k_netburst_20260322.json
# K from SLURM_PROCID (0→10, 1→100, 2→500, 3→1000 with default k_values).
# Requires: one task per K — len(k_values) must match --nodes / --ntasks. Seed 42; run_1 from JSON.
#
# Submit from anywhere:
#   sbatch <external-netburst-root>/src/analysis/clustering_then_invariance_sweep_from_json/per_cluster_cka_netburst_seed42_multinode.slurm
#
#SBATCH -q premium
#SBATCH -C cpu
# Must match len(k_values) in the JSON (default netburst JSON: 10, 100, 500, 1000).
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_pccka_s42
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=256
#SBATCH --output=<external-netburst-root>/output/clustering_then_invariance_cosine_netburst_20260322/logs/per_cluster_cka_seed42_%j_%t.out
#SBATCH --error=<external-netburst-root>/output/clustering_then_invariance_cosine_netburst_20260322/logs/per_cluster_cka_seed42_%j_%t.err

set -euo pipefail

module load conda
module load pytorch/2.6.0

WORKDIR=<external-netburst-root>/src/analysis
JSON="${WORKDIR}/clustering_then_invariance_sweep_from_json/clustering_then_invariance_multi_node_per_run_parallel_k_netburst_20260322.json"
TASK_PY="${WORKDIR}/clustering_then_invariance_sweep_from_json/run_per_cluster_cka_slurm_task.py"

mkdir -p <external-netburst-root>/output/clustering_then_invariance_cosine_netburst_20260322/logs

cd "${WORKDIR}"

srun python3 "${TASK_PY}" --json "${JSON}" --seed 42
