#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_retrieval_inputs
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load conda
module load pytorch/2.6.0

# ---- User inputs ----
REPO_ROOT="<repo-root>"
BI_IBG_PARQUET="/path/to/bi_ibg_parquet"
TWIN_HEAD_CKPT="/path/to/twin_head_checkpoint_dir"
ARTIFACT_ROOT="/path/to/artifacts/retrieval_inputs"
N_CLUSTERS=500
KMEANS_METRIC="cosine"
# ---------------------

REPR_OUT="${ARTIFACT_ROOT}/ip_representations.csv"
CLUSTER_OUT_DIR="${ARTIFACT_ROOT}/clustering"
CLUSTER_CSV="${CLUSTER_OUT_DIR}/k_${N_CLUSTERS}_labels.csv"

mkdir -p "${ARTIFACT_ROOT}" "${CLUSTER_OUT_DIR}"

echo "[$(date)] Starting NetBurst retrieval input prep"
echo "Repo: ${REPO_ROOT}"
echo "Representations output: ${REPR_OUT}"
echo "Cluster labels output: ${CLUSTER_CSV}"

cd "${REPO_ROOT}"

python src/train/extract_repr.py twin_ip "${BI_IBG_PARQUET}" \
  --model "${TWIN_HEAD_CKPT}" \
  --output_csv "${REPR_OUT}"

python src/analysis/clustering_analysis.py "${REPR_OUT}" \
  --method kmeans \
  --n-clusters "${N_CLUSTERS}" \
  --kmeans-metric "${KMEANS_METRIC}" \
  --minimal-cleaning \
  --out-dir "${CLUSTER_OUT_DIR}"

echo "[$(date)] Finished retrieval input prep"
echo "Cluster CSV ready at: ${CLUSTER_CSV}"
