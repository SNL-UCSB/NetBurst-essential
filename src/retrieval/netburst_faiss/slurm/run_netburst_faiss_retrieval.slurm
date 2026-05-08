#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_faiss_retrieval
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load conda
module load pytorch/2.6.0

# Optional: install if not already available in your env
# pip install faiss-cpu numpy pandas scipy pymilvus pyarrow

# ---- User inputs ----
REPO_ROOT="<repo-root>"
CLUSTER_CSV="/path/to/artifacts/retrieval_inputs/clustering/k_500_labels.csv"
PARQUET_PATH="/path/to/netreplica_parquet_root"
OUTPUT_DIR="/path/to/artifacts/faiss_runs/netburst_k500"
MODEL_NAME="netburst"
NLIST=500
NPROBE=32
TOPK=1
# Optional predefined split manifest:
SPLIT_MANIFEST=""
# ---------------------

mkdir -p "${OUTPUT_DIR}"

echo "[$(date)] Starting NetBurst FAISS retrieval"
echo "Repo: ${REPO_ROOT}"
echo "Cluster CSV: ${CLUSTER_CSV}"
echo "Output dir: ${OUTPUT_DIR}"
echo "nlist=${NLIST} nprobe=${NPROBE} topk=${TOPK}"

cd "${REPO_ROOT}"

CMD=(
  python src/retrieval/netburst_faiss/run_pipeline_faiss.py
  --cluster-csv "${CLUSTER_CSV}"
  --model "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --parquet-path "${PARQUET_PATH}"
  --nlist "${NLIST}"
  --nprobe "${NPROBE}"
  --topk "${TOPK}"
)

if [[ -n "${SPLIT_MANIFEST}" ]]; then
  CMD+=(--split-manifest "${SPLIT_MANIFEST}")
fi

"${CMD[@]}"

echo "[$(date)] Finished NetBurst FAISS retrieval"
echo "Artifacts in: ${OUTPUT_DIR}"
