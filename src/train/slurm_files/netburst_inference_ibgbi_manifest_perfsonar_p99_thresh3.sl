#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_inf_perf_p99_t3
#SBATCH --time=4:00:00
#SBATCH --licenses=scratch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=premium

module load conda
module load pytorch/2.6.0

PARQUET_ROOT="<data-root>/RnE-NetFlow_Summer_25/preprocess/splits_perfsonar_p99_thresh3/ibgbi_test"
# Allow consolidated launchers to pin a specific checkpoint path (default matches NetBurst pretrain SAVE_DIR).
MODEL_PATH="${CHRONOS_IBGBI_MODEL_PATH:-<external-netburst-root>/src/train/NetBurstIBGBI_perfsonar_p99_thresh3_netburst_CE_intK_manifest}"
NZ_THRESH=3

set -euo pipefail

echo "Starting NetBurst inference (NetBurst-essential ar_predict.py): perfsonar_p99_thresh3"

cd <repo-root>/src/train

torchrun --nproc_per_node=4 ar_predict.py "${PARQUET_ROOT}" \
  --model "${MODEL_PATH}" \
  --use_precomputed_context_forecast \
  --no_sampling \
  --nz_thresh ${NZ_THRESH} --bi_thresh ${NZ_THRESH} \
  --save_pkl netburst_ibgbi_manifest_perfsonar_p99_thresh3.pkl \
  --per_example_csv per_example_metrics_netburst_ibgbi_manifest_perfsonar_p99_thresh3.csv \
  --aggregate_json final_metrics_netburst_ibgbi_manifest_perfsonar_p99_thresh3.json
