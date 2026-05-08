#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_inf_ip_1s_main
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

PARQUET_ROOT="<data-root>/output/splits_1s_ip/ibgbi_test"
MODEL_PATH="<external-netburst-root>/src/train/NetBurstIBGBI_ip_1s_main_netburst_CE_intK_manifest"
NZ_THRESH=100

set -euo pipefail

echo "Starting NetBurst inference (NetBurst-essential ar_predict.py): ip_1s_main"

cd <repo-root>/src/train

torchrun --nproc_per_node=4 ar_predict.py "${PARQUET_ROOT}" \
  --model "${MODEL_PATH}" \
  --use_precomputed_context_forecast \
  --no_sampling \
  --nz_thresh ${NZ_THRESH} --bi_thresh ${NZ_THRESH} \
  --save_pkl netburst_ibgbi_manifest_ip_1s_main.pkl \
  --per_example_csv per_example_metrics_netburst_ibgbi_manifest_ip_1s_main.csv \
  --aggregate_json final_metrics_netburst_ibgbi_manifest_ip_1s_main.json
