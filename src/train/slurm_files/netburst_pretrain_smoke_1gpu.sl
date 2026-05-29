#!/bin/bash
#SBATCH -q debug
#SBATCH --account=${SLURM_ACCOUNT:-<SLURM_ACCOUNT>}
#SBATCH --job-name=netburst_smoke
#SBATCH --time=00:30:00
#SBATCH --licenses=scratch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Quick smoke: 1 GPU, few epochs. Use tiny ibgbi splits from examples/tiny.
#
#   export PARQUET_ROOT=${REPO_ROOT}/examples/tiny/outputs/splits_1s_ip_tiny/ibgbi_train
#   export NUM_LOCAL_BINS=16   # or 0 for legacy path
#   sbatch slurm_files/netburst_pretrain_smoke_1gpu.sl

set -euo pipefail

module load conda
module load pytorch/2.6.0

REPO_ROOT="${REPO_ROOT:-<repo-root>}"
TRAIN_DIR="${REPO_ROOT}/src/train"

PARQUET_ROOT="${PARQUET_ROOT:-${REPO_ROOT}/examples/tiny/outputs/splits_1s_ip_tiny/ibgbi_train}"
MODEL_PATH="${MODEL_PATH:-amazon/chronos-t5-small}"
SAVE_DIR="${SAVE_DIR:-${TRAIN_DIR}/checkpoints/smoke_pretrain}"
NUM_LOCAL_BINS="${NUM_LOCAL_BINS:-0}"

cd "${TRAIN_DIR}"

torchrun --nproc_per_node=1 pretrain_twin.py "${PARQUET_ROOT}" \
    --model "${MODEL_PATH}" \
    --epochs 1 \
    --batch_size 4 \
    --limit 32 \
    --min_len 3 \
    --max_len 128 \
    --num_tokens 256 \
    --save_dir "${SAVE_DIR}" \
    --ibg_integer_bins -1 \
    --num_local_bins "${NUM_LOCAL_BINS}" \
    --use_ce_loss \
    --freeze_soft_ce_sharpness
