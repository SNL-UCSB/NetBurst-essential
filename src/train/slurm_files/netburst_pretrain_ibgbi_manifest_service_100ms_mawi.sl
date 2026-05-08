#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_pre_service_100ms_mawi_minlen2
#SBATCH --time=24:00:00
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

PARQUET_ROOT="<data-root>/output/splits_100ms_mawi_service/ibgbi_train"
MODEL_PATH="<data-root>/models--amazon--chronos-t5-small/snapshots/476a71b73e6205f7987e811a81f355b9791c9256"
PRETRAIN_CKPT_ROOT="<repo-root>/src/train/checkpoints/netburst_ibgbi_manifest"
SAVE_DIR="${PRETRAIN_CKPT_ROOT}/NetBurstIBGBI_service_100ms_mawi_netburst_CE_intK_manifest_minlen2"
EPOCHS=100
BATCH_SIZE=16
MAX_LEN=128
TRAIN_FRAC=0.8
MIN_LEN=2
NUM_TOKENS=4096
CENTER_CLIP=1000000
LR=1e-3
IBG_INTEGER_BINS=-1
LOSS_WEIGHT_BI=1.0
LOSS_WEIGHT_IBG=2.0

set -euo pipefail

echo "Starting NetBurst pretraining (NetBurst-essential pretrain_twin.py): service_100ms_mawi (CE + IntegerIBGBins + per-head projections)"

cd <repo-root>/src/train

torchrun --nproc_per_node=4 pretrain_twin.py "${PARQUET_ROOT}" \
    --model "${MODEL_PATH}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --train_frac ${TRAIN_FRAC} \
    --min_len ${MIN_LEN} \
    --max_len ${MAX_LEN} \
    --num_tokens ${NUM_TOKENS} \
    --mse_mode centers \
    --center_clip ${CENTER_CLIP} \
    --save_dir "${SAVE_DIR}" \
    --lr ${LR} \
    --ibg_integer_bins ${IBG_INTEGER_BINS} \
    --use_ce_loss \
    --freeze_soft_ce_sharpness \
    --loss_weight_bi ${LOSS_WEIGHT_BI} \
    --loss_weight_ibg ${LOSS_WEIGHT_IBG}
