#!/usr/bin/env bash
# Submit NetBurst pretrain jobs. Usage:
#
#   cd <repo-root>/src/train/slurm_files
#   export SLURM_ACCOUNT=<SLURM_ACCOUNT>
#   export REPO_ROOT=<repo-root>
#   export PARQUET_ROOT=/path/to/ibgbi_train
#   export MODEL_PATH=/path/to/chronos-t5-small/snapshot
#
#   ./submit_pretrain.sh baseline          # num_local_bins=0
#   ./submit_pretrain.sh local16           # num_local_bins=16
#   ./submit_pretrain.sh smoke             # 1-GPU debug queue smoke
#   ./submit_pretrain.sh finetune          # finetune from CKPT_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-baseline}"

case "${MODE}" in
  baseline)
    sbatch --export=ALL "${SCRIPT_DIR}/netburst_pretrain_ibgbi_ip_1s_main.sl"
    ;;
  local16|local)
    export NUM_LOCAL_BINS="${NUM_LOCAL_BINS:-16}"
    sbatch --export=ALL "${SCRIPT_DIR}/netburst_pretrain_ibgbi_local_bins_ip_1s_main.sl"
    ;;
  smoke)
    sbatch --export=ALL "${SCRIPT_DIR}/netburst_pretrain_smoke_1gpu.sl"
    ;;
  finetune)
    if [[ -z "${CKPT_DIR:-}" ]]; then
      echo "ERROR: set CKPT_DIR to a checkpoint directory for finetune mode." >&2
      exit 1
    fi
    sbatch --export=ALL "${SCRIPT_DIR}/netburst_finetune_ibgbi_ip_1s_main.sl"
    ;;
  *)
    echo "Usage: $0 {baseline|local16|smoke|finetune}" >&2
    exit 1
    ;;
esac
