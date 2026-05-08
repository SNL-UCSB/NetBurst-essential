#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH --nodes=16
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=pcap_mawi_to_parquet
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=256
#SBATCH --time=24:00:00

# Example Slurm job: MAWI-style PCAP split directory → Parquet (same driver as PCAPtoDF_multi_node.slurm).
#
# Defaults are placeholders under your scratch tree; override when submitting:
#   PCAP_DIR=/path/to/mawi_split OUT_DIR=/path/to/mawi_df sbatch PCAPtoDF_multi_node_mawi.slurm

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT:-$(cd "$HERE/.." && pwd)/PcapToDf_multi_node.py}"

PCAP_DIR="${PCAP_DIR:-<data-root>/mawi_split}"
OUT_DIR="${OUT_DIR:-<data-root>/mawi_df}"

module load conda
module load pytorch/2.3.1
module load parallel
module load cray-mpich

EXTRA=()
[[ -n "${COPY_WORKERS:-}" ]] && EXTRA+=(--copy_workers "${COPY_WORKERS}")
[[ -n "${TSHARK_WORKERS:-}" ]] && EXTRA+=(--tshark_workers "${TSHARK_WORKERS}")

srun python "$SCRIPT" "$PCAP_DIR" "$OUT_DIR" "${EXTRA[@]}"
