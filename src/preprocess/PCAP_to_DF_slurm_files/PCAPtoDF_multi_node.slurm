#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH --nodes=16
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=pcap_to_parquet
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=256
#SBATCH --time=24:00:00

# Example Slurm job: directory of PCAPs (one job typically uses pre-split PCAPs)
# → per-MPI-rank Parquet files via PcapToDf_multi_node.py (tshark + pyarrow).
#
# Before submitting: set account, nodes, ntasks, cpus-per-task, and paths.
# Optional env overrides at submit time:
#   PCAP_DIR=... OUT_DIR=... sbatch PCAPtoDF_multi_node.slurm
#   COPY_WORKERS=8 TSHARK_WORKERS=128 sbatch PCAPtoDF_multi_node.slurm
#   SCRIPT=/path/to/PcapToDf_multi_node.py sbatch PCAPtoDF_multi_node.slurm

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT:-$(cd "$HERE/.." && pwd)/PcapToDf_multi_node.py}"

PCAP_DIR="${PCAP_DIR:-/path/to/your_pcap_split}"
OUT_DIR="${OUT_DIR:-/path/to/your_parquet_output}"

module load conda
module load pytorch/2.3.1
module load parallel
module load cray-mpich

EXTRA=()
[[ -n "${COPY_WORKERS:-}" ]] && EXTRA+=(--copy_workers "${COPY_WORKERS}")
[[ -n "${TSHARK_WORKERS:-}" ]] && EXTRA+=(--tshark_workers "${TSHARK_WORKERS}")

srun python "$SCRIPT" "$PCAP_DIR" "$OUT_DIR" "${EXTRA[@]}"
