#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH --nodes=16
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=rne_netflow_prepare_sparse
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=256
#SBATCH --time=2:00:00
#
# Rename meta_dst_ip -> ip and add source_file for sparse_ip_out and sparse_out
# (feeds sparse_*_prepared for sparse_to_ibgbi_splits_manifest RnE-NetFlow jobs).

module load conda
module load pytorch/2.6.0
export PYTHONUNBUFFERED=1

# Data I/O stays under RnE-NetFlow preprocess by default; PrepareRnENetFlowSparse.py is this bundled copy.
RNE_NETFLOW_DATA="${RNE_NETFLOW_DATA:-${RNE_NETFLOW_PREPROCESS:-<data-root>/RnE-NetFlow_Summer_25/preprocess}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
NUM_TASKS=${SLURM_NTASKS:-1}

set -e

echo "[PrepareRnENetFlowSparse] sparse_ip_out -> sparse_ip_out_prepared"
srun -n "${NUM_TASKS}" -l python3 "${SCRIPT_DIR}/PrepareRnENetFlowSparse.py" \
    --input_dir  "${RNE_NETFLOW_DATA}/sparse_ip_out" \
    --output_dir "${RNE_NETFLOW_DATA}/sparse_ip_out_prepared" \
    --source_file_value rne_netflow_ip \
    --ip_col meta_dst_ip

echo "[PrepareRnENetFlowSparse] sparse_out -> sparse_out_prepared"
srun -n "${NUM_TASKS}" -l python3 "${SCRIPT_DIR}/PrepareRnENetFlowSparse.py" \
    --input_dir  "${RNE_NETFLOW_DATA}/sparse_out" \
    --output_dir "${RNE_NETFLOW_DATA}/sparse_out_prepared" \
    --source_file_value rne_netflow_service \
    --ip_col meta_dst_ip

echo "Done prepare_rne_netflow_sparse_only."
