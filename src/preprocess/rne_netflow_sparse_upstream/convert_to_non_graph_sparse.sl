#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH --nodes=16
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=convert_non_graph_sparse
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=256
#SBATCH --time=2:00:00

module load conda
module load pytorch/2.6.0
export PYTHONUNBUFFERED=1
# Python sources live with this .sl file; Parquet outputs go to RNE_NETFLOW_DATA (same layout as RnE-NetFlow repo).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
RNE_NETFLOW_DATA="${RNE_NETFLOW_DATA:-${RNE_NETFLOW_PREPROCESS:-<data-root>/RnE-NetFlow_Summer_25/preprocess}}"
cd "${RNE_NETFLOW_DATA}"

# Set FLOW_EDGE_HITS to your flow edge parquet input, e.g.:
#   sbatch --export=ALL,FLOW_EDGE_HITS=/path/to/flow_edge_hits.parquet convert_to_non_graph_sparse.sl
FLOW_EDGE_HITS="${FLOW_EDGE_HITS:-<data-root>/RnE-NetFlow_Summer_25/preprocess/flow_edge_hits_2.parquet}"

# Sharding derived from Slurm if available
TASK_ID=${SLURM_PROCID:-0}
NUM_TASKS=${SLURM_NTASKS:-1}

echo "Using sharding TASK_ID=${TASK_ID} of NUM_TASKS=${NUM_TASKS}"

# Service-level (dst_ip + service_port): build timeseries from flow-edge hits, then sparse arrays.
echo "[RUN] Timeseries (service) on ${NUM_TASKS} tasks"
srun -n ${NUM_TASKS} -l bash -lc 'python3 '"${SCRIPT_DIR}"'/ConvertToTimeseries.py \
  --flow_edge_hits_file "'"${FLOW_EDGE_HITS}"'" \
  --output_parquet timeseries_out \
  --num_tasks '"${NUM_TASKS}"' \
  --task_id ${SLURM_PROCID}'

echo "[RUN] Sparse (service) on ${NUM_TASKS} tasks"
srun -n ${NUM_TASKS} -l bash -lc 'python3 '"${SCRIPT_DIR}"'/ConvertToNonGraphSparseFormat.py \
  --input_parquet timeseries_out/task_${SLURM_PROCID} \
  --output_parquet sparse_out \
  --group_cols meta_dst_ip,service_port,session_id \
  --bin_seconds 60 \
  --min_seq_len 3 \
  --threshold 2080000.0 \
  --num_tasks 1 \
  --task_id ${SLURM_PROCID}'

# IP-level (dst only): build timeseries from flow-edge hits, then sparse arrays.
echo "[RUN] Timeseries (IP) on ${NUM_TASKS} tasks"
srun -n ${NUM_TASKS} -l bash -lc 'python3 '"${SCRIPT_DIR}"'/ConvertToTimeseriesIP.py \
  --flow_edge_hits_file "'"${FLOW_EDGE_HITS}"'" \
  --output_parquet timeseries_ip_out \
  --num_tasks '"${NUM_TASKS}"' \
  --task_id ${SLURM_PROCID}'

echo "[RUN] Sparse (IP) on ${NUM_TASKS} tasks"
srun -n ${NUM_TASKS} -l bash -lc 'python3 '"${SCRIPT_DIR}"'/ConvertToNonGraphSparseFormat.py \
  --input_parquet timeseries_ip_out/task_${SLURM_PROCID} \
  --output_parquet sparse_ip_out \
  --group_cols meta_dst_ip,session_id \
  --bin_seconds 60 \
  --min_seq_len 3 \
  --threshold 2136000.0 \
  --num_tasks 1 \
  --task_id ${SLURM_PROCID}'
