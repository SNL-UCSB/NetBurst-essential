#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=agg_flow_window
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00
#
# Single-node: AggregateToWindowSize.py runs one srun task (not sharded across nodes).
#
# Flow Parquet (src_ip,dst_ip,src_port,dst_port,timestamp,size) -> 100ms-style aggregates
# for NonGraphSparseTimeseries. Configure via environment:
#   AGGREGATE_INPUT   (required) comma-separated Parquet paths
#   AGGREGATE_OUTPUT  (required) output directory
#   AGGREGATE_WINDOW_MS  (default 100)
#   AGGREGATE_NO_RANK_PORTS  set to 1 to keep literal ports (--no-rank-service-ports)
#
# Example:
#   sbatch --export=ALL,AGGREGATE_INPUT=/path/to/mawi_parts,AGGREGATE_OUTPUT=/pscratch/.../mawi_aggregate_100ms \
#     aggregate_flow_to_window.sl

set -euo pipefail

module load conda
module load pytorch/2.6.0
export PYTHONUNBUFFERED=1

SCRIPT_DIR="${SCRIPT_DIR:-<repo-root>/src/preprocess/OtherDatasets_preprocessing}"
AGG_SCRIPT="${SCRIPT_DIR}/AggregateToWindowSize.py"

: "${AGGREGATE_INPUT:?Set AGGREGATE_INPUT to comma-separated Parquet input paths}"
: "${AGGREGATE_OUTPUT:?Set AGGREGATE_OUTPUT to output directory}"

WINDOW_MS="${AGGREGATE_WINDOW_MS:-100}"
RANK_EXTRA=()
if [[ "${AGGREGATE_NO_RANK_PORTS:-0}" == "1" ]]; then
  RANK_EXTRA=(--no-rank-service-ports)
fi

echo "AGGREGATE_INPUT=${AGGREGATE_INPUT}"
echo "AGGREGATE_OUTPUT=${AGGREGATE_OUTPUT}"
echo "AGGREGATE_WINDOW_MS=${WINDOW_MS}"

srun python3 "${AGG_SCRIPT}" \
  --input_dir "${AGGREGATE_INPUT}" \
  --output_dir "${AGGREGATE_OUTPUT}" \
  --window_ms "${WINDOW_MS}" \
  "${RANK_EXTRA[@]}"

echo "aggregate_flow_to_window complete."
