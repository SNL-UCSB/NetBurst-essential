#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH --nodes=16
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=ping_sparse_3m_t4
#SBATCH --ntasks=16
#SBATCH --time=12:00:00

# Convert ping latency CSV to sparse timeseries (3-minute bins).
# Optional: add --compute_quantiles to also write reference quantiles to OUT_DIR_BASE_quantiles
# for use with PingLatencyToIBGBI (--threshold_quantiles_path and --threshold_quantile).
#
# Override defaults at submit time, e.g.:
# INPUT_PATH_GLOB="/path/*.csv" OUT_DIR_BASE="/path/out" sbatch PingLatencyToSparseTimeseries_3min_threshold_4.slurm
# COMPUTE_QUANTILES=1 sbatch ...  # to enable --compute_quantiles

: "${INPUT_PATH_GLOB:=<data-root>/pinot_pings.csv}"
: "${OUT_DIR_BASE:=<data-root>/ping_latency_sparse_ts_3min_thresh_4ms}"
: "${BIN_SEC:=180}"
: "${THRESH_MS:=0}"
: "${MIN_SEQ_LEN:=3}"
: "${MAX_SEQ_LEN:=2147483647}"
: "${SEQ_LEN:=480}"
: "${SHUFFLE_PARTITIONS:=}"
: "${COMPUTE_QUANTILES:=0}"

SCRIPT="${PING_SPARSE_SCRIPT:-<repo-root>/src/preprocess/PingLatencyToSparseTimeseries.py}"

module load conda
module load pytorch/2.6.0

EXTRA_ARGS=()
[[ -n "${SHUFFLE_PARTITIONS}" ]] && EXTRA_ARGS+=(--shuffle_partitions "${SHUFFLE_PARTITIONS}")
[[ "${COMPUTE_QUANTILES}" == "1" ]] && EXTRA_ARGS+=(--compute_quantiles)

echo "Running sparse timeseries conversion with threshold ${THRESH_MS} ms"
srun python3 "${SCRIPT}" \
  --input_paths ${INPUT_PATH_GLOB} \
  --output_dir "${OUT_DIR_BASE}" \
  --threshold_ms "${THRESH_MS}" \
  --bin_sec "${BIN_SEC}" \
  --min_seq_len "${MIN_SEQ_LEN}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --seq_len "${SEQ_LEN}" \
  "${EXTRA_ARGS[@]}"
