#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMOKE_SCRIPT="${REPO_ROOT}/examples/tiny/run_smoke_full_pipeline.sh"
TINY_SPARSE_DIR="${REPO_ROOT}/examples/tiny/data/sparse_timeseries_1s_ip_tiny"

# Space-separated stage list override.
# Default runs the full smoke chain from checked-in sparse parquet.
STAGES="${STAGES:-preprocess tsfresh train forecast repr cluster interpret retrieval}"

# Runtime workaround path for Perlmutter-like MKL/OpenMP environments.
# Override with:
#   export GOMP_PRELOAD_PATH=/path/to/libgomp.so.1
GOMP_PRELOAD_PATH="${GOMP_PRELOAD_PATH:-/global/common/software/nersc9/pytorch/2.6.0/lib/libgomp.so.1}"

if [[ ! -x "${SMOKE_SCRIPT}" ]]; then
  echo "ERROR: missing smoke script: ${SMOKE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${TINY_SPARSE_DIR}" ]]; then
  echo "ERROR: missing tiny sparse dataset dir: ${TINY_SPARSE_DIR}" >&2
  exit 1
fi

shopt -s nullglob
task_dirs=("${TINY_SPARSE_DIR}"/task_*)
shopt -u nullglob
if [[ ${#task_dirs[@]} -eq 0 ]]; then
  echo "ERROR: no task_* parquet shards found under ${TINY_SPARSE_DIR}" >&2
  exit 1
fi

run_stage() {
  local stage="$1"
  echo
  echo "===== Running stage: ${stage} ====="

  # These two stages may need GNU OpenMP preload on certain MKL runtimes.
  if [[ "${stage}" == "tsfresh" || "${stage}" == "retrieval" ]]; then
    if [[ -f "${GOMP_PRELOAD_PATH}" ]]; then
      LD_PRELOAD="${GOMP_PRELOAD_PATH}" STAGE="${stage}" bash "${SMOKE_SCRIPT}"
      return
    fi
    echo "WARN: GOMP preload library not found at ${GOMP_PRELOAD_PATH}" >&2
    echo "WARN: continuing without LD_PRELOAD for stage=${stage}" >&2
  fi

  STAGE="${stage}" bash "${SMOKE_SCRIPT}"
}

echo "Running tiny end-to-end smoke from checked-in parquet fixture"
echo "Dataset root: ${TINY_SPARSE_DIR}"
echo "Stages: ${STAGES}"

for stage in ${STAGES}; do
  run_stage "${stage}"
done

echo
echo "Tiny end-to-end smoke from parquet completed successfully."
