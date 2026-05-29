#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCE_SPARSE="${SOURCE_SPARSE:-<data-root>/sparse_timeseries_1s_ip}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
MODE="${MODE:-direct}"  # direct | manifest
REBUILD_TINY_SPARSE="${REBUILD_TINY_SPARSE:-auto}"  # auto | 1 | 0

TINY_SPARSE_DIR="${REPO_ROOT}/examples/tiny/data/sparse_timeseries_1s_ip_tiny"
MANIFEST_PATH="${REPO_ROOT}/examples/tiny/sparse_to_ibgbi_splits_manifest_tiny.yaml"
PREPROCESS_DIR="${REPO_ROOT}/src/preprocess"
IBGBI_OUT="${REPO_ROOT}/examples/tiny/outputs/ibgbi_1s_ip_tiny"
SPLITS_OUT="${REPO_ROOT}/examples/tiny/outputs/splits_1s_ip_tiny"

shopt -s nullglob
tiny_task_matches=("${TINY_SPARSE_DIR}"/task_*)
shopt -u nullglob

should_rebuild="1"
if [[ "${REBUILD_TINY_SPARSE}" == "0" ]]; then
  should_rebuild="0"
elif [[ "${REBUILD_TINY_SPARSE}" == "auto" && ${#tiny_task_matches[@]} -gt 0 ]]; then
  should_rebuild="0"
fi

if [[ "${should_rebuild}" == "1" ]]; then
  echo "[1/3] Building tiny sparse dataset (${NUM_SAMPLES} rows) from ${SOURCE_SPARSE}"
  python "${REPO_ROOT}/examples/tiny/make_tiny_sparse_dataset.py" \
    --input_dir "${SOURCE_SPARSE}" \
    --output_dir "${TINY_SPARSE_DIR}" \
    --num_samples "${NUM_SAMPLES}"
else
  echo "[1/3] Reusing existing tiny sparse dataset at ${TINY_SPARSE_DIR}"
fi

if [[ "${MODE}" == "manifest" ]]; then
  echo "[2/3] Running preprocess flow through manifest driver"
  (
    cd "${PREPROCESS_DIR}"
    python run_sparse_to_ibgbi_splits_manifest.py --manifest "${MANIFEST_PATH}"
  )
else
  echo "[2/3] Running sparse -> IBG/BI (same script used by manifest jobs)"
  (
    cd "${PREPROCESS_DIR}"
    python SparseToIBGBIFromSparse.py \
      --sparse_input "${TINY_SPARSE_DIR}" \
      --entity_type ip \
      --output_dir "${IBGBI_OUT}" \
      --burst_column inbound \
      --threshold 100 \
      --bin_ms 1000 \
      --min_seq_len 3 \
      --max_seq_len 9000 \
      --disable_slurm_shard
  )

  echo "[3/3] Creating train/test + context/forecast splits"
  (
    cd "${PREPROCESS_DIR}"
    python create_train_test_splits.py \
      --ibgbi_dir "${IBGBI_OUT}/task_*" \
      --entity_type ip \
      --output_dir "${SPLITS_OUT}" \
      --min_bursts 3 \
      --train_ratio 0.7 \
      --context_ratio 0.7 \
      --seed 42 \
      --sparse_dir "${TINY_SPARSE_DIR}" \
      --alignment_mode burst_timestamp \
      --burst_column inbound \
      --burst_threshold 100 \
      --bin_ms 1000 \
      --min_seq_len 3 \
      --max_seq_len 9000
  )
fi

echo "Smoke preprocess pipeline complete."
echo "Tiny sparse dataset: ${TINY_SPARSE_DIR}"
echo "IBG/BI output: ${IBGBI_OUT}"
echo "Split outputs: ${SPLITS_OUT}"
