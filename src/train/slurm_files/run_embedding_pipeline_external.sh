#!/usr/bin/env bash
# External compute entrypoint for embedding pipeline.
# Supports:
#   - MODE=direct : run stages on current node/process
#   - MODE=slurm  : submit this same script as an sbatch job
#
# Required env:
#   PARQUET_PATH      Sparse parquet input path (task_* root, glob, dir, or parquet file)
#   EXPERIMENT_ID     Unique run identifier
#
# Optional env:
#   REPO_ROOT         NetBurst repo root (auto-detected)
#   OUTPUT_ROOT       Parent output directory (default: ${REPO_ROOT}/outputs/embedding_pipeline)
#   CHECKPOINT_DIR    Twin-head checkpoint dir
#   MODE              direct | slurm (default: direct)
#   SLURM_*           Standard sbatch knobs when MODE=slurm
#
# Output layout:
#   ${OUTPUT_ROOT}/${EXPERIMENT_ID}/
#     status.json
#     logs/pipeline.log
#     tsfresh/tsfresh_features.csv
#     ibgbi/task_*/
#     repr/ip_representations.csv
#     clustering/k_<K>_labels.csv
#     interpretability/{invariance_scores.csv,cluster_variance.csv,cluster_feature_rank.csv}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

MODE="${MODE:-direct}"
PARQUET_PATH="${PARQUET_PATH:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/embedding_pipeline}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/src/train/checkpoints/NetBurstIBGBI_ip_1s_main_local32}"

EMB_ENTITY_TYPE="${EMB_ENTITY_TYPE:-ip}"
EMB_BURST_COLUMN="${EMB_BURST_COLUMN:-inbound}"
EMB_THRESHOLD="${EMB_THRESHOLD:-100}"
EMB_BIN_MS="${EMB_BIN_MS:-1000}"
EMB_MIN_SEQ_LEN="${EMB_MIN_SEQ_LEN:-10}"
EMB_MAX_SEQ_LEN="${EMB_MAX_SEQ_LEN:-9000}"
EMB_TSFEATURE_SET="${EMB_TSFEATURE_SET:-efficient}"
EMB_TS_N_JOBS="${EMB_TS_N_JOBS:-0}"
EMB_TS_FIXED_LEN="${EMB_TS_FIXED_LEN:-600}"
NB_PYTHON="${NB_PYTHON:-python}"

EMB_TS_CHUNK_ROWS="${EMB_TS_CHUNK_ROWS:-500}"

EMB_BATCH_SIZE="${EMB_BATCH_SIZE:-16}"
EMB_N_CLUSTERS="${EMB_N_CLUSTERS:-8}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-<SLURM_ACCOUNT>}"
SLURM_QOS="${SLURM_QOS:-regular}"
SLURM_TIME="${SLURM_TIME:-06:00:00}"
SLURM_NODES="${SLURM_NODES:-1}"
SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-1}"
SLURM_CONSTRAINT="${SLURM_CONSTRAINT:-gpu}"
SLURM_PARTITION="${SLURM_PARTITION:-}"

if [[ -z "${PARQUET_PATH}" ]]; then
  echo "ERROR: PARQUET_PATH is required." >&2
  exit 1
fi
if [[ -z "${EXPERIMENT_ID}" ]]; then
  echo "ERROR: EXPERIMENT_ID is required." >&2
  exit 1
fi

RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT_ID}"
STATUS_JSON="${RUN_DIR}/status.json"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/pipeline.log"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

json_escape() {
  python - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

write_status() {
  local state="$1"
  local stage="$2"
  local stages_done="$3"
  local error_msg="${4:-}"
  local finished_at="${5:-null}"
  local error_json="null"
  if [[ -n "${error_msg}" ]]; then
    error_json="$(json_escape "${error_msg}")"
  fi
  cat > "${STATUS_JSON}" <<EOF
{
  "experiment_id": "$(printf '%s' "${EXPERIMENT_ID}")",
  "state": "$(printf '%s' "${state}")",
  "stage": "$(printf '%s' "${stage}")",
  "stages_done": ${stages_done},
  "started_at": "${STARTED_AT}",
  "finished_at": ${finished_at},
  "error": ${error_json}
}
EOF
}

STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

submit_slurm_job() {
  local slurm_script="${RUN_DIR}/run_embedding_pipeline_${EXPERIMENT_ID}.slurm"
  local sbatch_cmd=(sbatch
    --account="${SLURM_ACCOUNT}"
    --qos="${SLURM_QOS}"
    --time="${SLURM_TIME}"
    --nodes="${SLURM_NODES}"
    --ntasks-per-node="${SLURM_NTASKS_PER_NODE}"
    --constraint="${SLURM_CONSTRAINT}"
    --gpus-per-node="${SLURM_GPUS_PER_NODE}"
    --job-name="embed-${EXPERIMENT_ID}"
    --output="${LOG_DIR}/slurm-%x-%j.out"
    --error="${LOG_DIR}/slurm-%x-%j.err"
  )
  if [[ -n "${SLURM_PARTITION}" ]]; then
    sbatch_cmd+=(--partition="${SLURM_PARTITION}")
  fi

  cat > "${slurm_script}" <<EOF
#!/bin/bash
#SBATCH --licenses=scratch
set -euo pipefail
module load conda
module load pytorch/2.6.0

export MODE=direct
export REPO_ROOT="$(printf '%s' "${REPO_ROOT}")"
export PARQUET_PATH="$(printf '%s' "${PARQUET_PATH}")"
export EXPERIMENT_ID="$(printf '%s' "${EXPERIMENT_ID}")"
export OUTPUT_ROOT="$(printf '%s' "${OUTPUT_ROOT}")"
export CHECKPOINT_DIR="$(printf '%s' "${CHECKPOINT_DIR}")"
export EMB_ENTITY_TYPE="$(printf '%s' "${EMB_ENTITY_TYPE}")"
export EMB_BURST_COLUMN="$(printf '%s' "${EMB_BURST_COLUMN}")"
export EMB_THRESHOLD="$(printf '%s' "${EMB_THRESHOLD}")"
export EMB_BIN_MS="$(printf '%s' "${EMB_BIN_MS}")"
export EMB_MIN_SEQ_LEN="$(printf '%s' "${EMB_MIN_SEQ_LEN}")"
export EMB_MAX_SEQ_LEN="$(printf '%s' "${EMB_MAX_SEQ_LEN}")"
export EMB_TSFEATURE_SET="$(printf '%s' "${EMB_TSFEATURE_SET}")"
export EMB_TS_N_JOBS="$(printf '%s' "${EMB_TS_N_JOBS}")"
export EMB_TS_FIXED_LEN="$(printf '%s' "${EMB_TS_FIXED_LEN}")"
export EMB_TS_CHUNK_ROWS="$(printf '%s' "${EMB_TS_CHUNK_ROWS}")"
export EMB_BATCH_SIZE="$(printf '%s' "${EMB_BATCH_SIZE}")"
export EMB_N_CLUSTERS="$(printf '%s' "${EMB_N_CLUSTERS}")"
export NB_PYTHON="$(printf '%s' "${NB_PYTHON}")"

bash "$(printf '%s' "${SCRIPT_DIR}")/run_embedding_pipeline_external.sh"
EOF

  write_status "queued" "submit" "[]"
  local submit_out
  submit_out="$("${sbatch_cmd[@]}" "${slurm_script}")"
  local job_id
  job_id="$(awk '{print $4}' <<< "${submit_out}")"
  echo "${submit_out}" | tee -a "${LOG_FILE}"
  echo "{\"experiment_id\":\"${EXPERIMENT_ID}\",\"mode\":\"slurm\",\"job_id\":\"${job_id}\",\"status_json\":\"${STATUS_JSON}\"}"
}

if [[ "${MODE}" == "slurm" ]]; then
  submit_slurm_job
  exit 0
fi

TSFRESH_DIR="${RUN_DIR}/tsfresh"
IBGBI_DIR="${RUN_DIR}/ibgbi"
REPR_DIR="${RUN_DIR}/repr"
CLUSTER_DIR="${RUN_DIR}/clustering"
INTERPRET_DIR="${RUN_DIR}/interpretability"
RUNTIME_CFG_DIR="${RUN_DIR}/runtime_configs"

TSFRESH_CSV="${TSFRESH_DIR}/tsfresh_features.csv"
REPR_CSV="${REPR_DIR}/ip_representations.csv"
CLUSTER_LABELS_CSV="${CLUSTER_DIR}/k_${EMB_N_CLUSTERS}_labels.csv"
RUNTIME_CLUSTER_FEATURES_CFG="${RUNTIME_CFG_DIR}/cluster_features_runtime.json"
CLUSTER_FEATURES_TEMPLATE="${REPO_ROOT}/examples/tiny/configs/cluster_features_tiny.json"

mkdir -p "${TSFRESH_DIR}" "${IBGBI_DIR}" "${REPR_DIR}" "${CLUSTER_DIR}" "${INTERPRET_DIR}" "${RUNTIME_CFG_DIR}"

if [[ ! -f "${CLUSTER_FEATURES_TEMPLATE}" ]]; then
  echo "ERROR: Missing required config template: ${CLUSTER_FEATURES_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "ERROR: CHECKPOINT_DIR not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

SPARSE_ROOT="${PARQUET_PATH}"
SPARSE_GLOB="${PARQUET_PATH}"
if [[ -d "${PARQUET_PATH}" ]]; then
  if compgen -G "${PARQUET_PATH}/task_*" > /dev/null; then
    SPARSE_ROOT="${PARQUET_PATH}"
    SPARSE_GLOB="${PARQUET_PATH}/task_*"
  else
    SPARSE_ROOT="${RUN_DIR}/staging_sparse"
    mkdir -p "${SPARSE_ROOT}/task_0"
    SPARSE_GLOB="${SPARSE_ROOT}/task_0"
    python - "${PARQUET_PATH}" "${SPARSE_ROOT}/task_0" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
parquets = sorted(p for p in src.rglob("*.parquet") if p.is_file())
if not parquets:
    raise SystemExit(f"No parquet files found under directory: {src}")
for i, p in enumerate(parquets):
    target = dst / f"part-{i:06d}.parquet"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(p)
PY
  fi
elif [[ -f "${PARQUET_PATH}" ]]; then
  SPARSE_ROOT="${RUN_DIR}/staging_sparse"
  mkdir -p "${SPARSE_ROOT}/task_0"
  SPARSE_GLOB="${SPARSE_ROOT}/task_0"
  ln -sfn "${PARQUET_PATH}" "${SPARSE_ROOT}/task_0/input.parquet"
fi

run_stage() {
  local stage="$1"
  local done_json="$2"
  shift 2
  write_status "running" "${stage}" "${done_json}"
  "$@" >> "${LOG_FILE}" 2>&1
}

on_error() {
  local line="$1"
  local msg="Pipeline failed at line ${line}. See ${LOG_FILE}"
  write_status "failed" "${CURRENT_STAGE}" "${STAGES_DONE_JSON}" "${msg}" "\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\""
  echo "${msg}" >&2
}

CURRENT_STAGE="init"
STAGES_DONE_JSON="[]"
trap 'on_error ${LINENO}' ERR

echo "Running embedding pipeline for ${EXPERIMENT_ID}" > "${LOG_FILE}"
echo "PARQUET_PATH=${PARQUET_PATH}" >> "${LOG_FILE}"
echo "SPARSE_ROOT=${SPARSE_ROOT}" >> "${LOG_FILE}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}" >> "${LOG_FILE}"

CURRENT_STAGE="tsfresh"
run_stage "tsfresh" "${STAGES_DONE_JSON}" \
  "${NB_PYTHON}" "${REPO_ROOT}/src/preprocess/ConvertSparseParquetToTSFresh.py" \
    --parquet_root "${SPARSE_GLOB}" \
    --output_file "${TSFRESH_CSV}" \
    --feature_set "${EMB_TSFEATURE_SET}" \
    --n_jobs "${EMB_TS_N_JOBS}" \
    --time_scale_ms "${EMB_BIN_MS}" \
    --fixed_len "${EMB_TS_FIXED_LEN}" \
    --chunk_rows "${EMB_TS_CHUNK_ROWS}"
STAGES_DONE_JSON='["tsfresh"]'

CURRENT_STAGE="ibgbi"
run_stage "ibgbi" "${STAGES_DONE_JSON}" \
  "${NB_PYTHON}" "${REPO_ROOT}/src/preprocess/SparseToIBGBIFromSparse.py" \
    --sparse_input "${SPARSE_ROOT}" \
    --entity_type "${EMB_ENTITY_TYPE}" \
    --output_dir "${IBGBI_DIR}" \
    --burst_column "${EMB_BURST_COLUMN}" \
    --threshold "${EMB_THRESHOLD}" \
    --bin_ms "${EMB_BIN_MS}" \
    --min_seq_len "${EMB_MIN_SEQ_LEN}" \
    --max_seq_len "${EMB_MAX_SEQ_LEN}" \
    --disable_slurm_shard
STAGES_DONE_JSON='["tsfresh","ibgbi"]'

CURRENT_STAGE="repr"
run_stage "repr" "${STAGES_DONE_JSON}" \
  "${NB_PYTHON}" "${REPO_ROOT}/src/train/extract_repr.py" twin_ip \
    "${IBGBI_DIR}/task_*" \
    --model "${CHECKPOINT_DIR}" \
    --batch_size "${EMB_BATCH_SIZE}" \
    --output_csv "${REPR_CSV}"
STAGES_DONE_JSON='["tsfresh","ibgbi","repr"]'

CURRENT_STAGE="cluster"
run_stage "cluster" "${STAGES_DONE_JSON}" \
  "${NB_PYTHON}" "${REPO_ROOT}/src/analysis/clustering_analysis.py" \
    "${REPR_CSV}" \
    --method kmeans \
    --n-clusters "${EMB_N_CLUSTERS}" \
    --kmeans-metric cosine \
    --minimal-cleaning \
    --out-dir "${CLUSTER_DIR}"
STAGES_DONE_JSON='["tsfresh","ibgbi","repr","cluster"]'

CURRENT_STAGE="interpret"
run_stage "interpret" "${STAGES_DONE_JSON}" \
  "${NB_PYTHON}" - <<PY
import json
from pathlib import Path

cluster_cfg = json.loads(Path("${CLUSTER_FEATURES_TEMPLATE}").read_text(encoding="utf-8"))
cluster_cfg["clustering_csv"] = "${CLUSTER_LABELS_CSV}"
cluster_cfg["tsfresh_csv"] = "${TSFRESH_CSV}"
cluster_cfg["out_csv"] = "${INTERPRET_DIR}/invariance_scores.csv"
cluster_cfg["cluster_variance_out_csv"] = "${INTERPRET_DIR}/cluster_variance.csv"
cluster_cfg["cluster_feature_rank_out_csv"] = "${INTERPRET_DIR}/cluster_feature_rank.csv"
Path("${RUNTIME_CLUSTER_FEATURES_CFG}").write_text(json.dumps(cluster_cfg, indent=2), encoding="utf-8")
PY

run_stage "interpret" "${STAGES_DONE_JSON}" \
  "${NB_PYTHON}" "${REPO_ROOT}/src/analysis/run_analysis_pipeline.py" \
    cluster-features \
    --config "${RUNTIME_CLUSTER_FEATURES_CFG}"
STAGES_DONE_JSON='["tsfresh","ibgbi","repr","cluster","interpret"]'

FINISHED_AT="\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\""
write_status "completed" "done" "${STAGES_DONE_JSON}" "" "${FINISHED_AT}"
echo "Pipeline completed for ${EXPERIMENT_ID}" >> "${LOG_FILE}"
echo "{\"experiment_id\":\"${EXPERIMENT_ID}\",\"state\":\"completed\",\"output_dir\":\"${RUN_DIR}\",\"status_json\":\"${STATUS_JSON}\"}"
