#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ----------------------------
# Configurable knobs
# ----------------------------
TRAIN_LIMIT="${TRAIN_LIMIT:-256}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_MIN_LEN="${TRAIN_MIN_LEN:-3}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-256}"
N_CLUSTERS="${N_CLUSTERS:-4}"
N_EVAL="${N_EVAL:-32}"
TOPK="${TOPK:-1}"
NLIST="${NLIST:-8}"
NPROBE="${NPROBE:-4}"
NORMALIZED_TOP_K="${NORMALIZED_TOP_K:-5}"
TSFRESH_FEATURE_SET="${TSFRESH_FEATURE_SET:-efficient}"
TSFRESH_FIXED_LEN="${TSFRESH_FIXED_LEN:-600}"
TSFRESH_CHUNK_ROWS="${TSFRESH_CHUNK_ROWS:-500}"
TSFRESH_N_JOBS="${TSFRESH_N_JOBS:-0}"

# Stage control
# - STAGE unset: run everything
# - STAGE set: run one stage only (preprocess|tsfresh|train|forecast|repr|cluster|interpret|retrieval)
STAGE="${STAGE:-}"

FULL_ROOT="${REPO_ROOT}/examples/tiny/outputs/full_pipeline"
PREPROCESS_DIR="${REPO_ROOT}/src/preprocess"
TRAIN_DIR="${REPO_ROOT}/src/train"
ANALYSIS_DIR="${REPO_ROOT}/src/analysis"
RETRIEVAL_DIR="${REPO_ROOT}/src/retrieval/netburst_faiss"
TINY_DIR="${REPO_ROOT}/examples/tiny"

TINY_SPARSE_DIR="${REPO_ROOT}/examples/tiny/data/sparse_timeseries_1s_ip_tiny"
TINY_IBGBI_DIR="${FULL_ROOT}/preprocess/ibgbi_1s_ip_tiny"
TINY_SPLITS_DIR="${FULL_ROOT}/preprocess/splits_1s_ip_tiny"

TRAIN_OUT_DIR="${FULL_ROOT}/train/checkpoint"
FORECAST_OUT_STEM="${FULL_ROOT}/forecast/ar_results.pkl"
REPR_CSV="${FULL_ROOT}/analysis/ip_representations.csv"
TSFRESH_CSV="${FULL_ROOT}/analysis/tiny_tsfresh_features.csv"
CLUSTER_DIR="${FULL_ROOT}/analysis/clustering"
CLUSTER_LABELS_CSV="${CLUSTER_DIR}/k_${N_CLUSTERS}_labels.csv"
INTERPRET_DIR="${FULL_ROOT}/analysis/interpretability"
RETRIEVAL_OUT_DIR="${FULL_ROOT}/retrieval"

CLUSTER_FEATURES_CFG="${REPO_ROOT}/examples/tiny/configs/cluster_features_tiny.json"
NORMALIZED_CKA_CFG="${REPO_ROOT}/examples/tiny/configs/normalized_cka_cohen_tiny.json"
RUNTIME_CLUSTER_FEATURES_CFG="${INTERPRET_DIR}/cluster_features_runtime.json"
RUNTIME_NORMALIZED_CKA_CFG="${INTERPRET_DIR}/normalized_cka_cohen_runtime.json"

mkdir -p "${FULL_ROOT}" "${INTERPRET_DIR}" "${RETRIEVAL_OUT_DIR}"

run_stage() {
  local name="$1"
  if [[ -z "${STAGE}" || "${STAGE}" == "${name}" ]]; then
    return 0
  fi
  return 1
}

assert_path_exists() {
  local p="$1"
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: Required path missing: ${p}" >&2
    exit 1
  fi
}

assert_glob_has_files() {
  local g="$1"
  shopt -s nullglob
  local matches=(${g})
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "ERROR: No files matched: ${g}" >&2
    exit 1
  fi
}

echo "Running tiny full-pipeline smoke test"
echo "Output root: ${FULL_ROOT}"
echo "Selected stage: ${STAGE:-all}"

if run_stage "preprocess"; then
  echo "[preprocess] Reusing existing tiny sparse dataset and building split artifacts"
  assert_path_exists "${TINY_SPARSE_DIR}"
  assert_glob_has_files "${TINY_SPARSE_DIR}/task_*"

  (
    cd "${PREPROCESS_DIR}"
    python SparseToIBGBIFromSparse.py \
      --sparse_input "${TINY_SPARSE_DIR}" \
      --entity_type ip \
      --output_dir "${TINY_IBGBI_DIR}" \
      --burst_column inbound \
      --threshold 100 \
      --bin_ms 1000 \
      --min_seq_len 3 \
      --max_seq_len 9000 \
      --disable_slurm_shard
  )

  (
    cd "${PREPROCESS_DIR}"
    python create_train_test_splits.py \
      --ibgbi_dir "${TINY_IBGBI_DIR}/task_*" \
      --entity_type ip \
      --output_dir "${TINY_SPLITS_DIR}" \
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

  assert_path_exists "${TINY_SPLITS_DIR}/ibgbi_train"
  assert_path_exists "${TINY_SPLITS_DIR}/ibgbi_test"
fi

if run_stage "tsfresh"; then
  echo "[tsfresh] Extracting TSFresh features from tiny sparse parquet"
  assert_path_exists "${TINY_SPARSE_DIR}"
  (
    cd "${PREPROCESS_DIR}"
    python ConvertSparseParquetToTSFresh.py \
      --parquet_root "${TINY_SPARSE_DIR}/task_*" \
      --output_file "${TSFRESH_CSV}" \
      --feature_set "${TSFRESH_FEATURE_SET}" \
      --n_jobs "${TSFRESH_N_JOBS}" \
      --time_scale_ms 1000 \
      --fixed_len "${TSFRESH_FIXED_LEN}" \
      --chunk_rows "${TSFRESH_CHUNK_ROWS}"
  )
  assert_path_exists "${TSFRESH_CSV}"
fi

if run_stage "train"; then
  echo "[train] Running tiny pretraining"
  assert_path_exists "${TINY_SPLITS_DIR}/ibgbi_train"
  mkdir -p "${TRAIN_OUT_DIR}"
  (
    cd "${TRAIN_DIR}"
    torchrun --standalone --nproc_per_node=1 pretrain_twin.py \
      "${TINY_SPLITS_DIR}/ibgbi_train" \
      --model amazon/chronos-t5-small \
      --epochs "${TRAIN_EPOCHS}" \
      --batch_size "${TRAIN_BATCH_SIZE}" \
      --limit "${TRAIN_LIMIT}" \
      --min_len "${TRAIN_MIN_LEN}" \
      --max_len "${TRAIN_MAX_LEN}" \
      --save_dir "${TRAIN_OUT_DIR}"
  )
  assert_path_exists "${TRAIN_OUT_DIR}/chronos_best.pt"
  assert_path_exists "${TRAIN_OUT_DIR}/boundaries_bi.pkl"
  assert_path_exists "${TRAIN_OUT_DIR}/boundaries_ibg.pkl"
fi

if run_stage "forecast"; then
  echo "[forecast] Running autoregressive forecasting"
  assert_path_exists "${TRAIN_OUT_DIR}/chronos_best.pt"
  assert_path_exists "${TINY_SPLITS_DIR}/ibgbi_test"
  mkdir -p "${FULL_ROOT}/forecast"
  (
    cd "${TRAIN_DIR}"
    torchrun --standalone --nproc_per_node=1 ar_predict.py \
      "${TINY_SPLITS_DIR}/ibgbi_test" \
      --model "${TRAIN_OUT_DIR}" \
      --save_pkl "${FORECAST_OUT_STEM}" \
      --use_precomputed_context_forecast \
      --limit "${N_EVAL}" \
      --min_h 1 \
      --batch_size 16 \
      --top_k 20 \
      --top_p 1.0 \
      --temperature 1.0
  )
  assert_path_exists "${FULL_ROOT}/forecast/ar_results/final_metrics_chronos_netburst.json"
  assert_path_exists "${FULL_ROOT}/forecast/ar_results/per_example_metrics_chronos_netburst.csv"
fi

if run_stage "repr"; then
  echo "[repr] Extracting representations"
  assert_path_exists "${TRAIN_OUT_DIR}/chronos_best.pt"
  mkdir -p "${FULL_ROOT}/analysis"
  (
    cd "${TRAIN_DIR}"
    python extract_repr.py twin_ip \
      "${TINY_SPLITS_DIR}/ibgbi_train" \
      --model "${TRAIN_OUT_DIR}" \
      --output_csv "${REPR_CSV}"
  )
  assert_path_exists "${REPR_CSV}"
fi

if run_stage "cluster"; then
  echo "[cluster] Clustering tiny representations"
  assert_path_exists "${REPR_CSV}"
  mkdir -p "${CLUSTER_DIR}"
  (
    cd "${ANALYSIS_DIR}"
    python clustering_analysis.py \
      "${REPR_CSV}" \
      --method kmeans \
      --n-clusters "${N_CLUSTERS}" \
      --kmeans-metric cosine \
      --minimal-cleaning \
      --out-dir "${CLUSTER_DIR}"
  )
  assert_path_exists "${CLUSTER_LABELS_CSV}"
fi

if run_stage "interpret"; then
  echo "[interpret] Running cluster-features and normalized CKA x Cohen"
  assert_path_exists "${CLUSTER_LABELS_CSV}"
  assert_path_exists "${TSFRESH_CSV}"
  mkdir -p "${INTERPRET_DIR}"
  assert_path_exists "${CLUSTER_FEATURES_CFG}"
  assert_path_exists "${NORMALIZED_CKA_CFG}"

  python - <<PY
import json
from pathlib import Path

cluster_cfg = json.loads(Path("${CLUSTER_FEATURES_CFG}").read_text(encoding="utf-8"))
cluster_cfg["clustering_csv"] = "${CLUSTER_LABELS_CSV}"
cluster_cfg["tsfresh_csv"] = "${TSFRESH_CSV}"
cluster_cfg["out_csv"] = "${INTERPRET_DIR}/invariance_scores.csv"
cluster_cfg["cluster_variance_out_csv"] = "${INTERPRET_DIR}/cluster_variance.csv"
cluster_cfg["cluster_feature_rank_out_csv"] = "${INTERPRET_DIR}/cluster_feature_rank.csv"
Path("${RUNTIME_CLUSTER_FEATURES_CFG}").write_text(json.dumps(cluster_cfg, indent=2), encoding="utf-8")

norm_cfg = json.loads(Path("${NORMALIZED_CKA_CFG}").read_text(encoding="utf-8"))
norm_cfg["k_values"] = [int("${N_CLUSTERS}")]
norm_cfg["cluster_members_parent_dir_template"] = "${INTERPRET_DIR}"
norm_cfg["cohens_d_by_cluster_out_csv_template"] = "${INTERPRET_DIR}/cohens_d_across_clusters.csv"
norm_cfg["top_k"] = int("${NORMALIZED_TOP_K}")
Path("${RUNTIME_NORMALIZED_CKA_CFG}").write_text(json.dumps(norm_cfg, indent=2), encoding="utf-8")
PY

  (
    cd "${REPO_ROOT}"
    python src/analysis/run_analysis_pipeline.py \
      cluster-features \
      --config "${RUNTIME_CLUSTER_FEATURES_CFG}"
  )

  (
    cd "${REPO_ROOT}"
    python src/analysis/run_analysis_pipeline.py \
      normalized-cka-cohen \
      --config "${RUNTIME_NORMALIZED_CKA_CFG}" \
      --top-k "${NORMALIZED_TOP_K}"
  )

  assert_path_exists "${INTERPRET_DIR}/invariance_scores.csv"
  assert_path_exists "${INTERPRET_DIR}/cluster_variance.csv"
  assert_path_exists "${INTERPRET_DIR}/cluster_feature_rank.csv"
  assert_path_exists "${INTERPRET_DIR}/cohens_d_across_clusters.csv"
  assert_path_exists "${INTERPRET_DIR}/normalized_cka_cohen_cluster_importance.csv"
fi

if run_stage "retrieval"; then
  echo "[retrieval] Running FAISS retrieval smoke stage"
  assert_path_exists "${CLUSTER_LABELS_CSV}"
  assert_path_exists "${TINY_SPARSE_DIR}"
  mkdir -p "${RETRIEVAL_OUT_DIR}"
  (
    cd "${RETRIEVAL_DIR}"
    python run_pipeline_faiss.py \
      --cluster-csv "${CLUSTER_LABELS_CSV}" \
      --model netburst_tiny \
      --output-dir "${RETRIEVAL_OUT_DIR}" \
      --nlist "${NLIST}" \
      --nprobe "${NPROBE}" \
      --topk "${TOPK}" \
      --n-eval "${N_EVAL}" \
      --parquet-path "${TINY_SPARSE_DIR}/task_0"
  )

  assert_path_exists "${RETRIEVAL_OUT_DIR}/netburst_tiny_ivf_flat.faiss"
  assert_path_exists "${RETRIEVAL_OUT_DIR}/netburst_tiny_split_manifest.csv"
  assert_path_exists "${RETRIEVAL_OUT_DIR}/netburst_tiny_distances.csv"
  assert_path_exists "${RETRIEVAL_OUT_DIR}/netburst_tiny_query_times.csv"
fi

echo "Tiny full-pipeline smoke completed for stage: ${STAGE:-all}"
