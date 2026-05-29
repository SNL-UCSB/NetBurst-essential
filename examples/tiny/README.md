# Tiny Smoke Tests (`examples/tiny`)

This directory includes:

- a preprocess-only smoke test
- a full end-to-end smoke test (preprocess, pretrain, forecasting, clustering, interpretability, FAISS retrieval)

No raw packet/parquet conversion is run. Input starts from an existing sparse dataset.

## Files

- `run_smoke_pipeline.sh` - preprocess-only smoke test.
- `run_smoke_full_pipeline.sh` - full pipeline smoke test with per-stage artifact checks.
- `run_smoke_e2e_from_parquet.sh` - stage-orchestrated end-to-end smoke from checked-in tiny parquet.
- `configs/cluster_features_tiny.json` - tiny interpretability config for `cluster-features`.
- `configs/normalized_cka_cohen_tiny.json` - tiny config for normalized CKA x Cohen aggregation.
- `sparse_to_ibgbi_splits_manifest_tiny.yaml` - tiny manifest aligned with `ip_1s_main` defaults.

## Quick Start

From repo root:

```bash
bash examples/tiny/run_smoke_full_pipeline.sh
```

Preferred one-command end-to-end run from checked-in parquet:

```bash
bash examples/tiny/run_smoke_e2e_from_parquet.sh
```

Default tiny sparse input:

- `examples/tiny/data/sparse_timeseries_1s_ip_tiny`

For preprocess-only smoke (`run_smoke_pipeline.sh`), the script defaults to reusing the
existing tiny parquet when `task_*` shards are present. Use:

```bash
REBUILD_TINY_SPARSE=1 SOURCE_SPARSE=<data-root>/sparse_timeseries_1s_ip bash examples/tiny/run_smoke_pipeline.sh
```

to force rebuilding the tiny dataset from another sparse parquet source.

Primary output root:

- `examples/tiny/outputs/full_pipeline`

## Run Specific Stage

Use `STAGE=<name>`:

```bash
STAGE=preprocess bash examples/tiny/run_smoke_full_pipeline.sh
STAGE=train bash examples/tiny/run_smoke_full_pipeline.sh
STAGE=forecast bash examples/tiny/run_smoke_full_pipeline.sh
STAGE=repr bash examples/tiny/run_smoke_full_pipeline.sh
STAGE=cluster bash examples/tiny/run_smoke_full_pipeline.sh
STAGE=interpret bash examples/tiny/run_smoke_full_pipeline.sh
STAGE=retrieval bash examples/tiny/run_smoke_full_pipeline.sh
```

`tsfresh` is also available as an individual stage:

```bash
STAGE=tsfresh bash examples/tiny/run_smoke_full_pipeline.sh
```

## Useful Knobs

Set these environment variables as needed:

- `TRAIN_LIMIT` (default `256`)
- `TRAIN_EPOCHS` (default `1`)
- `TRAIN_BATCH_SIZE` (default `8`)
- `N_CLUSTERS` (default `4`)
- `N_EVAL` (default `32`)
- `TOPK` (default `1`)
- `NLIST` (default `8`)
- `NPROBE` (default `4`)
- `NORMALIZED_TOP_K` (default `5`)
- `TSFRESH_FEATURE_SET` (default `efficient`, options: `minimal|efficient|comprehensive`)
- `TSFRESH_FIXED_LEN` (default `600`)
- `TSFRESH_CHUNK_ROWS` (default `500`)
- `TSFRESH_N_JOBS` (default `0`, tsfresh uses all available cores)

Example:

```bash
TRAIN_LIMIT=384 N_CLUSTERS=6 N_EVAL=48 bash examples/tiny/run_smoke_full_pipeline.sh
```

For the end-to-end parquet wrapper, you can choose a subset of stages:

```bash
STAGES="preprocess tsfresh train" bash examples/tiny/run_smoke_e2e_from_parquet.sh
```

## Expected Artifacts

After a full run, check these:

- preprocess:
  - `examples/tiny/outputs/full_pipeline/preprocess/splits_1s_ip_tiny/ibgbi_train`
  - `examples/tiny/outputs/full_pipeline/preprocess/splits_1s_ip_tiny/ibgbi_test`
- training:
  - `examples/tiny/outputs/full_pipeline/train/checkpoint/chronos_best.pt`
- forecasting:
  - `examples/tiny/outputs/full_pipeline/forecast/ar_results/final_metrics_chronos_netburst.json`
- representations:
  - `examples/tiny/outputs/full_pipeline/analysis/ip_representations.csv`
- clustering:
  - `examples/tiny/outputs/full_pipeline/analysis/clustering/k_<K>_labels.csv`
- interpretability:
  - `examples/tiny/outputs/full_pipeline/analysis/interpretability/invariance_scores.csv`
  - `examples/tiny/outputs/full_pipeline/analysis/interpretability/normalized_cka_cohen_cluster_importance.csv`
- retrieval:
  - `examples/tiny/outputs/full_pipeline/retrieval/netburst_tiny_ivf_flat.faiss`
  - `examples/tiny/outputs/full_pipeline/retrieval/netburst_tiny_distances.csv`
  - `examples/tiny/outputs/full_pipeline/retrieval/netburst_tiny_query_times.csv`

## Prerequisites

- Local GPU runtime for `torchrun` stages (single process).
- Install Python dependencies from repo root:

```bash
pip install -r requirements.txt
```

## Troubleshooting

- If train/forecast fails with NCCL/CUDA issues, verify GPU visibility and torch runtime.
- If retrieval import fails on `pymilvus`, install it even for FAISS mode (module import requirement).
- If interpretability join is empty, verify `ip` and `source_file` keys between clustering CSV and TSFresh fixture CSV.
- On Perlmutter-like Intel MKL environments, `tsfresh` or `retrieval` may fail with
  `undefined symbol: omp_get_num_procs`; run those stages with:

```bash
LD_PRELOAD=/global/common/software/nersc9/pytorch/2.6.0/lib/libgomp.so.1 STAGE=tsfresh bash examples/tiny/run_smoke_full_pipeline.sh
LD_PRELOAD=/global/common/software/nersc9/pytorch/2.6.0/lib/libgomp.so.1 STAGE=retrieval bash examples/tiny/run_smoke_full_pipeline.sh
```
