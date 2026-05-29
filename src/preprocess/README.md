# Preprocessing

Preprocessing in this repo is built around **sparse -> IBG/BI -> train/test/context splits**.

## Canonical flow

1. Build sparse time-series (non-graph or ping).
2. Convert sparse to IBG/BI.
3. Create train/test + context/forecast splits.

Use the manifest driver when possible:

```bash
python run_sparse_to_ibgbi_splits_manifest.py \
  --manifest sparse_to_ibgbi_splits_manifest.yaml
```

Optional for Slurm wrappers:

- `--slurm_shard_jobs` to shard manifest jobs across tasks
- `--job_index N` for job arrays / single-job debugging

## Scripts in this directory

### Core pipeline

- `run_sparse_to_ibgbi_splits_manifest.py` — orchestrates per-job conversion and splits from YAML.
- `SparseToIBGBIFromSparse.py` — sparse parquet -> IBG/BI parquet.
- `create_train_test_splits.py` — IBG/BI -> splits (`splits_meta`, `ibgbi_train`, `ibgbi_test`, optional sparse mirrors).
- `select_bi_threshold.py` — threshold selection from quantiles parquet.
- `quantiles_utils.py` — helper functions for quantile computation/storage.
- `ConvertSparseParquetToTSFresh.py` — sparse parquet (`ip`, `source_file`, `inbound`) -> real TSFresh feature CSV.

### Sparse generation

- `NonGraphSparseTimeseries.py` — non-graph sparse generation (+ optional quantiles).
- `PingLatencyToSparseTimeseries.py` — ping sparse generation (+ optional quantiles).
- `OtherDatasets_preprocessing/` — additional sparse conversion scripts and slurm templates.
- `rne_netflow_sparse_upstream/` — RnE-NetFlow-specific sparse preparation helpers.

### Ping-specific converter

- `PingLatencyToIBGBI.py` — ping CSV or sparse input -> IBG/BI.
  - Supports `--threshold_ms` or quantile-derived thresholds.
  - Useful for ping-only workflows outside manifest orchestration.

### PCAP utilities

- `PcapToDf_multi_node.py` — PCAP -> packet-level parquet.
- `PCAP_to_DF_slurm_files/` — Slurm templates for PCAP conversion.
- `split_pcaps.sh` — split large PCAP files for parallel processing.

## Manifest notes

`sparse_to_ibgbi_splits_manifest.yaml` includes PINOT, RnE-NetFlow, PerfSONAR, and ping jobs.

- `pipeline: sparse_to_ibgbi` runs:
  1) `SparseToIBGBIFromSparse.py`
  2) `create_train_test_splits.py`
- `pipeline: existing_ibgbi` runs only split generation from pre-existing IBG/BI parquet.

Ping in the merged manifest uses the sparse->IBG/BI path (`entity_type: ip` over `*_label` sparse output).

## Data handoff to training

Training scripts and expected parquet schemas are documented in:

- `src/train/README.md`

## Dependencies

Primary dependencies: PySpark, pandas, PyYAML, and standard Python libraries used by these scripts.
