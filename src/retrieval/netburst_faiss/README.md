# NetBurst FAISS Retrieval

This directory contains the NetBurst retrieval pipeline using FAISS IVF-Flat, fully inside this repo.

## Scope

This pipeline does all of the following:

- reads NetBurst cluster/embedding CSV data,
- creates train/eval split artifacts,
- builds a FAISS index on train vectors,
- runs retrieval on eval vectors,
- writes query-time and distance outputs.

Primary entrypoint:

- `src/retrieval/netburst_faiss/run_pipeline_faiss.py`

Supporting modules:

- `src/retrieval/netburst_faiss/faiss_based/ingest_faiss.py`
- `src/retrieval/netburst_faiss/faiss_based/query_eval_faiss.py`
- `src/retrieval/netburst_faiss/faiss_based/search_ivf.py`
- `src/retrieval/netburst_faiss/faiss_based/cluster_allowed_ids.py`

---

## End-to-end checklist

Before running retrieval, make sure you have all four:

- a valid Python environment with required packages,
- a NetBurst embeddings CSV (`repr_dim_*`),
- a clustering labels CSV (`k_<K>_labels.csv`) with `label`,
- a parquet source with `ip`, `source_file`, `inbound` for WD/DTW scoring.

If any of these are missing, generate them using the steps below.

---

## Required input schema

The retrieval input CSV (`--cluster-csv`) must contain:

- `ip`
- `source_file`
- `label`
- vector columns, usually `repr_dim_0`, `repr_dim_1`, ...

Best source:

- `src/analysis/clustering_analysis.py` output file `k_<K>_labels.csv`

Optional but typically required for evaluation metrics:

- `--parquet-path` dataset with columns `ip`, `source_file`, `inbound`

Note: current `query_eval_faiss.py` computes Wasserstein/DTW, so parquet access is part of a normal run.

---

## Environment setup

From repo root:

```bash
cd <repo-root>
```

Install dependencies:

```bash
pip install faiss-cpu numpy pandas scipy pymilvus pyarrow
```

Why `pymilvus` is listed: this copied pipeline imports shared helpers that reference it at import time.

---

## Generate retrieval inputs in this repo

### 1) Create embeddings (`repr_dim_*`)

```bash
python src/train/extract_repr.py twin_ip /path/to/bi_ibg_parquet \
  --model /path/to/twin_head_checkpoint_dir \
  --output_csv /path/to/artifacts/retrieval_inputs/ip_representations.csv
```

Optional:

- add `--ips_csv /path/to/IpsToConsider.csv` to filter which IPs to embed.

Expected output columns include:

- `ip`
- `source_file`
- `repr_dim_*`

### 2) Cluster those embeddings to create labels

```bash
python src/analysis/clustering_analysis.py \
  /path/to/artifacts/retrieval_inputs/ip_representations.csv \
  --method kmeans \
  --n-clusters 500 \
  --kmeans-metric cosine \
  --minimal-cleaning \
  --out-dir /path/to/artifacts/retrieval_inputs/clustering
```

Expected output:

- `/path/to/artifacts/retrieval_inputs/clustering/k_500_labels.csv`

### 3) Validate cluster CSV before retrieval (recommended)

```bash
python - <<'PY'
import pandas as pd
fp = "/path/to/artifacts/retrieval_inputs/clustering/k_500_labels.csv"
df = pd.read_csv(fp, nrows=10)
required = {"ip", "source_file", "label"}
repr_cols = [c for c in df.columns if c.startswith("repr_dim_")]
print("missing_required:", sorted(required - set(df.columns)))
print("repr_dim_count:", len(repr_cols))
print("sample_columns:", list(df.columns[:12]))
PY
```

If `missing_required` is non-empty or `repr_dim_count == 0`, fix the input generation step first.

---

## Run retrieval locally

Basic run:

```bash
python src/retrieval/netburst_faiss/run_pipeline_faiss.py \
  --cluster-csv /path/to/artifacts/retrieval_inputs/clustering/k_500_labels.csv \
  --model netburst \
  --output-dir /path/to/artifacts/faiss_runs/netburst_k500 \
  --nlist 500 \
  --nprobe 32 \
  --topk 1 \
  --parquet-path /path/to/netreplica_parquet_root
```

### Important flag guidance

- `--nlist`: IVF coarse clusters; common choice is similar scale to clustering K.
- `--nprobe`: number of IVF lists visited per query; larger is slower but higher recall.
- `--topk`: neighbors per query.
- `--model`: used in output file names; use `netburst`.
- `--output-dir`: where all artifacts and metrics are written.

### Optional run modes

- global IVF search (disable cluster filter):

```bash
python src/retrieval/netburst_faiss/run_pipeline_faiss.py \
  --cluster-csv /path/to/.../k_500_labels.csv \
  --model netburst \
  --output-dir /path/to/.../global_search \
  --nlist 500 \
  --nprobe 32 \
  --topk 1 \
  --parquet-path /path/to/netreplica_parquet_root \
  --no-cluster-filter
```

- fixed split manifest:

```bash
python src/retrieval/netburst_faiss/run_pipeline_faiss.py \
  --cluster-csv /path/to/.../k_500_labels.csv \
  --model netburst \
  --output-dir /path/to/.../with_manifest \
  --split-manifest /path/to/predefined_split_manifest.csv \
  --nlist 500 \
  --nprobe 32 \
  --topk 1 \
  --parquet-path /path/to/netreplica_parquet_root
```

---

## Run with Slurm

Two templates are included:

- `src/retrieval/netburst_faiss/slurm/prepare_netburst_retrieval_artifacts.slurm`
- `src/retrieval/netburst_faiss/slurm/run_netburst_faiss_retrieval.slurm`

### Slurm step A: prepare embeddings + cluster labels

1. Edit variables at top of `prepare_netburst_retrieval_artifacts.slurm`:
   - `REPO_ROOT`
   - `BI_IBG_PARQUET`
   - `TWIN_HEAD_CKPT`
   - `ARTIFACT_ROOT`
   - `N_CLUSTERS`
2. Submit:

```bash
sbatch src/retrieval/netburst_faiss/slurm/prepare_netburst_retrieval_artifacts.slurm
```

### Slurm step B: run FAISS retrieval

1. Edit variables at top of `run_netburst_faiss_retrieval.slurm`:
   - `REPO_ROOT`
   - `CLUSTER_CSV`
   - `PARQUET_PATH`
   - `OUTPUT_DIR`
   - `MODEL_NAME` (keep `netburst`)
   - `NLIST`, `NPROBE`, `TOPK`
   - optional `SPLIT_MANIFEST`
2. Submit:

```bash
sbatch src/retrieval/netburst_faiss/slurm/run_netburst_faiss_retrieval.slurm
```

---

## Expected outputs

Inside `--output-dir`:

- `{model}_ivf_flat.faiss`
- `{model}_ivf_flat.faiss.meta.json`
- `{model}_split_manifest.csv`
- `{model}_eval_vecs.npy`
- `{model}_split.pkl`
- `{model}_distances.csv`
- `{model}_query_times.csv`
- `benchmark_config_{model}.json`

Quick checks:

- index exists and is non-empty: `{model}_ivf_flat.faiss`
- eval rows in manifest match eval vectors in `.npy`
- `query_times.csv` has one row per eval query
- `distances.csv` has `topk` rows per eval query

---

## Common failure modes and fixes

- Missing `label` column in cluster CSV:
  - rerun clustering step and use `k_<K>_labels.csv` output.
- No `repr_dim_*` columns:
  - ensure embeddings came from `extract_repr.py twin_ip`.
- `pymilvus` import error even for FAISS:
  - install `pymilvus` in current environment (temporary requirement of copied helpers).
- Parquet column errors (`ip`, `source_file`, `inbound`):
  - point `--parquet-path` to the correct dataset layout.
- Very slow query phase:
  - lower `topk`, lower `nprobe`, or run on smaller eval split (`--n-eval`).

---
