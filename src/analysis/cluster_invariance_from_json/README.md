# Cluster Invariance (Single Run)

## What this module does

Runs one JSON-configured cluster-features job locally or via Slurm.

Entrypoint:

- `submit_cluster_invariance_from_json.py`

## How to run

```bash
cd <repo-root>/src/analysis/cluster_invariance_from_json
python3 submit_cluster_invariance_from_json.py --json my_run.json --mode preview
python3 submit_cluster_invariance_from_json.py --json my_run.json --mode local
python3 submit_cluster_invariance_from_json.py --json my_run.json --mode slurm
```

Wrapper alternative:

```bash
cd <repo-root>
python src/analysis/run_analysis_pipeline.py cluster-features --config /path/to/my_run.json
```

## What to expect

- Run-specific output CSV files according to your JSON paths
- Optional per-cluster output directories when enabled
