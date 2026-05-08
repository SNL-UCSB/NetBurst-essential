# Clustering Then Invariance Sweep

## What this module does

Runs clustering first, then invariance analysis, using JSON sweep configs.

Entrypoints:

- `submit_clustering_then_invariance_sweep_from_json.py`
- `run_submit_with_nersc_modules.sh`

## How to run

```bash
cd <repo-root>
./src/analysis/clustering_then_invariance_sweep_from_json/run_submit_with_nersc_modules.sh \
  --json src/analysis/clustering_then_invariance_sweep_from_json/example_clustering_then_invariance_multi_node_per_run_parallel_k.json \
  --mode preview

./src/analysis/clustering_then_invariance_sweep_from_json/run_submit_with_nersc_modules.sh \
  --json src/analysis/clustering_then_invariance_sweep_from_json/example_clustering_then_invariance_multi_node_per_run_parallel_k.json \
  --mode slurm
```

## What to expect

- Clustering labels per configured run/K
- Invariance output CSVs based on templates in your JSON
