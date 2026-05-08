# Cluster Invariance Sweep

## What this module does

Runs many cluster-invariance jobs from one sweep JSON over `k_values`, `run_values`, and `seed_values`.

Entrypoint:

- `submit_cluster_invariance_sweep_from_json.py`

## How to run

```bash
cd <repo-root>/src/analysis/cluster_invariance_sweep_from_json
python3 submit_cluster_invariance_sweep_from_json.py --json example_cluster_invariance_multi_node_per_run_parallel_k.json --mode preview
python3 submit_cluster_invariance_sweep_from_json.py --json example_cluster_invariance_multi_node_per_run_parallel_k.json --mode slurm
```

Optional keys example:

- `optional_sweep_keys.example.json`

## What to expect

- Multiple run output CSVs/directories generated from templates in your sweep JSON
- Slurm scripts/jobs generated per selected submission style
