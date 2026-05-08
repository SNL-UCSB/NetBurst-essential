# Analysis (`src/analysis`)

## What this module does

This module evaluates NetBurst representations and clustering outputs.

Main workflows:

- `cluster-features`: clustering-based feature importance outputs
- `global-cka`: global CKA scoring
- `anisotropy`: representation geometry metrics
- `normalized-cka-cohen`: post-hoc cluster importance aggregation

## How to run

Run from repo root:

```bash
cd <repo-root>
python src/analysis/run_analysis_pipeline.py --help
```

Common commands:

```bash
python src/analysis/run_analysis_pipeline.py cluster-features --config /path/to/cluster_features.json
python src/analysis/run_analysis_pipeline.py global-cka --config /path/to/global_cka.json
python src/analysis/run_analysis_pipeline.py anisotropy --config /path/to/anisotropy.json
python src/analysis/run_analysis_pipeline.py normalized-cka-cohen --config /path/to/normalized_cka_cohen.json
```

Defaults and examples:

- `pipelines/cluster_features.defaults.json`
- `pipelines/global_cka_only.defaults.json`
- `pipelines/anisotropy_only.defaults.json`
- `pipelines/normalized_cka_cohen.defaults.json`
- `pipelines/examples/`

## What to expect

- Cluster-features CSV outputs (invariance, variance, rank outputs)
- Global CKA output CSVs
- Anisotropy output CSV/JSON files
- Optional per-cluster artifact directories for workflows that enable them
