# Analysis Example Configs

## What this module does

Contains minimal example JSON configs for analysis workflows.

## How to run

Run from repo root:

```bash
python src/analysis/run_analysis_pipeline.py cluster-features --config src/analysis/pipelines/examples/netburst_cluster_features_simple_external_labels.example.json
python src/analysis/run_analysis_pipeline.py cluster-features --config src/analysis/pipelines/examples/netburst_cluster_features_simple_run_clustering.example.json
python src/analysis/run_analysis_pipeline.py global-cka --config src/analysis/pipelines/examples/netburst_global_cka_simple.example.json
python src/analysis/run_analysis_pipeline.py anisotropy --config src/analysis/pipelines/examples/netburst_anisotropy_simple.example.json
python src/analysis/run_analysis_pipeline.py normalized-cka-cohen --config src/analysis/pipelines/examples/netburst_normalized_cka_cohen_simple.example.json
```

Before running, replace `<USER_Input>` placeholders in the selected JSON.

## What to expect

- Output files are written to the paths configured in each JSON.
- No output is written unless the corresponding path keys are set in your config.
