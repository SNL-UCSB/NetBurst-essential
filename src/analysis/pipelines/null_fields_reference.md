# Null Fields Reference

This document explains why some keys are intentionally `null` in pipeline JSON files.

## Why keep `null` keys

- `null` means the key is supported but intentionally unset.
- For optional features, `null` prevents accidental activation while preserving schema visibility.
- `null` keys should be replaced with `<USER_Input>` only when that feature is enabled.

## Cluster-features defaults

From [`cluster_features.defaults.json`](cluster_features.defaults.json):

- `artifact_root`: optional run-scoping base directory. Keep `null` unless you want scoped outputs.
- `run_tag`: optional run label under `artifact_root/runs/<run_tag>`. Keep `null` unless scoping is enabled.
- `clustering_csv`: optional only when `run_clustering_phase=true`; otherwise provide `<USER_Input>` labels CSV.
- `run_clustering_phase`: when `true`, labels are generated first via `clustering_phase.*` and then fed into cluster-features.
- `clustering_phase.data_csv`: required only when `run_clustering_phase=true` (representation CSV used by `clustering_analysis.py`).
- `clustering_phase.n_clusters`: required only when `run_clustering_phase=true`.
- `clustering_phase.out_dir`: optional output directory for generated labels; defaults under the `out_csv` parent.
- `cluster_members_parent_dir`: required only if member sampling or member plotting is enabled.
- `run_normalized_cka_cohen`: optional post-step; when true, computes top-k per-cluster importance and cluster importance sum from the run outputs.
- `normalized_cka_cohen_top_k`: optional top-k for the post-step (default `10`).
- `per_cluster_cka_workers`: optional thread count; `null` means sequential default behavior.
- `ip_suffix_filter`: defaults to `"/32"`; override only if your dataset IP suffix differs.
- `plot_member_timeseries_input_parquet`: required only when `run_member_timeseries_plots=true`.
- `plot_member_timeseries_config_out`: optional override output path for generated plot config.
- `plot_member_timeseries_y_fixed_max`: optional global y-axis max for member plots.
- `plot_member_timeseries_tsfresh_csv`: optional override TSFresh source for overlays; defaults to `tsfresh_csv`.
- `plot_member_overlay_legend_fontsize`: optional legend font-size override for overlay plots.

## Usage rule

- Keep `null` for optional features you are not using.
- Set `<USER_Input>` for required user paths and identifiers in active features.
