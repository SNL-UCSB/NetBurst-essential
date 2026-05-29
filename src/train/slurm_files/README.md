# NetBurst training Slurm jobs (NERSC)

Run from the repo:

```bash
cd <repo-root>/src/train/slurm_files
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export REPO_ROOT=<repo-root>
export PARQUET_ROOT=/path/to/ibgbi_train
export MODEL_PATH=/path/to/chronos-t5-small/snapshot
```

## Quick submit

```bash
chmod +x submit_pretrain.sh
./submit_pretrain.sh baseline    # global bins only (num_local_bins=0)
./submit_pretrain.sh local16     # global + local quantile bins (default K=16)
./submit_pretrain.sh smoke       # 1 GPU, debug queue, tiny data
./submit_pretrain.sh finetune    # requires CKPT_DIR
```

## External compute entrypoint (embedding pipeline)

Use `run_embedding_pipeline_external.sh` when a separate gateway/compute orchestration layer
needs a single shell entrypoint.

Direct mode (run stages in current shell):

```bash
PARQUET_PATH=/path/to/sparse_root_or_file \
EXPERIMENT_ID=exp-123 \
OUTPUT_ROOT=<data-root>/netburst-outputs \
CHECKPOINT_DIR=<repo-root>/src/train/checkpoints/NetBurstIBGBI_ip_1s_main_local32 \
bash run_embedding_pipeline_external.sh
```

Slurm mode (submit self as sbatch job):

```bash
MODE=slurm \
PARQUET_PATH=/path/to/sparse_root_or_file \
EXPERIMENT_ID=exp-123 \
OUTPUT_ROOT=<data-root>/netburst-outputs \
SLURM_ACCOUNT=<SLURM_ACCOUNT> \
SLURM_QOS=regular \
bash run_embedding_pipeline_external.sh
```

The script writes per-experiment outputs under:

- `${OUTPUT_ROOT}/${EXPERIMENT_ID}/status.json`
- `${OUTPUT_ROOT}/${EXPERIMENT_ID}/logs/pipeline.log`
- `${OUTPUT_ROOT}/${EXPERIMENT_ID}/tsfresh/tsfresh_features.csv`
- `${OUTPUT_ROOT}/${EXPERIMENT_ID}/repr/ip_representations.csv`
- `${OUTPUT_ROOT}/${EXPERIMENT_ID}/clustering/k_<K>_labels.csv`
- `${OUTPUT_ROOT}/${EXPERIMENT_ID}/interpretability/cluster_feature_rank.csv`

## Individual scripts

| Script | Purpose |
|--------|---------|
| `netburst_pretrain_ibgbi_ip_1s_main.sl` | 4-GPU pretrain, `NUM_LOCAL_BINS=0` (legacy-equivalent) |
| `netburst_pretrain_ibgbi_local_bins_ip_1s_main.sl` | 4-GPU pretrain with `NUM_LOCAL_BINS=16` |
| `netburst_finetune_ibgbi_ip_1s_main.sl` | Finetune from checkpoint (`finetune_twin.py`) |
| `netburst_pretrain_smoke_1gpu.sl` | Smoke test on debug queue |
| `run_embedding_pipeline_external.sh` | Stage chain entrypoint for external compute orchestration (`direct` or `slurm`) |

Override any hyperparameter via environment variables before `sbatch` (see comments in each `.sl` file).

## Multi-job launcher (JSON)

Edit `example_pretrain_jobs.json`, then:

```bash
cd <repo-root>/src/train
python build_pretrain_slurm.py --config slurm_files/example_pretrain_jobs.json --dry-run
python build_pretrain_slurm.py --config slurm_files/example_pretrain_jobs.json --submit
```

Per-job field `num_local_bins` (default `0`) is passed to `pretrain_twin.py`.

## Legacy manifest scripts

Older dataset-specific jobs remain under `netburst_pretrain_ibgbi_manifest_*.sl` and `netburst_pretrain_ibgbi_manifest_all_consolidated.sl`. The manifest `ip_1s_main` script includes `--num_local_bins`; regenerate others from JSON or copy the flag line if needed.
