# Training (`src/train`)

## What this module does

This module trains NetBurst models on IBG/BI parquet data, runs autoregressive inference, and extracts representation CSVs used by analysis and retrieval.

## How to run

Run from:

```bash
cd <repo-root>/src/train
```

Main scripts:

- `pretrain_twin.py` - pretrain from base pretrained weights
- `finetune_twin.py` - finetune from an existing NetBurst checkpoint
- `ar_predict.py` - distributed autoregressive inference
- `extract_repr.py` - representation extraction (`final`, `detailed`, `twin_ip`)
- `finetune_repr_mlp.py` - train MLP on saved representation CSVs
- `pretrain_netburst_head_2.py` - deprecated compatibility wrapper

Example commands:

```bash
torchrun --nproc_per_node=4 pretrain_twin.py /path/to/ibgbi_train --model amazon/chronos-t5-small --save_dir ./checkpoints/run1

torchrun --nproc_per_node=4 finetune_twin.py /path/to/ibgbi_train --model ./checkpoints/run1 --save_dir ./checkpoints/run1_ft

torchrun --nproc_per_node=4 ar_predict.py /path/to/ibgbi_test --model ./checkpoints/run1 --save_pkl ./ar_out/results.pkl

python extract_repr.py twin_ip /path/to/bi_ibg_parquet --model ./checkpoints/run1 --output_csv ./repr/ip_reps.csv
```

## Slurm (NERSC)

See [`slurm_files/README.md`](slurm_files/README.md). Typical pretrain with local bins:

```bash
cd src/train/slurm_files
export SLURM_ACCOUNT=<SLURM_ACCOUNT> REPO_ROOT=/path/to/NetBurst-essential
export PARQUET_ROOT=/path/to/ibgbi_train MODEL_PATH=/path/to/chronos-t5-small/snapshot
./submit_pretrain.sh local16
```

Use `./submit_pretrain.sh baseline` for `num_local_bins=0` (same as legacy). Finetune: `export CKPT_DIR=./checkpoints/run1 && ./submit_pretrain.sh finetune`.

## What to expect

- Checkpoint directories under your configured save path
- Inference outputs (`*.pkl`, per-example CSV, aggregate JSON)
- Representation CSV outputs with `repr_dim_*` columns
