# `netburst` package (`src/train/netburst`)

## What this module does

This package contains shared training code used by the top-level scripts in `src/train/`:

- model definitions
- data loading helpers
- training loop
- utility helpers

Use this package through the CLI scripts (`pretrain_twin.py`, `finetune_twin.py`, `ar_predict.py`, `extract_repr.py`) rather than calling internals directly.

## How to run

From `src/train/`, verify imports:

```bash
cd <repo-root>/src/train
python -c "import netburst"
```

For runnable commands, use:

- `../README.md`
