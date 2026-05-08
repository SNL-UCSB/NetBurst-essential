# NetBurst-essential

Standalone repository for running the NetBurst pipeline end-to-end:

- preprocess
- train
- analysis
- retrieval

## What each module does

- `src/preprocess/` - sparse time-series -> IBG/BI -> train/test/context splits
- `src/train/` - model training, finetuning, inference, representation extraction
- `src/analysis/` - clustering/CKA/anisotropy workflows and JSON-driven analysis pipelines
- `src/retrieval/netburst_faiss/` - FAISS index build and retrieval evaluation

## How to run

Follow module runbooks in this order:

1. `src/preprocess/README.md`
2. `src/train/README.md`
3. `src/analysis/README.md`
4. `src/retrieval/netburst_faiss/README.md`

## What to expect

- Preprocess outputs under configured output paths (commonly `<repo-root>/outputs/`)
- Train checkpoints and evaluation artifacts
- Analysis CSV outputs and optional plot artifacts
- Retrieval FAISS index artifacts and query/distance CSVs

## Citation

Paper link:

- [https://arxiv.org/abs/2510.22397](https://arxiv.org/abs/2510.22397)

BibTeX:

```bibtex
@misc{guthula2025netbursteventcentricforecastingbursty,
      title={NetBurst: Event-Centric Forecasting of Bursty, Intermittent Time Series}, 
      author={Satyandra Guthula and Jaber Daneshamooz and Charles Fleming and Ashish Kundu and Walter Willinger and Arpit Gupta},
      year={2025},
      eprint={2510.22397},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2510.22397}, 
}
```
