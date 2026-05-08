"""
Compute max-normalized local CKA × max-normalized |Cohen's d| feature importance
per cluster, then derive cluster importance as sum of top-k feature importances.

Algorithm
---------
For each (run, k, seed) combination specified in the JSON config:

1. Load per-cluster local CKA
   - Read selected_features_cka.csv from every cluster_{n}/ subdirectory.
   - Stack into a long DataFrame: (cluster_id, feature, cka_local).
   - Global max of cka_local across all clusters × features.
   - norm_cka = cka_local / global_max_cka   ∈ [0, 1]

2. Load Cohen's d
   - Read cohens_d_across_clusters.csv (wide: features × clusters).
   - Melt to long: (cluster_id, feature, cohens_d), abs_cohens_d = |cohens_d|.
   - Global max of abs_cohens_d across all clusters × features.
   - norm_cohen = abs_cohens_d / global_max_cohen   ∈ [0, 1]

3. Inner-join on (cluster_id, feature).
   feature_importance = norm_cka × norm_cohen

4. Top-k per cluster → cluster_importance = sum of those top-k scores.

Config keys
-----------
- ``top_k`` or ``normalized_cka_cohen_top_k`` (default: 10).
  Override with CLI ``--top-k``.

Outputs written to the same seed directory:
  - normalized_cka_cohen_feature_importance.csv   (all features, all clusters)
  - normalized_cka_cohen_top{k}_per_cluster.csv    (top-k per cluster)
  - normalized_cka_cohen_cluster_importance.csv     (one row per cluster; column cluster_importance_topk_sum)

Usage
-----
  python compute_normalized_cka_cohen_importance.py <config.json> [--k K] [--run RUN] [--seed SEED] [--top-k K]

If --k / --run / --seed are omitted the script runs over all combinations in the JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

_analysis_dir = Path(__file__).resolve().parent
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

from local_cka_cohen_utils import (
    cohens_d_wide_to_long,
    compute_feature_importance_long,
    top_k_per_cluster,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(template: str, *, run: int, k: int, seed: int) -> str:
    return template.format(run=run, k=k, seed=seed)


def load_per_cluster_cka(seed_dir: str, k: int) -> pd.DataFrame:
    """
    Read selected_features_cka.csv from every cluster_* subdirectory and concatenate.
    Returns long DataFrame with columns: cluster_id, feature, cka_local.
    """
    del k  # retained for API symmetry with callers
    frames = []
    for entry in sorted(os.listdir(seed_dir)):
        if not entry.startswith("cluster_"):
            continue
        cka_path = os.path.join(seed_dir, entry, "selected_features_cka.csv")
        if not os.path.exists(cka_path):
            continue
        df = pd.read_csv(cka_path, usecols=["feature", "cka_to_embedding"])
        cluster_id = int(entry.split("_", 1)[1])
        df["cluster_id"] = cluster_id
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No cluster_*/selected_features_cka.csv found in {seed_dir}")
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"cka_to_embedding": "cka_local"})
    print(f"  [CKA]   loaded {len(result):,} rows from {len(frames)} clusters")
    return result


def process_one(config: dict, run: int, k: int, seed: int, *, top_k: int) -> None:
    seed_dir = _fmt(config["cluster_members_parent_dir_template"], run=run, k=k, seed=seed)
    cohens_d_csv = _fmt(config["cohens_d_by_cluster_out_csv_template"], run=run, k=k, seed=seed)

    print(f"\n{'='*70}")
    print(f"  run={run}  k={k}  seed={seed}  top_k={top_k}")
    print(f"  seed_dir : {seed_dir}")
    print(f"  cohen csv: {cohens_d_csv}")

    if not os.path.isdir(seed_dir):
        print("  [SKIP] seed_dir not found")
        return
    if not os.path.exists(cohens_d_csv):
        print("  [SKIP] cohens_d_across_clusters.csv not found")
        return

    cka_long = load_per_cluster_cka(seed_dir, k)
    cohen_wide = pd.read_csv(cohens_d_csv, low_memory=False)
    cohen_long = cohens_d_wide_to_long(cohen_wide)
    print(
        f"  [Cohen] loaded {len(cohen_long):,} rows ({cohen_long['cluster_id'].nunique()} clusters, "
        f"{cohen_long['feature'].nunique()} features)"
    )

    g_cka = float(cka_long["cka_local"].max())
    g_co = float(cohen_long["abs_cohens_d"].max())
    print(f"  Global max CKA:     {g_cka:.6f}")
    print(f"  Global max |Cohen|: {g_co:.6f}")

    importance_df = compute_feature_importance_long(cka_long, cohen_long)
    print(f"  Merged: {len(importance_df):,} rows after inner join")

    top_df, cluster_imp_df = top_k_per_cluster(importance_df, top_k)

    # --- write outputs ---
    all_out = os.path.join(seed_dir, "normalized_cka_cohen_feature_importance.csv")
    top_out = os.path.join(seed_dir, f"normalized_cka_cohen_top{top_k}_per_cluster.csv")
    cimp_out = os.path.join(seed_dir, "normalized_cka_cohen_cluster_importance.csv")

    importance_df.to_csv(all_out, index=False)
    top_df.to_csv(top_out, index=False)
    cluster_imp_df.to_csv(cimp_out, index=False)

    print(f"  Wrote: {all_out}")
    print(f"  Wrote: {top_out}")
    print(f"  Wrote: {cimp_out}")
    col = "cluster_importance_topk_sum"
    print(
        f"  Cluster importance range: "
        f"[{cluster_imp_df[col].min():.6f}, {cluster_imp_df[col].max():.6f}]"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to the JSON config file")
    parser.add_argument("--k", type=int, default=None, help="Specific k value (default: all in config)")
    parser.add_argument("--run", type=int, default=None, help="Specific run value (default: all in config)")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed value (default: all in config)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k features per cluster (overrides config top_k / normalized_cka_cohen_top_k).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    k_values = [args.k] if args.k is not None else config["k_values"]
    run_values = [args.run] if args.run is not None else config["run_values"]
    seed_values = [args.seed] if args.seed is not None else config["seed_values"]

    top_k_cfg = args.top_k
    if top_k_cfg is None:
        top_k_cfg = int(
            config.get("top_k", config.get("normalized_cka_cohen_top_k", 10))
        )

    print(f"Config: {args.config}")
    print(f"k_values:    {k_values}")
    print(f"run_values:  {run_values}")
    print(f"seed_values: {seed_values}")
    print(f"top_k:       {top_k_cfg}")

    for run in run_values:
        for k in k_values:
            for seed in seed_values:
                try:
                    process_one(config, run=run, k=k, seed=seed, top_k=top_k_cfg)
                except Exception as exc:
                    print(f"  [ERROR] run={run} k={k} seed={seed}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
