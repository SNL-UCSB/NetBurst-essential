#!/usr/bin/env python3
"""
Aggregate mean Fisher ratio (over all scored features) per (run, k) or per (run, seed, k)
from invariant_scores.csv under a clustering-then-invariance sweep output tree, pick the best
configuration per k (max mean Fisher), and write summary files in the sweep root directory.

Layouts supported:
  Legacy:
    <root>/run_<r>/K_<k>/invariant_scores.csv
  With per-seed outputs (recommended when seed_values has multiple entries):
    <root>/run_<r>/K_<k>/seed_<s>/invariant_scores.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


RUN_DIR_RE = re.compile(r"^run_(\d+)$")
K_DIR_RE = re.compile(r"^K_(\d+)$")
SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def _mean_fisher_from_invariant_scores(path: Path) -> Optional[Tuple[float, int]]:
    if not path.is_file():
        return None
    ratios = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "fisher_ratio" not in reader.fieldnames:
            return None
        for row in reader:
            try:
                ratios.append(float(row["fisher_ratio"]))
            except (TypeError, ValueError):
                continue
    if not ratios:
        return None
    return sum(ratios) / len(ratios), len(ratios)


# (k, run_id, seed_id or None for legacy flat layout, mean_fisher, n_features)
Record = Tuple[int, int, Optional[int], float, int]


def collect_matrix(root: Path) -> Tuple[List[Record], Set[int], Set[int], Set[int]]:
    """
    Returns (records, k set, run set, seed set — seed set only filled when seed subdirs exist).
    """
    records: List[Record] = []
    ks: Set[int] = set()
    runs: Set[int] = set()
    seeds: Set[int] = set()

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        m_run = RUN_DIR_RE.match(run_dir.name)
        if not m_run:
            continue
        run_id = int(m_run.group(1))
        runs.add(run_id)
        for k_dir in sorted(run_dir.iterdir()):
            if not k_dir.is_dir():
                continue
            m_k = K_DIR_RE.match(k_dir.name)
            if not m_k:
                continue
            k = int(m_k.group(1))
            ks.add(k)

            seed_dirs = [
                d
                for d in k_dir.iterdir()
                if d.is_dir() and SEED_DIR_RE.match(d.name)
            ]
            if seed_dirs:
                for sd in sorted(seed_dirs):
                    m_seed = SEED_DIR_RE.match(sd.name)
                    if not m_seed:
                        continue
                    seed_id = int(m_seed.group(1))
                    seeds.add(seed_id)
                    inv = sd / "invariant_scores.csv"
                    parsed = _mean_fisher_from_invariant_scores(inv)
                    if parsed is None:
                        continue
                    mf, n_feat = parsed
                    records.append((k, run_id, seed_id, mf, n_feat))
            else:
                inv = k_dir / "invariant_scores.csv"
                parsed = _mean_fisher_from_invariant_scores(inv)
                if parsed is None:
                    continue
                mf, n_feat = parsed
                records.append((k, run_id, None, mf, n_feat))

    return records, ks, runs, seeds


def best_per_k(records: List[Record]) -> Dict[int, Dict[str, Any]]:
    """For each k, configs with maximum mean Fisher; ties keep all winning (run, seed) keys."""
    by_k: Dict[int, List[Tuple[Tuple[int, Optional[int]], float]]] = {}
    for k, run_id, seed_id, mf, _n in records:
        key = (run_id, seed_id)
        by_k.setdefault(k, []).append((key, mf))

    out: Dict[int, Dict[str, Any]] = {}
    for k in sorted(by_k):
        pairs = by_k[k]
        best_mf = max(mf for _, mf in pairs)
        winners = sorted({key for key, mf in pairs if mf == best_mf})
        # Single winner: unpack run / seed
        best_run: Optional[int] = None
        best_seed: Optional[int] = None
        if len(winners) == 1:
            best_run, best_seed = winners[0]
        tied_repr = [_format_run_seed_key(r, s) for r, s in winners]
        out[k] = {
            "k": k,
            "best_mean_fisher_all_features": best_mf,
            "best_run": best_run,
            "best_seed": best_seed,
            "best_runs_tied": tied_repr,
            "best_runs_tied_pairs": [{"run": r, "seed": s} for r, s in winners],
            "n_configs_with_scores": len({key for key, _ in pairs}),
        }
    return out


def _format_run_seed_key(run_id: int, seed_id: Optional[int]) -> str:
    if seed_id is None:
        return f"run_{run_id}"
    return f"run_{run_id}_seed_{seed_id}"


def main():
    ap = argparse.ArgumentParser(
        description="Best (run[, seed]) per k by mean Fisher ratio over all features (invariant_scores.csv)."
    )
    ap.add_argument(
        "root",
        type=Path,
        help="Sweep output root, e.g. .../clustering_then_invariance_cosine_netburst_20260322",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only; do not write files.",
    )
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit("not a directory: {}".format(root))

    records, ks, runs, seeds = collect_matrix(root)
    if not records:
        raise SystemExit("no invariant_scores.csv found under {}".format(root))

    by_k = best_per_k(records)

    mean_csv = root / "mean_fisher_all_features_by_run_k.csv"
    best_csv = root / "best_run_per_k_mean_fisher.csv"
    best_json = root / "best_run_per_k_mean_fisher.json"

    if args.dry_run:
        print("Would write: {}, {}, {}".format(mean_csv, best_csv, best_json))
        print(json.dumps(by_k, indent=2))
        return

    with mean_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "run", "seed", "mean_fisher_all_features", "n_features"])
        for k, run_id, seed_id, mf, n_feat in sorted(records):
            w.writerow([k, run_id, "" if seed_id is None else seed_id, mf, n_feat])

    with best_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "k",
                "best_run",
                "best_seed",
                "best_mean_fisher_all_features",
                "best_configs_tied",
                "n_configs_compared",
            ]
        )
        for k in sorted(by_k):
            info = by_k[k]
            br = info["best_run"]
            bs = info["best_seed"]
            tied = ";".join(info["best_runs_tied"])
            w.writerow(
                [
                    k,
                    "" if br is None else br,
                    "" if bs is None else bs,
                    info["best_mean_fisher_all_features"],
                    tied,
                    info["n_configs_with_scores"],
                ]
            )

    payload = {
        "root": str(root),
        "k_values": sorted(ks),
        "run_values": sorted(runs),
        "seed_values": sorted(seeds) if seeds else None,
        "best_by_k": {str(k): v for k, v in sorted(by_k.items())},
    }
    with best_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print("Wrote {}".format(mean_csv))
    print("Wrote {}".format(best_csv))
    print("Wrote {}".format(best_json))


if __name__ == "__main__":
    main()
