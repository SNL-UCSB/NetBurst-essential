#!/usr/bin/env python3
"""
Entry point for clustering (KMeans labels) + cluster-invariance sweeps.

Implements the same CLI as cluster_invariance_sweep_from_json/submit_cluster_invariance_sweep_from_json.py;
enable clustering_phase in the JSON (see example JSON in this folder).
"""
import importlib.util
from pathlib import Path


def _load_sweep_main():
    sweep_path = Path(__file__).resolve().parents[1] / "cluster_invariance_sweep_from_json" / "submit_cluster_invariance_sweep_from_json.py"
    spec = importlib.util.spec_from_file_location("submit_cluster_invariance_sweep_from_json", str(sweep_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load sweep driver from {sweep_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.main


def main() -> None:
    main_fn = _load_sweep_main()
    main_fn()


if __name__ == "__main__":
    main()
