#!/usr/bin/env python3
"""
Thin dispatcher for analysis scripts with JSON-driven defaults under ``src/analysis/pipelines/``.

Examples:

  python src/analysis/run_analysis_pipeline.py global-cka --config my_paths.json

  python src/analysis/run_analysis_pipeline.py anisotropy --config pipelines/anisotropy_only.defaults.json

  python src/analysis/run_analysis_pipeline.py cluster-features --config my_cluster_run.json

  python src/analysis/run_analysis_pipeline.py normalized-cka-cohen --config norm_cka.json

Each ``--config`` JSON is merged with the defaults file for that subcommand.
Underscore-prefixed keys are stripped before dispatch.

For ``cluster-features``, this invokes
``submit_cluster_invariance_from_json.py --mode local`` with the merged JSON (same pipeline as
interactive Slurm submitter: optional clustering_phase → cluster_analysis → optional overlay → optional member plots).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

_ANALYSIS = Path(__file__).resolve().parent
_ROOT = _ANALYSIS.parent.parent
_PIPELINES = _ANALYSIS / "pipelines"
_SUBMIT_CLUSTER = _ANALYSIS / "cluster_invariance_from_json" / "submit_cluster_invariance_from_json.py"


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _strip_meta_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop underscore-prefixed keys (comments) so argparse ``set_defaults`` does not see them."""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def _load_merged_config(defaults_path: Path | None, config_path: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if defaults_path is not None and defaults_path.is_file():
        with open(defaults_path, encoding="utf-8") as f:
            merged.update(_strip_meta_keys(json.load(f)))
    with open(config_path, encoding="utf-8") as f:
        merged = _deep_merge(merged, _strip_meta_keys(json.load(f)))
    return _strip_meta_keys(merged)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_g = sub.add_parser("global-cka", help="Run global_cka_fisher_metrics.run_from_config_dict")
    p_g.add_argument("--config", required=True, help="JSON with CLI-equivalent keys.")
    p_g.add_argument(
        "--defaults",
        default=str(_PIPELINES / "global_cka_only.defaults.json"),
        help=f"Base JSON merged before --config (default: {_PIPELINES / 'global_cka_only.defaults.json'}).",
    )

    p_a = sub.add_parser("anisotropy", help="Run anisotropy_scores.run_from_config_dict")
    p_a.add_argument("--config", required=True)
    p_a.add_argument(
        "--defaults",
        default=str(_PIPELINES / "anisotropy_only.defaults.json"),
        help=f"Base JSON merged before --config (default: {_PIPELINES / 'anisotropy_only.defaults.json'}).",
    )

    p_n = sub.add_parser(
        "normalized-cka-cohen",
        help="Run compute_normalized_cka_cohen_importance main() with merged JSON config file.",
    )
    p_n.add_argument("--config", required=True)
    p_n.add_argument(
        "--defaults",
        default=str(_PIPELINES / "normalized_cka_cohen.defaults.json"),
        help="Base JSON merged before --config.",
    )
    p_n.add_argument("--k", type=int, default=None)
    p_n.add_argument("--run", type=int, default=None)
    p_n.add_argument("--seed", type=int, default=None)
    p_n.add_argument("--top-k", type=int, default=None, dest="top_k_override")

    p_cf = sub.add_parser(
        "cluster-features",
        help="Run cluster invariance pipeline locally via submit_cluster_invariance_from_json.py --mode local.",
    )
    p_cf.add_argument("--config", required=True, help="JSON matching submit_cluster_invariance_from_json schema.")
    p_cf.add_argument(
        "--defaults",
        default=str(_PIPELINES / "cluster_features.defaults.json"),
        help=f"Base JSON merged before --config (default: {_PIPELINES / 'cluster_features.defaults.json'}).",
    )
    p_cf.add_argument(
        "--artifact-root",
        default=None,
        help="Optional base directory merged into config as artifact_root (run-scoped outputs).",
    )
    p_cf.add_argument(
        "--run-tag",
        default=None,
        help='Optional run subdirectory tag merged into config; use "auto" for a timestamp.',
    )

    args = ap.parse_args()

    if args.command == "global-cka":
        cfg_path = Path(os.path.expanduser(args.config)).resolve()
        def_path = Path(os.path.expanduser(args.defaults)).resolve() if args.defaults else None
        merged = _load_merged_config(def_path, cfg_path)
        sys.path.insert(0, str(_ANALYSIS))
        from global_cka_fisher_metrics import run_from_config_dict

        run_from_config_dict(merged)
        return

    if args.command == "anisotropy":
        cfg_path = Path(os.path.expanduser(args.config)).resolve()
        def_path = Path(os.path.expanduser(args.defaults)).resolve() if args.defaults else None
        merged = _load_merged_config(def_path, cfg_path)
        sys.path.insert(0, str(_ANALYSIS))
        from anisotropy_scores import run_from_config_dict

        run_from_config_dict(merged)
        return

    if args.command == "normalized-cka-cohen":
        cfg_path = Path(os.path.expanduser(args.config)).resolve()
        def_path = Path(os.path.expanduser(args.defaults)).resolve() if args.defaults else None
        merged = _load_merged_config(def_path, cfg_path)
        if args.top_k_override is not None:
            merged["top_k"] = int(args.top_k_override)
        tmp = cfg_path.with_name(cfg_path.stem + "_merged_tmp.json")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            sys.path.insert(0, str(_ANALYSIS))
            import compute_normalized_cka_cohen_importance as norm_mod

            argv = [str(tmp)]
            if args.k is not None:
                argv += ["--k", str(args.k)]
            if args.run is not None:
                argv += ["--run", str(args.run)]
            if args.seed is not None:
                argv += ["--seed", str(args.seed)]
            if args.top_k_override is not None:
                argv += ["--top-k", str(args.top_k_override)]
            sys.argv = [norm_mod.__file__] + argv
            norm_mod.main()
        finally:
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
        return

    if args.command == "cluster-features":
        cfg_path = Path(os.path.expanduser(args.config)).resolve()
        def_path = Path(os.path.expanduser(args.defaults)).resolve() if args.defaults else None
        merged = _load_merged_config(def_path, cfg_path)
        if getattr(args, "artifact_root", None):
            merged["artifact_root"] = args.artifact_root
        if getattr(args, "run_tag", None):
            merged["run_tag"] = args.run_tag
        tmp = cfg_path.with_name(cfg_path.stem + "_merged_tmp.json")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            subprocess.run(
                [
                    sys.executable,
                    str(_SUBMIT_CLUSTER),
                    "--json",
                    str(tmp),
                    "--mode",
                    "local",
                ],
                check=True,
                cwd=str(_ROOT),
            )
        finally:
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
        return


if __name__ == "__main__":
    main()
