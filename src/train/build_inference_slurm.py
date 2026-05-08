#!/usr/bin/env python3
"""Render a single multi-node Slurm AR inference launcher from a JSON jobs config.

Each (job, model) pair becomes one srun on its own node. Total --nodes equals
the sum over jobs of len(job["models"]). Hardcoded SBATCH defaults match
existing netburst_inference_ibgbi_manifest_*.sl scripts. The script either
submits the rendered file via sbatch (--submit) or just writes it (--dry-run).

JSON schema (top-level "jobs" list):

    {
      "jobs": [
        {
          "name": "ip_1s_main",
          "parquet_root": ".../ibgbi_test",
          "models": [".../NetBurstIBGBI_ip_1s_main_netburst_CE_intK_manifest"],
          "nz_thresh": 100,
          "bi_thresh": 100,
          "min_len": 2,
          "use_precomputed_context_forecast": true,
          "no_sampling": true,
          "save_pkl": "netburst_ibgbi_manifest_ip_1s_main.pkl",
          "per_example_csv": "per_example_metrics_netburst_ibgbi_manifest_ip_1s_main.csv",
          "aggregate_json": "final_metrics_netburst_ibgbi_manifest_ip_1s_main.json"
        },
        ...
      ]
    }

Required per job: name, parquet_root, models (non-empty list).
Defaults: nz_thresh=0, bi_thresh=nz_thresh, use_precomputed_context_forecast=true,
no_sampling=true. Output filenames default to netburst_ibgbi_manifest_<name>{...}
when omitted. When a job has multiple models, save_pkl / per_example_csv /
aggregate_json are suffixed _m0, _m1, ... before the file extension.

Examples:
    python build_inference_slurm.py --config example_inference_jobs.json --dry-run
    python build_inference_slurm.py --config my.json --submit --out /tmp/my.sl
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_TRAIN_DIR = "<repo-root>/src/train"
DEFAULT_OUT_DIR = Path(REPO_TRAIN_DIR) / "slurm_files" / "from_config"

SBATCH_JOB_NAME = "netburst_inf_from_config"
SBATCH_WALLTIME = "10:00:00"

SBATCH_HEADER_TEMPLATE = """\
#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --licenses=scratch
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=premium

module load conda
module load pytorch/2.6.0
"""

INFERENCE_DEFAULTS = {
    "nz_thresh": 0,
    "use_precomputed_context_forecast": True,
    "no_sampling": True,
}


def _err(msg: str) -> None:
    sys.stderr.write(f"ERROR: {msg}\n")
    sys.exit(2)


def _check_no_single_quote(name: str, value) -> None:
    if isinstance(value, str) and "'" in value:
        _err(
            f"field {name!r} contains a single quote, which would break the "
            f"`bash -c '...'` wrapper. Offending value: {value!r}"
        )


def _walk_check_no_quote(prefix: str, value) -> None:
    if isinstance(value, str):
        _check_no_single_quote(prefix, value)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _walk_check_no_quote(f"{prefix}[{i}]", v)
    elif isinstance(value, dict):
        for k, v in value.items():
            _walk_check_no_quote(f"{prefix}.{k}", v)


def _validate_inference_job(idx: int, job: dict) -> None:
    if not isinstance(job, dict):
        _err(f"jobs[{idx}] must be an object, got {type(job).__name__}.")
    for required in ("name", "parquet_root", "models"):
        if required not in job:
            _err(f"jobs[{idx}] missing required field {required!r}.")
    for k in ("name", "parquet_root"):
        if not isinstance(job[k], str) or not job[k]:
            _err(f"jobs[{idx}].{k} must be a non-empty string.")
    if not isinstance(job["models"], list) or not job["models"]:
        _err(f"jobs[{idx}].models must be a non-empty list.")
    for j, m in enumerate(job["models"]):
        if not isinstance(m, str) or not m:
            _err(f"jobs[{idx}].models[{j}] must be a non-empty string.")


def _suffix_basename(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _render_inference_srun(job: dict, model: str, model_idx: int, n_models: int) -> str:
    name = job["name"]
    parquet_root = job["parquet_root"]

    nz = job.get("nz_thresh", INFERENCE_DEFAULTS["nz_thresh"])
    bi = job.get("bi_thresh", nz)

    save_pkl = job.get("save_pkl", f"netburst_ibgbi_manifest_{name}.pkl")
    per_example_csv = job.get(
        "per_example_csv", f"per_example_metrics_netburst_ibgbi_manifest_{name}.csv"
    )
    aggregate_json = job.get(
        "aggregate_json", f"final_metrics_netburst_ibgbi_manifest_{name}.json"
    )

    if n_models > 1:
        sfx = f"_m{model_idx}"
        save_pkl = _suffix_basename(save_pkl, sfx)
        per_example_csv = _suffix_basename(per_example_csv, sfx)
        aggregate_json = _suffix_basename(aggregate_json, sfx)

    flags = [f'--model "{model}"']
    if bool(
        job.get(
            "use_precomputed_context_forecast",
            INFERENCE_DEFAULTS["use_precomputed_context_forecast"],
        )
    ):
        flags.append("--use_precomputed_context_forecast")
    if bool(job.get("no_sampling", INFERENCE_DEFAULTS["no_sampling"])):
        flags.append("--no_sampling")
    flags.append(f"--nz_thresh {nz}")
    flags.append(f"--bi_thresh {bi}")
    if "min_len" in job:
        flags.append(f"--min_len {int(job['min_len'])}")
    flags.append(f'--save_pkl "{save_pkl}"')
    flags.append(f'--per_example_csv "{per_example_csv}"')
    flags.append(f'--aggregate_json "{aggregate_json}"')

    cmd = (
        f"cd {REPO_TRAIN_DIR} && "
        f'torchrun --nproc_per_node=4 ar_predict.py "{parquet_root}" '
        + " ".join(flags)
    )
    comment = f"# job={name} model_idx={model_idx} model={model}"
    return f"{comment}\nsrun --exclusive -N 1 -n 1 bash -c '{cmd}' &\n"


def render_inference_slurm(cfg: dict) -> str:
    jobs = cfg.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        _err("top-level `jobs` must be a non-empty list.")
    for i, job in enumerate(jobs):
        _validate_inference_job(i, job)
    _walk_check_no_quote("config", cfg)

    n_total = sum(len(j["models"]) for j in jobs)
    head = SBATCH_HEADER_TEMPLATE.format(
        job_name=SBATCH_JOB_NAME, walltime=SBATCH_WALLTIME, nodes=n_total
    )

    body_blocks = []
    for job in jobs:
        n_m = len(job["models"])
        for k, model in enumerate(job["models"]):
            body_blocks.append(_render_inference_srun(job, model, k, n_m))

    tail = f'\nwait\necho "All {n_total} (job, model) inference runs completed."\n'
    return head + "\n" + "\n".join(body_blocks) + tail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to JSON config with top-level `jobs: [...]`.",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--submit", action="store_true", help="Submit the rendered .sl with sbatch."
    )
    grp.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only write the .sl; do not submit.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .sl path (default: src/train/slurm_files/from_config/<json_stem>_inference_<TS>.sl).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        _err(f"config not found: {cfg_path}")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        _err(f"failed to parse JSON {cfg_path}: {e}")

    sl_text = render_inference_slurm(cfg)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = DEFAULT_OUT_DIR / f"{cfg_path.stem}_inference_{ts}.sl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(sl_text)
    try:
        out_path.chmod(0o755)
    except OSError:
        pass
    print(f"Wrote: {out_path}")

    n_total = sum(len(j["models"]) for j in cfg["jobs"])
    print(f"Nodes: {n_total} (one srun per (job, model) pair)")

    if args.submit:
        print(f"Submitting via sbatch: {out_path}")
        result = subprocess.run(["sbatch", str(out_path)])
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
