#!/usr/bin/env python3
"""
Run sparse -> IBG/BI conversion and split generation from a YAML manifest.

Each job is processed independently.

Default pipeline (`pipeline: sparse_to_ibgbi` or omitted):
1) Converts sparse series to IBG/BI using SparseToIBGBIFromSparse.py
2) Creates train/test + context/forecast splits using create_train_test_splits.py

Alternate pipeline (`pipeline: existing_ibgbi`): IBG-BI already exists (e.g. PerfSONAR, ping).
Runs create_train_test_splits.py only. Optional ``sparse_dir`` in the job enables sparse train/test mirroring.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple


ENTITY_SUFFIX = {
    "ip": "ip",
    "ip_service": "service",
    "subnet": "subnet",
    "ip2ip": "ip2ip",
    "ping_latency": "ping_latency",
}


def _load_manifest(path: str) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Manifest must be a YAML object")
    if "jobs" not in data or not isinstance(data["jobs"], list) or not data["jobs"]:
        raise ValueError("Manifest must contain non-empty 'jobs' list")
    return data


def _require(job: Dict, key: str):
    if key not in job or job[key] in (None, ""):
        raise ValueError(f"Missing required field '{key}' in job: {job}")


def _run(cmd: List[str], dry_run: bool):
    print("[CMD] " + " ".join(cmd))
    if dry_run:
        return ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result.stdout.strip()


def _resolve_threshold(
    preprocess_dir: str,
    python_exec: str,
    job: Dict,
    dry_run: bool,
) -> float:
    mode = str(job.get("threshold_mode", "value")).strip().lower()
    if mode == "value":
        _require(job, "threshold_value")
        threshold_value = float(job["threshold_value"])
        return threshold_value

    if mode == "quantile":
        _require(job, "threshold_quantiles_path")
        _require(job, "threshold_quantile")
        quantiles_path = str(job["threshold_quantiles_path"])
        quantile = float(job["threshold_quantile"])
        entity_type = str(job["entity_type"])

        selector = os.path.join(preprocess_dir, "select_bi_threshold.py")
        selector_cmd = [
            python_exec,
            selector,
            "--quantiles_path", quantiles_path,
            "--entity_type", entity_type,
            "--quantile", str(quantile),
        ]
        out = _run(selector_cmd, dry_run=dry_run)
        threshold_value = 0.0 if dry_run else float(out.splitlines()[-1].strip())
        return threshold_value

    raise ValueError(f"Unsupported threshold_mode '{mode}' (use 'value' or 'quantile')")


def _run_existing_ibgbi_splits_only(
    job: Dict,
    preprocess_dir: str,
    python_exec: str,
    dry_run: bool,
) -> None:
    """Splits from pre-built IBG-BI (no SparseToIBGBIFromSparse step).

    If ``sparse_dir`` is set in the job, also mirrors sparse train/test under the output dir.
    If ``sparse_dir`` is omitted or null, only ``splits_meta``, ``ibgbi_train``, and ``ibgbi_test`` are written.

    If ``mirror_sparse_only`` is true, runs create_train_test_splits.py with ``--mirror_sparse_only``:
    reads existing ``splits_meta`` and writes only ``sparse_train`` / ``sparse_test`` (same train/test
    keys as the prior split).
    """
    split_script = os.path.join(preprocess_dir, "create_train_test_splits.py")

    if bool(job.get("mirror_sparse_only")):
        _require(job, "split_output_dir")
        _require(job, "entity_type")
        _require(job, "sparse_dir")
        alignment = str(job.get("alignment_mode", "burst_timestamp"))
        split_cmd = [
            python_exec,
            split_script,
            "--mirror_sparse_only",
            "--entity_type",
            str(job["entity_type"]),
            "--output_dir",
            str(job["split_output_dir"]),
            "--sparse_dir",
            str(job["sparse_dir"]),
            "--alignment_mode",
            alignment,
            "--burst_column",
            str(job.get("burst_column", "inbound")),
        ]
        if alignment == "burst_timestamp":
            _require(job, "burst_threshold")
            split_cmd.extend(["--burst_threshold", str(job["burst_threshold"])])
        smd = job.get("splits_meta_dir")
        if smd not in (None, ""):
            split_cmd.extend(["--splits_meta_dir", str(smd)])
        _run(split_cmd, dry_run=dry_run)
        return

    required = [
        "ibgbi_dir",
        "split_output_dir",
        "min_bursts",
        "train_ratio",
        "context_ratio",
        "seed",
        "bin_ms",
    ]
    for k in required:
        _require(job, k)

    split_cmd = [
        python_exec,
        split_script,
        "--ibgbi_dir",
        str(job["ibgbi_dir"]),
        "--entity_type",
        str(job["entity_type"]),
        "--output_dir",
        str(job["split_output_dir"]),
        "--min_bursts",
        str(job["min_bursts"]),
        "--train_ratio",
        str(job["train_ratio"]),
        "--context_ratio",
        str(job["context_ratio"]),
        "--seed",
        str(job["seed"]),
        "--bin_ms",
        str(job["bin_ms"]),
        "--alignment_mode",
        str(job.get("alignment_mode", "index")),
        "--burst_column",
        str(job.get("burst_column", "inbound")),
    ]
    sparse_dir = job.get("sparse_dir")
    if sparse_dir not in (None, ""):
        _require(job, "burst_threshold")
        split_cmd.extend(
            [
                "--sparse_dir",
                str(sparse_dir),
                "--burst_threshold",
                str(job["burst_threshold"]),
            ]
        )
    if bool(job.get("ibg_in_ms", False)):
        split_cmd.append("--ibg_in_ms")
    if job.get("min_seq_len") is not None:
        split_cmd.extend(["--min_seq_len", str(job["min_seq_len"])])
    if job.get("max_seq_len") is not None:
        split_cmd.extend(["--max_seq_len", str(job["max_seq_len"])])
    _run(split_cmd, dry_run=dry_run)


def _job_defaults(job: Dict) -> Dict:
    out = dict(job)
    out.setdefault("bin_ms", 1000)
    out.setdefault("min_seq_len", 10)
    out.setdefault("max_seq_len", 9000)
    out.setdefault("min_bursts", 10)
    out.setdefault("train_ratio", 0.7)
    out.setdefault("context_ratio", 0.7)
    out.setdefault("seed", 42)
    out.setdefault("alignment_mode", "burst_timestamp")
    out.setdefault("ibg_in_ms", False)
    out.setdefault("outbound", False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run sparse->IBG/BI conversion and split generation from YAML manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to YAML manifest")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--slurm_shard_jobs",
        action="store_true",
        help="Shard manifest jobs across SLURM tasks: job i runs on task (i % SLURM_NTASKS).",
    )
    parser.add_argument(
        "--job_index", type=int, default=None,
        help="0-based index of the single job to run (for SLURM job arrays). Omit to run all jobs.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    all_jobs = manifest["jobs"]

    if args.job_index is not None:
        if args.job_index < 0 or args.job_index >= len(all_jobs):
            raise SystemExit(
                f"--job_index {args.job_index} out of range; manifest has {len(all_jobs)} jobs (0-{len(all_jobs)-1})"
            )
        jobs_to_run = [all_jobs[args.job_index]]
        print(f"Job array mode: running job index {args.job_index} of {len(all_jobs)}")
    else:
        jobs_to_run = all_jobs

    if args.slurm_shard_jobs:
        task_rank = int(os.getenv("SLURM_PROCID", "0"))
        num_tasks = int(os.getenv("SLURM_NTASKS", "1"))
        pre_shard_count = len(jobs_to_run)
        jobs_to_run = [j for i, j in enumerate(jobs_to_run) if (i % num_tasks) == task_rank]
        print(
            f"SLURM shard mode: task_rank={task_rank}/{num_tasks}, "
            f"assigned {len(jobs_to_run)} of {pre_shard_count} jobs"
        )

    preprocess_dir = os.path.dirname(os.path.abspath(__file__))
    python_exec = sys.executable

    summary = []
    for idx, raw_job in enumerate(jobs_to_run, start=1):
        pipeline = str(raw_job.get("pipeline") or "sparse_to_ibgbi").strip().lower()
        if pipeline in ("existing_ibgbi", "splits_from_existing"):
            job = dict(raw_job)
            job["pipeline"] = "existing_ibgbi"
            _require(job, "name")
            _require(job, "entity_type")
            entity_type = str(job["entity_type"])
            if entity_type not in ENTITY_SUFFIX:
                raise ValueError(f"Unsupported entity_type '{entity_type}' in job '{job['name']}'")

            print(f"\n=== Job {idx}/{len(jobs_to_run)}: {job['name']} [pipeline=existing_ibgbi] ===")
            _run_existing_ibgbi_splits_only(job, preprocess_dir, python_exec, args.dry_run)

            split_output_dir = str(job["split_output_dir"])
            os.makedirs(split_output_dir, exist_ok=True)
            run_cfg_path = os.path.join(split_output_dir, "run_config.json")
            run_cfg = {
                "manifest": os.path.abspath(args.manifest),
                "job": job,
                "pipeline": "existing_ibgbi",
            }
            if not args.dry_run:
                with open(run_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(run_cfg, f, indent=2)

            summary.append({
                "name": job["name"],
                "entity_type": entity_type,
                "pipeline": "existing_ibgbi",
                "ibgbi_dir": str(job["ibgbi_dir"]),
                "sparse_dir": str(job["sparse_dir"]),
                "split_output_dir": split_output_dir,
            })
            continue

        job = _job_defaults(raw_job)
        _require(job, "name")
        _require(job, "entity_type")
        _require(job, "sparse_input_dir")
        _require(job, "ibg_output_dir")
        _require(job, "split_output_dir")

        entity_type = str(job["entity_type"])
        if entity_type not in ENTITY_SUFFIX:
            raise ValueError(f"Unsupported entity_type '{entity_type}' in job '{job['name']}'")

        sparse_input_dir = str(job["sparse_input_dir"])
        ibg_output_dir = str(job["ibg_output_dir"])
        split_output_dir = str(job["split_output_dir"])

        threshold_value = _resolve_threshold(
            preprocess_dir=preprocess_dir,
            python_exec=python_exec,
            job=job,
            dry_run=args.dry_run,
        )

        print(f"\n=== Job {idx}/{len(jobs_to_run)}: {job['name']} [pipeline=sparse_to_ibgbi] ===")

        ibg_script = os.path.join(preprocess_dir, "SparseToIBGBIFromSparse.py")
        ibg_cmd = [
            "srun",
            python_exec,
            ibg_script,
            "--sparse_input", sparse_input_dir,
            "--entity_type", entity_type,
            "--output_dir", ibg_output_dir,
            "--burst_column", "outbound" if bool(job.get("outbound", False)) else "inbound",
            "--threshold", str(threshold_value),
            "--bin_ms", str(job["bin_ms"]),
            "--min_seq_len", str(job["min_seq_len"]),
            "--max_seq_len", str(job["max_seq_len"]),
        ]
        if bool(job.get("ibg_in_ms", False)):
            ibg_cmd.append("--ibg_in_ms")
        _run(ibg_cmd, dry_run=args.dry_run)

        ibgbi_dir = os.path.join(ibg_output_dir, "task_*")
        split_script = os.path.join(preprocess_dir, "create_train_test_splits.py")
        split_cmd = [
            python_exec,
            split_script,
            "--ibgbi_dir", ibgbi_dir,
            "--entity_type", entity_type,
            "--output_dir", split_output_dir,
            "--min_bursts", str(job["min_bursts"]),
            "--train_ratio", str(job["train_ratio"]),
            "--context_ratio", str(job["context_ratio"]),
            "--seed", str(job["seed"]),
            "--sparse_dir", sparse_input_dir,
            "--alignment_mode", str(job["alignment_mode"]),
            "--burst_column", "outbound" if bool(job.get("outbound", False)) else "inbound",
            "--burst_threshold", str(threshold_value),
            "--bin_ms", str(job["bin_ms"]),
        ]
        if bool(job.get("ibg_in_ms", False)):
            split_cmd.append("--ibg_in_ms")
        if job.get("min_seq_len") is not None:
            split_cmd.extend(["--min_seq_len", str(job["min_seq_len"])])
        if job.get("max_seq_len") is not None:
            split_cmd.extend(["--max_seq_len", str(job["max_seq_len"])])
        _run(split_cmd, dry_run=args.dry_run)

        os.makedirs(split_output_dir, exist_ok=True)
        run_cfg_path = os.path.join(split_output_dir, "run_config.json")
        run_cfg = {
            "manifest": os.path.abspath(args.manifest),
            "job": job,
            "resolved_threshold": threshold_value,
            "resolved_ibgbi_dir": ibgbi_dir,
            "pipeline": "sparse_to_ibgbi",
        }
        if not args.dry_run:
            with open(run_cfg_path, "w", encoding="utf-8") as f:
                json.dump(run_cfg, f, indent=2)

        summary.append({
            "name": job["name"],
            "entity_type": entity_type,
            "pipeline": "sparse_to_ibgbi",
            "sparse_input_dir": sparse_input_dir,
            "ibg_output_dir": ibg_output_dir,
            "split_output_dir": split_output_dir,
            "resolved_threshold": threshold_value,
        })

    print("\nCompleted manifest run.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()