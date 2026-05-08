#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import uuid


_ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))


def merge_sbatch_dependency(
    precursor_slurm_job_id: Optional[str],
    dependency: Optional[str],
) -> Optional[str]:
    deps: list[str] = []
    if dependency and str(dependency).strip():
        deps.append(str(dependency).strip())
    if precursor_slurm_job_id and str(precursor_slurm_job_id).strip():
        deps.append(f"afterok:{str(precursor_slurm_job_id).strip()}")
    return ",".join(deps) if deps else None

_CLUSTER_INVARIANCE_DRIVER_MOD = None
DEFAULT_ENVIRONMENT = "nersc"


def _get_cluster_invariance_driver_module():
    """
    Load the single-job driver module by filesystem path.
    This avoids needing Python packages (__init__.py) under `src/`.
    """
    global _CLUSTER_INVARIANCE_DRIVER_MOD
    if _CLUSTER_INVARIANCE_DRIVER_MOD is not None:
        return _CLUSTER_INVARIANCE_DRIVER_MOD

    analysis_dir = Path(__file__).resolve().parents[1]  # .../src/analysis
    driver_path = analysis_dir / "cluster_invariance_from_json" / "submit_cluster_invariance_from_json.py"
    spec = importlib.util.spec_from_file_location("cluster_invariance_submit", str(driver_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load driver module from {driver_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    _CLUSTER_INVARIANCE_DRIVER_MOD = mod
    return mod


def _load_config(json_path: str) -> dict:
    p = Path(json_path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
    return default


def _invariance_phase_enabled(sweep_cfg: dict) -> bool:
    """When false, only clustering_phase runs (no tsfresh / invariance / plots). Default true."""
    return _parse_bool((sweep_cfg.get("invariance_phase") or {}).get("enabled"), default=True)


def _format_tmpl(template: str, *, k: int, run: Optional[int], seed: Optional[int] = None) -> str:
    # Supports templates that contain "{k}", optionally "{run}", and optionally "{seed}".
    return template.format(k=k, run=run, seed=seed)


def _clustering_phase_bash_string(
    sweep_cfg: dict,
    chunk: List[dict],
    bash_quote: Any,
) -> str:
    """
    Bash snippet: load env + one clustering_analysis.py invocation per k in chunk (sequential).
    Uses clustering_labels_template to derive --out-dir (parent of k_{k}_labels.csv path).
    Empty string if clustering_phase.enabled is false.
    """
    cp = sweep_cfg.get("clustering_phase") or {}
    if not isinstance(cp, dict) or not _parse_bool(cp.get("enabled"), default=False):
        return ""
    if not chunk:
        return ""
    data_csv = cp.get("data_csv")
    if not data_csv:
        raise ValueError("clustering_phase.enabled=true requires clustering_phase.data_csv")
    tmpl = sweep_cfg.get("clustering_labels_template")
    if not tmpl:
        raise ValueError("clustering_phase requires sweep clustering_labels_template")

    workdir = str(sweep_cfg.get("workdir", "<external-netburst-root>/src/analysis"))
    netburst_root = cp.get("netburst_root")
    if not netburst_root:
        netburst_root = str(Path(workdir).resolve().parents[1])
    script_path = str(Path(str(netburst_root)).resolve() / "src" / "analysis" / "clustering_analysis.py")

    run0 = chunk[0]["run"]
    seed0 = chunk[0]["seed"]
    k_list = sorted({int(m["k"]) for m in chunk})

    kmeans_metric = str(cp.get("kmeans_metric", "cosine"))
    minimal_cleaning = _parse_bool(cp.get("minimal_cleaning"), default=True)
    ip_sfx_cluster: Optional[str] = None
    if isinstance(cp, dict):
        v = cp.get("ip_suffix_filter")
        if v is not None and str(v).strip():
            ip_sfx_cluster = str(v).strip()
    if ip_sfx_cluster is None:
        v = sweep_cfg.get("ip_suffix_filter")
        if v is not None and str(v).strip():
            ip_sfx_cluster = str(v).strip()
    if cp.get("random_state") is not None:
        rs = int(cp["random_state"])
    elif seed0 is not None:
        rs = int(seed0)
    else:
        rs = 42

    data_q = bash_quote(str(Path(str(data_csv)).expanduser()))
    parts: list[str] = [
        'THREADS_PER_RANK="${SLURM_CPUS_PER_TASK:-64}"',
        'if [ "$THREADS_PER_RANK" -gt 64 ]; then THREADS_PER_RANK=64; fi',
        'export OMP_NUM_THREADS="${THREADS_PER_RANK}"',
        'export OPENBLAS_NUM_THREADS="${THREADS_PER_RANK}"',
        'export GOTO_NUM_THREADS="${THREADS_PER_RANK}"',
        'export MKL_NUM_THREADS="${THREADS_PER_RANK}"',
        'export BLIS_NUM_THREADS="${THREADS_PER_RANK}"',
        'export VECLIB_MAXIMUM_THREADS="${THREADS_PER_RANK}"',
        'export NUMEXPR_MAX_THREADS="${THREADS_PER_RANK}"',
        'export NUMEXPR_NUM_THREADS="${THREADS_PER_RANK}"',
        "export SLURM_CPU_BIND=cores",
        f'export PYTHONHASHSEED="{rs}"',
    ]
    for k in k_list:
        labels_path = _format_tmpl(str(tmpl), k=int(k), run=run0, seed=seed0)
        out_dir = str(Path(labels_path).expanduser().parent)
        parts.append(f"mkdir -p {bash_quote(out_dir)}")
        parts.append(
            f'echo "[clustering_phase] K={int(k)} run={run0} seed={seed0} out_dir={out_dir}"'
        )
        extra = " --minimal-cleaning" if minimal_cleaning else ""
        ip_extra = f" --ip-suffix-filter {bash_quote(ip_sfx_cluster)}" if ip_sfx_cluster else ""
        use_all = _parse_bool(cp.get("use_all_features"), default=False)
        all_feat = " --use-all-features" if use_all else ""
        parts.append(
            "python3 -u "
            f"{bash_quote(script_path)} "
            f"{data_q} "
            f"--n-clusters {int(k)} "
            f"--kmeans-metric {bash_quote(kmeans_metric)}"
            f"{extra}"
            f"{all_feat}"
            f"{ip_extra} "
            f"--out-dir {bash_quote(out_dir)} "
            f"--random-state {rs}"
        )
    return " && ".join(parts)


def _quote_bash(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _resolve_runner(*, mode: str, cfg_runner: Any) -> str:
    if cfg_runner is None:
        if mode == "preview":
            return "slurm"
        return mode
    runner = str(cfg_runner).strip().lower()
    if runner not in {"local", "slurm"}:
        raise ValueError("runner must be one of: local, slurm")
    return runner


def _resolve_environment(value: Any) -> str:
    env = str(value if value is not None else DEFAULT_ENVIRONMENT).strip().lower()
    if env not in {"nersc", "local"}:
        raise ValueError("environment must be one of: nersc, local")
    return env


def _run_bash(command: str) -> None:
    subprocess.run(["bash", "-lc", command], check=True)


def _ensure_dir(path: str) -> None:
    Path(path).expanduser().mkdir(parents=True, exist_ok=True)


def _derive_out_dir(out_csv: str) -> str:
    return str(Path(out_csv).expanduser().parent)


def _norm_run_seed_pair(run: Any, seed: Any) -> Tuple[Any, Any]:
    """
    Normalize (run, seed) for chunk grouping so one Slurm task is allocated per pair.
    Coerces numeric run/seed to int (e.g. numpy scalars vs Python int hash the same bucket).
    """
    def _one(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, bool):
            return x
        try:
            return int(x)
        except (TypeError, ValueError):
            return x

    return (_one(run), _one(seed))


def _group_members_by_run_seed(member_cfgs: list[dict]) -> List[List[dict]]:
    """
    Group sweep members that share the same (run, seed); within each group sort by k.
    Each group is one Slurm chunk: all k values for that (run, seed) run in parallel on one node.
    """
    d: Dict[Tuple[Any, Any], List[dict]] = defaultdict(list)
    for m in member_cfgs:
        key = _norm_run_seed_pair(m.get("run"), m.get("seed"))
        d[key].append(m)

    keys = sorted(d.keys(), key=lambda t: (str(t[0]), str(t[1])))
    return [sorted(d[k], key=lambda x: int(x["k"])) for k in keys]


def _format_per_run_job_name(template: str, sweep_job_name: str, run: Any, seed: Any) -> str:
    return template.format(
        sweep_job_name=sweep_job_name,
        run=run if run is not None else "none",
        seed=seed if seed is not None else "none",
    )


def _bash_quote_path_for_shell(path: str, bash_quote: Any) -> str:
    """
    Paths that use ${STAGE_DIR}/... must be double-quoted so bash expands them after STAGE_DIR
    is set. Single-quoted shlex-style quoting passes a literal ${STAGE_DIR}/... to Python, which
    is not a real filesystem path; unquoted or broken quoting with an empty STAGE_DIR yields
    /clustering_j.csv at the filesystem root (FileNotFoundError).
    """
    if "${STAGE_DIR}" in path:
        return '"' + path.replace('"', '\\"') + '"'
    return bash_quote(path)


def _staging_shell_lines(staging_cfg: dict) -> list[str]:
    """
    Bash snippet for one chunk: STAGE_DIR + export + mkdir.
    export STAGE_DIR puts the path in the environment for Python os.path.expandvars and for
    child processes; the shell still expands "${STAGE_DIR}/..." on the argv before exec.
    """
    cfg = staging_cfg if isinstance(staging_cfg, dict) else {}
    shm_root = str(cfg.get("shm_root", "/dev/shm")).strip().rstrip("/") or "/dev/shm"
    prefix = str(cfg.get("stage_dir_prefix", "netburst_invariance_")).strip() or "netburst_invariance_"
    if "/" in prefix or prefix.startswith(".."):
        raise ValueError(
            "input_staging.stage_dir_prefix must be a single path component (no slashes); "
            f"got {prefix!r}"
        )
    if not shm_root.startswith("/"):
        raise ValueError(f"input_staging.shm_root must be an absolute path; got {shm_root!r}")
    # sbatch forwards the submission environment; a stray STAGE_DIR (e.g. from a local test)
    # must not be used in export NB_CLUSTERING_j="${STAGE_DIR}/...". bash -lc may also load
    # profiles that export STAGE_DIR; drop readonly if present, then unset and reassign.
    lines = [
        "declare +r STAGE_DIR 2>/dev/null || true",
        "unset STAGE_DIR NETBURST_STAGE_DIR 2>/dev/null || true",
        f'STAGE_DIR="{shm_root}/{prefix}${{SLURM_JOB_ID}}_${{SLURM_PROCID}}"',
        "export STAGE_DIR",
        # Duplicate for Python: some srun/conda invocations do not pass STAGE_DIR to the child.
        'export NETBURST_STAGE_DIR="${STAGE_DIR}"',
        'mkdir -p "${STAGE_DIR}"',
    ]
    if _parse_bool(cfg.get("debug_echo"), default=False):
        lines.append(
            'echo "[cluster_invariance] STAGE_DIR=${STAGE_DIR} python3=$(command -v python3)"'
        )
    return lines


def _staging_abs_path_bash_quoted(suffix: str, staging_cfg: dict) -> str:
    """
    Bash double-quoted file path under the staging dir (same dir as STAGE_DIR in _staging_shell_lines),
    without referencing the STAGE_DIR variable (avoids submit-time STAGE_DIR pollution).
    suffix: e.g. tsfresh.csv, clustering_0.csv
    """
    cfg = staging_cfg if isinstance(staging_cfg, dict) else {}
    shm_root = str(cfg.get("shm_root", "/dev/shm")).strip().rstrip("/") or "/dev/shm"
    prefix = str(cfg.get("stage_dir_prefix", "netburst_invariance_")).strip() or "netburst_invariance_"
    if "/" in prefix or prefix.startswith(".."):
        raise ValueError(
            "input_staging.stage_dir_prefix must be a single path component (no slashes); "
            f"got {prefix!r}"
        )
    if not shm_root.startswith("/"):
        raise ValueError(f"input_staging.shm_root must be an absolute path; got {shm_root!r}")
    return f'"{shm_root}/{prefix}${{SLURM_JOB_ID}}_${{SLURM_PROCID}}/{suffix}"'


def _heredoc_delimiter_unique(body: str, prefix: str = "EOF_CI_CHUNK") -> str:
    """Pick a delimiter that does not appear in body (single-quoted heredoc, line-at-start safe)."""
    for _ in range(32):
        d = f"{prefix}_{uuid.uuid4().hex}"
        if d not in body:
            return d
    raise ValueError("Could not find a unique heredoc delimiter; shorten or sanitize script body.")


def _bash_emit_chunk_script_file(*, chunk_index: int, script_body: str) -> list[str]:
    """
    Write one chunk script under $JOB_TMP/chunk_{i}.sh via heredoc.
    Keeps srun --multi-prog lines short (Slurm max line length).
    JOB_TMP must be on a filesystem visible to every node (see LOG_BASE in generated script).
    """
    delim = _heredoc_delimiter_unique(script_body)
    path = f'"$JOB_TMP/chunk_{chunk_index}.sh"'
    return [
        f"cat > {path} <<'{delim}'",
        script_body.rstrip("\n"),
        delim,
        f"chmod +x {path}",
    ]


def _multi_node_task_debug_lines(sweep_cfg: dict) -> list[str]:
    """
    Echo Slurm / host context after cd workdir, before staging or member pipelines.
    Enable with sweep JSON: \"multi_node_debug\": true
    """
    if not _parse_bool(sweep_cfg.get("multi_node_debug"), default=False):
        return []
    return [
        'echo "[cluster_invariance] task host=$(hostname) pwd=$(pwd -P) SLURM_JOB_ID=${SLURM_JOB_ID:-?} SLURM_PROCID=${SLURM_PROCID:-?} SLURM_NODEID=${SLURM_NODEID:-?}"',
    ]


def _member_plot_write_and_run_bash(m: dict, bash_quote: Any) -> str:
    """
    write_member_plot_config.py && plot_member_timeseries_from_parquet.py (fragment after mkdirs).
    """
    parent = m.get("cluster_members_parent_dir")
    if not parent:
        raise ValueError("member plot pipeline requires cluster_members_parent_dir on the member config.")
    parent_exp = str(Path(parent).expanduser())
    glob_pat = str(Path(parent_exp) / "cluster_*" / "members.json")
    cfg_out = m.get("plot_member_timeseries_config_out") or str(
        Path(parent_exp) / "plot_member_timeseries_config.json"
    )
    ipq = bash_quote(str(m["plot_member_timeseries_input_parquet"]))
    cfg_q = bash_quote(str(cfg_out))
    glob_q = bash_quote(glob_pat)
    kc = bash_quote(str(m.get("plot_member_timeseries_key_columns", "ip,source_file")))
    vc = bash_quote(str(m.get("plot_member_timeseries_value_column", "inbound")))
    nw = int(m.get("plot_member_timeseries_num_workers", 0))
    cmd = "python3 write_member_plot_config.py"
    cmd += f" --out {cfg_q}"
    cmd += f" --input-parquet {ipq}"
    cmd += f" --members-json-glob {glob_q}"
    cmd += f" --key-columns {kc}"
    cmd += f" --value-column {vc}"
    cmd += f" --num-workers {nw}"
    if not m.get("plot_member_timeseries_y_floor_zero", True):
        cmd += " --no-y-floor-zero"
    if m.get("plot_member_timeseries_share_y_max_within_k"):
        cmd += " --share-y-max-within-k"
    if m.get("plot_member_timeseries_y_fixed_max") is not None:
        cmd += f" --y-fixed-max {bash_quote(str(float(m['plot_member_timeseries_y_fixed_max'])))}"
    if m.get("plot_member_timeseries_fisher_overlay"):
        tsf = m.get("plot_member_timeseries_tsfresh_csv") or m.get("tsfresh_csv")
        if not tsf:
            raise ValueError(
                "plot_member_timeseries_fisher_overlay requires plot_member_timeseries_tsfresh_csv "
                "or member tsfresh_csv."
            )
        fn = int(m.get("plot_member_timeseries_fisher_top_n", 5))
        hm = bash_quote(str(m.get("plot_member_timeseries_heatmap_filename", "feature_order_fisher.csv")))
        cmd += " --fisher-overlay"
        cmd += f" --tsfresh-csv {bash_quote(str(tsf))}"
        cmd += f" --fisher-top-n {fn}"
        cmd += f" --heatmap-filename {hm}"
    else:
        use_dual = bool(
            m.get("global_filter_overlay_plots") or m.get("plot_member_dual_global_filter_overlays")
        )
        if use_dual:
            tsf = m.get("plot_member_timeseries_tsfresh_csv") or m.get("tsfresh_csv")
            if not tsf:
                raise ValueError(
                    "dual global-filter member plots require plot_member_timeseries_tsfresh_csv "
                    "or member tsfresh_csv."
                )
            cmd += " --dual-global-filter-overlays"
            cmd += f" --tsfresh-csv {bash_quote(str(tsf))}"
            cmd += f" --overlay-top-n {int(m.get('overlay_local_top_k', 10))}"
            olfs = m.get("plot_member_overlay_legend_fontsize")
            if olfs is not None:
                cmd += f" --overlay-legend-fontsize {bash_quote(str(float(olfs)))}"
    if _parse_bool(m.get("plot_member_compact_metric_tsfresh_legend"), default=False):
        cmd += " --compact-metric-tsfresh-legend"
    cmd += " && python3 plot_member_timeseries_from_parquet.py"
    cmd += f" --config {cfg_q}"
    return cmd


def _member_bash_pipeline(
    m: dict,
    *,
    bash_quote: Any,
    clustering_csv: str,
    tsfresh_csv: str,
    staging_clustering_ref: Optional[str] = None,
    staging_tsfresh_ref: Optional[str] = None,
    staging_parallel_literal_bash: bool = False,
) -> str:
    """
    Bash snippet for one member: mkdirs + cluster_analysis.py [+ cluster_variance_rank_to_json.py].
    Paths may point at staged copies under /dev/shm.

    If staging_*_ref is set (e.g. NB_CLUSTERING_1), that variable is assumed already exported in the
    parent shell before parallel subshells — use with _build_chunk_command_parallel + staging.

    If staging_parallel_literal_bash is True, clustering_csv/tsfresh_csv are already full bash
    double-quoted path expressions from _staging_abs_path_bash_quoted (no NB_* vars).
    Slurm forwards submit env; stray NB_CLUSTERING_j from the login shell must not affect argv.
    """
    if _parse_bool(m.get("member_timeseries_plots_only"), default=False):
        if not (m.get("run_member_timeseries_plots") and m.get("plot_member_timeseries_input_parquet")):
            raise ValueError(
                "member_timeseries_plots_only requires run_member_timeseries_plots and "
                "plot_member_timeseries_input_parquet."
            )
        parent = m.get("cluster_members_parent_dir")
        if not parent:
            raise ValueError("member_timeseries_plots_only requires cluster_members_parent_dir.")
        parent_q = bash_quote(str(Path(parent).expanduser()))
        inner = _member_plot_write_and_run_bash(m, bash_quote)
        return f"mkdir -p {parent_q} && {inner}"

    cmd = (
        f"mkdir -p {bash_quote(str(Path(m['out_csv']).expanduser().parent))}"
        f" && mkdir -p {bash_quote(str(Path(m['cluster_variance_out_csv']).expanduser().parent))}"
        f" && mkdir -p {bash_quote(str(Path(m['cluster_feature_rank_out_csv']).expanduser().parent))}"
    )
    if m.get("write_cluster_member_samples") and m.get("cluster_members_parent_dir"):
        cmd += f" && mkdir -p {bash_quote(str(Path(m['cluster_members_parent_dir']).expanduser()))}"
    # Assign NB_* in-shell then pass "$NB_*" to python. Avoids rare cases where "${STAGE_DIR}/..."
    # in argv expands wrong for later parallel members while STAGE_DIR is still set.
    staging_assigns: list[str] = []
    if not staging_parallel_literal_bash:
        if staging_clustering_ref is None and "${STAGE_DIR}" in clustering_csv:
            staging_assigns.append(f'NB_CLUSTERING="{clustering_csv}"')
        if staging_tsfresh_ref is None and "${STAGE_DIR}" in tsfresh_csv:
            staging_assigns.append(f'NB_TSFRESH="{tsfresh_csv}"')
    if staging_assigns:
        cmd += " && " + " && ".join(staging_assigns)
    use_staging_python = (
        staging_parallel_literal_bash
        or bool(staging_assigns)
        or bool(staging_clustering_ref or staging_tsfresh_ref)
    )
    # env ensures STAGE_DIR reaches Python even when the shell export is dropped before exec.
    if use_staging_python:
        cmd += ' && env STAGE_DIR="${STAGE_DIR}" NETBURST_STAGE_DIR="${STAGE_DIR}" python3 cluster_analysis.py'
    else:
        cmd += " && python3 cluster_analysis.py"
    if staging_parallel_literal_bash:
        cmd += f" --clustering {clustering_csv}"
    elif staging_clustering_ref is not None:
        cmd += f' --clustering "${{{staging_clustering_ref}}}"'
    elif "${STAGE_DIR}" in clustering_csv:
        cmd += ' --clustering "$NB_CLUSTERING"'
    else:
        cmd += f" --clustering {bash_quote(clustering_csv)}"
    if staging_parallel_literal_bash:
        cmd += f" --tsfresh {tsfresh_csv}"
    elif staging_tsfresh_ref is not None:
        cmd += f' --tsfresh "${{{staging_tsfresh_ref}}}"'
    elif "${STAGE_DIR}" in tsfresh_csv:
        cmd += ' --tsfresh "$NB_TSFRESH"'
    else:
        cmd += f" --tsfresh {bash_quote(tsfresh_csv)}"
    cmd += f" --out {bash_quote(m['out_csv'])}"
    cmd += f" --min-cluster-size {int(m['min_cluster_size'])}"
    cmd += f" --global-fisher-min {float(m.get('global_fisher_min', 0.1))}"
    if use_staging_python:
        # Live job path from bash; avoids relying on SLURM_* or stale env inside Python.
        cmd += ' --netburst-stage-dir "${STAGE_DIR}"'
    cmd += f" --cluster-variance-out {bash_quote(m['cluster_variance_out_csv'])}"
    cmd += f" --cluster-feature-rank-out {bash_quote(m['cluster_feature_rank_out_csv'])}"
    cmd += f" --id-ip {bash_quote(m['id_ip'])}"
    cmd += f" --id-source-file {bash_quote(m['id_source_file'])}"
    if m.get("ip_suffix_filter"):
        cmd += f" --ip-suffix-filter {bash_quote(str(m['ip_suffix_filter']).strip())}"
    if m.get("write_cluster_member_samples") and m.get("cluster_members_parent_dir"):
        cmd += " --write-cluster-member-samples"
        cmd += (
            f" --cluster-members-parent-dir {bash_quote(str(Path(m['cluster_members_parent_dir']).expanduser()))}"
        )
        cmd += f" --member-sample-n {int(m['member_sample_n'])}"
        cmd += f" --member-sample-seed {int(m['member_sample_seed'])}"
    if m.get("write_per_cluster_cohens_d_cka"):
        th_dir = str(Path(m["out_csv"]).expanduser().parent)
        cmd += " --write-per-cluster-cohens-d-cka"
        cmd += f" --per-cluster-thresholds-dir {bash_quote(str(Path(th_dir).expanduser()))}"
        cmd += f" --repr-prefix {bash_quote(str(m.get('repr_prefix', 'repr_dim')))}"
    if m.get("global_filter_overlay_plots"):
        if not m.get("cluster_members_parent_dir"):
            raise ValueError("global_filter_overlay_plots requires cluster_members_parent_dir")
        gfm = str(m.get("global_filter_mode", "fisher_min")).strip().lower()
        if gfm != "fisher_min":
            raise ValueError("global_filter_mode must be 'fisher_min'")
        parent_q = bash_quote(str(Path(m["cluster_members_parent_dir"]).expanduser()))
        cmd += " && python3 global_filtered_overlay_prep.py"
        if staging_parallel_literal_bash:
            cmd += f" --clustering-csv {clustering_csv}"
        elif staging_clustering_ref is not None:
            cmd += f' --clustering-csv "${{{staging_clustering_ref}}}"'
        elif "${STAGE_DIR}" in clustering_csv:
            cmd += ' --clustering-csv "$NB_CLUSTERING"'
        else:
            cmd += f" --clustering-csv {bash_quote(clustering_csv)}"
        if staging_parallel_literal_bash:
            cmd += f" --tsfresh-csv {tsfresh_csv}"
        elif staging_tsfresh_ref is not None:
            cmd += f' --tsfresh-csv "${{{staging_tsfresh_ref}}}"'
        elif "${STAGE_DIR}" in tsfresh_csv:
            cmd += ' --tsfresh-csv "$NB_TSFRESH"'
        else:
            cmd += f" --tsfresh-csv {bash_quote(tsfresh_csv)}"
        cmd += f" --parent-out {parent_q}"
        if m.get("global_filtered_features_csv"):
            cmd += (
                f" --global-filtered-features-csv "
                f"{bash_quote(str(m['global_filtered_features_csv']))}"
            )
        cmd += f" --global-filter-mode {bash_quote(gfm)}"
        cmd += f" --global-fisher-min {float(m.get('global_fisher_min', 0.1))}"
        cmd += f" --local-overlay-top-k {int(m.get('overlay_local_top_k', 10))}"
        if m.get("global_cka_all_csv"):
            cmd += f" --global-cka-all-csv {bash_quote(str(m['global_cka_all_csv']))}"
        if m.get("global_cka_all_feature_col"):
            cmd += f" --global-cka-all-feature-col {bash_quote(str(m['global_cka_all_feature_col']))}"
        if m.get("global_cka_all_value_col"):
            cmd += f" --global-cka-all-value-col {bash_quote(str(m['global_cka_all_value_col']))}"
        cmd += f" --repr-prefix {bash_quote(str(m.get('repr_prefix', 'repr_dim')))}"
        if m.get("global_all_features_csv"):
            cmd += f" --all-features-csv {bash_quote(str(m['global_all_features_csv']))}"
        if m.get("global_metrics_csv"):
            cmd += f" --global-metrics-csv {bash_quote(str(m['global_metrics_csv']))}"
        if m.get("global_fisher_cdf_png"):
            cmd += f" --global-fisher-cdf-png {bash_quote(str(m['global_fisher_cdf_png']))}"
        if m.get("global_cka_cdf_png"):
            cmd += f" --global-cka-cdf-png {bash_quote(str(m['global_cka_cdf_png']))}"
        if m.get("cluster_importance_csv"):
            cmd += f" --cluster-importance-csv {bash_quote(str(m['cluster_importance_csv']))}"
        if m.get("cluster_importance_cdf_png"):
            cmd += f" --cluster-importance-cdf-png {bash_quote(str(m['cluster_importance_cdf_png']))}"
        if m.get("local_importance_all_filename"):
            cmd += f" --local-importance-all-filename {bash_quote(str(m['local_importance_all_filename']))}"
        if m.get("local_importance_top_filename"):
            cmd += f" --local-importance-top-filename {bash_quote(str(m['local_importance_top_filename']))}"
        cmd += f" --label-col {bash_quote(str(m.get('label_col', 'label')))}"
        cmd += f" --id-ip {bash_quote(str(m['id_ip']))}"
        cmd += f" --id-source-file {bash_quote(str(m['id_source_file']))}"
        cmd += f" --cka-feature-col {bash_quote(str(m.get('global_cka_feature_col', 'tsfresh_feature')))}"
        if m.get("cohens_d_tanh_d0") is not None:
            cmd += f" --cohens-d-tanh-d0 {bash_quote(str(m['cohens_d_tanh_d0']))}"
        if m.get("cohens_d_max_norm"):
            cmd += " --cohens-d-max-norm"
        if m.get("ip_suffix_filter"):
            cmd += f" --ip-suffix-filter {bash_quote(str(m['ip_suffix_filter']).strip())}"
        if m.get("skip_global_cdf_plots") is False:
            cmd += " --enable-cdf-plots"
    if m.get("run_member_timeseries_plots") and m.get("plot_member_timeseries_input_parquet"):
        cmd += " && " + _member_plot_write_and_run_bash(m, bash_quote)
    return cmd


def _build_single_member_local_cmds(*, member_cfg: dict) -> tuple[str, Optional[str]]:
    if _parse_bool(member_cfg.get("member_timeseries_plots_only"), default=False):
        mod = _get_cluster_invariance_driver_module()
        _bash_quote = mod._bash_quote  # type: ignore[attr-defined]
        inner = _member_plot_write_and_run_bash(member_cfg, _bash_quote)
        return inner, None
    # Reuse the already-implemented logic from the single-job driver.
    mod = _get_cluster_invariance_driver_module()
    return mod._build_local_cmds(  # type: ignore[attr-defined]
        clustering_csv=member_cfg["clustering_csv"],
        tsfresh_csv=member_cfg["tsfresh_csv"],
        out_csv=member_cfg["out_csv"],
        cluster_variance_out_csv=member_cfg["cluster_variance_out_csv"],
        cluster_feature_rank_out_csv=member_cfg["cluster_feature_rank_out_csv"],
        min_cluster_size=int(member_cfg["min_cluster_size"]),
        id_ip=member_cfg["id_ip"],
        id_source_file=member_cfg["id_source_file"],
        write_cluster_member_samples=bool(member_cfg.get("write_cluster_member_samples", False)),
        cluster_members_parent_dir=member_cfg.get("cluster_members_parent_dir"),
        member_sample_n=int(member_cfg.get("member_sample_n", 5)),
        member_sample_seed=int(member_cfg.get("member_sample_seed", 0)),
        write_per_cluster_cohens_d_cka=bool(member_cfg.get("write_per_cluster_cohens_d_cka", False)),
        repr_prefix=str(member_cfg.get("repr_prefix", "repr_dim")),
        ip_suffix_filter=member_cfg.get("ip_suffix_filter"),
        run_member_timeseries_plots=bool(member_cfg.get("run_member_timeseries_plots")),
        plot_member_timeseries_input_parquet=member_cfg.get("plot_member_timeseries_input_parquet"),
        plot_member_timeseries_config_out=member_cfg.get("plot_member_timeseries_config_out"),
        plot_member_timeseries_key_columns=str(
            member_cfg.get("plot_member_timeseries_key_columns", "ip,source_file")
        ),
        plot_member_timeseries_value_column=str(member_cfg.get("plot_member_timeseries_value_column", "inbound")),
        plot_member_timeseries_num_workers=int(member_cfg.get("plot_member_timeseries_num_workers", 0)),
        plot_member_timeseries_y_floor_zero=bool(member_cfg.get("plot_member_timeseries_y_floor_zero", True)),
        plot_member_timeseries_y_fixed_max=member_cfg.get("plot_member_timeseries_y_fixed_max"),
        plot_member_timeseries_share_y_max_within_k=bool(
            member_cfg.get("plot_member_timeseries_share_y_max_within_k", False)
        ),
        plot_member_timeseries_fisher_overlay=bool(member_cfg.get("plot_member_timeseries_fisher_overlay", False)),
        plot_member_timeseries_tsfresh_csv=member_cfg.get("plot_member_timeseries_tsfresh_csv"),
        plot_member_timeseries_fisher_top_n=int(member_cfg.get("plot_member_timeseries_fisher_top_n", 5)),
        plot_member_timeseries_heatmap_filename=str(
            member_cfg.get("plot_member_timeseries_heatmap_filename", "feature_order_fisher.csv")
        ),
        global_filter_overlay_plots=bool(member_cfg.get("global_filter_overlay_plots", False)),
        global_filter_mode=str(member_cfg.get("global_filter_mode", "fisher_min")),
        global_fisher_min=float(member_cfg.get("global_fisher_min", 0.1)),
        overlay_local_top_k=int(member_cfg.get("overlay_local_top_k", 10)),
        label_col=str(member_cfg.get("label_col", "label")),
        global_cka_feature_col=str(member_cfg.get("global_cka_feature_col", "tsfresh_feature")),
        plot_member_overlay_legend_fontsize=member_cfg.get("plot_member_overlay_legend_fontsize"),
    )


def _build_single_member_slurm_text(*, member_cfg: dict, sbatch_cfg: dict, modules: list[str]) -> str:
    mod = _get_cluster_invariance_driver_module()
    _build_slurm_script = mod._build_slurm_script  # type: ignore[attr-defined]

    out_dir = _derive_out_dir(member_cfg["out_csv"])
    job_name = member_cfg["job_name"]

    log_out = str(Path(out_dir) / "logs" / f"{job_name}_%j.out")
    log_err = str(Path(out_dir) / "logs" / f"{job_name}_%j.err")

    environment = str(member_cfg.get("environment", DEFAULT_ENVIRONMENT))
    # Ensure we pass int where needed.
    return _build_slurm_script(
        job_name=job_name,
        clustering_csv=member_cfg["clustering_csv"],
        tsfresh_csv=member_cfg["tsfresh_csv"],
        out_csv=member_cfg["out_csv"],
        cluster_variance_out_csv=member_cfg["cluster_variance_out_csv"],
        cluster_feature_rank_out_csv=member_cfg["cluster_feature_rank_out_csv"],
        min_cluster_size=int(member_cfg["min_cluster_size"]),
        id_ip=member_cfg["id_ip"],
        id_source_file=member_cfg["id_source_file"],
        write_cluster_member_samples=bool(member_cfg.get("write_cluster_member_samples", False)),
        cluster_members_parent_dir=member_cfg.get("cluster_members_parent_dir"),
        member_sample_n=int(member_cfg.get("member_sample_n", 5)),
        member_sample_seed=int(member_cfg.get("member_sample_seed", 0)),
        write_per_cluster_cohens_d_cka=bool(member_cfg.get("write_per_cluster_cohens_d_cka", False)),
        repr_prefix=str(member_cfg.get("repr_prefix", "repr_dim")),
        ip_suffix_filter=member_cfg.get("ip_suffix_filter"),
        run_member_timeseries_plots=bool(member_cfg.get("run_member_timeseries_plots")),
        plot_member_timeseries_input_parquet=member_cfg.get("plot_member_timeseries_input_parquet"),
        plot_member_timeseries_config_out=member_cfg.get("plot_member_timeseries_config_out"),
        plot_member_timeseries_key_columns=str(
            member_cfg.get("plot_member_timeseries_key_columns", "ip,source_file")
        ),
        plot_member_timeseries_value_column=str(member_cfg.get("plot_member_timeseries_value_column", "inbound")),
        plot_member_timeseries_num_workers=int(member_cfg.get("plot_member_timeseries_num_workers", 0)),
        plot_member_timeseries_y_floor_zero=bool(member_cfg.get("plot_member_timeseries_y_floor_zero", True)),
        plot_member_timeseries_y_fixed_max=member_cfg.get("plot_member_timeseries_y_fixed_max"),
        plot_member_timeseries_share_y_max_within_k=bool(
            member_cfg.get("plot_member_timeseries_share_y_max_within_k", False)
        ),
        plot_member_timeseries_fisher_overlay=bool(member_cfg.get("plot_member_timeseries_fisher_overlay", False)),
        plot_member_timeseries_tsfresh_csv=member_cfg.get("plot_member_timeseries_tsfresh_csv"),
        plot_member_timeseries_fisher_top_n=int(member_cfg.get("plot_member_timeseries_fisher_top_n", 5)),
        plot_member_timeseries_heatmap_filename=str(
            member_cfg.get("plot_member_timeseries_heatmap_filename", "feature_order_fisher.csv")
        ),
        global_filter_overlay_plots=bool(member_cfg.get("global_filter_overlay_plots", False)),
        global_filter_mode=str(member_cfg.get("global_filter_mode", "fisher_min")),
        global_fisher_min=float(member_cfg.get("global_fisher_min", 0.1)),
        overlay_local_top_k=int(member_cfg.get("overlay_local_top_k", 10)),
        label_col=str(member_cfg.get("label_col", "label")),
        global_cka_feature_col=str(member_cfg.get("global_cka_feature_col", "tsfresh_feature")),
        plot_member_overlay_legend_fontsize=member_cfg.get("plot_member_overlay_legend_fontsize"),
        time_limit=str(sbatch_cfg.get("time", "1:00:00")),
        queue=str(sbatch_cfg.get("queue", "premium")),
        account=str(sbatch_cfg.get("account", "<SLURM_ACCOUNT>")),
        cpus_per_task=int(sbatch_cfg.get("cpus_per_task", 8)),
        log_out=log_out,
        log_err=log_err,
        modules=modules,
        environment=environment,
    )


def _write_preview_slurm(slurm_text: str, tmp_dir: str, job_name: str) -> str:
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    slurm_path = str(Path(tmp_dir) / f"{job_name}_preview.slurm")
    Path(slurm_path).write_text(slurm_text, encoding="utf-8")
    return slurm_path


def _build_parallel_slurm_on_one_node(
    *,
    member_cfgs: list[dict],
    job_name: str,
    header_comment_lines: Optional[list[str]],
    sbatch_cfg: dict,
    modules: list[str],
    environment: str,
    workdir: str,
    sbatch_output: Optional[str] = None,
    sbatch_error: Optional[str] = None,
    clustering_prefix: Optional[str] = None,
    invariance_enabled: bool = True,
) -> str:
    """
    One Slurm allocation (--nodes=1): run every member pipeline in parallel (& ... & wait).
    """
    mod = _get_cluster_invariance_driver_module()
    _bash_quote = mod._bash_quote  # type: ignore[attr-defined]

    modules_lines: list[str] = []
    if environment == "nersc":
        for m in modules:
            if m == "conda":
                modules_lines.append("module load conda")
            else:
                modules_lines.append(f"module load {m}")

    queue = str(sbatch_cfg.get("queue", "premium"))
    account = str(sbatch_cfg.get("account", "<SLURM_ACCOUNT>"))
    time_limit = str(sbatch_cfg.get("time", "2:00:00"))
    cpus_per_task = int(sbatch_cfg.get("cpus_per_task", 32))

    member_lines: list[str] = []
    if invariance_enabled:
        for m in member_cfgs:
            member_job_name = m["job_name"]
            out_dir = _derive_out_dir(m["out_csv"])
            redirect_path = str(Path(out_dir) / f"cluster_metrics_sweep_{member_job_name}.log")

            inner = _member_bash_pipeline(
                m,
                bash_quote=_bash_quote,
                clustering_csv=m["clustering_csv"],
                tsfresh_csv=m["tsfresh_csv"],
            )
            # mkdir before redirect: bash opens the log file before the inner pipeline runs.
            member_lines.append(
                f"mkdir -p {_bash_quote(out_dir)} && ({inner}) &> {_bash_quote(redirect_path)} &"
            )
    else:
        member_lines.append(
            'echo "[invariance_phase] disabled — clustering-only sweep (no member pipelines)."'
        )

    out_err_lines: list[str] = []
    if sbatch_output:
        out_err_lines.extend(
            [
                f"#SBATCH --output={sbatch_output}",
                f"#SBATCH --error={sbatch_error or sbatch_output}",
                "",
            ]
        )

    header = (header_comment_lines or []) + (
        [
            "# Parallel on this node: one background process per member (typically one per k).",
        ]
        if invariance_enabled
        else [
            "# invariance_phase disabled: no parallel member pipelines on this node.",
        ]
    )

    mid_lines: list[str] = []
    if clustering_prefix:
        mid_lines.extend(
            [
                "# clustering_phase (sequential K) then invariance members below",
                clustering_prefix,
                "",
            ]
        )

    return "\n".join(
        [
            "#!/bin/bash",
            *header,
            f"#SBATCH -q {queue}",
            "#SBATCH -C cpu",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --account={account}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            *out_err_lines,
            "set -euo pipefail",
            "",
            *modules_lines,
            "",
            f"cd {_bash_quote(workdir)}",
            "",
            *mid_lines,
            "mkdir -p ./logs || true",
            "",
            *member_lines,
            "",
            "wait",
            f"echo \"All member pipelines finished for job: {job_name}\"",
            "",
        ]
    )


def _build_single_job_parallel_slurm(
    *,
    sweep_cfg: dict,
    member_cfgs: list[dict],
    sbatch_cfg: dict,
    modules: list[str],
    environment: str,
    invariance_enabled: bool = True,
) -> str:
    workdir = str(sweep_cfg.get("workdir", "<external-netburst-root>/src/analysis"))
    sweep_job_name = str(sweep_cfg.get("sweep_job_name", "cluster_invariance_sweep"))
    return _build_parallel_slurm_on_one_node(
        member_cfgs=member_cfgs,
        job_name=sweep_job_name,
        header_comment_lines=None,
        sbatch_cfg=sbatch_cfg,
        modules=modules,
        environment=environment,
        workdir=workdir,
        sbatch_output=None,
        sbatch_error=None,
        invariance_enabled=invariance_enabled,
    )


def _chunk_members(member_cfgs: list[dict], chunk_size: int) -> list[list[dict]]:
    if chunk_size < 1:
        raise ValueError("members_per_node (chunk size) must be >= 1.")
    return [member_cfgs[i : i + chunk_size] for i in range(0, len(member_cfgs), chunk_size)]


def _build_chunk_command(
    chunk: list[dict],
    *,
    bash_quote: Any,
    staging_enabled: bool,
    staging_cfg: Optional[dict],
    tsfresh_src: str,
    invariance_enabled: bool = True,
) -> str:
    """Sequential pipelines for all members in one chunk (one Slurm task / node by default)."""
    if not invariance_enabled:
        return "true"
    if staging_enabled:
        lines = [
            *_staging_shell_lines(staging_cfg or {}),
            f"cp -f {bash_quote(tsfresh_src)} \"${{STAGE_DIR}}/tsfresh.csv\"",
        ]
        for j, m in enumerate(chunk):
            lines.append(
                f"cp -f {bash_quote(m['clustering_csv'])} \"${{STAGE_DIR}}/clustering_{j}.csv\""
            )
        ts_use = "${STAGE_DIR}/tsfresh.csv"
        pipes: list[str] = []
        for j, m in enumerate(chunk):
            pipes.append(
                _member_bash_pipeline(
                    m,
                    bash_quote=bash_quote,
                    clustering_csv=f"${{STAGE_DIR}}/clustering_{j}.csv",
                    tsfresh_csv=ts_use,
                )
            )
        return " && ".join(lines) + " && " + " && ".join(pipes) + ' && rm -rf "${STAGE_DIR}"'

    pipes = [
        _member_bash_pipeline(
            m,
            bash_quote=bash_quote,
            clustering_csv=m["clustering_csv"],
            tsfresh_csv=m["tsfresh_csv"],
        )
        for m in chunk
    ]
    return " && ".join(pipes)


def _build_chunk_command_parallel(
    chunk: list[dict],
    *,
    bash_quote: Any,
    staging_enabled: bool,
    staging_cfg: Optional[dict],
    tsfresh_src: str,
    invariance_enabled: bool = True,
) -> str:
    """
    Parallel pipelines on one node: ( ... ) &> log & ... wait.

    With input_staging, stage clustering + TSFresh under STAGE_DIR before parallel members.
    """
    if not invariance_enabled:
        return "true"

    def _one_bg(m: dict, clustering_csv: str, tsfresh: str) -> str:
        out_dir = _derive_out_dir(m["out_csv"])
        redirect_path = str(Path(out_dir) / f"cluster_metrics_sweep_{m['job_name']}.log")
        inner = _member_bash_pipeline(
            m,
            bash_quote=bash_quote,
            clustering_csv=clustering_csv,
            tsfresh_csv=tsfresh,
        )
        # mkdir before redirect: bash opens the log file before the inner pipeline runs.
        return f"mkdir -p {bash_quote(out_dir)} && ( {inner} ) &> {bash_quote(redirect_path)} &"

    if staging_enabled:
        nb_unset = " ".join(f"NB_CLUSTERING_{j}" for j in range(len(chunk)))
        lines = [
            f"unset NB_TSFRESH {nb_unset} 2>/dev/null || true",
            *_staging_shell_lines(staging_cfg or {}),
            f"cp -f {bash_quote(tsfresh_src)} \"${{STAGE_DIR}}/tsfresh.csv\"",
        ]
        for j, m in enumerate(chunk):
            lines.append(
                f"cp -f {bash_quote(m['clustering_csv'])} \"${{STAGE_DIR}}/clustering_{j}.csv\""
            )
        prefix = " && ".join(lines)
        bg = []
        for j, m in enumerate(chunk):
            inner = _member_bash_pipeline(
                m,
                bash_quote=bash_quote,
                clustering_csv=_staging_abs_path_bash_quoted(f"clustering_{j}.csv", staging_cfg or {}),
                tsfresh_csv=_staging_abs_path_bash_quoted("tsfresh.csv", staging_cfg or {}),
                staging_parallel_literal_bash=True,
            )
            out_dir = _derive_out_dir(m["out_csv"])
            redirect_path = str(Path(out_dir) / f"cluster_metrics_sweep_{m['job_name']}.log")
            bg.append(f"mkdir -p {bash_quote(out_dir)} && ( {inner} ) &> {bash_quote(redirect_path)} &")
        # Brace group: prefix must finish (all cp) before any background member; bare
        # `prefix && job0 & job1 & wait` can parse so a later & runs before prefix completes.
        return prefix + " && { " + " ".join(bg) + " wait; }" + ' && rm -rf "${STAGE_DIR}"'

    bg = [_one_bg(m, m["clustering_csv"], m["tsfresh_csv"]) for m in chunk]
    return " ".join(bg) + " wait"


def _build_multi_node_slurm(
    *,
    sweep_cfg: dict,
    member_cfgs: list[dict],
    sbatch_cfg: dict,
    modules: list[str],
    environment: str,
    tsfresh_csv: str,
    invariance_enabled: bool = True,
) -> str:
    """
    One Slurm job: --nodes == --ntasks == number of chunks; one task per node (--ntasks-per-node=1).
    Each task runs one chunk (members_per_node members, sequentially on that node).
    """
    mod = _get_cluster_invariance_driver_module()
    bash_quote = mod._bash_quote  # type: ignore[attr-defined]

    workdir = str(sweep_cfg.get("workdir", "<external-netburst-root>/src/analysis"))

    members_per_node = int(sweep_cfg.get("members_per_node", 1))
    chunks = _chunk_members(member_cfgs, members_per_node)
    num_chunks = len(chunks)
    if num_chunks == 0:
        raise ValueError("multi_node_single_job: no sweep members to run.")

    nodes_opt = sweep_cfg.get("nodes")
    if nodes_opt is not None and int(nodes_opt) != num_chunks:
        raise ValueError(
            f"nodes={nodes_opt} must equal the computed chunk count {num_chunks} "
            f"(len(members)={len(member_cfgs)}, members_per_node={members_per_node}). "
            "Omit 'nodes' to auto-set."
        )

    staging = sweep_cfg.get("input_staging") or {}
    if not isinstance(staging, dict):
        staging = {}
    staging_enabled = _parse_bool(staging.get("enabled"), default=False)

    queue = str(sbatch_cfg.get("queue", "premium"))
    account = str(sbatch_cfg.get("account", "<SLURM_ACCOUNT>"))
    time_limit = str(sbatch_cfg.get("time", "4:00:00"))
    cpus_per_task = int(sbatch_cfg.get("cpus_per_task", 64))

    sweep_job_name = str(sweep_cfg.get("sweep_job_name", "cluster_invariance_sweep"))

    modules_lines: list[str] = []
    if environment == "nersc":
        for m in modules:
            if m == "conda":
                modules_lines.append("module load conda")
            else:
                modules_lines.append(f"module load {m}")

    log_dir = sweep_cfg.get("multi_node_log_dir")
    if log_dir:
        log_base = Path(log_dir).expanduser()
    else:
        log_base = Path(_derive_out_dir(member_cfgs[0]["out_csv"])) / "logs"
    log_out = str(log_base / f"{sweep_job_name}_%j_%t.out")
    log_err = str(log_base / f"{sweep_job_name}_%j_%t.err")

    # Slurm --multi-prog: one short line per task id (long bodies go to $JOB_TMP/chunk_N.sh).
    comment_lines: list[str] = [
        "# Task groups (same Slurm job, one chunk per node by default):",
    ]
    chunk_write_lines: list[str] = []
    for i, chunk in enumerate(chunks):
        desc = ", ".join(f"k={m['k']} run={m['run']} seed={m['seed']} ({m['job_name']})" for m in chunk)
        comment_lines.append(f"#   [{i}] {len(chunk)} member(s): {desc}")
        chunk_cmd = _build_chunk_command(
            chunk,
            bash_quote=bash_quote,
            staging_enabled=staging_enabled,
            staging_cfg=staging,
            tsfresh_src=str(Path(tsfresh_csv).expanduser()),
            invariance_enabled=invariance_enabled,
        )
        out0 = _derive_out_dir(chunk[0]["out_csv"])
        if log_dir:
            log_chunk = str(log_base / f"sweep_multinode_chunk_{i}.log")
        else:
            log_chunk = str(Path(out0) / f"sweep_multinode_chunk_{i}.log")
        run_parts = [
            "set -euo pipefail",
            *modules_lines,
            f"cd {bash_quote(workdir)}",
            *_multi_node_task_debug_lines(sweep_cfg),
            chunk_cmd,
        ]
        inner = " && ".join(run_parts)
        log_parent = bash_quote(str(Path(log_chunk).parent))
        full = f"mkdir -p {bash_quote(out0)} && mkdir -p {log_parent} && ( {inner} ) &> {bash_quote(log_chunk)}"
        script_body = "#!/bin/bash\n" + full
        chunk_write_lines.extend(_bash_emit_chunk_script_file(chunk_index=i, script_body=script_body))
        chunk_write_lines.append("")

    mp_lines_str = "\n".join(f"{i} bash $JOB_TMP/chunk_{i}.sh" for i in range(num_chunks))

    dbg_launch = []
    if _parse_bool(sweep_cfg.get("multi_node_debug"), default=False):
        dbg_launch = [
            'echo "[cluster_invariance] launcher JOBID=${SLURM_JOB_ID:-?} NNODES=${SLURM_JOB_NUM_NODES:-?} pwd=$(pwd -P)"',
            "",
        ]

    header_lines = [
        "#!/bin/bash",
        "# Cluster invariance sweep: one allocation; srun --multi-prog maps chunks to nodes.",
        "# Chunk scripts under $JOB_TMP on shared LOG_BASE (not node-local /tmp; all nodes must see files).",
        f"# members_per_node={members_per_node}  nodes/ntasks={num_chunks}",
        f"# input_staging_to_shm={'on' if staging_enabled else 'off'}",
        *comment_lines,
        f"#SBATCH -q {queue}",
        "#SBATCH -C cpu",
        f"#SBATCH --nodes={num_chunks}",
        f"#SBATCH --ntasks={num_chunks}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --account={account}",
        f"#SBATCH --job-name={sweep_job_name}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --output={log_out}",
        f"#SBATCH --error={log_err}",
        "",
        "set -euo pipefail",
        "",
        *modules_lines,
        "",
        f"cd {bash_quote(workdir)}",
        "",
        f"mkdir -p {bash_quote(str(log_base))}",
        "",
        *dbg_launch,
        f"LOG_BASE={bash_quote(str(log_base))}",
        'JOB_TMP="${LOG_BASE}/multiprog_job_${SLURM_JOB_ID}"',
        'mkdir -p "$JOB_TMP"',
        'MULTI="${LOG_BASE}/sweep_multiprog_${SLURM_JOB_ID}.conf"',
        "",
        *chunk_write_lines,
        'cat > "$MULTI" <<MP_EOF',
        mp_lines_str,
        "MP_EOF",
        "",
        'srun --multi-prog "$MULTI"',
        'rm -f "$MULTI"',
        'rm -rf "$JOB_TMP"',
        "",
        f'echo "All tasks finished for sweep: {sweep_job_name}"',
        "",
    ]

    return "\n".join(header_lines)


def _build_multi_node_slurm_per_run_parallel_k(
    *,
    sweep_cfg: dict,
    member_cfgs: list[dict],
    sbatch_cfg: dict,
    modules: list[str],
    environment: str,
    tsfresh_csv: str,
    invariance_enabled: bool = True,
) -> str:
    """
    One Slurm job spanning n nodes: each node = one (run, seed) group; all k for that run
    run in parallel on that node (same as one_node_per_run_parallel_k, but one sbatch).
    """
    mod = _get_cluster_invariance_driver_module()
    bash_quote = mod._bash_quote  # type: ignore[attr-defined]

    workdir = str(sweep_cfg.get("workdir", "<external-netburst-root>/src/analysis"))

    chunks = _group_members_by_run_seed(member_cfgs)
    num_chunks = len(chunks)
    if num_chunks == 0:
        raise ValueError("multi_node_per_run_parallel_k: no sweep members to run.")

    nodes_opt = sweep_cfg.get("nodes")
    if nodes_opt is not None and int(nodes_opt) != num_chunks:
        raise ValueError(
            f"nodes={nodes_opt} must equal the number of (run, seed) groups {num_chunks}. "
            "Omit 'nodes' to auto-set."
        )

    staging = sweep_cfg.get("input_staging") or {}
    if not isinstance(staging, dict):
        staging = {}
    staging_enabled = _parse_bool(staging.get("enabled"), default=False)

    queue = str(sbatch_cfg.get("queue", "premium"))
    account = str(sbatch_cfg.get("account", "<SLURM_ACCOUNT>"))
    time_limit = str(sbatch_cfg.get("time", "4:00:00"))
    cpus_per_task = int(sbatch_cfg.get("cpus_per_task", 64))

    sweep_job_name = str(sweep_cfg.get("sweep_job_name", "cluster_invariance_sweep"))

    modules_lines: list[str] = []
    if environment == "nersc":
        for m in modules:
            if m == "conda":
                modules_lines.append("module load conda")
            else:
                modules_lines.append(f"module load {m}")

    log_dir = sweep_cfg.get("multi_node_log_dir")
    if log_dir:
        log_base = Path(log_dir).expanduser()
    else:
        log_base = Path(_derive_out_dir(member_cfgs[0]["out_csv"])) / "logs"
    log_out = str(log_base / f"{sweep_job_name}_%j_%t.out")
    log_err = str(log_base / f"{sweep_job_name}_%j_%t.err")

    comment_lines: list[str] = [
        "# One Slurm allocation: one node per (run, seed); on each node all k run in parallel.",
    ]
    if _parse_bool((sweep_cfg.get("clustering_phase") or {}).get("enabled"), default=False):
        comment_lines.insert(
            0,
            "# clustering_phase: sequential clustering_analysis.py (per K) before parallel invariance on each node.",
        )
    chunk_write_lines: list[str] = []
    for i, chunk in enumerate(chunks):
        run0, seed0 = chunk[0]["run"], chunk[0]["seed"]
        ks = [m["k"] for m in chunk]
        comment_lines.append(
            f"#   [{i}] run={run0} seed={seed0}  |  parallel k processes ({len(chunk)}): {ks}"
        )
        chunk_cmd = _build_chunk_command_parallel(
            chunk,
            bash_quote=bash_quote,
            staging_enabled=staging_enabled,
            staging_cfg=staging,
            tsfresh_src=str(Path(tsfresh_csv).expanduser()),
            invariance_enabled=invariance_enabled,
        )
        cp_bash = _clustering_phase_bash_string(sweep_cfg, chunk, bash_quote)
        out0 = _derive_out_dir(chunk[0]["out_csv"])
        if log_dir:
            log_chunk = str(log_base / f"sweep_multinode_per_run_chunk_{i}.log")
        else:
            log_chunk = str(Path(out0) / f"sweep_multinode_per_run_chunk_{i}.log")
        run_parts = [
            "set -euo pipefail",
            *modules_lines,
            f"cd {bash_quote(workdir)}",
            *_multi_node_task_debug_lines(sweep_cfg),
        ]
        if cp_bash:
            run_parts.append(cp_bash)
        run_parts.append(chunk_cmd)
        inner = " && ".join(run_parts)
        # Ensure output dirs and chunk log parent exist before shell redirection.
        log_parent = bash_quote(str(Path(log_chunk).parent))
        full = f"mkdir -p {bash_quote(out0)} && mkdir -p {log_parent} && ( {inner} ) &> {bash_quote(log_chunk)}"
        script_body = "#!/bin/bash\n" + full
        chunk_write_lines.extend(_bash_emit_chunk_script_file(chunk_index=i, script_body=script_body))
        chunk_write_lines.append("")

    mp_lines_str = "\n".join(f"{i} bash $JOB_TMP/chunk_{i}.sh" for i in range(num_chunks))

    dbg_launch = []
    if _parse_bool(sweep_cfg.get("multi_node_debug"), default=False):
        dbg_launch = [
            'echo "[cluster_invariance] launcher JOBID=${SLURM_JOB_ID:-?} NNODES=${SLURM_JOB_NUM_NODES:-?} pwd=$(pwd -P)"',
            "",
        ]

    header_lines = [
        "#!/bin/bash",
        "# Single job, n nodes: each task runs one (run, seed) group's k-list in parallel on that node.",
        "# Chunk scripts under $JOB_TMP on shared LOG_BASE (not node-local /tmp; all nodes must see files).",
        f"# nodes/ntasks={num_chunks}  (one per distinct (run, seed) from the sweep grid)",
        f"# input_staging_to_shm={'on' if staging_enabled else 'off'}",
        *comment_lines,
        f"#SBATCH -q {queue}",
        "#SBATCH -C cpu",
        f"#SBATCH --nodes={num_chunks}",
        f"#SBATCH --ntasks={num_chunks}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --account={account}",
        f"#SBATCH --job-name={sweep_job_name}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --output={log_out}",
        f"#SBATCH --error={log_err}",
        "",
        "set -euo pipefail",
        "",
        *modules_lines,
        "",
        f"cd {bash_quote(workdir)}",
        "",
        f"mkdir -p {bash_quote(str(log_base))}",
        "",
        *dbg_launch,
        f"LOG_BASE={bash_quote(str(log_base))}",
        'JOB_TMP="${LOG_BASE}/multiprog_job_${SLURM_JOB_ID}"',
        'mkdir -p "$JOB_TMP"',
        'MULTI="${LOG_BASE}/sweep_multiprog_${SLURM_JOB_ID}.conf"',
        "",
        *chunk_write_lines,
        'cat > "$MULTI" <<MP_EOF',
        mp_lines_str,
        "MP_EOF",
        "",
        'srun --multi-prog "$MULTI"',
        'rm -f "$MULTI"',
        'rm -rf "$JOB_TMP"',
        "",
        f'echo "All tasks finished for sweep: {sweep_job_name}"',
        "",
    ]

    return "\n".join(header_lines)


def _run_sbatch(slurm_path: str, dependency: Optional[str] = None) -> None:
    cmd = ["sbatch"]
    if dependency is not None and str(dependency).strip():
        cmd.append(f"--dependency={str(dependency).strip()}")
    cmd.append(slurm_path)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster invariance sweep scheduler from JSON.")
    ap.add_argument("--json", required=True, help="Path to sweep JSON config.")
    ap.add_argument("--mode", choices=["local", "slurm", "preview"], default="slurm")
    ap.add_argument("--job-name", default=None, help="Override sweep job name (optional).")
    ap.add_argument("--tmp-dir", default=None, help="Directory to write preview slurm files.")
    ap.add_argument(
        "--dependency",
        default=None,
        help="Slurm dependency for each sbatch (e.g. afterok:12345). Waits until that job exits successfully.",
    )
    ap.add_argument(
        "--k",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of k values to override the JSON 'k_values' (no extra JSON files).",
    )
    args = ap.parse_args()

    sweep_cfg = _load_config(args.json)
    _get_cluster_invariance_driver_module()._apply_run_scoping(sweep_cfg)  # type: ignore[attr-defined]
    k_values = sweep_cfg.get("k_values", [])
    if not isinstance(k_values, list) or not k_values:
        raise ValueError("Sweep JSON must contain non-empty list 'k_values'.")

    if args.k is not None and len(args.k) > 0:
        k_values = [int(x) for x in args.k]

    run_values = sweep_cfg.get("run_values")
    if run_values is None:
        run_values = [None]
    if not isinstance(run_values, list) or len(run_values) == 0:
        raise ValueError("Sweep JSON 'run_values' must be a non-empty list if provided.")
    run_values_int: list[Optional[int]] = []
    for r in run_values:
        if r is None:
            run_values_int.append(None)
        else:
            run_values_int.append(int(r))

    seed_values = sweep_cfg.get("seed_values")
    if seed_values is None:
        seed_values = [None]
    if not isinstance(seed_values, list) or len(seed_values) == 0:
        raise ValueError("Sweep JSON 'seed_values' must be a non-empty list if provided.")
    seed_values_int: list[Optional[int]] = []
    for s in seed_values:
        if s is None:
            seed_values_int.append(None)
        else:
            seed_values_int.append(int(s))

    invariance_enabled = _invariance_phase_enabled(sweep_cfg)
    if not invariance_enabled:
        cp0 = sweep_cfg.get("clustering_phase") or {}
        if not isinstance(cp0, dict) or not _parse_bool(cp0.get("enabled"), default=False):
            raise ValueError(
                "invariance_phase.enabled=false requires clustering_phase.enabled=true (clustering-only sweep)."
            )
        sweep_cfg = dict(sweep_cfg)
        sweep_cfg["global_filter_overlay_plots"] = False
        sweep_cfg["run_member_timeseries_plots"] = False
        sweep_cfg["write_cluster_member_samples"] = False
        sweep_cfg["write_per_cluster_cohens_d_cka"] = False
        sweep_cfg["input_staging"] = {**(sweep_cfg.get("input_staging") or {}), "enabled": False}

    runner = _resolve_runner(mode=args.mode, cfg_runner=sweep_cfg.get("runner"))
    environment = _resolve_environment(sweep_cfg.get("environment"))
    if args.mode == "preview":
        runner = "slurm"

    submission_style = str(sweep_cfg.get("submission_style", "separate_jobs"))
    if submission_style not in {
        "separate_jobs",
        "single_job_parallel",
        "multi_node_single_job",
        "multi_node_per_run_parallel_k",
        "one_node_per_run_parallel_k",
    }:
        raise ValueError(
            "submission_style must be one of: separate_jobs, single_job_parallel, "
            "multi_node_single_job, multi_node_per_run_parallel_k, one_node_per_run_parallel_k."
        )

    if not invariance_enabled:
        if submission_style not in ("multi_node_per_run_parallel_k", "one_node_per_run_parallel_k"):
            raise ValueError(
                "invariance_phase.enabled=false is only supported with submission_style "
                "multi_node_per_run_parallel_k or one_node_per_run_parallel_k."
            )

    _cp = sweep_cfg.get("clustering_phase") or {}
    if isinstance(_cp, dict) and _parse_bool(_cp.get("enabled"), default=False):
        if submission_style not in ("multi_node_per_run_parallel_k", "one_node_per_run_parallel_k"):
            raise ValueError(
                "clustering_phase.enabled requires submission_style "
                "multi_node_per_run_parallel_k or one_node_per_run_parallel_k."
            )
        if not _cp.get("data_csv"):
            raise ValueError("clustering_phase.enabled requires clustering_phase.data_csv")

    if invariance_enabled:
        tsfresh_csv = sweep_cfg["tsfresh_csv"]
    else:
        tsfresh_csv = str(sweep_cfg.get("tsfresh_csv") or "")

    clustering_labels_template = sweep_cfg["clustering_labels_template"]
    out_csv_template = sweep_cfg["out_csv_template"]
    cluster_variance_out_csv_template = sweep_cfg["cluster_variance_out_csv_template"]
    cluster_feature_rank_out_csv_template = sweep_cfg["cluster_feature_rank_out_csv_template"]
    min_cluster_size = int(sweep_cfg.get("min_cluster_size", 10))
    id_ip = str(sweep_cfg.get("id_ip", "ip"))
    id_source_file = str(sweep_cfg.get("id_source_file", "source_file"))

    write_cluster_member_samples_sweep = _parse_bool(
        sweep_cfg.get("write_cluster_member_samples"), default=False
    )
    cluster_members_parent_dir_template = sweep_cfg.get("cluster_members_parent_dir_template")
    member_sample_n_sweep = int(sweep_cfg.get("member_sample_n", 5))
    sweep_default_member_seed = int(sweep_cfg.get("member_sample_seed", 0))

    write_per_cluster_cohens_d_cka_sweep = _parse_bool(
        sweep_cfg.get("write_per_cluster_cohens_d_cka"),
        default=False,
    )
    repr_prefix_sweep = str(sweep_cfg.get("repr_prefix", "repr_dim"))

    ip_suffix_filter_sweep = sweep_cfg.get("ip_suffix_filter", "/32")
    if ip_suffix_filter_sweep is not None:
        ip_suffix_filter_sweep = str(ip_suffix_filter_sweep).strip() or None

    raw_run_member_plots = sweep_cfg.get("run_member_timeseries_plots")
    plot_member_timeseries_input_parquet_sweep = sweep_cfg.get("plot_member_timeseries_input_parquet")
    parquet_ok = bool(
        plot_member_timeseries_input_parquet_sweep
        and str(plot_member_timeseries_input_parquet_sweep).strip()
    )
    if raw_run_member_plots is None:
        run_member_timeseries_plots_sweep = False
    else:
        run_member_timeseries_plots_sweep = _parse_bool(raw_run_member_plots, default=False)
        if run_member_timeseries_plots_sweep and not parquet_ok:
            raise ValueError(
                "run_member_timeseries_plots requires sweep key 'plot_member_timeseries_input_parquet' "
                "(non-empty path)."
            )
    plot_member_timeseries_config_out_template = sweep_cfg.get("plot_member_timeseries_config_out_template")
    plot_member_timeseries_key_columns_sweep = str(
        sweep_cfg.get("plot_member_timeseries_key_columns", "ip,source_file")
    )
    plot_member_timeseries_value_column_sweep = str(
        sweep_cfg.get("plot_member_timeseries_value_column", "inbound")
    )
    plot_member_timeseries_num_workers_sweep = int(sweep_cfg.get("plot_member_timeseries_num_workers", 0))
    plot_member_timeseries_y_floor_zero_sweep = _parse_bool(
        sweep_cfg.get("plot_member_timeseries_y_floor_zero"), default=True
    )
    plot_member_timeseries_share_y_max_sweep = _parse_bool(
        sweep_cfg.get("plot_member_timeseries_share_y_max_within_k"), default=False
    )
    _yfm_sw = sweep_cfg.get("plot_member_timeseries_y_fixed_max")
    plot_member_timeseries_y_fixed_max_sweep: Optional[float] = None
    if _yfm_sw is not None and str(_yfm_sw).strip() != "":
        try:
            plot_member_timeseries_y_fixed_max_sweep = float(_yfm_sw)
        except (TypeError, ValueError):
            plot_member_timeseries_y_fixed_max_sweep = None
    if plot_member_timeseries_y_fixed_max_sweep is not None and (
        not math.isfinite(plot_member_timeseries_y_fixed_max_sweep)
        or plot_member_timeseries_y_fixed_max_sweep <= 0
    ):
        plot_member_timeseries_y_fixed_max_sweep = None
    plot_member_timeseries_fisher_overlay_sweep = _parse_bool(
        sweep_cfg.get("plot_member_timeseries_fisher_overlay"), default=False
    )
    plot_member_timeseries_tsfresh_csv_sweep = sweep_cfg.get("plot_member_timeseries_tsfresh_csv")
    plot_member_timeseries_fisher_top_n_sweep = int(sweep_cfg.get("plot_member_timeseries_fisher_top_n", 5))
    plot_member_timeseries_heatmap_filename_sweep = str(
        sweep_cfg.get("plot_member_timeseries_heatmap_filename", "feature_order_fisher.csv")
    )

    global_filter_overlay_plots_sweep = _parse_bool(
        sweep_cfg.get("global_filter_overlay_plots"), default=False
    )
    cohens_d_tanh_d0_sweep = sweep_cfg.get("cohens_d_tanh_d0")
    cohens_d_max_norm_sweep = _parse_bool(sweep_cfg.get("cohens_d_max_norm"), default=False)
    global_filtered_features_csv_sweep = sweep_cfg.get("global_filtered_features_csv")
    global_filtered_features_csv_template = sweep_cfg.get("global_filtered_features_csv_template")
    global_filter_mode_sweep = str(sweep_cfg.get("global_filter_mode", "fisher_min")).strip().lower()
    global_fisher_min_sweep = float(sweep_cfg.get("global_fisher_min", 0.1))
    global_cka_all_csv_sweep = sweep_cfg.get("global_cka_all_csv")
    global_cka_all_feature_col_sweep = sweep_cfg.get("global_cka_all_feature_col")
    global_cka_all_value_col_sweep = sweep_cfg.get("global_cka_all_value_col")
    global_all_features_csv_sweep = sweep_cfg.get("global_all_features_csv")
    global_metrics_csv_sweep = sweep_cfg.get("global_metrics_csv")
    global_fisher_cdf_png_sweep = sweep_cfg.get("global_fisher_cdf_png")
    global_cka_cdf_png_sweep = sweep_cfg.get("global_cka_cdf_png")
    cluster_importance_csv_sweep = sweep_cfg.get("cluster_importance_csv")
    cluster_importance_cdf_png_sweep = sweep_cfg.get("cluster_importance_cdf_png")
    local_importance_all_filename_sweep = sweep_cfg.get("local_importance_all_filename")
    local_importance_top_filename_sweep = sweep_cfg.get("local_importance_top_filename")
    overlay_local_top_k_sweep = int(sweep_cfg.get("overlay_local_top_k", 10))
    skip_global_cdf_plots_sweep = _parse_bool(sweep_cfg.get("skip_global_cdf_plots"), default=False)
    label_col_sweep = str(sweep_cfg.get("label_col", "label"))
    global_cka_feature_col_sweep = str(sweep_cfg.get("global_cka_feature_col", "tsfresh_feature"))
    plot_member_overlay_legend_fontsize_sweep = sweep_cfg.get("plot_member_overlay_legend_fontsize")

    member_timeseries_plots_only_sweep = _parse_bool(
        sweep_cfg.get("member_timeseries_plots_only"), default=False
    )
    plot_member_dual_global_filter_overlays_sweep = _parse_bool(
        sweep_cfg.get("plot_member_dual_global_filter_overlays"), default=False
    )
    plot_member_compact_metric_tsfresh_legend_sweep = _parse_bool(
        sweep_cfg.get("plot_member_compact_metric_tsfresh_legend"), default=False
    )
    precomputed_global_filter_sweep = bool(
        (global_filtered_features_csv_sweep and str(global_filtered_features_csv_sweep).strip())
        or (
            global_filtered_features_csv_template
            and str(global_filtered_features_csv_template).strip()
        )
    )

    if global_filter_overlay_plots_sweep:
        if not precomputed_global_filter_sweep:
            if global_filter_mode_sweep != "fisher_min":
                raise ValueError("global_filter_mode must be 'fisher_min'.")
        if not write_per_cluster_cohens_d_cka_sweep:
            raise ValueError(
                "global_filter_overlay_plots requires write_per_cluster_cohens_d_cka."
            )
    if global_filter_overlay_plots_sweep and run_member_timeseries_plots_sweep:
        if plot_member_timeseries_fisher_overlay_sweep:
            raise ValueError(
                "Do not set plot_member_timeseries_fisher_overlay together with global_filter_overlay_plots."
            )

    modules = sweep_cfg.get("modules", ["conda", "pytorch/2.6.0"])
    if not isinstance(modules, list):
        modules = ["conda", "pytorch/2.6.0"]
    modules = [str(x) for x in modules]

    effective_sbatch_dependency = merge_sbatch_dependency(None, args.dependency)

    if run_member_timeseries_plots_sweep:
        if not member_timeseries_plots_only_sweep and not write_cluster_member_samples_sweep:
            raise ValueError(
                "run_member_timeseries_plots requires write_cluster_member_samples and cluster_members_parent_dir "
                "(unless member_timeseries_plots_only is true)."
            )
    if member_timeseries_plots_only_sweep and plot_member_timeseries_fisher_overlay_sweep:
        raise ValueError("member_timeseries_plots_only cannot be combined with plot_member_timeseries_fisher_overlay.")
    if plot_member_dual_global_filter_overlays_sweep and not run_member_timeseries_plots_sweep:
        raise ValueError("plot_member_dual_global_filter_overlays requires run_member_timeseries_plots.")
    if plot_member_dual_global_filter_overlays_sweep and global_filter_overlay_plots_sweep:
        raise ValueError("Use either global_filter_overlay_plots or plot_member_dual_global_filter_overlays, not both.")

    # Build member configs.
    member_cfgs: list[dict] = []
    job_name_template = str(sweep_cfg.get("job_name_template", "cluster_invariance_k{k}"))
    sweep_job_name = str(sweep_cfg.get("sweep_job_name", "cluster_invariance_sweep"))
    if args.job_name:
        sweep_job_name = args.job_name

    # We'll store in sweep_cfg for parallel script builder.
    sweep_cfg["sweep_job_name"] = sweep_job_name

    for k in k_values:
        for run in run_values_int:
            for seed in seed_values_int:
                run_pass: Optional[int] = run
                seed_pass: Optional[int] = seed
                member_job_name = job_name_template.format(k=int(k), run=run_pass, seed=seed_pass)
                out_csv_path = _format_tmpl(out_csv_template, k=int(k), run=run_pass, seed=seed_pass)
                parent_for_members: Optional[str] = None
                if write_cluster_member_samples_sweep:
                    if cluster_members_parent_dir_template:
                        parent_for_members = _format_tmpl(
                            cluster_members_parent_dir_template, k=int(k), run=run_pass, seed=seed_pass
                        )
                    else:
                        parent_for_members = str(Path(out_csv_path).expanduser().parent)
                elif run_member_timeseries_plots_sweep:
                    if cluster_members_parent_dir_template:
                        parent_for_members = _format_tmpl(
                            cluster_members_parent_dir_template, k=int(k), run=run_pass, seed=seed_pass
                        )
                    else:
                        parent_for_members = str(Path(out_csv_path).expanduser().parent)
                elif global_filter_overlay_plots_sweep:
                    if cluster_members_parent_dir_template:
                        parent_for_members = _format_tmpl(
                            cluster_members_parent_dir_template, k=int(k), run=run_pass, seed=seed_pass
                        )
                    else:
                        parent_for_members = str(Path(out_csv_path).expanduser().parent)
                member_seed_effective = (
                    int(seed_pass) if seed_pass is not None else sweep_default_member_seed
                )
                if write_per_cluster_cohens_d_cka_sweep:
                    th_check = str(Path(out_csv_path).expanduser().parent)
                    if not th_check:
                        raise ValueError(
                            "write_per_cluster_cohens_d_cka requires a valid parent directory next to out_csv."
                        )
                plot_cfg_out_member = None
                if plot_member_timeseries_config_out_template:
                    plot_cfg_out_member = _format_tmpl(
                        str(plot_member_timeseries_config_out_template),
                        k=int(k),
                        run=run_pass,
                        seed=seed_pass,
                    )
                member_cfgs.append(
                    {
                        "k": int(k),
                        "run": run_pass,
                        "seed": seed_pass,
                        "job_name": member_job_name,
                        "clustering_csv": _format_tmpl(clustering_labels_template, k=int(k), run=run_pass, seed=seed_pass),
                        "tsfresh_csv": tsfresh_csv,
                        "out_csv": out_csv_path,
                        "cluster_variance_out_csv": _format_tmpl(
                            cluster_variance_out_csv_template, k=int(k), run=run_pass, seed=seed_pass
                        ),
                        "cluster_feature_rank_out_csv": _format_tmpl(
                            cluster_feature_rank_out_csv_template, k=int(k), run=run_pass, seed=seed_pass
                        ),
                        "min_cluster_size": min_cluster_size,
                        "id_ip": id_ip,
                        "id_source_file": id_source_file,
                        "environment": environment,
                        "write_cluster_member_samples": write_cluster_member_samples_sweep,
                        "cluster_members_parent_dir": parent_for_members,
                        "member_sample_n": member_sample_n_sweep,
                        "member_sample_seed": member_seed_effective,
                        "write_per_cluster_cohens_d_cka": write_per_cluster_cohens_d_cka_sweep,
                        "repr_prefix": repr_prefix_sweep,
                        "ip_suffix_filter": ip_suffix_filter_sweep,
                        "run_member_timeseries_plots": run_member_timeseries_plots_sweep,
                        "plot_member_timeseries_input_parquet": plot_member_timeseries_input_parquet_sweep,
                        "plot_member_timeseries_config_out": plot_cfg_out_member,
                        "plot_member_timeseries_key_columns": plot_member_timeseries_key_columns_sweep,
                        "plot_member_timeseries_value_column": plot_member_timeseries_value_column_sweep,
                        "plot_member_timeseries_num_workers": plot_member_timeseries_num_workers_sweep,
                        "plot_member_timeseries_y_floor_zero": plot_member_timeseries_y_floor_zero_sweep,
                        "plot_member_timeseries_y_fixed_max": plot_member_timeseries_y_fixed_max_sweep,
                        "plot_member_timeseries_share_y_max_within_k": plot_member_timeseries_share_y_max_sweep,
                        "plot_member_timeseries_fisher_overlay": plot_member_timeseries_fisher_overlay_sweep,
                        "plot_member_timeseries_tsfresh_csv": plot_member_timeseries_tsfresh_csv_sweep,
                        "plot_member_timeseries_fisher_top_n": plot_member_timeseries_fisher_top_n_sweep,
                        "plot_member_timeseries_heatmap_filename": plot_member_timeseries_heatmap_filename_sweep,
                        "global_filter_overlay_plots": global_filter_overlay_plots_sweep,
                        "skip_global_cdf_plots": skip_global_cdf_plots_sweep,
                        "cohens_d_tanh_d0": cohens_d_tanh_d0_sweep,
                        "cohens_d_max_norm": cohens_d_max_norm_sweep,
                        "global_filtered_features_csv": (
                            _format_tmpl(
                                str(global_filtered_features_csv_template),
                                k=int(k),
                                run=run_pass,
                                seed=seed_pass,
                            )
                            if global_filtered_features_csv_template
                            and str(global_filtered_features_csv_template).strip()
                            else global_filtered_features_csv_sweep
                        ),
                        "global_filter_mode": global_filter_mode_sweep,
                        "global_fisher_min": global_fisher_min_sweep,
                        "global_cka_all_csv": global_cka_all_csv_sweep,
                        "global_cka_all_feature_col": global_cka_all_feature_col_sweep,
                        "global_cka_all_value_col": global_cka_all_value_col_sweep,
                        "global_all_features_csv": global_all_features_csv_sweep,
                        "global_metrics_csv": global_metrics_csv_sweep,
                        "global_fisher_cdf_png": global_fisher_cdf_png_sweep,
                        "global_cka_cdf_png": global_cka_cdf_png_sweep,
                        "cluster_importance_csv": cluster_importance_csv_sweep,
                        "cluster_importance_cdf_png": cluster_importance_cdf_png_sweep,
                        "local_importance_all_filename": local_importance_all_filename_sweep,
                        "local_importance_top_filename": local_importance_top_filename_sweep,
                        "overlay_local_top_k": overlay_local_top_k_sweep,
                        "label_col": label_col_sweep,
                        "global_cka_feature_col": global_cka_feature_col_sweep,
                        "plot_member_overlay_legend_fontsize": plot_member_overlay_legend_fontsize_sweep,
                        "member_timeseries_plots_only": member_timeseries_plots_only_sweep,
                        "plot_member_dual_global_filter_overlays": plot_member_dual_global_filter_overlays_sweep,
                        "plot_member_compact_metric_tsfresh_legend": plot_member_compact_metric_tsfresh_legend_sweep,
                    }
                )

    if runner == "local":
        if not invariance_enabled and submission_style != "one_node_per_run_parallel_k":
            raise ValueError(
                "invariance_phase.enabled=false with runner=local requires submission_style "
                "one_node_per_run_parallel_k."
            )
        if submission_style == "one_node_per_run_parallel_k":
            mod_bq = _get_cluster_invariance_driver_module()
            _bash_quote_local = mod_bq._bash_quote  # type: ignore[attr-defined]
            groups = _group_members_by_run_seed(member_cfgs)
            for group in groups:
                cp_local = _clustering_phase_bash_string(sweep_cfg, group, _bash_quote_local)
                if cp_local:
                    _run_bash("set -euo pipefail && " + cp_local)
                if not invariance_enabled:
                    continue
                bg_parts: list[str] = []
                for m in group:
                    _cmd1, _cmd2 = _build_single_member_local_cmds(member_cfg=m)
                    core = _cmd1 if _cmd2 is None else f"{_cmd1} && {_cmd2}"
                    out_dir = _derive_out_dir(m["out_csv"])
                    _ensure_dir(out_dir)
                    if m.get("write_cluster_member_samples") and m.get("cluster_members_parent_dir"):
                        _ensure_dir(str(Path(m["cluster_members_parent_dir"]).expanduser()))
                    _ensure_dir(str(Path(m["cluster_variance_out_csv"]).expanduser().parent))
                    _ensure_dir(str(Path(m["cluster_feature_rank_out_csv"]).expanduser().parent))
                    log_path = str(Path(out_dir) / f"cluster_metrics_sweep_{m['job_name']}.log")
                    bg_parts.append(f"({core}) &> {_quote_bash(log_path)} &")
                run_line = "set -euo pipefail && " + " ".join(bg_parts) + " wait"
                _run_bash(run_line)
            return
        if submission_style in ("multi_node_single_job", "multi_node_per_run_parallel_k"):
            raise ValueError(f"{submission_style} requires runner=slurm (not local).")
        local_commands: list[tuple[str, str]] = []
        for m in member_cfgs:
            _cmd1, _cmd2 = _build_single_member_local_cmds(member_cfg=m)
            local_cmd_core = _cmd1 if _cmd2 is None else f"{_cmd1} && {_cmd2}"
            out_dir = _derive_out_dir(m["out_csv"])
            _ensure_dir(out_dir)
            if m.get("write_cluster_member_samples") and m.get("cluster_members_parent_dir"):
                _ensure_dir(str(Path(m["cluster_members_parent_dir"]).expanduser()))
            _ensure_dir(str(Path(m["cluster_variance_out_csv"]).expanduser().parent))
            _ensure_dir(str(Path(m["cluster_feature_rank_out_csv"]).expanduser().parent))
            log_path = str(Path(out_dir) / f"cluster_metrics_sweep_{m['job_name']}.log")
            local_cmd = f"({local_cmd_core}) &> {_quote_bash(log_path)}"
            local_commands.append((m["job_name"], local_cmd))

        # Default behavior: run all local sweep jobs in parallel.
        local_max_workers = int(sweep_cfg.get("local_max_workers", len(local_commands)))
        if local_max_workers <= 0:
            local_max_workers = len(local_commands)
        local_max_workers = min(local_max_workers, len(local_commands))

        with concurrent.futures.ThreadPoolExecutor(max_workers=local_max_workers) as executor:
            futures = {
                executor.submit(_run_bash, command): job_name
                for job_name, command in local_commands
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return

    if submission_style == "multi_node_single_job":
        sbatch_cfg = sweep_cfg.get("sbatch_multi_node", sweep_cfg.get("sbatch_parallel", sweep_cfg.get("sbatch", {})))
        if not isinstance(sbatch_cfg, dict):
            sbatch_cfg = {}
        slurm_text = _build_multi_node_slurm(
            sweep_cfg=sweep_cfg,
            member_cfgs=member_cfgs,
            sbatch_cfg=sbatch_cfg,
            modules=modules,
            environment=environment,
            tsfresh_csv=tsfresh_csv,
            invariance_enabled=invariance_enabled,
        )
        tmp_dir = args.tmp_dir or tempfile.gettempdir()
        slurm_path = _write_preview_slurm(slurm_text=slurm_text, tmp_dir=tmp_dir, job_name=sweep_job_name)
        if args.mode == "preview":
            print(f"Preview slurm written to: {slurm_path}")
            return
        _run_sbatch(slurm_path, dependency=effective_sbatch_dependency)
        return

    if submission_style == "multi_node_per_run_parallel_k":
        sbatch_cfg = sweep_cfg.get("sbatch_multi_node", sweep_cfg.get("sbatch_parallel", sweep_cfg.get("sbatch", {})))
        if not isinstance(sbatch_cfg, dict):
            sbatch_cfg = {}
        slurm_text = _build_multi_node_slurm_per_run_parallel_k(
            sweep_cfg=sweep_cfg,
            member_cfgs=member_cfgs,
            sbatch_cfg=sbatch_cfg,
            modules=modules,
            environment=environment,
            tsfresh_csv=tsfresh_csv,
            invariance_enabled=invariance_enabled,
        )
        tmp_dir = args.tmp_dir or tempfile.gettempdir()
        slurm_path = _write_preview_slurm(slurm_text=slurm_text, tmp_dir=tmp_dir, job_name=sweep_job_name)
        if args.mode == "preview":
            print(f"Preview slurm written to: {slurm_path}")
            return
        _run_sbatch(slurm_path, dependency=effective_sbatch_dependency)
        return

    if submission_style == "one_node_per_run_parallel_k":
        sbatch_cfg = sweep_cfg.get(
            "sbatch_per_run_parallel", sweep_cfg.get("sbatch_parallel", sweep_cfg.get("sbatch", {}))
        )
        if not isinstance(sbatch_cfg, dict):
            sbatch_cfg = {}
        groups = _group_members_by_run_seed(member_cfgs)
        workdir = str(sweep_cfg.get("workdir", "<external-netburst-root>/src/analysis"))
        tmpl = str(sweep_cfg.get("per_run_job_name_template", "{sweep_job_name}_run_{run}_seed_{seed}"))
        tmp_dir = args.tmp_dir or tempfile.gettempdir()

        def _one_group_slurm(group: List[dict]) -> str:
            mod_bq = _get_cluster_invariance_driver_module()
            _bash_quote_1n = mod_bq._bash_quote  # type: ignore[attr-defined]
            run0, seed0 = group[0]["run"], group[0]["seed"]
            group_job_name = _format_per_run_job_name(tmpl, sweep_job_name, run0, seed0)
            ks = [m["k"] for m in group]
            header = [
                f"# One Slurm job = one node = one (run, seed); all k run in parallel on that node.",
                f"# run={run0} seed={seed0}",
                f"# k_values this job ({len(group)} processes): {ks}",
            ]
            log_dir = Path(_derive_out_dir(group[0]["out_csv"])) / "logs"
            cp_one = _clustering_phase_bash_string(sweep_cfg, group, _bash_quote_1n)
            return _build_parallel_slurm_on_one_node(
                member_cfgs=group,
                job_name=group_job_name,
                header_comment_lines=header,
                sbatch_cfg=sbatch_cfg,
                modules=modules,
                environment=environment,
                workdir=workdir,
                sbatch_output=str(log_dir / f"{group_job_name}_%j.out"),
                sbatch_error=str(log_dir / f"{group_job_name}_%j.err"),
                clustering_prefix=cp_one or None,
                invariance_enabled=invariance_enabled,
            )

        if args.mode == "preview":
            for group in groups:
                gj = _format_per_run_job_name(
                    tmpl, sweep_job_name, group[0]["run"], group[0]["seed"]
                )
                _write_preview_slurm(slurm_text=_one_group_slurm(group), tmp_dir=tmp_dir, job_name=gj)
            print(f"Preview slurm written for {len(groups)} one-node-per-run jobs to: {tmp_dir}")
            return

        with tempfile.TemporaryDirectory(prefix=f"{sweep_job_name}_") as td:
            for group in groups:
                gj = _format_per_run_job_name(
                    tmpl, sweep_job_name, group[0]["run"], group[0]["seed"]
                )
                _ensure_dir(str(Path(_derive_out_dir(group[0]["out_csv"])) / "logs"))
                slurm_path = _write_preview_slurm(
                    slurm_text=_one_group_slurm(group), tmp_dir=td, job_name=gj
                )
                _run_sbatch(slurm_path, dependency=effective_sbatch_dependency)
        return

    if submission_style == "single_job_parallel":
        # Choose sbatch config for parallel style
        sbatch_cfg = sweep_cfg.get("sbatch_parallel", sweep_cfg.get("sbatch", {}))
        if not isinstance(sbatch_cfg, dict):
            sbatch_cfg = {}
        slurm_text = _build_single_job_parallel_slurm(
            sweep_cfg=sweep_cfg,
            member_cfgs=member_cfgs,
            sbatch_cfg=sbatch_cfg,
            modules=modules,
            environment=environment,
            invariance_enabled=invariance_enabled,
        )
        tmp_dir = args.tmp_dir or tempfile.gettempdir()
        slurm_path = _write_preview_slurm(slurm_text=slurm_text, tmp_dir=tmp_dir, job_name=sweep_job_name)
        if args.mode == "preview":
            print(f"Preview slurm written to: {slurm_path}")
            return
        _run_sbatch(slurm_path, dependency=effective_sbatch_dependency)
        return

    # separate_jobs
    sbatch_cfg = sweep_cfg.get("sbatch_separate", sweep_cfg.get("sbatch", {}))
    if not isinstance(sbatch_cfg, dict):
        sbatch_cfg = {}
    tmp_dir = args.tmp_dir or tempfile.gettempdir()

    if args.mode == "preview":
        for m in member_cfgs:
            slurm_text = _build_single_member_slurm_text(member_cfg=m, sbatch_cfg=sbatch_cfg, modules=modules)
            _write_preview_slurm(slurm_text=slurm_text, tmp_dir=tmp_dir, job_name=m["job_name"])
        print(f"Preview slurm written for {len(member_cfgs)} jobs to: {tmp_dir}")
        return

    # args.mode == "slurm"
    # In slurm mode we create per-member temp slurm files, sbatch them, and delete.
    with tempfile.TemporaryDirectory(prefix=f"{sweep_job_name}_") as td:
        for m in member_cfgs:
            slurm_text = _build_single_member_slurm_text(member_cfg=m, sbatch_cfg=sbatch_cfg, modules=modules)
            slurm_path = _write_preview_slurm(slurm_text=slurm_text, tmp_dir=td, job_name=m["job_name"])
            # Ensure log dirs exist before sbatch (output/error redirection paths).
            out_dir = _derive_out_dir(m["out_csv"])
            _ensure_dir(str(Path(out_dir) / "logs"))
            _run_sbatch(slurm_path, dependency=effective_sbatch_dependency)


if __name__ == "__main__":
    main()

