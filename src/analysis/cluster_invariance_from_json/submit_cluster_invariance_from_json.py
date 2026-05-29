#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))


DEFAULT_MIN_CLUSTER_SIZE = 10
DEFAULT_ID_IP = "ip"
DEFAULT_ID_SOURCE_FILE = "source_file"
DEFAULT_JOB_NAME = "cluster_invariance"
DEFAULT_SQUEUE = "premium"
DEFAULT_ACCOUNT = "<SLURM_ACCOUNT>"
DEFAULT_TIME = "04:00:00"
DEFAULT_CPUS_PER_TASK = 8
DEFAULT_WRITE_CLUSTER_MEMBER_SAMPLES = False
DEFAULT_MEMBER_SAMPLE_N = 5
DEFAULT_MEMBER_SAMPLE_SEED = 0
DEFAULT_WRITE_PER_CLUSTER_COHENS_D_CKA = False
DEFAULT_RUN_MEMBER_TIMESERIES_PLOTS = False
DEFAULT_REPR_PREFIX = "repr_dim"
DEFAULT_ENVIRONMENT = "nersc"
DEFAULT_RUN_NORMALIZED_CKA_COHEN = False
DEFAULT_NORMALIZED_CKA_COHEN_TOP_K = 10
DEFAULT_MODULES = [
    "conda",
    "pytorch/2.6.0",
]


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


def _load_config(json_path: str) -> dict:
    p = Path(json_path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bash_quote(s: str) -> str:
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


def _cmd_list_to_bash(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _run_bash(command: str) -> None:
    subprocess.run(["bash", "-lc", command], check=True)


def _apply_run_scoping(cfg: dict) -> None:
    """
    When artifact_root and run_tag are set, prefix relative output paths with
    artifact_root/runs/<run_tag>/. Use run_tag \"auto\" for a timestamp suffix.
    """
    tag = cfg.get("run_tag")
    if tag == "auto":
        cfg["run_tag"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = cfg["run_tag"]
    root = cfg.get("artifact_root")
    if not root or not str(root).strip():
        return
    if not tag or not str(tag).strip():
        return
    root_p = Path(os.path.expandvars(str(root))).expanduser()
    run_dir = root_p / "runs" / str(tag)
    for key in (
        "out_csv",
        "cluster_variance_out_csv",
        "cluster_feature_rank_out_csv",
        "cluster_members_parent_dir",
        "plot_member_timeseries_config_out",
    ):
        v = cfg.get(key)
        if v is None or str(v).strip() == "":
            continue
        p = Path(str(v))
        if p.is_absolute():
            continue
        cfg[key] = str(run_dir / p)
    cp = cfg.get("clustering_phase")
    if isinstance(cp, dict):
        cp_out = cp.get("out_dir")
        if cp_out is not None and str(cp_out).strip() != "":
            cp_p = Path(str(cp_out))
            if not cp_p.is_absolute():
                cp["out_dir"] = str(run_dir / cp_p)


def _derive_log_paths(out_csv: str, job_name: str) -> tuple[str, str, str]:
    out_path = Path(out_csv).expanduser()
    out_dir = out_path.parent
    log_dir = out_dir / "logs"
    log_out = log_dir / f"{job_name}_%j.out"
    log_err = log_dir / f"{job_name}_%j.err"
    return str(out_dir), str(log_out), str(log_err)


def _require_str(cfg: dict, key: str) -> str:
    v = cfg.get(key)
    if v is None or str(v) == "" or str(v) == "None":
        raise ValueError(f"JSON must contain non-empty '{key}'.")
    return str(v)


def _maybe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _build_clustering_phase_command(
    *,
    cfg: dict,
    out_csv: str,
    default_ip_suffix_filter: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns:
      (resolved_clustering_csv, clustering_phase_cmd_bash, clustering_phase_out_dir)

    One-of-two contract:
      - Either provide clustering_csv (external labels), OR
      - set run_clustering_phase=true and provide clustering_phase inputs.
    """
    run_cp = _parse_bool(cfg.get("run_clustering_phase"), default=False)
    clustering_csv_external = _maybe_str(cfg.get("clustering_csv"))

    if run_cp and clustering_csv_external:
        raise ValueError(
            "Choose exactly one cluster-label source: either non-empty 'clustering_csv' "
            "or 'run_clustering_phase=true', not both."
        )
    if (not run_cp) and (not clustering_csv_external):
        raise ValueError(
            "Cluster-features requires cluster labels. Provide non-empty 'clustering_csv' "
            "or set 'run_clustering_phase=true' with a valid 'clustering_phase' block."
        )
    if not run_cp:
        return clustering_csv_external, None, None

    cp = cfg.get("clustering_phase")
    if not isinstance(cp, dict):
        raise ValueError("run_clustering_phase=true requires a JSON object 'clustering_phase'.")
    data_csv = _maybe_str(cp.get("data_csv"))
    if not data_csv:
        raise ValueError("run_clustering_phase=true requires non-empty 'clustering_phase.data_csv'.")
    method = str(cp.get("method", "kmeans")).strip().lower()
    if method != "kmeans":
        raise ValueError(
            "run_clustering_phase currently supports clustering_phase.method='kmeans' only."
        )
    n_clusters_raw = cp.get("n_clusters")
    if n_clusters_raw is None or str(n_clusters_raw).strip() == "":
        raise ValueError("run_clustering_phase=true requires integer 'clustering_phase.n_clusters'.")
    try:
        n_clusters = int(n_clusters_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("clustering_phase.n_clusters must be an integer.") from exc
    if n_clusters <= 1:
        raise ValueError("clustering_phase.n_clusters must be > 1.")

    kmeans_metric = str(cp.get("kmeans_metric", "cosine")).strip().lower()
    if kmeans_metric not in {"euclidean", "cosine"}:
        raise ValueError("clustering_phase.kmeans_metric must be one of: euclidean, cosine.")
    random_state = int(cp.get("random_state", 42))
    minimal_cleaning = _parse_bool(cp.get("minimal_cleaning"), default=True)
    use_all_features = _parse_bool(cp.get("use_all_features"), default=False)
    out_dir = _maybe_str(cp.get("out_dir")) or str(Path(out_csv).expanduser().parent / "clustering_phase")

    cp_ip = _maybe_str(cp.get("ip_suffix_filter"))
    if cp_ip is None:
        cp_ip = default_ip_suffix_filter

    clustering_script = Path(__file__).resolve().parents[1] / "clustering_analysis.py"
    cmd: list[str] = [
        sys.executable,
        str(clustering_script),
        str(data_csv),
        "--method",
        "kmeans",
        "--n-clusters",
        str(n_clusters),
        "--kmeans-metric",
        kmeans_metric,
        "--out-dir",
        str(out_dir),
        "--random-state",
        str(random_state),
    ]
    if minimal_cleaning:
        cmd.append("--minimal-cleaning")
    if use_all_features:
        cmd.append("--use-all-features")
    if cp_ip:
        cmd.extend(["--ip-suffix-filter", str(cp_ip)])

    resolved_clustering_csv = str(Path(out_dir).expanduser() / f"k_{n_clusters}_labels.csv")
    return resolved_clustering_csv, _cmd_list_to_bash(cmd), str(Path(out_dir).expanduser())


def _build_normalized_cka_cohen_bash(
    *,
    out_csv: str,
    top_k: int,
    quote_fn: Any,
) -> str:
    parent = str(Path(out_csv).expanduser().parent)
    cfg_path = str(Path(parent) / "normalized_cka_cohen_single_run.config.json")
    cohens_csv = str(Path(parent) / "cohens_d_across_clusters.csv")
    cfg_json = json.dumps(
        {
            "k_values": [1],
            "run_values": [1],
            "seed_values": [1],
            "cluster_members_parent_dir_template": parent,
            "cohens_d_by_cluster_out_csv_template": cohens_csv,
            "top_k": int(top_k),
        },
        indent=2,
    )
    py = str(Path(__file__).resolve().parents[1] / "compute_normalized_cka_cohen_importance.py")
    lines = [
        f"cat > {quote_fn(cfg_path)} <<'EOF'",
        cfg_json,
        "EOF",
        f"python3 {quote_fn(py)} {quote_fn(cfg_path)} --top-k {int(top_k)}",
    ]
    return "\n".join(lines)


def _build_local_cmds(
    *,
    clustering_csv: str,
    tsfresh_csv: str,
    out_csv: str,
    cluster_variance_out_csv: str,
    cluster_feature_rank_out_csv: str,
    min_cluster_size: int,
    id_ip: str,
    id_source_file: str,
    write_cluster_member_samples: bool = False,
    cluster_members_parent_dir: Optional[str] = None,
    member_sample_n: int = DEFAULT_MEMBER_SAMPLE_N,
    member_sample_seed: int = DEFAULT_MEMBER_SAMPLE_SEED,
    write_per_cluster_cohens_d_cka: bool = False,
    repr_prefix: str = DEFAULT_REPR_PREFIX,
    ip_suffix_filter: Optional[str] = None,
    run_member_timeseries_plots: bool = DEFAULT_RUN_MEMBER_TIMESERIES_PLOTS,
    plot_member_timeseries_input_parquet: Optional[str] = None,
    plot_member_timeseries_config_out: Optional[str] = None,
    plot_member_timeseries_key_columns: str = "ip,source_file",
    plot_member_timeseries_value_column: str = "inbound",
    plot_member_timeseries_num_workers: int = 0,
    plot_member_timeseries_y_floor_zero: bool = True,
    plot_member_timeseries_y_fixed_max: Optional[float] = None,
    plot_member_timeseries_share_y_max_within_k: bool = False,
    plot_member_timeseries_fisher_overlay: bool = False,
    plot_member_timeseries_tsfresh_csv: Optional[str] = None,
    plot_member_timeseries_fisher_top_n: int = 5,
    plot_member_timeseries_heatmap_filename: str = "feature_order_fisher.csv",
    global_filter_overlay_plots: bool = False,
    global_filter_mode: str = "fisher_min",
    global_fisher_min: float = 0.1,
    overlay_local_top_k: int = 10,
    label_col: str = "label",
    global_cka_feature_col: str = "tsfresh_feature",
    plot_member_overlay_legend_fontsize: Optional[float] = None,
) -> tuple[str, Optional[str]]:
    analysis_dir = Path(__file__).resolve().parents[1]  # .../src/analysis
    cluster_script = analysis_dir / "cluster_analysis.py"
    plot_writer_script = analysis_dir / "write_member_plot_config.py"
    plot_main_script = analysis_dir / "plot_member_timeseries_from_parquet.py"
    overlay_script = analysis_dir / "global_filtered_overlay_prep.py"

    cmd1: list[str] = [
        sys.executable,
        str(cluster_script),
        "--clustering",
        clustering_csv,
        "--tsfresh",
        tsfresh_csv,
        "--out",
        out_csv,
        "--min-cluster-size",
        str(min_cluster_size),
        "--global-fisher-min",
        str(float(global_fisher_min)),
        "--cluster-variance-out",
        cluster_variance_out_csv,
        "--cluster-feature-rank-out",
        cluster_feature_rank_out_csv,
        "--id-ip",
        id_ip,
        "--id-source-file",
        id_source_file,
    ]
    if ip_suffix_filter:
        cmd1.extend(["--ip-suffix-filter", str(ip_suffix_filter)])
    if write_cluster_member_samples:
        cmd1.extend(
            [
                "--write-cluster-member-samples",
                "--cluster-members-parent-dir",
                cluster_members_parent_dir or "",
                "--member-sample-n",
                str(int(member_sample_n)),
                "--member-sample-seed",
                str(int(member_sample_seed)),
            ]
        )

    if write_per_cluster_cohens_d_cka:
        th_dir = str(Path(out_csv).expanduser().parent)
        cmd1.extend(
            [
                "--write-per-cluster-cohens-d-cka",
                "--per-cluster-thresholds-dir",
                th_dir or "",
                "--repr-prefix",
                str(repr_prefix),
            ]
        )
    cmd2: Optional[list[str]] = None

    overlay_cmd: Optional[list[str]] = None
    if global_filter_overlay_plots:
        if not cluster_members_parent_dir:
            raise ValueError("global_filter_overlay_plots requires cluster_members_parent_dir.")
        gfm = str(global_filter_mode).strip().lower()
        if gfm != "fisher_min":
            raise ValueError("global_filter_mode must be 'fisher_min'.")
        overlay_cmd = [
            sys.executable,
            str(overlay_script),
            "--clustering-csv",
            clustering_csv,
            "--tsfresh-csv",
            tsfresh_csv,
            "--parent-out",
            str(Path(cluster_members_parent_dir).expanduser()),
            "--global-filter-mode",
            gfm,
            "--local-overlay-top-k",
            str(int(overlay_local_top_k)),
            "--label-col",
            str(label_col),
            "--id-ip",
            id_ip,
            "--id-source-file",
            id_source_file,
            "--cka-feature-col",
            str(global_cka_feature_col),
        ]
        if gfm == "fisher_min":
            overlay_cmd.extend(["--global-fisher-min", str(float(global_fisher_min))])
        if ip_suffix_filter:
            overlay_cmd.extend(["--ip-suffix-filter", str(ip_suffix_filter)])

    if global_filter_overlay_plots and run_member_timeseries_plots:
        if not write_per_cluster_cohens_d_cka:
            raise ValueError(
                "global_filter_overlay_plots with run_member_timeseries_plots requires write_per_cluster_cohens_d_cka."
            )
        if plot_member_timeseries_fisher_overlay:
            raise ValueError(
                "Do not combine plot_member_timeseries_fisher_overlay with global_filter_overlay_plots."
            )

    if run_member_timeseries_plots:
        if not write_cluster_member_samples or not cluster_members_parent_dir:
            raise ValueError(
                "run_member_timeseries_plots requires write_cluster_member_samples and cluster_members_parent_dir."
            )
        if not plot_member_timeseries_input_parquet:
            raise ValueError("run_member_timeseries_plots requires plot_member_timeseries_input_parquet.")
        parent_exp = str(Path(cluster_members_parent_dir).expanduser())
        glob_pat = str(Path(parent_exp) / "cluster_*" / "members.json")
        cfg_out = plot_member_timeseries_config_out or str(Path(parent_exp) / "plot_member_timeseries_config.json")
        write_cfg_cmd = [
            sys.executable,
            str(plot_writer_script),
            "--out",
            cfg_out,
            "--input-parquet",
            str(plot_member_timeseries_input_parquet),
            "--members-json-glob",
            glob_pat,
            "--key-columns",
            plot_member_timeseries_key_columns,
            "--value-column",
            plot_member_timeseries_value_column,
            "--num-workers",
            str(int(plot_member_timeseries_num_workers)),
        ]
        if not plot_member_timeseries_y_floor_zero:
            write_cfg_cmd.append("--no-y-floor-zero")
        if plot_member_timeseries_share_y_max_within_k:
            write_cfg_cmd.append("--share-y-max-within-k")
        if plot_member_timeseries_y_fixed_max is not None:
            write_cfg_cmd.extend(
                ["--y-fixed-max", str(float(plot_member_timeseries_y_fixed_max))]
            )
        if plot_member_timeseries_fisher_overlay:
            tsf_o = plot_member_timeseries_tsfresh_csv or tsfresh_csv
            write_cfg_cmd.extend(
                [
                    "--fisher-overlay",
                    "--tsfresh-csv",
                    str(tsf_o),
                    "--fisher-top-n",
                    str(int(plot_member_timeseries_fisher_top_n)),
                    "--heatmap-filename",
                    str(plot_member_timeseries_heatmap_filename),
                ]
            )
        elif global_filter_overlay_plots:
            tsf_o = plot_member_timeseries_tsfresh_csv or tsfresh_csv
            write_cfg_cmd.extend(
                [
                    "--dual-global-filter-overlays",
                    "--tsfresh-csv",
                    str(tsf_o),
                    "--overlay-top-n",
                    str(int(overlay_local_top_k)),
                ]
            )
            if plot_member_overlay_legend_fontsize is not None:
                write_cfg_cmd.extend(
                    ["--overlay-legend-fontsize", str(float(plot_member_overlay_legend_fontsize))]
                )
        plot_cmd = [sys.executable, str(plot_main_script), "--config", cfg_out]
        plot_bash = _cmd_list_to_bash(write_cfg_cmd) + " && " + _cmd_list_to_bash(plot_cmd)
    else:
        plot_bash = ""

    cmd1_bash = _cmd_list_to_bash(cmd1)
    post: list[str] = []
    if overlay_cmd is not None:
        post.append(_cmd_list_to_bash(overlay_cmd))
    if cmd2 is not None:
        post.append(_cmd_list_to_bash(cmd2))
    if plot_bash:
        post.append(plot_bash)
    cmd2_bash = " && ".join(post) if post else None
    return cmd1_bash, cmd2_bash


def _build_slurm_script(
    *,
    job_name: str,
    clustering_csv: str,
    tsfresh_csv: str,
    out_csv: str,
    cluster_variance_out_csv: str,
    cluster_feature_rank_out_csv: str,
    min_cluster_size: int,
    id_ip: str,
    id_source_file: str,
    write_cluster_member_samples: bool = False,
    cluster_members_parent_dir: Optional[str] = None,
    member_sample_n: int = DEFAULT_MEMBER_SAMPLE_N,
    member_sample_seed: int = DEFAULT_MEMBER_SAMPLE_SEED,
    write_per_cluster_cohens_d_cka: bool = False,
    repr_prefix: str = DEFAULT_REPR_PREFIX,
    ip_suffix_filter: Optional[str] = None,
    run_member_timeseries_plots: bool = DEFAULT_RUN_MEMBER_TIMESERIES_PLOTS,
    plot_member_timeseries_input_parquet: Optional[str] = None,
    plot_member_timeseries_config_out: Optional[str] = None,
    plot_member_timeseries_key_columns: str = "ip,source_file",
    plot_member_timeseries_value_column: str = "inbound",
    plot_member_timeseries_num_workers: int = 0,
    plot_member_timeseries_y_floor_zero: bool = True,
    plot_member_timeseries_y_fixed_max: Optional[float] = None,
    plot_member_timeseries_share_y_max_within_k: bool = False,
    plot_member_timeseries_fisher_overlay: bool = False,
    plot_member_timeseries_tsfresh_csv: Optional[str] = None,
    plot_member_timeseries_fisher_top_n: int = 5,
    plot_member_timeseries_heatmap_filename: str = "feature_order_fisher.csv",
    global_filter_overlay_plots: bool = False,
    global_filter_mode: str = "fisher_min",
    global_fisher_min: float = 0.1,
    overlay_local_top_k: int = 10,
    label_col: str = "label",
    global_cka_feature_col: str = "tsfresh_feature",
    plot_member_overlay_legend_fontsize: Optional[float] = None,
    clustering_phase_cmd: Optional[str] = None,
    clustering_phase_out_dir: Optional[str] = None,
    normalized_cka_cohen_cmd: Optional[str] = None,
    time_limit: str,
    queue: str,
    account: str,
    cpus_per_task: int,
    log_out: str,
    log_err: str,
    modules: list[str],
    environment: str,
) -> str:
    # Use this repo's analysis directory instead of a hardcoded external path.
    workdir = str(_ANALYSIS_ROOT)

    modules_lines: list[str] = []
    if environment == "nersc":
        for m in modules:
            if m == "conda":
                modules_lines.append("module load conda")
            else:
                modules_lines.append(f"module load {m}")

    th_parent = str(Path(out_csv).expanduser().parent)
    member_extra = ""
    if write_cluster_member_samples and cluster_members_parent_dir:
        member_extra = (
            "\n  --write-cluster-member-samples \\"
            + f"\n  --cluster-members-parent-dir {_bash_quote(str(Path(cluster_members_parent_dir).expanduser()))} \\"
            + f"\n  --member-sample-n {int(member_sample_n)} \\"
            + f"\n  --member-sample-seed {int(member_sample_seed)}"
        )
        if write_per_cluster_cohens_d_cka and th_parent:
            member_extra += " \\"
    per_cluster_extra = ""
    if write_per_cluster_cohens_d_cka and th_parent:
        per_cluster_extra = (
            "\n  --write-per-cluster-cohens-d-cka \\"
            + f"\n  --per-cluster-thresholds-dir {_bash_quote(str(Path(th_parent).expanduser()))} \\"
            + f"\n  --repr-prefix {_bash_quote(repr_prefix)}"
        )
    id_line = f"  --id-source-file {_bash_quote(id_source_file)}"
    if member_extra or per_cluster_extra or ip_suffix_filter:
        id_line += " \\"
    ip_suffix_block = ""
    if ip_suffix_filter:
        ip_suffix_block = f"\n  --ip-suffix-filter {_bash_quote(str(ip_suffix_filter))}"
        if member_extra or per_cluster_extra:
            ip_suffix_block += " \\"

    cluster_analysis_cmd = (
        "python3 cluster_analysis.py \\"
        + f"\n  --clustering {_bash_quote(clustering_csv)} \\"
        + f"\n  --tsfresh {_bash_quote(tsfresh_csv)} \\"
        + f"\n  --out {_bash_quote(out_csv)} \\"
        + f"\n  --min-cluster-size {int(min_cluster_size)} \\"
        + f"\n  --global-fisher-min {float(global_fisher_min)} \\"
        + f"\n  --cluster-variance-out {_bash_quote(cluster_variance_out_csv)} \\"
        + f"\n  --cluster-feature-rank-out {_bash_quote(cluster_feature_rank_out_csv)} \\"
        + f"\n  --id-ip {_bash_quote(id_ip)} \\"
        + f"\n{id_line}"
        + ip_suffix_block
        + member_extra
        + per_cluster_extra
    )
    overlay_slurm_lines: list[str] = []
    if global_filter_overlay_plots:
        if not cluster_members_parent_dir:
            raise ValueError("global_filter_overlay_plots requires cluster_members_parent_dir.")
        gfm = str(global_filter_mode).strip().lower()
        if gfm != "fisher_min":
            raise ValueError("global_filter_mode must be 'fisher_min'.")
        parent_o = _bash_quote(str(Path(cluster_members_parent_dir).expanduser()))
        core_lines = [
            "python3 global_filtered_overlay_prep.py \\",
            f"  --clustering-csv {_bash_quote(clustering_csv)} \\",
            f"  --tsfresh-csv {_bash_quote(tsfresh_csv)} \\",
            f"  --parent-out {parent_o} \\",
            f"  --global-filter-mode {_bash_quote(gfm)} \\",
            f"  --local-overlay-top-k {int(overlay_local_top_k)} \\",
            f"  --label-col {_bash_quote(str(label_col))} \\",
            f"  --id-ip {_bash_quote(id_ip)} \\",
            f"  --id-source-file {_bash_quote(id_source_file)} \\",
            f"  --cka-feature-col {_bash_quote(str(global_cka_feature_col))}",
        ]
        if gfm == "fisher_min":
            core_lines[-1] = core_lines[-1].replace(
                f"  --cka-feature-col {_bash_quote(str(global_cka_feature_col))}",
                f"  --global-fisher-min {float(global_fisher_min)} \\",
            )
            core_lines.append(f"  --cka-feature-col {_bash_quote(str(global_cka_feature_col))}")
        else:
            core_lines.append(f"  --cka-feature-col {_bash_quote(str(global_cka_feature_col))}")
        overlay_slurm_lines = core_lines + [""]
        if ip_suffix_filter:
            overlay_slurm_lines[-2] += " \\"
            overlay_slurm_lines.insert(-1, f"  --ip-suffix-filter {_bash_quote(str(ip_suffix_filter))}")

    if global_filter_overlay_plots and run_member_timeseries_plots:
        if not write_per_cluster_cohens_d_cka:
            raise ValueError(
                "global_filter_overlay_plots with run_member_timeseries_plots requires write_per_cluster_cohens_d_cka."
            )
        if plot_member_timeseries_fisher_overlay:
            raise ValueError(
                "Do not combine plot_member_timeseries_fisher_overlay with global_filter_overlay_plots."
            )

    plot_block: list[str] = []
    if run_member_timeseries_plots:
        if not write_cluster_member_samples or not cluster_members_parent_dir:
            raise ValueError(
                "run_member_timeseries_plots requires write_cluster_member_samples and cluster_members_parent_dir."
            )
        if not plot_member_timeseries_input_parquet:
            raise ValueError("run_member_timeseries_plots requires plot_member_timeseries_input_parquet.")
        parent_exp = str(Path(cluster_members_parent_dir).expanduser())
        glob_pat = str(Path(parent_exp) / "cluster_*" / "members.json")
        cfg_out = plot_member_timeseries_config_out or str(Path(parent_exp) / "plot_member_timeseries_config.json")
        wlines = [
            "python3 write_member_plot_config.py \\",
            f"  --out {_bash_quote(cfg_out)} \\",
            f"  --input-parquet {_bash_quote(str(plot_member_timeseries_input_parquet))} \\",
            f"  --members-json-glob {_bash_quote(glob_pat)} \\",
            f"  --key-columns {_bash_quote(plot_member_timeseries_key_columns)} \\",
            f"  --value-column {_bash_quote(plot_member_timeseries_value_column)} \\",
            f"  --num-workers {int(plot_member_timeseries_num_workers)}",
        ]
        if not plot_member_timeseries_y_floor_zero:
            wlines[-1] += " \\"
            wlines.append("  --no-y-floor-zero")
        if plot_member_timeseries_share_y_max_within_k:
            wlines[-1] += " \\"
            wlines.append("  --share-y-max-within-k")
        if plot_member_timeseries_y_fixed_max is not None:
            wlines[-1] += " \\"
            wlines.append(f"  --y-fixed-max {float(plot_member_timeseries_y_fixed_max)}")
        if plot_member_timeseries_fisher_overlay:
            tsf_o = plot_member_timeseries_tsfresh_csv or tsfresh_csv
            wlines[-1] += " \\"
            wlines.append("  --fisher-overlay \\")
            wlines.append(f"  --tsfresh-csv {_bash_quote(str(tsf_o))} \\")
            wlines.append(f"  --fisher-top-n {int(plot_member_timeseries_fisher_top_n)} \\")
            wlines.append(f"  --heatmap-filename {_bash_quote(str(plot_member_timeseries_heatmap_filename))}")
        elif global_filter_overlay_plots:
            tsf_o = plot_member_timeseries_tsfresh_csv or tsfresh_csv
            wlines[-1] += " \\"
            wlines.append("  --dual-global-filter-overlays \\")
            wlines.append(f"  --tsfresh-csv {_bash_quote(str(tsf_o))} \\")
            wlines.append(f"  --overlay-top-n {int(overlay_local_top_k)}")
            if plot_member_overlay_legend_fontsize is not None:
                wlines[-1] += " \\"
                wlines.append(f"  --overlay-legend-fontsize {float(plot_member_overlay_legend_fontsize)}")
        plot_block = wlines + [
            "",
            "python3 plot_member_timeseries_from_parquet.py \\",
            f"  --config {_bash_quote(cfg_out)}",
            "",
        ]

    return "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH -q {queue}",
            "#SBATCH -C cpu",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --account={account}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --output={log_out}",
            f"#SBATCH --error={log_err}",
            "",
            "set -euo pipefail",
            "",
            *modules_lines,
            "",
            f"cd {workdir}",
            "",
            f"mkdir -p { _bash_quote(str(Path(out_csv).expanduser().parent)) }",
            f"mkdir -p { _bash_quote(str(Path(cluster_variance_out_csv).expanduser().parent)) }",
            f"mkdir -p { _bash_quote(str(Path(cluster_feature_rank_out_csv).expanduser().parent)) }",
            *(
                [
                    f"mkdir -p { _bash_quote(str(Path(cluster_members_parent_dir).expanduser())) }",
                ]
                if write_cluster_member_samples and cluster_members_parent_dir
                else []
            ),
            *(
                [
                    f"mkdir -p { _bash_quote(str(Path(th_parent).expanduser())) }",
                ]
                if write_per_cluster_cohens_d_cka and th_parent
                else []
            ),
            *(
                [
                    f"mkdir -p { _bash_quote(str(Path(clustering_phase_out_dir).expanduser())) }",
                ]
                if clustering_phase_out_dir
                else []
            ),
            "",
            *( [clustering_phase_cmd, ""] if clustering_phase_cmd else [] ),
            "",
            cluster_analysis_cmd,
            "",
            *overlay_slurm_lines,
            "",
            *plot_block,
            "",
            *( [normalized_cka_cohen_cmd, ""] if normalized_cka_cohen_cmd else [] ),
        ]
    )


def _write_preview_slurm(*, slurm_text: str, tmp_dir: str, job_name: str) -> str:
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    slurm_path = str(Path(tmp_dir) / f"{job_name}_preview.slurm")
    Path(slurm_path).write_text(slurm_text, encoding="utf-8")
    return slurm_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run/submit cluster invariance from a JSON config.")
    ap.add_argument("--json", required=True, help="Path to JSON config.")
    ap.add_argument("--mode", choices=["local", "slurm", "preview"], default="slurm")
    ap.add_argument("--job-name", default=None, help="Override SLURM job name (optional).")
    ap.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory to write preview slurm file (preview mode). If omitted, uses a system temp dir.",
    )
    ap.add_argument(
        "--dependency",
        default=None,
        help="Slurm --dependency=... for the main job (merged with afterok:<id> if a precursor CKA job is submitted).",
    )
    args = ap.parse_args()

    cfg = _load_config(args.json)
    _apply_run_scoping(cfg)

    tsfresh_csv = _require_str(cfg, "tsfresh_csv")
    out_csv = _require_str(cfg, "out_csv")
    cluster_variance_out_csv = _require_str(cfg, "cluster_variance_out_csv")
    cluster_feature_rank_out_csv = _require_str(cfg, "cluster_feature_rank_out_csv")

    min_cluster_size = int(cfg.get("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE))
    id_ip = str(cfg.get("id_ip", DEFAULT_ID_IP))
    id_source_file = str(cfg.get("id_source_file", DEFAULT_ID_SOURCE_FILE))
    runner = _resolve_runner(mode=args.mode, cfg_runner=cfg.get("runner"))
    environment = _resolve_environment(cfg.get("environment"))

    write_cluster_member_samples = _parse_bool(
        cfg.get("write_cluster_member_samples"), default=DEFAULT_WRITE_CLUSTER_MEMBER_SAMPLES
    )
    cluster_members_parent_dir_raw = cfg.get("cluster_members_parent_dir")
    cluster_members_parent_dir = (
        str(cluster_members_parent_dir_raw).strip()
        if cluster_members_parent_dir_raw is not None and str(cluster_members_parent_dir_raw).strip() != ""
        else None
    )
    member_sample_n = int(cfg.get("member_sample_n", DEFAULT_MEMBER_SAMPLE_N))
    member_sample_seed = int(cfg.get("member_sample_seed", DEFAULT_MEMBER_SAMPLE_SEED))
    if write_cluster_member_samples and not cluster_members_parent_dir:
        raise ValueError(
            "write_cluster_member_samples=true requires non-empty JSON key 'cluster_members_parent_dir'."
        )

    write_per_cluster_cohens_d_cka = _parse_bool(
        cfg.get("write_per_cluster_cohens_d_cka"),
        default=DEFAULT_WRITE_PER_CLUSTER_COHENS_D_CKA,
    )
    run_normalized_cka_cohen = _parse_bool(
        cfg.get("run_normalized_cka_cohen"),
        default=DEFAULT_RUN_NORMALIZED_CKA_COHEN,
    )
    normalized_cka_cohen_top_k = int(
        cfg.get("normalized_cka_cohen_top_k", DEFAULT_NORMALIZED_CKA_COHEN_TOP_K)
    )
    repr_prefix = str(cfg.get("repr_prefix", DEFAULT_REPR_PREFIX))

    raw_run_member_plots = cfg.get("run_member_timeseries_plots")
    plot_member_timeseries_input_parquet = cfg.get("plot_member_timeseries_input_parquet")
    parquet_ok = bool(
        plot_member_timeseries_input_parquet and str(plot_member_timeseries_input_parquet).strip()
    )
    if raw_run_member_plots is None:
        # Opt-in: set run_member_timeseries_plots true explicitly (or set parquet + legacy sweep behavior).
        run_member_timeseries_plots = False
    else:
        run_member_timeseries_plots = _parse_bool(raw_run_member_plots, default=False)
        if run_member_timeseries_plots and not parquet_ok:
            raise ValueError(
                "run_member_timeseries_plots=true requires non-empty 'plot_member_timeseries_input_parquet'."
            )
    plot_member_timeseries_config_out = cfg.get("plot_member_timeseries_config_out")
    plot_member_timeseries_key_columns = str(cfg.get("plot_member_timeseries_key_columns", "ip,source_file"))
    plot_member_timeseries_value_column = str(cfg.get("plot_member_timeseries_value_column", "inbound"))
    plot_member_timeseries_num_workers = int(cfg.get("plot_member_timeseries_num_workers", 0))
    plot_member_timeseries_y_floor_zero = _parse_bool(
        cfg.get("plot_member_timeseries_y_floor_zero"), default=True
    )
    _yfm = cfg.get("plot_member_timeseries_y_fixed_max")
    plot_member_timeseries_y_fixed_max: Optional[float] = None
    if _yfm is not None and str(_yfm).strip() != "":
        try:
            plot_member_timeseries_y_fixed_max = float(_yfm)
        except (TypeError, ValueError):
            plot_member_timeseries_y_fixed_max = None
    if plot_member_timeseries_y_fixed_max is not None and (
        not math.isfinite(plot_member_timeseries_y_fixed_max) or plot_member_timeseries_y_fixed_max <= 0
    ):
        plot_member_timeseries_y_fixed_max = None
    plot_member_timeseries_share_y_max_within_k = _parse_bool(
        cfg.get("plot_member_timeseries_share_y_max_within_k"), default=False
    )
    plot_member_timeseries_fisher_overlay = _parse_bool(
        cfg.get("plot_member_timeseries_fisher_overlay"), default=False
    )
    plot_member_timeseries_tsfresh_csv = cfg.get("plot_member_timeseries_tsfresh_csv")
    plot_member_timeseries_fisher_top_n = int(cfg.get("plot_member_timeseries_fisher_top_n", 5))
    plot_member_timeseries_heatmap_filename = str(
        cfg.get("plot_member_timeseries_heatmap_filename", "feature_order_fisher.csv")
    )
    global_filter_overlay_plots = _parse_bool(cfg.get("global_filter_overlay_plots"), default=False)
    global_filter_mode = str(cfg.get("global_filter_mode", "fisher_min")).strip().lower()
    global_fisher_min = float(cfg.get("global_fisher_min", 0.1))
    overlay_local_top_k = int(cfg.get("overlay_local_top_k", 10))
    label_col = str(cfg.get("label_col", "label"))
    global_cka_feature_col = str(cfg.get("global_cka_feature_col", "tsfresh_feature"))
    _polf = cfg.get("plot_member_overlay_legend_fontsize")
    plot_member_overlay_legend_fontsize: Optional[float] = None
    if _polf is not None and str(_polf).strip() != "":
        try:
            plot_member_overlay_legend_fontsize = float(_polf)
        except (TypeError, ValueError):
            plot_member_overlay_legend_fontsize = None
    if global_filter_overlay_plots:
        if global_filter_mode != "fisher_min":
            raise ValueError("global_filter_mode must be 'fisher_min'.")
    if global_filter_overlay_plots and not write_per_cluster_cohens_d_cka:
        raise ValueError(
            "global_filter_overlay_plots requires write_per_cluster_cohens_d_cka."
        )
    if run_normalized_cka_cohen and not write_per_cluster_cohens_d_cka:
        raise ValueError(
            "run_normalized_cka_cohen=true requires write_per_cluster_cohens_d_cka=true."
        )
    ip_suffix_filter = cfg.get("ip_suffix_filter", "/32")
    if ip_suffix_filter is not None:
        ip_suffix_filter = str(ip_suffix_filter).strip() or None
    clustering_csv, clustering_phase_cmd, clustering_phase_out_dir = _build_clustering_phase_command(
        cfg=cfg,
        out_csv=out_csv,
        default_ip_suffix_filter=ip_suffix_filter,
    )
    if not clustering_csv:
        raise ValueError(
            "Internal error resolving clustering labels. Provide clustering_csv or run_clustering_phase=true."
        )

    modules = cfg.get("modules", DEFAULT_MODULES)
    if not isinstance(modules, list):
        modules = DEFAULT_MODULES
    modules_str = [str(x) for x in modules]

    effective_sbatch_dependency = merge_sbatch_dependency(None, args.dependency)

    job_name = args.job_name or str(cfg.get("job_name", DEFAULT_JOB_NAME))

    sbatch_cfg = cfg.get("sbatch", {})
    if not isinstance(sbatch_cfg, dict):
        sbatch_cfg = {}
    queue = str(sbatch_cfg.get("queue", DEFAULT_SQUEUE))
    account = str(sbatch_cfg.get("account", DEFAULT_ACCOUNT))
    time_limit = str(sbatch_cfg.get("time", DEFAULT_TIME))
    cpus_per_task = int(sbatch_cfg.get("cpus_per_task", DEFAULT_CPUS_PER_TASK))

    if args.mode == "preview":
        runner = "slurm"

    if runner == "local":
        Path(Path(out_csv).expanduser().parent).mkdir(parents=True, exist_ok=True)
        if clustering_phase_out_dir:
            Path(clustering_phase_out_dir).expanduser().mkdir(parents=True, exist_ok=True)
        if write_cluster_member_samples and cluster_members_parent_dir:
            Path(cluster_members_parent_dir).expanduser().mkdir(parents=True, exist_ok=True)
        th_parent = str(Path(out_csv).expanduser().parent)
        if write_per_cluster_cohens_d_cka and th_parent:
            Path(th_parent).expanduser().mkdir(parents=True, exist_ok=True)
        if clustering_phase_cmd:
            _run_bash(clustering_phase_cmd)
        cmd1, cmd2 = _build_local_cmds(
            clustering_csv=clustering_csv,
            tsfresh_csv=tsfresh_csv,
            out_csv=out_csv,
            cluster_variance_out_csv=cluster_variance_out_csv,
            cluster_feature_rank_out_csv=cluster_feature_rank_out_csv,
            min_cluster_size=min_cluster_size,
            id_ip=id_ip,
            id_source_file=id_source_file,
            write_cluster_member_samples=write_cluster_member_samples,
            cluster_members_parent_dir=cluster_members_parent_dir,
            member_sample_n=member_sample_n,
            member_sample_seed=member_sample_seed,
            write_per_cluster_cohens_d_cka=write_per_cluster_cohens_d_cka,
            repr_prefix=repr_prefix,
            run_member_timeseries_plots=run_member_timeseries_plots,
            plot_member_timeseries_input_parquet=str(plot_member_timeseries_input_parquet)
            if plot_member_timeseries_input_parquet
            else None,
            plot_member_timeseries_config_out=str(plot_member_timeseries_config_out)
            if plot_member_timeseries_config_out
            else None,
            plot_member_timeseries_key_columns=plot_member_timeseries_key_columns,
            plot_member_timeseries_value_column=plot_member_timeseries_value_column,
            plot_member_timeseries_num_workers=plot_member_timeseries_num_workers,
            plot_member_timeseries_y_floor_zero=plot_member_timeseries_y_floor_zero,
            plot_member_timeseries_y_fixed_max=plot_member_timeseries_y_fixed_max,
            plot_member_timeseries_share_y_max_within_k=plot_member_timeseries_share_y_max_within_k,
            plot_member_timeseries_fisher_overlay=plot_member_timeseries_fisher_overlay,
            plot_member_timeseries_tsfresh_csv=str(plot_member_timeseries_tsfresh_csv)
            if plot_member_timeseries_tsfresh_csv
            else None,
            plot_member_timeseries_fisher_top_n=plot_member_timeseries_fisher_top_n,
            plot_member_timeseries_heatmap_filename=plot_member_timeseries_heatmap_filename,
            ip_suffix_filter=ip_suffix_filter,
            global_filter_overlay_plots=global_filter_overlay_plots,
            global_filter_mode=global_filter_mode,
            global_fisher_min=global_fisher_min,
            overlay_local_top_k=overlay_local_top_k,
            label_col=label_col,
            global_cka_feature_col=global_cka_feature_col,
            plot_member_overlay_legend_fontsize=plot_member_overlay_legend_fontsize,
        )
        Path(Path(cluster_variance_out_csv).expanduser().parent).mkdir(parents=True, exist_ok=True)
        Path(Path(cluster_feature_rank_out_csv).expanduser().parent).mkdir(parents=True, exist_ok=True)
        _run_bash(cmd1)
        if cmd2:
            _run_bash(cmd2)
        if run_normalized_cka_cohen:
            _run_bash(
                _build_normalized_cka_cohen_bash(
                    out_csv=out_csv,
                    top_k=normalized_cka_cohen_top_k,
                    quote_fn=shlex.quote,
                )
            )
        return

    out_dir, log_out, log_err = _derive_log_paths(out_csv, job_name)
    normalized_cka_cohen_cmd = None
    if run_normalized_cka_cohen:
        normalized_cka_cohen_cmd = _build_normalized_cka_cohen_bash(
            out_csv=out_csv,
            top_k=normalized_cka_cohen_top_k,
            quote_fn=_bash_quote,
        )
    slurm_text = _build_slurm_script(
        job_name=job_name,
        clustering_csv=clustering_csv,
        tsfresh_csv=tsfresh_csv,
        out_csv=out_csv,
        cluster_variance_out_csv=cluster_variance_out_csv,
        cluster_feature_rank_out_csv=cluster_feature_rank_out_csv,
        min_cluster_size=min_cluster_size,
        id_ip=id_ip,
        id_source_file=id_source_file,
        write_cluster_member_samples=write_cluster_member_samples,
        cluster_members_parent_dir=cluster_members_parent_dir,
        member_sample_n=member_sample_n,
        member_sample_seed=member_sample_seed,
        write_per_cluster_cohens_d_cka=write_per_cluster_cohens_d_cka,
        repr_prefix=repr_prefix,
        run_member_timeseries_plots=run_member_timeseries_plots,
        plot_member_timeseries_input_parquet=str(plot_member_timeseries_input_parquet)
        if plot_member_timeseries_input_parquet
        else None,
        plot_member_timeseries_config_out=str(plot_member_timeseries_config_out)
        if plot_member_timeseries_config_out
        else None,
        plot_member_timeseries_key_columns=plot_member_timeseries_key_columns,
        plot_member_timeseries_value_column=plot_member_timeseries_value_column,
        plot_member_timeseries_num_workers=plot_member_timeseries_num_workers,
        plot_member_timeseries_y_floor_zero=plot_member_timeseries_y_floor_zero,
        plot_member_timeseries_y_fixed_max=plot_member_timeseries_y_fixed_max,
        plot_member_timeseries_share_y_max_within_k=plot_member_timeseries_share_y_max_within_k,
        plot_member_timeseries_fisher_overlay=plot_member_timeseries_fisher_overlay,
        plot_member_timeseries_tsfresh_csv=str(plot_member_timeseries_tsfresh_csv)
        if plot_member_timeseries_tsfresh_csv
        else None,
        plot_member_timeseries_fisher_top_n=plot_member_timeseries_fisher_top_n,
        plot_member_timeseries_heatmap_filename=plot_member_timeseries_heatmap_filename,
        ip_suffix_filter=ip_suffix_filter,
        global_filter_overlay_plots=global_filter_overlay_plots,
        global_filter_mode=global_filter_mode,
        global_fisher_min=global_fisher_min,
        overlay_local_top_k=overlay_local_top_k,
        label_col=label_col,
        global_cka_feature_col=global_cka_feature_col,
        plot_member_overlay_legend_fontsize=plot_member_overlay_legend_fontsize,
        clustering_phase_cmd=clustering_phase_cmd,
        clustering_phase_out_dir=clustering_phase_out_dir,
        normalized_cka_cohen_cmd=normalized_cka_cohen_cmd,
        time_limit=time_limit,
        queue=queue,
        account=account,
        cpus_per_task=cpus_per_task,
        log_out=log_out,
        log_err=log_err,
        modules=modules_str,
        environment=environment,
    )

    if args.mode == "preview":
        tmp_dir = args.tmp_dir or tempfile.gettempdir()
        slurm_path = _write_preview_slurm(slurm_text=slurm_text, tmp_dir=tmp_dir, job_name=job_name)
        print(f"Preview slurm written to: {slurm_path}")
        return

    # slurm mode: generate a temporary file, submit, and remove.
    with tempfile.TemporaryDirectory(prefix=f"{job_name}_") as td:
        slurm_path = _write_preview_slurm(slurm_text=slurm_text, tmp_dir=td, job_name=job_name)
        sbatch_cmd = ["sbatch"]
        if effective_sbatch_dependency:
            sbatch_cmd.extend(["--dependency", effective_sbatch_dependency])
        sbatch_cmd.append(slurm_path)
        subprocess.run(sbatch_cmd, check=True)


if __name__ == "__main__":
    main()

