#!/usr/bin/env python3

"""
MPI clustering runner with per-rank parameter assignment.

Supported methods:
- KMeans: distribute over a list of --n-clusters values; compute silhouette.
- OPTICS: distribute over a list of --min-samples values; compute silhouette on
                    non-noise clusters (labels >= 0) with cluster size >= 2; write labels including noise.

Key behavior:
- Root (rank 0) reads the CSV/PKL once and selects features (repr_dim_* or all feature cols from PKL).
- Root broadcasts the feature array and metadata (ip, source_file) to all ranks using mpi4py.
- Each MPI rank performs at most one clustering run for a single parameter value (indexed by rank).
- No sampling, PCA, scaling, or other optimizations: silhouette is computed on the data directly.
- Rank 0 writes a scores file and per-parameter CSVs containing labels, ip, source_file, and feature columns.
"""

import argparse
import os
import sys
import warnings
from typing import List, Optional, Tuple

import numpy as np

# Cap thread usage before importing numpy/pandas/sklearn to honor library limits
# We respect NUMEXPR_MAX_THREADS (often 64) and common OpenBLAS/MKL limits.
try:
    # Determine a safe per-rank thread cap (default 64; never exceed NUMEXPR_MAX_THREADS if set)
    env_cap = int(os.environ.get("OMP_NUM_THREADS", "64"))
    numexpr_max = int(os.environ.get("NUMEXPR_MAX_THREADS", "64"))
    max_threads = max(1, min(env_cap, 64, numexpr_max))
except Exception:
    max_threads = 64

# Export consistent limits for all major math backends
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(max_threads))
os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
os.environ["OMP_NUM_THREADS"] = str(max_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
os.environ["GOTO_NUM_THREADS"] = str(max_threads)
os.environ["MKL_NUM_THREADS"] = str(max_threads)
os.environ["BLIS_NUM_THREADS"] = str(max_threads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(max_threads))

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="MPI clustering (KMeans or OPTICS) with silhouette scoring")
    p.add_argument(
        "data_path",
        type=str,
        help=(
            "Path to CSV (repr_dim_* + ip,source_file) or PKL DataFrame "
            "(col 0=ip, col 1=source_file, col 2..=features)"
        ),
    )
    p.add_argument(
        "--method",
        type=str,
        choices=["kmeans", "optics"],
        default="kmeans",
        help="Clustering method to use",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        nargs="+",
        help="KMeans only: list of K values (one per rank).",
    )
    p.add_argument(
        "--kmeans-metric",
        type=str,
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="KMeans: metric for clustering ('euclidean' default, or 'cosine' via spherical k-means with row L2 normalization)",
    )
    p.add_argument("--random-state", type=int, default=42, help="Random seed for KMeans")
    p.add_argument("--out-dir", type=str, default=".", help="Directory to write results")
    # OPTICS-specific parameters
    p.add_argument(
        "--min-samples",
        type=int,
        nargs="+",
        help="OPTICS only: list of min_samples values (one per rank). Defaults to [5] if not provided.",
    )
    p.add_argument("--xi", type=float, default=0.05, help="OPTICS: xi parameter for cluster extraction")
    p.add_argument(
        "--min-cluster-size",
        type=float,
        default=None,
        help="OPTICS: minimum cluster size (int or float fraction). Use None to disable.",
    )
    p.add_argument(
        "--cluster-method",
        type=str,
        choices=["xi", "dbscan"],
        default="xi",
        help="OPTICS: cluster extraction method",
    )
    p.add_argument("--metric", type=str, default="minkowski", help="OPTICS: distance metric")
    p.add_argument("--max-eps", type=float, default=np.inf, help="OPTICS: maximum epsilon")
    p.add_argument(
        "--use-all-features",
        action="store_true",
        help=(
            "CSV only: treat all columns except 'ip' and 'source_file' as features. "
            "By default, only columns with prefix 'repr_dim_' are used."
        ),
    )
    p.add_argument(
        "--minimal-cleaning",
        action="store_true",
        help=(
            "CSV only: use minimal feature cleaning (coerce to numeric and replace NaNs/Infs), "
            "without dropping >10% NaN columns, zero-variance columns, or all-zero rows. "
            "By default, stricter cleaning is applied."
        ),
    )
    p.add_argument(
        "--ip-suffix-filter",
        type=str,
        default=None,
        help=(
            "If set, keep only rows whose ip string ends with this suffix (e.g. /32 for single-host CIDRs)."
        ),
    )
    args = p.parse_args()

    # Conditional validation
    if args.method == "kmeans":
        if not args.n_clusters:
            p.error("--n-clusters is required when --method kmeans")
    else:  # optics
        if not args.min_samples:
            args.min_samples = [5]
    return args


args = parse_args()

import pandas as pd
from sklearn.cluster import KMeans, OPTICS

# As an extra guard, attempt to limit threadpools at runtime
try:
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=max_threads)
except Exception:
    pass

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
except Exception:
    comm = None
    rank = 0
    world = 1


def _log(msg: str) -> None:
    try:
        sys.stdout.write(f"[rank {rank}/{world}] {msg}\n")
        sys.stdout.flush()
    except Exception:
        pass


def _select_repr_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("repr_dim_")]
    if not cols:
        raise ValueError("No columns starting with 'repr_dim_' found in the CSV.")
    return cols


def _load_features_and_meta_from_pkl(path: str) -> Tuple[np.ndarray, List[str], pd.Series, pd.Series]:
    """Load pandas DataFrame from PKL.

    Expected schema by position:
    - col 0: ip (any type; cast to str)
    - col 1: source_file (any type; cast to str)
    - col 2..: feature columns (treated as repr_dim_*)
    """
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Pickle must contain a pandas DataFrame.")
    if df.shape[1] < 3:
        raise ValueError("Pickle DataFrame must have at least 3 columns (ip, source_file, features...).")

    ips = df.iloc[:, 0].astype(str)
    source_files = df.iloc[:, 1].astype(str)
    if args.ip_suffix_filter:
        suf = str(args.ip_suffix_filter)
        mask = ips.str.endswith(suf)
        n0 = len(df)
        df = df.loc[mask].reset_index(drop=True)
        ips = df.iloc[:, 0].astype(str)
        source_files = df.iloc[:, 1].astype(str)
        _log(f"ip_suffix_filter={suf!r}: kept {len(df)}/{n0} PKL rows")
        if len(df) == 0:
            raise ValueError(f"ip_suffix_filter={suf!r} removed all rows from PKL.")
    # Coerce to numeric and drop columns that are entirely NaN
    X_df = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    all_nan_mask = X_df.isna().all(axis=0)
    if bool(all_nan_mask.any()):
        dropped = int(all_nan_mask.sum())
        X_df = X_df.loc[:, ~all_nan_mask]
        _log(f"Dropped {dropped} all-NaN feature columns (PKL)")
    # Drop columns with >10% NaNs
    nan_frac = X_df.isna().mean(axis=0)
    high_nan_mask = nan_frac > 0.10
    if bool(high_nan_mask.any()):
        dropped = int(high_nan_mask.sum())
        X_df = X_df.loc[:, ~high_nan_mask]
        _log(f"Dropped {dropped} >10% NaN feature columns (PKL)")
    # Sanitize for variance check
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop zero-variance columns
    zero_var_mask = X_df.std(ddof=0) == 0
    if bool(zero_var_mask.any()):
        removed_cols = X_df.columns[zero_var_mask].tolist()
        _log(f"Dropped {len(removed_cols)} zero-variance feature columns (PKL)")
        X_df = X_df.loc[:, ~zero_var_mask]
    _log(f"Feature columns after cleaning (PKL): {X_df.shape[1]}")
    X = X_df.to_numpy(copy=False)
    # sanitize numeric array
    if np.any(~np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    # Drop rows that are all zeros (after NaN fill)
    row_mask = (X != 0).any(axis=1)
    if bool((~row_mask).any()):
        removed = int((~row_mask).sum())
        _log(f"Removed {removed} all-zero rows after NaN fill (PKL)")
        X = X[row_mask]
        ips = ips[row_mask].reset_index(drop=True)
        source_files = source_files[row_mask].reset_index(drop=True)
    feat_cols = [f"repr_dim_{i}" for i in range(X.shape[1])]
    # Diagnostics: rows and approximate distinct rows
    try:
        distinct_rows = int(pd.DataFrame(X).drop_duplicates().shape[0])
        _log(f"Loaded PKL: samples={X.shape[0]}, features={X.shape[1]}, distinct_rows~={distinct_rows}")
    except Exception:
        pass
    return X.astype(np.float32, copy=False), feat_cols, ips, source_files


def _load_features_and_meta(path: str) -> Tuple[np.ndarray, List[str], pd.Series, pd.Series]:
    """Load features and metadata from CSV or PKL.

    - CSV: expects repr_dim_* columns and explicit 'ip' and 'source_file' columns.
    - PKL: expects DataFrame with col 0=ip, col 1=source_file, col 2..=features.
    """
    lower = path.lower()
    if lower.endswith(".pkl") or lower.endswith(".pickle"):
        return _load_features_and_meta_from_pkl(path)

    # CSV path
    df = pd.read_csv(path)
    # Validate presence of metadata columns
    # Keep ip and source_file for output
    if "ip" not in df.columns or "source_file" not in df.columns:
        raise ValueError("CSV must contain 'ip' and 'source_file' columns")
    if args.ip_suffix_filter:
        suf = str(args.ip_suffix_filter)
        mask = df["ip"].astype(str).str.endswith(suf)
        n0 = len(df)
        df = df.loc[mask].reset_index(drop=True)
        _log(f"ip_suffix_filter={suf!r}: kept {len(df)}/{n0} CSV rows")
        if len(df) == 0:
            raise ValueError(f"ip_suffix_filter={suf!r} removed all rows from CSV.")
    # Feature column selection
    if args.use_all_features:
        feat_cols = [c for c in df.columns if c not in ["ip", "source_file"]]
        if not feat_cols:
            raise ValueError("No feature columns found when excluding 'ip' and 'source_file'.")
    else:
        feat_cols = _select_repr_columns(df)
    ips = df["ip"].astype(str)
    source_files = df["source_file"].astype(str)
    # Ensure numeric features; coerce non-numeric to NaN
    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    if args.minimal_cleaning:
        # Minimal cleaning: just coerce, replace NaN/Inf, and keep all rows/columns
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X_df.to_numpy(copy=False)
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        try:
            distinct_rows = int(pd.DataFrame(X).drop_duplicates().shape[0])
            _log(
                f"Loaded CSV (minimal cleaning): samples={X.shape[0]}, features={X.shape[1]}, distinct_rows~={distinct_rows}"
            )
        except Exception:
            pass
        return X.astype(np.float32, copy=False), feat_cols, ips, source_files

    # Default: stricter cleaning, as before
    # Drop columns that are entirely NaN to avoid normalization issues
    all_nan_mask = X_df.isna().all(axis=0)
    if bool(all_nan_mask.any()):
        dropped = int(all_nan_mask.sum())
        feat_cols = [c for c in feat_cols if c in X_df.columns and not all_nan_mask[c]]
        X_df = X_df[feat_cols]
        _log(f"Dropped {dropped} all-NaN feature columns (CSV)")
    # Drop columns with >10% NaNs
    nan_frac = X_df.isna().mean(axis=0)
    high_nan_mask = nan_frac > 0.10
    if bool(high_nan_mask.any()):
        dropped = int(high_nan_mask.sum())
        keep_cols = [c for c in X_df.columns if not high_nan_mask[c]]
        X_df = X_df[keep_cols]
        feat_cols = X_df.columns.tolist()
        _log(f"Dropped {dropped} >10% NaN feature columns (CSV)")
        if not feat_cols:
            raise ValueError("No feature columns remain after >10% NaN filtering.")
    # Sanitize for variance check
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Drop zero-variance columns
    zero_var_mask = X_df.std(ddof=0) == 0
    if bool(zero_var_mask.any()):
        removed_cols = X_df.columns[zero_var_mask].tolist()
        _log(f"Dropped {len(removed_cols)} zero-variance feature columns (CSV)")
        X_df = X_df.loc[:, ~zero_var_mask]
        feat_cols = X_df.columns.tolist()
    _log(f"Feature columns after cleaning (CSV): {X_df.shape[1]}")
    X = X_df.to_numpy(copy=False)
    # sanitize
    if np.any(~np.isfinite(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    # Drop rows that are all zeros (after NaN fill)
    row_mask = (X != 0).any(axis=1)
    if bool((~row_mask).any()):
        removed = int((~row_mask).sum())
        _log(f"Removed {removed} all-zero rows after NaN fill (CSV)")
        X = X[row_mask]
        ips = ips.loc[row_mask].reset_index(drop=True)
        source_files = source_files.loc[row_mask].reset_index(drop=True)
    # Diagnostics: rows and approximate distinct rows
    try:
        distinct_rows = int(pd.DataFrame(X).drop_duplicates().shape[0])
        _log(f"Loaded CSV: samples={X.shape[0]}, features={X.shape[1]}, distinct_rows~={distinct_rows}")
    except Exception:
        pass
    return X.astype(np.float32, copy=False), feat_cols, ips, source_files


def _broadcast_numpy_array_and_meta(
    X: Optional[np.ndarray], feat_cols: Optional[List[str]], ips: Optional[pd.Series], sources: Optional[pd.Series]
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Broadcast numpy array X (float32/64) and metadata columns to all ranks.

    Returns: X, feat_cols, ips_list, sources_list (ips and sources as lists of strings)
    """
    if world == 1:
        return X, feat_cols, ips.tolist() if ips is not None else [], sources.tolist() if sources is not None else []

    # Broadcast shape and dtype information from root
    if rank == 0:
        shape = tuple(X.shape)
        dtype_name = str(X.dtype)
        feat_cols_local = feat_cols
        ips_local = ips.tolist()
        sources_local = sources.tolist()
    else:
        shape = None
        dtype_name = None
        feat_cols_local = None
        ips_local = None
        sources_local = None

    shape = comm.bcast(shape, root=0)
    dtype_name = comm.bcast(dtype_name, root=0)
    feat_cols_local = comm.bcast(feat_cols_local, root=0)
    ips_local = comm.bcast(ips_local, root=0)
    sources_local = comm.bcast(sources_local, root=0)

    dtype = np.dtype(dtype_name)
    if rank != 0:
        X = np.empty(shape, dtype=dtype)

    # Broadcast raw data buffer
    if dtype == np.float32:
        comm.Bcast([X, MPI.FLOAT], root=0)
    elif dtype == np.float64:
        comm.Bcast([X, MPI.DOUBLE], root=0)
    else:
        # fallback: treat as bytes
        buf = X.view(np.uint8)
        comm.Bcast([buf, MPI.BYTE], root=0)

    return X, feat_cols_local, ips_local, sources_local


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # avoid division by zero; keep zero rows unchanged
    norms[norms == 0] = 1.0
    return X / norms


def _run_kmeans(X: np.ndarray, k: int, random_state: int, metric: str) -> np.ndarray:
    if k < 2 or k >= X.shape[0]:
        raise ValueError(f"Invalid n_clusters={k} for n_samples={X.shape[0]}")
    X_fit = _l2_normalize_rows(X) if metric == "cosine" else X
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return km.fit_predict(X_fit)


def _run_optics(
    X: np.ndarray,
    min_samples: int,
    xi: float,
    min_cluster_size: Optional[float],
    cluster_method: str,
    metric: str,
    max_eps: float,
) -> np.ndarray:
    optics = OPTICS(
        min_samples=int(min_samples),
        xi=xi,
        min_cluster_size=min_cluster_size,
        cluster_method=cluster_method,
        metric=metric,
        max_eps=max_eps,
        n_jobs=None,
    )
    return optics.fit_predict(X)


def _slurm_step_num_tasks() -> int:
    v = os.environ.get("SLURM_STEP_NUM_TASKS")
    if v is not None and str(v).strip().isdigit():
        try:
            return max(1, int(v))
        except ValueError:
            pass
    return 1


def _slurm_step_task_index() -> int:
    """Index of this process in the current srun step (0 .. step-1), or -1 if unknown."""
    for key in ("SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(key)
        if v is None or v == "":
            continue
        try:
            return int(v)
        except ValueError:
            continue
    return -1


def main():
    global comm, rank, world
    # Slurm `srun --multi-prog` (and similar) can place unrelated tasks in one MPI job so
    # COMM_WORLD has one rank per node (global ranks 0..N-1). For a single --n-clusters or
    # single min_samples value, the script only keeps rank 0 after trim, so only global
    # rank 0 would write labels; other nodes exit without producing CSVs. Use COMM_SELF so
    # each process is a standalone rank-0 world for single-parameter runs.
    if comm is not None and world > 1:
        single_param = False
        if args.method == "kmeans" and args.n_clusters and len(args.n_clusters) == 1:
            single_param = True
        elif args.method == "optics" and args.min_samples and len(args.min_samples) == 1:
            single_param = True
        if single_param:
            from mpi4py import MPI as _MPI

            comm = _MPI.COMM_SELF
            rank = comm.Get_rank()
            world = comm.Get_size()

    # Root reads the CSV once and prepares numpy array and metadata
    if rank == 0:
        _log("Loading data on root")
        X_root, feat_cols, ips, sources = _load_features_and_meta(args.data_path)
    else:
        X_root = None
        feat_cols = None
        ips = None
        sources = None

    # Broadcast X and metadata
    _log("Broadcasting data and metadata")
    X, feat_cols, ips_list, sources_list = _broadcast_numpy_array_and_meta(X_root, feat_cols, ips, sources)

    # Broadcast parameter list based on method
    if args.method == "kmeans":
        if world > 1:
            if rank == 0:
                param_list = list(args.n_clusters)
            else:
                param_list = None
            param_list = comm.bcast(param_list, root=0)
        else:
            param_list = list(args.n_clusters)
    else:  # optics
        if world > 1:
            if rank == 0:
                param_list = list(args.min_samples)
            else:
                param_list = None
            param_list = comm.bcast(param_list, root=0)
        else:
            param_list = list(args.min_samples)

    # If more MPI ranks than parameters, trim to only the first len(param_list) ranks
    # so idle ranks don't spam logs or block.
    active_world = min(world, len(param_list))
    if comm is not None and world > active_world:
        try:
            # Update global communicator, rank, and world to a reduced communicator
            from mpi4py import MPI as _MPI

            color = 0 if rank < active_world else _MPI.UNDEFINED
            new_comm = comm.Split(color=color, key=rank)
            if rank >= active_world:
                # Non-active ranks exit main quietly
                return
            comm = new_comm
            rank = comm.Get_rank()
            world = comm.Get_size()
            _log(f"Trimmed active ranks to {world} to match parameters ({len(param_list)})")
        except Exception:
            # If communicator split fails, fall back to original behavior
            pass

    # Assign one parameter per worker:
    # - world>1: use MPI rank (normal mpi4py + Slurm PMI).
    # - world==1 but SLURM_STEP_NUM_TASKS>1: mpi4py often sees singleton worlds; use Slurm task id
    #   (SLURM_PROCID / PMI_RANK) so each srun task gets one K.
    # - world==1 and a single-task step: run all parameters sequentially on that process.
    step_n = _slurm_step_num_tasks()
    slurm_tid = _slurm_step_task_index()

    if world > 1:
        if rank < len(param_list):
            params_for_rank = [param_list[rank]]
        else:
            params_for_rank = []
    elif world == 1 and len(param_list) > 1:
        if step_n > 1 and slurm_tid >= 0:
            if slurm_tid < len(param_list):
                params_for_rank = [param_list[slurm_tid]]
                _log(
                    f"MPI world size=1; using Slurm step task id={slurm_tid} (step tasks={step_n}) "
                    f"for one parameter"
                )
            else:
                params_for_rank = []
        elif step_n > 1 and slurm_tid < 0:
            _log(
                "WARN: multi-task srun but SLURM_PROCID/PMI_RANK not set; "
                "running all parameters on this task (duplicate work if multiple tasks)."
            )
            params_for_rank = list(param_list)
        else:
            params_for_rank = list(param_list)
            _log(f"MPI world size=1; running all {len(param_list)} parameters sequentially on this task")
    elif rank < len(param_list):
        params_for_rank = [param_list[rank]]
    else:
        params_for_rank = []

    # Note: silhouette scoring intentionally disabled. Each rank writes its own labels CSVs.
    for my_param in params_for_rank:
        try:
            # Ensure out dir exists per-rank for immediate writes
            try:
                os.makedirs(args.out_dir, exist_ok=True)
            except Exception:
                pass

            if args.method == "kmeans":
                _log(f"Starting KMeans for k={int(my_param)} (metric={args.kmeans_metric})")
                # Use a consistent seed across all k values (deterministic init)
                labels = _run_kmeans(X, int(my_param), args.random_state, args.kmeans_metric)
                # Early write: labels right after clustering
                out_df = pd.DataFrame(
                    {
                        "ip": ips_list,
                        "source_file": sources_list,
                        "label": labels.tolist(),
                    }
                )
                feat_arr = X if isinstance(X, np.ndarray) else np.array(X)
                for i, col in enumerate(feat_cols):
                    out_df[col] = feat_arr[:, i].tolist()
                out_file = os.path.join(args.out_dir, f"k_{int(my_param)}_labels.csv")
                out_df.to_csv(out_file, index=False)
                _log(f"Wrote labels to {out_file}")
                uniq, cnts = np.unique(labels, return_counts=True)
                _log(f"Cluster label distribution: {dict(zip(uniq.tolist(), cnts.tolist()))}")
                # (Silhouette scoring intentionally disabled)

                # Also perform clustering on per-row L2 norms (1D)
                _log("Computing L2-norm-based clustering")
                # Compute per-row L2 norms with robust sanitization
                X_norm = np.linalg.norm(X.astype(np.float64, copy=False), axis=1)
                inf_before = int(np.isinf(X_norm).sum())
                if inf_before:
                    _log(f"L2-norm had {inf_before} inf values before sanitization")
                X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1e12, neginf=0.0)
                X_norm = np.clip(X_norm, 0.0, 1e12)
                X_norm = X_norm.astype(np.float32, copy=False).reshape(-1, 1)
                labels_norm = _run_kmeans(
                    X_norm, int(my_param), args.random_state, "euclidean"
                )
                uniq_n, cnts_n = np.unique(labels_norm, return_counts=True)
                _log(f"L2-norm KMeans label distribution: {dict(zip(uniq_n.tolist(), cnts_n.tolist()))}")
                # Write L2-norm labels CSV for KMeans
                try:
                    out_df_norm = pd.DataFrame(
                        {
                            "ip": ips_list,
                            "source_file": sources_list,
                            "l2_norm": X_norm.reshape(-1).tolist(),
                            "label": labels_norm.tolist(),
                        }
                    )
                    out_file_norm = os.path.join(args.out_dir, f"k_{int(my_param)}_labels_l2norm.csv")
                    out_df_norm.to_csv(out_file_norm, index=False)
                    _log(f"Wrote L2-norm labels to {out_file_norm}")
                except Exception as e:
                    _log(f"Failed writing L2-norm labels CSV: {e}")
            else:
                _log(f"Starting OPTICS for min_samples={int(my_param)}")
                labels = _run_optics(
                    X,
                    int(my_param),
                    args.xi,
                    args.min_cluster_size,
                    args.cluster_method,
                    args.metric,
                    args.max_eps,
                )
                # Early write: labels right after clustering
                out_df = pd.DataFrame(
                    {
                        "ip": ips_list,
                        "source_file": sources_list,
                        "label": labels.tolist(),
                    }
                )
                _log("Writing feature columns to output")
                feat_arr = X if isinstance(X, np.ndarray) else np.array(X)
                for i, col in enumerate(feat_cols):
                    out_df[col] = feat_arr[:, i].tolist()
                out_file = os.path.join(args.out_dir, f"optics_minSamples_{int(my_param)}_labels.csv")
                out_df.to_csv(out_file, index=False)
                _log(f"Wrote labels to {out_file}")

                # Also perform OPTICS on per-row L2 norms (1D)
                _log("Computing L2-norm-based OPTICS")
                X_norm = np.linalg.norm(X.astype(np.float64, copy=False), axis=1)
                inf_before = int(np.isinf(X_norm).sum())
                if inf_before:
                    _log(f"L2-norm had {inf_before} inf values before sanitization")
                X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=1e12, neginf=0.0)
                X_norm = np.clip(X_norm, 0.0, 1e12)
                X_norm = X_norm.astype(np.float32, copy=False).reshape(-1, 1)
                labels_norm = _run_optics(
                    X_norm,
                    int(my_param),
                    args.xi,
                    args.min_cluster_size,
                    args.cluster_method,
                    args.metric,
                    args.max_eps,
                )
                # Write L2-norm labels CSV for OPTICS
                try:
                    out_df_norm = pd.DataFrame(
                        {
                            "ip": ips_list,
                            "source_file": sources_list,
                            "l2_norm": X_norm.reshape(-1).tolist(),
                            "label": labels_norm.tolist(),
                        }
                    )
                    out_file_norm = os.path.join(args.out_dir, f"optics_minSamples_{int(my_param)}_labels_l2norm.csv")
                    out_df_norm.to_csv(out_file_norm, index=False)
                    _log(f"Wrote L2-norm labels to {out_file_norm}")
                except Exception as e:
                    _log(f"Failed writing L2-norm labels CSV: {e}")
            _log(f"Finished param={int(my_param)} (silhouette disabled)")
        except Exception:
            # print traceback
            import traceback

            traceback.print_exc()
            _log(f"Error occurred for param={int(my_param)}")

    if rank == 0:
        _log("Finished all clustering runs (silhouette disabled)")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
