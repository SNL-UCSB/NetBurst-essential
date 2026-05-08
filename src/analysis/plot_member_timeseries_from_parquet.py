#!/usr/bin/env python3
"""
Plot per-member inbound byte time series from a Parquet dataset for cluster member IDs.

Designed to pair with cluster_analysis members.json (list of {ip, source_file}) and the
same inbound-only Parquet layout used by DeepARNoGraph.from_inbound_only / Slurm
extract_ip_representations_deepar.sl (INPUT_PARQUET).

JSON-driven: pass --config path/to/config.json

Figure style matches ``plot_graphs/plot_utils.py`` (golden-ratio size, font size 14, line width)
and save settings used in ``plot_mape_vs_wd_by_dataset.py`` (e.g. savefig dpi 300).
Config: width_in, font_size, line_width, savefig_dpi; optional figsize [w, h] to override ratio.

Plots are written in each cluster directory next to ``members.json`` when using
``members_json_paths`` or ``members_json_glob``. Use ``output_dir`` only for inline
``members`` (required) or for an optional combined manifest (``write_combined_manifest``).

Parallelism: single-process unless ``num_workers`` > 1 or ``0``/``"auto"``. Auto uses
``SLURM_CPUS_PER_TASK`` when set (Slurm allocation), else ``os.cpu_count()``. Optional
``max_workers`` caps auto. Set ``num_workers`` explicitly (e.g. 64) to override. This is
not PySpark—only independent Parquet lookups + plots per process.

Parquet loading: ``ds.dataset(...)`` is **lazy** (schema + file layout; it does **not** read
the whole table into RAM). **Sequential:** one dataset handle in the parent, reused for
every member. **Parallel:** each worker calls ``ds.dataset`` **once** in the pool
initializer, then reuses that handle for all tasks assigned to that worker; each task only
runs a **filtered** read (``to_table`` / scan) for the matching row(s). The parent process
opens the dataset briefly for schema checks, then releases it before workers run. You get
many small filtered reads across processes (OS page cache is shared), not a full Parquet
reload on every plot.

Y-axis (methodology): default ``y_floor_zero`` / ``y_axis.floor_zero`` pins the left axis at
zero so cross-cluster intensity comparisons are not distorted by different auto-scaled
floors. Optional ``share_y_max_within_k_parent`` / ``y_axis.share_max_within_k_parent`` sets
a common maximum within each ``K_*`` directory (all cluster subfolders under the same K).

Optional ``y_axis.fixed_max`` (or top-level ``y_fixed_max``) sets a global y-axis maximum for
every plot (e.g. 750000 for bytes), with the minimum at 0 when ``floor_zero`` is true. This
overrides per-series and shared-K max logic so all figures use the same scale.

Fisher overlay (optional ``fisher_overlay``): reads ``feature_order_fisher.csv``
beside each ``K_*`` dir (from cluster_analysis) and the merged TSFresh table; draws the top
``top_n`` Fisher-ranked features as horizontal reference lines on a twin axis (values
min–max normalized column-wise over the full TSFresh table), with legend showing Fisher
ratios — scalar features per series, aligned with the cluster invariance pipeline.

Lowest normalized-variance overlay (optional ``lowest_norm_var_overlay``): reads the
per-cluster variance CSV (same as ``cluster_analysis --cluster-variance-out``), applies
the same population law-of-total-variance normalization as ``cluster_analysis.ipynb`` /
``cluster_variance_stats.heatmap_population_z_alpha_from_variance_df``, ranks features by
``V_within / var_pop`` per cluster (lowest = most invariant among selected features), and draws the top
``top_n`` (default 3) on the twin axis. Mutually exclusive with ``fisher_overlay``.
Can be combined with ``top_cohens_d_overlay`` (dashed vs dash-dot lines; per-line legend
abbrev can differ, e.g. ``value`` vs ``z``). By default the plotted number is the cluster
metric used to pick the feature; ``legend_value_source`` = ``"member"`` uses the member's
TSFresh value on the twin axis.

Top-Cohen overlay (optional ``top_cohens_d_overlay``): reads each cluster's
``overlay_cohen_d_top10.csv`` (or configured filename), takes the top ``top_n``
rows by ``abs_z``, draws reference lines on the twin axis. Combine with
``lowest_norm_var_overlay`` for lowest ``V/var_pop`` plus high-|z| features (do not
enable ``top_cka_overlay`` at the same time).

Top-CKA overlay (optional ``top_cka_overlay``): reads ``selected_features_cka.csv``. Mutually
exclusive with ``top_cohens_d_overlay`` when combined with ``lowest_norm_var_overlay``.

``plot_variants`` (optional list): run multiple overlay modes in one pass. Each element is a
JSON object merged over the base config (excluding ``plot_variants``); set
``output_suffix`` (e.g. ``_ckaorder``) so PNG/PDF stems do not collide. Each variant must
enable at most one of ``fisher_overlay``, ``lowest_norm_var_overlay`` + ``top_cka_overlay``,
``lowest_norm_var_overlay`` + ``top_cohens_d_overlay``, or a single of ``top_cka_overlay`` /
``top_cohens_d_overlay`` — same mutual-exclusion rules as the flat config.

Optional ``overlay_legend_fontsize`` (top-level or per-variant): overrides legend size for
twin-axis overlays; when unset, legend matches the axis tick label size (``xtick.labelsize``
from ``setup_plot_style``, same as the left/bottom tick numbers).

By default the overlay legend is drawn **inside** the main axes (upper right), with a
**transparent** frame so the time series stays large. Set ``overlay_legend_bbox_anchor_y`` to
a number (e.g. ``-0.14``) to restore the older **below-the-axes** legend and reserve bottom
margin for it.

Dual ``plot_variants`` stems: e.g. ``..._ckaorder.png`` vs ``..._zorder.png`` when using
``write_member_plot_config --dual-global-filter-overlays``. A path with **no** suffix is
usually from a single-variant (Cohen-only) run, not the dual second variant.

When twin-axis overlays are drawn, also writes ``{stem}_overlay_features.csv`` next to each
PNG/PDF: ``cluster_dir``, ``plot_tag``, ``series_id``, key columns (e.g. ``ip``,
``source_file``), then ``rank``, ``feature``, ``metric_*``, ``tsfresh_value_member``.

By default (unless ``write_member_sequence_ids_manifest`` is false), writes
``member_plot_sequence_ids.csv`` under the shared parent of all ``cluster_*`` dirs (e.g.
``seed_42``): one row per ``members.json`` entry — use this file to subset other models to
exactly the same sequences. Override path with ``member_sequence_ids_csv``.

By default each plotted series is also saved as ``{stem}_timeseries.csv`` (columns
``time_index`` and the value column) next to the figures for reproducibility; set
``write_timeseries_csv`` to false to disable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

_analysis_dir = Path(__file__).resolve().parent
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))
from typing import Any

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from cluster_variance_stats import population_stats_from_variance_df
from cluster_fs import safe_cluster_dir_name
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set in each Pool worker by _pool_init_worker.
_WORKER_DATASET: Any = None
# Group key for inline members when sharing y-axis max across a run.
_INLINE_K_GROUP = Path("__inline__")


def _ensure_plot_utils() -> None:
    """Allow `from plot_utils import ...` (same package as plot_mape_vs_wd_by_dataset.py)."""
    pg = Path(__file__).resolve().parent.parent / "plot_graphs"
    s = str(pg)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _expand_members_sources(
    cfg: dict[str, Any],
) -> list[tuple[str, dict[str, Any], Path | None]]:
    """
    Returns list of (source_tag, member_record, cluster_dir).

    cluster_dir is the directory containing members.json (plots are written here), or None
    for inline ``members`` (then output_dir/output_subdir is used).
    """
    out: list[tuple[str, dict[str, Any], Path | None]] = []

    if "members_json_paths" in cfg:
        for mp in cfg["members_json_paths"]:
            p = Path(os.path.expandvars(str(mp))).expanduser().resolve()
            tag = p.parent.name if p.parent.name else p.stem
            cluster_dir = p.parent
            with open(p, encoding="utf-8") as f:
                rows = json.load(f)
            for i, rec in enumerate(rows):
                out.append((f"{tag}_m{i}", dict(rec), cluster_dir))

    if "members_json_glob" in cfg:
        import glob

        pattern = os.path.expandvars(cfg["members_json_glob"])
        for mp in sorted(glob.glob(pattern)):
            p = Path(mp).resolve()
            tag = p.parent.name
            cluster_dir = p.parent
            with open(p, encoding="utf-8") as f:
                rows = json.load(f)
            for i, rec in enumerate(rows):
                out.append((f"{tag}_m{i}", dict(rec), cluster_dir))

    if "members" in cfg:
        for i, rec in enumerate(cfg["members"]):
            out.append((f"inline_m{i}", dict(rec), None))

    return out


def _key_columns(cfg: dict[str, Any]) -> list[str]:
    kc = cfg.get("key_columns")
    if kc:
        return list(kc)
    if cfg.get("use_service_port"):
        return ["ip", "service_port", "source_file"]
    return ["ip", "source_file"]


def _value_column(cfg: dict[str, Any]) -> str:
    return str(cfg.get("value_column", "inbound"))


def _series_id(record: dict[str, Any], key_cols: list[str]) -> str:
    return "::".join(str(record[c]) for c in key_cols if c in record)


def _safe_stem(tag: str, series_id: str, max_len: int = 120) -> str:
    h = hashlib.sha256(series_id.encode("utf-8")).hexdigest()[:12]
    base = f"{tag}_{h}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:max_len]
    return safe or "member"


def _scalar_for_column(dataset: ds.Dataset, col: str, value: Any) -> pa.Scalar:
    field = dataset.schema.field(col)
    typ = field.type
    if pa.types.is_integer(typ):
        try:
            return pa.scalar(int(value), type=typ)
        except (TypeError, ValueError):
            return pa.scalar(str(value), type=pa.string())
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        return pa.scalar(str(value), type=typ)
    return pa.scalar(value, type=typ)


def _filter_for_record(dataset: ds.Dataset, record: dict[str, Any], key_cols: list[str]) -> pc.Expression | None:
    exprs: list[pc.Expression] = []
    for c in key_cols:
        if c not in record:
            raise KeyError(f"Member record missing key {c!r}; have {list(record.keys())}")
        exprs.append(pc.field(c) == _scalar_for_column(dataset, c, record[c]))
    if not exprs:
        return None
    out = exprs[0]
    for e in exprs[1:]:
        out = out & e
    return out


def _inbound_to_array(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    arr = np.array(
        [float(x) for x in raw if x is not None and x != -1],
        dtype=np.float64,
    )
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _row_matches(record: dict[str, Any], key_cols: list[str], table: pa.Table, row_idx: int) -> bool:
    for c in key_cols:
        v = table.column(c)[row_idx].as_py()
        w = record[c]
        if v != w and str(v) != str(w):
            return False
    return True


def _fetch_inbound_for_member(
    dataset: ds.Dataset,
    record: dict[str, Any],
    key_cols: list[str],
    value_col: str,
) -> np.ndarray | None:
    cols = key_cols + [value_col]
    filt = _filter_for_record(dataset, record, key_cols)
    table: pa.Table | None = None
    if filt is not None:
        try:
            table = dataset.to_table(filter=filt, columns=cols)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            table = None

    if table is None or len(table) == 0:
        # Type mismatch (e.g. source_file int vs JSON string): narrow by ip then match in Python.
        if "ip" not in record:
            return None
        try:
            ip_f = pc.field("ip") == _scalar_for_column(dataset, "ip", record["ip"])
            table = dataset.to_table(filter=ip_f, columns=cols)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            return None
        if len(table) == 0:
            return None
        mask = pa.array([_row_matches(record, key_cols, table, i) for i in range(len(table))])
        table = table.filter(mask)

    if len(table) == 0:
        return None
    if len(table) > 1:
        # Rare duplicate keys; first row wins (matches typical single-series semantics).
        table = table.slice(0, 1)

    col = table.column(value_col)
    return _inbound_to_array(col[0])


def _member_key(rec: dict[str, Any], key_cols: list[str]) -> tuple[str, ...]:
    return tuple(str(rec[c]) for c in key_cols)


def _parse_plot_member_index(plot_tag: str) -> int | None:
    """``cluster_10_m3`` -> 3; ``inline_m0`` -> 0."""
    if "_m" not in plot_tag:
        return None
    suf = plot_tag.rsplit("_m", 1)[-1]
    return int(suf) if suf.isdigit() else None


def _member_manifest_parent_dir(members: list[tuple[str, dict[str, Any], Path | None]]) -> Path | None:
    """Single shared parent of all ``cluster_*`` dirs (e.g. ``seed_42``), or None if mixed/inline-only."""
    parents = {cd.resolve().parent for _, _, cd in members if cd is not None}
    if len(parents) == 1:
        return parents.pop()
    return None


def _write_member_sequence_ids_manifest(
    out_path: Path,
    members: list[tuple[str, dict[str, Any], Path | None]],
    key_cols: list[str],
) -> None:
    """
    One row per member row in ``members.json`` (same order as plot expansion).
    Use this CSV to subset other models/datasets to exactly these sequences.
    """
    rows: list[dict[str, Any]] = []
    for plot_tag, rec, cluster_dir in members:
        sid = _series_id(rec, key_cols)
        mjson = ""
        cdir = ""
        if cluster_dir is not None:
            cdir = cluster_dir.name
            mj = cluster_dir / "members.json"
            mjson = str(mj.resolve())
        row: dict[str, Any] = {
            "cluster_dir": cdir,
            "member_index": _parse_plot_member_index(plot_tag),
            "plot_tag": plot_tag,
            "series_id": sid,
            "members_json": mjson,
        }
        for k in key_cols:
            row[k] = str(rec[k]) if k in rec else ""
        rows.append(row)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote member sequence manifest ({len(rows)} rows) -> {out_path}", flush=True)


def _write_timeseries_csv(
    y: np.ndarray,
    out_csv: Path,
    *,
    value_column: str,
) -> None:
    """Save the plotted samples as CSV: time_index + value column (same data as the line plot)."""
    out_csv = Path(out_csv)
    # Avoid problematic CSV header characters
    vcol = str(value_column).replace("\n", " ").strip() or "value"
    df = pd.DataFrame(
        {
            "time_index": np.arange(len(y), dtype=np.int64),
            vcol: np.asarray(y, dtype=np.float64),
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _load_heatmap_fisher_order(heatmap_path: Path, top_n: int) -> list[tuple[str, float]]:
    """Return [(feature_base, fisher_ratio_vis), ...] in Fisher-descending order (CSV order)."""
    df = pd.read_csv(heatmap_path, low_memory=False)
    if "feature_base" not in df.columns or "fisher_ratio_vis" not in df.columns:
        raise ValueError(
            f"heatmap CSV {heatmap_path} needs columns feature_base, fisher_ratio_vis; "
            f"have {list(df.columns)}"
        )
    out: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        feat = str(row["feature_base"]).strip()
        fr = float(row["fisher_ratio_vis"])
        out.append((feat, fr))
        if len(out) >= top_n:
            break
    return out


def _read_tsfresh_for_fisher_overlay(
    tsfresh_path: str,
    key_cols: list[str],
    feature_names: list[str],
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Load TSFresh table with only id + requested features; return indexed frame + per-column min/max."""
    import pyarrow.parquet as pq

    path = Path(os.path.expandvars(str(tsfresh_path))).expanduser()
    cols = list(dict.fromkeys(list(key_cols) + list(feature_names)))
    if path.suffix.lower() == ".parquet":
        avail = set(pq.ParquetFile(path).schema.names)
        use = [c for c in cols if c in avail]
        df = pd.read_parquet(path, columns=use)
    else:
        df = pd.read_csv(path, usecols=lambda c: c in set(cols), low_memory=False)
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"TSFresh table missing id columns {missing}")
    feats_ok = [f for f in feature_names if f in df.columns]
    if not feats_ok:
        raise ValueError("None of the Fisher-ranked feature columns exist in the TSFresh table.")
    for c in key_cols:
        df[c] = df[c].astype(str)
    df = df.set_index(list(key_cols), drop=True)
    minmax: dict[str, tuple[float, float]] = {}
    for f in feats_ok:
        s = pd.to_numeric(df[f], errors="coerce")
        lo = float(np.nanmin(s.values)) if s.notna().any() else 0.0
        hi = float(np.nanmax(s.values)) if s.notna().any() else 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        minmax[f] = (lo, hi)
    return df[feats_ok], minmax


def _y_axis_fixed_max(cfg: dict[str, Any]) -> float | None:
    """Global ymax for all plots; None = use per-series or share-within-K logic."""
    yax = cfg.get("y_axis") if isinstance(cfg.get("y_axis"), dict) else {}
    raw = cfg.get("y_fixed_max")
    if raw is None:
        raw = yax.get("fixed_max")
    if raw is None:
        raw = yax.get("max")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return v


def _y_axis_flags(cfg: dict[str, Any]) -> tuple[bool, bool]:
    """floor_y_zero, share_y_max_within_k_parent."""
    yax = cfg.get("y_axis") if isinstance(cfg.get("y_axis"), dict) else {}
    floor = _parse_bool(cfg.get("y_floor_zero", yax.get("floor_zero")), default=True)
    share = _parse_bool(
        cfg.get("share_y_max_within_k_parent", yax.get("share_max_within_k_parent")),
        default=False,
    )
    return floor, share


def _parse_bool(val: Any, *, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
    return default


def _fisher_overlay_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    fo = cfg.get("fisher_overlay")
    if not isinstance(fo, dict) or not _parse_bool(fo.get("enabled"), default=False):
        return None
    ts = fo.get("tsfresh_csv")
    if not ts:
        raise ValueError("fisher_overlay.enabled requires fisher_overlay.tsfresh_csv")
    return fo


def _build_fisher_overlay_by_member_key(
    members: list[tuple[str, dict[str, Any], Path | None]],
    key_cols: list[str],
    fo: dict[str, Any],
) -> dict[tuple[str, ...], list[dict[str, Any]] | None]:
    """
    For each member key, precompute Fisher overlay line specs (or None).
    Requires cluster_dir so feature_order_fisher.csv can be resolved under K_*.
    """
    top_n = int(fo.get("top_n", 5))
    heatmap_name = str(fo.get("heatmap_filename", "feature_order_fisher.csv"))
    k_to_order: dict[Path, list[tuple[str, float]]] = {}
    for _, _, cluster_dir in members:
        if cluster_dir is None:
            continue
        kp = cluster_dir.resolve().parent
        if kp in k_to_order:
            continue
        hp = kp / heatmap_name
        if not hp.is_file():
            continue
        try:
            k_to_order[kp] = _load_heatmap_fisher_order(hp, top_n)
        except (OSError, ValueError) as e:
            print(f"[plot_member_timeseries] skip Fisher overlay for {kp}: {e}", flush=True)

    if not k_to_order:
        print(
            "[plot_member_timeseries] fisher_overlay enabled but no heatmap CSVs found "
            f"(expected {heatmap_name} next to cluster_* dirs).",
            flush=True,
        )
        return {}

    feat_union: list[str] = []
    seen: set[str] = set()
    for ordered in k_to_order.values():
        for feat, _ in ordered:
            if feat not in seen:
                seen.add(feat)
                feat_union.append(feat)

    ts_path = str(os.path.expandvars(str(fo["tsfresh_csv"])))
    ts_indexed, minmax = _read_tsfresh_for_fisher_overlay(ts_path, key_cols, feat_union)

    out: dict[tuple[str, ...], list[dict[str, Any]] | None] = {}
    for _, rec, cluster_dir in members:
        mk = _member_key(rec, key_cols)
        if mk in out:
            continue
        if cluster_dir is None:
            out[mk] = None
            continue
        kp = cluster_dir.resolve().parent
        ordered = k_to_order.get(kp)
        if not ordered:
            out[mk] = None
            continue
        lines = _fisher_lines_for_member(ts_indexed, minmax, ordered, mk)
        out[mk] = lines
    return out


def _fisher_lines_for_member(
    ts_indexed: pd.DataFrame,
    minmax: dict[str, tuple[float, float]],
    ordered: list[tuple[str, float] | tuple[str, float, str]],
    member_key: tuple[str, ...],
    eps: float = 1e-12,
    legend_value_source: str = "metric",
) -> list[dict[str, Any]] | None:
    """Build overlay specs: each feature -> y_norm in [0,1] for twin axis.

    ``ordered`` entries may be ``(feature, metric)`` or ``(feature, metric, legend_abbrev)``
    for per-line legend labels (e.g. ``value`` vs ``z``).
    """
    try:
        row = ts_indexed.loc[member_key]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    lines: list[dict[str, Any]] = []
    for item in ordered:
        if len(item) >= 3:
            feat, fr, legend_abbrev = item[0], float(item[1]), str(item[2])
        else:
            feat, fr = item[0], float(item[1])
            legend_abbrev = ""
        if feat not in ts_indexed.columns:
            continue
        v = float(pd.to_numeric(row[feat], errors="coerce"))
        if not np.isfinite(v):
            continue
        lo, hi = minmax.get(feat, (0.0, 1.0))
        y_norm = (v - lo) / (hi - lo + eps)
        y_norm = float(np.clip(y_norm, 0.0, 1.0))
        label_value = v if str(legend_value_source).lower() == "member" else fr
        li: dict[str, Any] = {
            "name": feat,
            "fr": float(label_value),
            "metric_from_file": float(fr),
            "y_norm": y_norm,
            "raw": v,
        }
        if legend_abbrev:
            li["metric_abbrev"] = legend_abbrev
        lines.append(li)
    return lines or None


def _lowest_norm_var_overlay_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    lo = cfg.get("lowest_norm_var_overlay")
    if not isinstance(lo, dict) or not _parse_bool(lo.get("enabled"), default=False):
        return None
    vc = lo.get("variance_csv")
    ts = lo.get("tsfresh_csv")
    if not vc or not ts:
        raise ValueError(
            "lowest_norm_var_overlay.enabled requires lowest_norm_var_overlay.variance_csv "
            "and lowest_norm_var_overlay.tsfresh_csv"
        )
    return lo


def _cluster_row_index_for_variance_overlay(cluster_dir: Path, cluster_ids: list[str]) -> int | None:
    """Map .../cluster_*/ to row index in the variance / heatmap matrices."""
    want = cluster_dir.name
    for i, cid in enumerate(cluster_ids):
        if safe_cluster_dir_name(cid) == want:
            return i
    return None


def _build_lowest_norm_var_overlay_by_member_key(
    members: list[tuple[str, dict[str, Any], Path | None]],
    key_cols: list[str],
    lo: dict[str, Any],
) -> dict[tuple[str, ...], list[dict[str, Any]] | None]:
    """
    Per cluster, top_n features with smallest V_within / var_pop (same var_pop as heatmap).
    TSFresh values drawn as global min–max norm on twin axis (same as Fisher overlay).
    """
    top_n = int(lo.get("top_n", 3))
    fisher_reorder = not _parse_bool(lo.get("no_fisher_reorder"), default=False)
    eps = float(lo.get("eps", 1e-12))
    legend_value_source = str(lo.get("legend_value_source", "metric"))

    vpath = Path(os.path.expandvars(str(lo["variance_csv"]))).expanduser().resolve()
    if not vpath.is_file():
        print(f"[plot_member_timeseries] lowest_norm_var_overlay: missing {vpath}", flush=True)
        return {}

    df = pd.read_csv(vpath, low_memory=False)
    if "cluster_id" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "cluster_id"})
    try:
        hm = population_stats_from_variance_df(
            df,
            cluster_id_col="cluster_id",
            fisher_reorder=fisher_reorder,
        )
    except (OSError, ValueError) as e:
        print(f"[plot_member_timeseries] lowest_norm_var overlay failed: {e}", flush=True)
        return {}

    cluster_ids = hm["cluster_ids"]
    M = hm["M"]
    V = hm["V"]
    var_pop = hm["std_pop"].astype(float) ** 2
    vnorm = V / (var_pop[None, :] + eps)
    vnorm = np.nan_to_num(vnorm, nan=np.inf, posinf=np.inf, neginf=np.inf)
    bases = hm["feature_bases"]
    n_c, n_f = vnorm.shape

    # Optional summary CSV (one row per cluster × rank)
    summary_path = lo.get("write_summary_csv")
    summary_rows: list[dict[str, Any]] = []

    cid_to_ordered: dict[str, list[tuple[str, float]]] = {}
    for i in range(n_c):
        order = np.argsort(vnorm[i, :], kind="mergesort")
        take = order[:top_n].tolist()
        feats_metrics: list[tuple[str, float]] = []
        for j in take:
            # Legend value = within-cluster mean (ranking still by lowest V/var_pop).
            m_abbrev = str(lo.get("metric_abbrev") or "value")
        feats_metrics.append((str(bases[j]), float(M[i, j]), m_abbrev))
        cid_to_ordered[str(cluster_ids[i])] = feats_metrics
        for rank, j in enumerate(take, start=1):
            summary_rows.append(
                {
                    "cluster_id": cluster_ids[i],
                    "rank": rank,
                    "feature": str(bases[j]),
                    "mean_within": float(M[i, j]),
                    "v_norm": float(vnorm[i, j]),
                    "var_pop": float(var_pop[j]),
                    "var_within": float(V[i, j]),
                }
            )

    if summary_path:
        sp = Path(os.path.expandvars(str(summary_path))).expanduser()
        sp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(sp, index=False)
        print(f"[plot_member_timeseries] wrote lowest V/var_pop summary -> {sp}", flush=True)

    feat_union: list[str] = []
    seen: set[str] = set()
    for ordered in cid_to_ordered.values():
        for item in ordered:
            feat = item[0]
            if feat not in seen:
                seen.add(feat)
                feat_union.append(feat)

    ts_path = str(os.path.expandvars(str(lo["tsfresh_csv"])))
    ts_indexed, minmax = _read_tsfresh_for_fisher_overlay(ts_path, key_cols, feat_union)

    out: dict[tuple[str, ...], list[dict[str, Any]] | None] = {}
    for _, rec, cluster_dir in members:
        mk = _member_key(rec, key_cols)
        if mk in out:
            continue
        if cluster_dir is None:
            out[mk] = None
            continue
        ri = _cluster_row_index_for_variance_overlay(cluster_dir, cluster_ids)
        if ri is None:
            out[mk] = None
            continue
        cid = str(cluster_ids[ri])
        ordered = cid_to_ordered.get(cid)
        if not ordered:
            out[mk] = None
            continue
        lines = _fisher_lines_for_member(
            ts_indexed,
            minmax,
            ordered,
            mk,
            legend_value_source=legend_value_source,
        )
        out[mk] = lines
    return out


def _top_cka_overlay_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    c = cfg.get("top_cka_overlay")
    if not isinstance(c, dict) or not _parse_bool(c.get("enabled"), default=False):
        return None
    if not c.get("tsfresh_csv"):
        raise ValueError("top_cka_overlay.enabled requires top_cka_overlay.tsfresh_csv")
    return c


def _top_cohens_d_overlay_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    c = cfg.get("top_cohens_d_overlay")
    if not isinstance(c, dict) or not _parse_bool(c.get("enabled"), default=False):
        return None
    if not c.get("tsfresh_csv"):
        raise ValueError("top_cohens_d_overlay.enabled requires top_cohens_d_overlay.tsfresh_csv")
    return c


def _tag_overlay_lines(
    lines: list[dict[str, Any]] | None,
    linestyle: str,
) -> list[dict[str, Any]] | None:
    if not lines:
        return None
    out: list[dict[str, Any]] = []
    for li in lines:
        d = dict(li)
        d["linestyle"] = linestyle
        out.append(d)
    return out


def _merge_vnorm_and_cka_lines(
    vnorm: list[dict[str, Any]] | None,
    cka: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    parts: list[dict[str, Any]] = []
    t1 = _tag_overlay_lines(vnorm, "--")
    if t1:
        parts.extend(t1)
    t2 = _tag_overlay_lines(cka, "-.")
    if t2:
        parts.extend(t2)
    return parts or None


def _build_top_cka_overlay_by_member_key(
    members: list[tuple[str, dict[str, Any], Path | None]],
    key_cols: list[str],
    cka_cfg: dict[str, Any],
) -> dict[tuple[str, ...], list[dict[str, Any]] | None]:
    """
    Per cluster, top_n features by rank in selected_features_cka.csv (rank 1 = highest CKA in file).
    """
    top_n = int(cka_cfg.get("top_n", 3))
    filename = str(cka_cfg.get("filename", "selected_features_cka.csv"))
    ts_path = str(os.path.expandvars(str(cka_cfg["tsfresh_csv"])))
    legend_value_source = str(cka_cfg.get("legend_value_source", "metric"))

    cluster_dir_to_ordered: dict[Path, list[tuple[str, float]]] = {}
    feat_union: list[str] = []
    seen: set[str] = set()
    for _, _, cluster_dir in members:
        if cluster_dir is None:
            continue
        cd = cluster_dir.resolve()
        if cd in cluster_dir_to_ordered:
            continue
        p = cd / filename
        if not p.is_file():
            cluster_dir_to_ordered[cd] = []
            continue
        df = pd.read_csv(p, low_memory=False)
        if df.empty or "feature" not in df.columns or "cka_to_embedding" not in df.columns:
            cluster_dir_to_ordered[cd] = []
            continue
        if "rank" in df.columns:
            df = df.sort_values("rank", ascending=True, kind="mergesort")
        else:
            df = df.sort_values("cka_to_embedding", ascending=False, kind="mergesort")
        sub = df.head(top_n)
        cka_abbrev = str(cka_cfg.get("metric_abbrev", "cka"))
        ordered: list[tuple[str, float, str]] = []
        for _, r in sub.iterrows():
            feat = str(r["feature"])
            cka = float(pd.to_numeric(r["cka_to_embedding"], errors="coerce"))
            if np.isfinite(cka):
                ordered.append((feat, cka, cka_abbrev))
        cluster_dir_to_ordered[cd] = ordered
        for item in ordered:
            feat = item[0]
            if feat not in seen:
                seen.add(feat)
                feat_union.append(feat)

    if not feat_union:
        print(
            "[plot_member_timeseries] top_cka_overlay: no features found in "
            f"{filename} under member cluster dirs.",
            flush=True,
        )
        return {}

    ts_indexed, minmax = _read_tsfresh_for_fisher_overlay(ts_path, key_cols, feat_union)
    out: dict[tuple[str, ...], list[dict[str, Any]] | None] = {}
    for _, rec, cluster_dir in members:
        mk = _member_key(rec, key_cols)
        if mk in out:
            continue
        if cluster_dir is None:
            out[mk] = None
            continue
        ordered = cluster_dir_to_ordered.get(cluster_dir.resolve(), [])
        if not ordered:
            out[mk] = None
            continue
        lines = _fisher_lines_for_member(
            ts_indexed,
            minmax,
            ordered,
            mk,
            legend_value_source=legend_value_source,
        )
        out[mk] = lines
    return out


def _build_top_cohens_d_overlay_by_member_key(
    members: list[tuple[str, dict[str, Any], Path | None]],
    key_cols: list[str],
    zs_cfg: dict[str, Any],
) -> dict[tuple[str, ...], list[dict[str, Any]] | None]:
    """
    Per cluster, top_n features by effect size in the configured overlay CSV.
    """
    top_n = int(zs_cfg.get("top_n", 3))
    filename = str(zs_cfg.get("filename", "overlay_cohen_d_top10.csv"))
    ts_path = str(os.path.expandvars(str(zs_cfg["tsfresh_csv"])))
    legend_value_source = str(zs_cfg.get("legend_value_source", "metric"))
    metric_abbrev = str(zs_cfg.get("metric_abbrev", "d"))

    cluster_dir_to_ordered: dict[Path, list[tuple[str, float, str]]] = {}
    feat_union: list[str] = []
    seen: set[str] = set()
    for _, _, cluster_dir in members:
        if cluster_dir is None:
            continue
        cd = cluster_dir.resolve()
        if cd in cluster_dir_to_ordered:
            continue
        p = cd / filename
        if not p.is_file():
            cluster_dir_to_ordered[cd] = []
            continue
        try:
            df = pd.read_csv(p, low_memory=False)
        except pd.errors.EmptyDataError:
            cluster_dir_to_ordered[cd] = []
            continue
        if df.empty or "feature" not in df.columns:
            cluster_dir_to_ordered[cd] = []
            continue
        metric_col = None
        abs_metric_col = None
        if "cohen_d" in df.columns:
            metric_col = "cohen_d"
            abs_metric_col = "abs_cohen_d" if "abs_cohen_d" in df.columns else None
        elif "z" in df.columns:
            metric_col = "z"
            abs_metric_col = "abs_z" if "abs_z" in df.columns else None
        if metric_col is None:
            cluster_dir_to_ordered[cd] = []
            continue
        work = df.copy()
        if abs_metric_col is not None:
            work["_sort_metric"] = pd.to_numeric(work[abs_metric_col], errors="coerce")
        else:
            work["_sort_metric"] = np.abs(pd.to_numeric(work[metric_col], errors="coerce"))
        work = work.sort_values("_sort_metric", ascending=False, kind="mergesort")
        sub = work.head(top_n)
        ordered: list[tuple[str, float, str]] = []
        for _, r in sub.iterrows():
            feat = str(r["feature"])
            metric_value = float(pd.to_numeric(r[metric_col], errors="coerce"))
            if np.isfinite(metric_value):
                ordered.append((feat, metric_value, metric_abbrev))
        cluster_dir_to_ordered[cd] = ordered
        for feat, _, _ in ordered:
            if feat not in seen:
                seen.add(feat)
                feat_union.append(feat)

    if not feat_union:
        print(
            "[plot_member_timeseries] top_cohens_d_overlay: no features found in "
            f"{filename} under member cluster dirs.",
            flush=True,
        )
        return {}

    ts_indexed, minmax = _read_tsfresh_for_fisher_overlay(ts_path, key_cols, feat_union)
    out: dict[tuple[str, ...], list[dict[str, Any]] | None] = {}
    for _, rec, cluster_dir in members:
        mk = _member_key(rec, key_cols)
        if mk in out:
            continue
        if cluster_dir is None:
            out[mk] = None
            continue
        ordered = cluster_dir_to_ordered.get(cluster_dir.resolve(), [])
        if not ordered:
            out[mk] = None
            continue
        lines = _fisher_lines_for_member(
            ts_indexed,
            minmax,
            ordered,
            mk,
            legend_value_source=legend_value_source,
        )
        out[mk] = lines
    return out


def _write_member_overlay_features_csv(
    out_path: Path,
    fisher_lines: list[dict[str, Any]],
    *,
    member_record: dict[str, Any] | None,
    member_key_cols: list[str] | None,
    cluster_dir_name: str = "",
    plot_tag: str = "",
    series_id: str = "",
) -> None:
    """One row per overlay line; leading columns identify cluster/member (IP, source_file, etc.)."""
    rows: list[dict[str, Any]] = []
    kcols = member_key_cols or []
    for i, li in enumerate(fisher_lines):
        mabbr = li.get("metric_abbrev")
        if mabbr is None:
            mabbr = ""
        row: dict[str, Any] = {
            "cluster_dir": cluster_dir_name,
            "plot_tag": plot_tag,
            "series_id": series_id,
        }
        for k in kcols:
            v = member_record.get(k) if member_record else None
            row[k] = "" if v is None else str(v)
        row["rank"] = i + 1
        row["feature"] = li["name"]
        row["metric_name"] = str(mabbr).lower()
        row["metric_value"] = float(li.get("metric_from_file", li["fr"]))
        row["tsfresh_value_member"] = float(li["raw"])
        rows.append(row)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _axes_tick_fontsize_pt() -> float:
    """Tick label size in points (matches left/bottom axis after setup_plot_style)."""
    v = plt.rcParams["xtick.labelsize"]
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(plt.rcParams["font.size"])


def _plot_series(
    y: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    *,
    width_in: float,
    figsize: tuple[float, float] | None,
    line_width: float,
    savefig_dpi: int,
    y_max: float | None = None,
    floor_y_zero: bool = True,
    fisher_lines: list[dict[str, Any]] | None = None,
    overlay_twin_ylabel: str | None = None,
    overlay_metric_abbrev: str = "FR",
    overlay_legend_ncol: int = 2,
    overlay_width_in: float | None = None,
    overlay_hide_twin_y_axis: bool = True,
    overlay_print_z_tsfresh_panel: bool = True,
    overlay_show_twin_reference_lines: bool = True,
    overlay_legend_text_only: bool = False,
    overlay_legend_fontsize: float | None = None,
    overlay_legend_bbox_anchor_y: float | None = None,
    overlay_legend_metric_comma_tsfresh: bool = False,
    member_overlay_features_csv: Path | None = None,
    member_record: dict[str, Any] | None = None,
    member_key_cols: list[str] | None = None,
    member_overlay_cluster_dir: str = "",
    member_overlay_plot_tag: str = "",
    member_overlay_series_id: str = "",
) -> None:
    """Line plot with plot_graphs/plot_utils styling (golden-ratio figure, fonts)."""
    from plot_utils import create_figure

    x = np.arange(len(y), dtype=np.float64)
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        w_plot = width_in
        if fisher_lines:
            w_plot = float(overlay_width_in) if overlay_width_in is not None else max(width_in * 2.35, 11.0)
        fig, ax = create_figure(width_in=w_plot)
    ax.plot(x, y, color="#d62728", linewidth=line_width, zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("bytes")
    import matplotlib.ticker as _mticker

    def _si_bytes_fmt(val: float, _pos: int) -> str:
        for threshold, suffix in (
            (1e12, "T"),
            (1e9, "G"),
            (1e6, "M"),
            (1e3, "k"),
        ):
            if val >= threshold:
                scaled = val / threshold
                if scaled == int(scaled):
                    return f"{int(scaled)}{suffix}"
                return f"{scaled:g}{suffix}"
        return f"{int(val)}" if val == int(val) else f"{val:g}"

    ax.yaxis.set_major_formatter(_mticker.FuncFormatter(_si_bytes_fmt))

    _cdir = member_overlay_cluster_dir or ""
    _cnum_match = re.search(r"\d+", _cdir)
    if _cnum_match:
        ax.text(
            0.03, -0.14,
            f"Cluster {_cnum_match.group()}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=plt.rcParams.get("axes.labelsize", 9),
        )

    ax.tick_params(axis="both", which="major", labelleft=True, labelbottom=True)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.5, color="lightgray", alpha=0.5)
    if fisher_lines:
        ax.spines["right"].set_visible(True)
        n = len(fisher_lines)
        colors = ["#d62728"] * max(n, 1)
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, 1.0)
        if overlay_hide_twin_y_axis:
            ax2.set_ylabel("")
            ax2.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
                labelright=False,
                length=0,
            )
            ax2.set_yticks([])
            ax2.spines["right"].set_visible(False)
        else:
            if overlay_twin_ylabel is None:
                twin_ylabel = "Fisher-top features (global min–max norm)"
            else:
                twin_ylabel = overlay_twin_ylabel
            ax2.set_ylabel(twin_ylabel)
            ax2.tick_params(axis="y", labelsize=max(8, int(12 - n)))
            ax2.spines["right"].set_visible(True)
        for li in fisher_lines:
            if overlay_show_twin_reference_lines:
                ax2.axhline(
                    li["y_norm"],
                    color=colors[0],
                    linestyle=str(li.get("linestyle", "--")),
                    linewidth=line_width * 0.45,
                    alpha=0.9,
                    zorder=1,
                )
    else:
        ax.spines["right"].set_visible(False)

    if floor_y_zero:
        if y_max is not None:
            ax.set_ylim(0.0, float(y_max))
        else:
            top = float(np.nanmax(y)) * 1.05 if y.size else 1.0
            ax.set_ylim(0.0, top)
    elif y_max is not None:
        ax.set_ylim(None, float(y_max))

    ax.margins(x=0.01, y=0.0 if floor_y_zero else 0.02)
    fig.tight_layout()
    # Full figure bounds match figsize (no tight crop); legend stays in axes coords.
    fig.savefig(out_png, dpi=savefig_dpi)
    fig.savefig(out_pdf, dpi=savefig_dpi)
    plt.close(fig)
    if (
        fisher_lines
        and member_overlay_features_csv is not None
        and len(fisher_lines) > 0
    ):
        _write_member_overlay_features_csv(
            member_overlay_features_csv,
            fisher_lines,
            member_record=member_record,
            member_key_cols=member_key_cols,
            cluster_dir_name=member_overlay_cluster_dir,
            plot_tag=member_overlay_plot_tag,
            series_id=member_overlay_series_id,
        )


def _detect_cpus_for_pool() -> int:
    """Prefer Slurm allocation so we do not oversubscribe the cgroup."""
    v = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
    if v.isdigit():
        return max(1, int(v))
    return max(1, os.cpu_count() or 8)


def _resolve_num_workers(cfg: dict[str, Any]) -> int:
    raw = cfg.get("num_workers", 1)
    if raw is None:
        return 1
    if raw == 0 or raw == "auto":
        n = _detect_cpus_for_pool()
        cap = cfg.get("max_workers")
        if cap is not None:
            n = min(n, int(cap))
        return max(1, n)
    return max(1, int(raw))


def _normalize_plot_variant_jobs(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    raw = cfg.get("plot_variants")
    if not raw:
        return [("", cfg)]
    if not isinstance(raw, list) or not raw:
        raise SystemExit("plot_variants must be a non-empty list of objects when set.")
    base = {k: v for k, v in cfg.items() if k != "plot_variants"}
    out: list[tuple[str, dict[str, Any]]] = []
    for i, var in enumerate(raw):
        if not isinstance(var, dict):
            raise SystemExit(f"plot_variants[{i}] must be an object")
        suffix = var.get("output_suffix", "")
        if suffix is None:
            suffix = ""
        suffix = str(suffix)
        merged = {**base, **{k: v for k, v in var.items() if k != "output_suffix"}}
        out.append((suffix, merged))
    return out


def _build_overlay_bundle(
    cfg: dict[str, Any],
    members: list[tuple[str, dict[str, Any], Path | None]],
    key_cols: list[str],
) -> dict[str, Any]:
    fo_cfg = _fisher_overlay_config(cfg)
    lo_cfg = _lowest_norm_var_overlay_config(cfg)
    cka_cfg = _top_cka_overlay_config(cfg)
    zs_cfg = _top_cohens_d_overlay_config(cfg)
    if fo_cfg is not None and (lo_cfg is not None or cka_cfg is not None or zs_cfg is not None):
        raise SystemExit(
            "fisher_overlay cannot be combined with lowest_norm_var_overlay, top_cka_overlay, or top_cohens_d_overlay."
        )
    if lo_cfg is not None and cka_cfg is not None and zs_cfg is not None:
        raise SystemExit(
            "lowest_norm_var_overlay: enable at most one of top_cka_overlay and top_cohens_d_overlay."
        )
    overlay_by_key: dict[tuple[str, ...], list[dict[str, Any]] | None] | None = None
    overlay_twin_ylabel_effective: str | None = None
    overlay_metric_abbrev_effective = "FR"
    if fo_cfg is not None:
        overlay_by_key = _build_fisher_overlay_by_member_key(members, key_cols, fo_cfg)
        overlay_twin_ylabel_effective = cfg.get("overlay_twin_ylabel")
        if cfg.get("overlay_metric_abbrev") is not None:
            overlay_metric_abbrev_effective = str(cfg["overlay_metric_abbrev"])
    elif lo_cfg is not None and zs_cfg is not None:
        vnorm_map = _build_lowest_norm_var_overlay_by_member_key(members, key_cols, lo_cfg)
        zs_map = _build_top_cohens_d_overlay_by_member_key(members, key_cols, zs_cfg)
        overlay_by_key = {}
        for _, rec, _cd in members:
            mk = _member_key(rec, key_cols)
            if mk in overlay_by_key:
                continue
            overlay_by_key[mk] = _merge_vnorm_and_cka_lines(vnorm_map.get(mk), zs_map.get(mk))
        overlay_twin_ylabel_effective = str(cfg.get("overlay_twin_ylabel", ""))
        overlay_metric_abbrev_effective = str(cfg.get("overlay_metric_abbrev") or "")
    elif lo_cfg is not None and cka_cfg is not None:
        vnorm_map = _build_lowest_norm_var_overlay_by_member_key(members, key_cols, lo_cfg)
        cka_map = _build_top_cka_overlay_by_member_key(members, key_cols, cka_cfg)
        overlay_by_key = {}
        for _, rec, _cd in members:
            mk = _member_key(rec, key_cols)
            if mk in overlay_by_key:
                continue
            overlay_by_key[mk] = _merge_vnorm_and_cka_lines(vnorm_map.get(mk), cka_map.get(mk))
        overlay_twin_ylabel_effective = str(cfg.get("overlay_twin_ylabel", ""))
        overlay_metric_abbrev_effective = str(cfg.get("overlay_metric_abbrev") or "")
    elif lo_cfg is not None:
        raw = _build_lowest_norm_var_overlay_by_member_key(members, key_cols, lo_cfg)
        overlay_by_key = {mk: _tag_overlay_lines(v, "--") for mk, v in raw.items()}
        overlay_twin_ylabel_effective = str(
            lo_cfg.get("twin_ylabel") or cfg.get("overlay_twin_ylabel", "")
        )
        overlay_metric_abbrev_effective = str(
            lo_cfg.get("metric_abbrev") or cfg.get("overlay_metric_abbrev") or ""
        )
    elif cka_cfg is not None:
        raw = _build_top_cka_overlay_by_member_key(members, key_cols, cka_cfg)
        overlay_by_key = {mk: _tag_overlay_lines(v, "-.") for mk, v in raw.items()}
        overlay_twin_ylabel_effective = str(cka_cfg.get("twin_ylabel") or cfg.get("overlay_twin_ylabel", ""))
        overlay_metric_abbrev_effective = str(cfg.get("overlay_metric_abbrev") or "")
    elif zs_cfg is not None:
        raw = _build_top_cohens_d_overlay_by_member_key(members, key_cols, zs_cfg)
        overlay_by_key = {mk: _tag_overlay_lines(v, "-.") for mk, v in raw.items()}
        overlay_twin_ylabel_effective = str(zs_cfg.get("twin_ylabel") or cfg.get("overlay_twin_ylabel", ""))
        overlay_metric_abbrev_effective = str(cfg.get("overlay_metric_abbrev") or "")

    has_any = (
        fo_cfg is not None or lo_cfg is not None or cka_cfg is not None or zs_cfg is not None
    )
    return {
        "fo_cfg": fo_cfg,
        "lo_cfg": lo_cfg,
        "cka_cfg": cka_cfg,
        "zs_cfg": zs_cfg,
        "overlay_by_key": overlay_by_key,
        "overlay_twin_ylabel_effective": overlay_twin_ylabel_effective,
        "overlay_metric_abbrev_effective": overlay_metric_abbrev_effective,
        "has_any_overlay": has_any,
    }


def _pool_init_worker(
    parquet_path: str,
    width_in: float,
    font_size: float,
    line_width: float,
) -> None:
    global _WORKER_DATASET
    import matplotlib as _mpl

    _mpl.use("Agg")
    _ensure_plot_utils()
    from plot_utils import setup_plot_style

    setup_plot_style(width_in=width_in, font_size=font_size, linewidth=line_width)
    _WORKER_DATASET = ds.dataset(parquet_path, format="parquet")


def _pool_worker_task(task: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_DATASET
    if _WORKER_DATASET is None:
        raise RuntimeError("Pool worker dataset not initialized")
    tag = task["tag"]
    rec = task["record"]
    plot_dir = Path(task["plot_dir"])
    key_cols = task["key_cols"]
    value_col = task["value_col"]
    width_in = float(task["width_in"])
    figsize = tuple(task["figsize"]) if task.get("figsize") else None
    line_width = float(task["line_width"])
    savefig_dpi = int(task["savefig_dpi"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    sid = _series_id(rec, key_cols)
    y = _fetch_inbound_for_member(_WORKER_DATASET, rec, key_cols, value_col)
    stem = _safe_stem(tag, sid) + str(task.get("stem_suffix", "") or "")
    if y is None:
        return {"plot_dir": str(plot_dir), "series_id": sid, "ok": False}
    out_png = plot_dir / f"{stem}.png"
    out_pdf = plot_dir / f"{stem}.pdf"
    out_csv = plot_dir / f"{stem}_timeseries.csv"
    ymax = task.get("y_max")
    floor_y_zero = bool(task.get("floor_y_zero", True))
    flines = task.get("fisher_lines")
    overlay_feat_csv = (
        plot_dir / f"{stem}_overlay_features.csv" if flines else None
    )
    cdir_name = plot_dir.name
    _plot_series(
        y,
        out_png,
        out_pdf,
        width_in=width_in,
        figsize=figsize,
        line_width=line_width,
        savefig_dpi=savefig_dpi,
        y_max=float(ymax) if ymax is not None else None,
        floor_y_zero=floor_y_zero,
        fisher_lines=flines,
        overlay_twin_ylabel=task.get("overlay_twin_ylabel"),
        overlay_metric_abbrev=str(task.get("overlay_metric_abbrev", "FR")),
        overlay_legend_ncol=int(task.get("overlay_legend_ncol", 2)),
        overlay_width_in=(
            float(task["overlay_width_in"]) if task.get("overlay_width_in") is not None else None
        ),
        overlay_hide_twin_y_axis=bool(task.get("overlay_hide_twin_y_axis", True)),
        overlay_print_z_tsfresh_panel=bool(task.get("overlay_print_z_tsfresh_panel", True)),
        overlay_show_twin_reference_lines=bool(task.get("overlay_show_twin_reference_lines", True)),
        overlay_legend_text_only=bool(task.get("overlay_legend_text_only", False)),
        overlay_legend_fontsize=(
            float(task["overlay_legend_fontsize"]) if task.get("overlay_legend_fontsize") is not None else None
        ),
        overlay_legend_bbox_anchor_y=(
            float(task["overlay_legend_bbox_anchor_y"])
            if task.get("overlay_legend_bbox_anchor_y") is not None
            else None
        ),
        overlay_legend_metric_comma_tsfresh=bool(task.get("overlay_legend_metric_comma_tsfresh", False)),
        member_overlay_features_csv=overlay_feat_csv,
        member_record=rec,
        member_key_cols=key_cols,
        member_overlay_cluster_dir=cdir_name,
        member_overlay_plot_tag=tag,
        member_overlay_series_id=sid,
    )
    if task.get("write_timeseries_csv", True):
        _write_timeseries_csv(y, out_csv, value_column=value_col)
    ret: dict[str, Any] = {
        "plot_dir": str(plot_dir),
        "series_id": sid,
        "ok": True,
        "png": out_png.name,
        "pdf": out_pdf.name,
    }
    if task.get("write_timeseries_csv", True):
        ret["csv"] = out_csv.name
    if overlay_feat_csv is not None:
        ret["overlay_features_csv"] = overlay_feat_csv.name
    return ret


def run(cfg: dict[str, Any]) -> None:
    _ensure_plot_utils()
    from plot_utils import setup_plot_style

    parquet_path = os.path.expandvars(str(cfg["input_parquet"]))
    out_dir: Path | None = None
    if cfg.get("output_dir"):
        out_dir = Path(os.path.expandvars(str(cfg["output_dir"]))).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

    key_cols = _key_columns(cfg)
    value_col = _value_column(cfg)
    width_in = float(cfg.get("width_in", 5.0))
    font_size = float(cfg.get("font_size", 9))
    line_width = float(cfg.get("line_width", 1.0))
    savefig_dpi = int(cfg.get("savefig_dpi", 300))
    figsize = cfg.get("figsize")
    if figsize is not None:
        figsize = tuple(float(x) for x in figsize)

    n_workers = _resolve_num_workers(cfg)
    members = _expand_members_sources(cfg)
    if not members:
        raise SystemExit(
            "No members: set members_json_paths, members_json_glob, or members in JSON."
        )
    if any(cd is None for _, _, cd in members) and out_dir is None:
        raise SystemExit(
            "output_dir is required when using inline \"members\" (no members.json path)."
        )

    dataset = ds.dataset(parquet_path, format="parquet")
    if value_col not in dataset.schema.names:
        raise SystemExit(
            f"Value column {value_col!r} not in dataset. Columns: {dataset.schema.names}"
        )
    for c in key_cols:
        if c not in dataset.schema.names:
            raise SystemExit(
                f"Key column {c!r} not in dataset. Columns: {dataset.schema.names}"
            )
    del dataset

    subdir = cfg.get("output_subdir", "member_plots")
    fallback_dir = (out_dir / subdir) if out_dir is not None else None

    floor_y_zero, share_y_max = _y_axis_flags(cfg)
    y_fixed_max = _y_axis_fixed_max(cfg)
    if y_fixed_max is not None:
        share_y_max = False

    variant_jobs = _normalize_plot_variant_jobs(cfg)

    parent_max: dict[Path, float] = {}
    if share_y_max and y_fixed_max is None:
        scan = ds.dataset(parquet_path, format="parquet")
        for _, rec, cluster_dir in members:
            y = _fetch_inbound_for_member(scan, rec, key_cols, value_col)
            if y is None or y.size == 0:
                continue
            gk = cluster_dir.resolve().parent if cluster_dir is not None else _INLINE_K_GROUP
            my = float(np.nanmax(y))
            parent_max[gk] = max(parent_max.get(gk, 0.0), my)
        for g in list(parent_max.keys()):
            parent_max[g] = parent_max[g] * 1.05
        del scan

    write_ts_csv = _parse_bool(cfg.get("write_timeseries_csv"), default=True)

    manifest_path: Path | None = None
    raw_manifest = cfg.get("member_sequence_ids_csv")
    if raw_manifest is not None and str(raw_manifest).strip():
        manifest_path = Path(os.path.expandvars(str(raw_manifest))).expanduser()
    elif _parse_bool(cfg.get("write_member_sequence_ids_manifest"), default=True):
        mparent = _member_manifest_parent_dir(members)
        if mparent is not None:
            manifest_path = mparent / "member_plot_sequence_ids.csv"
        elif fallback_dir is not None:
            manifest_path = Path(fallback_dir) / "member_plot_sequence_ids.csv"
    if manifest_path is not None:
        _write_member_sequence_ids_manifest(manifest_path, members, key_cols)

    cluster_plots: dict[Path, list[dict[str, str]]] = defaultdict(list)
    cluster_missing: dict[Path, set[str]] = defaultdict(set)
    missing_all: set[str] = set()

    def _legend_fs_raw(eff: dict[str, Any]) -> float | None:
        v = eff.get("overlay_legend_fontsize")
        if v is None:
            return None
        return float(v)

    def _legend_bbox_y_raw(eff: dict[str, Any]) -> float | None:
        v = eff.get("overlay_legend_bbox_anchor_y")
        if v is None:
            return None
        return float(v)

    if n_workers <= 1:
        setup_plot_style(width_in=width_in, font_size=font_size, linewidth=line_width)
        dataset = ds.dataset(parquet_path, format="parquet")
        variant_bundles: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for stem_suffix, eff_cfg in variant_jobs:
            variant_bundles.append(
                (stem_suffix, eff_cfg, _build_overlay_bundle(eff_cfg, members, key_cols))
            )
        for tag, rec, cluster_dir in members:
            plot_dir = cluster_dir if cluster_dir is not None else fallback_dir
            assert plot_dir is not None
            plot_dir.mkdir(parents=True, exist_ok=True)

            sid = _series_id(rec, key_cols)
            y = _fetch_inbound_for_member(dataset, rec, key_cols, value_col)
            if y is None:
                missing_all.add(sid)
                cluster_missing[plot_dir].add(sid)
                continue
            gk = cluster_dir.resolve().parent if cluster_dir is not None else _INLINE_K_GROUP
            if y_fixed_max is not None:
                ymax = float(y_fixed_max)
            else:
                ymax = parent_max.get(gk) if share_y_max else None
            mk = _member_key(rec, key_cols)
            for stem_suffix, eff_cfg, bundle in variant_bundles:
                overlay_by_key = bundle["overlay_by_key"]
                overlay_twin_ylabel_effective = bundle["overlay_twin_ylabel_effective"]
                overlay_metric_abbrev_effective = bundle["overlay_metric_abbrev_effective"]
                has_any_overlay = bundle["has_any_overlay"]
                overlay_legend_ncol = max(1, int(eff_cfg.get("overlay_legend_ncol", 2)))
                overlay_width_in = eff_cfg.get("overlay_width_in")
                overlay_hide_twin_y_axis = _parse_bool(eff_cfg.get("overlay_hide_twin_y_axis"), default=True)
                overlay_print_z_tsfresh_panel = _parse_bool(
                    eff_cfg.get("overlay_print_z_tsfresh_panel"), default=True
                )
                overlay_show_twin_reference_lines = _parse_bool(
                    eff_cfg.get("overlay_show_twin_reference_lines"), default=True
                )
                overlay_legend_text_only = _parse_bool(eff_cfg.get("overlay_legend_text_only"), default=False)
                leg_fs = _legend_fs_raw(eff_cfg)
                leg_bbox_y = _legend_bbox_y_raw(eff_cfg)
                stem = _safe_stem(tag, sid) + stem_suffix
                flines = None
                if overlay_by_key is not None:
                    flines = overlay_by_key.get(mk)
                out_png = plot_dir / f"{stem}.png"
                out_pdf = plot_dir / f"{stem}.pdf"
                out_csv = plot_dir / f"{stem}_timeseries.csv"
                overlay_feat_csv = plot_dir / f"{stem}_overlay_features.csv" if flines else None
                _plot_series(
                    y,
                    out_png,
                    out_pdf,
                    width_in=width_in,
                    figsize=figsize,
                    line_width=line_width,
                    savefig_dpi=savefig_dpi,
                    y_max=ymax,
                    floor_y_zero=floor_y_zero,
                    fisher_lines=flines,
                    overlay_twin_ylabel=overlay_twin_ylabel_effective if has_any_overlay else None,
                    overlay_metric_abbrev=overlay_metric_abbrev_effective,
                    overlay_legend_ncol=overlay_legend_ncol,
                    overlay_width_in=float(overlay_width_in) if overlay_width_in is not None else None,
                    overlay_hide_twin_y_axis=overlay_hide_twin_y_axis,
                    overlay_print_z_tsfresh_panel=overlay_print_z_tsfresh_panel,
                    overlay_show_twin_reference_lines=overlay_show_twin_reference_lines,
                    overlay_legend_text_only=overlay_legend_text_only,
                    overlay_legend_fontsize=leg_fs,
                    overlay_legend_bbox_anchor_y=leg_bbox_y,
                    overlay_legend_metric_comma_tsfresh=_parse_bool(
                        eff_cfg.get("overlay_legend_metric_comma_tsfresh"), default=False
                    ),
                    member_overlay_features_csv=overlay_feat_csv,
                    member_record=rec,
                    member_key_cols=key_cols,
                    member_overlay_cluster_dir=plot_dir.name,
                    member_overlay_plot_tag=tag,
                    member_overlay_series_id=sid,
                )
                entry: dict[str, str] = {
                    "series_id": sid,
                    "png": out_png.name,
                    "pdf": out_pdf.name,
                }
                if write_ts_csv:
                    _write_timeseries_csv(y, out_csv, value_column=value_col)
                    entry["csv"] = out_csv.name
                cluster_plots[plot_dir].append(entry)
    else:
        print(f"[plot_member_timeseries] parallel num_workers={n_workers}", flush=True)
        variant_bundles_par: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for stem_suffix, eff_cfg in variant_jobs:
            variant_bundles_par.append(
                (stem_suffix, eff_cfg, _build_overlay_bundle(eff_cfg, members, key_cols))
            )
        tasks: list[dict[str, Any]] = []
        for tag, rec, cluster_dir in members:
            plot_dir = cluster_dir if cluster_dir is not None else fallback_dir
            assert plot_dir is not None
            gk = cluster_dir.resolve().parent if cluster_dir is not None else _INLINE_K_GROUP
            if y_fixed_max is not None:
                ymax = float(y_fixed_max)
            else:
                ymax = parent_max.get(gk) if share_y_max else None
            mk = _member_key(rec, key_cols)
            for stem_suffix, eff_cfg, bundle in variant_bundles_par:
                overlay_by_key = bundle["overlay_by_key"]
                overlay_twin_ylabel_effective = bundle["overlay_twin_ylabel_effective"]
                overlay_metric_abbrev_effective = bundle["overlay_metric_abbrev_effective"]
                has_any_overlay = bundle["has_any_overlay"]
                overlay_legend_ncol = max(1, int(eff_cfg.get("overlay_legend_ncol", 2)))
                overlay_width_in = eff_cfg.get("overlay_width_in")
                overlay_hide_twin_y_axis = _parse_bool(eff_cfg.get("overlay_hide_twin_y_axis"), default=True)
                overlay_print_z_tsfresh_panel = _parse_bool(
                    eff_cfg.get("overlay_print_z_tsfresh_panel"), default=True
                )
                overlay_show_twin_reference_lines = _parse_bool(
                    eff_cfg.get("overlay_show_twin_reference_lines"), default=True
                )
                overlay_legend_text_only = _parse_bool(eff_cfg.get("overlay_legend_text_only"), default=False)
                leg_fs = _legend_fs_raw(eff_cfg)
                leg_bbox_y = _legend_bbox_y_raw(eff_cfg)
                flines = None
                if overlay_by_key is not None:
                    flines = overlay_by_key.get(mk)
                sid_par = _series_id(rec, key_cols)
                tasks.append(
                    {
                        "tag": tag,
                        "record": rec,
                        "plot_dir": str(plot_dir),
                        "member_overlay_cluster_dir": plot_dir.name,
                        "member_overlay_plot_tag": tag,
                        "member_overlay_series_id": sid_par,
                        "key_cols": key_cols,
                        "value_col": value_col,
                        "width_in": width_in,
                        "figsize": list(figsize) if figsize is not None else None,
                        "line_width": line_width,
                        "savefig_dpi": savefig_dpi,
                        "y_max": ymax,
                        "floor_y_zero": floor_y_zero,
                        "fisher_lines": flines,
                        "write_timeseries_csv": write_ts_csv,
                        "stem_suffix": stem_suffix,
                        "overlay_twin_ylabel": overlay_twin_ylabel_effective if has_any_overlay else None,
                        "overlay_metric_abbrev": overlay_metric_abbrev_effective,
                        "overlay_legend_ncol": overlay_legend_ncol,
                        "overlay_width_in": overlay_width_in,
                        "overlay_hide_twin_y_axis": overlay_hide_twin_y_axis,
                        "overlay_print_z_tsfresh_panel": overlay_print_z_tsfresh_panel,
                        "overlay_show_twin_reference_lines": overlay_show_twin_reference_lines,
                        "overlay_legend_text_only": overlay_legend_text_only,
                        "overlay_legend_fontsize": leg_fs,
                        "overlay_legend_bbox_anchor_y": leg_bbox_y,
                        "overlay_legend_metric_comma_tsfresh": _parse_bool(
                            eff_cfg.get("overlay_legend_metric_comma_tsfresh"), default=False
                        ),
                    }
                )
        initargs = (parquet_path, width_in, font_size, line_width)
        n_tasks = len(tasks)
        chunksize = max(1, min(500, n_tasks // (n_workers * 8) or 1))
        with Pool(
            processes=n_workers,
            initializer=_pool_init_worker,
            initargs=initargs,
        ) as pool:
            results = pool.map(_pool_worker_task, tasks, chunksize=chunksize)
        for r in results:
            pd = Path(r["plot_dir"])
            sid = r["series_id"]
            if r.get("ok"):
                pe: dict[str, str] = {
                    "series_id": sid,
                    "png": r["png"],
                    "pdf": r["pdf"],
                }
                if r.get("csv"):
                    pe["csv"] = r["csv"]
                cluster_plots[pd].append(pe)
            else:
                missing_all.add(sid)
                cluster_missing[pd].add(sid)

    manifest_name = cfg.get("manifest_name", "member_plots_manifest.json")
    all_cluster_dirs = sorted(set(cluster_plots) | set(cluster_missing))
    for d in all_cluster_dirs:
        mpath = d / manifest_name
        miss = cluster_missing.get(d, set())
        manifest = {
            "input_parquet": parquet_path,
            "cluster_dir": str(d),
            "n_plotted": len(cluster_plots.get(d, [])),
            "n_missing": len(miss),
            "plots": cluster_plots.get(d, []),
            "missing_series_ids": sorted(miss),
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if out_dir is not None and cfg.get("write_combined_manifest", True):
        combined_path = out_dir / manifest_name
        combined = {
            "input_parquet": parquet_path,
            "n_plotted": sum(len(v) for v in cluster_plots.values()),
            "n_missing": len(missing_all),
            "missing_series_ids": sorted(missing_all),
            "per_cluster_manifests": [str(d / manifest_name) for d in all_cluster_dirs],
        }
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

    miss_sorted = sorted(missing_all)
    print(
        f"Wrote plots next to members.json under {len(all_cluster_dirs)} cluster dir(s); "
        f"missing series: {len(miss_sorted)}."
    )
    if miss_sorted:
        for s in miss_sorted[:20]:
            print(f"  missing: {s}")
        if len(miss_sorted) > 20:
            print(f"  ... and {len(miss_sorted) - 20} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        required=True,
        help="Path to JSON config (input_parquet, members_*; output_dir optional unless inline members)",
    )
    args = ap.parse_args()
    cfg = _load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
