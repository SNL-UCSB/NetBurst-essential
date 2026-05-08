#!/usr/bin/env python3
"""Emit a JSON config for plot_member_timeseries_from_parquet.py (Slurm / sweep pipelines)."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Path to write the JSON config.")
    ap.add_argument(
        "--input-parquet",
        required=True,
        help="Parquet dataset root (same as plot_member_timeseries input_parquet).",
    )
    ap.add_argument(
        "--members-json-glob",
        required=True,
        help="Glob for cluster_*/members.json (e.g. .../K_10/cluster_*/members.json).",
    )
    ap.add_argument("--value-column", default="inbound")
    ap.add_argument(
        "--key-columns",
        default="ip,source_file",
        help="Comma-separated key columns (default ip,source_file).",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--width-in", type=float, default=5.0)
    ap.add_argument("--font-size", type=float, default=14.0)
    ap.add_argument("--line-width", type=float, default=2.0)
    ap.add_argument("--savefig-dpi", type=int, default=300)
    ap.add_argument(
        "--no-y-floor-zero",
        dest="y_floor_zero",
        action="store_false",
        default=True,
        help="Allow auto y-axis floor (default: pin at zero for cross-cluster comparability).",
    )
    ap.add_argument(
        "--share-y-max-within-k",
        action="store_true",
        help="Use one ymax per K_* parent (all cluster_* dirs under the same K).",
    )
    ap.add_argument(
        "--y-fixed-max",
        type=float,
        default=None,
        help="Fixed y-axis maximum for every plot (e.g. 750000 bytes); min stays 0 with default floor_zero. Overrides --share-y-max-within-k.",
    )
    ap.add_argument(
        "--fisher-overlay",
        action="store_true",
        help="Overlay top Fisher-ranked TSFresh scalars (needs heatmap CSV next to K_* and --tsfresh-csv).",
    )
    ap.add_argument(
        "--tsfresh-csv",
        default=None,
        help="Merged TSFresh table (same as cluster_analysis); required with --fisher-overlay.",
    )
    ap.add_argument("--fisher-top-n", type=int, default=5)
    ap.add_argument("--heatmap-filename", default="feature_order_fisher.csv")
    ap.add_argument(
        "--no-write-timeseries-csv",
        dest="write_timeseries_csv",
        action="store_false",
        default=True,
        help="Do not write {stem}_timeseries.csv next to each figure (default: write CSV).",
    )
    ap.add_argument(
        "--dual-global-filter-overlays",
        action="store_true",
        help="Emit plot_variants: top_cka and top_cohens_d (default filenames overlay_cka_top{N}.csv / overlay_cohen_d_top{N}.csv).",
    )
    ap.add_argument(
        "--overlay-top-n",
        type=int,
        default=10,
        help="top_n for each overlay variant (default 10; must match global_filtered_overlay_prep --local-overlay-top-k).",
    )
    ap.add_argument(
        "--cka-overlay-filename",
        default=None,
        help="Per-cluster CKA-order overlay CSV (default overlay_cka_top{overlay-top-n}.csv).",
    )
    ap.add_argument(
        "--z-overlay-filename",
        default=None,
        help="Per-cluster Cohen-d overlay CSV (default overlay_cohen_d_top{overlay-top-n}.csv).",
    )
    ap.add_argument(
        "--overlay-legend-fontsize",
        type=float,
        default=None,
        help="Optional legend fontsize for twin-axis overlays; omit to match axis tick size in plot_member_timeseries_from_parquet.",
    )
    ap.add_argument(
        "--compact-metric-tsfresh-legend",
        action="store_true",
        help="Legend lines: metric,tsfresh_value feature_name (CKA and Cohen overlays); disables extra metric text panel.",
    )
    ap.add_argument(
        "--member-sequence-ids-csv",
        default=None,
        help="Path for member_plot_sequence_ids.csv (one row per members.json entry). Default: parent of cluster_* dirs.",
    )
    ap.add_argument(
        "--no-member-sequence-ids-manifest",
        dest="write_member_sequence_ids_manifest",
        action="store_false",
        default=True,
        help="Disable writing the member ID manifest CSV used to subset other models.",
    )
    args = ap.parse_args()

    if args.fisher_overlay and args.dual_global_filter_overlays:
        raise SystemExit("Use either --fisher-overlay or --dual-global-filter-overlays, not both.")
    if args.fisher_overlay and not args.tsfresh_csv:
        raise SystemExit("--fisher-overlay requires --tsfresh-csv")
    if args.dual_global_filter_overlays and not args.tsfresh_csv:
        raise SystemExit("--dual-global-filter-overlays requires --tsfresh-csv")

    cfg = {
        "input_parquet": os.path.expandvars(str(args.input_parquet)),
        "members_json_glob": os.path.expandvars(str(args.members_json_glob)),
        "key_columns": [c.strip() for c in str(args.key_columns).split(",") if c.strip()],
        "value_column": str(args.value_column),
        "num_workers": int(args.num_workers),
        "width_in": float(args.width_in),
        "font_size": float(args.font_size),
        "line_width": float(args.line_width),
        "savefig_dpi": int(args.savefig_dpi),
    }
    if (not args.y_floor_zero) or args.share_y_max_within_k or args.y_fixed_max is not None:
        cfg["y_axis"] = {
            "floor_zero": bool(args.y_floor_zero),
            "share_max_within_k_parent": bool(args.share_y_max_within_k),
        }
        if args.y_fixed_max is not None:
            cfg["y_axis"]["fixed_max"] = float(args.y_fixed_max)
    if args.fisher_overlay:
        cfg["fisher_overlay"] = {
            "enabled": True,
            "tsfresh_csv": os.path.expandvars(str(args.tsfresh_csv)),
            "top_n": int(args.fisher_top_n),
            "heatmap_filename": str(args.heatmap_filename),
        }
    if args.dual_global_filter_overlays:
        ts_exp = os.path.expandvars(str(args.tsfresh_csv))
        tn = int(args.overlay_top_n)
        cka_fn = args.cka_overlay_filename or f"overlay_cka_top{tn}.csv"
        z_fn = args.z_overlay_filename or f"overlay_cohen_d_top{tn}.csv"
        cfg["plot_variants"] = [
            {
                "output_suffix": "_ckaorder",
                "top_cka_overlay": {
                    "enabled": True,
                    "tsfresh_csv": ts_exp,
                    "top_n": tn,
                    "filename": str(cka_fn),
                },
            },
            {
                "output_suffix": "_zorder",
                "top_cohens_d_overlay": {
                    "enabled": True,
                    "tsfresh_csv": ts_exp,
                    "top_n": tn,
                    "filename": str(z_fn),
                },
            },
        ]
        if args.overlay_legend_fontsize is not None:
            cfg["overlay_legend_fontsize"] = float(args.overlay_legend_fontsize)
        # CKA/z legends: feature name, then metric value, then member TSFresh value (see plot_member_timeseries).
        cfg["overlay_legend_metric_comma_tsfresh"] = True
        cfg["overlay_print_z_tsfresh_panel"] = False
    if args.compact_metric_tsfresh_legend:
        cfg["overlay_legend_metric_comma_tsfresh"] = True
        cfg["overlay_print_z_tsfresh_panel"] = False
    if not args.write_timeseries_csv:
        cfg["write_timeseries_csv"] = False
    if args.member_sequence_ids_csv:
        cfg["member_sequence_ids_csv"] = os.path.expandvars(str(args.member_sequence_ids_csv))
    cfg["write_member_sequence_ids_manifest"] = bool(args.write_member_sequence_ids_manifest)
    out = Path(os.path.expanduser(os.path.expandvars(args.out)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Wrote plot config -> {out}", flush=True)


if __name__ == "__main__":
    main()
