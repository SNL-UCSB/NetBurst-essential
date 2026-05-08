#!/usr/bin/env python3
"""
CLI to read stored quantiles Parquet and return a scalar BI (or latency) threshold
for a requested quantile and entity type. Used by IBG/BI processing and Slurm drivers.
"""

import argparse
import json
import os
import sys

# Read Parquet without Spark (pyarrow or pandas)
try:
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _read_quantiles_parquet(path: str):
    """Read quantiles Parquet; return list of (entity_type, levels, quantiles)."""
    if _HAS_PYARROW:
        t = pq.read_table(path)
        df = t.to_pandas() if hasattr(t, "to_pandas") else pd.DataFrame(t)
    elif _HAS_PANDAS:
        df = pd.read_parquet(path)
    else:
        raise RuntimeError("Need pyarrow or pandas to read Parquet")
    out = []
    for _, row in df.iterrows():
        et = row["entity_type"]
        lv = row["levels"]
        qv = row["quantiles"]
        if hasattr(lv, "tolist"):
            lv = lv.tolist()
        if hasattr(qv, "tolist"):
            qv = qv.tolist()
        out.append((et, list(lv), list(qv)))
    return out


def get_threshold_for_quantile(levels, quantiles, q: float) -> float:
    """Interpolate threshold for quantile q from (levels, quantiles)."""
    if not levels or not quantiles or len(levels) != len(quantiles):
        raise ValueError("levels and quantiles must be non-empty and same length")
    if q <= levels[0]:
        return float(quantiles[0])
    if q >= levels[-1]:
        return float(quantiles[-1])
    for i in range(len(levels) - 1):
        if levels[i] <= q <= levels[i + 1]:
            if levels[i + 1] == levels[i]:
                return float(quantiles[i])
            t = (q - levels[i]) / (levels[i + 1] - levels[i])
            return float(quantiles[i]) + t * (float(quantiles[i + 1]) - float(quantiles[i]))
    return float(quantiles[-1])


def main():
    parser = argparse.ArgumentParser(
        description="Select BI/latency threshold from stored quantiles by entity type and quantile"
    )
    parser.add_argument(
        "--quantiles_path",
        type=str,
        required=True,
        help="Path to quantiles Parquet (entity_type, levels, quantiles)",
    )
    parser.add_argument(
        "--entity_type",
        type=str,
        required=True,
        help="Entity type key (e.g. ip, ip_service, subnet, ip2ip, ping_latency)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        required=True,
        help="Quantile in [0,1] (e.g. 0.9) to get threshold for",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional: write threshold and metadata to this JSON path for reproducibility",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.quantiles_path) and not os.path.isfile(args.quantiles_path):
        # Parquet can be a dir or a file
        print(f"Error: quantiles path not found: {args.quantiles_path}", file=sys.stderr)
        sys.exit(1)

    rows = _read_quantiles_parquet(args.quantiles_path)
    chosen = None
    for et, levels, quantiles in rows:
        if et == args.entity_type:
            chosen = (levels, quantiles)
            break
    if chosen is None:
        available = [r[0] for r in rows]
        print(
            f"Error: entity_type '{args.entity_type}' not in quantiles. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    levels, quantiles = chosen
    threshold = get_threshold_for_quantile(levels, quantiles, args.quantile)
    print(threshold)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(
                {
                    "entity_type": args.entity_type,
                    "quantile": args.quantile,
                    "threshold": threshold,
                    "quantiles_path": args.quantiles_path,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
