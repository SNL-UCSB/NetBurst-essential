"""Extract series representations: modes ``final``, ``detailed`` (single-stream Chronos+quantiles), or ``twin_ip`` (NetBurst)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Optional, Set

import numpy as np
import pandas as pd
import torch

from netburst.data import load_cleaned_series, load_ibg_bi_with_ip
from netburst.model import load_chronos_bin_predictor_from_checkpoint, load_twin_head_from_dir


def _parser_final() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-stream Chronos: last-token embeddings (dim_* columns).")
    p.add_argument("parquet_root", help="Root path to parquet data (or CSV when --isCSV)")
    p.add_argument("--model", default=".", help="Directory with boundaries.pkl and chronos_best.pt")
    p.add_argument(
        "--base_chronos",
        default="amazon/chronos-t5-small",
        help="HF model id or local path for base Chronos weights",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_len", type=int, default=None, help="Maximum length of series")
    p.add_argument("--nonhierarchical", action="store_true")
    p.add_argument("--output_csv", default="final_representations.csv")
    p.add_argument("--isCSV", type=int, default=None)
    p.add_argument("--outbound_only", action="store_true")
    return p


def run_final() -> None:
    args = _parser_final().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.isCSV:
        df = pd.read_csv(args.parquet_root)
        all_series = df.values.tolist()
    else:
        all_series = load_cleaned_series(
            args.parquet_root,
            min_len=10,
            max_len=args.max_len,
            limit=args.limit,
            nonHierarchical=args.nonhierarchical,
            outbound_only=args.outbound_only,
        )

    print(f"Loaded {len(all_series)} series")
    print("Loading trained model...")
    predictor = load_chronos_bin_predictor_from_checkpoint(args.model, args.base_chronos, device)
    print("Model loaded successfully")

    all_representations = []
    series_ids = []

    print("Extracting representations...")
    with torch.no_grad():
        for i in range(0, len(all_series), args.batch_size):
            batch_series = all_series[i : i + args.batch_size]
            try:
                batch_data = []
                for series in batch_series:
                    batch_data.append(np.array(series, dtype=np.float32))
                batch_np = np.array(batch_data, dtype=np.float32)
                batch_tensor = torch.from_numpy(batch_np).to(device)
                batch_emb, _tokenizer_state = predictor.pipeline.embed(batch_tensor)
                representations = batch_emb[:, -1, :].detach().cpu().numpy()
                all_representations.extend(representations)
                for j in range(len(batch_series)):
                    series_ids.append(i + j)
                if (i // args.batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch_series)}/{len(all_series)} series")
            except Exception as e:
                print(f"Error processing batch {i//args.batch_size}: {e}")
                continue

    representations_array = np.array(all_representations)
    n_dims = representations_array.shape[1]
    df_representations = pd.DataFrame()
    df_representations["series_id"] = series_ids
    for i in range(n_dims):
        df_representations[f"dim_{i}"] = representations_array[:, i]

    df_representations.to_csv(args.output_csv, index=False)
    print(f"Saved {len(representations_array)} representations to {args.output_csv}")
    print(f"Representation shape: {representations_array.shape}")


def _parser_detailed() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-stream Chronos: detailed rows with optional token-level columns.")
    p.add_argument("parquet_root")
    p.add_argument("--model", default=".")
    p.add_argument("--base_chronos", default="amazon/chronos-t5-small")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--nonhierarchical", action="store_true")
    p.add_argument("--output_csv", default="detailed_representations.csv")
    p.add_argument("--isCSV", type=int, default=None)
    p.add_argument("--outbound_only", action="store_true")
    p.add_argument("--include_raw_data", action="store_true")
    p.add_argument("--include_token_level", action="store_true")
    return p


def run_detailed() -> None:
    args = _parser_detailed().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.isCSV:
        df = pd.read_csv(args.parquet_root)
        all_series = df.values.tolist()
        series_metadata = [{"source": "csv", "row_idx": i} for i in range(len(all_series))]
    else:
        all_series = load_cleaned_series(
            args.parquet_root,
            min_len=10,
            max_len=args.max_len,
            limit=args.limit,
            nonHierarchical=args.nonhierarchical,
            outbound_only=args.outbound_only,
        )
        series_metadata = [{"source": "parquet", "series_idx": i} for i in range(len(all_series))]

    print(f"Loaded {len(all_series)} series")
    print("Loading trained model...")
    predictor = load_chronos_bin_predictor_from_checkpoint(args.model, args.base_chronos, device)
    print("Model loaded successfully")

    results_data = []
    print("Extracting detailed representations...")
    with torch.no_grad():
        for i in range(0, len(all_series), args.batch_size):
            batch_series = all_series[i : i + args.batch_size]
            batch_metadata = series_metadata[i : i + args.batch_size]
            try:
                batch_data = [np.array(s, dtype=np.float32) for s in batch_series]
                batch_np = np.array(batch_data, dtype=np.float32)
                batch_tensor = torch.from_numpy(batch_np).to(device)
                batch_emb, _ = predictor.pipeline.embed(batch_tensor)
                embeddings = batch_emb.detach().cpu().numpy()
                for j, (series, metadata) in enumerate(zip(batch_series, batch_metadata)):
                    series_length = len(series)
                    series_embeddings = embeddings[j]
                    series_repr = series_embeddings[-1, :]
                    result = {
                        "series_id": i + j,
                        "series_length": series_length,
                        "n_tokens": series_embeddings.shape[0],
                        **metadata,
                    }
                    for dim_idx, val in enumerate(series_repr):
                        result[f"repr_dim_{dim_idx}"] = val
                    if args.include_raw_data:
                        result["raw_series"] = json.dumps(series)
                    if args.include_token_level:
                        for dim_idx in range(series_embeddings.shape[1]):
                            dim_values = series_embeddings[:, dim_idx]
                            result[f"token_repr_dim_{dim_idx}"] = ",".join(map(str, dim_values))
                    results_data.append(result)
                if (i // args.batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch_series)}/{len(all_series)} series")
            except Exception as e:
                print(f"Error processing batch {i//args.batch_size}: {e}")
                continue

    pd.DataFrame(results_data).to_csv(args.output_csv, index=False)
    print(f"Saved {len(results_data)} detailed representations to {args.output_csv}")


def _parser_twin_ip() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Twin-head: last hidden state per (bi, ibg) series with IP columns.")
    p.add_argument("parquet_root")
    p.add_argument("--model", default=".", help="Twin-head checkpoint directory")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ips_csv", default="IpsToConsider.csv")
    p.add_argument("--output_csv", default="ip_representations.csv")
    p.add_argument(
        "--ip_suffix_filter",
        type=str,
        default=None,
        help="Keep rows whose ip ends with this suffix",
    )
    return p


def run_twin_ip() -> None:
    args = _parser_twin_ip().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    allowed_ips: Optional[Set[str]] = None
    if os.path.exists(args.ips_csv):
        tmp_df = pd.read_csv(args.ips_csv)
        if "ip" in tmp_df.columns:
            allowed_ips = set(tmp_df["ip"])
            print(f"Loaded {len(allowed_ips)} allowed IPs from {args.ips_csv}")
    if allowed_ips is None:
        print("No IP filter found, processing all IPs")

    ip_series_data = load_ibg_bi_with_ip(
        args.parquet_root,
        allowed_ips,
        min_len=10,
        ip_suffix_filter=args.ip_suffix_filter,
    )
    print(f"Loaded {len(ip_series_data)} IP-(bi,ibg) pairs")

    print("Loading trained model...")
    predictor = load_twin_head_from_dir(args.model, device, print_head_norms=True)

    results_data = []
    print("Extracting IP-based representations...")
    t_repr_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(ip_series_data), args.batch_size):
            print(f"Processing batch {i//args.batch_size + 1}...")
            batch_data = ip_series_data[i : i + args.batch_size]
            batch_ips = [ip for ip, _, _ in batch_data]
            batch_source_files = [sf for _, sf, _ in batch_data]
            batch_pairs = [(bi, ibg) for _, _, (bi, ibg) in batch_data]
            try:
                bi_series, ibg_series = predictor._prep_context(batch_pairs)
                bi_tensors = [torch.tensor(bi, dtype=torch.float32) for bi in bi_series]
                ibg_tensors = [torch.tensor(ibg, dtype=torch.float32) for ibg in ibg_series]

                ctx_bi = predictor.pipeline._prepare_and_validate_context(context=bi_tensors)
                ids_bi, mask_bi, _ = predictor.tokenizer_bi.context_input_transform(ctx_bi)

                ctx_ibg = predictor.pipeline._prepare_and_validate_context(context=ibg_tensors)
                ids_ibg, mask_ibg, _ = predictor.tokenizer_ibg.context_input_transform(ctx_ibg)

                comb_mask = (mask_bi & mask_ibg).to(device)

                dummy = torch.empty((ids_bi.size(0), 1), dtype=torch.long, device=device)
                _, hidden = predictor.model.forward_with_embeddings(
                    input_ids1=ids_bi.to(device),
                    input_ids2=ids_ibg.to(device),
                    attention_mask=comb_mask,
                    decoder_input_ids=dummy,
                    type_id=0,
                    cross_attend=predictor.use_cross_attn,
                )
                hidden = hidden.squeeze(1)
                series_representations = hidden[:, -1, :].detach().cpu().numpy()

                for j, (ip, source_file, (bi_ser, ibg_ser)) in enumerate(
                    zip(batch_ips, batch_source_files, batch_pairs)
                ):
                    series_repr = series_representations[j]
                    result = {
                        "ip": ip,
                        "source_file": source_file,
                        "series_id": i + j,
                        "bi_length": len(bi_ser),
                        "ibg_length": len(ibg_ser),
                        "n_tokens": hidden.shape[1],
                    }
                    for dim_idx, val in enumerate(series_repr):
                        result[f"repr_dim_{dim_idx}"] = val
                    results_data.append(result)

                if (i // args.batch_size + 1) % 10 == 0:
                    print(f"Processed {i + len(batch_data)}/{len(ip_series_data)} IP-(bi,ibg) pairs")
            except Exception as e:
                print(f"Error processing batch {i//args.batch_size}: {e}")
                import traceback

                traceback.print_exc()
                continue

    t_repr_elapsed = time.perf_counter() - t_repr_start
    print(
        f"Representation extraction wall time: {t_repr_elapsed:.3f} s; "
        f"computed {len(results_data)} representations"
    )

    pd.DataFrame(results_data).to_csv(args.output_csv, index=False)
    print(f"Saved {len(results_data)} IP representations to {args.output_csv}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python extract_repr.py {final|detailed|twin_ip} [args...]\n"
            "  final     — single-stream Chronos + quantile checkpoint\n"
            "  detailed  — same with optional token-level / raw columns\n"
            "  twin_ip   — NetBurst checkpoint on bi/ibg parquet with IP metadata",
            file=sys.stderr,
        )
        sys.exit(2)
    mode = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if mode == "final":
        run_final()
    elif mode == "detailed":
        run_detailed()
    elif mode == "twin_ip":
        run_twin_ip()
    else:
        print(f"Unknown mode {mode!r}. Use final, detailed, or twin_ip.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
