"""Extract series representations: modes ``final``, ``detailed`` (single-stream Chronos+quantiles), or ``twin_ip`` (NetBurst)."""

from __future__ import annotations

import argparse
import concurrent.futures
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


def _write_parquet_shard(rows, shard_path: str) -> int:
    pd.DataFrame(rows).to_parquet(shard_path, index=False)
    return len(rows)


class _AsyncParquetShardWriter:
    def __init__(self, output_dir: str, workers: int = 2, max_inflight: int = 8):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.max_inflight = max(1, int(max_inflight))
        self._futures = []
        self._shard_idx = 0
        self.total_rows = 0

    def submit_rows(self, rows) -> None:
        if not rows:
            return
        shard_path = os.path.join(self.output_dir, f"part-{self._shard_idx:06d}.parquet")
        self._shard_idx += 1
        fut = self.executor.submit(_write_parquet_shard, rows, shard_path)
        self._futures.append(fut)
        if len(self._futures) >= self.max_inflight:
            done, not_done = concurrent.futures.wait(
                self._futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for d in done:
                self.total_rows += int(d.result())
            self._futures = list(not_done)

    def close(self) -> None:
        if self._futures:
            done, _ = concurrent.futures.wait(self._futures)
            for d in done:
                self.total_rows += int(d.result())
            self._futures = []
        self.executor.shutdown(wait=True)


def _write_output(df: pd.DataFrame, output_csv: str, output_parquet: Optional[str]) -> None:
    if output_parquet:
        df.to_parquet(output_parquet, index=False)
        print(f"Saved {len(df)} rows to {output_parquet} (parquet)")
    else:
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} rows to {output_csv} (csv)")


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
    p.add_argument("--output_parquet", default=None, help="If set, write parquet instead of CSV")
    p.add_argument("--isCSV", type=int, default=None)
    p.add_argument("--outbound_only", action="store_true")
    p.add_argument(
        "--all_timesteps",
        action="store_true",
        help="Store one representation row per time step (adds `time_step` column).",
    )
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
    time_steps = []

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
                embeddings = batch_emb.detach().cpu().numpy()
                if args.all_timesteps:
                    for j in range(len(batch_series)):
                        series_idx = i + j
                        series_embeddings = embeddings[j]
                        n_tokens = series_embeddings.shape[0]
                        all_representations.extend(series_embeddings)
                        series_ids.extend([series_idx] * n_tokens)
                        time_steps.extend(range(n_tokens))
                else:
                    representations = embeddings[:, -1, :]
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
    if args.all_timesteps:
        df_representations["time_step"] = time_steps
    for i in range(n_dims):
        df_representations[f"dim_{i}"] = representations_array[:, i]

    _write_output(df_representations, args.output_csv, args.output_parquet)
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
    p.add_argument("--output_parquet", default=None, help="If set, write parquet instead of CSV")
    p.add_argument("--isCSV", type=int, default=None)
    p.add_argument("--outbound_only", action="store_true")
    p.add_argument("--include_raw_data", action="store_true")
    p.add_argument("--include_token_level", action="store_true")
    p.add_argument(
        "--all_timesteps",
        action="store_true",
        help="Store one row per time step (`time_step` + repr_dim_*).",
    )
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
                    if args.all_timesteps:
                        for ts_idx, ts_repr in enumerate(series_embeddings):
                            result = {
                                "series_id": i + j,
                                "series_length": series_length,
                                "n_tokens": series_embeddings.shape[0],
                                "time_step": ts_idx,
                                **metadata,
                            }
                            for dim_idx, val in enumerate(ts_repr):
                                result[f"repr_dim_{dim_idx}"] = val
                            if args.include_raw_data:
                                result["raw_series"] = json.dumps(series)
                            results_data.append(result)
                    else:
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

    _write_output(pd.DataFrame(results_data), args.output_csv, args.output_parquet)


def _parser_twin_ip() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Twin-head: last hidden state per (bi, ibg) series with IP columns.")
    p.add_argument("parquet_root")
    p.add_argument("--model", default=".", help="Twin-head checkpoint directory")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ips_csv", default="IpsToConsider.csv")
    p.add_argument("--output_csv", default="ip_representations.csv")
    p.add_argument("--output_parquet", default=None, help="If set, write parquet instead of CSV")
    p.add_argument(
        "--output_parquet_dir",
        default=None,
        help="If set, stream sharded parquet writes to this directory (lower memory than single-file parquet).",
    )
    p.add_argument("--write_workers", type=int, default=2, help="Parallel writer workers for --output_parquet_dir")
    p.add_argument("--max_inflight_writes", type=int, default=8, help="Max queued async shard writes")
    p.add_argument(
        "--ip_prefix_filter",
        type=str,
        default=None,
        help="Keep rows whose ip starts with this prefix",
    )
    p.add_argument(
        "--ip_suffix_filter",
        type=str,
        default=None,
        help="Keep rows whose ip ends with this suffix",
    )
    p.add_argument(
        "--all_timesteps",
        action="store_true",
        help="Store one row per hidden time step (`time_step` + repr_dim_*).",
    )
    p.add_argument(
        "--include_targets",
        action="store_true",
        help="Include aligned target_bi/target_ibg columns in output rows.",
    )
    p.add_argument(
        "--repr_source",
        type=str,
        choices=("decoder", "encoder"),
        default="decoder",
        help="Representation source: decoder hidden (default) or encoder hidden states.",
    )
    return p


def run_twin_ip() -> None:
    args = _parser_twin_ip().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.output_parquet and args.output_parquet_dir:
        raise ValueError("Use only one of --output_parquet or --output_parquet_dir.")

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
        ip_prefix_filter=args.ip_prefix_filter,
        ip_suffix_filter=args.ip_suffix_filter,
    )
    print(f"Loaded {len(ip_series_data)} IP-(bi,ibg) pairs")

    print("Loading trained model...")
    predictor = load_twin_head_from_dir(args.model, device, print_head_norms=True)
    print(f"Representation source: {args.repr_source}")

    results_data = []
    shard_writer = None
    if args.output_parquet_dir:
        shard_writer = _AsyncParquetShardWriter(
            output_dir=args.output_parquet_dir,
            workers=args.write_workers,
            max_inflight=args.max_inflight_writes,
        )
        print(
            f"Streaming parquet shards to {args.output_parquet_dir} "
            f"(workers={args.write_workers}, max_inflight={args.max_inflight_writes})"
        )
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
                ctx_ibg = predictor.pipeline._prepare_and_validate_context(context=ibg_tensors)
                (
                    ids_bi,
                    mask_bi,
                    ids_ibg,
                    mask_ibg,
                    local_bi,
                    local_ibg,
                    *_,
                ) = predictor._tokenize_dual_streams(ctx_bi, ctx_ibg)

                comb_mask = (mask_bi & mask_ibg).to(device)

                dummy = torch.empty((ids_bi.size(0), 1), dtype=torch.long, device=device)
                fwd_kw = dict(
                    input_ids1=ids_bi.to(device),
                    input_ids2=ids_ibg.to(device),
                    attention_mask=comb_mask,
                    decoder_input_ids=dummy,
                    type_id=0,
                    cross_attend=predictor.use_cross_attn,
                )
                if local_bi is not None:
                    fwd_kw["local_ids1"] = local_bi.to(device)
                    fwd_kw["local_ids2"] = local_ibg.to(device)
                if args.repr_source == "encoder":
                    tok_emb = predictor.model.fused_stream_embeddings(
                        ids_bi.to(device),
                        ids_ibg.to(device),
                        local_ids1=local_bi.to(device) if local_bi is not None else None,
                        local_ids2=local_ibg.to(device) if local_ibg is not None else None,
                    )
                    enc_out = predictor.model.model.encoder(
                        inputs_embeds=tok_emb,
                        attention_mask=comb_mask,
                        return_dict=True,
                    )
                    hidden = enc_out.last_hidden_state
                else:
                    _, hidden = predictor.model.forward_with_embeddings(**fwd_kw)
                    hidden = hidden.squeeze(1)
                hidden_np = hidden.detach().cpu().numpy()
                series_representations = hidden_np[:, -1, :]

                batch_results = []
                for j, (ip, source_file, (bi_ser, ibg_ser)) in enumerate(
                    zip(batch_ips, batch_source_files, batch_pairs)
                ):
                    base_result = {
                        "ip": ip,
                        "source_file": source_file,
                        "series_id": i + j,
                        "bi_length": len(bi_ser),
                        "ibg_length": len(ibg_ser),
                        "n_tokens": hidden.shape[1],
                        "repr_source": args.repr_source,
                    }
                    if args.all_timesteps:
                        max_ts = len(hidden_np[j])
                        if args.include_targets:
                            max_ts = min(max_ts, len(bi_ser), len(ibg_ser))
                        for ts_idx in range(max_ts):
                            ts_repr = hidden_np[j][ts_idx]
                            result = {**base_result, "time_step": ts_idx}
                            if args.include_targets:
                                result["target_bi"] = float(bi_ser[ts_idx])
                                result["target_ibg"] = float(ibg_ser[ts_idx])
                            for dim_idx, val in enumerate(ts_repr):
                                result[f"repr_dim_{dim_idx}"] = val
                            batch_results.append(result)
                    else:
                        series_repr = series_representations[j]
                        result = dict(base_result)
                        if args.include_targets and len(bi_ser) > 0 and len(ibg_ser) > 0:
                            result["target_bi"] = float(bi_ser[-1])
                            result["target_ibg"] = float(ibg_ser[-1])
                        for dim_idx, val in enumerate(series_repr):
                            result[f"repr_dim_{dim_idx}"] = val
                        batch_results.append(result)
                if shard_writer is not None:
                    shard_writer.submit_rows(batch_results)
                else:
                    results_data.extend(batch_results)

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
        f"computed {len(results_data) if shard_writer is None else 'streamed'} representations"
    )

    if shard_writer is not None:
        shard_writer.close()
        print(
            f"Saved {shard_writer.total_rows} rows as parquet shards under "
            f"{args.output_parquet_dir}"
        )
    else:
        _write_output(pd.DataFrame(results_data), args.output_csv, args.output_parquet)


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
