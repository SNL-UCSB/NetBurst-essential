"""Distributed autoregressive inference for the NetBurst model."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pickle
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from eval_metrics_utils import mae_nonzero_gt, mape_nonzero_gt, wasserstein_distance_tails

from netburst.data import (
    load_fires_ibg_bi,
    load_ibg_bi_series,
    load_filtered_ibg_bi,
    load_precomputed_ibgbi_context_forecast,
)
from netburst.model import IntegerIBGBins, load_twin_head_from_dir
from netburst.utils import fano_factor_numpy


def ddp_init(timeout_minutes: int = 60):
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=max(1, int(timeout_minutes))),
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    return rank, world_size, device


def ddp_cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def shard_list_by_rank(x, rank, world_size):
    return [v for i, v in enumerate(x) if (i % world_size) == rank]


def save_on_rank0(obj, path, rank):
    if rank == 0:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_root", help="Root path of your Parquet series data")
    parser.add_argument("--save_pkl", default="ar_inference_compressed2.pkl",
                        help="Base filename; a folder with this stem will contain BI/IBG rank shards")
    parser.add_argument("--model", default="amazon/chronos-t5-small",
                        help="Directory with trained NetBurst model (expects boundaries_bi.pkl, boundaries_ibg.pkl, chronos_best.pt)")
    parser.add_argument("--ips_csv", default=None,
                        help="CSV of allowed keys with columns: ip, or ip,service_port, or subnet")
    parser.add_argument("--fires_parquet", type=str, default=None,
                        help="If set, load (bi, ibg) pairs via Fires schema; else use generic IBG/BI loader")
    parser.add_argument("--ctx_frac", type=float, default=0.70,
                        help="Fraction of each series used as context; remainder is AR horizon "
                             "(ignored with --use_precomputed_context_forecast; lengths come from parquet). "
                             "Also ignored when --max_context_split is set.")
    parser.add_argument(
        "--max_context_split",
        action="store_true",
        help=(
            "For raw (bi, ibg) series only: use the longest possible context prefix — "
            "L = max(min_ctx, T - min_h), H = T - L — so the AR horizon is only the last min_h "
            "steps (subject to T >= min_ctx + min_h). Overrides --ctx_frac."
        ),
    )
    parser.add_argument("--min_ctx", type=int, default=10,
                        help="Minimum context steps for non-precomputed runs and internal 70/30 split. "
                             "Ignored when --use_precomputed_context_forecast is set.")
    parser.add_argument("--min_h", type=int, default=1,
                        help="Minimum forecast horizon steps (precomputed: minimum len(forecast_*); default 1).")
    parser.add_argument("--min_len", type=int, default=2,
                        help="Minimum total length after load: for --use_precomputed_context_forecast, "
                             "minimum len(context)+len(forecast) (default 2 = one context + one forecast). "
                             "For raw (bi,ibg) series, minimum series length.")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Optional cap on series length before splitting into context/horizon")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of (bi, ibg) pairs to load")
    parser.add_argument("--use_precomputed_context_forecast", action="store_true",
                        help="Use context_bi/forecast_bi/context_ibg/forecast_ibg columns from split parquet instead of internal 70/30 split")
    parser.add_argument("--test_split_ratio", type=float, default=None,
                        help="If set, fraction of data to hold out as test split (from the end)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nz_thresh", type=float, default=0.0,
                        help="Threshold used for non-zero metrics and WD masking")
    parser.add_argument("--bi_thresh", type=float, default=0.0,
                        help="BI values (forecast and GT independently) below this threshold are zeroed out "
                             "before computing any metrics (per-head BI, expanded merged, oracle). "
                             "Default 0 means no zeroing.")
    parser.add_argument("--mape_scale", type=float, default=1.0,
                        help="Scale used in NZ_SCALED (MAPE-style) metric")
    parser.add_argument("--max_zero_fill", type=int, default=10000,
                        help="Cap for expansion zero-fill length to avoid huge allocations")
    parser.add_argument("--merged_eval_max_len", type=int, default=-1,
                        help="Merged-timeline eval prefix length (GT merge steps). Negative values "
                             "(default -1) use the full GT merged length. 0 disables merged-timeline "
                             "metrics. Positive N caps to min(N, len(GT merge)); then k is the smallest "
                             "number of GT bursts whose expansion covers that prefix; forecast/oracle "
                             "merges use the first k burst pairs.")
    parser.add_argument("--per_example_csv", type=str, default="per_example_metrics_chronos_netburst.csv",
                        help="Per-example CSV path (relative paths are resolved under save_pkl stem directory)")
    parser.add_argument("--aggregate_json", type=str, default="final_metrics_chronos_netburst.json",
                        help="Aggregate metrics JSON path (relative paths are resolved under save_pkl stem directory)")
    # Sampling / decoding controls
    parser.add_argument("--sampling", dest="sampling", action="store_true",
                        help="Enable sampling instead of greedy argmax during AR decoding")
    parser.add_argument("--no_sampling", dest="sampling", action="store_false",
                        help="Disable sampling and use greedy argmax during AR decoding")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples per AR step; median in value-space is used")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for sampling; <=0 switches to greedy")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K filtering (0 disables)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-P (nucleus) filtering in [0,1] (1.0 disables)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed for reproducible sampling")
    parser.add_argument("--debug_sanity", type=int, default=0,
                        help="If >0, compute training-style next-token accuracy on context for the first N series to verify checkpoint loading and tokenization alignment")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example BI/IBG forecasts vs ground truth and per-example metrics.")
    parser.add_argument("--verbose_head", type=int, default=20,
                        help="Number of leading forecast/GT values to print per head when --verbose is set.")
    parser.add_argument(
        "--dist_timeout_minutes",
        type=int,
        default=60,
        help="Process-group timeout for NCCL collectives during inference (default 60 minutes).",
    )
    parser.set_defaults(sampling=True)
    args = parser.parse_args()

    # -------- DDP setup --------
    rank, world_size, device = ddp_init(timeout_minutes=args.dist_timeout_minutes)
    os.makedirs(args.save_pkl.split(".")[0], exist_ok=True)

    # -------- Load NetBurst model on this rank's GPU --------
    predictor = load_twin_head_from_dir(args.model, device, print_head_norms=True)

    # Load trained weights (full NetBurst state dict)
    # state_path = os.path.join(args.model, "chronos_best.pt")
    # state = torch.load(state_path, map_location=device)
    # loaded = False
    # try:
    #     predictor.load_state_dict(state, strict=True)
    #     loaded = True
    # except Exception:
    #     # Fallback: try loading only the inner Chronos weights (while keeping heads)
    #     try:
    #         core = strip_prefix_if_present(state, ["model.model.", "model."])
    #         predictor.model.load_state_dict(core, strict=False)
    #         loaded = True
    #     except Exception:
    #         pass
    # if not loaded:
    #     raise RuntimeError(f"Failed to load weights from {state_path}")

    # --------- Load (bi, ibg) series pairs ---------
    if rank == 0:
        if args.use_precomputed_context_forecast:
            # Loader only enforces >=1 context, >=1 forecast, and total >= max(2, min_len).
            # Use min_len=2 here so parquet rows are not dropped by a high default --min_len;
            # optional stricter floor is applied in cleaned below via args.min_len.
            all_pairs = load_precomputed_ibgbi_context_forecast(
                args.parquet_root, min_len=2, limit=args.limit
            )
        elif args.fires_parquet:
            all_pairs = load_fires_ibg_bi(args.fires_parquet, min_len=args.min_len, max_len=args.max_len, limit=args.limit)
        else:
            # If ips_csv is provided and exists, filter by it; otherwise load all
            if args.ips_csv and os.path.exists(args.ips_csv):
                # Restrict to allowed keys from CSV and then extract (key, bi, ibg)
                all_pairs = load_filtered_ibg_bi(args.parquet_root, args.ips_csv, min_len=args.min_len, max_len=args.max_len, limit=args.limit)
            else:
                # Load all data without filtering
                all_pairs = load_ibg_bi_series(args.parquet_root, min_len=args.min_len, max_len=args.max_len, limit=args.limit)
                print(f"Rank 0: Loaded {len(all_pairs)} (bi, ibg) pairs from parquet")

        # Ensure each pair has equal length (loaders enforce this, but double-check)
        cleaned = []
        if args.use_precomputed_context_forecast:
            for (key, ctx_bi, fc_bi, ctx_ibg, fc_ibg) in all_pairs:
                lctx = min(len(ctx_bi), len(ctx_ibg))
                lfc = min(len(fc_bi), len(fc_ibg))
                mh = max(1, int(args.min_h))
                tot = lctx + lfc
                if lctx >= 1 and lfc >= mh and tot >= max(2, int(args.min_len)):
                    cleaned.append((key, ctx_bi[:lctx], fc_bi[:lfc], ctx_ibg[:lctx], fc_ibg[:lfc]))
        elif args.fires_parquet:
            for (bi, ibg) in all_pairs:
                L = min(len(bi), len(ibg))
                if L >= args.min_len:
                    cleaned.append((None, bi[:L], ibg[:L]))  # no key available; use None
        elif args.ips_csv and os.path.exists(args.ips_csv):
            # Filtered data has keys
            for (key, bi, ibg) in all_pairs:
                L = min(len(bi), len(ibg))
                if L >= args.min_len:
                    cleaned.append((key, bi[:L], ibg[:L]))
        else:
            # Unfiltered data has no keys (just bi, ibg pairs)
            for (bi, ibg) in all_pairs:
                L = min(len(bi), len(ibg))
                if L >= args.min_len:
                    cleaned.append((None, bi[:L], ibg[:L]))
        #if test split ratio is given, split into train and test
        if args.test_split_ratio is not None:
            split_idx = int(len(cleaned) * (1 - args.test_split_ratio))
            cleaned = cleaned[split_idx:]  # keep only test split
        print(f"Rank 0: Loaded {len(cleaned)} cleaned (key, bi, ibg) triplets from parquet")

        shards = [cleaned[i::world_size] for i in range(world_size)]
        input_list = shards
    else:
        input_list = []

    # Every rank receives its shard
    out = [None]
    torch.distributed.scatter_object_list(out, input_list, src=0)
    my_pairs = out[0] if out[0] is not None else []
    torch.distributed.barrier()

    # --------- Autoregression over this rank's shard ---------
    bs = args.batch_size
    thr_nz = float(args.nz_thresh)

    def _resolve_out_path(path_like: str, base_dir: str) -> str:
        if os.path.isabs(path_like):
            return path_like
        return os.path.join(base_dir, path_like)

    def _wd_on_pdfs(a, b):
        # wasserstein_distance_tails handles eps-addition and normalization internally;
        # pre-normalizing here would cause double-normalization and destroy timing.
        return wasserstein_distance_tails(np.asarray(a, dtype=float), np.asarray(b, dtype=float), zero_thresh=0.0)

    def _expand_series(iei_vals, compressed_vals):
        iei = np.asarray(iei_vals, dtype=float)
        cmpv = np.asarray(compressed_vals, dtype=float)
        n = min(len(iei), len(cmpv))
        if n == 0:
            return np.array([], dtype=float)
        chunks = []
        for i in range(n):
            v = iei[i]
            if not np.isfinite(v):
                g = 1
            else:
                try:
                    g = int(v)
                except Exception:
                    g = 1
                if g < 1:
                    g = 1
            g = min(g, int(args.max_zero_fill))
            # IBG stores an inclusive gap count (1 means no explicit zero slot before BI),
            # so expand as zeros for (IBG - 1), then place the BI value.
            chunks.extend([np.zeros(g - 1, dtype=float), np.array([cmpv[i]], dtype=float)])
        return np.concatenate(chunks, axis=0)

    def _ibg_seg_len(v):
        """One burst's contribution to merged length: (IBG-1) zeros + 1 BI slot == IBG (g)."""
        if not np.isfinite(v):
            g = 1
        else:
            try:
                g = int(v)
            except Exception:
                g = 1
            if g < 1:
                g = 1
        return min(g, int(args.max_zero_fill))

    def _num_bursts_for_merged_prefix(ibg_vals, bi_vals, prefix_len: int) -> int:
        """Smallest k such that len(_expand_series(ibg[:k], bi[:k])) >= prefix_len (or all bursts)."""
        iei = np.asarray(ibg_vals, dtype=float)
        cmpv = np.asarray(bi_vals, dtype=float)
        n = min(len(iei), len(cmpv))
        if n == 0 or prefix_len <= 0:
            return 0
        acc = 0
        for i in range(n):
            acc += _ibg_seg_len(iei[i])
            if acc >= prefix_len:
                return i + 1
        return n

    def split_70_30(length, ctx_frac=0.70, min_ctx=100, min_h=1):
        L = max(min_ctx, int(math.ceil(ctx_frac * length)))
        H = length - L
        if H < min_h:
            take_back = min(min_ctx, (min_h - H))
            L = max(min_ctx, L - take_back)
            H = length - L
        if H <= 0 and length > min_ctx:
            L = length - 1
            H = 1
        return L, max(0, H)

    def _sample_from_logits(logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float,
                            generator: Optional[torch.Generator] = None,
                            num_samples: int = 1) -> torch.Tensor:
        """Return sampled token ids [B, S] from logits [B,V] using temperature, top-k, top-p."""
        n_samp = max(1, int(num_samples))
        if (temperature is None) or (temperature <= 0):
            return logits.argmax(dim=-1, keepdim=True).repeat(1, n_samp)
        # scale by temperature
        scaled = logits / max(1e-6, float(temperature))
        # Top-K filtering
        if isinstance(top_k, int) and top_k > 0 and top_k < scaled.size(-1):
            kth_vals, _ = torch.topk(scaled, k=top_k, dim=-1)
            thresh = kth_vals[..., -1, None]
            scaled = torch.where(scaled < thresh, torch.full_like(scaled, float('-inf')), scaled)
        # Top-P (nucleus) filtering
        if (top_p is not None) and (0.0 < top_p < 1.0):
            sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cprobs = torch.cumsum(probs, dim=-1)
            # mask tokens with cumulative prob > top_p
            mask = cprobs > top_p
            # ensure at least one token retained
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
            # map back to original order
            unsorted = torch.full_like(scaled, float('-inf'))
            unsorted.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
            scaled = unsorted
        # sample
        probs = torch.softmax(scaled, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=n_samp, generator=generator)
        return next_ids

    def ar_one_pair(bi_vals, ibg_vals, predictor, ctx_frac, min_ctx, min_h,
                    do_sample: bool, temperature: float, top_k: int, top_p: float,
                    seed: Optional[int],
                    context_bi_vals=None, forecast_bi_vals=None,
                    context_ibg_vals=None, forecast_ibg_vals=None):
        """Autoregress in TOKEN space for stability; combined mask across streams; optional sampling."""
        _empty_meta = {"loss_bi": float("nan"), "loss_ibg": float("nan"),
                       "series_len": 0, "context_len": 0, "horizon": 0,
                       "fano_input_bi": float("nan"), "fano_gt_bi": float("nan"), "fano_forecast_bi": float("nan"),
                       "fano_input_ibg": float("nan"), "fano_gt_ibg": float("nan"), "fano_forecast_ibg": float("nan")}
        use_pre = all(x is not None for x in [context_bi_vals, forecast_bi_vals, context_ibg_vals, forecast_ibg_vals])
        if use_pre:
            L = min(len(context_bi_vals), len(context_ibg_vals))
            H = min(len(forecast_bi_vals), len(forecast_ibg_vals))
            T = L + H
            # Precomputed splits: only require >=1 context and >=1 forecast; do not apply min_ctx or T>=(min_ctx+min_h).
            if L < 1 or H < max(1, int(min_h)):
                return [], [], [], [], 0, 0, 0, 0, _empty_meta, [], []
            bi_full = list(map(float, context_bi_vals[:L])) + list(map(float, forecast_bi_vals[:H]))
            ibg_full = list(map(float, context_ibg_vals[:L])) + list(map(float, forecast_ibg_vals[:H]))
        else:
            T = min(len(bi_vals), len(ibg_vals))
            mh = max(1, int(min_h))
            if T < (min_ctx + mh):
                return [], [], [], [], 0, 0, 0, 0, _empty_meta, [], []
            if getattr(args, "max_context_split", False):
                # Longest context such that horizon has at least mh steps (usually H == mh when T is large enough).
                L = max(int(min_ctx), T - mh)
                H = T - L
            else:
                L, H = split_70_30(T, ctx_frac=ctx_frac, min_ctx=min_ctx, min_h=mh)
            if H <= 0:
                return [], [], [], [], 0, 0, 0, 0, _empty_meta, [], []
            bi_full = list(map(float, bi_vals[:L+H]))
            ibg_full = list(map(float, ibg_vals[:L+H]))

        H_gt = H           # ground-truth horizon matches generation horizon

        device = predictor.device
        # Float contexts (for reporting only)
        ctx_bi_vals  = list(map(float, bi_full[:L]))
        ctx_ibg_vals = list(map(float, ibg_full[:L]))
        truth_bi  = list(map(float, bi_full[L:L+H_gt]))
        truth_ibg = list(map(float, ibg_full[L:L+H_gt]))

        # Tokenize FULL (context + horizon) once; then slice
        full_bi = predictor.pipeline._prepare_and_validate_context(
            context=[torch.tensor(bi_full[:L+H], dtype=torch.float32)]
        )
        ids_bi_full, mask_bi_full, _ = predictor.tokenizer_bi.context_input_transform(full_bi)
        full_ibg = predictor.pipeline._prepare_and_validate_context(
            context=[torch.tensor(ibg_full[:L+H], dtype=torch.float32)]
        )
        ids_ibg_full, mask_ibg_full, _ = predictor.tokenizer_ibg.context_input_transform(full_ibg)

        # Drop EOS once, like training
        if ids_bi_full.size(1) > 0:
            ids_bi_full  = ids_bi_full[:, :-1]
            mask_bi_full = mask_bi_full[:, :-1]
        if ids_ibg_full.size(1) > 0:
            ids_ibg_full  = ids_ibg_full[:, :-1]
            mask_ibg_full = mask_ibg_full[:, :-1]

        # Slice into context and horizon targets
        ids_bi_ctx  = ids_bi_full[:, :L]
        ids_ibg_ctx = ids_ibg_full[:, :L]
        gt_tok_bi   = ids_bi_full[:, L:L+H]
        gt_tok_ibg  = ids_ibg_full[:, L:L+H]
        mask_bi_ctx  = mask_bi_full[:, :L]
        mask_ibg_ctx = mask_ibg_full[:, :L]

        ids_bi  = ids_bi_ctx.to(device)
        ids_ibg = ids_ibg_ctx.to(device)
        mask_bi  = mask_bi_ctx.to(device)
        mask_ibg = mask_ibg_ctx.to(device)
        comb_mask = (mask_bi & mask_ibg)

        # Build decoder inputs for AR: [PAD] + tokens_so_far for both streams
        pad_id = int(predictor.pipeline.model.config.pad_token_id)
        dec_bi  = torch.full((1, 1), pad_id, dtype=torch.long, device=device)
        dec_ibg = torch.full((1, 1), pad_id, dtype=torch.long, device=device)
        dec_bi  = torch.cat([dec_bi,  ids_bi],  dim=1)  # [1, L+1]
        dec_ibg = torch.cat([dec_ibg, ids_ibg], dim=1)  # [1, L+1]

        fc_bi, fc_ibg = [], []
        corr_bi = corr_ibg = 0
        tot_bi = tot_ibg = 0

        with torch.no_grad():
            # Move precomputed horizon tokens to device for accuracy checks
            gt_tok_bi = gt_tok_bi.to(device)
            gt_tok_ibg = gt_tok_ibg.to(device)

            # optional deterministic generator for reproducibility
            gen = None
            if seed is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(int(seed))

            for step in range(H):
                # AR tuple decoder path: local residual not applied (see forward_with_embeddings).
                _, hidden = predictor.model.forward_with_embeddings(
                    input_ids1=ids_bi,
                    input_ids2=ids_ibg,
                    attention_mask=comb_mask,
                    decoder_input_ids=(dec_bi, dec_ibg),
                    type_id=0,
                    cross_attend=predictor.use_cross_attn,
                )
                h = hidden.squeeze(1)  # [1, T, D]
                # Apply per-head stream projections so inference matches training.
                h_bi = predictor.stream_proj_bi(h)
                h_ibg = predictor.stream_proj_ibg(h)
                logits_bi = predictor.head_bi(h_bi)[:, -1, :]   # [1, V]
                logits_ibg = predictor.head_ibg(h_ibg)[:, -1, :]
                # Disallow special tokens (PAD/EOS/etc.) at generation time
                cfg = predictor.pipeline.model.config
                n_spec = int(getattr(cfg, "n_special_tokens", 0))
                eos_id = int(getattr(cfg, "eos_token_id", -1))
                if n_spec > 0:
                    logits_bi[:, :n_spec] = float('-inf')
                    logits_ibg[:, :n_spec] = float('-inf')
                if eos_id >= 0 and eos_id < logits_bi.size(-1):
                    logits_bi[:, eos_id] = float('-inf')
                    logits_ibg[:, eos_id] = float('-inf')
                # If IBG uses the integer tokenizer, mask out token ids beyond
                # the active [n_spec, n_spec + K) range so the model cannot emit
                # never-trained indices.
                if isinstance(predictor.tokenizer_ibg, IntegerIBGBins):
                    K_ibg = int(predictor.tokenizer_ibg.num_int_bins)
                    tail_start = n_spec + K_ibg
                    if tail_start < logits_ibg.size(-1):
                        logits_ibg[:, tail_start:] = float('-inf')

                if do_sample:
                    sampled_bi = _sample_from_logits(
                        logits_bi, temperature=temperature, top_k=top_k, top_p=top_p,
                        generator=gen, num_samples=int(args.num_samples)
                    ).squeeze(0)  # [S]
                    sampled_ibg = _sample_from_logits(
                        logits_ibg, temperature=temperature, top_k=top_k, top_p=top_p,
                        generator=gen, num_samples=int(args.num_samples)
                    ).squeeze(0)  # [S]
                    vals_bi = predictor.token_values_bi[sampled_bi]
                    vals_ibg = predictor.token_values_ibg[sampled_ibg]
                    med_bi = torch.median(vals_bi)
                    med_ibg = torch.median(vals_ibg)
                    # Choose token whose decoded value is closest to median value.
                    idx_bi = torch.argmin(torch.abs(vals_bi - med_bi))
                    idx_ibg = torch.argmin(torch.abs(vals_ibg - med_ibg))
                    next_id_bi = sampled_bi[idx_bi].view(1)
                    next_id_ibg = sampled_ibg[idx_ibg].view(1)
                else:
                    next_id_bi  = logits_bi.argmax(dim=-1)    # [1]
                    next_id_ibg = logits_ibg.argmax(dim=-1)   # [1]

                # optional first-step debug
                if args.debug_sanity and step == 0 and gt_tok_bi.shape[1] > 0:
                    try:
                        print(f"[dbg] rank {rank} first BI pred={int(next_id_bi.item())} true={int(gt_tok_bi[0,0].item())}")
                    except Exception:
                        pass

                # Append TOKEN IDs (not floats) and update masks
                ids_bi  = torch.cat([ids_bi,  next_id_bi.view(1, 1)], dim=1)
                ids_ibg = torch.cat([ids_ibg, next_id_ibg.view(1, 1)], dim=1)
                mask_bi  = torch.cat([mask_bi,  torch.ones((1, 1), dtype=mask_bi.dtype, device=device)], dim=1)
                mask_ibg = torch.cat([mask_ibg, torch.ones((1, 1), dtype=mask_ibg.dtype, device=device)], dim=1)
                comb_mask = (mask_bi & mask_ibg)

                # Also append to decoder inputs (teacher-forcing on generated tokens)
                dec_bi  = torch.cat([dec_bi,  next_id_bi.view(1, 1)], dim=1)
                dec_ibg = torch.cat([dec_ibg, next_id_ibg.view(1, 1)], dim=1)

                # Convert predicted tokens to numeric values for outputs
                fc_bi.append(float(predictor.token_values_bi[next_id_bi].item()))
                fc_ibg.append(float(predictor.token_values_ibg[next_id_ibg].item()))

                # Optional token-accuracy vs ground truth step token
                if gt_tok_bi is not None and step < gt_tok_bi.shape[1]:
                    corr_bi += int((next_id_bi.item() == gt_tok_bi[0, step].item()))
                    tot_bi  += 1
                if gt_tok_ibg is not None and step < gt_tok_ibg.shape[1]:
                    corr_ibg += int((next_id_ibg.item() == gt_tok_ibg[0, step].item()))
                    tot_ibg  += 1

        # Per-example loss (L1/MAE) and Fano factors for pkl
        loss_bi = float(np.mean(np.abs(np.array(fc_bi) - np.array(truth_bi)))) if len(fc_bi) else float("nan")
        loss_ibg = float(np.mean(np.abs(np.array(fc_ibg) - np.array(truth_ibg)))) if len(fc_ibg) else float("nan")
        meta = {
            "loss_bi": loss_bi,
            "loss_ibg": loss_ibg,
            "series_len": int(T),
            "context_len": int(L),
            "horizon": int(H),
            "fano_input_bi": fano_factor_numpy(ctx_bi_vals),
            "fano_gt_bi": fano_factor_numpy(truth_bi),
            "fano_forecast_bi": fano_factor_numpy(fc_bi),
            "fano_input_ibg": fano_factor_numpy(ctx_ibg_vals),
            "fano_gt_ibg": fano_factor_numpy(truth_ibg),
            "fano_forecast_ibg": fano_factor_numpy(fc_ibg),
        }
        return fc_bi, truth_bi, fc_ibg, truth_ibg, corr_bi, tot_bi, corr_ibg, tot_ibg, meta, ctx_bi_vals, ctx_ibg_vals

    def _context_sanity(bi_vals, ibg_vals, predictor) -> Optional[Tuple[float, float, float]]:
        """Compute training-style next-token token-accuracy on the context only (no AR),
        to validate the loaded checkpoint and tokenization alignment.
        Returns (acc_bi, acc_ibg, acc_avg) or None on failure.
        """
        try:
            # tokenize full context (drop EOS in targets step below, so keep EOS here like training)
            bi_ctx = predictor.pipeline._prepare_and_validate_context(
                context=[torch.tensor(bi_vals, dtype=torch.float32)]
            )
            ibg_ctx = predictor.pipeline._prepare_and_validate_context(
                context=[torch.tensor(ibg_vals, dtype=torch.float32)]
            )
            ids_bi, mask_bi, ids_ibg, mask_ibg, local_bi, local_ibg = (
                predictor._tokenize_dual_streams(bi_ctx, ibg_ctx)
            )

            comb_mask = (mask_bi & mask_ibg).to(predictor.device)
            dummy = torch.empty((ids_bi.size(0), 1), dtype=torch.long, device=predictor.device)
            fwd_kw = dict(
                input_ids1=ids_bi.to(predictor.device),
                input_ids2=ids_ibg.to(predictor.device),
                attention_mask=comb_mask,
                decoder_input_ids=dummy,
                type_id=0,
                cross_attend=predictor.use_cross_attn,
            )
            if local_bi is not None:
                fwd_kw["local_ids1"] = local_bi.to(predictor.device)
                fwd_kw["local_ids2"] = local_ibg.to(predictor.device)
            _, hidden = predictor.model.forward_with_embeddings(**fwd_kw)
            h = hidden.squeeze(1)
            h_bi = predictor.stream_proj_bi(h)
            h_ibg = predictor.stream_proj_ibg(h)
            logits_bi = predictor.head_bi(h_bi)[:, :-1, :]   # [B, T-1, V]
            logits_ibg = predictor.head_ibg(h_ibg)[:, :-1, :]
            targets_bi = ids_bi[:, 1:].to(predictor.device)
            targets_ibg = ids_ibg[:, 1:].to(predictor.device)
            valid_bi = mask_bi[:, 1:].to(predictor.device)
            valid_ibg = mask_ibg[:, 1:].to(predictor.device)

            preds_bi = logits_bi.argmax(dim=-1)
            preds_ibg = logits_ibg.argmax(dim=-1)
            tot_bi = valid_bi.sum().item()
            tot_ibg = valid_ibg.sum().item()
            corr_bi = (((preds_bi == targets_bi) & valid_bi).sum().item()) if tot_bi > 0 else 0
            corr_ibg = (((preds_ibg == targets_ibg) & valid_ibg).sum().item()) if tot_ibg > 0 else 0
            acc_bi = (corr_bi / max(1, tot_bi))
            acc_ibg = (corr_ibg / max(1, tot_ibg))
            return acc_bi, acc_ibg, 0.5*(acc_bi+acc_ibg)
        except Exception:
            return None

    def batch_autoreg(pairs_batch):
        out = []
        for local_idx, item in enumerate(pairs_batch):
            use_pre = isinstance(item, tuple) and len(item) == 4
            if use_pre:
                ctx_bi, fc_bi, ctx_ibg, fc_ibg = item
                bi = list(ctx_bi) + list(fc_bi)
                ibg = list(ctx_ibg) + list(fc_ibg)
            else:
                bi, ibg = item
            if args.debug_sanity and local_idx < args.debug_sanity:
                s = _context_sanity(bi, ibg, predictor)
                if s is not None:
                    acc_bi_s, acc_ibg_s, acc_avg_s = s
                    print(f"[sanity ctx] rank {rank} series {local_idx}: BI={acc_bi_s:.4f}, IBG={acc_ibg_s:.4f}, AVG={acc_avg_s:.4f}")
            res = ar_one_pair(
                bi, ibg, predictor,
                ctx_frac=args.ctx_frac, min_ctx=args.min_ctx, min_h=args.min_h,
                do_sample=bool(args.sampling), temperature=float(args.temperature),
                top_k=int(args.top_k), top_p=float(args.top_p), seed=args.seed,
                context_bi_vals=(ctx_bi if use_pre else None),
                forecast_bi_vals=(fc_bi if use_pre else None),
                context_ibg_vals=(ctx_ibg if use_pre else None),
                forecast_ibg_vals=(fc_ibg if use_pre else None),
            )
            fc_bi, tr_bi, fc_ibg, tr_ibg, cb, tb, cig, tig, meta, ctx_bi_r, ctx_ibg_r = res
            mase_bi = np.mean((np.abs(np.array(fc_bi) - np.array(tr_bi)))/np.abs(np.array(tr_bi)+1e-2)) if len(fc_bi) > 0 else float('nan')
            # print(f"rank {rank} local_series {local_idx}: BI MASE={mase_bi:.4f}")
            out.append(res)
        return out

    # Run over my shard
    results_bi = {}   # idx -> (forecasts, truths, meta) with meta = loss + Fano per example
    results_ibg = {}  # idx -> (forecasts, truths, meta)
    per_rows = []
    with torch.inference_mode():
        for i in range(0, len(my_pairs), bs):
            chunk = my_pairs[i:i+bs]
            # print(f"inference batch: {i}-{i+len(chunk)-1} ({len(chunk)} pairs)")

            # Split keys and (bi, ibg)
            if len(chunk) and isinstance(chunk[0], tuple) and len(chunk[0]) == 3:
                keys = [entry[0] for entry in chunk]
                pairs = [(entry[1], entry[2]) for entry in chunk]
            elif len(chunk) and isinstance(chunk[0], tuple) and len(chunk[0]) == 5:
                keys = [entry[0] for entry in chunk]
                pairs = [(entry[1], entry[2], entry[3], entry[4]) for entry in chunk]
            else:
                keys = None
                pairs = chunk  # assumed (bi, ibg)

            out = batch_autoreg(pairs)
            for j, res in enumerate(out):
                fc_bi, tr_bi, fc_ibg, tr_ibg, cb, tb, cig, tig, meta, ctx_bi_r, ctx_ibg_r = res
                global_idx = i + j
                out_key = keys[j] if keys is not None and keys[j] is not None else global_idx
                out_key = str(out_key)+"_rank"+str(rank)
                # print(
                #     f"rank {rank} series {global_idx}: BI f={len(fc_bi)}/t={len(tr_bi)} (acc={100.0*cb/tb if tb>0 else 0:.1f}%), "
                #     f"IBG f={len(fc_ibg)}/t={len(tr_ibg)} (acc={100.0*cig/tig if tig>0 else 0:.1f}%)"
                # )
                results_bi[out_key] = (fc_bi, tr_bi, meta)
                results_ibg[out_key] = (fc_ibg, tr_ibg, meta)

                # Per-example metrics for BI, IBG, and expanded merged timeline
                # IBG = inter-burst gap (zero-fill multiplier), BI = byte value
                fc_bi_arr = np.asarray(fc_bi, dtype=np.float32)
                tr_bi_arr = np.asarray(tr_bi, dtype=np.float32)
                fc_ibg_arr = np.asarray(fc_ibg, dtype=np.float32)
                tr_ibg_arr = np.asarray(tr_ibg, dtype=np.float32)
                ctx_bi_arr = np.asarray(ctx_bi_r, dtype=np.float32)
                ctx_ibg_arr = np.asarray(ctx_ibg_r, dtype=np.float32)

                # Apply BI threshold: zero out BI values below bi_thresh independently
                # in both forecast and GT before any metric is computed.
                thr_bi = float(args.bi_thresh)
                if thr_bi > 0.0:
                    fc_bi_arr = np.where(fc_bi_arr < thr_bi, 0.0, fc_bi_arr).astype(np.float32)
                    tr_bi_arr = np.where(tr_bi_arr < thr_bi, 0.0, tr_bi_arr).astype(np.float32)
                    ctx_bi_arr = np.where(ctx_bi_arr < thr_bi, 0.0, ctx_bi_arr).astype(np.float32)

                # Skip series with no qualifying GT BI bursts after thresholding.
                # Such series have no signal to evaluate against.
                if thr_bi > 0.0 and not np.any(tr_bi_arr > 0.0):
                    continue

                mae_bi = float(np.mean(np.abs(fc_bi_arr - tr_bi_arr))) if len(fc_bi_arr) else np.nan
                mae_ibg = float(np.mean(np.abs(fc_ibg_arr - tr_ibg_arr))) if len(fc_ibg_arr) else np.nan
                nz_scaled_bi = mape_nonzero_gt(fc_bi_arr, tr_bi_arr, nz_thresh=thr_nz, scale=args.mape_scale)
                nz_scaled_ibg = mape_nonzero_gt(fc_ibg_arr, tr_ibg_arr, nz_thresh=thr_nz, scale=args.mape_scale)
                wd_bi = wasserstein_distance_tails(tr_bi_arr, fc_bi_arr, zero_thresh=thr_nz)
                wd_ibg = wasserstein_distance_tails(tr_ibg_arr, fc_ibg_arr, zero_thresh=thr_nz)

                # Merged timeline: prefix length from GT (--merged_eval_max_len), then k bursts
                # on GT so that expand(tr_ibg[:k], tr_bi[:k]) covers that prefix; forecast uses
                # the same first k burst pairs (other models compare on ≤500 merged steps).
                # merged_eval_max_len < 0 means use the full GT merged length (default -1).
                merged_gt_full = _expand_series(tr_ibg_arr, tr_bi_arr)
                full_merged_len = int(len(merged_gt_full))
                if int(args.merged_eval_max_len) < 0:
                    merged_max = full_merged_len
                else:
                    merged_max = max(0, int(args.merged_eval_max_len))
                effective_len = min(merged_max, full_merged_len) if merged_max else 0
                expanded_gt = merged_gt_full[:effective_len]
                k_bursts = _num_bursts_for_merged_prefix(tr_ibg_arr, tr_bi_arr, effective_len)
                k_fc = min(
                    k_bursts,
                    int(fc_bi_arr.shape[0]),
                    int(fc_ibg_arr.shape[0]),
                )
                # k_bursts = bursts needed on GT to cover merged prefix; k_fc caps by forecast length
                expanded_fc = _expand_series(fc_ibg_arr[:k_fc], fc_bi_arr[:k_fc])
                Lm = min(effective_len, len(expanded_fc), len(expanded_gt))
                expanded_ctx = _expand_series(ctx_ibg_arr, ctx_bi_arr)[:merged_max] if merged_max else np.array([], dtype=float)

                if Lm > 0:
                    mae_expanded = float(np.mean(np.abs(expanded_fc[:Lm] - expanded_gt[:Lm])))
                    wd_expanded = _wd_on_pdfs(expanded_fc[:Lm], expanded_gt[:Lm])
                    nz_scaled_merged = mape_nonzero_gt(
                        expanded_fc[:Lm], expanded_gt[:Lm],
                        nz_thresh=thr_nz, scale=args.mape_scale)
                else:
                    mae_expanded = np.nan
                    wd_expanded = np.nan
                    nz_scaled_merged = np.nan

                # Oracle IBG: GT IBG (timing) + forecasted BI (byte values) vs GT merged
                # Tests how much error comes purely from the BI head
                expanded_oracle_ibg = _expand_series(tr_ibg_arr[:k_fc], fc_bi_arr[:k_fc])
                Lo_ibg = min(Lm, len(expanded_oracle_ibg))
                if Lo_ibg > 0:
                    mape_oracle_ibg = mape_nonzero_gt(
                        expanded_oracle_ibg[:Lo_ibg], expanded_gt[:Lo_ibg],
                        nz_thresh=thr_nz, scale=args.mape_scale)
                    wd_oracle_ibg = _wd_on_pdfs(expanded_oracle_ibg[:Lo_ibg], expanded_gt[:Lo_ibg])
                else:
                    mape_oracle_ibg = np.nan
                    wd_oracle_ibg = np.nan

                # Oracle BI: forecasted IBG (timing) + GT BI (byte values) vs GT merged
                # Tests how much error comes purely from the IBG head
                expanded_oracle_bi = _expand_series(fc_ibg_arr[:k_fc], tr_bi_arr[:k_fc])
                Lo_bi = min(Lm, len(expanded_oracle_bi))
                if Lo_bi > 0:
                    mape_oracle_bi = mape_nonzero_gt(
                        expanded_oracle_bi[:Lo_bi], expanded_gt[:Lo_bi],
                        nz_thresh=thr_nz, scale=args.mape_scale)
                    wd_oracle_bi = _wd_on_pdfs(expanded_oracle_bi[:Lo_bi], expanded_gt[:Lo_bi])
                else:
                    mape_oracle_bi = np.nan
                    wd_oracle_bi = np.nan

                per_rows.append({
                    "series_id": str(out_key),
                    "length": int(meta.get("series_len", 0)),
                    "context_len": int(meta.get("context_len", 0)),
                    "horizon": int(meta.get("horizon", len(fc_bi_arr))),
                    "MAE_BI": mae_bi,
                    "NZ_MAE_BI": mae_nonzero_gt(fc_bi_arr, tr_bi_arr, thr=thr_nz),
                    "NZ_SCALED_BI": nz_scaled_bi,
                    "WD_BI": wd_bi,
                    "MAE_IBG": mae_ibg,
                    "NZ_MAE_IBG": mae_nonzero_gt(fc_ibg_arr, tr_ibg_arr, thr=thr_nz),
                    "NZ_SCALED_IBG": nz_scaled_ibg,
                    "WD_IBG": wd_ibg,
                    "MAE_EXPANDED_MERGED": mae_expanded,
                    "NZ_SCALED_MERGED": nz_scaled_merged,
                    "WD_EXPANDED_MERGED": wd_expanded,
                    "merged_eval_len": int(Lm) if Lm > 0 else 0,
                    "merged_eval_bursts_k_gt": int(k_bursts),
                    "merged_eval_bursts_k": int(k_fc),
                    "MAPE_ORACLE_IBG": mape_oracle_ibg,
                    "WD_ORACLE_IBG": wd_oracle_ibg,
                    "MAPE_ORACLE_BI": mape_oracle_bi,
                    "WD_ORACLE_BI": wd_oracle_bi,
                    "gt_tail_bi": json.dumps(tr_bi_arr.astype(float).tolist()),
                    "pred_tail_bi": json.dumps(fc_bi_arr.astype(float).tolist()),
                    "gt_tail_ibg": json.dumps(tr_ibg_arr.astype(float).tolist()),
                    "pred_tail_ibg": json.dumps(fc_ibg_arr.astype(float).tolist()),
                    "context_merged": json.dumps(expanded_ctx.astype(float).tolist()),
                    "forecast_merged": json.dumps(expanded_fc.astype(float).tolist()),
                    "gt_merged": json.dumps(expanded_gt.astype(float).tolist()),
                })

                if args.verbose:
                    def _fmt_vec(arr, n, prec=4):
                        if arr is None or len(arr) == 0:
                            return "[]"
                        head = arr[:n]
                        body = ", ".join(f"{float(v):.{prec}f}" for v in head)
                        suffix = f" ... (+{len(arr) - n} more)" if len(arr) > n else ""
                        return f"[{body}]{suffix}"
                    n_head = max(1, int(args.verbose_head))
                    acc_bi_s = (100.0 * cb / tb) if tb > 0 else float("nan")
                    acc_ibg_s = (100.0 * cig / tig) if tig > 0 else float("nan")
                    print(
                        f"[rank {rank}] series={out_key} "
                        f"ctx_len={int(meta.get('context_len', 0))} "
                        f"horizon={int(meta.get('horizon', len(fc_bi_arr)))} "
                        f"| BI  tok_acc={acc_bi_s:.1f}% "
                        f"MAE={mae_bi:.4f} NZ_SCALED={nz_scaled_bi:.4f} WD={wd_bi:.4f} "
                        f"| IBG tok_acc={acc_ibg_s:.1f}% "
                        f"MAE={mae_ibg:.4f} NZ_SCALED={nz_scaled_ibg:.4f} WD={wd_ibg:.4f} "
                        f"| MERGED MAE={mae_expanded if not (isinstance(mae_expanded, float) and np.isnan(mae_expanded)) else float('nan'):.4f} "
                        f"WD={wd_expanded if not (isinstance(wd_expanded, float) and np.isnan(wd_expanded)) else float('nan'):.4f} "
                        f"(k_fc={int(k_fc)}/k_gt={int(k_bursts)}, Lm={int(Lm)})",
                        flush=True,
                    )
                    print(f"    BI  fc  ={_fmt_vec(fc_bi_arr,  n_head, prec=2)}", flush=True)
                    print(f"    BI  gt  ={_fmt_vec(tr_bi_arr,  n_head, prec=2)}", flush=True)
                    print(f"    IBG fc  ={_fmt_vec(fc_ibg_arr, n_head, prec=2)}", flush=True)
                    print(f"    IBG gt  ={_fmt_vec(tr_ibg_arr, n_head, prec=2)}", flush=True)

    # --------- Calculate and print MAPE for BI ---------
    all_mapes_bi = []
    for out_key, entry in results_bi.items():
        fc_bi, tr_bi = entry[0], entry[1]
        if len(fc_bi) > 0 and len(tr_bi) > 0:
            # Calculate MAPE for this series
            fc_arr = np.array(fc_bi)
            tr_arr = np.array(tr_bi)
            # Avoid division by zero: only include points where truth > epsilon
            epsilon = 0.5
            valid_mask = np.abs(tr_arr) > epsilon
            if valid_mask.sum() > 0:
                mape = np.mean(np.abs((tr_arr[valid_mask] - fc_arr[valid_mask]) / tr_arr[valid_mask])) * 100
                all_mapes_bi.append(mape)
    
    if len(all_mapes_bi) > 0:
        avg_mape_bi = np.mean(all_mapes_bi)
        print(f"[rank {rank}] Average MAPE for BI across {len(all_mapes_bi)} samples: {avg_mape_bi:.4f}%")
    else:
        print(f"[rank {rank}] No valid MAPE samples for BI")

    # --------- Save per-rank shards (separate files) ---------
    base = args.save_pkl.split(".")[0]
    rank_out_bi  = os.path.join(base, f"bi_rank{rank:02d}.pkl")
    rank_out_ibg = os.path.join(base, f"ibg_rank{rank:02d}.pkl")
    rank_metrics_csv = os.path.join(base, f"per_example_metrics_rank{rank:02d}.csv")
    with open(rank_out_bi, "wb") as f:
        pickle.dump(results_bi, f)
    with open(rank_out_ibg, "wb") as f:
        pickle.dump(results_ibg, f)
    pd.DataFrame(per_rows).to_csv(rank_metrics_csv, index=False)
    print(f"[rank {rank}] wrote {len(results_bi)} BI series → {rank_out_bi}")
    print(f"[rank {rank}] wrote {len(results_ibg)} IBG series → {rank_out_ibg}")
    print(f"[rank {rank}] wrote {len(per_rows)} per-example rows → {rank_metrics_csv}")

    # Merge per-rank metric rows on rank 0 and write final CSV/JSON
    torch.distributed.barrier()
    if rank == 0:
        merged = []
        for r in range(world_size):
            rp = os.path.join(base, f"per_example_metrics_rank{r:02d}.csv")
            if os.path.exists(rp):
                try:
                    merged.append(pd.read_csv(rp))
                except Exception:
                    pass
        if len(merged) > 0:
            per_df = pd.concat(merged, ignore_index=True)
        else:
            per_df = pd.DataFrame()

        out_csv = _resolve_out_path(args.per_example_csv, base)
        out_json = _resolve_out_path(args.aggregate_json, base)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
        os.makedirs(os.path.dirname(out_json), exist_ok=True) if os.path.dirname(out_json) else None
        per_df.to_csv(out_csv, index=False)

        metrics = {
            "num_series_test": int(len(per_df)),
            "mean_MAE_BI": float(np.nanmean(per_df["MAE_BI"])) if len(per_df) else np.nan,
            "mean_NZ_MAE_BI": float(np.nanmean(per_df["NZ_MAE_BI"])) if len(per_df) else np.nan,
            "mean_NZ_SCALED_BI": float(np.nanmean(per_df["NZ_SCALED_BI"])) if len(per_df) else np.nan,
            "mean_MAPE": float(np.nanmean(per_df["NZ_SCALED_BI"])) if len(per_df) else np.nan,
            "mean_WD_BI": float(np.nanmean(per_df["WD_BI"])) if len(per_df) else np.nan,
            "mean_MAE_IBG": float(np.nanmean(per_df["MAE_IBG"])) if len(per_df) else np.nan,
            "mean_NZ_MAE_IBG": float(np.nanmean(per_df["NZ_MAE_IBG"])) if len(per_df) else np.nan,
            "mean_NZ_SCALED_IBG": float(np.nanmean(per_df["NZ_SCALED_IBG"])) if len(per_df) else np.nan,
            "mean_WD_IBG": float(np.nanmean(per_df["WD_IBG"])) if len(per_df) else np.nan,
            "mean_MAE_EXPANDED_MERGED": float(np.nanmean(per_df["MAE_EXPANDED_MERGED"])) if len(per_df) else np.nan,
            "mean_NZ_SCALED_MERGED": float(np.nanmean(per_df["NZ_SCALED_MERGED"])) if len(per_df) else np.nan,
            "mean_WD_EXPANDED_MERGED": float(np.nanmean(per_df["WD_EXPANDED_MERGED"])) if len(per_df) else np.nan,
            # Oracle IBG: GT timing + forecasted BI — isolates BI-head error
            "mean_MAPE_ORACLE_IBG": float(np.nanmean(per_df["MAPE_ORACLE_IBG"])) if len(per_df) else np.nan,
            "mean_WD_ORACLE_IBG": float(np.nanmean(per_df["WD_ORACLE_IBG"])) if len(per_df) else np.nan,
            # Oracle BI: forecasted timing + GT BI — isolates IBG-head error
            "mean_MAPE_ORACLE_BI": float(np.nanmean(per_df["MAPE_ORACLE_BI"])) if len(per_df) else np.nan,
            "mean_WD_ORACLE_BI": float(np.nanmean(per_df["WD_ORACLE_BI"])) if len(per_df) else np.nan,
            "median_NZ_SCALED": float(np.nanmedian(per_df["NZ_SCALED_BI"])) if len(per_df) else np.nan,
            "median_MAPE": float(np.nanmedian(per_df["NZ_SCALED_BI"])) if len(per_df) else np.nan,
            "median_WD": float(np.nanmedian(per_df["WD_BI"])) if len(per_df) else np.nan,
            "median_NZ_SCALED_MERGED": float(np.nanmedian(per_df["NZ_SCALED_MERGED"])) if len(per_df) else np.nan,
            "median_WD_merged": float(np.nanmedian(per_df["WD_EXPANDED_MERGED"])) if len(per_df) else np.nan,
            "median_WD_oracle_ibg": float(np.nanmedian(per_df["WD_ORACLE_IBG"])) if len(per_df) else np.nan,
            "median_WD_oracle_bi": float(np.nanmedian(per_df["WD_ORACLE_BI"])) if len(per_df) else np.nan,
        }
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[rank 0] wrote merged per-example metrics → {out_csv}")
        print(f"[rank 0] wrote aggregate metrics → {out_json}")

    torch.distributed.barrier()

    ddp_cleanup()


if __name__ == "__main__":
    main()
