"""Distributed NCCL training for NetBurst Chronos."""

from __future__ import annotations

import argparse
import copy
import datetime
import os
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from netburst.data import (
    PairSeriesDataset,
    collate_as_list,
    iterative_quantile_bins,
    load_fires_ibg_bi,
    load_ibg_bi_series,
)
from netburst.model import MyChronosPipeline, TwinHeadChronosPredictor, soft_ce_rank
from netburst.utils import fano_factor_tensor


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_root")
    parser.add_argument("--model", default="amazon/chronos-t5-small")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of data to use for training",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Maximum length of series to consider",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--save_dir",
        default="checkpoints",
        help="Where to save best model",
    )
    parser.add_argument("--retrain", type=str, default=None)
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=512,
        help="Effective number of quantile bins per stream (<= model capacity)",
    )
    parser.add_argument(
        "--mse_mode",
        choices=["bin", "centers"],
        default="bin",
        help="How to compute the auxiliary MSE: 'bin' for bin-index space, 'centers' for weighted token centers",
    )
    parser.add_argument(
        "--center_clip",
        type=float,
        default=1e6,
        help="Clamp value for center-based MSE to avoid exploding values",
    )
    parser.add_argument(
        "--soft_ce_alpha_init",
        type=float,
        default=1.0,
        help="Initial sharpness alpha for BI/IBG soft-CE kernel (higher => sharper)",
    )
    parser.add_argument(
        "--soft_ce_denom_floor_bi",
        type=float,
        default=1e-3,
        help="Minimum denominator for BI relative-error soft-CE",
    )
    parser.add_argument(
        "--soft_ce_denom_floor_ibg",
        type=float,
        default=1.0,
        help="Minimum denominator for IBG relative-error soft-CE",
    )
    parser.add_argument(
        "--freeze_soft_ce_sharpness",
        action="store_true",
        help="Disable automatic tuning of soft-CE sharpness parameters",
    )
    parser.add_argument(
        "--use_ce_loss",
        action="store_true",
        help="Use standard cross-entropy (hard targets) instead of soft CE for BI/IBG token prediction",
    )
    parser.add_argument(
        "--fires_parquet",
        type=str,
        default=None,
        help="Path to Fires parquet preprocessed by ConvertFiresToIBGBI.py (with 'bi' and 'ibg' arrays). If provided, overrides parquet_root for training data.",
    )
    parser.add_argument(
        "--dist_timeout_minutes",
        type=int,
        default=360,
        help="Process-group timeout for NCCL collectives (rank 0 loads data + quantiles before broadcast; large parquet needs more than 45m).",
    )
    parser.add_argument(
        "--quantile_max_series",
        type=int,
        default=50_000,
        help="Max number of (bi,ig) series used only for BI/IBG quantile bin estimation. "
        "Full loaded set is still used for train/val. Cuts cost/time for huge loads.",
    )
    parser.add_argument(
        "--loss_weight_bi",
        type=float,
        default=1.0,
        help="Multiply BI CE loss by this weight (use 0 to train only the IBG head).",
    )
    parser.add_argument(
        "--loss_weight_ibg",
        type=float,
        default=1.0,
        help="Multiply IBG CE loss by this weight (use 0 to train only the BI head).",
    )
    parser.add_argument(
        "--loss_weight_local_bi",
        type=float,
        default=0.0,
        help="Multiply BI local auxiliary soft-CE by this weight (0 disables).",
    )
    parser.add_argument(
        "--loss_weight_local_ibg",
        type=float,
        default=0.0,
        help="Multiply IBG local auxiliary soft-CE by this weight (0 disables).",
    )
    parser.add_argument(
        "--soft_ce_sigma_local",
        type=float,
        default=1.5,
        help="Rank-space Gaussian sigma for local auxiliary soft-CE.",
    )
    parser.add_argument(
        "--local_loss_min_context",
        type=int,
        default=16,
        help="Mask out first N target positions from local auxiliary loss.",
    )
    parser.add_argument(
        "--ibg_integer_bins",
        type=int,
        default=0,
        help=(
            "Controls the IBG tokenizer. "
            "-1: auto — K = max(observed IBG) + 1 on rank 0 (capped by --num_tokens). "
            ">0: explicit K; tokenize IBG as integer indices in [0, K-1] using IntegerIBGBins. "
            "0 (default): legacy quantile tokenizer (GlobalQuantileBins) on IBG. "
            "Integer-bin modes also keep IBG=0 as a valid observation."
        ),
    )
    parser.add_argument(
        "--num_local_bins",
        type=int,
        default=0,
        help="Per-series local quantile bins (0 = disabled, identical to legacy).",
    )
    parser.add_argument(
        "--soft_ce_alpha_mode",
        type=str,
        default="scalar",
        choices=["scalar", "perbin", "parametric"],
        help="Soft-CE sharpness: 'scalar' (one alpha/stream, legacy), "
        "'perbin' (learnable alpha per bin), "
        "'parametric' (alpha = linear of per-bin width/center features).",
    )
    return parser


def run_training(from_checkpoint: bool) -> None:
    parser = build_train_parser()
    args = parser.parse_args()


    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=max(1, args.dist_timeout_minutes)),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print("[run_training] Parsed args:")
        for key, value in sorted(vars(args).items()):
            print(f"  {key}={value}")

    # 1) Load (bi, ibg) pairs and compute separate quantile boundaries
    if rank == 0:
        # Load full sequences first (do not truncate at loader level); we will window below.
        if args.fires_parquet:
            all_pairs = load_fires_ibg_bi(args.fires_parquet, min_len=args.min_len, max_len=None, limit=args.limit)
        else:
            all_pairs = load_ibg_bi_series(args.parquet_root, min_len=args.min_len, max_len=None, limit=args.limit)

        # -------------------------------------------------------------
        # Window expansion: if max_len is specified and a sequence is
        # longer than max_len, break it into consecutive non-overlapping
        # chunks of length max_len (last chunk may be shorter). Keep any
        # chunk whose length >= min_len. This replaces earlier behavior
        # that only kept the first max_len slice.
        # -------------------------------------------------------------
        if args.max_len is not None and args.max_len > 0:
            expanded = []
            for bi, ibg in all_pairs:
                L = len(bi)
                if L != len(ibg):
                    # skip malformed pair
                    continue
                if L <= args.max_len:
                    expanded.append((bi, ibg))
                    continue
                for start in range(0, L, args.max_len):
                    end = start + args.max_len
                    bi_chunk = bi[start:end]
                    ibg_chunk = ibg[start:end]
                    if len(bi_chunk) >= args.min_len and len(bi_chunk) == len(ibg_chunk):
                        expanded.append((bi_chunk, ibg_chunk))
            print(f"Windowing applied: original pairs={len(all_pairs)}, expanded pairs={len(expanded)} (max_len={args.max_len})")
            all_pairs = expanded
        # Split into BI and IBG value pools for quantiles (full pools)
        bi_pool  = [np.asarray(bi, dtype=np.float32) for (bi, _ibg) in all_pairs]
        ibg_pool = [np.asarray(ibg, dtype=np.float32) for (_bi, ibg) in all_pairs]

        # Resolve --ibg_integer_bins == -1 ("auto") to a concrete K based on the
        # observed IBG max, so every seen integer value gets its own bin and
        # anything beyond falls into the last (overflow) bin. Capped by
        # --num_tokens so we can never exceed the effective vocab size.
        if args.ibg_integer_bins == -1:
            max_ibg = 0.0
            for _a in ibg_pool:
                if len(_a) == 0:
                    continue
                _m = float(np.max(_a))
                if np.isfinite(_m) and _m > max_ibg:
                    max_ibg = _m
            K_auto = int(np.ceil(max_ibg)) + 1
            K_auto = max(2, K_auto)
            K_cap = int(max(2, args.num_tokens))
            if K_auto > K_cap:
                print(
                    f"[auto K] observed max IBG = {max_ibg:.2f} -> K={K_auto}, "
                    f"capped to num_tokens={K_cap}. Overflow values will share the last bin."
                )
                K_auto = K_cap
            args.ibg_integer_bins = K_auto
            print(
                f"[auto K] using ibg_integer_bins = {args.ibg_integer_bins} "
                f"(observed max IBG = {max_ibg:.2f})"
            )

        # Quantile binning flattens all values and is very slow on hundreds of thousands of series;
        # subsample for boundary estimation only (train/val still use all_pairs).
        n_full = len(bi_pool)
        if n_full > args.quantile_max_series:
            rng = random.Random(42)
            pick = rng.sample(range(n_full), args.quantile_max_series)
            bi_pool_q = [bi_pool[i] for i in pick]
            ibg_pool_q = [ibg_pool[i] for i in pick]
            print(
                f"[quantiles] Using {args.quantile_max_series} / {n_full} series "
                f"for BI/IBG quantile bin estimation (full set unchanged for train/val)."
            )
        else:
            bi_pool_q = bi_pool
            ibg_pool_q = ibg_pool
        if not from_checkpoint:
            # Determine number of real bins from model config
            pipeline_tmp = MyChronosPipeline.from_pretrained(
                args.model,
                device_map={"": "cpu"},
                torch_dtype=torch.float32
            )
            import copy
            cfg = copy.deepcopy(pipeline_tmp.model.config)
            del pipeline_tmp
            B_model = cfg.n_tokens - cfg.n_special_tokens
            B_eff = int(min(B_model, max(4, args.num_tokens)))
            # Build quantile edges for BI
            cuts_bi  = iterative_quantile_bins(bi_pool_q,  B_eff)
            # Build quantile edges for IBG only if we are NOT using the integer
            # tokenizer; otherwise skip (expensive and unused).
            if args.ibg_integer_bins and args.ibg_integer_bins > 0:
                cuts_ibg = []
            else:
                cuts_ibg = iterative_quantile_bins(ibg_pool_q, B_eff)
            # Boundaries: first edge = min*0.8, last edge = max*1.2 so first/last bin centers
            # stay near the data range (avoids huge L1 when model predicts wrong bin).
            # Use per-series min/max (no full concat) to avoid O(total timesteps) RAM.
            if len(cuts_bi) > 0:
                bi_min = min(float(np.min(a)) for a in bi_pool)
                bi_max = max(float(np.max(a)) for a in bi_pool)
                boundaries_bi = np.concatenate(([bi_min * 0.8], cuts_bi[:B_eff - 2], [bi_max * 1.2]))
            else:
                boundaries_bi = np.array([0.0, 1e6])
            if args.ibg_integer_bins and args.ibg_integer_bins > 0:
                # Placeholder boundaries so downstream serialization is consistent;
                # the actual IBG tokenizer (IntegerIBGBins) ignores these.
                boundaries_ibg = np.array([0.0, float(args.ibg_integer_bins)], dtype=np.float32)
            elif len(cuts_ibg) > 0:
                ibg_min = min(float(np.min(a)) for a in ibg_pool)
                ibg_max = max(float(np.max(a)) for a in ibg_pool)
                boundaries_ibg = np.concatenate(([ibg_min * 0.8], cuts_ibg[:B_eff - 2], [ibg_max * 1.2]))
            else:
                boundaries_ibg = np.array([0.0, 1e4])

            # Save boundaries
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, "boundaries_bi.pkl"), "wb") as f:
                pickle.dump(boundaries_bi, f)
            with open(os.path.join(args.save_dir, "boundaries_ibg.pkl"), "wb") as f:
                pickle.dump(boundaries_ibg, f)
        else:
            # Load boundaries from retrain dir
            retrain_dir = args.model
            with open(os.path.join(retrain_dir, "boundaries_bi.pkl"), "rb") as f:
                boundaries_bi = pickle.load(f)
            with open(os.path.join(retrain_dir, "boundaries_ibg.pkl"), "rb") as f:
                boundaries_ibg = pickle.load(f)
        # Train/val split
        random.seed(42)
        random.shuffle(all_pairs)
        split = int(len(all_pairs) * args.train_frac)
        train_pairs = all_pairs[:split]
        val_pairs   = all_pairs[split:]
    else:
        train_pairs = None
        val_pairs   = None
        boundaries_bi = None
        boundaries_ibg = None

    # Include the (possibly auto-resolved) ibg_integer_bins so all ranks use
    # the same K — only rank 0 has the IBG data needed to compute it.
    ibg_integer_bins_resolved = int(args.ibg_integer_bins) if rank == 0 else None
    to_bcast = [train_pairs, val_pairs, boundaries_bi, boundaries_ibg, ibg_integer_bins_resolved]
    dist.broadcast_object_list(to_bcast, src=0)
    train_pairs, val_pairs, boundaries_bi, boundaries_ibg, ibg_integer_bins_resolved = to_bcast
    args.ibg_integer_bins = int(ibg_integer_bins_resolved)
    print(f"Rank {rank}: {len(train_pairs)} train pairs, {len(val_pairs)} val pairs")

    # 2) build Datasets & DistributedSamplers
    train_ds = PairSeriesDataset(train_pairs)
    val_ds   = PairSeriesDataset(val_pairs)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_as_list,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_as_list,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # 3) model + DDP
    if from_checkpoint:
        model = TwinHeadChronosPredictor.from_pretrained(args.model,device=device)
        print(f"Loaded pretrained model from {args.model}")
    else:
        model = TwinHeadChronosPredictor(
            boundaries_bi,
            boundaries_ibg,
            args.model,
            device,
            num_tokens=args.num_tokens,
            mse_mode=args.mse_mode,
            center_clip=args.center_clip,
            soft_ce_alpha_init=args.soft_ce_alpha_init,
            soft_ce_denom_floor_bi=args.soft_ce_denom_floor_bi,
            soft_ce_denom_floor_ibg=args.soft_ce_denom_floor_ibg,
            auto_tune_soft_ce_sharpness=(not args.freeze_soft_ce_sharpness),
            use_ce_loss=args.use_ce_loss,
            loss_weight_bi=args.loss_weight_bi,
            loss_weight_ibg=args.loss_weight_ibg,
            loss_weight_local_bi=args.loss_weight_local_bi,
            loss_weight_local_ibg=args.loss_weight_local_ibg,
            soft_ce_sigma_local=args.soft_ce_sigma_local,
            local_loss_min_context=args.local_loss_min_context,
            ibg_integer_bins=args.ibg_integer_bins,
            num_local_bins=args.num_local_bins,
            soft_ce_alpha_mode=args.soft_ce_alpha_mode,
        ).to(device)
        # load the model state dict from the given path
        # state = torch.load(args.retrain, map_location=device)
        # Allow partial load in case heads differ
        # try:
        #     state_core = strip_prefix_if_present(state, ["model.model.", "model."])
        #     model.model.load_state_dict(state_core, strict=False)
        # except Exception:
        #     model.load_state_dict(state, strict=False)
        # print(f"Loaded model from {args.retrain}")
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #add a learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)

    # 4) training + validation loop
    best_acc = 10000000
    os.makedirs(args.save_dir, exist_ok=True)
    global_val_ce_baseline = None

    for epoch in range(args.epochs):
        # — train —
        model.train()
        count = 0
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        local_sanity_checked = False
        local_trend_checked = False
        first_local_bi = None
        first_local_ibg = None
        step_idx = 0
        for batch in train_loader:
            # try:
            (
                logits_bi,
                logits_ibg,
                targets_bi,
                targets_ibg,
                valid_bi,
                valid_ibg,
                ids_bi,
                ids_ibg,
                logits_local_bi,
                logits_local_ibg,
                local_targets_bi,
                local_targets_ibg,
                local_valid_bi,
                local_valid_ibg,
            ) = model.module.forward(batch)
            base_loss, ce_bi, ce_ibg, l1_bi, l1_ibg = model.module.compute_loss(
                logits_bi, logits_ibg, targets_bi, targets_ibg, valid_bi, valid_ibg
            )
            loss = base_loss
            local_bi_loss = torch.zeros((), device=device, dtype=loss.dtype)
            local_ibg_loss = torch.zeros((), device=device, dtype=loss.dtype)

            aux_w_bi = float(getattr(model.module, "loss_weight_local_bi", 0.0))
            aux_w_ibg = float(getattr(model.module, "loss_weight_local_ibg", 0.0))
            aux_sigma = float(getattr(model.module, "soft_ce_sigma_local", 1.5))
            aux_min_ctx = int(getattr(model.module, "local_loss_min_context", 16))
            n_tokens = int(getattr(model.module.pipeline.model.config, "n_tokens"))
            n_special = int(getattr(model.module.pipeline.model.config, "n_special_tokens"))

            if (
                model.module.num_local_bins > 0
                and logits_local_bi is not None
                and local_targets_bi is not None
                and local_valid_bi is not None
            ):
                T_local = local_targets_bi.size(1)
                min_ctx = max(0, int(aux_min_ctx))
                pos = torch.arange(T_local, device=device).unsqueeze(0)
                min_ctx_mask = pos >= min_ctx
                local_mask_bi = local_valid_bi & min_ctx_mask
                local_mask_ibg = local_valid_ibg & min_ctx_mask

                if aux_w_bi > 0.0:
                    local_bi_loss = soft_ce_rank(
                        logits_local_bi,
                        local_targets_bi,
                        n_tokens=n_tokens,
                        sigma=aux_sigma,
                        n_special=n_special,
                        mask=local_mask_bi,
                    )
                    loss = loss + aux_w_bi * local_bi_loss

                if aux_w_ibg > 0.0:
                    local_ibg_loss = soft_ce_rank(
                        logits_local_ibg,
                        local_targets_ibg,
                        n_tokens=n_tokens,
                        sigma=aux_sigma,
                        n_special=n_special,
                        mask=local_mask_ibg,
                    )
                    loss = loss + aux_w_ibg * local_ibg_loss

                if rank == 0 and not local_sanity_checked:
                    if aux_w_bi == 0.0 and aux_w_ibg == 0.0:
                        delta = (loss.detach() - base_loss.detach()).abs().item()
                        assert delta <= 1e-6, f"Backward-compat loss mismatch with local weights 0: delta={delta}"
                    else:
                        assert logits_local_bi.shape == logits_bi.shape, (
                            f"local BI logits shape {logits_local_bi.shape} must match global {logits_bi.shape}"
                        )
                        assert logits_local_ibg.shape == logits_ibg.shape, (
                            f"local IBG logits shape {logits_local_ibg.shape} must match global {logits_ibg.shape}"
                        )
                        if aux_w_bi > 0.0:
                            assert torch.isfinite(local_bi_loss), "Local BI loss is non-finite"
                        if aux_w_ibg > 0.0:
                            assert torch.isfinite(local_ibg_loss), "Local IBG loss is non-finite"
                    local_sanity_checked = True

                if rank == 0 and (aux_w_bi > 0.0 or aux_w_ibg > 0.0):
                    if aux_w_bi > 0.0 and first_local_bi is None:
                        first_local_bi = float(local_bi_loss.detach().item())
                    if aux_w_ibg > 0.0 and first_local_ibg is None:
                        first_local_ibg = float(local_ibg_loss.detach().item())
                    if (not local_trend_checked) and step_idx >= 500:
                        if aux_w_bi > 0.0 and first_local_bi is not None:
                            if float(local_bi_loss.detach().item()) > first_local_bi * 1.05:
                                print(
                                    f"[WARN] local BI loss did not decrease by step 500 "
                                    f"(start={first_local_bi:.4f}, now={float(local_bi_loss.detach().item()):.4f})."
                                )
                        if aux_w_ibg > 0.0 and first_local_ibg is not None:
                            if float(local_ibg_loss.detach().item()) > first_local_ibg * 1.05:
                                print(
                                    f"[WARN] local IBG loss did not decrease by step 500 "
                                    f"(start={first_local_ibg:.4f}, now={float(local_ibg_loss.detach().item()):.4f})."
                                )
                        local_trend_checked = True

            optimizer.zero_grad()
            loss.backward()
            # Print loss every 100 steps plus Fano factors (input, ground truth, forecast)
            if rank == 0 and count % 100 == 0:
                alpha_bi_now = float(model.module._soft_ce_alpha("bi").mean().detach().item())
                alpha_ibg_now = float(model.module._soft_ce_alpha("ibg").mean().detach().item())
                # Context (input) values: positions 0..T-2
                ctx_ids_bi = ids_bi[:, :-1]
                ctx_ids_ibg = ids_ibg[:, :-1]
                input_val_bi = model.module.token_values_bi[ctx_ids_bi]
                input_val_ibg = model.module.token_values_ibg[ctx_ids_ibg]
                true_val_bi = model.module.token_values_bi[targets_bi].clone()
                true_val_ibg = model.module.token_values_ibg[targets_ibg].clone()
                pred_val_bi = model.module.token_values_bi[logits_bi.argmax(dim=-1)].clone()
                pred_val_ibg = model.module.token_values_ibg[logits_ibg.argmax(dim=-1)].clone()
                clip = getattr(model.module, "center_clip", 1e6)
                true_val_bi = torch.clamp(true_val_bi, min=-clip, max=clip)
                true_val_ibg = torch.clamp(true_val_ibg, min=-clip, max=clip)
                pred_val_bi = torch.clamp(pred_val_bi, min=-clip, max=clip)
                pred_val_ibg = torch.clamp(pred_val_ibg, min=-clip, max=clip)
                fano_in_bi = fano_factor_tensor(input_val_bi, valid_bi)
                fano_gt_bi = fano_factor_tensor(true_val_bi, valid_bi)
                fano_fc_bi = fano_factor_tensor(pred_val_bi, valid_bi)
                fano_in_ibg = fano_factor_tensor(input_val_ibg, valid_ibg)
                fano_gt_ibg = fano_factor_tensor(true_val_ibg, valid_ibg)
                fano_fc_ibg = fano_factor_tensor(pred_val_ibg, valid_ibg)
                print(
                    f"Epoch {epoch+1}/{args.epochs} step {count} — "
                    f"total={loss.item():.4f} | "
                    f"CE(bi)={ce_bi.item():.4f}, CE(ibg)={ce_ibg.item():.4f} | "
                    f"LocalCE(bi)={local_bi_loss.item():.4f}, LocalCE(ibg)={local_ibg_loss.item():.4f} | "
                    f"L1(bi)={l1_bi.item():.4f}, L1(ibg)={l1_ibg.item():.4f} | "
                    f"alpha(bi)={alpha_bi_now:.3f}, alpha(ibg)={alpha_ibg_now:.3f} | "
                    f"Fano_bi in/gt/fc={fano_in_bi:.4f}/{fano_gt_bi:.4f}/{fano_fc_bi:.4f} | "
                    f"Fano_ibg in/gt/fc={fano_in_ibg:.4f}/{fano_gt_ibg:.4f}/{fano_fc_ibg:.4f}"
                )
            optimizer.step()
            scheduler.step()  # step the scheduler
            total_loss += loss.cpu().item()
            del loss
            count += 1
            step_idx += 1
            # except:
            #     print(f"Skipping batch {count} due to error")
            #     try:
            #         print(f"Batch example shapes: bi={batch[0][0].shape}, ibg={batch[0][1].shape}")
            #     except Exception:
            #         pass
            #     continue

        # — validate —
        model.eval()
        val_sampler.set_epoch(epoch)
        correct_bi = torch.tensor(0, dtype=torch.long, device=device)
        total_bi   = torch.tensor(0, dtype=torch.long, device=device)
        correct_ibg = torch.tensor(0, dtype=torch.long, device=device)
        total_ibg   = torch.tensor(0, dtype=torch.long, device=device)
        correct_local_bi = torch.tensor(0, dtype=torch.long, device=device)
        total_local_bi = torch.tensor(0, dtype=torch.long, device=device)
        correct_local_ibg = torch.tensor(0, dtype=torch.long, device=device)
        total_local_ibg = torch.tensor(0, dtype=torch.long, device=device)
        val_ce_bi_sum = torch.tensor(0.0, device=device)
        val_ce_ibg_sum = torch.tensor(0.0, device=device)
        val_ce_count = torch.tensor(0, dtype=torch.long, device=device)

        with torch.no_grad():
            l1_sum_bi  = torch.tensor(0.0, device=device)
            l1_sum_ibg = torch.tensor(0.0, device=device)
            l1_cnt_bi  = torch.tensor(0, dtype=torch.long, device=device)
            l1_cnt_ibg = torch.tensor(0, dtype=torch.long, device=device)
            for batch in val_loader:
                (
                    logits_bi,
                    logits_ibg,
                    targets_bi,
                    targets_ibg,
                    valid_bi,
                    valid_ibg,
                    _ids_bi,
                    _ids_ibg,
                    logits_local_bi,
                    logits_local_ibg,
                    local_targets_bi,
                    local_targets_ibg,
                    local_valid_bi,
                    local_valid_ibg,
                ) = model.module.forward(batch)
                preds_bi  = logits_bi.argmax(dim=-1)
                preds_ibg = logits_ibg.argmax(dim=-1)
                correct_bi += ((preds_bi == targets_bi) & valid_bi).sum()
                total_bi   += valid_bi.sum()
                correct_ibg += ((preds_ibg == targets_ibg) & valid_ibg).sum()
                total_ibg   += valid_ibg.sum()
                _, ce_bi_val, ce_ibg_val, _l1_bi_val, _l1_ibg_val = model.module.compute_loss(
                    logits_bi, logits_ibg, targets_bi, targets_ibg, valid_bi, valid_ibg
                )
                val_ce_bi_sum += ce_bi_val
                val_ce_ibg_sum += ce_ibg_val
                val_ce_count += 1

                if (
                    model.module.num_local_bins > 0
                    and logits_local_bi is not None
                    and local_targets_bi is not None
                    and local_valid_bi is not None
                ):
                    T_local = local_targets_bi.size(1)
                    min_ctx = max(0, int(getattr(model.module, "local_loss_min_context", 16)))
                    pos = torch.arange(T_local, device=device).unsqueeze(0)
                    min_ctx_mask = pos >= min_ctx
                    eval_mask_bi = local_valid_bi & min_ctx_mask
                    eval_mask_ibg = local_valid_ibg & min_ctx_mask
                    pred_local_bi = logits_local_bi.argmax(dim=-1)
                    pred_local_ibg = logits_local_ibg.argmax(dim=-1)
                    correct_local_bi += ((pred_local_bi == local_targets_bi) & eval_mask_bi).sum()
                    total_local_bi += eval_mask_bi.sum()
                    correct_local_ibg += ((pred_local_ibg == local_targets_ibg) & eval_mask_ibg).sum()
                    total_local_ibg += eval_mask_ibg.sum()

                # Validation L1 according to selected mode
                probs_bi  = torch.softmax(logits_bi, dim=-1)
                probs_ibg = torch.softmax(logits_ibg, dim=-1)
                if args.mse_mode == "centers":
                    # weighted average of token centers (with clamping)
                    # pred_val_bi  = torch.einsum('btv,v->bt', probs_bi,  model.module.token_values_bi)
                    # pred_val_ibg = torch.einsum('btv,v->bt', probs_ibg, model.module.token_values_ibg)
                    true_val_bi  = model.module.token_values_bi[targets_bi]
                    true_val_ibg = model.module.token_values_ibg[targets_ibg]
                    pred_val_bi  = model.module.token_values_bi[preds_bi]
                    pred_val_ibg = model.module.token_values_ibg[preds_ibg]
                    # scale down center values
                    sf = 1
                    pred_val_bi  = pred_val_bi * sf
                    pred_val_ibg = pred_val_ibg
                    true_val_bi  = true_val_bi * sf
                    true_val_ibg = true_val_ibg
                    clip = float(args.center_clip)
                    pred_val_bi  = torch.clamp(pred_val_bi,  min=-clip, max=clip)
                    pred_val_ibg = torch.clamp(pred_val_ibg, min=-clip, max=clip)
                    true_val_bi  = torch.clamp(true_val_bi,  min=-clip, max=clip)
                    true_val_ibg = torch.clamp(true_val_ibg, min=-clip, max=clip)
                    diff_bi  = (pred_val_bi[valid_bi]  - true_val_bi[valid_bi])
                    diff_ibg = (pred_val_ibg[valid_ibg] - true_val_ibg[valid_ibg])
                else:
                    # bin-index expectation
                    vocab_size = logits_bi.size(-1)
                    bin_ids = torch.arange(vocab_size, dtype=torch.float32, device=device)
                    pred_bin_bi  = torch.einsum('btv,v->bt', probs_bi,  bin_ids)
                    pred_bin_ibg = torch.einsum('btv,v->bt', probs_ibg, bin_ids)
                    true_bin_bi  = targets_bi.float()
                    true_bin_ibg = targets_ibg.float()
                    diff_bi  = (pred_bin_bi[valid_bi]  - true_bin_bi[valid_bi])
                    diff_ibg = (pred_bin_ibg[valid_ibg] - true_bin_ibg[valid_ibg])
                # accumulate L1
                l1_sum_bi  += diff_bi.abs().sum()
                l1_sum_ibg += diff_ibg.abs().sum()
                l1_cnt_bi  += valid_bi.sum()
                l1_cnt_ibg += valid_ibg.sum()

        # aggregate across all ranks
        dist.all_reduce(correct_bi, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bi,   op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_ibg, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_ibg,   op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_local_bi, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_local_bi, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_local_ibg, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_local_ibg, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ce_bi_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ce_ibg_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ce_count, op=dist.ReduceOp.SUM)
        # also reduce L1 aggregates
        for t in [l1_sum_bi, l1_sum_ibg]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for t in [l1_cnt_bi, l1_cnt_ibg]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        if rank == 0:
            acc_bi  = (correct_bi.float()  / total_bi.clamp_min(1).float()).item()
            acc_ibg = (correct_ibg.float() / total_ibg.clamp_min(1).float()).item()
            w_bi = float(args.loss_weight_bi)
            w_ibg = float(args.loss_weight_ibg)
            denom_w = w_bi + w_ibg
            if denom_w > 0:
                acc = (w_bi * acc_bi + w_ibg * acc_ibg) / denom_w
            else:
                acc = 0.5 * (acc_bi + acc_ibg)
            l1_bi_val  = (l1_sum_bi / l1_cnt_bi.clamp_min(1)).item()
            l1_ibg_val = (l1_sum_ibg / l1_cnt_ibg.clamp_min(1)).item()
            local_acc_bi = (correct_local_bi.float() / total_local_bi.clamp_min(1).float()).item()
            local_acc_ibg = (correct_local_ibg.float() / total_local_ibg.clamp_min(1).float()).item()
            val_ce_bi = (val_ce_bi_sum / val_ce_count.clamp_min(1)).item()
            val_ce_ibg = (val_ce_ibg_sum / val_ce_count.clamp_min(1)).item()
            print(
                f"Epoch {epoch+1}/{args.epochs} — val accuracy: bi={acc_bi:.4f}, ibg={acc_ibg:.4f}, avg={acc:.4f} | "
                f"val L1: bi={l1_bi_val:.4f}, ibg={l1_ibg_val:.4f} | "
                f"val CE: bi={val_ce_bi:.4f}, ibg={val_ce_ibg:.4f} | "
                f"val local acc (post-min-context): bi={local_acc_bi:.4f}, ibg={local_acc_ibg:.4f}"
            )

            w_bi_ce = float(getattr(model.module, "loss_weight_bi", 1.0))
            w_ibg_ce = float(getattr(model.module, "loss_weight_ibg", 1.0))
            denom_ce = max(w_bi_ce + w_ibg_ce, 1e-12)
            global_val_ce = (w_bi_ce * val_ce_bi + w_ibg_ce * val_ce_ibg) / denom_ce
            if global_val_ce_baseline is None:
                global_val_ce_baseline = global_val_ce
            elif (
                (float(getattr(model.module, "loss_weight_local_bi", 0.0)) > 0.0
                 or float(getattr(model.module, "loss_weight_local_ibg", 0.0)) > 0.0)
                and global_val_ce > 1.02 * global_val_ce_baseline
            ):
                print(
                    f"[WARN] Weighted global val CE rose by >2% vs run-start baseline proxy "
                    f"({global_val_ce:.4f} vs {global_val_ce_baseline:.4f}). "
                    "Aux local loss may be too strong."
                )

            if (
                model.module.num_local_bins > 0
                and (float(getattr(model.module, "loss_weight_local_bi", 0.0)) > 0.0
                     or float(getattr(model.module, "loss_weight_local_ibg", 0.0)) > 0.0)
                and epoch + 1 >= 5
            ):
                if total_local_bi.item() > 0 and (local_acc_bi < 0.30 or local_acc_bi > 0.85):
                    print(
                        f"[WARN] BI local val top-1={local_acc_bi:.3f} outside expected [0.30, 0.85]. "
                        "Consider tuning soft_ce_sigma_local or local loss weight."
                    )
                if total_local_ibg.item() > 0 and (local_acc_ibg < 0.30 or local_acc_ibg > 0.85):
                    print(
                        f"[WARN] IBG local val top-1={local_acc_ibg:.3f} outside expected [0.30, 0.85]. "
                        "Consider tuning soft_ce_sigma_local or local loss weight."
                    )

            # save best (weighted L1 matches training loss weighting)
            score_val = w_bi * l1_bi_val + w_ibg * l1_ibg_val
            if score_val < best_acc:
                best_acc = score_val
                path = os.path.join(args.save_dir, "chronos_best.pt")
                #save pretrained
                model.module.save_pretrained(args.save_dir)
                torch.save(model.module.state_dict(), path)
                print(f"→ New best model (acc={acc:.4f}) saved to {path}")

    dist.destroy_process_group()
