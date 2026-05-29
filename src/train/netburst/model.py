"""NetBurst model, tokenizers, and checkpoint loaders."""

from __future__ import annotations

import json
import math
import os
import pickle
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from chronos import ChronosConfig, ChronosModel, ChronosPipeline, ChronosTokenizer


def soft_ce_rank(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    n_tokens: int,
    sigma: float,
    n_special: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Soft cross-entropy with Gaussian-over-rank targets.

    logits:     [B, T, V]
    target_ids: [B, T] (token ids, including special-token offset)
    mask:       [B, T] bool (optional). False positions are excluded.
    """
    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError(
            f"soft_ce_rank expects logits[B,T,V] and targets[B,T], got {logits.shape} and {target_ids.shape}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    B, T, V = logits.shape
    if target_ids.shape[0] != B or target_ids.shape[1] != T:
        raise ValueError(
            f"target_ids shape {target_ids.shape} must match logits[:2] {(B, T)}"
        )

    device = logits.device
    dtype = logits.dtype
    target_ids = target_ids.to(device=device)
    if mask is None:
        mask = torch.ones_like(target_ids, dtype=torch.bool, device=device)
    else:
        mask = mask.to(device=device, dtype=torch.bool)

    valid_targets = (
        (target_ids >= int(n_special))
        & (target_ids < int(n_tokens))
    )
    valid = mask & valid_targets
    valid_count = valid.sum()
    if valid_count.item() == 0:
        # Preserve graph/device semantics.
        return logits.sum() * 0.0

    vocab_ids = torch.arange(V, device=device, dtype=dtype).view(1, 1, V)
    target_rank = target_ids.to(dtype=dtype).unsqueeze(-1)  # [B, T, 1]
    inv_two_sigma2 = 1.0 / (2.0 * float(sigma) * float(sigma))
    dist2 = (vocab_ids - target_rank) ** 2
    target_dist = torch.exp(-dist2 * inv_two_sigma2)  # [B, T, V]

    real_vocab = (
        (vocab_ids >= float(n_special))
        & (vocab_ids < float(n_tokens))
    ).to(dtype=dtype)
    target_dist = target_dist * real_vocab
    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    log_probs = torch.log_softmax(logits, dim=-1)
    per_pos = -(target_dist * log_probs).sum(dim=-1)  # [B, T]
    per_pos = per_pos * valid.to(dtype=dtype)
    return per_pos.sum() / valid_count.to(dtype=dtype).clamp_min(1.0)

class MyChronosPipeline(ChronosPipeline):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Use the parent `from_pretrained` method and replace the model with MyChronosModel.
        """
        # Call the parent `from_pretrained` method to get the pipeline
        pipeline = super().from_pretrained(*args, **kwargs)
        
        # Replace the model in the pipeline with MyChronosModel
        pipeline.model = MyChronosModel(
            config=pipeline.model.config,  # Use the same configuration
            model=pipeline.model.model,   # Use the underlying pretrained model
        )
        
        # Return the updated pipeline
        return pipeline


class GlobalQuantileBins(ChronosTokenizer):
    def __init__(self, boundaries: np.ndarray, config: ChronosConfig):
        self.config     = config
        # boundaries: 1D numpy array of length B+1: [-inf, q1, q2, ..., inf]
        # convert to torch tensor on CPU; will be moved in bucketize
        self.boundaries = torch.tensor(boundaries, dtype=torch.float32)

    def context_input_transform(self, context: torch.Tensor):
        # identical to uniform but with quantile boundaries
        context = context.to(dtype=torch.float32)
        attention_mask = ~torch.isnan(context) & (context >= 0.000000000001)

        # compute per-series scale as before
        # scale = torch.nansum(torch.abs(context) * attention_mask, dim=-1)  \
        #         / torch.nansum(attention_mask, dim=-1)
        # scale[~(scale > 0)] = 1.0

        # scaled = context / scale.unsqueeze(-1)
        scaled = context 
        # bucketize against global quantile boundaries
        # bucketize expects sorted boundaries; right=True to match half-open
        token_ids = torch.bucketize(
            context,
            self.boundaries.to(scaled.device),
            right=False
        )
        # shift by special tokens
        token_ids = token_ids + self.config.n_special_tokens
        token_ids.clamp_(0, self.config.n_tokens - 1)
        token_ids[~attention_mask] = self.config.pad_token_id

        # append EOS if needed
        if self.config.use_eos_token and self.config.model_type == 'seq2seq':
            eos = torch.full((context.shape[0],1), self.config.eos_token_id)
            mask_eos = torch.ones_like(eos, dtype=torch.bool)
            token_ids     = torch.cat([token_ids, eos], dim=1)
            attention_mask= torch.cat([attention_mask, mask_eos], dim=1)
        return token_ids, attention_mask, 1

    def label_input_transform(self, label: torch.Tensor, scale: torch.Tensor):
        # same binning on labels
        label = label.to(dtype=torch.float32)
        token_ids = torch.bucketize(
            label / scale.unsqueeze(-1),
            self.boundaries.to(label.device),
            right=True
        ) + self.config.n_special_tokens
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        if self.config.use_eos_token:
            eos = torch.full((label.shape[0],1), self.config.eos_token_id)
            token_ids      = torch.cat([token_ids, eos], dim=1)
            attention_mask = torch.cat([attention_mask, eos==eos], dim=1)
        return token_ids, attention_mask

    def output_transform(self, samples: torch.Tensor, scale: Optional[torch.Tensor]):
        # Map tokens back to finite bin representatives (centers between boundaries)
        # Compute centers purely in torch to keep device/dtype consistent
        b = self.boundaries.to(samples.device, dtype=torch.float32)  # [B+1]
        centers = 0.5 * (b[:-1] + b[1:])                             # [B]
        # Avoid using a sentinel huge last boundary (e.g., 1e20) for the last center
        last_edge = b[-1]
        if not torch.isfinite(last_edge) or (last_edge > 1e19):
            centers[-1] = b[-2]

        # Align centers with how tokens were created: we added n_special_tokens during
        # context_input_transform, so we subtract exactly that here. No extra -1.
        idx = samples.long() - self.config.n_special_tokens
        idx = idx.clamp(0, centers.numel() - 1)
        return centers[idx]
# --- end tokenizer ---


class LocalQuantileBins(ChronosTokenizer):
    """Per-series empirical quantile binning (input-only residual stream)."""

    def __init__(
        self,
        num_local_bins: int,
        config: ChronosConfig,
        min_valid: float = 1e-12,
    ):
        self.config = config
        max_real = int(getattr(config, "n_tokens")) - int(getattr(config, "n_special_tokens"))
        self.num_local_bins = int(max(0, min(int(num_local_bins), max_real)))
        self.min_valid = float(min_valid)

    def _valid_mask(self, x: torch.Tensor) -> torch.Tensor:
        return ~torch.isnan(x) & (x >= self.min_valid)

    def _compute_edges(self, context: torch.Tensor, attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        K = self.num_local_bins
        if K <= 0:
            return None
        q_levels = torch.arange(1, K, device=context.device, dtype=torch.float32) / K
        ctx_nan = context.masked_fill(~attention_mask, float("nan"))
        edges = torch.nanquantile(ctx_nan, q_levels, dim=1)  # [K-1, B]
        edges = edges.transpose(0, 1).contiguous()  # [B, K-1]
        edges = torch.nan_to_num(edges, nan=0.0)
        edges, _ = torch.sort(edges, dim=1)
        return edges

    def _tokenize_with_edges(
        self,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        edges: Optional[torch.Tensor],
    ) -> torch.Tensor:
        K = self.num_local_bins
        if K <= 0 or edges is None:
            token_ids = torch.full(
                values.shape, self.config.pad_token_id, dtype=torch.long, device=values.device
            )
            token_ids[attention_mask] = self.config.n_special_tokens
            return token_ids

        token_ids = torch.searchsorted(edges, values, right=False)
        token_ids = token_ids + self.config.n_special_tokens
        token_ids.clamp_(0, self.config.n_tokens - 1)
        token_ids[~attention_mask] = self.config.pad_token_id
        return token_ids

    def context_input_transform(self, context: torch.Tensor, return_edges: bool = False):
        context = context.to(dtype=torch.float32)
        attention_mask = self._valid_mask(context)
        edges = self._compute_edges(context, attention_mask)
        token_ids = self._tokenize_with_edges(context, attention_mask, edges)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            eos = torch.full((context.shape[0], 1), self.config.eos_token_id)
            mask_eos = torch.ones_like(eos, dtype=torch.bool)
            token_ids = torch.cat([token_ids, eos], dim=1)
            attention_mask = torch.cat([attention_mask, mask_eos], dim=1)
        if return_edges:
            return token_ids, attention_mask, 1, edges
        return token_ids, attention_mask, 1

    def label_input_transform(
        self,
        label: torch.Tensor,
        edges: Optional[torch.Tensor],
        append_eos: bool = False,
    ):
        label = label.to(dtype=torch.float32)
        attention_mask = self._valid_mask(label)
        token_ids = self._tokenize_with_edges(label, attention_mask, edges)
        if append_eos and self.config.use_eos_token and self.config.model_type == "seq2seq":
            eos = torch.full((label.shape[0], 1), self.config.eos_token_id)
            mask_eos = torch.ones_like(eos, dtype=torch.bool)
            token_ids = torch.cat([token_ids, eos], dim=1)
            attention_mask = torch.cat([attention_mask, mask_eos], dim=1)
        return token_ids, attention_mask
# --- end LocalQuantileBins ---


class IntegerIBGBins(ChronosTokenizer):
    """Tokenize IBG (inter-burst-gap) as non-negative integer indices.

    IBG values are naturally integer-valued indices (burst positions); quantile
    binning collapses the heavy tail to a single plateau token and wastes
    vocabulary on duplicates. Here we map value ``v`` -> token id
    ``n_special + clamp(round(v), 0, K-1)``. ``K`` is the cap; values >= K are
    clamped to the last bin. Values of 0 are VALID (not masked out) — they
    simply map to bin 0. Negative values and NaNs are masked to PAD.
    """

    def __init__(self, num_int_bins: int, config: ChronosConfig):
        self.config = config
        # K = effective integer-bin count; cannot exceed real-token capacity.
        max_real = int(getattr(config, "n_tokens")) - int(getattr(config, "n_special_tokens"))
        self.num_int_bins = int(max(1, min(int(num_int_bins), max_real)))

    # Metadata dict used for save / restore. Keeping a single source of truth
    # makes `TwinHeadChronosPredictor.save_pretrained` reconstruction robust.
    @property
    def boundaries_like(self):
        return {"num_int_bins": self.num_int_bins}

    def context_input_transform(self, context: torch.Tensor):
        context = context.to(dtype=torch.float32)
        # Valid iff finite AND non-negative. 0 is valid (unlike the quantile path).
        attention_mask = torch.isfinite(context) & (context >= 0.0)

        # Round to nearest int, clamp to [0, K-1], offset by n_special_tokens.
        ids = torch.round(context).long()
        ids.clamp_(0, self.num_int_bins - 1)
        token_ids = ids + self.config.n_special_tokens
        token_ids.clamp_(0, self.config.n_tokens - 1)
        token_ids[~attention_mask] = self.config.pad_token_id

        if self.config.use_eos_token and self.config.model_type == 'seq2seq':
            eos = torch.full((context.shape[0], 1), self.config.eos_token_id)
            mask_eos = torch.ones_like(eos, dtype=torch.bool)
            token_ids = torch.cat([token_ids, eos], dim=1)
            attention_mask = torch.cat([attention_mask, mask_eos], dim=1)
        return token_ids, attention_mask, 1

    def label_input_transform(self, label: torch.Tensor, scale: torch.Tensor):
        label = label.to(dtype=torch.float32)
        ids = torch.round(label).long().clamp(0, self.num_int_bins - 1)
        token_ids = ids + self.config.n_special_tokens
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        if self.config.use_eos_token:
            eos = torch.full((label.shape[0], 1), self.config.eos_token_id)
            token_ids = torch.cat([token_ids, eos], dim=1)
            attention_mask = torch.cat([attention_mask, eos == eos], dim=1)
        return token_ids, attention_mask

    def output_transform(self, samples: torch.Tensor, scale: Optional[torch.Tensor]):
        # Token k (k >= n_special) decodes to integer (k - n_special).
        # Tokens in [0, n_special) or beyond (n_special + K) are clamped to valid range.
        idx = samples.long() - self.config.n_special_tokens
        idx = idx.clamp(0, self.num_int_bins - 1)
        return idx.to(dtype=torch.float32)
# --- end IntegerIBGBins ---


class MyChronosModel(ChronosModel):

    def __init__(
        self,
        config: ChronosConfig,
        model: Optional[torch.nn.Module] = None,
        num_local_bins: int = 0,
    ):
        super().__init__(config, model)
        if "d_model" not in config.__dict__:
            config.d_model = 512  # default value
        self.num_local_bins = int(num_local_bins)
        self.embedding2 = nn.Embedding(config.n_tokens, config.d_model)
        # Warm-start the IBG (stream-2) embedding table from the pretrained BI/T5
        # token embeddings so both streams start from the same well-conditioned prior,
        # instead of stream-2 being random noise relative to the pretrained stream-1.
        # Guard on availability (self.model may be None if constructed without a
        # backing HF model) and on matching shape (n_tokens may be smaller than the
        # backbone's vocab_size when num_tokens is used to cap the effective vocab).
        try:
            if getattr(self, "model", None) is not None:
                src_emb = self.model.get_input_embeddings()
                if src_emb is not None and src_emb.weight is not None:
                    src_w = src_emb.weight.detach()
                    n_copy = min(self.embedding2.weight.shape[0], src_w.shape[0])
                    d_copy = min(self.embedding2.weight.shape[1], src_w.shape[1])
                    with torch.no_grad():
                        self.embedding2.weight[:n_copy, :d_copy].copy_(
                            src_w[:n_copy, :d_copy].to(
                                dtype=self.embedding2.weight.dtype,
                                device=self.embedding2.weight.device,
                            )
                        )
        except Exception as _e:
            # Non-fatal: fall back to the default random init for embedding2.
            print(f"[MyChronosModel] embedding2 warm-start skipped: {_e}")
        self.pre = nn.LayerNorm(2*config.d_model)
        self.fc1 = nn.Linear(2*config.d_model, 4*config.d_model, bias=False)
        self.act = nn.GELU()  # or nn.SiLU()
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(4*config.d_model, config.d_model, bias=False)
        self.drop2 = nn.Dropout(0.1)
        self.skip = nn.Linear(2*config.d_model, config.d_model, bias=False)

        if self.num_local_bins > 0:
            self.embedding1_local = nn.Embedding(config.n_tokens, config.d_model)
            self.embedding2_local = nn.Embedding(config.n_tokens, config.d_model)
            nn.init.zeros_(self.embedding1_local.weight)
            nn.init.zeros_(self.embedding2_local.weight)
            self.lm_head_local_bi = nn.Linear(config.d_model, config.n_tokens, bias=False)
            self.lm_head_local_ibg = nn.Linear(config.d_model, config.n_tokens, bias=False)
            self._tie_local_head_weights()

    def _tie_local_head_weights(self) -> None:
        if hasattr(self, "lm_head_local_bi") and hasattr(self, "embedding1_local"):
            self.lm_head_local_bi.weight = self.embedding1_local.weight
        if hasattr(self, "lm_head_local_ibg") and hasattr(self, "embedding2_local"):
            self.lm_head_local_ibg.weight = self.embedding2_local.weight

    def heads(self, decoder_hidden: torch.Tensor) -> dict:
        out = {
            "logits_global_bi": self.model.lm_head(decoder_hidden),
        }
        if self.num_local_bins > 0 and hasattr(self, "lm_head_local_bi"):
            out["logits_local_bi"] = self.lm_head_local_bi(decoder_hidden)
            out["logits_local_ibg"] = self.lm_head_local_ibg(decoder_hidden)
        return out

    @classmethod
    def from_pretrained(cls, *args, boundaries=None, **kwargs):
        pipe = super().from_pretrained(*args, **kwargs)
        pipe.model = MyChronosModel(config=pipe.model.config, model=pipe.model.model)
        if boundaries is not None:
            pipe.tokenizer = GlobalQuantileBins(boundaries, pipe.model.config)
        return pipe
        
    
    def forward_with_embeddings(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        type_id: Optional[int] = None,
        cross_attend: bool = False,
        local_ids1: Optional[torch.Tensor] = None,
        local_ids2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.model.device
        input_ids1 = input_ids1.to(device)
        input_ids2 = input_ids2.to(device)
        attention_mask = attention_mask.to(device).long()

        B, T = input_ids1.size()

        if self.num_local_bins > 0 and not isinstance(decoder_input_ids, tuple):
            if local_ids1 is None or local_ids2 is None:
                raise ValueError(
                    "num_local_bins > 0 requires local_ids1 and local_ids2 on teacher-forcing "
                    "paths (decoder_input_ids not a tuple). AR tuple decoding intentionally "
                    "omits local ids."
                )

        # ----- Token embedding fusion for both streams -----
        def tok_emb_layer(
            ids1,
            ids2,
            loc_ids1=None,
            loc_ids2=None,
        ):
            return self.fused_stream_embeddings(
                ids1,
                ids2,
                local_ids1=loc_ids1,
                local_ids2=loc_ids2,
            )

        # ----- Optional encoder (for cross-attention) -----
        # By default, we DISABLE cross-attention (decoder-only) to avoid future-token leakage
        # when source and target are the same (next-token prediction). Set cross_attend=True
        # if you explicitly want to use encoder cross-attention.
        if cross_attend:
            tok_emb = tok_emb_layer(input_ids1, input_ids2, local_ids1, local_ids2)  # [B,T,D]
            enc_out = self.model.encoder(
                inputs_embeds=tok_emb,
                attention_mask=attention_mask,
                return_dict=True
            )
            enc_h = enc_out.last_hidden_state  # [B, T, D]
            enc_attn_mask = attention_mask
        else:
            enc_h = None
            enc_attn_mask = None

        # ----- Decoder (causal) -----
        # If caller provides decoder_input_ids as a tuple (ids1, ids2), use AR mode.
        # Otherwise, build teacher-forcing shifted inputs from encoder inputs.
        if isinstance(decoder_input_ids, tuple):
            dec_input_ids1, dec_input_ids2 = decoder_input_ids
            dec_input_ids1 = dec_input_ids1.to(device)
            dec_input_ids2 = dec_input_ids2.to(device)
            # AR tuple path: local residual not applied in decoder (per spec).
            dec_in = tok_emb_layer(dec_input_ids1, dec_input_ids2, None, None)  # [B, T_dec, D]
            # Decoder padding mask (float32): assume all valid in AR tuple path
            dec_pad = torch.ones((B, dec_in.size(1)), device=device, dtype=torch.float32)
        else:
            # Teacher-forcing mode: shifted encoder token IDs
            dec_input_ids1 = torch.cat([
                torch.full((B, 1), self.model.config.pad_token_id, dtype=torch.long, device=device),
                input_ids1[:, :-1]
            ], dim=1)
            dec_input_ids2 = torch.cat([
                torch.full((B, 1), self.model.config.pad_token_id, dtype=torch.long, device=device),
                input_ids2[:, :-1]
            ], dim=1)
            dec_local_ids1 = dec_local_ids2 = None
            if local_ids1 is not None:
                dec_local_ids1 = torch.cat([
                    torch.full((B, 1), self.model.config.pad_token_id, dtype=torch.long, device=device),
                    local_ids1[:, :-1].to(device),
                ], dim=1)
                dec_local_ids2 = torch.cat([
                    torch.full((B, 1), self.model.config.pad_token_id, dtype=torch.long, device=device),
                    local_ids2[:, :-1].to(device),
                ], dim=1)
            dec_in = tok_emb_layer(
                dec_input_ids1, dec_input_ids2, dec_local_ids1, dec_local_ids2
            )
            # Decoder padding mask (float32): leading token is valid; rest from encoder mask shifted
            dec_pad = torch.cat([
                torch.ones((B, 1), device=device, dtype=torch.float32),
                attention_mask[:, :-1].float()
            ], dim=1)  # [B, T_dec]
        # Build explicit causal + padding mask for decoder self-attention (4D additive mask)
        #   - causal: upper triangle masked (no look-ahead)
        #   - padding: time steps with dec_pad == 0 masked
        T_dec = dec_in.size(1)
        # Use a large finite negative to avoid NaNs from all -inf rows in softmax
        attn_dtype = dec_in.dtype
        neg_large = torch.finfo(attn_dtype).min if attn_dtype in (torch.float16, torch.bfloat16, torch.float32) else -1e9

        causal = torch.triu(
            torch.full((T_dec, T_dec), neg_large, device=device, dtype=attn_dtype), diagonal=1
        )  # [T_dec, T_dec]
        pad_keep = dec_pad.to(attn_dtype)[:, None, None, :]   # [B, 1, 1, T_dec]
        pad_mask = (1.0 - pad_keep) * neg_large               # neg_large where PAD, 0 elsewhere
        dec_attn = causal.unsqueeze(0).unsqueeze(0) + pad_mask  # [B, 1, T_dec, T_dec]

        # Ensure at least one unmasked element per row (the diagonal)
        idx = torch.arange(T_dec, device=device)
        dec_attn[:, :, idx, idx] = 0.0

        # Decoder forward pass with enforced causal mask
        dec_out = self.model.decoder(
            inputs_embeds=dec_in,
            attention_mask=dec_attn,
            encoder_hidden_states=enc_h,            # None disables cross-attention
            encoder_attention_mask=enc_attn_mask,   # None
            use_cache=False,
            return_dict=True,
        )
        dec_h = dec_out.last_hidden_state  # [B, T, D]

        # ----- LM head -----
        logits = self.model.lm_head(dec_h)  # [B, T, V]

        return logits.unsqueeze(1), dec_h.unsqueeze(1)

    def fused_stream_embeddings(
        self,
        input_ids1: torch.Tensor,
        input_ids2: torch.Tensor,
        local_ids1: Optional[torch.Tensor] = None,
        local_ids2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build fused BI/IBG token embeddings before encoder/decoder blocks."""
        device = self.model.device
        emb1 = self.model.get_input_embeddings()(input_ids1.to(device))
        emb2 = self.embedding2(input_ids2.to(device))
        if local_ids1 is not None and hasattr(self, "embedding1_local"):
            emb1 = emb1 + self.embedding1_local(local_ids1.to(device))
            emb2 = emb2 + self.embedding2_local(local_ids2.to(device))
        emb = torch.cat([emb1, emb2], dim=-1)
        x = self.pre(emb)
        x2 = self.fc1(x)
        x2 = self.act(x2)
        x2 = self.drop1(x2)
        x2 = self.fc2(x2)
        x2 = self.drop2(x2)
        skip = self.skip(emb)
        return x2 + skip

class ChronosBinPredictor(nn.Module):
    """
    Wraps ChronosPipeline to:
      1) prepare inputs,
      2) run Chronos to get per‐step logits over bins,
      3) expose logits and true token IDs,
      4) compute cross‐entropy loss.
    """
    def __init__(self, boundaries, pretrained_model_or_path: str, device: torch.device):
        super().__init__()
        # load Chronos T5 pipeline & swap in ChronosModel if needed
        self.pipeline: MyChronosPipeline = MyChronosPipeline.from_pretrained(
            pretrained_model_or_path,
            device_map={"": device},
            torch_dtype=torch.float32
        )
        self.pipeline.tokenizer = GlobalQuantileBins(boundaries, self.pipeline.model.config)
        # ensure model is on the right device
        orig_model = self.pipeline.model.to(device)
        self.device = device

        # if it’s not already our subclass, build one, copy weights, swap it in
        if isinstance(orig_model, MyChronosModel):
            self.model = orig_model
        else:
            # instantiate a MyChronosModel with the same config
            new_model = MyChronosModel(orig_model.config)
            # copy across all pretrained weights
            new_model.load_state_dict(orig_model.state_dict(), strict=True)
            # move it to device
            new_model.to(device)
            # swap it into the pipeline
            self.pipeline.model = new_model
            self.model = new_model

        # CE loss for bins (kept for potential fallback)
        self.ce_loss = nn.CrossEntropyLoss()

        # Precompute token->value map for soft targets
        with torch.no_grad():
            vocab_size = int(getattr(self.pipeline.model.config, "n_tokens"))
            tok_ids = torch.arange(vocab_size, dtype=torch.long, device=self.device)
            vals = self.pipeline.tokenizer.output_transform(tok_ids, None).to(self.device).float()
        self.register_buffer("token_values", vals)

    # def forward(self, timeseries: torch.Tensor):
    #     """
    #     timeseries: [B, T] float tensor of raw values
    #     valid_mask: [B, T] bool mask (True where real data exists, False for padding)
    #     Returns:
    #       logits:    [B, T', V]  where V=|vocab| (# of bins) and T' = T-1 (we predict next‐token)
    #       target:    [B, T'] long tensor of true bin ids
    #     """
    #     # 1) Use ChronosPipeline to turn floats → input_ids, attention_mask, scale
    #     #    `context` expects a list or tuple of one or more time‐series arrays.
    #     #check nan in timeseries
    #     context = self.pipeline._prepare_and_validate_context(context=timeseries)
    #     # if context has 4 in it print it
    #     input_ids, attention_mask, scale = self.pipeline.tokenizer.context_input_transform(context)

    #     # 2) chop off the last EOS from inputs (Chronos expects decoder input shifted by 1)
    #     decoder_input_ids = torch.full((len(timeseries), 1), 
    #                                    self.model.config.pad_token_id, 
    #                                    dtype=torch.long)

    #     # 3) run ChronosModel.forward_with_embeddings to get logits
    #     #    returns: logits: [B, 1, T, V], _hidden = ...
    #     forinput = torch.concat([input_ids[:,:-2],input_ids[:,-1:]], dim = -1)
    #     firstAttentionMask = torch.concat([attention_mask[:,:-2],attention_mask[:,-1:]], dim = -1)
    #     logits, _hidden = self.model.forward_with_embeddings(
    #         input_ids=forinput.to(self.device),
    #         attention_mask=firstAttentionMask.to(self.device),
    #         decoder_input_ids=decoder_input_ids.to(self.device)
    #     )

    #     # squeeze out the extra “step” dim: → [B, T, V]
    #     logits = logits.squeeze(1)

    #     # 4) build target tokens: Chronos tokenizer output_transform inversion
    #     #    the “output_transform” returns continuous floats, but underlying the pipeline
    #     #    the “labels” are the token IDs in input_ids[:,1:]
    #     # here we grab the true token IDs directly:
    #     target_ids = input_ids[:, 1 : logits.size(1)].to(self.device)

    #     return logits[:,1:,:], target_ids, firstAttentionMask[:,1:]

    def forward(self, timeseries: torch.Tensor):
        context = self.pipeline._prepare_and_validate_context(context=timeseries)
        input_ids, attention_mask, _ = self.pipeline.tokenizer.context_input_transform(context)

        dummy = torch.empty((input_ids.size(0), 1), dtype=torch.long, device=self.device)
        logits, _ = self.model.forward_with_embeddings(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            decoder_input_ids=dummy
        )
        logits = logits.squeeze(1)                   # [B, T, V]

        # Next-token targets (shifted-left)
        targets = input_ids[:, 1:].to(self.device)   # [B, T-1]
        logits  = logits[:, :-1, :]                  # [B, T-1, V]

        # Valid positions for loss (reflects ≥0.1 threshold): both t and t+1 must be valid
        valid = attention_mask[:, 1:].to(self.device)  # [B, T-1]

        return logits, targets, valid

    def compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor, valid_mask: torch.Tensor):
        """
        logits:    [B, T, V]
        target_ids:[B, T]  long
        valid_mask:[B, T+1] bool (True where we had data; we drop the first step)
        """
        # Flatten only the valid positions
        logits_flat  = logits[valid_mask]           # [N_valid, V]
        targets_flat = target_ids[valid_mask]       # [N_valid]

        # Soft target weights based on numeric bin values
        gt_vals = self.token_values[targets_flat]   # [N_valid]
        denom = torch.clamp(gt_vals, min=1e-12)
        bin_vals = self.token_values.unsqueeze(0).expand(gt_vals.size(0), -1)  # [N_valid, V]
        weights = torch.exp(- torch.abs(bin_vals - gt_vals.unsqueeze(1)) / denom.unsqueeze(1))
        soft_targets = weights / weights.sum(dim=1, keepdim=True)

        log_probs = torch.log_softmax(logits_flat, dim=-1)  # [N_valid, V]
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss


class TwinHeadChronosPredictor(nn.Module):
    def __init__(
        self,
        boundaries_bi,
        boundaries_ibg,
        pretrained_model_or_path: str,
        device: torch.device,
        num_tokens: Optional[int] = None,
        mse_mode: str = "bin",
        center_clip: float = 1e6,
        use_cross_attn: bool = False,
        soft_ce_alpha_init: float = 1.0,
        soft_ce_denom_floor_bi: float = 1e-3,
        soft_ce_denom_floor_ibg: float = 1.0,
        auto_tune_soft_ce_sharpness: bool = True,
        use_ce_loss: bool = False,
        loss_weight_bi: float = 1.0,
        loss_weight_ibg: float = 1.0,
        loss_weight_local_bi: float = 0.0,
        loss_weight_local_ibg: float = 0.0,
        soft_ce_sigma_local: float = 1.5,
        local_loss_min_context: int = 16,
        ibg_integer_bins: int = 0,
        num_local_bins: int = 0,
        soft_ce_alpha_mode: str = "scalar",
    ):
        super().__init__()

        # --- store for save_pretrained / from_pretrained ---
        self._boundaries_bi = boundaries_bi
        self._boundaries_ibg = boundaries_ibg
        self._pretrained_model_or_path = pretrained_model_or_path
        self._num_tokens = num_tokens
        self.mse_mode = mse_mode
        self.center_clip = float(center_clip)
        self.device = device
        self.soft_ce_denom_floor_bi = float(max(soft_ce_denom_floor_bi, 1e-12))
        self.soft_ce_denom_floor_ibg = float(max(soft_ce_denom_floor_ibg, 1e-12))
        self.auto_tune_soft_ce_sharpness = bool(auto_tune_soft_ce_sharpness)
        self.use_ce_loss = bool(use_ce_loss)
        self.soft_ce_alpha_mode = str(soft_ce_alpha_mode).lower()
        if self.soft_ce_alpha_mode not in ("scalar", "perbin", "parametric"):
            raise ValueError(
                "soft_ce_alpha_mode must be one of scalar|perbin|parametric, "
                f"got {soft_ce_alpha_mode!r}"
            )
        self.loss_weight_bi = float(loss_weight_bi)
        self.loss_weight_ibg = float(loss_weight_ibg)
        self.loss_weight_local_bi = float(loss_weight_local_bi)
        self.loss_weight_local_ibg = float(loss_weight_local_ibg)
        self.soft_ce_sigma_local = float(soft_ce_sigma_local)
        self.local_loss_min_context = int(local_loss_min_context)
        # If > 0, use IntegerIBGBins with K = ibg_integer_bins (integer tokenizer for IBG).
        # If 0 (default), use the original GlobalQuantileBins on IBG (backward-compat).
        self.ibg_integer_bins = int(ibg_integer_bins)
        self.num_local_bins = int(num_local_bins)
        # Whether to enable encoder cross-attention in the inner model
        self.use_cross_attn = bool(use_cross_attn)

        # Load pipeline & Chronos model
        self.pipeline = MyChronosPipeline.from_pretrained(
            pretrained_model_or_path,
            device_map={"": device},
            torch_dtype=torch.float32,
        )

        cfg = self.pipeline.model.config
        if num_tokens is not None:
            total_eff = int(
                min(
                    getattr(cfg, "n_tokens"),
                    getattr(cfg, "n_special_tokens")
                    + int(num_tokens),
                )
            )
            cfg.n_tokens = total_eff

        self.tokenizer_bi = GlobalQuantileBins(boundaries_bi, cfg)
        if self.ibg_integer_bins and self.ibg_integer_bins > 0:
            # Integer-index tokenizer: valid K bins starting at 0; IBG=0 is kept.
            self.tokenizer_ibg = IntegerIBGBins(self.ibg_integer_bins, cfg)
        else:
            self.tokenizer_ibg = GlobalQuantileBins(boundaries_ibg, cfg)

        if self.num_local_bins > 0:
            self.local_binner_bi = LocalQuantileBins(self.num_local_bins, cfg, min_valid=1e-12)
            ibg_min_valid = 0.0 if (self.ibg_integer_bins and self.ibg_integer_bins > 0) else 1e-12
            self.local_binner_ibg = LocalQuantileBins(
                self.num_local_bins, cfg, min_valid=ibg_min_valid
            )
        else:
            self.local_binner_bi = None
            self.local_binner_ibg = None

        orig_model = self.pipeline.model.to(device)
        backbone = getattr(orig_model, "model", None)
        if self.num_local_bins > 0:
            new_model = MyChronosModel(
                orig_model.config, model=backbone, num_local_bins=self.num_local_bins
            )
            new_model.load_state_dict(orig_model.state_dict(), strict=False)
            new_model._tie_local_head_weights()
            new_model.to(device)
            self.pipeline.model = new_model
            self.model = new_model
        elif isinstance(orig_model, MyChronosModel):
            self.model = orig_model
        else:
            new_model = MyChronosModel(orig_model.config, model=backbone)
            new_model.load_state_dict(orig_model.state_dict(), strict=True)
            new_model.to(device)
            self.pipeline.model = new_model
            self.model = new_model

        d_model = self.model.model.config.d_model
        vocab_size = int(
            getattr(self.pipeline.model.config, "n_tokens",
                    self.model.model.lm_head.out_features)
        )
        self.head_bi = nn.Linear(d_model, vocab_size)
        self.head_ibg = nn.Linear(d_model, vocab_size)
        # Per-head stream projection so each head sees a task-specific view of
        # the shared decoder hidden state. Initialized to identity so behavior
        # at step 0 matches the prior code path (head(hidden)); the network can
        # then learn to route BI- vs IBG-relevant features independently.
        self.stream_proj_bi = nn.Linear(d_model, d_model, bias=True)
        self.stream_proj_ibg = nn.Linear(d_model, d_model, bias=True)
        with torch.no_grad():
            nn.init.eye_(self.stream_proj_bi.weight)
            nn.init.zeros_(self.stream_proj_bi.bias)
            nn.init.eye_(self.stream_proj_ibg.weight)
            nn.init.zeros_(self.stream_proj_ibg.bias)
        self.ce_loss = nn.CrossEntropyLoss()
        # Use L1 loss instead of MSE for the numeric/bin regression components
        self.l1_loss = nn.L1Loss()

        # Precompute token->value maps and register as buffers
        with torch.no_grad():
            tok_ids = torch.arange(vocab_size, dtype=torch.long, device=self.device)
            vals_bi = self.tokenizer_bi.output_transform(tok_ids, None).to(self.device).float()
            vals_ibg = self.tokenizer_ibg.output_transform(tok_ids, None).to(self.device).float()

        self.register_buffer("token_values_bi", vals_bi)
        self.register_buffer("token_values_ibg", vals_ibg)

        init_log_alpha = math.log(max(float(soft_ce_alpha_init), 1e-6))
        req_alpha = self.auto_tune_soft_ce_sharpness
        if self.soft_ce_alpha_mode == "perbin":
            # Option A: one learnable log-alpha per bin, gathered by ground-truth bin id.
            self.log_soft_ce_alpha_bi = nn.Parameter(
                torch.full((vocab_size,), init_log_alpha, dtype=torch.float32, device=self.device),
                requires_grad=req_alpha,
            )
            self.log_soft_ce_alpha_ibg = nn.Parameter(
                torch.full((vocab_size,), init_log_alpha, dtype=torch.float32, device=self.device),
                requires_grad=req_alpha,
            )
        elif self.soft_ce_alpha_mode == "parametric":
            # Option B: log-alpha = linear(per-bin width/center features). Weight=0,
            # bias=init_log_alpha => initial alpha == soft_ce_alpha_init for every bin.
            self.register_buffer(
                "alpha_feat_bi", self._build_alpha_features(self.token_values_bi).to(self.device)
            )
            self.register_buffer(
                "alpha_feat_ibg", self._build_alpha_features(self.token_values_ibg).to(self.device)
            )
            self.alpha_head_bi = nn.Linear(2, 1).to(self.device)
            self.alpha_head_ibg = nn.Linear(2, 1).to(self.device)
            with torch.no_grad():
                for _head in (self.alpha_head_bi, self.alpha_head_ibg):
                    nn.init.zeros_(_head.weight)
                    nn.init.constant_(_head.bias, init_log_alpha)
            for _p in list(self.alpha_head_bi.parameters()) + list(self.alpha_head_ibg.parameters()):
                _p.requires_grad_(req_alpha)
        else:  # "scalar" (legacy): one learnable log-alpha per stream
            self.log_soft_ce_alpha_bi = nn.Parameter(
                torch.tensor(init_log_alpha, dtype=torch.float32, device=self.device),
                requires_grad=req_alpha,
            )
            self.log_soft_ce_alpha_ibg = nn.Parameter(
                torch.tensor(init_log_alpha, dtype=torch.float32, device=self.device),
                requires_grad=req_alpha,
            )

    @staticmethod
    def _build_alpha_features(centers: torch.Tensor) -> torch.Tensor:
        """Per-bin features for parametric soft-CE alpha: [z(log spacing), z(center)] -> [V, 2]."""
        c = centers.detach().to(torch.float32)
        spacing = torch.ones_like(c)
        if c.numel() >= 3:
            spacing[1:-1] = 0.5 * (c[2:] - c[:-2])
            spacing[0] = c[1] - c[0]
            spacing[-1] = c[-1] - c[-2]
        spacing = spacing.abs().clamp_min(1e-9)
        log_w = torch.log(spacing)

        def _z(x: torch.Tensor) -> torch.Tensor:
            return (x - x.mean()) / (x.std() + 1e-6)

        return torch.stack([_z(log_w), _z(c)], dim=-1)

    def _soft_ce_alpha(self, stream: str) -> torch.Tensor:
        """Soft-CE sharpness alpha. Returns 0-dim (scalar mode) or [V] (perbin/parametric)."""
        if self.soft_ce_alpha_mode == "parametric":
            feat = getattr(self, f"alpha_feat_{stream}")
            head = getattr(self, f"alpha_head_{stream}")
            log_alpha = head(feat).squeeze(-1)
        else:
            log_alpha = getattr(self, f"log_soft_ce_alpha_{stream}")
        return torch.exp(log_alpha).clamp(max=10000.0)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        # 1) Save weights (Chronos + heads + buffers)
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

        # 2) Save a small config so we can reconstruct the wrapper
        # If boundaries are numpy / torch, convert to plain lists
        def to_list(x):
            import numpy as np
            if isinstance(x, torch.Tensor):
                return x.cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return list(x)

        cfg = {
            "pretrained_model_or_path": self._pretrained_model_or_path,
            "boundaries_bi": to_list(self._boundaries_bi),
            "boundaries_ibg": to_list(self._boundaries_ibg),
            "num_tokens": self._num_tokens,
            "mse_mode": self.mse_mode,
            "center_clip": self.center_clip,
            "d_model": self.model.model.config.d_model,
            "use_cross_attn": self.use_cross_attn,
            "soft_ce_alpha_init": float(self._soft_ce_alpha("bi").mean().detach().cpu().item()),
            "soft_ce_alpha_mode": self.soft_ce_alpha_mode,
            "soft_ce_denom_floor_bi": self.soft_ce_denom_floor_bi,
            "soft_ce_denom_floor_ibg": self.soft_ce_denom_floor_ibg,
            "auto_tune_soft_ce_sharpness": self.auto_tune_soft_ce_sharpness,
            "use_ce_loss": self.use_ce_loss,
            "loss_weight_bi": self.loss_weight_bi,
            "loss_weight_ibg": self.loss_weight_ibg,
            "loss_weight_local_bi": self.loss_weight_local_bi,
            "loss_weight_local_ibg": self.loss_weight_local_ibg,
            "soft_ce_sigma_local": self.soft_ce_sigma_local,
            "local_loss_min_context": self.local_loss_min_context,
            "ibg_integer_bins": int(self.ibg_integer_bins),
            "num_local_bins": int(self.num_local_bins),
        }

        cfg_path = os.path.join(save_directory, "netburst_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str, device: torch.device):
        # 1) Load wrapper config
        cfg_path = os.path.join(load_directory, "netburst_config.json")
        if not os.path.exists(cfg_path):
            # Backward-compatibility with older checkpoints saved as twinhead_config.json
            legacy_cfg_path = os.path.join(load_directory, "twinhead_config.json")
            if os.path.exists(legacy_cfg_path):
                cfg_path = legacy_cfg_path
            else:
                raise FileNotFoundError(
                    f"Missing config in {load_directory}. Expected 'netburst_config.json' "
                    f"or legacy 'twinhead_config.json'."
                )
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        boundaries_bi = np.array(cfg["boundaries_bi"], dtype=np.float32)
        boundaries_ibg = np.array(cfg["boundaries_ibg"], dtype=np.float32)
        pretrained_model_or_path = cfg["pretrained_model_or_path"]
        num_tokens = cfg.get("num_tokens", None)
        mse_mode = cfg.get("mse_mode", "bin")
        d_model = cfg.get("d_model", 512)
        center_clip = cfg.get("center_clip", 1e6)
        use_cross_attn = cfg.get("use_cross_attn", False)
        soft_ce_alpha_init = cfg.get("soft_ce_alpha_init", 1.0)
        soft_ce_denom_floor_bi = cfg.get("soft_ce_denom_floor_bi", 1e-3)
        soft_ce_denom_floor_ibg = cfg.get("soft_ce_denom_floor_ibg", 1.0)
        auto_tune_soft_ce_sharpness = cfg.get("auto_tune_soft_ce_sharpness", True)
        use_ce_loss = cfg.get("use_ce_loss", False)
        loss_weight_bi = cfg.get("loss_weight_bi", 1.0)
        loss_weight_ibg = cfg.get("loss_weight_ibg", 1.0)
        loss_weight_local_bi = cfg.get("loss_weight_local_bi", 0.0)
        loss_weight_local_ibg = cfg.get("loss_weight_local_ibg", 0.0)
        soft_ce_sigma_local = cfg.get("soft_ce_sigma_local", 1.5)
        local_loss_min_context = cfg.get("local_loss_min_context", 16)
        ibg_integer_bins = int(cfg.get("ibg_integer_bins", 0))
        num_local_bins = int(cfg.get("num_local_bins", 0))
        soft_ce_alpha_mode = cfg.get("soft_ce_alpha_mode", "scalar")

        # 2) Recreate the module with *exactly* the same ctor args
        model = cls(
            boundaries_bi=boundaries_bi,
            boundaries_ibg=boundaries_ibg,
            pretrained_model_or_path=pretrained_model_or_path,
            device=device,
            num_tokens=num_tokens,
            mse_mode=mse_mode,
            center_clip=center_clip,
            use_cross_attn=use_cross_attn,
            soft_ce_alpha_init=soft_ce_alpha_init,
            soft_ce_denom_floor_bi=soft_ce_denom_floor_bi,
            soft_ce_denom_floor_ibg=soft_ce_denom_floor_ibg,
            auto_tune_soft_ce_sharpness=auto_tune_soft_ce_sharpness,
            use_ce_loss=use_ce_loss,
            loss_weight_bi=loss_weight_bi,
            loss_weight_ibg=loss_weight_ibg,
            loss_weight_local_bi=loss_weight_local_bi,
            loss_weight_local_ibg=loss_weight_local_ibg,
            soft_ce_sigma_local=soft_ce_sigma_local,
            local_loss_min_context=local_loss_min_context,
            ibg_integer_bins=ibg_integer_bins,
            num_local_bins=num_local_bins,
            soft_ce_alpha_mode=soft_ce_alpha_mode,
        )

        # 3) Load weights
        weights_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location=device)
        incompat = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(incompat, "missing_keys", []))
        unexpected = list(getattr(incompat, "unexpected_keys", []))
        if missing:
            print(
                f"[WARN] from_pretrained: missing {len(missing)} key(s) when loading {weights_path}. "
                f"First keys: {missing[:20]}"
            )
        if unexpected:
            print(
                f"[WARN] from_pretrained: unexpected {len(unexpected)} key(s) when loading {weights_path}. "
                f"First keys: {unexpected[:20]}"
            )
        if hasattr(model, "model") and hasattr(model.model, "_tie_local_head_weights"):
            model.model._tie_local_head_weights()
        model.to(device)
        model.eval()
        return model


    def _prep_context(self, batch_list):
        # batch_list is a python list of tuples (bi_t, ibg_t)
        bi_series  = [bi for (bi, _ibg) in batch_list]
        ibg_series = [ibg for (_bi, ibg) in batch_list]
        return bi_series, ibg_series

    def _tokenize_dual_streams(self, ctx_bi, ctx_ibg):
        """Tokenize BI/IBG global ids and optional per-series local ids."""
        ids_bi, mask_bi, _ = self.tokenizer_bi.context_input_transform(ctx_bi)
        ids_ibg, mask_ibg, _ = self.tokenizer_ibg.context_input_transform(ctx_ibg)
        local_bi = local_ibg = None
        local_mask_bi = local_mask_ibg = None
        local_label_bi = local_label_ibg = None
        local_label_mask_bi = local_label_mask_ibg = None
        if self.num_local_bins > 0:
            local_bi, local_mask_bi, _, edges_bi = self.local_binner_bi.context_input_transform(
                ctx_bi, return_edges=True
            )
            local_ibg, local_mask_ibg, _, edges_ibg = self.local_binner_ibg.context_input_transform(
                ctx_ibg, return_edges=True
            )
            local_label_bi, local_label_mask_bi = self.local_binner_bi.label_input_transform(
                ctx_bi, edges_bi, append_eos=self.local_binner_bi.config.use_eos_token
            )
            local_label_ibg, local_label_mask_ibg = self.local_binner_ibg.label_input_transform(
                ctx_ibg, edges_ibg, append_eos=self.local_binner_ibg.config.use_eos_token
            )
            assert local_bi.shape == ids_bi.shape and local_ibg.shape == ids_ibg.shape
            assert local_label_bi.shape == local_bi.shape and local_label_ibg.shape == local_ibg.shape
        return (
            ids_bi,
            mask_bi,
            ids_ibg,
            mask_ibg,
            local_bi,
            local_ibg,
            local_mask_bi,
            local_mask_ibg,
            local_label_bi,
            local_label_ibg,
            local_label_mask_bi,
            local_label_mask_ibg,
        )

    def forward(self, batch_list):
        """
        batch_list: list of (bi_tensor, ibg_tensor), each 1D float tensors on CPU
        Returns:
          logits_bi:  [B, T-1, V]
          logits_ibg: [B, T-1, V]
          targets_bi:  [B, T-1]
          targets_ibg: [B, T-1]
          valid_bi:    [B, T-1] bool
          valid_ibg:   [B, T-1] bool
        """
        # Prepare separate contexts
        bi_series, ibg_series = self._prep_context(batch_list)

        ctx_bi = self.pipeline._prepare_and_validate_context(context=bi_series)
        ctx_ibg = self.pipeline._prepare_and_validate_context(context=ibg_series)
        (
            ids_bi,
            mask_bi,
            ids_ibg,
            mask_ibg,
            local_bi,
            local_ibg,
            local_mask_bi,
            local_mask_ibg,
            local_label_ids_bi,
            local_label_ids_ibg,
            local_label_mask_bi,
            local_label_mask_ibg,
        ) = self._tokenize_dual_streams(ctx_bi, ctx_ibg)

        dummy = torch.empty((ids_bi.size(0), 1), dtype=torch.long, device=self.device)
        comb_mask = (mask_bi & mask_ibg).to(self.device)

        fwd_kw = dict(
            input_ids1=ids_bi.to(self.device),
            input_ids2=ids_ibg.to(self.device),
            attention_mask=comb_mask,
            decoder_input_ids=dummy,
            type_id=0,
            cross_attend=self.use_cross_attn,
        )
        if local_bi is not None:
            fwd_kw["local_ids1"] = local_bi.to(self.device)
            fwd_kw["local_ids2"] = local_ibg.to(self.device)

        logits_bi_unused, hidden = self.model.forward_with_embeddings(**fwd_kw)
        # shapes: logits_bi_unused: [B,1,T,V], hidden: [B,1,T,D]
        hidden = hidden.squeeze(1)  # [B,T,D]

        # Per-head projection of the shared decoder state so each head gets a
        # task-specific representation (see __init__; identity at step 0).
        h_bi = self.stream_proj_bi(hidden)
        h_ibg = self.stream_proj_ibg(hidden)

        # Next-token shift
        logits_bi = self.head_bi(h_bi)[:, :-1, :]   # predict t+1 using state at t
        logits_ibg = self.head_ibg(h_ibg)[:, :-1, :]

        logits_local_bi = logits_local_ibg = None
        local_targets_bi = local_targets_ibg = None
        local_valid_bi = local_valid_ibg = None
        if self.num_local_bins > 0 and local_label_ids_bi is not None:
            # Expose optional local-head logits for training-only auxiliary loss.
            self.model._tie_local_head_weights()
            local_heads_bi = self.model.heads(h_bi)
            local_heads_ibg = self.model.heads(h_ibg)
            logits_local_bi = local_heads_bi["logits_local_bi"][:, :-1, :]
            logits_local_ibg = local_heads_ibg["logits_local_ibg"][:, :-1, :]
            local_targets_bi = local_label_ids_bi[:, 1:].to(self.device)
            local_targets_ibg = local_label_ids_ibg[:, 1:].to(self.device)
            local_valid_bi = local_label_mask_bi[:, 1:].to(self.device)
            local_valid_ibg = local_label_mask_ibg[:, 1:].to(self.device)

        targets_bi = ids_bi[:, 1:].to(self.device)
        targets_ibg = ids_ibg[:, 1:].to(self.device)

        valid_bi  = mask_bi[:, 1:].to(self.device)
        valid_ibg = mask_ibg[:, 1:].to(self.device)

        # Return context token ids for Fano-factor logging (input = ids[:, :-1])
        ids_bi = ids_bi.to(self.device)
        ids_ibg = ids_ibg.to(self.device)
        return (
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
        )

    def compute_loss(self, logits_bi, logits_ibg, targets_bi, targets_ibg, valid_bi, valid_ibg):
        # ----- CE branch: standard cross-entropy (hard targets) or soft CE (value-based weights) -----
        bi_log = logits_bi[valid_bi]        # [N_valid_bi, V]
        bi_tgt = targets_bi[valid_bi]       # [N_valid_bi]
        ibg_log = logits_ibg[valid_ibg]     # [N_valid_ibg, V]
        ibg_tgt = targets_ibg[valid_ibg]    # [N_valid_ibg]

        if self.use_ce_loss:
            # Standard cross-entropy with one-hot (hard) targets
            ce_bi = self.ce_loss(bi_log, bi_tgt)
            ce_ibg = self.ce_loss(ibg_log, ibg_tgt)
        else:
            # Soft cross-entropy over token IDs with value-based weights
            gt_vals_bi = self.token_values_bi[bi_tgt]                 # [N_valid_bi]
            alpha_vec_bi = self._soft_ce_alpha("bi")
            alpha_bi = alpha_vec_bi[bi_tgt].unsqueeze(1) if alpha_vec_bi.dim() > 0 else alpha_vec_bi
            denom_bi = torch.clamp(torch.abs(gt_vals_bi), min=self.soft_ce_denom_floor_bi)
            bin_vals_bi = self.token_values_bi.unsqueeze(0).expand(gt_vals_bi.size(0), -1)  # [N_valid_bi, V]
            rel_err_bi = torch.abs(bin_vals_bi - gt_vals_bi.unsqueeze(1)) / denom_bi.unsqueeze(1)
            weights_bi = torch.exp(-alpha_bi * rel_err_bi)
            soft_targets_bi = weights_bi / weights_bi.sum(dim=1, keepdim=True)
            log_probs_bi = torch.log_softmax(bi_log, dim=-1)
            ce_bi = -(soft_targets_bi * log_probs_bi).sum(dim=-1).mean()

            gt_vals_ibg = self.token_values_ibg[ibg_tgt]               # [N_valid_ibg]
            alpha_vec_ibg = self._soft_ce_alpha("ibg")
            alpha_ibg = alpha_vec_ibg[ibg_tgt].unsqueeze(1) if alpha_vec_ibg.dim() > 0 else alpha_vec_ibg
            denom_ibg = torch.clamp(torch.abs(gt_vals_ibg), min=self.soft_ce_denom_floor_ibg)
            bin_vals_ibg = self.token_values_ibg.unsqueeze(0).expand(gt_vals_ibg.size(0), -1)  # [N_valid_ibg, V]
            rel_err_ibg = torch.abs(bin_vals_ibg - gt_vals_ibg.unsqueeze(1)) / denom_ibg.unsqueeze(1)
            weights_ibg = torch.exp(-alpha_ibg * rel_err_ibg)
            soft_targets_ibg = weights_ibg / weights_ibg.sum(dim=1, keepdim=True)
            log_probs_ibg = torch.log_softmax(ibg_log, dim=-1)
            ce_ibg = -(soft_targets_ibg * log_probs_ibg).sum(dim=-1).mean()

    # ----- L1 branch (was MSE) -----
        probs_bi  = torch.softmax(logits_bi, dim=-1)
        probs_ibg = torch.softmax(logits_ibg, dim=-1)

        if self.mse_mode == "centers":
            # Predicted numeric value via weighted average of token centers
            # pred_val_bi  = torch.einsum('btv,v->bt', probs_bi,  self.token_values_bi)
            # pred_val_ibg = torch.einsum('btv,v->bt', probs_ibg, self.token_values_ibg)
            pred_val_bi  = self.token_values_bi[probs_bi.argmax(dim=-1)]
            pred_val_ibg = self.token_values_ibg[probs_ibg.argmax(dim=-1)]

            # Target numeric value is the center value corresponding to the target token id
            true_val_bi  = self.token_values_bi[targets_bi]
            true_val_ibg = self.token_values_ibg[targets_ibg]

            # Scale down center values to stabilize loss
            sf = 1
            pred_val_bi  = pred_val_bi * sf
            pred_val_ibg = pred_val_ibg
            true_val_bi  = true_val_bi * sf
            true_val_ibg = true_val_ibg

            # Clamp both predictions and targets to avoid exploding MSE due to extreme centers
            clip = self.center_clip
            pred_val_bi  = torch.clamp(pred_val_bi,  min=-clip, max=clip)
            pred_val_ibg = torch.clamp(pred_val_ibg, min=-clip, max=clip)
            true_val_bi  = torch.clamp(true_val_bi,  min=-clip, max=clip)
            true_val_ibg = torch.clamp(true_val_ibg, min=-clip, max=clip)

            l1_bi  = self.l1_loss(pred_val_bi[valid_bi],  true_val_bi[valid_bi])
            l1_ibg = self.l1_loss(pred_val_ibg[valid_ibg], true_val_ibg[valid_ibg])
        else:
            # Default: L1 on bin indices via expected bin id
            vocab_size = logits_bi.size(-1)
            bin_ids = torch.arange(vocab_size, dtype=torch.float32, device=self.device)
            pred_bin_bi  = torch.einsum('btv,v->bt', probs_bi,  bin_ids)
            pred_bin_ibg = torch.einsum('btv,v->bt', probs_ibg, bin_ids)
            true_bin_bi  = targets_bi.float()
            true_bin_ibg = targets_ibg.float()
            l1_bi  = self.l1_loss(pred_bin_bi[valid_bi],  true_bin_bi[valid_bi])
            l1_ibg = self.l1_loss(pred_bin_ibg[valid_ibg], true_bin_ibg[valid_ibg])

        # ----- Weight CE losses based on corresponding L1 losses -----
        # If l1 is large, take log(l1) and clamp weights to the range [1, 5]; else weight = 1
        one_const = torch.tensor(1.0, device=self.device)
        # Handle scalar tensors robustly with torch.where to avoid logging <=1
        # w_bi = torch.where(
        #     l1_bi > 1.0,
        #     torch.clamp(l1_bi, min=1.0, max=5.0),
        #     one_const,
        # )
        # w_ibg = torch.where(
        #     l1_ibg > 1.0,
        #     torch.clamp(l1_ibg, min=1.0, max=5.0),
        #     one_const,
        # )

        w_bi = float(getattr(self, "loss_weight_bi", 1.0))
        w_ibg = float(getattr(self, "loss_weight_ibg", 1.0))
        total = w_bi * ce_bi + w_ibg * ce_ibg  # + l1_bi*1e-4 +l1_ibg*1e-4
        return total, ce_bi.detach(), ce_ibg.detach(), l1_bi.detach(), l1_ibg.detach()


def load_twin_head_from_dir(
    model_dir: str,
    device: torch.device,
    *,
    print_head_norms: bool = True,
) -> TwinHeadChronosPredictor:
    """Load NetBurst checkpoint from a directory (same as ``TwinHeadChronosPredictor.from_pretrained``)."""
    predictor = TwinHeadChronosPredictor.from_pretrained(model_dir, device)
    if print_head_norms:
        try:
            with torch.no_grad():
                hn_bi = float(predictor.head_bi.weight.norm().item())
                hn_ibg = float(predictor.head_ibg.weight.norm().item())
            print(f"Head norms — BI: {hn_bi:.4f}, IBG: {hn_ibg:.4f}")
        except Exception:
            pass
    return predictor


def load_chronos_bin_predictor_from_checkpoint(
    checkpoint_dir: str,
    base_pretrained_id_or_path: str,
    device: torch.device,
    *,
    boundaries_filename: str = "boundaries.pkl",
    weights_filename: str = "chronos_best.pt",
    prefixes_to_strip: Tuple[str, ...] = ("model.model.", "model."),
) -> ChronosBinPredictor:
    """Load ``ChronosBinPredictor`` with quantile boundaries and fine-tuned weights."""
    from netburst.utils import strip_prefix_if_present

    with open(os.path.join(checkpoint_dir, boundaries_filename), "rb") as f:
        boundaries = pickle.load(f)
    predictor = ChronosBinPredictor(boundaries, base_pretrained_id_or_path, device).to(device)
    state = torch.load(os.path.join(checkpoint_dir, weights_filename), map_location=device)
    state = strip_prefix_if_present(state, list(prefixes_to_strip))
    predictor.model.model.load_state_dict(state, strict=True)
    predictor.model.eval()
    return predictor
