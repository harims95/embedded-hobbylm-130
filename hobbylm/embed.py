"""Text embeddings from the MoE LM: padding-aware attention, pooling, InfoNCE.

Nothing in model.py changes. Embedder walks the same blocks MoETransformer.forward does,
but stops at final_norm (no 50k-vocab head) and pools the hidden states into one vector.

Two attention modes, chosen per call:
  causal (default)   : the pretrained behaviour. Pair with pooling="last" (text must end in EOT).
  bidirectional=True : every token sees the whole text. Pair with pooling="mean". Use this on a
                       checkpoint adapted with the masked-token objective (train.py --set diffusion=true).

Things the caller owns:
  - Freeze the aux-free expert bias first (model.set_bias_update_rate(0.0)). MoE._route updates
    that buffer on EVERY training forward, and GradCache runs two forwards per batch.
  - Padding tokens are still routed, so they count in the MoE balance loss returned here.
    Keep padding low (similar lengths per batch) when that loss is weighted in.
  - Use one precision for training, indexing and querying: bf16 vs fp32 changes which experts fire.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import MoETransformer

EOT = 50256                 # <|endoftext|>; appended to every text, and used as the pad id


def build_padding_bias(lengths: Tensor, S: int, dtype: torch.dtype, causal: bool) -> Tensor:
    """Additive SDPA mask (B, 1, S, S): 0 where the key is a real token (and, if causal,
    not in the future), dtype-min elsewhere. Rows are right-padded; lengths (B,) >= 1.
    Padding queries still see the real keys, so no row is fully masked (no NaNs)."""
    B = lengths.shape[0]
    pos = torch.arange(S, device=lengths.device)
    key_ok = pos.unsqueeze(0) < lengths.unsqueeze(1)                 # (B, S)
    allowed = key_ok.unsqueeze(1).expand(B, S, S)                    # (B, S_query, S_key)
    if causal:
        allowed = allowed & torch.tril(torch.ones(S, S, dtype=torch.bool, device=lengths.device))
    bias = torch.zeros(B, 1, S, S, dtype=dtype, device=lengths.device)
    bias.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)
    return bias


def pool(x: Tensor, lengths: Tensor, pooling: str) -> Tensor:
    """x: (B, S, d) hidden states -> (B, d). 'last' = final real token; 'mean' = mean over real tokens."""
    B, S, _ = x.shape
    if pooling == "last":
        return x[torch.arange(B, device=x.device), lengths - 1]
    if pooling == "mean":
        real = (torch.arange(S, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        return (x * real).sum(dim=1) / lengths.unsqueeze(1).to(x.dtype)
    raise ValueError(f"unknown pooling {pooling!r}; use 'last' or 'mean'")


class Embedder(nn.Module):
    """Wraps a MoETransformer. forward(idx, lengths) -> (unit-norm embeddings (B, d) fp32, MoE aux loss)."""

    def __init__(self, lm: MoETransformer, pooling: str = "last", bidirectional: bool = False):
        super().__init__()
        self.lm = lm
        self.pooling = pooling
        self.bidirectional = bidirectional

    def forward(self, idx: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        lm = self.lm
        x = lm.embed(idx)
        if lm.cfg.scale_embeddings:
            x = x * (lm.cfg.d_model ** 0.5)
        S = x.shape[1]
        cos, sin = lm.rope(S, x.device, x.dtype)
        # SDPA needs the mask dtype to match the query dtype, which is bf16 only under CUDA autocast
        # (same rule as build_block_diagonal_bias's caller in training/train.py).
        bf16 = x.is_cuda and torch.is_autocast_enabled()
        bias = build_padding_bias(lengths, S, torch.bfloat16 if bf16 else x.dtype,
                                  causal=not self.bidirectional)
        aux_sum = x.new_zeros(())
        for blk in lm.blocks:
            x, aux = blk(x, cos, sin, attn_bias=bias)
            aux_sum = aux_sum + aux
        x = lm.final_norm(x).float()
        return F.normalize(pool(x, lengths, self.pooling), dim=-1), aux_sum


def info_nce(q: Tensor, cands: Tensor, labels: Tensor, temperature: float,
             fn_margin: float | None = None) -> tuple[Tensor, Tensor]:
    """q: (Q, d) queries, cands: (C, d) candidates (positives + negatives), both unit-norm.
    labels: (Q,) index into cands of each query's positive. Every other candidate is a negative.
    fn_margin: if set, drop any negative scoring above (positive score + fn_margin) -- likely an
    unlabeled true match (Qwen3-Embedding uses 0.1). Returns (mean loss, top-1 accuracy)."""
    sims = q @ cands.T
    if fn_margin is not None:
        with torch.no_grad():
            pos = sims.gather(1, labels.unsqueeze(1))                     # (Q, 1)
            drop = sims > pos + fn_margin
            drop.scatter_(1, labels.unsqueeze(1), False)                 # never drop the positive
        sims = sims.masked_fill(drop, float("-inf"))
    logits = sims / temperature
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, acc.detach()


_PREFIX_LEN: dict[str, int] = {}


def text_ids(tokenize, prefix: str, text: str, width: int) -> list[int]:
    """prefix + text, truncated to width - 1 tokens, then EOT (last-token pooling reads it).
    Refuses a prefix (e.g. a long instruction) that leaves fewer than 8 tokens for the text,
    which would silently make every example look the same."""
    if prefix not in _PREFIX_LEN:
        _PREFIX_LEN[prefix] = len(tokenize(prefix))
    if _PREFIX_LEN[prefix] > width - 9:
        raise ValueError(f"prefix {prefix!r} is {_PREFIX_LEN[prefix]} tokens; width {width} leaves no room "
                         f"for the text. Raise the width (--q_len / --p_len).")
    return tokenize(prefix + text)[:width - 1] + [EOT]


def pad_batch(token_lists: list[list[int]], pad_to: int, device=None) -> tuple[Tensor, Tensor]:
    """Right-pad token id lists to a fixed width (a fixed width keeps shapes static for compile)."""
    B = len(token_lists)
    idx = torch.full((B, pad_to), EOT, dtype=torch.long)
    lengths = torch.empty(B, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        assert 1 <= len(toks) <= pad_to, f"row {i}: length {len(toks)} not in [1, {pad_to}]"
        idx[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
        lengths[i] = len(toks)
    if device is not None:
        idx, lengths = idx.to(device), lengths.to(device)
    return idx, lengths
