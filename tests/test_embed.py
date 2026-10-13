"""CPU checks for the embedding path (hobbylm/embed.py). Runs in well under a minute.

    python tests/test_embed.py

1. Padding invariance: a text's embedding must not change when the row is padded wider.
2. Attention modes differ: bidirectional must actually see later tokens; causal must not.
3. GradCache: the two-pass gradient must equal the plain single-pass gradient.
4. Overfit one batch: InfoNCE must drive to ~0 on a fixed batch.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hobbylm.config import ModelConfig
from hobbylm.embed import EOT, Embedder, info_nce, pad_batch
from hobbylm.model import MoETransformer

torch.manual_seed(0)
VOCAB = 256
cfg = ModelConfig(
    vocab_size=VOCAB, d_model=96, n_layers=3, n_dense_layers=1,
    n_q_heads=4, n_kv_heads=2, head_dim=24,
    dense_ffn=192, expert_ffn=48, n_experts=8, top_k=2, n_shared=1,
    expert_backend="bmm",
)
PAD = EOT % VOCAB       # pad_batch pads with EOT; remap into the toy vocab


def make(pooling, bidirectional):
    torch.manual_seed(0)
    lm = MoETransformer(cfg)
    lm.set_bias_update_rate(0.0)
    return Embedder(lm, pooling=pooling, bidirectional=bidirectional)


def batch(rows, width):
    idx, lengths = pad_batch(rows, width)
    return idx.clamp_max(VOCAB - 1), lengths


def rand_rows(n, lo, hi):
    return [torch.randint(0, 200, (int(torch.randint(lo, hi + 1, (1,))),)).tolist() for _ in range(n)]


# ---- 1. padding invariance (both modes) ----
rows = rand_rows(5, 3, 12)
for pooling, bi in (("last", False), ("mean", False), ("mean", True), ("last", True)):
    emb = make(pooling, bi).eval()
    with torch.no_grad():
        a, _ = emb(*batch(rows, 12))
        b, _ = emb(*batch(rows, 40))
    diff = (a - b).abs().max().item()
    print(f"padding invariance  pooling={pooling:4s} bidirectional={bi!s:5s}  max|diff|={diff:.2e}")
    assert diff < 1e-5, "embedding changed with padding width"

# ---- 2. the two attention modes really differ ----
x1 = [5, 6, 7, 8, 9, 10]
x2 = [5, 6, 7, 8, 9, 11]            # differs only in the LAST token
for bi, expect_same in ((False, True), (True, False)):
    lm_emb = make("mean", bi).eval()
    with torch.no_grad():
        h = []
        for x in (x1, x2):
            idx, lengths = batch([x], 6)
            # hidden state of the FIRST token: can it see the last one?
            xx = lm_emb.lm.embed(idx)
            from hobbylm.embed import build_padding_bias
            cos, sin = lm_emb.lm.rope(6, xx.device, xx.dtype)
            bias = build_padding_bias(lengths, 6, xx.dtype, causal=not bi)
            for blk in lm_emb.lm.blocks:
                xx, _ = blk(xx, cos, sin, attn_bias=bias)
            h.append(xx[0, 0])
    same = torch.allclose(h[0], h[1], atol=1e-6)
    print(f"first token sees last token: bidirectional={bi!s:5s} -> {'no' if same else 'yes'}")
    assert same == expect_same

# ---- 3. GradCache two-pass gradient == direct gradient ----
B, K, CHUNK, TEMP = 6, 1, 4, 0.05
q_rows, p_rows = rand_rows(B, 3, 8), rand_rows(B * (1 + K), 5, 16)
labels = torch.arange(B)


def direct_grads(emb):
    emb.zero_grad()
    q, aq = emb(*batch(q_rows, 8))
    p, ap = emb(*batch(p_rows, 16))
    loss, _ = info_nce(q, p, labels, TEMP)
    loss.backward()
    return loss.item(), {n: p_.grad.clone() for n, p_ in emb.named_parameters() if p_.grad is not None}


def gradcache_grads(emb):
    emb.zero_grad()
    chunks = [batch(q_rows[i:i + CHUNK], 8) for i in range(0, B, CHUNK)]
    chunks += [batch(p_rows[i:i + CHUNK], 16) for i in range(0, len(p_rows), CHUNK)]
    with torch.no_grad():
        reps = torch.cat([emb(i, l)[0] for i, l in chunks])
    reps.requires_grad_(True)
    loss, _ = info_nce(reps[:B], reps[B:], labels, TEMP)
    (g,) = torch.autograd.grad(loss, reps)
    row = 0
    for i, l in chunks:
        r, _ = emb(i, l)
        (r * g[row:row + r.shape[0]]).sum().backward()
        row += r.shape[0]
    return loss.item(), {n: p_.grad.clone() for n, p_ in emb.named_parameters() if p_.grad is not None}


emb = make("mean", True).train()
l1, g1 = direct_grads(emb)
l2, g2 = gradcache_grads(emb)
worst = max((g1[n] - g2[n]).abs().max().item() for n in g1)
scale = max(g1[n].abs().max().item() for n in g1)
print(f"gradcache: loss direct={l1:.6f} cached={l2:.6f}  max|grad diff|={worst:.2e} (grad scale {scale:.2e})")
assert abs(l1 - l2) < 1e-5 and set(g1) == set(g2) and worst < 1e-4 * max(scale, 1.0)

# ---- 4. overfit one batch ----
emb = make("last", False).train()
opt = torch.optim.AdamW(emb.parameters(), lr=1e-3, weight_decay=0.0)
qb, pb = batch([r + [PAD] for r in q_rows], 9), batch([r + [PAD] for r in p_rows], 17)
for step in range(300):
    q, aq = emb(*qb)
    p, ap = emb(*pb)
    loss, acc = info_nce(q, p, labels, TEMP)
    opt.zero_grad(); (loss + aq + ap).backward(); opt.step()
    if step == 0:
        first = loss.item()
print(f"overfit one batch: loss {first:.4f} -> {loss.item():.4f}, acc {acc.item():.2f}")
assert loss.item() < 0.05 and acc.item() == 1.0

print("ALL EMBED CHECKS PASSED")
