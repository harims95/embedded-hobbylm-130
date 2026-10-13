"""CPU checks for the ablation kit: false-negative masking, distillation GradCache, teacher
pooling/merge, eval metrics + encoder, data prep. No network: Hugging Face loaders are stubbed.

    python tests/test_ablation_kit.py
"""
import json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hobbylm.config import ModelConfig
from hobbylm.embed import Embedder, info_nce, pad_batch
from hobbylm.model import MoETransformer

torch.manual_seed(0)
TMP = tempfile.mkdtemp()
cfg = ModelConfig(vocab_size=50304, d_model=96, n_layers=3, n_dense_layers=1, n_q_heads=4, n_kv_heads=2,
                  head_dim=24, dense_ffn=192, expert_ffn=48, n_experts=8, top_k=2, n_shared=0,
                  expert_backend="bmm")


def tiny_embedder(pooling="mean", bi=True):
    torch.manual_seed(0)
    lm = MoETransformer(cfg); lm.set_bias_update_rate(0.0)
    return Embedder(lm, pooling, bi)


def rows(n, lo, hi):
    return [torch.randint(0, 200, (int(torch.randint(lo, hi + 1, (1,))),)).tolist() for _ in range(n)]


# ---- 1. false-negative masking ----
q = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
c = F.normalize(torch.tensor([[0.6, 0.8], [0.0, 1.0], [1.0, 0.05]]), dim=-1)   # cand 2 ~= query 0
labels = torch.tensor([0, 1])
l_plain, _ = info_nce(q, c, labels, 0.05)
l_mask, _ = info_nce(q, c, labels, 0.05, fn_margin=0.1)
sims = q @ c.T
dropped = (sims > sims.gather(1, labels[:, None]) + 0.1)
print(f"fn masking: dropped {dropped.sum().item()} candidate(s); loss {l_plain:.3f} -> {l_mask:.3f}")
assert dropped[0, 2] and not dropped[0, 0] and not dropped[1, 1] and l_mask < l_plain

# ---- 2. distillation: GradCache gradient == direct gradient (model AND projection) ----
from training.train_distill import distill_losses
B, D, CH = 8, 40, 3
texts = rows(B, 3, 12)
teacher = F.normalize(torch.randn(B, D), dim=-1)


def grads_direct():
    emb = tiny_embedder(); proj = nn.Linear(96, D, bias=False); torch.manual_seed(1); nn.init.normal_(proj.weight, std=0.1)
    s, _ = emb(*pad_batch(texts, 12))
    cl, sl = distill_losses(s, proj, teacher); (cl + 2.0 * sl).backward()
    return {**{n: p.grad.clone() for n, p in emb.named_parameters() if p.grad is not None}, "proj": proj.weight.grad.clone()}


def grads_cached():
    emb = tiny_embedder(); proj = nn.Linear(96, D, bias=False); torch.manual_seed(1); nn.init.normal_(proj.weight, std=0.1)
    chunks = [pad_batch(texts[i:i + CH], 12) for i in range(0, B, CH)]
    with torch.no_grad():
        reps = torch.cat([emb(i, l)[0] for i, l in chunks])
    reps.requires_grad_(True)
    cl, sl = distill_losses(reps, proj, teacher); (cl + 2.0 * sl).backward()
    row = 0
    for i, l in chunks:
        r, _ = emb(i, l); (r * reps.grad[row:row + r.shape[0]]).sum().backward(); row += r.shape[0]
    return {**{n: p.grad.clone() for n, p in emb.named_parameters() if p.grad is not None}, "proj": proj.weight.grad.clone()}


g1, g2 = grads_direct(), grads_cached()
worst = max((g1[n] - g2[n]).abs().max().item() for n in g1)
print(f"distill gradcache: {len(g1)} tensors, max|grad diff|={worst:.2e}")
assert set(g1) == set(g2) and worst < 1e-5

# ---- 3. teacher: last-token pooling under left padding, Matryoshka truncation, shard merge ----
from scripts.teacher_embed import encode as teacher_encode, last_token_pool, merge as teacher_merge
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
words = "[PAD] [UNK] the cat sat on mat a dog ran fast Instruct: Query: find pets <eos>".split()
tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]", padding_side="left")
qm = Qwen3Model(Qwen3Config(vocab_size=len(words), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16)).eval()
sents = ["the cat sat on the mat", "a dog ran", "Instruct: find pets Query: the cat"]
batch = teacher_encode(qm, tok, sents, 32, None, torch.device("cpu"))
alone = torch.cat([teacher_encode(qm, tok, [s], 32, None, torch.device("cpu")) for s in sents])
mrl = teacher_encode(qm, tok, sents, 32, 16, torch.device("cpu"))
print(f"teacher: batched vs single max|diff|={(batch - alone).abs().max().item():.2e}, "
      f"truncated dim={mrl.shape[1]} norms={mrl.norm(dim=-1).min().item():.4f}")
assert (batch - alone).abs().max().item() < 1e-4 and mrl.shape[1] == 16 and torch.allclose(mrl.norm(dim=-1), torch.ones(3), atol=1e-5)
full = np.random.randn(7, 4).astype(np.float16)


class A: pass


a = A(); a.out = os.path.join(TMP, "t.npy"); a.merge = 3
for s in range(3):
    np.save(f"{a.out}.part{s}of3.npy", full[s::3])
teacher_merge(a)
assert np.array_equal(np.load(a.out), full)
print("teacher shard merge restores original row order")

# ---- 4. eval metrics + encoder on a stubbed NanoBEIR ----
import scripts.eval_embed as ev
assert abs(ev.ndcg_at_k(["d1", "d2"], {"d1": 1.0, "d2": 1.0}) - 1.0) < 1e-9
assert abs(ev.ndcg_at_k(["x", "d1"], {"d1": 1.0}) - 1 / math.log2(3)) < 1e-9
assert abs(ev.spearman(torch.tensor([1., 2, 3, 4]), torch.tensor([10., 20, 30, 40])) - 1.0) < 1e-9
assert abs(ev.spearman(torch.tensor([1., 2, 3, 4]), torch.tensor([4., 3, 2, 1])) + 1.0) < 1e-9
from datasets import Dataset
fake = {
    "corpus": Dataset.from_dict({"_id": ["d1", "d2", "d3"], "text": ["cats sit on mats", "dogs run fast", "stocks fell"]}),
    "queries": Dataset.from_dict({"_id": ["q1", "q2"], "text": ["where do cats sit", "fast dogs"]}),
    "qrels": Dataset.from_dict({"query-id": ["q1", "q2"], "corpus-id": ["d1", "d2"], "score": [1, 1]}),
    None: Dataset.from_dict({"sentence1": ["a cat", "a dog", "the sky"], "sentence2": ["a kitten", "a puppy", "stocks"],
                             "score": [0.9, 0.8, 0.0]}),
}
ev.load_split = lambda name, config, split: fake[config]
emb = tiny_embedder("mean", True)
ck = os.path.join(TMP, "emb.pt")
torch.save({"model": emb.lm.state_dict(), "config": {**cfg.to_dict(), "preset": "tiny"},
            "embed": {"pooling": "mean", "bidirectional": True, "query_prefix": "query: ",
                      "passage_prefix": "passage: ", "q_len": 160, "p_len": 64}}, ck)
enc = ev.Encoder(ck, torch.device("cpu"), tokenize=lambda s: list(s.encode()), batch_size=2)
nd = ev.eval_nanobeir(enc, "NanoMSMARCO", ev.NANOBEIR["NanoMSMARCO"])
st = ev.eval_stsb(enc)
# encoder must reproduce the Embedder exactly (prefix + EOT + per-batch padding)
direct, _ = emb.eval()(*pad_batch([list(("passage: " + "dogs run fast").encode()) + [50256]], 64))
via_enc = enc.encode(["dogs run fast", "cats sit on mats"], "passage: ", 64)[0]
print(f"eval: nDCG@10={nd:.3f} stsb={st:.3f}; encoder vs direct max|diff|={(direct[0] - via_enc).abs().max().item():.2e}")
assert 0.0 <= nd <= 1.0 and -1.0 <= st <= 1.0 and (direct[0] - via_enc).abs().max().item() < 1e-5

# ---- 5. data prep on stubbed datasets ----
import datasets as hfds
import scripts.prep_embed_data as prep
stub = {
    ("sentence-transformers/msmarco-bm25", "triplet"): Dataset.from_dict(
        {"query": ["q a", "q b", "q a"], "positive": ["p1", "p2", "p3"], "negative": ["n1", "p2", "n3"]}),
    ("sentence-transformers/all-nli", "triplet"): Dataset.from_dict(
        {"anchor": ["s1"], "positive": ["e1"], "negative": ["c1"]}),
}
hfds.load_dataset = lambda name, config, split: stub[(name, config)]
sys.argv = ["prep", "--out_dir", os.path.join(TMP, "embed")]
prep.main()
pairs = [json.loads(l) for l in open(os.path.join(TMP, "embed", "pairs.jsonl"))]
texts = [json.loads(l) for l in open(os.path.join(TMP, "embed", "texts.jsonl"))]
assert len(pairs) == 3                       # the row with positive == negative is dropped
assert all(len(p["neg"]) == 1 and p["instruction"] for p in pairs)
assert sum(t["kind"] == "query" for t in texts) == 2   # "q a" counted once
assert all(t.get("instruction") for t in texts if t["kind"] == "query")
print("prep: drops degenerate rows, dedups queries, keeps instructions on queries")

print("ALL ABLATION-KIT CHECKS PASSED")
