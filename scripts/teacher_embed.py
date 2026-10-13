"""Embed texts with a teacher model (default Qwen3-Embedding-4B) for distillation.

  # one process per GPU, each takes every num_shards-th text
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/teacher_embed.py --texts texts.jsonl --out teacher.npy \
        --shard $i --num_shards 4 &
  done; wait
  python scripts/teacher_embed.py --texts texts.jsonl --out teacher.npy --merge 4

Input: JSONL with {"text": str, "kind": "query"|"passage", "instruction": str (optional)}.
Queries with an instruction use the Qwen3-Embedding format "Instruct: {instruction}\\nQuery:{text}";
passages are embedded as-is. Output: float16 .npy of shape (N, dim), row i = line i, unit-norm.
--dim truncates (Matryoshka) and renormalizes; Qwen3-Embedding is trained to allow this.

--check_st compares this script's embeddings with sentence-transformers' encode() on a few texts
and stops if they disagree, so a pooling/formatting mistake can't silently poison the targets.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


def format_text(ex: dict) -> str:
    if ex.get("kind") == "query" and ex.get("instruction"):
        return f"Instruct: {ex['instruction']}\nQuery:{ex['text']}"
    return ex["text"]


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of the last real token; works for left or right padding."""
    if bool((attention_mask[:, -1] == 1).all()):            # left padded: last column is real everywhere
        return hidden[:, -1]
    last = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


@torch.no_grad()
def encode(model, tokenizer, texts: list[str], max_len: int, dim: int | None, device) -> torch.Tensor:
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    out = model(**batch)
    emb = last_token_pool(out.last_hidden_state, batch["attention_mask"]).float()
    emb = F.normalize(emb, dim=-1)
    if dim:
        emb = F.normalize(emb[:, :dim], dim=-1)
    return emb


def read_texts(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_shard(args):
    from transformers import AutoModel, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.to(device).eval()
    rows = read_texts(args.texts)
    idx = list(range(args.shard, len(rows), args.num_shards))
    texts = [format_text(rows[i]) for i in idx]

    if args.check_st and args.shard == 0:
        check_against_sentence_transformers(args, model, tokenizer, rows, device)

    # sort by length so each batch pads little; write results back in original order
    order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
    dim = args.dim or model.config.hidden_size
    out = np.zeros((len(texts), dim), dtype=np.float16)
    t0 = time.time()
    for b in range(0, len(order), args.batch_size):
        js = order[b:b + args.batch_size]
        emb = encode(model, tokenizer, [texts[j] for j in js], args.max_len, args.dim, device)
        out[js] = emb.cpu().numpy().astype(np.float16)
        if (b // args.batch_size) % 50 == 0:
            done = b + len(js)
            rate = done / max(1e-9, time.time() - t0)
            print(f"[shard {args.shard}] {done}/{len(texts)} texts, {rate:.0f}/s", flush=True)
    part = args.out if args.num_shards == 1 else f"{args.out}.part{args.shard}of{args.num_shards}.npy"
    np.save(part, out)
    print(f"[shard {args.shard}] wrote {part} {out.shape}", flush=True)


def check_against_sentence_transformers(args, model, tokenizer, rows, device):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(args.model, device=str(device))
    sample = rows[:8]
    mine = encode(model, tokenizer, [format_text(r) for r in sample], args.max_len, None, device)
    ref = []
    for r in sample:
        if r.get("kind") == "query" and r.get("instruction"):
            ref.append(st.encode([r["text"]], prompt=f"Instruct: {r['instruction']}\nQuery:", convert_to_tensor=True))
        else:
            ref.append(st.encode([r["text"]], convert_to_tensor=True))
    ref = F.normalize(torch.cat(ref).float().to(device), dim=-1)
    cos = (mine * ref).sum(-1)
    print(f"check vs sentence-transformers: cosine min={cos.min().item():.4f} mean={cos.mean().item():.4f}")
    if cos.min().item() < 0.99:
        raise SystemExit("teacher embeddings disagree with sentence-transformers; fix formatting/pooling first")


def merge(args):
    n = args.merge
    parts = [np.load(f"{args.out}.part{s}of{n}.npy", mmap_mode="r") for s in range(n)]
    total = sum(p.shape[0] for p in parts)
    dim = parts[0].shape[1]
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(total, dim))
    for s, p in enumerate(parts):
        out[s::n] = p                     # shard s held rows s, s+n, s+2n, ...
    out.flush()
    norms = np.linalg.norm(out[: min(total, 1000)].astype(np.float32), axis=1)
    print(f"merged {n} parts -> {args.out} {out.shape}; first-1000 norms {norms.min():.3f}..{norms.max():.3f}")
    for s in range(n):
        os.remove(f"{args.out}.part{s}of{n}.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--dim", type=int, default=1024, help="Matryoshka truncation (0 = full width)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--merge", type=int, default=0, help="merge N shard parts into --out and exit")
    ap.add_argument("--check_st", action="store_true", help="verify against sentence-transformers first")
    args = ap.parse_args()
    if args.dim == 0:
        args.dim = None
    if args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
