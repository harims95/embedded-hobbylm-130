"""Fast embedding eval for ablations: NanoBEIR (13 retrieval sets, nDCG@10) + STS-B test (Spearman).

  python scripts/eval_embed.py --ckpt runs/emb_A/model.pt
  python scripts/eval_embed.py --ckpt runs/emb_A/model.pt --quick     # 4 NanoBEIR sets + STS-B

Reads pooling / attention mode / prefixes / widths from the checkpoint's "embed" block, so a
model is always evaluated the way it was trained. Writes eval.json next to the checkpoint.
Needs `datasets` and `tiktoken`. NanoMSMARCO overlaps the MS MARCO training data, so it is not
zero-shot; the average is still comparable across ablation arms trained on the same data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path
from hobbylm.config import ModelConfig
from hobbylm.embed import EOT, Embedder, pad_batch, text_ids
from hobbylm.model import MoETransformer

# Query instructions in the E5-mistral / Qwen3-Embedding style. Documents get no instruction.
NANOBEIR = {
    "NanoArguAna": "Given a claim, find documents that refute the claim",
    "NanoClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "NanoDBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "NanoFEVER": "Given a claim, retrieve documents that support or refute the claim",
    "NanoFiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "NanoHotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "NanoMSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NanoNFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NanoNQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "NanoQuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "NanoSCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "NanoSciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "NanoTouche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
}
QUICK = ["NanoMSMARCO", "NanoNQ", "NanoSciFact", "NanoFiQA2018"]


def load_split(name: str, config: str | None, split: str):
    from datasets import load_dataset
    return load_dataset(name, config, split=split)


def pick(cols: list[str], *options: str) -> str:
    for o in options:
        if o in cols:
            return o
    raise SystemExit(f"none of {options} in columns {cols}")


def load_nanobeir(name: str):
    repo = f"zeta-alpha-ai/{name}"
    corpus = load_split(repo, "corpus", "train")
    queries = load_split(repo, "queries", "train")
    qrels = load_split(repo, "qrels", "train")
    cid, ctext = pick(corpus.column_names, "_id", "id"), pick(corpus.column_names, "text")
    qid, qtext = pick(queries.column_names, "_id", "id"), pick(queries.column_names, "text")
    rq, rc = pick(qrels.column_names, "query-id", "query_id", "qid"), pick(qrels.column_names, "corpus-id", "corpus_id", "doc_id")
    rs = "score" if "score" in qrels.column_names else None
    docs = {str(r[cid]): r[ctext] for r in corpus if r[ctext]}
    qs = {str(r[qid]): r[qtext] for r in queries if r[qtext]}
    rel: dict[str, dict[str, float]] = {}
    for r in qrels:
        rel.setdefault(str(r[rq]), {})[str(r[rc])] = float(r[rs]) if rs else 1.0
    qs = {k: v for k, v in qs.items() if k in rel}
    return docs, qs, rel


def ndcg_at_k(ranked: list[str], rel: dict[str, float], k: int = 10) -> float:
    dcg = sum(rel.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    def ranks(x):
        r = torch.empty_like(x)
        r[x.argsort()] = torch.arange(len(x), dtype=x.dtype)
        return r
    ra, rb = ranks(a.double()), ranks(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


class Encoder:
    def __init__(self, ckpt_path: str, device, tokenize=None, batch_size: int = 128):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_d = dict(ck["config"]); cfg_d.pop("preset", None)
        cfg = ModelConfig(**cfg_d)
        cfg.expert_backend = "grouped" if device.type == "cuda" else "bmm"
        lm = MoETransformer(cfg)
        lm.load_state_dict(ck["model"])
        lm.set_bias_update_rate(0.0)
        self.e = ck.get("embed") or {}
        self.model = Embedder(lm, pooling=self.e.get("pooling", "last"),
                              bidirectional=self.e.get("bidirectional", False)).to(device).eval()
        self.device, self.batch_size = device, batch_size
        if tokenize is None:
            import tiktoken
            tokenize = tiktoken.get_encoding("gpt2").encode_ordinary
        self.tokenize = tokenize
        self.amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    @torch.no_grad()
    def encode(self, texts: list[str], prefix: str, width: int) -> torch.Tensor:
        ids = [text_ids(self.tokenize, prefix, t, width) for t in texts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = torch.empty(len(ids), self.model.lm.cfg.d_model)
        for b in range(0, len(order), self.batch_size):
            js = order[b:b + self.batch_size]
            w = max(len(ids[j]) for j in js)                # pad per batch: eval has no compile
            idx, lengths = pad_batch([ids[j] for j in js], w, self.device)
            with self.amp:
                emb, _ = self.model(idx, lengths)
            out[js] = emb.float().cpu()
        return out

    def query_prefix(self, instruction: str | None) -> str:
        return f"Instruct: {instruction}\nQuery: " if instruction else self.e.get("query_prefix", "query: ")

    @property
    def passage_prefix(self) -> str:
        return self.e.get("passage_prefix", "passage: ")


def eval_nanobeir(enc: Encoder, name: str, instruction: str) -> float:
    docs, qs, rel = load_nanobeir(name)
    dids, qids = list(docs), list(qs)
    D = enc.encode([docs[d] for d in dids], enc.passage_prefix, enc.e.get("p_len", 256))
    Q = enc.encode([qs[q] for q in qids], enc.query_prefix(instruction), enc.e.get("q_len", 128))
    top = (Q @ D.T).topk(min(10, len(dids)), dim=1).indices
    return sum(ndcg_at_k([dids[j] for j in top[i].tolist()], rel[q]) for i, q in enumerate(qids)) / len(qids)


def eval_stsb(enc: Encoder) -> float:
    ds = load_split("sentence-transformers/stsb", None, "test")
    width = enc.e.get("p_len", 256)
    a = enc.encode(list(ds["sentence1"]), enc.passage_prefix, width)
    b = enc.encode(list(ds["sentence2"]), enc.passage_prefix, width)
    return spearman((a * b).sum(-1), torch.tensor(ds["score"], dtype=torch.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--quick", action="store_true", help="4 NanoBEIR sets instead of 13")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--debug_byte_tokenizer", action="store_true", help="CPU smoke tests only")
    ap.add_argument("--q_len", type=int, default=0, help="override the checkpoint's query width (0 = keep)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenize = (lambda t: list(t.encode("utf-8"))) if args.debug_byte_tokenizer else None
    enc = Encoder(args.ckpt, device, tokenize=tokenize, batch_size=args.batch_size)
    if args.q_len:
        enc.e["q_len"] = args.q_len
    t0 = time.time()
    names = QUICK if args.quick else list(NANOBEIR)
    res = {}
    for n in names:
        res[n] = eval_nanobeir(enc, n, NANOBEIR[n])
        print(f"{n:22s} nDCG@10 {100*res[n]:6.2f}", flush=True)
    res["nanobeir_mean"] = sum(res[n] for n in names) / len(names)
    res["stsb_spearman"] = eval_stsb(enc)
    print(f"{'NanoBEIR mean':22s} nDCG@10 {100*res['nanobeir_mean']:6.2f}")
    print(f"{'STS-B test':22s} Spearman {100*res['stsb_spearman']:6.2f}   ({time.time()-t0:.0f}s)")
    out = Path(args.ckpt).with_name("eval_quick.json" if args.quick else "eval.json")
    out.write_text(json.dumps({"ckpt": args.ckpt, "embed": enc.e, **res}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
