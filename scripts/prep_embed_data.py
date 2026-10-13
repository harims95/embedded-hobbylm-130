"""Build the ablation datasets from Hugging Face (run on the training machine; needs `datasets`).

  python scripts/prep_embed_data.py --out_dir data/embed

Writes:
  pairs.jsonl  contrastive data, {"instruction", "query", "pos", "neg": [..]} per line
               - sentence-transformers/msmarco-bm25  (triplet)  web search -> passage, BM25 hard negative
               - sentence-transformers/all-nli      (triplet)  premise -> entailed hypothesis, contradiction as negative
  texts.jsonl  unique texts for distillation, {"text", "kind": "query"|"passage", "instruction"?}
               queries keep their instruction so the teacher and student see the same format.

Every row has exactly one hard negative (BM25-mined for MS MARCO, contradiction for NLI),
so train_embed.py --n_neg 1 uses all rows.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SOURCES = [
    # (dataset, config, split, query col, pos col, neg col, instruction)
    ("sentence-transformers/msmarco-bm25", "triplet", "train", "query", "positive", "negative",
     "Given a web search query, retrieve relevant passages that answer the query"),
    ("sentence-transformers/all-nli", "triplet", "train", "anchor", "positive", "negative",
     "Given a premise, retrieve a hypothesis that is entailed by the premise"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/embed")
    ap.add_argument("--max_rows_per_source", type=int, default=0, help="0 = all")
    ap.add_argument("--max_distill_texts", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    from datasets import load_dataset

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pairs, queries, passages = [], {}, set()
    for name, config, split, qc, pc, nc, inst in SOURCES:
        ds = load_dataset(name, config, split=split)
        missing = {qc, pc, nc} - set(ds.column_names)
        if missing:
            raise SystemExit(f"{name}/{config}: missing columns {missing}; has {ds.column_names}")
        n = len(ds) if not args.max_rows_per_source else min(len(ds), args.max_rows_per_source)
        kept = 0
        for row in ds.select(range(n)):
            q, p, neg = row[qc], row[pc], row[nc]
            if not (q and p and neg) or p == neg:
                continue
            pairs.append({"instruction": inst, "query": q, "pos": p, "neg": [neg]})
            queries[(q, inst)] = None
            passages.update((p, neg))
            kept += 1
        print(f"{name}/{config}: kept {kept}/{n} rows")

    rng.shuffle(pairs)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for ex in pairs:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    texts = [{"text": q, "kind": "query", "instruction": inst} for (q, inst) in queries]
    texts += [{"text": p, "kind": "passage"} for p in passages]
    rng.shuffle(texts)
    texts = texts[: args.max_distill_texts]
    with open(out / "texts.jsonl", "w", encoding="utf-8") as f:
        for ex in texts:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    nq = sum(t["kind"] == "query" for t in texts)
    print(f"wrote {len(pairs)} pairs -> {out/'pairs.jsonl'}")
    print(f"wrote {len(texts)} distill texts ({nq} queries, {len(texts)-nq} passages) -> {out/'texts.jsonl'}")


if __name__ == "__main__":
    main()
