"""Contrastive (InfoNCE) fine-tuning of the MoE LM into a text embedder.

  # causal + last-token pooling (baseline), single GPU
  python training/train_embed.py --init_from runs/base/model.pt --train_file pairs.jsonl --run_name emb_v0

  # bidirectional + mean pooling, from a masked-token-adapted checkpoint, 4 GPUs
  torchrun --standalone --nproc_per_node=4 training/train_embed.py --init_from runs/mntp/model.pt \
      --train_file pairs.jsonl --bidirectional --pooling mean --batch_size 2048 --chunk 64

Data: JSONL, one example per line: {"query": str, "pos": str, "neg": [str, ...], "instruction": str}
("neg" and "instruction" optional; an instruction formats the query as "Instruct: ...\nQuery: ...").
With --n_neg K > 0, lines with no negatives are skipped; lines with fewer than K are sampled with repeats.

Large batches use GradCache (Gao et al. 2021): embed everything without grad, take the loss
gradient w.r.t. the embeddings, then re-embed chunk by chunk with grad. Peak memory is set by
--chunk, not --batch_size. Multi-GPU: every rank scores against ALL ranks' candidates, and
gradients are averaged with an explicit all_reduce (no DDP wrapper).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path
from hobbylm.config import ModelConfig, TrainConfig, get_config
from hobbylm.embed import EOT, Embedder, info_nce, pad_batch, text_ids
from hobbylm.model import MoETransformer, count_params
from hobbylm.optim import build_optimizers


def lr_mult(step: int, max_steps: int, warmup: int, final_frac: float) -> float:
    """Linear warmup, then linear decay to final_frac."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return 1.0 + t * (final_frac - 1.0)


def make_tokenizer(debug_bytes: bool):
    if debug_bytes:                       # CPU smoke tests only: no tokenizer download needed
        return lambda text: list(text.encode("utf-8"))
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode_ordinary


def example_stream(path: str, rank: int, world: int, seed: int, shuffle_buf: int, n_neg: int):
    """Endless stream of (instruction, query, pos, [negs]) from a JSONL file. Each rank takes every
    world-th line. Shuffles through a fixed-size buffer so the file never has to fit in memory;
    the buffer is flushed at the end of every pass, so files smaller than the buffer still stream."""
    rng = random.Random(seed + rank)
    buf = []
    while True:
        seen = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line_no % world != rank or not line.strip():
                    continue
                ex = json.loads(line)
                negs = ex.get("neg") or []
                if n_neg > 0 and not negs:
                    continue
                seen += 1
                item = (ex.get("instruction"), ex["query"], ex["pos"],
                        [rng.choice(negs) for _ in range(n_neg)] if n_neg else [])
                if len(buf) < shuffle_buf:
                    buf.append(item)
                    continue
                j = rng.randrange(shuffle_buf)
                buf[j], item = item, buf[j]
                yield item
        if seen == 0:
            raise SystemExit(f"{path}: no usable examples for rank {rank} (n_neg={n_neg})")
        rng.shuffle(buf)
        yield from buf
        buf = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default="", help="LM checkpoint .pt (train.py format). Empty = random init (tests only)")
    ap.add_argument("--preset", default="1B", help="only used when --init_from is empty")
    ap.add_argument("--set", nargs="*", default=[], help="model config overrides key=value (random init only)")
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--run_name", default="embed")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--bidirectional", action="store_true",
                    help="full attention over the text (use with a masked-token-adapted checkpoint)")
    ap.add_argument("--query_prefix", default="query: ")
    ap.add_argument("--passage_prefix", default="passage: ")
    ap.add_argument("--q_len", type=int, default=64, help="query width in tokens (truncate + pad)")
    ap.add_argument("--p_len", type=int, default=256, help="passage width in tokens (truncate + pad)")
    ap.add_argument("--batch_size", type=int, default=256, help="queries per step PER GPU")
    ap.add_argument("--chunk", type=int, default=32, help="texts per forward pass (memory knob)")
    ap.add_argument("--n_neg", type=int, default=0, help="hard negatives per query")
    ap.add_argument("--temperature", type=float, default=0.03)
    ap.add_argument("--fn_margin", type=float, default=None,
                    help="drop negatives scoring above positive + this margin (Qwen3-Embedding: 0.1); off by default")
    ap.add_argument("--aux_weight", type=float, default=1.0, help="weight on the MoE balance/z losses")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--final_lr_frac", type=float, default=0.1)
    ap.add_argument("--muon_lr", type=float, default=1e-3, help="peak Muon LR (pretraining used 0.02+; sweep this)")
    ap.add_argument("--adam_lr", type=float, default=1e-5, help="peak AdamW LR (embeddings, norms, router)")
    ap.add_argument("--freeze_router", action="store_true", help="keep the pretrained routing fixed")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--shuffle_buf", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--debug_byte_tokenizer", action="store_true", help="CPU smoke tests: bytes as tokens")
    args = ap.parse_args()

    # ---- distributed setup (gloo on CPU so the multi-rank path is testable without GPUs) ----
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    master = rank == 0

    def log(*a):
        if master:
            print(*a, flush=True)

    # ---- model: identical weights on every rank (same checkpoint, or same seed) ----
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg_d = dict(ck["config"]); preset = cfg_d.pop("preset", None)
        cfg = ModelConfig(**cfg_d)
    else:
        from training.train import parse_overrides
        cfg, preset = get_config(args.preset), args.preset
        for k, v in parse_overrides(args.set).items():
            setattr(cfg, k, v)
        cfg.__post_init__()
    cfg.expert_backend = "grouped" if device.type == "cuda" else "bmm"
    lm = MoETransformer(cfg)
    if args.init_from:
        lm.load_state_dict(ck["model"])
        log(f"loaded {args.init_from} (step={ck.get('step')}, val={ck.get('val_loss')})")
    lm.to(device)
    lm.set_bias_update_rate(0.0)          # GradCache forwards twice per batch; the bias must not move
    if args.freeze_router:
        for name, p in lm.named_parameters():
            if name.endswith("gate.weight"):
                p.requires_grad_(False)
    model = Embedder(lm, pooling=args.pooling, bidirectional=args.bidirectional).to(device)
    model.train()
    pc = count_params(lm)
    log(f"total={pc['total']/1e6:.1f}M active={pc['active']/1e6:.1f}M pooling={args.pooling} "
        f"bidirectional={args.bidirectional} world={world} device={device}")

    tc = TrainConfig(muon_lr=args.muon_lr, adam_lr=args.adam_lr)
    muon, adamw, (nm, na) = build_optimizers(lm, tc)
    log(f"optimizers: Muon over {nm} tensors, AdamW over {na} tensors")
    params = [p for p in lm.parameters() if p.requires_grad]
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.type == "cuda" and not args.no_bf16 else nullcontext())

    # ---- data ----
    tokenize = make_tokenizer(args.debug_byte_tokenizer)
    stream = example_stream(args.train_file, rank, world, args.seed, args.shuffle_buf, args.n_neg)

    def toks(prefix: str, text: str, width: int) -> list[int]:
        return text_ids(tokenize, prefix, text, width)

    def query_prefix(instruction):
        # per-example instruction (Qwen3-Embedding format) wins over the fixed --query_prefix
        return f"Instruct: {instruction}\nQuery: " if instruction else args.query_prefix

    def next_batch():
        """Returns the chunk list [(idx, lengths), ...] ordered queries | positives | negatives."""
        qs, ps, ns = [], [], []
        for _ in range(args.batch_size):
            inst, q, p, negs = next(stream)
            qs.append(toks(query_prefix(inst), q, args.q_len))
            ps.append(toks(args.passage_prefix, p, args.p_len))
            ns.extend(toks(args.passage_prefix, n, args.p_len) for n in negs)
        chunks = []
        for rows, width in ((qs, args.q_len), (ps + ns, args.p_len)):
            for i in range(0, len(rows), args.chunk):
                chunks.append(pad_batch(rows[i:i + args.chunk], width, device))
        return chunks

    B = args.batch_size
    n_cand = B * (1 + args.n_neg)         # per rank: positives then hard negatives

    def global_loss(reps: torch.Tensor):
        """reps: this rank's (B + n_cand, d). Loss over ALL ranks' queries vs ALL ranks' candidates,
        with this rank's rows carrying grad. Every rank computes the same number."""
        if world == 1:
            q, cands = reps[:B], reps[B:]
            labels = torch.arange(B, device=reps.device)
        else:
            gathered = [torch.zeros_like(reps) for _ in range(world)]
            dist.all_gather(gathered, reps.detach())
            gathered[rank] = reps                              # keep grad for our own rows
            q = torch.cat([g[:B] for g in gathered])
            cands = torch.cat([g[B:] for g in gathered])
            # query i of rank r has its positive at r * n_cand + i
            labels = torch.cat([r * n_cand + torch.arange(B, device=reps.device) for r in range(world)])
        return info_nce(q, cands, labels, args.temperature, args.fn_margin)

    out_dir = Path(args.out_dir) / args.run_name
    embed_cfg = {"pooling": args.pooling, "bidirectional": args.bidirectional,
                 "query_prefix": args.query_prefix, "passage_prefix": args.passage_prefix,
                 "fn_margin": args.fn_margin,
                 "q_len": args.q_len, "p_len": args.p_len, "temperature": args.temperature}
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "embed": embed_cfg,
                                                         "args": vars(args)}, indent=2))

    def save_ckpt(fname: str, step: int):
        if not master:
            return
        torch.save({"model": lm.state_dict(), "config": {**cfg.to_dict(), "preset": preset},
                    "embed": embed_cfg, "step": step}, out_dir / fname)
        log(f"saved checkpoint -> {out_dir / fname}")

    log(f"batch={B}x{world} queries, {n_cand * world} candidates/step, chunk={args.chunk}, "
        f"q_len={args.q_len} p_len={args.p_len} muon_lr={args.muon_lr} adam_lr={args.adam_lr}")

    # ---- train ----
    t0 = time.time()
    for step in range(args.max_steps):
        m = lr_mult(step, args.max_steps, args.warmup_steps, args.final_lr_frac)
        for g in muon.param_groups:
            g["lr"] = args.muon_lr * m
        for g in adamw.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * m   # base recorded on first step

        chunks = next_batch()
        # pass 1: embeddings only, no graph
        with torch.no_grad(), amp:
            reps = torch.cat([model(idx, lengths)[0] for idx, lengths in chunks])
        # loss gradient w.r.t. the embeddings
        reps.requires_grad_(True)
        loss, acc = global_loss(reps)
        (rep_grad,) = torch.autograd.grad(loss, reps)
        # pass 2: re-embed with grad, chunk by chunk; d(loss)/d(params) = sum_chunks d(rep . rep_grad)
        aux_log = 0.0
        row = 0
        for idx, lengths in chunks:
            with amp:
                r, aux = model(idx, lengths)
            g = rep_grad[row:row + r.shape[0]]
            row += r.shape[0]
            # aux is divided by world because the ranks' gradients are SUMMED below
            surrogate = (r * g).sum() + args.aux_weight * aux / (len(chunks) * world)
            surrogate.backward()
            aux_log += aux.item() / len(chunks)

        if world > 1:
            # each rank holds d(global loss)/d(params) through ITS rows only: the true gradient is the
            # SUM over ranks.
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            dt = (time.time() - t0) / (step + 1)
            log(f"step {step:5d} | loss {loss.item():.4f} | acc {acc.item():.3f} | aux {aux_log:.4f} | "
                f"grad_norm {grad_norm.item():.4f} | {dt*1000:.0f}ms/step")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(f"ckpt_{step+1}.pt", step + 1)

    save_ckpt("model.pt", args.max_steps)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
