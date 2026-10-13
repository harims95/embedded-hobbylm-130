"""Embedding distillation: train the MoE LM so its embeddings match a teacher's.

  python training/train_distill.py --init_from runs/130M_5B/model.pt --texts texts.jsonl \
      --teacher teacher.npy --bidirectional --pooling mean --run_name distill_v0

Inputs (from scripts/prep_embed_data.py + scripts/teacher_embed.py):
  --texts    JSONL, line i = {"text", "kind": "query"|"passage", "instruction"?}
  --teacher  float16 .npy (N, D), row i = teacher embedding of line i (unit-norm)

Loss, per batch of B texts (student s = our embedding, t = teacher embedding):
  cosine     : mean(1 - cos(P s, t))          P = a learned linear map d -> D (dims differ)
  similarity : MSE between the B x B similarity matrices  s s^T  and  t t^T
The similarity term needs no projection: it teaches the student the teacher's geometry directly.
P is only a training aid; the saved checkpoint's native embeddings are what get evaluated.

Same mechanics as train_embed.py: GradCache (memory set by --chunk), frozen expert bias,
router on AdamW, explicit gradient all_reduce across ranks. Output checkpoint has the same
format, so it can be passed straight to train_embed.py --init_from for a contrastive stage.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root on path
from hobbylm.config import ModelConfig, TrainConfig, get_config
from hobbylm.embed import EOT, Embedder, pad_batch, text_ids
from hobbylm.model import MoETransformer, count_params
from hobbylm.optim import build_optimizers
from training.train_embed import lr_mult, make_tokenizer


def distill_losses(s: torch.Tensor, proj: nn.Linear, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """s: (B, d) unit-norm student, t: (B, D) unit-norm teacher. Returns (cosine loss, similarity loss)."""
    ps = F.normalize(proj(s), dim=-1)
    cos_loss = (1.0 - (ps * t).sum(-1)).mean()
    sim_loss = F.mse_loss(s @ s.T, t @ t.T)
    return cos_loss, sim_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_from", default="", help="LM or embedder checkpoint. Empty = random init (tests only)")
    ap.add_argument("--preset", default="130M")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--texts", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--run_name", default="distill")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--pooling", default="mean", choices=["last", "mean"])
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--query_prefix", default="query: ")
    ap.add_argument("--passage_prefix", default="passage: ")
    ap.add_argument("--q_len", type=int, default=64)
    ap.add_argument("--p_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=512, help="texts per step PER GPU")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--w_cos", type=float, default=1.0)
    ap.add_argument("--w_sim", type=float, default=1.0)
    ap.add_argument("--aux_weight", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--final_lr_frac", type=float, default=0.1)
    ap.add_argument("--muon_lr", type=float, default=1e-3)
    ap.add_argument("--adam_lr", type=float, default=1e-5)
    ap.add_argument("--proj_lr", type=float, default=1e-3, help="AdamW LR for the projection head")
    ap.add_argument("--freeze_router", action="store_true")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--debug_byte_tokenizer", action="store_true")
    args = ap.parse_args()

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

    # ---- model (identical on every rank) ----
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
        log(f"loaded {args.init_from} (step={ck.get('step')})")
    lm.to(device)
    lm.set_bias_update_rate(0.0)
    if args.freeze_router:
        for name, p in lm.named_parameters():
            if name.endswith("gate.weight"):
                p.requires_grad_(False)
    model = Embedder(lm, pooling=args.pooling, bidirectional=args.bidirectional).to(device).train()

    # ---- data ----
    with open(args.texts, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    teacher = np.load(args.teacher, mmap_mode="r")
    assert teacher.shape[0] == len(rows), f"teacher rows {teacher.shape[0]} != texts {len(rows)}"
    D = teacher.shape[1]
    proj = nn.Linear(cfg.d_model, D, bias=False).to(device)
    nn.init.normal_(proj.weight, std=cfg.d_model ** -0.5)
    log(f"texts={len(rows)} teacher_dim={D} student_dim={cfg.d_model} pooling={args.pooling} "
        f"bidirectional={args.bidirectional} world={world}")

    tc = TrainConfig(muon_lr=args.muon_lr, adam_lr=args.adam_lr)
    muon, adamw, _ = build_optimizers(lm, tc)
    proj_opt = torch.optim.AdamW(proj.parameters(), lr=args.proj_lr, weight_decay=0.0)
    params = [p for p in lm.parameters() if p.requires_grad] + list(proj.parameters())
    amp = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device.type == "cuda" and not args.no_bf16 else nullcontext())
    tokenize = make_tokenizer(args.debug_byte_tokenizer)
    rng = np.random.default_rng(args.seed + rank)

    def format_ids(r: dict) -> tuple[list[int], int]:
        if r.get("kind") == "query":
            prefix = f"Instruct: {r['instruction']}\nQuery: " if r.get("instruction") else args.query_prefix
            width = args.q_len
        else:
            prefix, width = args.passage_prefix, args.p_len
        return text_ids(tokenize, prefix, r["text"], width), width

    def next_batch():
        idx = rng.integers(0, len(rows), size=args.batch_size)   # with replacement: O(B), duplicates are rare
        groups = {args.q_len: [], args.p_len: []}
        order = {args.q_len: [], args.p_len: []}
        for i in idx:
            ids, width = format_ids(rows[i])
            groups[width].append(ids)
            order[width].append(i)
        chunks, rows_in_order = [], []
        for width in (args.q_len, args.p_len):
            g = groups[width]
            for c in range(0, len(g), args.chunk):
                chunks.append(pad_batch(g[c:c + args.chunk], width, device))
            rows_in_order.extend(order[width])
        t = torch.from_numpy(np.asarray(teacher[np.asarray(rows_in_order)], dtype=np.float32)).to(device)
        return chunks, F.normalize(t, dim=-1)

    out_dir = Path(args.out_dir) / args.run_name
    embed_cfg = {"pooling": args.pooling, "bidirectional": args.bidirectional,
                 "query_prefix": args.query_prefix, "passage_prefix": args.passage_prefix,
                 "q_len": args.q_len, "p_len": args.p_len, "distilled_from": args.teacher}
    if master:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "embed": embed_cfg,
                                                         "args": vars(args)}, indent=2))

    def save_ckpt(fname: str, step: int):
        if master:
            torch.save({"model": lm.state_dict(), "config": {**cfg.to_dict(), "preset": preset},
                        "embed": embed_cfg, "proj": proj.state_dict(), "step": step}, out_dir / fname)
            log(f"saved checkpoint -> {out_dir / fname}")

    t0 = time.time()
    for step in range(args.max_steps):
        m = lr_mult(step, args.max_steps, args.warmup_steps, args.final_lr_frac)
        for g in muon.param_groups:
            g["lr"] = args.muon_lr * m
        for g in adamw.param_groups:
            g["lr"] = g.setdefault("base_lr", g["lr"]) * m   # base recorded on first step
        for g in proj_opt.param_groups:
            g["lr"] = args.proj_lr * m

        chunks, t = next_batch()
        with torch.no_grad(), amp:
            reps = torch.cat([model(i, l)[0] for i, l in chunks])
        reps.requires_grad_(True)
        cos_loss, sim_loss = distill_losses(reps, proj, t)
        loss = args.w_cos * cos_loss + args.w_sim * sim_loss
        loss.backward()                                   # grads -> proj.weight and reps.grad
        rep_grad = reps.grad
        aux_log, row = 0.0, 0
        for i, l in chunks:
            with amp:
                r, aux = model(i, l)
            g = rep_grad[row:row + r.shape[0]]
            row += r.shape[0]
            ((r * g).sum() + args.aux_weight * aux / len(chunks)).backward()
            aux_log += aux.item() / len(chunks)

        if world > 1:
            # every loss here is local to the rank's own batch: average the gradients
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        muon.step(); adamw.step(); proj_opt.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True); proj_opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            dt = (time.time() - t0) / (step + 1)
            log(f"step {step:5d} | cos {cos_loss.item():.4f} | sim {sim_loss.item():.5f} | aux {aux_log:.4f} | "
                f"grad_norm {grad_norm.item():.4f} | {dt*1000:.0f}ms/step")
        if args.save_every and (step + 1) % args.save_every == 0:
            save_ckpt(f"ckpt_{step+1}.pt", step + 1)

    save_ckpt("model.pt", args.max_steps)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
