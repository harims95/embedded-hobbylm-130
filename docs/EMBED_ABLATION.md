# 130M embedding ablations

Goal: settle the embedding recipe on the 130M model (same architecture family as HobbyLM-1B,
minutes per run instead of hours), then run only the winning recipe on the 1B.

## What each arm answers

| Arm | Recipe | Question |
|---|---|---|
| base | untrained 130M, causal, last token | floor: training must beat this |
| A | contrastive, causal attention, last-token pooling | baseline recipe |
| B | contrastive, bidirectional, mean pooling | does dropping the causal mask help, with no adaptation step? |
| C | masked-token adaptation (~260M tokens), then B | is the adaptation step worth it? |
| D | distillation from Qwen3-Embedding-4B only | how far does distillation alone get? |
| E | D, then contrastive | do the two stack? |
| F | E with false-negative masking (margin 0.1) | does dropping suspicious negatives help? |

All contrastive arms use the same data (MS MARCO + NLI, 1 hard negative each) and the same steps,
so each comparison isolates one change. Set `MODE` for D-F from the A/B/C result
(default: bidirectional + mean).

## Run

```bash
pip install tiktoken datasets transformers sentence-transformers
CKPT=/path/to/130M/model.pt FINEWEB=/path/to/fineweb10B NGPU=4 bash scripts/ablate_130m.sh
```

Stages can be run one at a time (`... bash scripts/ablate_130m.sh A B eval report`). Every
knob (steps, batch sizes, widths, learning rates) is an environment variable at the top of
the script.

## Reading the result

`report` prints NanoBEIR mean nDCG@10 (13 small retrieval sets) and STS-B Spearman per arm.
NanoMSMARCO overlaps the training data, so absolute numbers are not zero-shot; differences
between arms are what matter. With 50 queries per NanoBEIR set, treat gaps under ~1 point as
noise: re-run the two arms with another `--seed` before deciding.

## Before the first real run

- Decode a few rows of `pairs.jsonl` and `texts.jsonl` and read them.
- `teacher` checks its embeddings against sentence-transformers on shard 0 and stops on a mismatch.
- First log line of each arm: loss near ln(number of candidates) for contrastive, then falling.
- Learning rates (`MUON_LR=2e-3`, `ADAM_LR=2e-5`) are untested starting points. If arm A's loss
  stalls or spikes, sweep these on A before running the rest.
