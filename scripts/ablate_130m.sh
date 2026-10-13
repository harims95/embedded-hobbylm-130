#!/usr/bin/env bash
# 130M embedding ablations, end to end. Run from anywhere; stages can be picked individually.
#
#   pip install tiktoken datasets transformers sentence-transformers
#   CKPT=/data/runs/130M_10B_resume_3000/model.pt FINEWEB=/data/fineweb10B NGPU=4 \
#       bash scripts/ablate_130m.sh                 # everything, in order
#   ... bash scripts/ablate_130m.sh A B eval report # just some stages
#
# Stages
#   topup   continue Prajan's 130M from ~1B to ~5B FineWeb tokens       (~1-1.5 h on 4xH200)
#   prep    MS MARCO + NLI pairs and distillation texts from Hugging Face
#   teacher Qwen3-Embedding-4B embeddings of the distillation texts       (~20-30 min)
#   mntp    ~260M tokens of masked-token training (bidirectional adapt)   (~10 min)
#   A       contrastive, causal attention + last-token pooling            (baseline)
#   B       contrastive, bidirectional + mean pooling, straight from base
#   C       contrastive, bidirectional + mean pooling, after mntp
#   D       distillation only                                             (attention mode = $MODE)
#   E       distillation, then contrastive
#   F       E + false-negative masking (margin 0.1)
#   eval    NanoBEIR (13 sets) + STS-B for the untrained base and every finished arm
#   report  one table of all results
#
# Every arm trains on the same data for the same number of steps, so differences are the variable.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CKPT:?set CKPT to the latest 130M model.pt from Prajan}"
: "${FINEWEB:?set FINEWEB to the directory holding fineweb_train_*.bin / fineweb_val_*.bin}"
NGPU=${NGPU:-4}
WORK=${WORK:-runs/embed_ablation}
DATA=$WORK/data
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-Embedding-4B}
MODE=${MODE:---bidirectional --pooling mean}      # attention/pooling for D, E, F (set after reading A-C)
RUN=${RUN:-torchrun --standalone --nproc_per_node=$NGPU}
BASE=$WORK/130M_5B/model.pt

# Knobs (defaults sized for 4xH200). EXTRA is appended to every embedding train/eval command.
TOPUP_STEPS=${TOPUP_STEPS:-15000}; MNTP_STEPS=${MNTP_STEPS:-1000}
STEPS_C=${STEPS_C:-1000}; STEPS_D=${STEPS_D:-1500}; BS_C=${BS_C:-256}; BS_D=${BS_D:-512}; CHUNK=${CHUNK:-128}
Q_LEN=${Q_LEN:-128}; P_LEN=${P_LEN:-256}; BATCH_TOKENS=${BATCH_TOKENS:-262144}; SEQ_LEN=${SEQ_LEN:-1024}; MICRO=${MICRO:-$((BATCH_TOKENS / SEQ_LEN / NGPU))}; EXTRA=${EXTRA:-}
COMMON="--q_len $Q_LEN --p_len $P_LEN --log_every 50 --muon_lr ${MUON_LR:-2e-3} --adam_lr ${ADAM_LR:-2e-5} --chunk $CHUNK $EXTRA"
CONTRAST="$COMMON --train_file $DATA/pairs.jsonl --n_neg 1 --batch_size $BS_C --max_steps $STEPS_C --warmup_steps 50"
DISTILL="$COMMON --texts $DATA/texts.jsonl --teacher $DATA/teacher.npy --batch_size $BS_D --max_steps $STEPS_D --warmup_steps 50"
PRETRAIN_EXTRA=${PRETRAIN_EXTRA:-}; EVAL_EXTRA=${EVAL_EXTRA:-}

stage_topup() {
  # init_from = weights only, fresh optimizer + schedule: 15000 x 262144 tokens = ~3.9B more tokens.
  $RUN training/train.py --preset 130M --run_name 130M_5B --out_dir "$WORK" --data_dir "$FINEWEB" \
    --init_from "$CKPT" --max_steps "$TOPUP_STEPS" --batch_tokens "$BATCH_TOKENS" --seq_len "$SEQ_LEN" \
    --micro_batch_seqs "$MICRO" --val_every 1000 --save_every 5000 $PRETRAIN_EXTRA
}

stage_prep() { python scripts/prep_embed_data.py --out_dir "$DATA"; }

stage_teacher() {
  # first shard checks the formatting against sentence-transformers before embedding anything
  for i in $(seq 0 $((NGPU - 1))); do
    extra=""; [ "$i" = 0 ] && extra="--check_st"
    CUDA_VISIBLE_DEVICES=$i python scripts/teacher_embed.py --texts "$DATA/texts.jsonl" \
      --out "$DATA/teacher.npy" --model "$TEACHER_MODEL" --shard "$i" --num_shards "$NGPU" $extra &
  done
  wait
  if [ "$NGPU" -gt 1 ]; then
    python scripts/teacher_embed.py --texts "$DATA/texts.jsonl" --out "$DATA/teacher.npy" --merge "$NGPU"
  fi
}

stage_mntp() {
  $RUN training/train.py --preset 130M --run_name mntp --out_dir "$WORK" --data_dir "$FINEWEB" \
    --init_from "$BASE" --max_steps "$MNTP_STEPS" --batch_tokens "$BATCH_TOKENS" --seq_len "$SEQ_LEN" \
    --micro_batch_seqs "$MICRO" --set diffusion=true --lr_mult 0.3 --val_every 500 $PRETRAIN_EXTRA
}

stage_A() { $RUN training/train_embed.py --init_from "$BASE" --run_name A_causal_last --out_dir "$WORK" $CONTRAST --pooling last; }
stage_B() { $RUN training/train_embed.py --init_from "$BASE" --run_name B_bi_mean --out_dir "$WORK" $CONTRAST --bidirectional --pooling mean; }
stage_C() { $RUN training/train_embed.py --init_from "$WORK/mntp/model.pt" --run_name C_mntp_bi_mean --out_dir "$WORK" $CONTRAST --bidirectional --pooling mean; }
stage_D() { $RUN training/train_distill.py --init_from "$BASE" --run_name D_distill --out_dir "$WORK" $DISTILL $MODE; }
stage_E() { $RUN training/train_embed.py --init_from "$WORK/D_distill/model.pt" --run_name E_distill_contrast --out_dir "$WORK" $CONTRAST $MODE; }
stage_F() { $RUN training/train_embed.py --init_from "$WORK/D_distill/model.pt" --run_name F_distill_contrast_fn --out_dir "$WORK" $CONTRAST $MODE --fn_margin 0.1; }

stage_eval() {
  python scripts/eval_embed.py --ckpt "$BASE" $EVAL_EXTRA   # untrained reference (causal, last token)
  for arm in A_causal_last B_bi_mean C_mntp_bi_mean D_distill E_distill_contrast F_distill_contrast_fn; do
    if [ -f "$WORK/$arm/model.pt" ]; then python scripts/eval_embed.py --ckpt "$WORK/$arm/model.pt" $EVAL_EXTRA; fi
  done
}

stage_report() {
  python - "$WORK" <<'EOF'
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
names = [("130M_5B", "untrained base"), ("A_causal_last", "A causal + last"), ("B_bi_mean", "B bi + mean"),
         ("C_mntp_bi_mean", "C mntp + bi + mean"), ("D_distill", "D distill only"),
         ("E_distill_contrast", "E distill -> contrastive"), ("F_distill_contrast_fn", "F E + fn masking")]
print(f"{'arm':28s} {'NanoBEIR nDCG@10':>17s} {'STS-B':>7s}")
for d, label in names:
    f = work / d / "eval.json"
    if f.exists():
        r = json.loads(f.read_text())
        print(f"{label:28s} {100*r['nanobeir_mean']:17.2f} {100*r['stsb_spearman']:7.2f}")
EOF
}

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(topup prep teacher mntp A B C D E F eval report)
for s in "${STAGES[@]}"; do
  echo "================ stage $s ================"
  "stage_$s"
done
