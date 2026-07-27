#!/usr/bin/env bash
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/miniconda3/envs/diffsynth/bin/python; NEED=64000
gate(){ local t=0; while true; do local f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$NEED" ] && return 0; t=$((t+1)); [ $t -gt 120 ] && return 1; sleep 30; done; }
echo "[B1c] STEP3 train (token_mlp, grid-agnostic) $(date +%H:%M:%S)"
$PY scripts/train_score_residual_student.py --manifest results/score_residual_ca.jsonl \
  --out-dir results/student_ca --epochs 100 --model-type token_mlp --hidden-dim 384 \
  --prediction-mode residual --device cuda 2>&1 | tail -14
echo "[B1c] ckpt: $(ls results/student_ca/*.pt 2>/dev/null)"
[ -f results/student_ca/student_best.pt ] || { echo "[B1c] NO CKPT, abort"; exit 1; }
echo "[B1c] STEP4 eval $(date +%H:%M:%S)"
gate && $PY scripts/evaluate_score_residual_student.py --manifest results/score_residual_ca.jsonl \
  --checkpoint results/student_ca/student_best.pt --out-prefix ca_student \
  --student-scales 0,3,6,9 2>&1 | tail -14
echo "[B1c] DONE $(date +%H:%M:%S)"
