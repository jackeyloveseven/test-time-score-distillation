#!/usr/bin/env bash
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/miniconda3/envs/diffsynth/bin/python; NEED=64000
gate(){ local t=0; while true; do local f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$NEED" ] && return 0; t=$((t+1)); [ $t -gt 120 ] && return 1; sleep 30; done; }
echo "[B1b] STEP2 manifest $(date +%H:%M:%S)"
$PY scripts/build_score_residual_manifest.py --field-dumps 'results/field_dumps/ca_smoke_fields/*.pt' \
  --teacher-kind native_delta_score --out-prefix score_residual_ca --results-dir results 2>&1 | tail -8
echo "[B1b] manifest rows: $(wc -l < results/score_residual_ca.jsonl 2>/dev/null || echo 0)"
[ -s results/score_residual_ca.jsonl ] || { echo "[B1b] MANIFEST EMPTY, abort"; exit 1; }
echo "[B1b] STEP3 train $(date +%H:%M:%S)"
$PY scripts/train_score_residual_student.py --manifest results/score_residual_ca.jsonl \
  --out-dir results/student_ca --epochs 80 --model-type token_mixer --hidden-dim 384 \
  --prediction-mode residual --device cuda 2>&1 | tail -12
echo "[B1b] STEP4 eval $(date +%H:%M:%S)"
gate && $PY scripts/evaluate_score_residual_student.py --manifest results/score_residual_ca.jsonl \
  --checkpoint results/student_ca/student_best.pt --out-prefix ca_student \
  --student-scales 0,3,6,9 2>&1 | tail -12
echo "[B1b] DONE $(date +%H:%M:%S)"
