#!/usr/bin/env bash
# B1 smoke: distill the CA-driver's per-step direction (native_delta_score) into the
# amortized token-mixer student, then inject at inference. Minimal: 4 styles x 3 prompts.
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/miniconda3/envs/diffsynth/bin/python; NEED=64000
gate(){ local t=0; while true; do local f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$NEED" ] && return 0; t=$((t+1)); [ $t -gt 120 ] && return 1; sleep 30; done; }
echo "[B1] === STEP 1 dump fields === $(date +%H:%M:%S)"
gate && $PY scripts/probe_native_edit_teacher.py --style-ids 04,16,17,27 --prompt-ids kp0,kp2,kp4 \
  --num-steps 12 --seed 65 --dump-fields --dump-latents \
  --output-prefix results/field_dumps/ca_smoke 2>&1 | tail -8
echo "[B1] dumps: $(ls results/field_dumps/ca_smoke*.pt 2>/dev/null | wc -l)"
echo "[B1] === STEP 2 manifest (teacher-kind native_delta_score) === $(date +%H:%M:%S)"
$PY scripts/build_score_residual_manifest.py --field-dumps 'results/field_dumps/ca_smoke*.pt' \
  --teacher-kind native_delta_score --out-prefix score_residual_ca --results-dir results 2>&1 | tail -6
echo "[B1] manifest rows: $(wc -l < results/score_residual_ca.jsonl 2>/dev/null || echo 0)"
echo "[B1] === STEP 3 train student === $(date +%H:%M:%S)"
$PY scripts/train_score_residual_student.py --manifest results/score_residual_ca.jsonl \
  --out-dir results/student_ca --epochs 80 --model-type token_mixer --hidden-dim 384 \
  --prediction-mode residual --device cuda 2>&1 | tail -10
echo "[B1] === STEP 4 eval (inject at scales) === $(date +%H:%M:%S)"
gate && $PY scripts/evaluate_score_residual_student.py --manifest results/score_residual_ca.jsonl \
  --checkpoint results/student_ca/student_best.pt --out-prefix ca_student \
  --student-scales 0,3,6,9 2>&1 | tail -10
echo "[B1] DONE $(date +%H:%M:%S)"
