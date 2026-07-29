#!/usr/bin/env bash
# Generate the new content prompt kp5 ("A lighthouse on a rocky coast") for the
# 5 media-diverse styles {00,09,10,23,34}: native (sem+gst) + TSD (Semantic q25s
# / Global pg) at seeds {65,202}. For the 2-prompt main Figure 3 (panda|lighthouse).
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=agent_runs/kp5_gen
STYLES=(00 09 10 23 34)
SEEDS=(65 202)
MAXJ=4

# ---- (1) native baseline (one process, batches sem+gst x 5 styles) ----
ONLY="00:sem,00:gst,09:sem,09:gst,10:sem,10:gst,23:sem,23:gst,34:sem,34:gst"
echo "[kp5] native START $(date -u +%FT%TZ)"
$PY scripts/diagnostic_kleinedit_sysprompt_baseline.py --prompt-id kp5 --only "$ONLY" \
    > "$OUT/native_kp5.log" 2>&1 &
NATIVE_PID=$!

# ---- (2) TSD (parallel) ----
tsd() {  # mode seed sid
  local mode=$1 seed=$2 sid=$3 prefix steps step
  if [ "$mode" = semantic ]; then prefix=q25s; steps=25; step=0.05
  else prefix=pg_n8_s12_r; steps=8; step=0.12; fi
  local png="results/diagnostic_${prefix}_ca_only_style${sid}_kp5_seed${seed}.png"
  [ -f "$png" ] && { echo "[skip] $png"; return 0; }
  $PY scripts/run_tpami_with_style_teacher.py \
    --style-id "$sid" --prompt-id kp5 --seed "$seed" \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 \
    --pseudo-step-size "$step" --num-steps "$steps" --iters 5 \
    --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain \
    --without-prompt-mode full_style_prompt --pseudo-grad-normalization rms \
    --reverse-step-strength 1 \
    --main-model-root models/klein9b-base --teacher-model-root models/klein9b-base \
    --vram-limit-gb 38 --main-vram-limit-gb 38 --teacher-vram-limit-gb 38 \
    --out-prefix "$prefix" > "$OUT/${prefix}_style${sid}_seed${seed}.log" 2>&1
  test -f "$png" && echo "[ok] $png" || echo "[FAIL] $png"
}
export -f tsd; export PY OUT
echo "[kp5] TSD START $(date -u +%FT%TZ)"
for seed in "${SEEDS[@]}"; do for sid in "${STYLES[@]}"; do for mode in semantic global; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
  tsd "$mode" "$seed" "$sid" &
done; done; done
wait $NATIVE_PID
wait
echo "[kp5] ALL DONE $(date -u +%FT%TZ)"
