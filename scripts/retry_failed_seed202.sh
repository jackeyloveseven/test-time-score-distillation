#!/usr/bin/env bash
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
OUTDIR=agent_runs/figure_seed_sweep
gen() {  # mode sid  (seed fixed 202)
  local mode=$1 sid=$2 seed=202 prefix steps step_size
  if [ "$mode" = semantic ]; then prefix=q25s; steps=25; step_size=0.05
  else prefix=pg_n8_s12_r; steps=8; step_size=0.12; fi
  local png="results/diagnostic_${prefix}_ca_only_style${sid}_kp0_seed${seed}.png"
  [ -f "$png" ] && { echo "[skip] $png"; return 0; }
  "$PY" scripts/run_tpami_with_style_teacher.py \
    --style-id "$sid" --prompt-id kp0 --seed "$seed" \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 \
    --pseudo-step-size "$step_size" --num-steps "$steps" --iters 5 \
    --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain \
    --without-prompt-mode full_style_prompt --pseudo-grad-normalization rms \
    --reverse-step-strength 1 \
    --main-model-root models/klein9b-base \
    --teacher-model-root models/klein9b-base \
    --vram-limit-gb 38 --main-vram-limit-gb 38 --teacher-vram-limit-gb 38 \
    --out-prefix "$prefix" >"$OUTDIR/${prefix}_style${sid}_seed${seed}.retry.log" 2>&1
  test -f "$png" && echo "[ok] $png" || echo "[FAIL] $png"
}
echo "[retry] start $(date -u +%FT%TZ)"
gen semantic 05
gen semantic 09
gen semantic 10
gen semantic 12
gen global   03
gen global   10
echo "[retry] complete $(date -u +%FT%TZ)"
