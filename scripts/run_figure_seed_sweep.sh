#!/usr/bin/env bash
# Seed sweep for the QUALITATIVE figures only (TSD Semantic + Global) so we can
# pick the best-looking seed per style. Quantitative benchmark stays seed 65.
# Parallel (GPU has ~143GB free); each TSD cell ~24GB -> up to 5 concurrent.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
STYLES=(00 02 03 05 07 09 10 12 16 20 23 27 30 34 37)
SEEDS=(202 777)
MAXJ=${MAXJ:-5}
OUTDIR=agent_runs/figure_seed_sweep
mkdir -p "$OUTDIR"

gen() {   # mode seed sid
  local mode=$1 seed=$2 sid=$3 prefix steps step_size
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
    --main-model-root /mnt/local/klein9b-base \
    --teacher-model-root /mnt/local/klein9b-base \
    --vram-limit-gb 38 --main-vram-limit-gb 38 --teacher-vram-limit-gb 38 \
    --out-prefix "$prefix" >"$OUTDIR/${prefix}_style${sid}_seed${seed}.log" 2>&1
  test -f "$png" && echo "[ok] $png" || echo "[FAIL] $png"
}
export -f gen; export PY

echo "[sweep] start $(date -u +%FT%TZ)  styles=${#STYLES[@]} seeds=${SEEDS[*]}"
for seed in "${SEEDS[@]}"; do
  for sid in "${STYLES[@]}"; do
    for mode in semantic global; do
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do sleep 5; done
      gen "$mode" "$seed" "$sid" &
    done
  done
done
wait
echo "[sweep] complete $(date -u +%FT%TZ)"
