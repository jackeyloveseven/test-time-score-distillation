#!/usr/bin/env bash
# Causal test of the frozen reverse process as an inference-time regularizer.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3

PY=/home/ubuntu/chengyanli/flux/bin/python3
NEED_MIB=${NEED_MIB:-24000}
SLEEP_S=${SLEEP_S:-30}
STYLES=(00 09 10 23 34)
OUTDIR=agent_runs/reverse_shield
mkdir -p "$OUTDIR" results/reverse_shield

wait_gpu() {
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    free=${free:-0}
    if [ "$free" -ge "$NEED_MIB" ]; then return 0; fi
    echo "[shield-gate] free=${free}MiB < ${NEED_MIB}MiB; sleep ${SLEEP_S}s" >&2
    sleep "$SLEEP_S"
  done
}

run_cell() {
  local setting=$1 rho=$2 tag=$3 sid=$4 iters step_size dynamic_flag prefix png
  if [ "$setting" = "standard" ]; then
    iters=5; step_size=0.08; dynamic_flag=""
  else
    iters=8; step_size=0.12; dynamic_flag="--disable-dynamic-iters"
  fi
  prefix="shield_${setting}_rho${tag}"
  png="results/diagnostic_${prefix}_ca_only_style${sid}_kp0_seed65.png"
  if [ -f "$png" ]; then echo "[shield-skip] $png"; return 0; fi
  wait_gpu
  "$PY" scripts/run_tpami_with_style_teacher.py \
    --style-id "$sid" --prompt-id kp0 --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1 \
    --pseudo-step-size "$step_size" --num-steps 25 --iters "$iters" \
    --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain \
    --without-prompt-mode full_style_prompt --pseudo-grad-normalization rms \
    --reverse-step-strength "$rho" $dynamic_flag \
    --main-model-root /mnt/local/klein9b-base \
    --teacher-model-root /mnt/local/klein9b-base \
    --vram-limit-gb 38 --main-vram-limit-gb 38 --teacher-vram-limit-gb 38 \
    --out-prefix "$prefix" >"$OUTDIR/${prefix}_style${sid}.log" 2>&1
  test -f "$png"
}

for setting in standard stress; do
  for spec in "1.0 100" "0.5 050" "0.0 000"; do
    set -- $spec; rho=$1; tag=$2
    for sid in "${STYLES[@]}"; do
      run_cell "$setting" "$rho" "$tag" "$sid" || exit 20
    done
  done
done

"$PY" scripts/reverse_shield_ablation.py inventory
"$PY" scripts/reverse_shield_ablation.py manifest --require-complete
"$PY" scripts/evaluate_matched_teacher_outputs.py \
  --manifest-csv results/reverse_shield/manifest.csv \
  --out-csv results/reverse_shield/metrics.csv \
  --summary-json results/reverse_shield/metrics_summary.json --device cuda
"$PY" scripts/reverse_shield_ablation.py diagnostics
"$PY" scripts/reverse_shield_ablation.py vlm
"$PY" scripts/reverse_shield_ablation.py aggregate \
  --metrics-csv results/reverse_shield/metrics.csv
