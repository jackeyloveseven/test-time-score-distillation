#!/usr/bin/env bash
# Paired 40-style benchmark: TSD Semantic / Global vs the same native MMDiT edit path.
# Reuses all seed-65 outputs and generates only missing cells for seeds 64 and 66.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3

PY=/home/ubuntu/chengyanli/flux/bin/python3
SEEDS=(64 66)
STYLES=($(seq -w 0 39))
TSD_NEED_MIB=${TSD_NEED_MIB:-24000}
NATIVE_NEED_MIB=${NATIVE_NEED_MIB:-45000}
SLEEP_S=${SLEEP_S:-30}
OUTDIR=agent_runs/benchmark_multiseed
mkdir -p "$OUTDIR" results/benchmark_multiseed

wait_gpu() {
  local need=$1 tag=$2 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    free=${free:-0}
    if [ "$free" -ge "$need" ]; then
      echo "[gate] $tag free=${free}MiB >= ${need}MiB" >&2
      return 0
    fi
    echo "[gate] $tag free=${free}MiB < ${need}MiB; sleep ${SLEEP_S}s" >&2
    sleep "$SLEEP_S"
  done
}

run_tsd_cell() {
  local method=$1 seed=$2 sid=$3 prefix steps step_size
  if [ "$method" = "semantic" ]; then
    prefix=q25s; steps=25; step_size=0.05
  else
    prefix=pg_n8_s12_r; steps=8; step_size=0.12
  fi
  local png="results/diagnostic_${prefix}_ca_only_style${sid}_kp0_seed${seed}.png"
  local log="$OUTDIR/${prefix}_style${sid}_seed${seed}.log"
  if [ -f "$png" ]; then
    echo "[skip] $png"
    return 0
  fi
  wait_gpu "$TSD_NEED_MIB" "TSD-${method} style${sid} seed${seed}"
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
    --out-prefix "$prefix" >"$log" 2>&1
  test -f "$png"
}

echo "[benchmark] start $(date -u +%FT%TZ)"
for seed in "${SEEDS[@]}"; do
  for sid in "${STYLES[@]}"; do
    run_tsd_cell semantic "$seed" "$sid" || exit 10
    run_tsd_cell global "$seed" "$sid" || exit 11
  done
done

for seed in "${SEEDS[@]}"; do
  wait_gpu "$NATIVE_NEED_MIB" "native MMDiT seed${seed}"
  "$PY" scripts/diagnostic_kleinedit_sysprompt_baseline.py --seed "$seed" \
    >"$OUTDIR/native_seed${seed}.log" 2>&1 || exit 12
done

"$PY" scripts/benchmark_multiseed.py inventory
"$PY" scripts/benchmark_multiseed.py manifest --require-complete
"$PY" scripts/evaluate_matched_teacher_outputs.py \
  --manifest-csv results/benchmark_multiseed/manifest.csv \
  --out-csv results/benchmark_multiseed/metrics.csv \
  --summary-json results/benchmark_multiseed/metrics_summary.json \
  --device cuda
"$PY" scripts/benchmark_multiseed.py aggregate \
  --metrics-csv results/benchmark_multiseed/metrics.csv
echo "[benchmark] complete $(date -u +%FT%TZ)"
