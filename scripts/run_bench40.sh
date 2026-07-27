#!/usr/bin/env bash
# Complete the base-method (score_match) benchmark to ALL 40 available style refs.
# The grid already covers 15 styles (00,02,05,07,11,13,15,18,22,25,27,29,34,36,39);
# this runs the 25 REMAINING styles x rotating prompts kp0-kp4 x seed 64, score_match
# ONLY. Directly closes the "eval corpus too thin (11-15 styles)" AAAI gap flagged
# in docs/LOOP_PLAN.md. Direction-independent (A/C-aligned); NOT carrier/kv. Config
# byte-identical to breadth/breadthext so all score_match cells pool. Each run is
# headroom-GATED (>=NEED free) with OOM retry to survive external-job churn (proven
# pattern from rerun_sched_failed.sh). Idempotent (skips existing PNGs). 1-wide.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
OUTDIR=agent_runs/bench40
mkdir -p "$OUTDIR"
PREFIX=bench40
NEED=66000
MAX_TRIES=200

COMMON=(--num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1 --carrier-strength 0.75
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain)

# 25 remaining styles (40 total minus the 15 already in the grid)
STYLES=(01 03 04 06 08 09 10 12 14 16 17 19 20 21 23 24 26 28 30 31 32 33 35 37 38)
PROMPTS=(kp0 kp1 kp2 kp3 kp4)   # rotate across styles
SEED=${1:-64}                   # optional seed arg (default 64); 2nd seed for CIs

echo "[bench40] START $(date +%F_%H:%M:%S)  ${#STYLES[@]} score_match runs (seed ${SEED})"
i=0
for sid in "${STYLES[@]}"; do
  pid=${PROMPTS[$(( i % ${#PROMPTS[@]} ))]}
  i=$(( i + 1 ))
  png="results/${PREFIX}_score_match_style${sid}_${pid}_seed${SEED}.png"
  if [ -f "$png" ]; then echo "[bench40] SKIP existing $png"; continue; fi
  tries=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      log="$OUTDIR/${PREFIX}_score_match_style${sid}_${pid}_seed${SEED}.log"
      echo "[bench40] LAUNCH style${sid} ${pid} (free=${free}MiB) $(date +%H:%M:%S)"
      "$PY" scripts/run_tpami_with_style_teacher.py \
          --style-id "$sid" --prompt-id "$pid" --seed "$SEED" \
          --style-mode score_match "${COMMON[@]}" --out-prefix "$PREFIX" > "$log" 2>&1
      rc=$?
      if [ $rc -eq 0 ] && [ -f "$png" ]; then echo "[bench40] OK style${sid} ${pid} $(date +%H:%M:%S)"; break; fi
      echo "[bench40] FAIL style${sid} ${pid} rc=${rc} (likely OOM race) -> retry"
    fi
    tries=$(( tries + 1 ))
    if [ "$tries" -gt "$MAX_TRIES" ]; then echo "[bench40] TIMEOUT style${sid} ${pid}"; break; fi
    sleep 40
  done
done
echo "[bench40] ALL_GEN $(date +%H:%M:%S)"

"$PY" scripts/build_matrix_manifest.py --prefix "$PREFIX" \
    --out-csv "results/${PREFIX}_manifest.csv" || { echo "[bench40] manifest FAIL"; exit 4; }
"$PY" scripts/evaluate_matched_teacher_outputs.py \
    --manifest-csv "results/${PREFIX}_manifest.csv" \
    --out-csv "results/${PREFIX}_metrics.csv" \
    --summary-json "results/${PREFIX}_metrics_summary.json" --device cuda \
    || echo "[bench40] eval FAIL (rerun later)"
echo "[bench40] DONE $(date +%F_%H:%M:%S)"
