#!/usr/bin/env bash
# Direction-INDEPENDENT breadth extension for the BASE method only (score_match) —
# strengthens the "native score residual transfers across many styles" table under
# any paper framing (A/B/C). 4 NEW styles not yet in the grid x rotating prompts x
# seeds {64,65} = 8 runs, score_match ONLY (K/V dropped, carrier dead — not run).
# Each run is headroom-GATED (>= NEED MiB free) with OOM retry so it survives the
# churny external GPU jobs (same pattern as scripts/rerun_sched_failed.sh). Config
# byte-identical to the existing breadth grid so new cells pool with results/
# breadth_*. Idempotent (skips existing PNGs). Prefix "breadthext" keeps this batch
# self-contained; pool score_match cells across breadth + breadthext for the table.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
OUTDIR=agent_runs/breadthext
mkdir -p "$OUTDIR"
PREFIX=breadthext
NEED=68000
MAX_TRIES=160

COMMON=(--num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1 --carrier-strength 0.75
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain)

STYLE_PROMPT=("05 kp0" "11 kp1" "18 kp2" "25 kp3")
SEEDS=(64 65)

echo "[breadthext] START $(date +%F_%H:%M:%S)  $(( ${#STYLE_PROMPT[@]} * ${#SEEDS[@]} )) score_match runs"
for seed in "${SEEDS[@]}"; do
  for sp in "${STYLE_PROMPT[@]}"; do
    # shellcheck disable=SC2086
    set -- $sp; sid=$1; pid=$2
    png="results/${PREFIX}_score_match_style${sid}_${pid}_seed${seed}.png"
    if [ -f "$png" ]; then echo "[breadthext] SKIP existing $png"; continue; fi
    tries=0
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
      free=${free:-0}
      if [ "$free" -ge "$NEED" ]; then
        log="$OUTDIR/${PREFIX}_score_match_style${sid}_${pid}_seed${seed}.log"
        echo "[breadthext] LAUNCH style${sid} ${pid} seed${seed} (free=${free}MiB) $(date +%H:%M:%S)"
        "$PY" scripts/run_tpami_with_style_teacher.py \
            --style-id "$sid" --prompt-id "$pid" --seed "$seed" \
            --style-mode score_match "${COMMON[@]}" --out-prefix "$PREFIX" > "$log" 2>&1
        rc=$?
        if [ $rc -eq 0 ] && [ -f "$png" ]; then echo "[breadthext] OK style${sid} ${pid} seed${seed} $(date +%H:%M:%S)"; break; fi
        echo "[breadthext] FAIL style${sid} ${pid} seed${seed} rc=${rc} (likely OOM race) -> retry"
      fi
      tries=$((tries+1))
      if [ "$tries" -gt "$MAX_TRIES" ]; then echo "[breadthext] TIMEOUT style${sid} ${pid} seed${seed}"; break; fi
      sleep 45
    done
  done
done
echo "[breadthext] ALL_GEN $(date +%H:%M:%S)"

"$PY" scripts/build_matrix_manifest.py --prefix "$PREFIX" \
    --out-csv "results/${PREFIX}_manifest.csv" || { echo "[breadthext] manifest FAIL"; exit 4; }
"$PY" scripts/evaluate_matched_teacher_outputs.py \
    --manifest-csv "results/${PREFIX}_manifest.csv" \
    --out-csv "results/${PREFIX}_metrics.csv" \
    --summary-json "results/${PREFIX}_metrics_summary.json" --device cuda \
    || echo "[breadthext] eval FAIL (rerun later)"
echo "[breadthext] DONE $(date +%F_%H:%M:%S)"
