#!/usr/bin/env bash
# Serial (1-wide) rerun of the score-type matrix cells so they coexist safely
# with any external GPU job. Each score-type run peaks ~63 GB; serial keeps
# 63 + external headroom well under 143 GB. kv_match cells (already done at
# 2-wide, 34 GB) are NOT rerun. After gen, builds the full 27-cell manifest
# (scans ALL matrix_* summaries incl. the 9 kv) and evaluates + aggregates.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
PREFIX=matrix
OUTDIR=agent_runs/matrix
mkdir -p "$OUTDIR"

COMMON=(--num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1 --carrier-strength 0.75
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain)

MODES=(score_match score_carrier)
PAIRS=("27 kp0" "00 kp1" "13 kp3")
SEEDS=(64 65 66)

echo "[serial] START $(date +%F_%H:%M:%S)  18 score-type runs, 1-wide"
for mode in "${MODES[@]}"; do
  for pair in "${PAIRS[@]}"; do
    # shellcheck disable=SC2086
    set -- $pair; sid=$1; pid=$2
    for seed in "${SEEDS[@]}"; do
      log="$OUTDIR/${PREFIX}_${mode}_style${sid}_${pid}_seed${seed}.log"
      "$PY" scripts/run_tpami_with_style_teacher.py \
          --style-id "$sid" --prompt-id "$pid" --seed "$seed" \
          --style-mode "$mode" "${COMMON[@]}" --out-prefix "$PREFIX" > "$log" 2>&1
      echo "[serial] DONE ${mode} style${sid} ${pid} seed${seed} exit=$? $(date +%H:%M:%S)"
    done
  done
done
echo "[serial] ALL_GEN_DONE $(date +%H:%M:%S)"

"$PY" scripts/build_matrix_manifest.py --prefix "$PREFIX" \
    --out-csv results/matrix_manifest.csv || { echo "[serial] manifest FAILED"; exit 4; }
"$PY" scripts/evaluate_matched_teacher_outputs.py \
    --manifest-csv results/matrix_manifest.csv \
    --out-csv results/matrix_metrics.csv \
    --summary-json results/matrix_metrics_summary.json \
    --device cuda || { echo "[serial] eval FAILED"; exit 5; }
"$PY" scripts/analyze_matched_matrix.py \
    --metrics-csv results/matrix_metrics.csv \
    --out-json results/matrix_aggregate.json | tee "$OUTDIR/aggregate_table.txt"
echo "[serial] MATRIX_COMPLETE $(date +%F_%H:%M:%S)"
