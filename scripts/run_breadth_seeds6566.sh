#!/usr/bin/env bash
# Multi-seed extension of the breadth benchmark: seeds 65 & 66 for the same 8
# fresh styles x {kv_match, score_match}, SAME out-prefix "breadth" so the final
# manifest spans all three seeds (24 cells/method). Serial (1-wide) to coexist
# with the external ~56 GB worker.py job. Rebuilds + re-evaluates the full breadth
# set at the end and aggregates with --expect-cells 24.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
PREFIX=breadth
OUTDIR=agent_runs/breadth
mkdir -p "$OUTDIR"

COMMON=(--num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1 --carrier-strength 0.75
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain)

STYLE_PROMPT=("02 kp0" "07 kp1" "15 kp2" "22 kp3" "29 kp4" "34 kp0" "36 kp1" "39 kp2")
MODES=(kv_match score_match)
SEEDS=(65 66)

echo "[breadth2] START $(date +%F_%H:%M:%S)  $(( ${#STYLE_PROMPT[@]} * ${#MODES[@]} * ${#SEEDS[@]} )) runs, 1-wide"
for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    for sp in "${STYLE_PROMPT[@]}"; do
      # shellcheck disable=SC2086
      set -- $sp; sid=$1; pid=$2
      log="$OUTDIR/${PREFIX}_${mode}_style${sid}_${pid}_seed${seed}.log"
      "$PY" scripts/run_tpami_with_style_teacher.py \
          --style-id "$sid" --prompt-id "$pid" --seed "$seed" \
          --style-mode "$mode" "${COMMON[@]}" --out-prefix "$PREFIX" > "$log" 2>&1
      echo "[breadth2] DONE ${mode} style${sid} ${pid} seed${seed} exit=$? $(date +%H:%M:%S)"
    done
  done
done
echo "[breadth2] ALL_GEN_DONE $(date +%H:%M:%S)"

"$PY" scripts/build_matrix_manifest.py --prefix "$PREFIX" \
    --out-csv results/breadth_manifest.csv || { echo "[breadth2] manifest FAILED"; exit 4; }
"$PY" scripts/evaluate_matched_teacher_outputs.py \
    --manifest-csv results/breadth_manifest.csv \
    --out-csv results/breadth_metrics.csv \
    --summary-json results/breadth_metrics_summary.json --device cuda \
    || { echo "[breadth2] eval FAILED"; exit 5; }
"$PY" scripts/analyze_matched_matrix.py \
    --metrics-csv results/breadth_metrics.csv \
    --out-json results/breadth_aggregate.json \
    --expect-cells 24 | tee "$OUTDIR/aggregate_table_multiseed.txt"
echo "[breadth2] BREADTH_MULTISEED_COMPLETE $(date +%F_%H:%M:%S)"
