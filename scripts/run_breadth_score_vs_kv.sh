#!/usr/bin/env bash
# Direction-INDEPENDENT breadth: the one solid result is "native same-state
# score-matching beats K/V replacement". Broaden it from 3 references (mechanism
# study) toward a benchmark: 8 fresh styles x rotating prompts x seed 64, methods
# {kv_match, score_match}. Serial (1-wide) to coexist with any external GPU job.
# score_carrier is intentionally EXCLUDED: the strength sweep proved it inert, so
# it adds no evidence for the robust claim; add it back only if the paper keeps a
# carrier story.
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

# fresh styles (not 00/13/27); rotate content prompts to test generality
STYLE_PROMPT=("02 kp0" "07 kp1" "15 kp2" "22 kp3" "29 kp4" "34 kp0" "36 kp1" "39 kp2")
MODES=(kv_match score_match)
SEED=64

echo "[breadth] START $(date +%F_%H:%M:%S)  $(( ${#STYLE_PROMPT[@]} * ${#MODES[@]} )) runs, 1-wide"
for mode in "${MODES[@]}"; do
  for sp in "${STYLE_PROMPT[@]}"; do
    # shellcheck disable=SC2086
    set -- $sp; sid=$1; pid=$2
    log="$OUTDIR/${PREFIX}_${mode}_style${sid}_${pid}_seed${SEED}.log"
    "$PY" scripts/run_tpami_with_style_teacher.py \
        --style-id "$sid" --prompt-id "$pid" --seed "$SEED" \
        --style-mode "$mode" "${COMMON[@]}" --out-prefix "$PREFIX" > "$log" 2>&1
    echo "[breadth] DONE ${mode} style${sid} ${pid} exit=$? $(date +%H:%M:%S)"
  done
done
echo "[breadth] ALL_GEN_DONE $(date +%H:%M:%S)"

"$PY" scripts/build_matrix_manifest.py --prefix "$PREFIX" \
    --out-csv results/breadth_manifest.csv || { echo "[breadth] manifest FAILED"; exit 4; }
"$PY" scripts/evaluate_matched_teacher_outputs.py \
    --manifest-csv results/breadth_manifest.csv \
    --out-csv results/breadth_metrics.csv \
    --summary-json results/breadth_metrics_summary.json --device cuda \
    || { echo "[breadth] eval FAILED"; exit 5; }
"$PY" scripts/analyze_matched_matrix.py \
    --metrics-csv results/breadth_metrics.csv \
    --out-json results/breadth_aggregate.json \
    --expect-cells 8 | tee "$OUTDIR/aggregate_table.txt"
echo "[breadth] BREADTH_COMPLETE $(date +%F_%H:%M:%S)"
