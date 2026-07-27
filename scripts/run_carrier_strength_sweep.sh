#!/usr/bin/env bash
# Decisive carrier diagnostic: does the content-affinity carrier beat plain
# score-matching at ANY strength? The matched matrix showed carrier@0.75 is a
# wash vs score_match (mean CSD 0.5935 vs 0.5993). Here we sweep carrier-strength
# in {0.5, 1.0} over the same 9 cells. If neither mean beats score_match's
# 0.5993, the carrier is confirmed inert/harmful across strengths (-> demote it).
# Serial (1-wide) to coexist with any external GPU job. score_match/kv baselines
# already exist from the matrix; here we only add the two carrier strengths.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
OUTDIR=agent_runs/sweep
mkdir -p "$OUTDIR"

COMMON=(--num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain)

PAIRS=("27 kp0" "00 kp1" "13 kp3")
SEEDS=(64 65 66)

for s in 0.5 1.0; do
  tag="cs${s//./}"   # cs05 / cs10
  echo "[sweep] === carrier-strength $s (prefix sweep_${tag}) $(date +%H:%M:%S) ==="
  for pair in "${PAIRS[@]}"; do
    # shellcheck disable=SC2086
    set -- $pair; sid=$1; pid=$2
    for seed in "${SEEDS[@]}"; do
      log="$OUTDIR/sweep_${tag}_style${sid}_${pid}_seed${seed}.log"
      "$PY" scripts/run_tpami_with_style_teacher.py \
          --style-id "$sid" --prompt-id "$pid" --seed "$seed" \
          --style-mode score_carrier "${COMMON[@]}" \
          --carrier-strength "$s" --out-prefix "sweep_${tag}" > "$log" 2>&1
      echo "[sweep] DONE ${tag} style${sid} ${pid} seed${seed} exit=$? $(date +%H:%M:%S)"
    done
  done
  "$PY" scripts/build_matrix_manifest.py --prefix "sweep_${tag}" \
      --out-csv "results/sweep_${tag}_manifest.csv"
  "$PY" scripts/evaluate_matched_teacher_outputs.py \
      --manifest-csv "results/sweep_${tag}_manifest.csv" \
      --out-csv "results/sweep_${tag}_metrics.csv" \
      --summary-json "results/sweep_${tag}_metrics_summary.json" --device cuda
  # print mean CSD for quick comparison against score_match 0.5993
  "$PY" - "results/sweep_${tag}_metrics.csv" "$s" <<'PYEOF'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1])))
ck=[c for c in rows[0] if 'csd' in c.lower()][0]
vals=[float(r[ck]) for r in rows]
print(f"[sweep] carrier-strength {sys.argv[2]}: mean CSD = {sum(vals)/len(vals):.4f} over {len(vals)} cells  (score_match baseline = 0.5993)")
PYEOF
done
echo "[sweep] SWEEP_COMPLETE $(date +%F_%H:%M:%S)"
