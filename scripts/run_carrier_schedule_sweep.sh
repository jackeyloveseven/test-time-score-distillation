#!/usr/bin/env bash
# Stage 1 decisive diagnostic (docs/research/SYNTHESIS.md "Stage 1 -- Timestep-
# scheduled strength"): does scheduling the carrier's strength + loss weight over
# denoise progress (protect content early, push style late) revive score_carrier
# out of its current flat-strength near-wash vs plain score_match? Sweeps the new
# --carrier-schedule knob in {const,back,front,sqrt,cosine} at the established
# carrier-strength=0.75 operating point, over the same 3 matched cells used by
# run_matched_seed_matrix.sh / run_carrier_strength_sweep.sh. Seed 64 only
# (single-seed smoke -- widen to {64,65,66} only after this passes the eyeball).
# Serial (1-wide) to coexist with any external GPU job.
#
# *** DO NOT LAUNCH until `nvidia-smi` shows > 70 GB free headroom on the target
# GPU. A score_carrier run (score-type) needs ~63 GB; launching under that
# headroom risks OOM-colliding with whatever external job is currently using the
# GPU. ***
#
# NOTE ON METHOD LABELING (read before trusting the aggregate table): style_mode
# is fixed to score_carrier for every run here, but build_matrix_manifest.py
# labels each row's "method" from style_mode alone -- it cannot see
# --carrier-schedule. A single combined manifest would therefore collapse all 5
# schedules into one method="score_carrier" bucket (n=15), which is useless for
# comparing schedules against each other. To keep build_matrix_manifest.py /
# analyze_matched_matrix.py untouched, this script instead:
#   1. gives each schedule its own out-prefix (sched_<mode>) so output files
#      don't collide across schedules (they would if out-prefix were literally
#      "sched" for all, since style_mode never changes in the filename),
#   2. relabels the "method" column of each per-mode manifest CSV to the mode
#      name (const/back/front/sqrt/cosine) right after build_matrix_manifest --
#      only the "method" field is touched, nothing else,
#   3. runs evaluate_matched_teacher_outputs.py per mode (method flows through
#      unchanged from the relabeled manifest),
#   4. concatenates the 5 per-mode metrics CSVs and runs analyze_matched_matrix.py
#      ONCE on the combined CSV, so each schedule shows up as its own "method"
#      row (n=3 cells) with mean/std/95%CI on CSD, CLIP-T, DINO, Gram, layout-leak.
# Caveat: analyze_matched_matrix.py's paired win-rate block is hardcoded to
# compare against methods literally named kv_match/score_match; none of our
# method labels match those, so the win-rate columns will print "n/a" -- that is
# expected, not a bug. The per-method mean/std/CI table + coverage gate is the
# useful output and is what the Stage-1 decision gate (SYNTHESIS.md) should read:
# does any schedule (esp. back/sqrt/cosine, on textured styles) beat const on
# CSD with lower dino_layout_leak?
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
OUTDIR=agent_runs/carrier_schedule_sweep
mkdir -p "$OUTDIR"

COMMON=(--num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1 --carrier-strength 0.75
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain)

PAIRS=("27 kp0" "00 kp1" "13 kp3")
SEED=64
MODES=(const back front sqrt cosine)

for mode in "${MODES[@]}"; do
  echo "[sched-sweep] === carrier-schedule $mode (prefix sched_${mode}) $(date +%H:%M:%S) ==="
  for pair in "${PAIRS[@]}"; do
    # shellcheck disable=SC2086
    set -- $pair; sid=$1; pid=$2
    log="$OUTDIR/sched_${mode}_style${sid}_${pid}_seed${SEED}.log"
    "$PY" scripts/run_tpami_with_style_teacher.py \
        --style-id "$sid" --prompt-id "$pid" --seed "$SEED" \
        --style-mode score_carrier "${COMMON[@]}" \
        --carrier-schedule "$mode" --out-prefix "sched_${mode}" > "$log" 2>&1
    echo "[sched-sweep] DONE ${mode} style${sid} ${pid} seed${SEED} exit=$? $(date +%H:%M:%S)"
  done

  "$PY" scripts/build_matrix_manifest.py --prefix "sched_${mode}" \
      --out-csv "results/sched_${mode}_manifest.csv"

  # Relabel the manifest's "method" column (currently "score_carrier" for every
  # row, since style_mode never changes across schedules) to the schedule mode
  # name, so the downstream metrics CSV -- and the combined analyze call below --
  # can tell the five schedules apart. Only the "method" field is touched.
  "$PY" - "results/sched_${mode}_manifest.csv" "$mode" <<'PYEOF'
import csv, sys
path, mode = sys.argv[1], sys.argv[2]
with open(path, newline="") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames
    rows = list(reader)
for row in rows:
    row["method"] = mode
with open(path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[sched-sweep] relabeled method -> {mode} in {path} ({len(rows)} rows)")
PYEOF

  "$PY" scripts/evaluate_matched_teacher_outputs.py \
      --manifest-csv "results/sched_${mode}_manifest.csv" \
      --out-csv "results/sched_${mode}_metrics.csv" \
      --summary-json "results/sched_${mode}_metrics_summary.json" --device cuda
done

# Merge the 5 per-mode metrics CSVs (method column already relabeled per mode
# above) into one combined CSV so analyze_matched_matrix.py only needs one call.
"$PY" - "${MODES[@]}" <<'PYEOF'
import csv, sys
modes = sys.argv[1:]
out_path = "results/sched_metrics_combined.csv"
fieldnames = None
all_rows = []
for mode in modes:
    path = f"results/sched_{mode}_metrics.csv"
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        all_rows.extend(list(reader))
with open(out_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"[sched-sweep] wrote combined metrics -> {out_path} ({len(all_rows)} rows, {len(modes)} methods)")
PYEOF

"$PY" scripts/analyze_matched_matrix.py \
    --metrics-csv results/sched_metrics_combined.csv \
    --out-json results/sched_aggregate.json \
    --expect-cells 3 | tee "$OUTDIR/aggregate_table.txt"

echo "[sched-sweep] SWEEP_COMPLETE $(date +%F_%H:%M:%S)"
