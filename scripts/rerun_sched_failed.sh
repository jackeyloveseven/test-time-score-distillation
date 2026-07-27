#!/usr/bin/env bash
# Re-run the 3 carrier-schedule modes that OOM'd in run_carrier_schedule_sweep.sh
# (back/front/sqrt) — the failure was transient external-job memory contention
# during model load, NOT a code issue (const/cosine + the manual smoke all
# succeeded with identical config). Each cell is headroom-GATED: we poll
# nvidia-smi and only launch when free >= NEED MiB, with OOM retry, so runs slot
# into windows between the churny external jobs. Config is byte-identical to the
# sweep (full_cuda, no vram limit) so these 9 cells are directly comparable to the
# already-good const/cosine cells — no memory-config confound. Skips any cell whose
# PNG already exists (idempotent / resumable).
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
OUTDIR=agent_runs/carrier_schedule_sweep
mkdir -p "$OUTDIR"
NEED=68000          # MiB free required before launching a score-type run (~63GB peak + buffer)
MAX_TRIES=160       # ~2h of 45s polls per cell, hard cap

COMMON=(--num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1 --carrier-strength 0.75
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain)

PAIRS=("27 kp0" "00 kp1" "13 kp3")
MODES=(back front sqrt)

echo "[rerun] START $(date +%F_%H:%M:%S)  NEED=${NEED}MiB"
for mode in "${MODES[@]}"; do
  for pair in "${PAIRS[@]}"; do
    # shellcheck disable=SC2086
    set -- $pair; sid=$1; pid=$2
    png="results/sched_${mode}_score_carrier_style${sid}_${pid}_seed64.png"
    if [ -f "$png" ]; then echo "[rerun] SKIP existing $png"; continue; fi
    tries=0
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
      free=${free:-0}
      if [ "$free" -ge "$NEED" ]; then
        log="$OUTDIR/sched_${mode}_style${sid}_${pid}_seed64.log"
        echo "[rerun] LAUNCH ${mode} style${sid} ${pid} (free=${free}MiB) $(date +%H:%M:%S)"
        "$PY" scripts/run_tpami_with_style_teacher.py \
            --style-id "$sid" --prompt-id "$pid" --seed 64 \
            --style-mode score_carrier "${COMMON[@]}" \
            --carrier-schedule "$mode" --out-prefix "sched_${mode}" > "$log" 2>&1
        rc=$?
        if [ $rc -eq 0 ] && [ -f "$png" ]; then echo "[rerun] OK ${mode} ${sid} ${pid} $(date +%H:%M:%S)"; break; fi
        echo "[rerun] FAIL ${mode} ${sid} ${pid} rc=${rc} (likely OOM race) -> retry"
      fi
      tries=$((tries+1))
      if [ "$tries" -gt "$MAX_TRIES" ]; then echo "[rerun] TIMEOUT ${mode} ${sid} ${pid}"; break; fi
      sleep 45
    done
  done
done
echo "[rerun] ALL_GEN $(date +%H:%M:%S)"

# Rebuild manifest + evaluate for the 3 re-run modes (method relabel like the sweep).
for mode in "${MODES[@]}"; do
  "$PY" scripts/build_matrix_manifest.py --prefix "sched_${mode}" \
      --out-csv "results/sched_${mode}_manifest.csv" || { echo "[rerun] manifest $mode FAIL"; continue; }
  "$PY" - "results/sched_${mode}_manifest.csv" "$mode" <<'PYEOF'
import csv, sys
path, mode = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path, newline="")))
fn = rows[0].keys() if rows else []
for r in rows: r["method"] = mode
w = csv.DictWriter(open(path, "w", newline=""), fieldnames=list(fn)); w.writeheader(); w.writerows(rows)
print(f"[rerun] relabeled method -> {mode} ({len(rows)} rows)")
PYEOF
  "$PY" scripts/evaluate_matched_teacher_outputs.py \
      --manifest-csv "results/sched_${mode}_manifest.csv" \
      --out-csv "results/sched_${mode}_metrics.csv" \
      --summary-json "results/sched_${mode}_metrics_summary.json" --device cuda \
      || echo "[rerun] eval $mode FAIL (rerun eval later)"
done
echo "[rerun] DONE $(date +%F_%H:%M:%S)"
