#!/usr/bin/env bash
# B2 smoke: ca_dmd inference (CA + DM with the REAL KleinFullFakeBackend online critic)
# vs ca_only, on style04/16. Tests whether the DM/fake half adds anything (our finding
# predicts ~nothing at inference). NOT the amortized student (that's days) — the CA+DM
# component test. Higher headroom gate (loads main+teacher+fake DiT).
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; NEED=90000
for sp in "04 kp2 65" "16 kp4 65"; do
  set -- $sp; sid=$1;pid=$2;seed=$3
  png="results/b2dmd_ca_dmd_style${sid}_${pid}_seed${seed}.png"
  [ -f "$png" ] && { echo "SKIP $sid"; continue; }
  t=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      echo "[B2] LAUNCH s$sid ca_dmd (free=$free) $(date +%H:%M:%S)"
      $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed $seed \
        --style-mode ca_dmd --cfg-scale 7 --lambda-ca 1.0 --lambda-dm 1.0 \
        --fake-score-backend klein_full_fake --fake-optimizer sgd --fake-lr 1e-3 --pseudo-step-size 0.05 \
        --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
        --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix b2dmd > agent_runs/b2_$sid.log 2>&1
      [ -f "$png" ] && { echo "[B2] OK s$sid"; break; }
      echo "[B2] FAIL s$sid"; tail -6 agent_runs/b2_$sid.log
    fi
    t=$((t+1)); [ $t -gt 120 ] && { echo "[B2] TIMEOUT s$sid"; break; }
    sleep 30
  done
done
echo "[B2] DONE $(date +%H:%M:%S)"
