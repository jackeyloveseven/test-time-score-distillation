#!/usr/bin/env bash
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; NEED=66000
for sp in "14 kp3 65" "21 kp3 64" "34 kp0 64"; do
  set -- $sp; sid=$1;pid=$2;seed=$3
  png="results/dmsched_ca_dmd_style${sid}_${pid}_seed${seed}.png"
  [ -f "$png" ] && { echo "SKIP $sid"; continue; }
  t=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      echo "[dmsched] LAUNCH s$sid front (free=$free) $(date +%H:%M:%S)"
      $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed $seed \
        --style-mode ca_dmd --dm-schedule front --cfg-scale 7 --lambda-ca 1.0 --lambda-dm 1.0 \
        --fake-score-backend klein_full_fake --fake-optimizer sgd --fake-lr 1e-3 --pseudo-step-size 0.05 \
        --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
        --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix dmsched > agent_runs/dmsched_$sid.log 2>&1
      [ -f "$png" ] && { echo "[dmsched] OK s$sid $(date +%H:%M:%S)"; break; }
      echo "[dmsched] FAIL s$sid"; tail -5 agent_runs/dmsched_$sid.log
    fi
    t=$((t+1)); [ $t -gt 120 ] && { echo "[dmsched] TIMEOUT s$sid"; break; }
    sleep 30
  done
done
echo "[dmsched] ALL DONE $(date +%H:%M:%S)"
