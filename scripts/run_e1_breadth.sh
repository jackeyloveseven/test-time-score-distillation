#!/usr/bin/env bash
# E1: TSD (ca_only cfg7) breadth main-results run — 40 styles x seeds {64,65}, one rotating
# prompt/style. = the paper's main table + regression check + B1-free StyleBench dataset.
# Headroom-gated (external GPU jobs never killed); skip-existing = resumable checkpoint.
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; NEED=64000
COMMON="--style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 --pseudo-step-size 0.05 --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix e1breadth"
PROMPTS=(kp0 kp1 kp2 kp3 kp4)
echo "[e1] START $(date +%F_%H:%M:%S)"
for seed in 64 65; do
 for i in $(seq 0 39); do
  sid=$(printf "%02d" $i); pid=${PROMPTS[$(( i % 5 ))]}
  png="results/e1breadth_ca_only_style${sid}_${pid}_seed${seed}.png"
  [ -f "$png" ] && { echo "[e1] SKIP s$sid seed$seed"; continue; }
  t=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      echo "[e1] LAUNCH s$sid $pid seed$seed (free=$free) $(date +%H:%M:%S)"
      $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed $seed $COMMON > agent_runs/e1_${sid}_${seed}.log 2>&1
      [ -f "$png" ] && { echo "[e1] OK s$sid seed$seed $(date +%H:%M:%S)"; break; }
      echo "[e1] FAIL s$sid seed$seed (retry)"
    fi
    t=$((t+1)); [ $t -gt 240 ] && { echo "[e1] TIMEOUT s$sid seed$seed"; break; }
    sleep 30
  done
 done
done
echo "[e1] ALL GEN DONE $(date +%F_%H:%M:%S). Next: StyleBench-score results/e1breadth_* (E8)."
