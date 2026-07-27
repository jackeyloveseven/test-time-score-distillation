#!/usr/bin/env bash
# CA-driver (ca_only cfg7) across ALL 40 styles = (1) teacher breadth/regression check
# vs score_match, (2) the B1 supervised-distillation dataset. Headroom-gated (external
# jobs present, never killed); checkpointed by skip-existing so a stall loses nothing.
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; NEED=64000
COMMON="--style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 --pseudo-step-size 0.05 --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix cabreadth"
PROMPTS=(kp0 kp1 kp2 kp3 kp4)
echo "[cabreadth] START $(date +%F_%H:%M:%S)"
for i in $(seq 0 39); do
  sid=$(printf "%02d" $i); pid=${PROMPTS[$(( i % 5 ))]}; seed=64
  png="results/cabreadth_ca_only_style${sid}_${pid}_seed${seed}.png"
  [ -f "$png" ] && { echo "[cabreadth] SKIP $sid"; continue; }
  t=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      echo "[cabreadth] LAUNCH s$sid $pid (free=$free) $(date +%H:%M:%S)"
      $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed $seed $COMMON > agent_runs/cabreadth_$sid.log 2>&1
      [ -f "$png" ] && { echo "[cabreadth] OK s$sid $(date +%H:%M:%S)"; break; }
      echo "[cabreadth] FAIL s$sid (retry)"
    fi
    t=$((t+1)); [ $t -gt 120 ] && { echo "[cabreadth] TIMEOUT s$sid"; break; }
    sleep 30
  done
done
echo "[cabreadth] ALL DONE $(date +%F_%H:%M:%S)"
