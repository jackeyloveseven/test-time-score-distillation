set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SID=16; PID=kp4; SEED=65; GATE=85000
wait_gpu () {  # wait for a real window; return 1 (skip, no OOM) if none in ~4h
  local t=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
    [ "$free" -ge "$GATE" ] && { echo "$free"; return 0; }
    t=$((t+1)); [ $t -gt 960 ] && return 1
    sleep 15
  done
}
CONFIGS=("puresw10 0 sw 10" "puresw50 0 sw 50" "puresw200 0 sw 200" "casw50 1 sw 50" "puregram50 0 gram 50")
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; label=$1; lca=$2; sl=$3; sw=$4
  pfx="statsprobe_${label}"; png="results/${pfx}_ca_only_style${SID}_${PID}_seed${SEED}.png"
  [ -f "$png" ] && { echo "SKIP $label (exists)"; continue; }
  free=$(wait_gpu) || { echo "[SP] TIMEOUT $label — skip (no OOM)"; continue; }
  echo "[SP] $label lca=$lca $sl w=$sw free=$free $(date +%H:%M:%S)"
  STATS_LOSS="$sl" STATS_WEIGHT="$sw" \
  $PY scripts/run_tpami_with_style_teacher.py --style-id $SID --prompt-id $PID --seed $SEED \
    --style-mode ca_only --cfg-scale 7 --lambda-ca $lca --pseudo-step-size 0.05 \
    --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix "$pfx" \
    > "agent_runs/sp_${label}.log" 2>&1
  [ -f "$png" ] && echo "[SP] OK $label" || { echo "[SP] FAIL $label"; tail -8 "agent_runs/sp_${label}.log"; }
done
echo "[SP] done $(date +%H:%M:%S)"
