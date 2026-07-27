set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SID=16; PID=kp4; SEED=65; GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
# label | lambda_ca norm STATS_LOSS STATS_WEIGHT step iters
CONFIGS=(
  "casw1|1 rms sw 1 0.05 5"
  "casw3|1 rms sw 3 0.05 5"
  "casw8|1 rms sw 8 0.05 5"
  "ptoksw3|1 rms_token sw 3 0.05 5"
  "ptokstep|1 rms_token none 0 0.15 8"
)
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"; set -- $rest; lca=$1; norm=$2; sl=$3; sw=$4; step=$5; it=$6
  pfx="r2_${label}"; png="results/${pfx}_ca_only_style${SID}_${PID}_seed${SEED}.png"
  [ -f "$png" ] && { echo "SKIP $label"; continue; }
  free=$(wait_gpu) || { echo "[R2] TIMEOUT $label skip"; continue; }
  echo "[R2] $label lca=$lca norm=$norm stats=$sl w=$sw step=$step it=$it free=$free $(date +%H:%M:%S)"
  env_stats=""; [ "$sl" != "none" ] && env_stats="STATS_LOSS=$sl STATS_WEIGHT=$sw"
  env $env_stats \
  $PY scripts/run_tpami_with_style_teacher.py --style-id $SID --prompt-id $PID --seed $SEED \
    --style-mode ca_only --cfg-scale 7 --lambda-ca $lca --pseudo-step-size $step \
    --num-steps 25 --iters $it --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain --pseudo-grad-normalization $norm \
    --out-prefix "$pfx" > "agent_runs/r2_${label}.log" 2>&1
  [ -f "$png" ] && echo "[R2] OK $label" || { echo "[R2] FAIL $label"; tail -6 "agent_runs/r2_${label}.log"; }
done
echo "[R2] done $(date +%H:%M:%S)"
