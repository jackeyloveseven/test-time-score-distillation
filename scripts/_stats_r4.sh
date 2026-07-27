set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
# label | SID PID STYLED_DENOISE lambda_ca norm STATS_LOSS STATS_WEIGHT
CONFIGS=(
  "s04_sd4|04 kp2 4 1 rms_token none 0"
  "s14_sd4|14 kp0 4 1 rms_token none 0"
  "s16_sd4gram3|16 kp4 4 1 rms_token gram 3"
  "s16_sd10|16 kp4 10 1 rms_token none 0"
  "s04_baseline_ref|04 kp2 0 1 rms none 0"
)
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"; set -- $rest; sid=$1; pid=$2; sd=$3; lca=$4; norm=$5; sl=$6; sw=$7
  pfx="r4_${label}"; png="results/${pfx}_ca_only_style${sid}_${pid}_seed65.png"
  [ -f "$png" ] && { echo "SKIP $label"; continue; }
  free=$(wait_gpu) || { echo "[R4] TIMEOUT $label skip"; continue; }
  echo "[R4] $label sid=$sid sd=$sd lca=$lca norm=$norm stats=$sl w=$sw free=$free $(date +%H:%M:%S)"
  env_sd=""; [ "$sd" != "0" ] && env_sd="STYLED_DENOISE=$sd"
  env_st=""; [ "$sl" != "none" ] && env_st="STATS_LOSS=$sl STATS_WEIGHT=$sw"
  env $env_sd $env_st \
  $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca $lca --pseudo-step-size 0.05 \
    --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain --pseudo-grad-normalization $norm \
    --out-prefix "$pfx" > "agent_runs/r4_${label}.log" 2>&1
  [ -f "$png" ] && echo "[R4] OK $label" || { echo "[R4] FAIL $label"; tail -6 "agent_runs/r4_${label}.log"; }
done
echo "[R4] done $(date +%H:%M:%S)"
