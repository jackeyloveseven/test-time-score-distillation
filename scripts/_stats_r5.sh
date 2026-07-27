set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
# label | SID PID STYLED_DENOISE lambda_ca norm STATS_LOSS STATS_WEIGHT step iters
CONFIGS=(
  "s16_gram10|16 kp4 4 1 rms_token gram 10 0.05 5"
  "s16_gram30|16 kp4 4 1 rms_token gram 30 0.05 5"
  "s16_push|16 kp4 4 1 rms_token none 0 0.10 8"
  "s04_gram10|04 kp2 4 1 rms_token gram 10 0.05 5"
  "s04_sd7|04 kp2 7 1 rms_token none 0 0.05 5"
)
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"; set -- $rest; sid=$1;pid=$2;sd=$3;lca=$4;norm=$5;sl=$6;sw=$7;step=$8;it=$9
  pfx="r5_${label}"; png="results/${pfx}_ca_only_style${sid}_${pid}_seed65.png"
  [ -f "$png" ] && { echo "SKIP $label"; continue; }
  free=$(wait_gpu) || { echo "[R5] TIMEOUT $label skip"; continue; }
  echo "[R5] $label sid=$sid sd=$sd stats=$sl w=$sw step=$step it=$it free=$free $(date +%H:%M:%S)"
  env_sd=""; [ "$sd" != "0" ] && env_sd="STYLED_DENOISE=$sd"
  env_st=""; [ "$sl" != "none" ] && env_st="STATS_LOSS=$sl STATS_WEIGHT=$sw"
  env $env_sd $env_st \
  $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca $lca --pseudo-step-size $step \
    --num-steps 25 --iters $it --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain --pseudo-grad-normalization $norm \
    --out-prefix "$pfx" > "agent_runs/r5_${label}.log" 2>&1
  [ -f "$png" ] && echo "[R5] OK $label" || { echo "[R5] FAIL $label"; tail -6 "agent_runs/r5_${label}.log"; }
done
echo "[R5] done $(date +%H:%M:%S)"
