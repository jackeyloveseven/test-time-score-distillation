set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
# label | SID PID STYLED_DENOISE START content_weight STATS_LOSS STATS_WEIGHT
CONFIGS=(
  "lateS8_04|04 kp4 2 8 0.2 none 0"
  "lateS12_04|04 kp4 2 12 0.2 none 0"
  "lateS8_09|09 kp4 2 8 0.2 none 0"
  "lateS12_09|09 kp4 2 12 0.2 none 0"
  "cw06_04|04 kp4 2 0 0.6 none 0"
  "cw06_09|09 kp4 2 0 0.6 none 0"
  "gram_04|04 kp2 2 0 0.2 gram 10"
  "gram_25|25 kp0 2 0 0.2 gram 10"
  "g3_00|00 kp0 3 0 0.2 none 0"
  "g3_33|33 kp3 3 0 0.2 none 0"
)
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"; set -- $rest; sid=$1;pid=$2;sd=$3;st=$4;cw=$5;sl=$6;sw=$7
  pfx="fork_${label}"; png="results/${pfx}_ca_only_style${sid}_${pid}_seed65.png"
  [ -f "$png" ] && { echo "SKIP $label"; continue; }
  free=$(wait_gpu) || { echo "[FK] TIMEOUT $label"; continue; }
  echo "[FK] $label sid=$sid sd=$sd start=$st cw=$cw stats=$sl w=$sw free=$free $(date +%H:%M:%S)"
  envs="STYLED_DENOISE=$sd STYLED_DENOISE_START=$st"; [ "$sl" != "none" ] && envs="$envs STATS_LOSS=$sl STATS_WEIGHT=$sw"
  env $envs $PY scripts/run_tpami_with_style_teacher.py --style-id "$sid" --prompt-id "$pid" --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1 --pseudo-step-size 0.05 --num-steps 25 --iters 5 \
    --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 \
    --disable-final-adain --pseudo-grad-normalization rms_token --content-weight "$cw" \
    --out-prefix "$pfx" > "agent_runs/fk_${label}.log" 2>&1
  [ -f "$png" ] && echo "[FK] OK $label" || { echo "[FK] FAIL $label"; tail -5 "agent_runs/fk_${label}.log"; }
done
echo "[FK] done $(date +%H:%M:%S)"
