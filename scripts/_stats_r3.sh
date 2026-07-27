set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SID=16; PID=kp4; SEED=65; GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
# label | STYLED_DENOISE lambda_ca norm
CONFIGS=(
  "sdonly4|4 0 rms"
  "sd2captok|2 1 rms_token"
  "sd4captok|4 1 rms_token"
  "sd7captok|7 1 rms_token"
  "sd4ca|4 1 rms"
)
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"; set -- $rest; sd=$1; lca=$2; norm=$3
  pfx="r3_${label}"; png="results/${pfx}_ca_only_style${SID}_${PID}_seed${SEED}.png"
  [ -f "$png" ] && { echo "SKIP $label"; continue; }
  free=$(wait_gpu) || { echo "[R3] TIMEOUT $label skip"; continue; }
  echo "[R3] $label sd=$sd lca=$lca norm=$norm free=$free $(date +%H:%M:%S)"
  STYLED_DENOISE=$sd \
  $PY scripts/run_tpami_with_style_teacher.py --style-id $SID --prompt-id $PID --seed $SEED \
    --style-mode ca_only --cfg-scale 7 --lambda-ca $lca --pseudo-step-size 0.05 \
    --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain --pseudo-grad-normalization $norm \
    --out-prefix "$pfx" > "agent_runs/r3_${label}.log" 2>&1
  [ -f "$png" ] && echo "[R3] OK $label" || { echo "[R3] FAIL $label"; tail -8 "agent_runs/r3_${label}.log"; }
done
echo "[R3] done $(date +%H:%M:%S)"
