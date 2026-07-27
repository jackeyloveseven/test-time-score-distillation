set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 1200 ] && return 1; sleep 15; done; }
for i in $(seq 0 39); do
  sid=$(printf "%02d" $i); pid="kp$((i % 5))"
  pfx="full_late8"; png="results/${pfx}_ca_only_style${sid}_${pid}_seed65.png"
  [ -f "$png" ] && { echo "SKIP s$sid"; continue; }
  free=$(wait_gpu) || { echo "[L8] TIMEOUT s$sid"; continue; }
  echo "[L8] s$sid $pid free=$free $(date +%H:%M:%S)"
  STYLED_DENOISE=2 STYLED_DENOISE_START=8 $PY scripts/run_tpami_with_style_teacher.py --style-id "$sid" --prompt-id "$pid" --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1 --pseudo-step-size 0.05 --num-steps 25 --iters 5 \
    --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 \
    --disable-final-adain --pseudo-grad-normalization rms_token --out-prefix "$pfx" > "agent_runs/l8_s${sid}.log" 2>&1
  [ -f "$png" ] && echo "[L8] OK s$sid" || echo "[L8] FAIL s$sid"
done
echo "[L8] ALL DONE $(date +%H:%M:%S)"
