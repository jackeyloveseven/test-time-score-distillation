set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 1200 ] && return 1; sleep 15; done; }
run(){
  local sid=$1; local pid=$2; local var=$3
  local pfx sd norm
  if [ "$var" = "base" ]; then pfx="full_base"; sd=""; norm="rms"; else pfx="full_sd2"; sd="STYLED_DENOISE=2"; norm="rms_token"; fi
  local png="results/${pfx}_ca_only_style${sid}_${pid}_seed65.png"
  [ -f "$png" ] && { echo "SKIP $var s$sid"; return; }
  local free; free=$(wait_gpu) || { echo "[FULL] TIMEOUT $var s$sid"; return; }
  echo "[FULL] $var(g2) s$sid $pid free=$free $(date +%H:%M:%S)"
  env $sd $PY scripts/run_tpami_with_style_teacher.py --style-id "$sid" --prompt-id "$pid" --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1 --pseudo-step-size 0.05 --num-steps 25 --iters 5 \
    --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 \
    --disable-final-adain --pseudo-grad-normalization "$norm" --out-prefix "$pfx" > "agent_runs/full_${var}_s${sid}.log" 2>&1
  [ -f "$png" ] && echo "[FULL] OK $var s$sid" || { echo "[FULL] FAIL $var s$sid"; tail -4 "agent_runs/full_${var}_s${sid}.log"; }
}
for i in $(seq 0 39); do sid=$(printf "%02d" $i); pid="kp$((i % 5))"; run "$sid" "$pid" base; run "$sid" "$pid" sd; done
echo "[FULL] ALL DONE $(date +%H:%M:%S)"
