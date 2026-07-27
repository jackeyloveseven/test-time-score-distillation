set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
run(){
  local sid=$1; local pid=$2; local g=$3
  local pfx="g${g}"
  local png="results/${pfx}_ca_only_style${sid}_${pid}_seed65.png"
  [ -f "$png" ] && { echo "SKIP g$g s$sid"; return; }
  local free; free=$(wait_gpu) || { echo "TIMEOUT g$g s$sid"; return; }
  echo "[GC] g$g s$sid $pid free=$free $(date +%H:%M:%S)"
  STYLED_DENOISE=$g $PY scripts/run_tpami_with_style_teacher.py --style-id "$sid" --prompt-id "$pid" --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1 --pseudo-step-size 0.05 --num-steps 25 --iters 5 \
    --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 \
    --disable-final-adain --pseudo-grad-normalization rms_token --out-prefix "$pfx" > "agent_runs/gc_g${g}_s${sid}.log" 2>&1
  [ -f "$png" ] && echo "[GC] OK g$g s$sid" || { echo "[GC] FAIL g$g s$sid"; tail -4 "agent_runs/gc_g${g}_s${sid}.log"; }
}
for sp in "01 kp1" "02 kp2" "04 kp4" "09 kp4"; do set -- $sp; run "$1" "$2" 2; run "$1" "$2" 3; done
echo "[GC] done"
