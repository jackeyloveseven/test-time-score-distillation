set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=85000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
run(){ # sid pid variant(base|sd)
  local sid=$1 pid=$2 var=$3
  local pfx="brd_${var}"; local png="results/${pfx}_ca_only_style${sid}_${pid}_seed65.png"
  [ -f "$png" ] && { echo "SKIP $var s$sid"; return; }
  local free; free=$(wait_gpu) || { echo "[R6] TIMEOUT $var s$sid"; return; }
  local sd_env="" norm="rms"
  [ "$var" = "sd" ] && { sd_env="STYLED_DENOISE=4"; norm="rms_token"; }
  echo "[R6] $var s$sid free=$free $(date +%H:%M:%S)"
  env $sd_env $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca 1 --pseudo-step-size 0.05 \
    --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain --pseudo-grad-normalization $norm \
    --out-prefix "$pfx" > "agent_runs/r6_${var}_s${sid}.log" 2>&1
  [ -f "$png" ] && echo "[R6] OK $var s$sid" || { echo "[R6] FAIL $var s$sid"; tail -6 "agent_runs/r6_${var}_s${sid}.log"; }
}
for sp in "07 kp1" "22 kp2" "31 kp3"; do set -- $sp; run $1 $2 base; run $1 $2 sd; done
echo "[R6] done $(date +%H:%M:%S)"
