set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GATE=90000
wait_gpu(){ local t=0 f; while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge "$GATE" ] && { echo "$f"; return 0; }; t=$((t+1)); [ $t -gt 960 ] && return 1; sleep 15; done; }
# label | lambda_ca STATS_WEIGHT
CONFIGS=("w200 1 200" "w2000 1 2000" "w20000 1 20000" "w2000pure 0 2000")
for entry in "${CONFIGS[@]}"; do
  label="${entry%%|*}"; set -- $entry; label=$1; lca=$2; sw=$3
  pfx="vgg_${label}"; png="results/${pfx}_ca_only_style04_kp2_seed65.png"
  [ -f "$png" ] && { echo "SKIP $label"; continue; }
  free=$(wait_gpu) || { echo "[VG] TIMEOUT $label"; continue; }
  echo "[VG] $label lca=$lca w=$sw free=$free $(date +%H:%M:%S)"
  STYLED_DENOISE=4 STATS_LOSS=vgg STATS_WEIGHT=$sw \
  $PY scripts/run_tpami_with_style_teacher.py --style-id 04 --prompt-id kp2 --seed 65 \
    --style-mode ca_only --cfg-scale 7 --lambda-ca $lca --pseudo-step-size 0.05 \
    --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
    --disable-slt --adain-alpha 0 --disable-final-adain --pseudo-grad-normalization rms_token \
    --out-prefix "$pfx" > "agent_runs/vg_${label}.log" 2>&1
  [ -f "$png" ] && echo "[VG] OK $label" || { echo "[VG] FAIL $label"; tail -6 "agent_runs/vg_${label}.log"; }
done
echo "[VG] done $(date +%H:%M:%S)"
