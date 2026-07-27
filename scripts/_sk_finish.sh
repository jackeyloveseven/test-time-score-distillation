set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1) wait for the running driver to produce style16 baseline (or 10min cap)
for i in $(seq 1 40); do [ -f results/skprobe_ca_only_style16_kp4_seed65.png ] && break; sleep 15; done
# 2) run the one still-missing cell: style04 mix05, once GPU has a clean window
png="results/skprobe_mix05_ca_only_style04_kp2_seed65.png"
if [ ! -f "$png" ]; then
  for i in $(seq 1 60); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
    if [ "$free" -ge 90000 ]; then
      echo "[FIN] run s04 mix05 free=$free $(date +%H:%M:%S)"
      SINKHORN_ATTN=1 SINKHORN_MIX=0.5 SINKHORN_ITERS=3 $PY scripts/run_tpami_with_style_teacher.py \
        --style-id 04 --prompt-id kp2 --seed 65 --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 \
        --pseudo-step-size 0.05 --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw \
        --latent-weight-decay 0 --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix skprobe_mix05 \
        > agent_runs/sk_s04_mix05_retry.log 2>&1
      break
    fi; sleep 15
  done
fi
echo "[FIN] done $(date +%H:%M:%S); matrix:"; ls results/skprobe*_ca_only_style{04,14,16}_*_seed65.png 2>/dev/null | grep -v reference
