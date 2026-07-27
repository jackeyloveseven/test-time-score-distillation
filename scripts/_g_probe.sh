set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 60); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
  [ "$free" -ge 85000 ] && break; sleep 15
done
echo "[GP] launch free=$free $(date +%H:%M:%S)"
DUMP_XHAT_STEPS="2,12,20" DUMP_G_STEPS="2,12,20" \
$PY scripts/run_tpami_with_style_teacher.py --style-id 16 --prompt-id kp4 --seed 65 \
  --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 --pseudo-step-size 0.05 \
  --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix gprobe > agent_runs/gprobe.log 2>&1
echo "[GP] done rc=$? $(date +%H:%M:%S)"; ls agent_runs/g_dump/ 2>/dev/null | wc -l | sed 's/^/g-dumps: /'
