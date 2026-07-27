set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 1 240); do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); f=${f:-0}; [ "$f" -ge 90000 ] && break; sleep 15; done
echo "[VGG] launch free=$f $(date +%H:%M:%S)"
STYLED_DENOISE=4 STATS_LOSS=vgg STATS_WEIGHT=2 \
$PY scripts/run_tpami_with_style_teacher.py --style-id 04 --prompt-id kp2 --seed 65 \
  --style-mode ca_only --cfg-scale 7 --lambda-ca 1 --pseudo-step-size 0.05 \
  --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
  --disable-slt --adain-alpha 0 --disable-final-adain --pseudo-grad-normalization rms_token \
  --out-prefix vggtest_w2 > agent_runs/vgg_test.log 2>&1
echo "[VGG] rc=$? $(date +%H:%M:%S)"; tail -5 agent_runs/vgg_test.log
ls -la results/vggtest_w2_ca_only_style04_kp2_seed65.png 2>/dev/null && echo OK || echo "NO IMAGE"
