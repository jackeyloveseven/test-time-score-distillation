set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run () {  # norm_mode out_prefix
  local nm="$1" pfx="$2"
  for sp in "16 kp4 65" "04 kp2 65"; do
    set -- $sp; sid=$1;pid=$2;seed=$3
    png="results/${pfx}_ca_only_style${sid}_${pid}_seed${seed}.png"
    [ -f "$png" ] && { echo "SKIP $pfx s$sid"; continue; }
    for i in $(seq 1 80); do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
      [ "$free" -ge 85000 ] && break; sleep 15
    done
    echo "[NF] $pfx s$sid free=$free $(date +%H:%M:%S)"
    $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed $seed \
      --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 --pseudo-step-size 0.05 \
      --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
      --disable-slt --adain-alpha 0 --disable-final-adain \
      --pseudo-grad-normalization "$nm" --out-prefix "$pfx" > "agent_runs/nf_${pfx}_s${sid}.log" 2>&1
    [ -f "$png" ] && echo "[NF] OK $pfx s$sid" || { echo "[NF] FAIL $pfx s$sid"; tail -6 "agent_runs/nf_${pfx}_s${sid}.log"; }
  done
}
run rms_token skprobe_rmstoken
run blend0.5   skprobe_blend50
echo "[NF] done $(date +%H:%M:%S)"
