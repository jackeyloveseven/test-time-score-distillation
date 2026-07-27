#!/usr/bin/env bash
# Sinkhorn-attention PROBE — small-sample, many-cell sweep.
# Q: does doubly-stochastic (de-concentrated) own-attention spread style off the
# background onto the prompt-fixed subject, and at what content cost?
# Per (style,prompt): baseline TSD vs Sinkhorn mix=0.5 vs mix=1.0 (iters=3).
# Renormalizes the model's OWN attention only — NEVER injects reference K/V.
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3; NEED=90000   # raised: dodge kaon5-server mid-load OOM race
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p agent_runs results

# style prompt seed  (3 hard/marginal styles where local-edit is clearest)
CASES=("04 kp2 65" "16 kp4 65" "14 kp0 65")
# label out_prefix SINKHORN_ATTN SINKHORN_MIX
CONFIGS=("baseline skprobe 0 0" "mix10 skprobe_mix10 1 1.0" "mix05 skprobe_mix05 1 0.5")

run_one () {  # sid pid seed  label prefix sk mix
  local sid="$1" pid="$2" seed="$3" label="$4" prefix="$5" sk="$6" mix="$7"
  local tag="s${sid}_${pid}_${label}"
  local png="results/${prefix}_ca_only_style${sid}_${pid}_seed${seed}.png"
  [ -f "$png" ] && { echo "[SK] SKIP $tag ($png exists)"; return 0; }
  local t=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      echo "[SK] LAUNCH $tag (free=$free) $(date +%H:%M:%S)"
      SINKHORN_ATTN="$sk" SINKHORN_MIX="$mix" SINKHORN_ITERS=3 \
        $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --prompt-id $pid --seed $seed \
        --style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 --pseudo-step-size 0.05 \
        --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 \
        --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix "$prefix" \
        > "agent_runs/sk_${tag}.log" 2>&1
      if [ -f "$png" ]; then echo "[SK] OK $tag -> $png"; else echo "[SK] FAIL $tag"; tail -8 "agent_runs/sk_${tag}.log"; fi
      return 0
    fi
    t=$((t+1)); [ $((t%20)) -eq 0 ] && echo "[SK] waiting $tag free=${free} (t=${t}) $(date +%H:%M:%S)"
    [ $t -gt 1200 ] && { echo "[SK] TIMEOUT $tag"; return 1; }   # ~5h at 15s
    sleep 15
  done
}

echo "[SK] sweep start $(date +%H:%M:%S)  need>=${NEED}MiB  ${#CASES[@]}x${#CONFIGS[@]} cells"
# Interleave: full config triad per style, style04 first (clearest local-edit).
for c in "${CASES[@]}"; do
  set -- $c; sid=$1; pid=$2; seed=$3
  for cf in "${CONFIGS[@]}"; do
    set -- $cf; label=$1; prefix=$2; sk=$3; mix=$4
    run_one "$sid" "$pid" "$seed" "$label" "$prefix" "$sk" "$mix"
  done
done
echo "[SK] sweep DONE $(date +%H:%M:%S)"
