#!/usr/bin/env bash
# Multi-prompt robustness benchmark (2026-07-21) -- ours-Global (pg_n8_s12_r) for the 5
# media-diverse styles {23,09,10,00,34} x a caller-supplied subset of {kp1,kp2,kp3,kp4}.
# Usage: _mp_bench_pg.sh kp1,kp2   (or kp3,kp4 for the other half -- split across 2 workers)
# Headroom-gated (external GPU jobs never killed); skip-existing = resumable checkpoint.
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEED=40000
STY=/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
COMMON="--style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 --pseudo-step-size 0.12 --num-steps 8 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 --disable-final-adain --seed 65 --without-prompt-mode full_style_prompt --pseudo-grad-normalization rms --main-model-root /mnt/local/klein9b-base --teacher-model-root /mnt/local/klein9b-base --vram-limit-gb 38 --main-vram-limit-gb 38 --teacher-vram-limit-gb 38 --out-prefix pg_n8_s12_r"
STYLES=(23 09 10 00 34)
IFS=',' read -r -a PROMPTS <<< "${1:-kp1,kp2,kp3,kp4}"
TAG="${2:-pg}"
echo "[$TAG] START $(date +%F_%H:%M:%S) prompts=${PROMPTS[*]}"
for sid in "${STYLES[@]}"; do
  for kp in "${PROMPTS[@]}"; do
    png="results/diagnostic_pg_n8_s12_r_ca_only_style${sid}_${kp}_seed65.png"
    if [ -f "$png" ]; then echo "[$TAG] SKIP $sid $kp (exists)"; continue; fi
    t=0
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
      if [ "$free" -ge "$NEED" ]; then
        echo "[$TAG] LAUNCH s$sid $kp (free=$free) $(date +%H:%M:%S)"
        $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --style-image ${STY}/${sid}.png --prompt-id $kp $COMMON > agent_runs/mp_${TAG}_${sid}_${kp}.log 2>&1
        if [ -f "$png" ]; then echo "[$TAG] OK s$sid $kp $(date +%H:%M:%S)"; break; fi
        echo "[$TAG] FAIL s$sid $kp (see agent_runs/mp_${TAG}_${sid}_${kp}.log) -- retrying after backoff"
      else
        echo "[$TAG] gate: free=$free < $NEED, sleep 20s"
      fi
      t=$((t+1)); if [ $t -gt 240 ]; then echo "[$TAG] TIMEOUT s$sid $kp -- giving up"; break; fi
      sleep 20
    done
  done
done
echo "[$TAG] ALL DONE $(date +%F_%H:%M:%S)"
