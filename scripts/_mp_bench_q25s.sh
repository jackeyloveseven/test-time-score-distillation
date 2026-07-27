#!/usr/bin/env bash
# Multi-prompt robustness benchmark (2026-07-21) -- worker B: ours-Semantic (q25s / ca_only cfg7)
# for the 5 media-diverse styles {23,09,10,00,34} x kp3,kp4 (kp0/kp1/kp2 already exist).
# Headroom-gated (external GPU jobs never killed); skip-existing = resumable checkpoint.
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEED=40000
STY=/home/ubuntu/chengyanli/items/save/attention_20260426_090948/sty
COMMON="--style-mode ca_only --cfg-scale 7 --lambda-ca 1.0 --pseudo-step-size 0.05 --num-steps 25 --iters 5 --start-time 1000 --latent-optimizer adamw --latent-weight-decay 0 --disable-slt --adain-alpha 0 --disable-final-adain --seed 65 --without-prompt-mode full_style_prompt --pseudo-grad-normalization rms --main-model-root /mnt/local/klein9b-base --teacher-model-root /mnt/local/klein9b-base --vram-limit-gb 38 --main-vram-limit-gb 38 --teacher-vram-limit-gb 38 --out-prefix q25s"
STYLES=(23 09 10 00 34)
PROMPTS=(kp3 kp4)
echo "[q25s] START $(date +%F_%H:%M:%S)"
for sid in "${STYLES[@]}"; do
  for kp in "${PROMPTS[@]}"; do
    png="results/diagnostic_q25s_ca_only_style${sid}_${kp}_seed65.png"
    if [ -f "$png" ]; then echo "[q25s] SKIP $sid $kp (exists)"; continue; fi
    t=0
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1); free=${free:-0}
      if [ "$free" -ge "$NEED" ]; then
        echo "[q25s] LAUNCH s$sid $kp (free=$free) $(date +%H:%M:%S)"
        $PY scripts/run_tpami_with_style_teacher.py --style-id $sid --style-image ${STY}/${sid}.png --prompt-id $kp $COMMON > agent_runs/mp_q25s_${sid}_${kp}.log 2>&1
        if [ -f "$png" ]; then echo "[q25s] OK s$sid $kp $(date +%H:%M:%S)"; break; fi
        echo "[q25s] FAIL s$sid $kp (see agent_runs/mp_q25s_${sid}_${kp}.log) -- retrying after backoff"
      else
        echo "[q25s] gate: free=$free < $NEED, sleep 20s"
      fi
      t=$((t+1)); if [ $t -gt 240 ]; then echo "[q25s] TIMEOUT s$sid $kp -- giving up"; break; fi
      sleep 20
    done
  done
done
echo "[q25s] ALL DONE $(date +%F_%H:%M:%S)"
