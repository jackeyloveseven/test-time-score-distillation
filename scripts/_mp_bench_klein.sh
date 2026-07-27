#!/usr/bin/env bash
# Multi-prompt robustness benchmark (2026-07-21) -- klein-sem / klein-gst native-edit baseline
# (diagnostic_kleinedit_sysprompt_baseline.py, now parametrized by --prompt-id) for the 5
# media-diverse styles {23,09,10,00,34} x kp1,kp2,kp3,kp4. One process per kp (batches both
# sem+gst x all 5 styles via --only in a single model load -- the script self-gates internally
# at GATE_FREE_MIB=40000, so no external gate wrapper is needed here).
set -u; cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709
PY=/home/ubuntu/chengyanli/flux/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ONLY="23:sem,23:gst,09:sem,09:gst,10:sem,10:gst,00:sem,00:gst,34:sem,34:gst"
echo "[klein] START $(date +%F_%H:%M:%S)"
for kp in kp1 kp2 kp3 kp4; do
  # quick pre-check: skip the whole kp invocation if all 10 outputs already exist
  need=0
  for sid in 23 09 10 00 34; do
    for pkey in sem gst; do
      [ -f "results/diagnostic_kleinsys_${pkey}_style${sid}_${kp}.png" ] || need=1
    done
  done
  if [ "$need" -eq 0 ]; then echo "[klein] SKIP $kp (all 10 exist)"; continue; fi
  echo "[klein] LAUNCH $kp $(date +%H:%M:%S)"
  $PY scripts/diagnostic_kleinedit_sysprompt_baseline.py --prompt-id $kp --only "$ONLY" > agent_runs/mp_klein_${kp}.log 2>&1
  echo "[klein] invocation for $kp exited rc=$? $(date +%H:%M:%S)"
done
echo "[klein] ALL DONE $(date +%F_%H:%M:%S)"
