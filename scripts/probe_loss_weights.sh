#!/usr/bin/env bash
# Free diagnostic (no new loss): does LOWERING content/dist regularization rescue the
# texture-collapse on ROBUST-failure styles? Tests the CUT-4 mechanism directly. If
# texture returns just by de-weighting the content anchor, the failure IS regularizer
# dominance → motivates a texture-targeting (Gram/VGG) loss next. score_match only.
# Config = bench40 COMMON except --content-weight / --dist-weight overridden.
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
OUTDIR=agent_runs/probe_lossw; mkdir -p "$OUTDIR"
NEED=66000; MAX_TRIES=120

BASE=(--num-steps 25 --iters 5 --start-time 1000
      --style-grad-rms-target 2.159326726314248e-06
      --latent-optimizer adamw --latent-weight-decay 0
      --carrier-temperature 0.1 --carrier-strength 0.75
      --carrier-highpass-gain 0 --carrier-loss-weight 0.25
      --disable-slt --adain-alpha 0 --disable-final-adain)

# robust-failure cells (fail on every seed): style04/kp2 (gold mosaic), style21/kp3 (3D geometric)
# variants: cw/dw = content-weight/dist-weight.  baseline was cw0.4 dw0.3.
run() {  # $1=style $2=prompt $3=seed $4=cw $5=dw $6=tag
  local png="results/probelw_${6}_score_match_style${1}_${2}_seed${3}.png"
  [ -f "$png" ] && { echo "[probe] SKIP $png"; return; }
  local tries=0
  while true; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      echo "[probe] LAUNCH style${1} ${2} cw=${4} dw=${5} (${6}) free=${free} $(date +%H:%M:%S)"
      "$PY" scripts/run_tpami_with_style_teacher.py \
        --style-id "$1" --prompt-id "$2" --seed "$3" --style-mode score_match \
        "${BASE[@]}" --content-weight "$4" --dist-weight "$5" \
        --out-prefix "probelw_${6}" > "$OUTDIR/style${1}_${2}_${6}.log" 2>&1
      [ $? -eq 0 ] && [ -f "$png" ] && { echo "[probe] OK style${1} ${6} $(date +%H:%M:%S)"; return; }
      echo "[probe] FAIL style${1} ${6} (retry)"
    fi
    tries=$((tries+1)); [ "$tries" -gt "$MAX_TRIES" ] && { echo "[probe] TIMEOUT style${1} ${6}"; return; }
    sleep 40
  done
}

echo "[probe] START $(date +%F_%H:%M:%S)"
# lowered regularization (cw0.1 dw0.1) and aggressive (cw0.0 dw0.05)
run 04 kp2 65 0.1  0.1  lowreg
run 04 kp2 65 0.0  0.05 minreg
run 21 kp3 64 0.1  0.1  lowreg
run 21 kp3 64 0.0  0.05 minreg
echo "[probe] ALL DONE $(date +%F_%H:%M:%S)"
