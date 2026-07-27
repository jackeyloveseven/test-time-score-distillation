#!/usr/bin/env bash
# Ceiling check (advisor gate for the decoupled-DMD route): decode the styled teacher's
# own one-step prediction x0^s at several noise levels on ROBUST-FAILURE styles. If x0^s
# is itself textureless at the operative sigma, NO fake score can conjure texture -> stop.
# Headroom-GATED: external GPU jobs must never be killed; we WAIT for free memory.
# Uses the DUMP_XHAT_STEPS instrumentation in run_tpami_with_style_teacher.py (env-gated).
set -u
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709 || exit 3
PY=/home/ubuntu/chengyanli/flux/bin/python3
LOGD=agent_runs/xhat_dump; mkdir -p "$LOGD"
NEED=70000; MAX_TRIES=1000
export DUMP_XHAT_STEPS="0,6,12,18,22"   # o24 never runs (loop breaks at last step); o22 is a real late step

COMMON=(--style-mode score_match --num-steps 25 --iters 5 --start-time 1000
        --style-grad-rms-target 2.159326726314248e-06
        --latent-optimizer adamw --latent-weight-decay 0
        --dist-weight 0.3 --content-weight 0.4
        --carrier-temperature 0.1 --carrier-strength 0.75
        --carrier-highpass-gain 0 --carrier-loss-weight 0.25
        --disable-slt --adain-alpha 0 --disable-final-adain --out-prefix xhatceil)

run() { # $1 style $2 prompt $3 seed
  local done="results/xhatceil_style${1}_${2}_o22_x0s_sig"*.png
  # crude done-check: any decoded dump for this style/prompt
  if ls results/xhatceil_style${1}_${2}_o22_x0s_sig*.png >/dev/null 2>&1; then echo "[ceil] SKIP style${1} ${2}"; return; fi
  local tries=0
  while true; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); free=${free:-0}
    if [ "$free" -ge "$NEED" ]; then
      echo "[ceil] LAUNCH style${1} ${2} seed${3} (free=${free}MiB) $(date +%H:%M:%S)"
      "$PY" scripts/run_tpami_with_style_teacher.py --style-id "$1" --prompt-id "$2" --seed "$3" \
        "${COMMON[@]}" > "$LOGD/ceil_style${1}_${2}.log" 2>&1
      if ls results/xhatceil_style${1}_${2}_o22_x0s_sig*.png >/dev/null 2>&1; then
        echo "[ceil] OK style${1} ${2} $(date +%H:%M:%S)"; return
      fi
      echo "[ceil] FAIL style${1} ${2} (likely OOM race) -> retry"
    fi
    tries=$((tries+1)); [ "$tries" -gt "$MAX_TRIES" ] && { echo "[ceil] TIMEOUT style${1} ${2}"; return; }
    sleep 45
  done
}

echo "[ceil] START $(date +%F_%H:%M:%S) — waiting for >=${NEED}MiB free (external jobs present, not killing)"
# run 04 done (textureless x0s confirmed)
run 16 kp4 65   # crystalline 3D-render (clear fail)
run 21 kp3 64   # 3D glossy geometric (robust fail)
echo "[ceil] ALL DONE $(date +%F_%H:%M:%S)"
