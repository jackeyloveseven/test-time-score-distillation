#!/usr/bin/env bash
# Serial, resumable validation queue. Never terminates foreign GPU processes.
set -euo pipefail
cd /home/ubuntu/chengyanli/items/style_residual_distill_20260709

echo "[remaining-validation] start $(date -u +%FT%TZ)"
./scripts/run_benchmark_multiseed.sh
./scripts/run_reverse_shield_ablation.sh
echo "[remaining-validation] complete $(date -u +%FT%TZ)"
