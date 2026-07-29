#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-core}"
command -v hf >/dev/null || {
  echo "Missing 'hf'. Install requirements.txt first." >&2
  exit 1
}

mkdir -p models
hf download black-forest-labs/FLUX.2-klein-9B --local-dir models/klein9b
hf download black-forest-labs/FLUX.2-klein-base-9B --local-dir models/klein9b-base

if [[ "$mode" == "all" ]]; then
  hf download black-forest-labs/FLUX.2-klein-base-4B --local-dir models/klein4b-base
  hf download openai/clip-vit-large-patch14 --local-dir models/clip
fi

echo "Style TSD model download complete (mode=$mode)."
