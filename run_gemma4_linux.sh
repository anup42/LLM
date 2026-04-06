#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Usage:
#   ./run_gemma4_linux.sh "/models/gemma-4-26B-A4B-it" "Write a short joke about saving RAM."
#   ./run_gemma4_linux.sh "" "Explain MoE in one sentence."
MODEL_PATH="${1:-}"
PROMPT="${2:-Write a short joke about saving RAM.}"

if [[ -n "${MODEL_PATH}" ]]; then
  python3 "${SCRIPT_DIR}/gemma4_runner.py" \
    --model-path "${MODEL_PATH}" \
    --local-files-only \
    --prompt "${PROMPT}"
else
  python3 "${SCRIPT_DIR}/gemma4_runner.py" \
    --model-id "google/gemma-4-26B-A4B-it" \
    --prompt "${PROMPT}"
fi
