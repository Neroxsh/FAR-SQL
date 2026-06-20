#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

args=(--config "$NDBC_ROOT/configs/default.yaml" --strategy autobudget)
if [ -n "${FALLBACK_LIMIT:-}" ]; then
  args+=(--limit "$FALLBACK_LIMIT")
fi
if [ -n "${FALLBACK_MODEL_KEY:-}" ]; then
  args+=(--model-key "$FALLBACK_MODEL_KEY")
fi
"$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" "${args[@]}"
