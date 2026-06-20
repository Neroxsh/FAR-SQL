#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

args=(
  --config "$NDBC_ROOT/configs/default.yaml"
  --strategy model_only
  --prompt-dir outputs/model_only_prompts
  --log-dir outputs/model_only_logs
  --log-suffix model_only
)
if [ -n "${MODEL_ONLY_LIMIT:-}" ]; then
  args+=(--limit "$MODEL_ONLY_LIMIT")
fi
if [ -n "${MODEL_ONLY_MODEL_KEY:-}" ]; then
  args+=(--model-key "$MODEL_ONLY_MODEL_KEY")
fi
"$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" "${args[@]}"
