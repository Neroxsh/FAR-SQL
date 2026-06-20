#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

STRATEGY="${1:-adaptive_accuracy}"
MODEL_KEY="${2:-qwen3_1_7b_100sft_cot}"

"$PYTHON_BIN" "$NDBC_ROOT/src/build_fallback_prompts.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --strategy "$STRATEGY" \
  --prompt-variant verieql_aware

"$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --strategy "$STRATEGY" \
  --model-key "$MODEL_KEY" \
  --log-suffix fallback_verieql_aware

"$PYTHON_BIN" "$NDBC_ROOT/src/fuse_decisions.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --verieql-strategy "$STRATEGY" \
  --log-suffix fallback_verieql_aware \
  --output-suffix fallback_verieql_aware
