#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

"$PYTHON_BIN" "$NDBC_ROOT/src/build_model_only_prompts.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --verieql-strategy autobudget
