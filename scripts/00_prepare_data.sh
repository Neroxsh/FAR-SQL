#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

args=(--config "$NDBC_ROOT/configs/default.yaml")
if [ -n "${SMOKE_LIMIT:-}" ]; then
  args+=(--limit "$SMOKE_LIMIT")
fi
"$PYTHON_BIN" "$NDBC_ROOT/src/prepare_data.py" "${args[@]}"
