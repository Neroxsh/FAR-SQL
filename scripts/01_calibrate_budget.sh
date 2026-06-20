#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

args=(--config "$NDBC_ROOT/configs/default.yaml")
if [ -n "${CALIBRATION_LIMIT:-}" ]; then
  args+=(--limit "$CALIBRATION_LIMIT")
fi
"$PYTHON_BIN" "$NDBC_ROOT/src/calibrate_budget.py" "${args[@]}"
