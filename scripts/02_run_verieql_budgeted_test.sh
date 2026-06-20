#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

datasets="${NDBC_DATASETS:-test_leetcode,test_calcite_spider}"
args=(
  --config "$NDBC_ROOT/configs/default.yaml" \
  --datasets "$datasets" \
  --strategy-name autobudget
)
if [ "${NDBC_SKIP_EXISTING:-0}" = "1" ]; then
  args+=(--skip-existing)
fi
"$PYTHON_BIN" "$NDBC_ROOT/src/run_verieql_budgeted.py" "${args[@]}"
