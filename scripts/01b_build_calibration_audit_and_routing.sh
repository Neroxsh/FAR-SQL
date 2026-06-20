#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

mkdir -p "$NDBC_ROOT/outputs/logs"

"$PYTHON_BIN" "$NDBC_ROOT/src/learn_routing_policy.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  > "$NDBC_ROOT/outputs/logs/learn_routing_policy.log"

"$PYTHON_BIN" "$NDBC_ROOT/src/analyze_calibration_buckets.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  > "$NDBC_ROOT/outputs/logs/analyze_calibration_buckets.log"

"$PYTHON_BIN" "$NDBC_ROOT/src/summarize_autobudget_routing.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  > "$NDBC_ROOT/outputs/logs/summarize_autobudget_routing.log"

cat "$NDBC_ROOT/outputs/budget/budget_table.json"
echo
echo "Routing plan: $NDBC_ROOT/outputs/tables/autobudget_routing_plan.md"
