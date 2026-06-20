#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

rm -rf "$NDBC_ROOT/outputs"
find "$NDBC_ROOT/src" -type d -name "__pycache__" -prune -exec rm -rf {} +

mkdir -p \
  "$NDBC_ROOT/outputs/preflight" \
  "$NDBC_ROOT/outputs/calibration" \
  "$NDBC_ROOT/outputs/budget" \
  "$NDBC_ROOT/outputs/tables" \
  "$NDBC_ROOT/outputs/metrics" \
  "$NDBC_ROOT/outputs/results" \
  "$NDBC_ROOT/outputs/verieql_budgeted" \
  "$NDBC_ROOT/outputs/decisions" \
  "$NDBC_ROOT/outputs/fallback_prompts" \
  "$NDBC_ROOT/outputs/fallback_logs" \
  "$NDBC_ROOT/outputs/model_only_prompts" \
  "$NDBC_ROOT/outputs/model_only_logs" \
  "$NDBC_ROOT/outputs/logs"

echo "Cleaned generated outputs under $NDBC_ROOT/outputs"
