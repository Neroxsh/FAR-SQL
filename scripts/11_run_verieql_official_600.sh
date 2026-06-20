#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

NDBC_VERIEQL_CORES="${NDBC_VERIEQL_CORES:-16}" "$PYTHON_BIN" "$NDBC_ROOT/src/run_verieql_budgeted.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --datasets "${NDBC_DATASETS:-test_leetcode,test_calcite_spider}" \
  --strategy-name official600 \
  --fixed-budget-sec 600 \
  --static-policy run \
  --skip-existing
