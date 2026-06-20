#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

for budget in ${FIXED_BUDGETS:-1 2 3 5 10 30 60 120}; do
  NDBC_VERIEQL_CORES="${NDBC_VERIEQL_CORES:-16}" "$PYTHON_BIN" "$NDBC_ROOT/src/run_verieql_budgeted.py" \
    --config "$NDBC_ROOT/configs/default.yaml" \
    --datasets test_leetcode,test_calcite_spider \
    --strategy-name "fixed${budget}s" \
    --fixed-budget-sec "$budget" \
    --static-policy run \
    --skip-existing
done
