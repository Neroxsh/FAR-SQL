#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

"$PYTHON_BIN" "$NDBC_ROOT/src/fuse_decisions.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --verieql-strategy autobudget
