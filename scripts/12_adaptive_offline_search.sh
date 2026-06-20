#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python src/adaptive_offline_search.py --config configs/default.yaml

