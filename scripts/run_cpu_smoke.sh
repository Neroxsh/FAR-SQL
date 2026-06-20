#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
export NDBC_DATASETS="${NDBC_DATASETS:-test_leetcode,test_calcite_spider}"
export NDBC_VERIEQL_CORES="${NDBC_VERIEQL_CORES:-2}"

bash "$ROOT/scripts/00_preflight.sh"
bash "$ROOT/scripts/00_prepare_data.sh"
bash "$ROOT/scripts/01_calibrate_budget.sh"
bash "$ROOT/scripts/01b_build_calibration_audit_and_routing.sh"
bash "$ROOT/scripts/02_run_verieql_budgeted_test.sh"
bash "$ROOT/scripts/03_build_fallback_prompts.sh"
bash "$ROOT/scripts/09_compute_all_experiments.sh"
