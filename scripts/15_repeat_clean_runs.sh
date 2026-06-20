#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
ROUTER_BASE_TAG="${ROUTER_BASE_TAG:?ROUTER_BASE_TAG is required}"
STRATEGY="${STRATEGY:-adaptive_accuracy}"
REPEATS="${REPEATS:-5}"

export FALLBACK_BATCH_SIZE="${FALLBACK_BATCH_SIZE:-192}"

for ((i=1; i<=REPEATS; i++)); do
  echo "[repeat ${i}/${REPEATS}] 纯模型"
  "$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
    --config "$NDBC_ROOT/configs/default.yaml" \
    --strategy model_only \
    --prompt-dir outputs/model_only_prompts \
    --log-dir outputs/model_only_logs \
    --log-suffix model_only \
    --model-key "$MODEL_KEY"

  echo "[repeat ${i}/${REPEATS}] Trace 引导"
  "$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
    --config "$NDBC_ROOT/configs/default.yaml" \
    --strategy "$STRATEGY" \
    --prompt-suffix trace_guided \
    --log-suffix fallback_trace_guided \
    --model-key "$MODEL_KEY"

  echo "[repeat ${i}/${REPEATS}] Witness 引导"
  "$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
    --config "$NDBC_ROOT/configs/default.yaml" \
    --strategy "$STRATEGY" \
    --prompt-suffix witness_guided \
    --log-suffix fallback_witness_guided \
    --model-key "$MODEL_KEY"

  echo "[repeat ${i}/${REPEATS}] 状态感知融合（干净版）"
  "$PYTHON_BIN" "$NDBC_ROOT/src/run_status_aware_router.py" \
    --config "$NDBC_ROOT/configs/default.yaml" \
    --strategy "$STRATEGY" \
    --model-key "$MODEL_KEY" \
    --output-tag "${ROUTER_BASE_TAG}_r${i}_clean"
done
