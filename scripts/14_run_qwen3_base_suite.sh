#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

MODEL_KEY="${MODEL_KEY:-qwen3_1_7b_base}"
STRATEGY="${STRATEGY:-adaptive_accuracy}"
ROUTER_BASE_TAG="${ROUTER_BASE_TAG:-status_aware_router_base}"

export FALLBACK_BATCH_SIZE="${FALLBACK_BATCH_SIZE:-192}"

if [ "${SKIP_MODEL_ONLY:-0}" != "1" ]; then
  echo "[1/6] 纯模型"
  "$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
    --config "$NDBC_ROOT/configs/default.yaml" \
    --strategy model_only \
    --prompt-dir outputs/model_only_prompts \
    --log-dir outputs/model_only_logs \
    --log-suffix model_only \
    --model-key "$MODEL_KEY"
else
  echo "[1/6] 跳过纯模型（沿用现有结果）"
fi

echo "[2/6] Trace 引导"
"$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --strategy "$STRATEGY" \
  --prompt-suffix trace_guided \
  --log-suffix fallback_trace_guided \
  --model-key "$MODEL_KEY"

echo "[3/6] Witness 引导"
"$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --strategy "$STRATEGY" \
  --prompt-suffix witness_guided \
  --log-suffix fallback_witness_guided \
  --model-key "$MODEL_KEY"

echo "[4/6] VeriEQL 极简弱提示"
"$PYTHON_BIN" "$NDBC_ROOT/src/run_fallback_gpu.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --strategy "$STRATEGY" \
  --prompt-suffix verieql_hint \
  --log-suffix fallback_verieql_hint \
  --model-key "$MODEL_KEY"

echo "[5/6] 状态感知融合（干净版）"
"$PYTHON_BIN" "$NDBC_ROOT/src/run_status_aware_router.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --strategy "$STRATEGY" \
  --model-key "$MODEL_KEY" \
  --output-tag "${ROUTER_BASE_TAG}_clean"

echo "[6/6] 状态感知融合（增强版）"
"$PYTHON_BIN" "$NDBC_ROOT/src/run_status_aware_router.py" \
  --config "$NDBC_ROOT/configs/default.yaml" \
  --strategy "$STRATEGY" \
  --model-key "$MODEL_KEY" \
  --use-timeout-aggregation-witness \
  --output-tag "${ROUTER_BASE_TAG}_fine"
