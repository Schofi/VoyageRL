#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${MODEL_PATH:?Set MODEL_PATH to a model directory or model ID}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/examples/inference.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/saved/inference/results.jsonl}"
require_file "${DATASET_PATH}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"
python run_tool_loop_infer.py \
  --infer_backend "${INFER_BACKEND:-transformers}" \
  --dataset_path "${DATASET_PATH}" \
  --model_dir "${MODEL_PATH}" \
  --tools_dir "${PROJECT_ROOT}/tools" \
  --start_idx "${START_IDX:-0}" \
  --num_samples "${NUM_SAMPLES:-1}" \
  --max_turns "${MAX_TURNS:-13}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-5000}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top_p 0.95 \
  --top_k 50 \
  --tool_first_enforce \
  --system_max_tool_calls "${TOOL_LOOP_SYSTEM_MAX_TOOL_CALLS:-13}" \
  --max_same_tool_call_rounds 3 \
  --save_full_messages \
  --metrics_output "${INFERENCE_METRICS_PATH:-${PROJECT_ROOT}/saved/monitor/inference.jsonl}" \
  --output_path "${OUTPUT_PATH}"
