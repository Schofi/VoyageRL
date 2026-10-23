#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${MODEL_PATH:?Set MODEL_PATH to the base model}"
: "${ADAPTER_PATH:?Set ADAPTER_PATH to the trained LoRA checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/saved/sft-merged}"
swift export \
  --model "${MODEL_PATH}" \
  --adapters "${ADAPTER_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --merge_lora true \
  --safe_serialization true
