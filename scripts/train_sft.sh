#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ===== SFT config =====
: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or model ID}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/final/sft_train.jsonl}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/final/sft_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/saved/sft}"
require_file "${TRAIN_DATA}"
require_file "${VAL_DATA}"

# Multi-GPU setup
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

swift sft \
  --model "${MODEL_PATH}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type lora \
  --lora_rank 128 \
  --num_train_epochs 2 \
  --learning_rate 8e-6 \
  --max_length 32768 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --dataloader_num_workers 8 \
  --save_steps 200 \
  --eval_steps 50 \
  --logging_steps 5 \
  --output_dir "${OUTPUT_DIR}"
