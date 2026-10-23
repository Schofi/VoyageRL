#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Base SFT checkpoint used for RL.
: "${MODEL_PATH:?Set MODEL_PATH to your merged SFT checkpoint}"

# Data and plugin paths.
SCHEDULER_PLUGIN="${SCHEDULER_PLUGIN:-${PROJECT_ROOT}/training/plugins/tooluse_multi_turn_scheduler.py}"

# Rollout server config.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

MAX_TURNS="${MAX_TURNS:-13}"
MAX_LENGTH="${MAX_LENGTH:-50000}"

# 设置正确的NCCL参数
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

# start rollout server in another terminal if it is not already running.
CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0,1}" \
swift rollout \
  --host "${VLLM_SERVER_HOST:-127.0.0.1}" \
  --port "${VLLM_SERVER_PORT:-8000}" \
  --model "${MODEL_PATH}" \
  --external_plugins "${SCHEDULER_PLUGIN}" \
  --multi_turn_scheduler travel_tool_loop \
  --vllm_use_async_engine true \
  --vllm_max_model_len "${MAX_LENGTH}" \
  --vllm_tensor_parallel_size "${ROLLOUT_TP_SIZE:-2}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max_turns "${MAX_TURNS}"
