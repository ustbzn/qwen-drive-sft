#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319}"
QWEN3_VL_PATH="${QWEN3_VL_PATH:-/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-mogo_trainval}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${DATA_ROOT}/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_cache}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-9675}"
BATCH_SIZE="${BATCH_SIZE:-80}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"

torchrun \
  --nnodes=1 \
  --node_rank="${MLP_ROLE_INDEX:-0}" \
  --master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MLP_WORKER_0_PORT:-${MASTER_PORT}}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.grpo=False \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.vlm_type=qwen3 \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=flow \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes=1 \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  experiment_name=training_recogdrive_qwen3_flow_IL \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  '+log_split@_global_=mogo' \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  "$@"


python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all