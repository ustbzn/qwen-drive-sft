#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20}"
QWEN3_VL_PATH="${QWEN3_VL_PATH:-/pcpt/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-trainval}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a Qwen3 ReCogDrive planner checkpoint}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${DATA_ROOT}/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-63669}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  agent=recogdrive_agent \
  agent.checkpoint_path="${CHECKPOINT}" \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  agent.cam_type=single \
  agent.grpo=False \
  agent.cache_hidden_state=False \
  agent.vlm_type=qwen3 \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  experiment_name=recogdrive_qwen3_agent_eval \
  "$@"
