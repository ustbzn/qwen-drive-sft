#!/usr/bin/env bash
set -euo pipefail

# default_training.yaml
# + common/default_evaluation.yaml
# + common/train_test_split/trainval.yaml
# + common/train_test_split/scene_filter/all_scenes.yaml
# + common/agent/recogdrive_agent.yaml
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

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_4frame_cache}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-9675}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py" \
  agent=recogdrive_agent \
  experiment_name=recogdrive_qwen3_agent_cache \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.cache_mode=True \
  agent.vlm_type=qwen3 \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  "agent.view_list=[cam_f0]" \
  agent.use_video_input=True \
  agent.num_frames=4 \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  force_cache_computation="${FORCE_CACHE_COMPUTATION:-False}" \
  "$@"
