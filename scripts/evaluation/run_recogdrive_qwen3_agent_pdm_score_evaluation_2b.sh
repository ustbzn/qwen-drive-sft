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

# Update CHECKPOINT to the HDMap RL run after training completes.
CHECKPOINT="${CHECKPOINT:-${NAVSIM_EXP_ROOT}/training_recogdrive_qwen3_hdmap_RL/2026.06.24.08.21.53/lightning_logs/version_0/checkpoints/epoch=19-step=10680.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v2}"
SQLITE_PATH="${SQLITE_PATH:-${OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite}"
MOGO_UTM_ZONE="${MOGO_UTM_ZONE:-50}"
GPUS="${GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-9696}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  train_test_split.data_split=val \
  agent=recogdrive_agent \
  agent.checkpoint_path=\"${CHECKPOINT}\" \
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
  agent.use_hdmap_vector=true \
  mogo_map.sqlite_path="${SQLITE_PATH}" \
  mogo_map.utm_zone="${MOGO_UTM_ZONE}" \
  experiment_name=recogdrive_qwen3_hdmap_ddim_IL_eval \
  "$@"
