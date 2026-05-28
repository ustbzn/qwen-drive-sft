#!/usr/bin/env bash
# 单 token 轨迹调试（IL vs RL、多采样、去噪链、PDM 指标）
# 用法: bash scripts/visualization/debug_token_trajectory.sh [TOKEN]

set -euo pipefail

TOKEN="${1:-1bdfacbfcff75c27}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/NXT/lst/vla/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/NXT/lst/vla/recog_drive"
export OPENSCENE_DATA_ROOT="/NXT/lst/vla/navsim/download"

CHECKPOINT_RL="/NXT/lst/vla/recog_drive/checkpoints/RL/train_rl_hs0_hsL_4.ckpt"
CHECKPOINT_IL="/NXT/lst/vla/recog_drive/checkpoints/IL/train_il_hs0_hsL_4.ckpt"
VLM_PATH="/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B"
METRIC_CACHE="/NXT/lst/vla/navsim/exp/metric_cache2"
OUT_DIR="vis_output/debug/${TOKEN}"

python -m navsim.planning.script.debug_token_trajectory \
  --token "${TOKEN}" \
  --checkpoint_rl "${CHECKPOINT_RL}" \
  --checkpoint_il "${CHECKPOINT_IL}" \
  --vlm_path "${VLM_PATH}" \
  --metric_cache_path "${METRIC_CACHE}" \
  --openscene_data_root "${OPENSCENE_DATA_ROOT}" \
  --split test \
  --scene_filter navtest \
  --num_stochastic 5 \
  --seed 42 \
  --out "${OUT_DIR}"

echo "Done. See ${OUT_DIR}/metrics.csv and compare_checkpoints.png"
