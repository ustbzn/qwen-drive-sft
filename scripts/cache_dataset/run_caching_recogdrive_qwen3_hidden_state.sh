#!/usr/bin/env bash
set -euo pipefail

# default_training.yaml
# + common/default_evaluation.yaml
# + common/train_test_split/trainval.yaml
# + common/train_test_split/scene_filter/all_scenes.yaml
# + common/agent/recogdrive_agent.yaml
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai}"
QWEN3_VL_PATH="${QWEN3_VL_PATH:-/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-mogo_trainval}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${DATA_ROOT}/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export USE_OBJECT_VECTOR=true
# 前视图像 ResNet encoder 开关（默认关闭，零影响；仅实验用）。USE_IMAGE_ENCODER=true 时
# 本阶段额外补出 front_image_feature_v1_<view>_<H>x<W>.gz（VLM 不重算，只多写图像 .gz）。
# 须与 IL/RL/eval 保持一致，否则 get_unique_name 对不上、CacheOnlyDataset 会报告缺失并过滤不完整样本。
export USE_IMAGE_ENCODER="${USE_IMAGE_ENCODER:-false}"

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_obj_hdmap_cache}"
SQLITE_PATH="${SQLITE_PATH:-${OPENSCENE_DATA_ROOT}/maps/hm_shanghai_1.9.1_20260522.sqlite}"
MOGO_UTM_ZONE="${MOGO_UTM_ZONE:-51}"
MOGO_UTM_SOUTH="${MOGO_UTM_SOUTH:-false}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-9675}"
SLIM_CACHE_ONLY="${SLIM_CACHE_ONLY:-false}"

CACHE_USE_VLM=true
CACHE_USE_HDMAP=true
CACHE_SQLITE_PATH="${SQLITE_PATH}"
if [ "${SLIM_CACHE_ONLY}" = "true" ]; then
  CACHE_USE_VLM=false
  CACHE_USE_HDMAP=false
  CACHE_SQLITE_PATH=""
fi

# --- 障碍物 obj token 开关（默认关闭，零影响）。USE_OBJECT_VECTOR=true 时补出 obj 缓存。
# object_encoder_type=temporal -> 生成 object_temporal_feature_v1.gz（差分+时序FPN，动态运动意图/
#   静态几何约束，需 IL/RL/eval 同设 object_encoder_type=temporal）；
# object_encoder_type=mlp      -> 生成单帧 object_vector_feature_v1.gz（旧行为）。
# 正常模式同时写 qwen3_feature*.gz 与 recogdrive_ego_feature_v1.gz。指向已有 CACHE_PATH
# 重跑时，已存在文件按 builder 逐个跳过，不重算 VLM；SLIM_CACHE_ONLY=true 时只补 ego
# sidecar，并关闭 Qwen、HDMap、object、image 与地图加载。旧 object 缓存残留无害。---
if [ "${SLIM_CACHE_ONLY}" != "true" ] && [ "${USE_OBJECT_VECTOR:-false}" = "true" ]; then
  set -- \
    agent.use_object_vector=true \
    agent.object_encoder_type="${OBJECT_ENCODER_TYPE:-temporal}" \
    agent.static_speed_threshold="${STATIC_SPEED_THRESHOLD:-0.5}" \
    agent.max_obj_tokens="${MAX_OBJ_TOKENS:-64}" \
    agent.include_static_objects="${INCLUDE_STATIC_OBJECTS:-false}" \
    agent.object_token_dropout="${OBJECT_TOKEN_DROPOUT:-0.1}" \
    "$@"
fi

# --- 前视图像 encoder 开关（默认关闭，零影响；实验用）。开启补出前视图 uint8 缓存。---
if [ "${SLIM_CACHE_ONLY}" != "true" ] && [ "${USE_IMAGE_ENCODER:-false}" = "true" ]; then
  set -- \
    agent.use_image_encoder=true \
    agent.image_encoder_arch="${IMAGE_ENCODER_ARCH:-resnet34}" \
    agent.image_encoder_pretrained="${IMAGE_ENCODER_PRETRAINED:-true}" \
    agent.image_height="${IMAGE_HEIGHT:-256}" \
    agent.image_width="${IMAGE_WIDTH:-704}" \
    agent.image_grid_h="${IMAGE_GRID_H:-8}" \
    agent.image_grid_w="${IMAGE_GRID_W:-16}" \
    agent.image_view="${IMAGE_VIEW:-cam_f0}" \
    "$@"
fi

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
  agent.use_vlm="${CACHE_USE_VLM}" \
  agent.vlm_type=qwen3 \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.use_hdmap_vector="${CACHE_USE_HDMAP}" \
  agent.use_lane_boundaries="${CACHE_USE_HDMAP}" \
  mogo_map.sqlite_path="${CACHE_SQLITE_PATH}" \
  mogo_map.utm_zone="${MOGO_UTM_ZONE}" \
  mogo_map.utm_south="${MOGO_UTM_SOUTH}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  force_cache_computation="${FORCE_CACHE_COMPUTATION:-False}" \
  "$@"

# "agent.view_list=[cam_f0]" \
#   agent.use_video_input=True \
#   agent.num_frames=4 \
if [ "${SLIM_CACHE_ONLY}" != "true" ]; then
  python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all
fi
