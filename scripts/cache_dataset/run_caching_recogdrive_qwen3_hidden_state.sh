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
export USE_OBJECT_VECTOR=true

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_hdmap_cache}"
SQLITE_PATH="${SQLITE_PATH:-${OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite}"
MOGO_UTM_ZONE="${MOGO_UTM_ZONE:-50}"
GPUS="${GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-9696}"

# --- 障碍物 obj token 开关（默认关闭，零影响）。USE_OBJECT_VECTOR=true 时补出 obj 缓存。
# object_encoder_type=temporal -> 生成 object_temporal_feature_v1.gz（差分+时序FPN，动态运动意图/
#   静态几何约束，需 IL/RL/eval 同设 object_encoder_type=temporal）；
# object_encoder_type=mlp      -> 生成单帧 object_vector_feature_v1.gz（旧行为）。
# 指向已有 CACHE_PATH 重跑时：token 目录里已存在的 qwen3_feature*.gz / hdmap*.gz 会按 builder
# 逐个跳过（不重算 VLM，省数小时），只新增缺失的 object_temporal_feature_v1.gz。旧的
# object_vector_feature_v1.gz（若曾生成）残留无害、不被 temporal 读取。---
if [ "${USE_OBJECT_VECTOR:-false}" = "true" ]; then
  set -- \
    agent.use_object_vector=true \
    agent.object_encoder_type="${OBJECT_ENCODER_TYPE:-temporal}" \
    agent.static_speed_threshold="${STATIC_SPEED_THRESHOLD:-0.5}" \
    agent.max_obj_tokens="${MAX_OBJ_TOKENS:-64}" \
    agent.include_static_objects="${INCLUDE_STATIC_OBJECTS:-false}" \
    agent.object_token_dropout="${OBJECT_TOKEN_DROPOUT:-0.1}" \
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
  agent.vlm_type=qwen3 \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.use_hdmap_vector=true \
  mogo_map.sqlite_path="${SQLITE_PATH}" \
  mogo_map.utm_zone="${MOGO_UTM_ZONE}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${CACHE_PATH}" \
  force_cache_computation="${FORCE_CACHE_COMPUTATION:-False}" \
  "$@"

# "agent.view_list=[cam_f0]" \
#   agent.use_video_input=True \
#   agent.num_frames=4 \
python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all