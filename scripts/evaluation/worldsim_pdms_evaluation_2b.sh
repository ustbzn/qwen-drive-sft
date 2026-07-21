#!/usr/bin/env bash
set -euo pipefail

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

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export USE_OBJECT_VECTOR=true
# 前视图像 ResNet encoder 开关（默认关闭，零影响；仅实验用）。eval 走在线 backbone
# (cache_hidden_state=False)，图像由 FrontImageFeatureBuilder 从当前帧 cam_f0 现场读取，
# 不依赖缓存。类型/分辨率须与训练一致，CHECKPOINT 必须来自带 image_encoder 的 IL/RL 训练，
# 否则 image_encoder 随机初始化（initialize() strict=False 静默跳过）。
export USE_IMAGE_ENCODER="${USE_IMAGE_ENCODER:-false}"
# 自车状态编码器：mlp(默认) | attention(SDE)。eval 时不 dropout(self.training=False)，但类型
# 必须与训练 ckpt 一致，否则 attention 层被 initialize() strict=False 丢成随机初始化。
export EGO_STATUS_ENCODER="${EGO_STATUS_ENCODER:-attention}"
# VLM 总开关：true(默认) | false(完全不用 VLM)。use_vlm=false 时 eval 不加载 Qwen3（省一次
# 2B 在线前向），planner 无 vl 投影层、DiT context 仅 obj/hdmap/img；vlm_ablation 为 no-op。
# CHECKPOINT 必须来自 use_vlm=false 的 IL/RL 训练（否则 vl 投影层被 initialize() strict=False
# 静默丢弃，其余共享层按有-VLM 训练的权重加载，结果错配）。
# 用法：USE_VLM=false CHECKPOINT=<novlm ckpt> bash scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh
export USE_VLM="${USE_VLM:-false}"
if [ "${USE_VLM}" = "false" ]; then VLM_TAG="_novlm"; else VLM_TAG=""; fi

# --- badcase 评测模式开关（默认关闭）。开启后仅评测指定 token 列表中的样本。---
export BADCASE_MODE="${BADCASE_MODE:-false}"
export BADCASE_TOKEN_LIST="${BADCASE_TOKEN_LIST:-${REPO_ROOT}/navsim/local_config/data_selection/0312_0319_10case.txt}"

# --- GT 进度参考开关（默认开启）。模型始终推理，仅用 GT 归一化模型轨迹的 EP。---
# 兼容旧环境变量 USE_GT_TRAJECTORY；新调用请使用语义更明确的 USE_GT_PROGRESS_REFERENCE。
export USE_GT_PROGRESS_REFERENCE="${USE_GT_PROGRESS_REFERENCE:-${USE_GT_TRAJECTORY:-false}}"

# --- 变道场景中心线收敛满分模式（默认开启）。按场景 CSV 的 scene_merge=lane_change 识别。---
export LANE_CHANGE_CENTERLINE_FULL_SCORE="${LANE_CHANGE_CENTERLINE_FULL_SCORE:-false}"
SCENARIO_TYPE_CSV_PATH="${SCENARIO_TYPE_CSV_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/val_scenario_type_token_map.csv}"

# Update CHECKPOINT to the HDMap RL run after training completes.
# CHECKPOINT="${CHECKPOINT:-${NAVSIM_EXP_ROOT}/training_2603_ego_SDE_obj_hdmap_novlm_RL/2026.07.15.06.51.23/lightning_logs/version_0/checkpoints/epoch=19-step=140.ckpt}"
CHECKPOINT="/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/exp/training_2603_ego_SDE_obj_hdmap_novlm_RL/2026.07.15.06.51.23/lightning_logs/version_0/checkpoints/epoch=19-step=140.ckpt"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1}"
SQLITE_PATH="${SQLITE_PATH:-${OPENSCENE_DATA_ROOT}/maps/hm_shanghai_1.9.1_20260522.sqlite}"
MOGO_UTM_ZONE="${MOGO_UTM_ZONE:-51}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-9676}"

# --- 障碍物 obj token 开关（默认关闭，零影响）。开启后 agent.requires_scene=True，
# eval 从 GT annotations 取障碍物（方案 A privileged 上限）。CHECKPOINT 必须来自带 obj 的
# IL/RL 训练，否则 object_encoder 随机初始化（initialize() strict=False 静默跳过）。---
if [ "${USE_OBJECT_VECTOR:-false}" = "true" ]; then
  set -- \
    agent.use_object_vector=true \
    agent.object_encoder_type="${OBJECT_ENCODER_TYPE:-temporal}" \
    agent.static_speed_threshold="${STATIC_SPEED_THRESHOLD:-0.5}" \
    agent.max_obj_tokens="${MAX_OBJ_TOKENS:-64}" \
    agent.include_static_objects="${INCLUDE_STATIC_OBJECTS:-true}" \
    agent.object_token_dropout="${OBJECT_TOKEN_DROPOUT:-0.1}" \
    "$@"
fi

# --- 前视图像 encoder 开关（默认关闭，零影响；实验用）。CHECKPOINT 须来自带 image_encoder 的训练。---
if [ "${USE_IMAGE_ENCODER:-false}" = "true" ]; then
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
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  train_test_split.data_split=trainval \
  agent=recogdrive_agent \
  agent.checkpoint_path=\"${CHECKPOINT}\" \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  agent.cam_type=single \
  agent.grpo=False \
  agent.cache_hidden_state=False \
  agent.use_vlm="${USE_VLM}" \
  agent.vlm_ablation=zero \
  agent.vlm_type=qwen3 \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  agent.use_hdmap_vector=true \
  agent.use_lane_boundaries=true \
  agent.ego_status_encoder_type="${EGO_STATUS_ENCODER}" \
  mogo_map.sqlite_path="${SQLITE_PATH}" \
  mogo_map.utm_zone="${MOGO_UTM_ZONE}" \
  +badcase_mode="${BADCASE_MODE}" \
  +badcase_token_list="${BADCASE_TOKEN_LIST}" \
  +use_gt_progress_reference="${USE_GT_PROGRESS_REFERENCE}" \
  +lane_change_centerline_full_score="${LANE_CHANGE_CENTERLINE_FULL_SCORE}" \
  +scenario_type_csv_path="${SCENARIO_TYPE_CSV_PATH}" \
  experiment_name=v1_alldata_SDE_obj_hdmap${VLM_TAG}_IL_worldsim_eval \
  "$@"

# "agent.view_list=[cam_f0]" \
#   agent.use_video_input=True \
#   agent.num_frames=4 \
