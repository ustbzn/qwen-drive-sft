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
export USE_OBJECT_VECTOR=true
# 自车状态编码器：mlp(默认，零影响) | attention(SDE，训练时随机遮蔽运动学量以抑制外推捷径)。
# 与 vlm_ablation 正交(none/zero 均生效)。IL/RL/eval 必须三处一致，否则 ckpt 加载错配
# (attention 层被 initialize() strict=False 丢成随机初始化)。
export EGO_STATUS_ENCODER="${EGO_STATUS_ENCODER:-attention}"

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_obj_hdmap_cache}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-9675}"
BATCH_SIZE="${BATCH_SIZE:-160}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"

# --- 障碍物 obj token 开关（默认关闭，零影响）。开启需与缓存端用同样的 obj 配置，
# 且 CACHE_PATH 必须已带 object_vector_feature_v1.gz，否则 CacheOnlyDataset 静默丢样本。---
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
  agent.sampling_method=ddim \
  agent.use_hdmap_vector=true \
  agent.use_lane_boundaries=true \
  agent.ego_status_encoder_type="${EGO_STATUS_ENCODER}" \
  agent.ego_state_dropout="${EGO_STATE_DROPOUT:-0.3}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes=1 \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  experiment_name=training_recogdrive_qwen3_ego_SDE_obj_hdmap_IL \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  '+log_split@_global_=mogo' \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  "$@"


python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all

# "agent.view_list=[cam_f0]" \
#   agent.use_video_input=True \
#   agent.num_frames=4 \
# agent.vlm_ablation=zero \
# trainer.params.strategy=ddp_find_unused_parameters_true \