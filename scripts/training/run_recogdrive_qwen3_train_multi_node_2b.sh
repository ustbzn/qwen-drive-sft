#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319}"
QWEN3_VL_PATH="${QWEN3_VL_PATH:-/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-mogo_trainval}"

# --- 多城市数据集：脚本内置默认 manifest，直接 bash 启动即多城市 ------------------------
# 直接 `bash scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh` 即用下面的 manifest，
# 自动派生 CACHE_PATHS、LOG_SPLIT(=manifest 文件名 stem)、NAVSIM_EXP_ROOT(=顶层 train_exp_root)。
# 覆盖：MOGO_MANIFEST=<其他 manifest> 换数据集；MOGO_MANIFEST= （置空）回到单城市旧行为；
# MOGO_NAMES="a,b" 只跑子集；显式传 CACHE_PATHS/LOG_SPLIT/NAVSIM_EXP_ROOT 亦优先。
MOGO_MANIFEST="${MOGO_MANIFEST-${REPO_ROOT}/navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml}"
LOG_SPLIT="${LOG_SPLIT:-}"
if [ -n "${MOGO_MANIFEST:-}" ]; then
  MOGO_PATHS_HELPER="${REPO_ROOT}/scripts/cache_dataset/mogo_manifest_paths.py"
  MOGO_NAMES_ARGS=""
  [ -n "${MOGO_NAMES:-}" ] && MOGO_NAMES_ARGS="--names ${MOGO_NAMES}"
  if [ -z "${CACHE_PATHS:-}" ]; then
    CACHE_PATHS="$(python "${MOGO_PATHS_HELPER}" --manifest "${MOGO_MANIFEST}" --field cache_path ${MOGO_NAMES_ARGS})"
  fi
  CACHE_PATH="${CACHE_PATH:-${CACHE_PATHS%%:*}}"
  [ -z "${LOG_SPLIT}" ] && LOG_SPLIT="$(basename "${MOGO_MANIFEST}" .yaml)"
  MOGO_TRAIN_EXP_ROOT="$(python "${MOGO_PATHS_HELPER}" --manifest "${MOGO_MANIFEST}" --field train_exp_root --top-level)"
  [ -n "${MOGO_TRAIN_EXP_ROOT}" ] && [ -z "${NAVSIM_EXP_ROOT:-}" ] && export NAVSIM_EXP_ROOT="${MOGO_TRAIN_EXP_ROOT}"
fi
LOG_SPLIT="${LOG_SPLIT:-mogo}"

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
# 前视图像 ResNet encoder 开关（默认关闭，零影响；仅实验用）。开启需与缓存端一致，且
# CACHE_PATH 必须已带 front_image_feature_*.gz，否则 CacheOnlyDataset 会报告缺失并过滤不完整样本。
# 典型实验：与 agent.vlm_ablation=zero 组合 —— 屏蔽 VLM 语义、保留前视图像视觉信息。
export USE_IMAGE_ENCODER="${USE_IMAGE_ENCODER:-false}"
# 自车状态编码器：mlp(默认，零影响) | attention(SDE，训练时随机遮蔽运动学量以抑制外推捷径)。
# 与 vlm_ablation 正交(none/zero 均生效)。IL/RL/eval 必须三处一致，否则 ckpt 加载错配
# (attention 层被 initialize() strict=False 丢成随机初始化)。
export EGO_STATUS_ENCODER="${EGO_STATUS_ENCODER:-attention}"
# VLM 总开关：true(默认，加载并使用 VLM，与历史一致) | false(完全不用 VLM：planner 不建 VLM
# 投影层、DiT cross-attention context 仅由 obj/hdmap/img token 组成、agent 不加载 Qwen3；模型更小)。
# use_vlm=false 时训练只读 recogdrive_ego_feature_v1.gz，不再检查或解压 qwen3_feature.gz；
# 旧 CACHE_PATH 须先用 --slim-only 补写 sidecar。vlm_ablation 为 no-op，ckpt 不可跨 use_vlm 移植，
# RL/eval 的 CHECKPOINT 必须来自 use_vlm=false 的上游（否则 vl 投影层缺失，行为错配）。
# 用法：USE_VLM=false bash scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh
export USE_VLM="${USE_VLM:-false}"
if [ "${USE_VLM}" = "false" ]; then VLM_TAG="_novlm"; else VLM_TAG=""; fi

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_obj_hdmap_cache}"
GPUS="${GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-9676}"
BATCH_SIZE="${BATCH_SIZE:-512}" #2400
MAX_EPOCHS="${MAX_EPOCHS:-20}"
# PRETRAINED_CKPT="${PRE_TRAINED_CKPT:-}"
PRETRAINED_CKPT="${PRE_TRAINED_CKPT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/mogo_multi/exp/v2_alldata_traffic_light_novlm_IL/2026.07.29.10.01.45/lightning_logs/version_0/checkpoints/epoch=69-step=9940.ckpt}"

CACHE_ARGS=(cache_path="${CACHE_PATH}")
if [ -n "${CACHE_PATHS:-}" ]; then
  IFS=':' read -r -a CACHE_PATH_ARRAY <<< "${CACHE_PATHS}"
  CACHE_PATH_LIST=""
  for CACHE_ITEM in "${CACHE_PATH_ARRAY[@]}"; do
    if [ -z "${CACHE_PATH_LIST}" ]; then
      CACHE_PATH_LIST="${CACHE_ITEM}"
    else
      CACHE_PATH_LIST="${CACHE_PATH_LIST},${CACHE_ITEM}"
    fi
  done
  CACHE_ARGS=(cache_path="${CACHE_PATH}" "cache_paths=[${CACHE_PATH_LIST}]")
fi

# --- 障碍物 obj token 开关（默认关闭，零影响）。开启需与缓存端用同样的 obj 配置，
# 且 CACHE_PATH 必须已带 object_vector_feature_v1.gz，否则 CacheOnlyDataset 会报告缺失并过滤不完整样本。---
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

# --- 前视图像 encoder 开关（默认关闭，零影响；实验用）。开启需 CACHE_PATH 已带前视图 .gz。---
if [ "${USE_IMAGE_ENCODER:-false}" = "true" ]; then
  set -- \
    agent.use_image_encoder=true \
    agent.image_encoder_arch="${IMAGE_ENCODER_ARCH:-resnet34}" \
    agent.image_encoder_pretrained="${IMAGE_ENCODER_PRETRAINED:-true}" \
    agent.image_height="${IMAGE_HEIGHT:-256}" \
    agent.image_width="${IMAGE_WIDTH:-704}" \
    agent.image_grid_h="${IMAGE_GRID_H:-8}" \
    agent.image_grid_w="${IMAGE_GRID_W:-16}" \
    agent.image_token_dropout="${IMAGE_TOKEN_DROPOUT:-0.0}" \
    agent.image_view="${IMAGE_VIEW:-cam_f0}" \
    "$@"
fi

#finetune lr=2e-5/orgin lr=1e-4
torchrun \
  --nnodes=1 \
  --node_rank="${MLP_ROLE_INDEX:-0}" \
  --master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MLP_WORKER_0_PORT:-${MASTER_PORT}}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive.py" \
  agent=recogdrive_agent \
  agent.lr=2e-5 \
  agent.grpo=False \
  agent.checkpoint_path=\"${PRETRAINED_CKPT}\" \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.vlm_type=qwen3 \
  agent.use_vlm="${USE_VLM}" \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  agent.use_hdmap_vector=true \
  agent.use_lane_boundaries=true \
  agent.hdmap_input_norm=false \
  agent.ego_status_encoder_type="${EGO_STATUS_ENCODER}" \
  agent.ego_state_dropout="${EGO_STATE_DROPOUT:-0.75}" \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  trainer.params.num_nodes=1 \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  experiment_name=v3_alldata_traffic_light_Pretrain_lowBS${VLM_TAG}_IL \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  "+log_split@_global_=${LOG_SPLIT}" \
  "${CACHE_ARGS[@]}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  "$@"


python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all

# "agent.view_list=[cam_f0]" \
#   agent.use_video_input=True \
#   agent.num_frames=4 \
# agent.vlm_ablation=zero \
# trainer.params.strategy=ddp_find_unused_parameters_true \