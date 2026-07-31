#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319}"
QWEN3_VL_PATH="${QWEN3_VL_PATH:-/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-mogo_trainval}"

# --- 多城市数据集：脚本内置默认 manifest，直接 bash 启动即多城市 ------------------------
# 直接 `bash scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh` 即用下面的 manifest，
# 自动派生 CACHE_PATHS、METRIC_CACHE_PATHS、NAVSIM_EXP_ROOT(=顶层 train_exp_root)。
# 覆盖：MOGO_MANIFEST=<其他 manifest> 换数据集；MOGO_MANIFEST= 置空回到单城市旧行为；MOGO_NAMES="a,b" 只跑子集。
# 注意：RL 用 token 白名单而非 log_split，多城市白名单须覆盖目标城市；MetricCacheLoader 多 metric root
# 间发现重复 token 会直接报错（数据生成需保证 token 全局唯一）。CHECKPOINT 须来自同结构多城市 IL。
MOGO_MANIFEST="${MOGO_MANIFEST-${REPO_ROOT}/navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml}"
if [ -n "${MOGO_MANIFEST:-}" ]; then
  MOGO_PATHS_HELPER="${REPO_ROOT}/scripts/cache_dataset/mogo_manifest_paths.py"
  MOGO_NAMES_ARGS=""
  [ -n "${MOGO_NAMES:-}" ] && MOGO_NAMES_ARGS="--names ${MOGO_NAMES}"
  if [ -z "${CACHE_PATHS:-}" ]; then
    CACHE_PATHS="$(python "${MOGO_PATHS_HELPER}" --manifest "${MOGO_MANIFEST}" --field cache_path ${MOGO_NAMES_ARGS})"
  fi
  CACHE_PATH="${CACHE_PATH:-${CACHE_PATHS%%:*}}"
  if [ -z "${METRIC_CACHE_PATHS:-}" ]; then
    METRIC_CACHE_PATHS="$(python "${MOGO_PATHS_HELPER}" --manifest "${MOGO_MANIFEST}" --field metric_cache_path ${MOGO_NAMES_ARGS})"
  fi
  MOGO_TRAIN_EXP_ROOT="$(python "${MOGO_PATHS_HELPER}" --manifest "${MOGO_MANIFEST}" --field train_exp_root --top-level)"
  [ -n "${MOGO_TRAIN_EXP_ROOT}" ] && [ -z "${NAVSIM_EXP_ROOT:-}" ] && export NAVSIM_EXP_ROOT="${MOGO_TRAIN_EXP_ROOT}"
fi

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${DATA_ROOT}/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_obj_hdmap_cache}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1}"

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
if [ -n "${METRIC_CACHE_PATHS:-}" ]; then
  METRIC_CACHE_PATH="${METRIC_CACHE_PATHS}"
fi
# Update CHECKPOINT to the HDMap IL run after training completes.
# CHECKPOINT="${CHECKPOINT:-${NAVSIM_EXP_ROOT}/training_recogdrive_ego_SDE_RL/2026.07.06.03.56.28/lightning_logs/version_0/checkpoints/epoch=19-step=640.ckpt}"
CHECKPOINT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/mogo_multi/exp/v1.1_alldata_traffic_light_novlm_IL/2026.07.24.08.56.05/lightning_logs/version_0/checkpoints/epoch=66-step=6231.ckpt"
MASTER_PORT="${MASTER_PORT:-9675}"
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}
export USE_OBJECT_VECTOR=true
# 前视图像 ResNet encoder 开关（默认关闭，零影响；仅实验用）。开启需与 IL 缓存端一致，且
# CHECKPOINT / reference_policy_checkpoint 必须来自带 image_encoder 的 IL 训练
# （_init_grpo strict=True，缺 image_encoder.* 会报缺键）。常与 vlm_ablation=zero 组合实验。
export USE_IMAGE_ENCODER="${USE_IMAGE_ENCODER:-false}"
# 自车状态编码器：mlp(默认) | attention(SDE)。与 vlm_ablation 正交；IL/RL/eval 必须一致，
# 且 CHECKPOINT/reference_policy_checkpoint 须来自同类型 IL(_init_grpo 对 ego_status_encoder 结构敏感)。
export EGO_STATUS_ENCODER="${EGO_STATUS_ENCODER:-attention}"
# VLM 总开关：true(默认) | false(完全不用 VLM，模型更小)。use_vlm=false 时 planner 无 vl 投影层，
# 缓存只读 recogdrive_ego_feature_v1.gz；旧 CACHE_PATH 须先用 --slim-only 补写 sidecar。
# vlm_ablation 为 no-op；且 _init_grpo 的 reference 加载是 strict=True——CHECKPOINT/reference
# 必须来自 use_vlm=false 的 IL 训练（novlm），否则会静默把带 VLM 的权重灌进无 VLM 结构。
# 用法：USE_VLM=false CHECKPOINT=<novlm IL ckpt> bash scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh
export USE_VLM="${USE_VLM:-false}"
if [ "${USE_VLM}" = "false" ]; then VLM_TAG="_novlm"; else VLM_TAG=""; fi

echo "NAVSIM_EXP_ROOT:    ${NAVSIM_EXP_ROOT}"
echo "CACHE_PATH:         ${CACHE_PATH}"
echo "METRIC_CACHE_PATH:  ${METRIC_CACHE_PATH}"
echo "CHECKPOINT:         ${CHECKPOINT}"
echo "GPUS:               ${GPUS}"

# --- 障碍物 obj token 开关（默认关闭，零影响）。开启需与 IL 缓存端 obj 配置一致，且
# CHECKPOINT / reference_policy_checkpoint 必须来自带 obj 的 IL 训练（_init_grpo strict=True，
# 缺 object_encoder.* 会报缺键）。---
OBJ_ARGS=()
if [ "${USE_OBJECT_VECTOR:-false}" = "true" ]; then
  OBJ_ARGS=(
    agent.use_object_vector=true
    agent.object_encoder_type="${OBJECT_ENCODER_TYPE:-temporal}" 
    agent.static_speed_threshold="${STATIC_SPEED_THRESHOLD:-0.5}" 
    agent.max_obj_tokens="${MAX_OBJ_TOKENS:-64}"
    agent.include_static_objects="${INCLUDE_STATIC_OBJECTS:-true}"
    agent.object_token_dropout="${OBJECT_TOKEN_DROPOUT:-0.1}"
  )
fi

# --- 前视图像 encoder 开关（默认关闭，零影响；实验用）。CHECKPOINT 须来自带 image_encoder 的 IL。---
IMG_ARGS=()
if [ "${USE_IMAGE_ENCODER:-false}" = "true" ]; then
  IMG_ARGS=(
    agent.use_image_encoder=true
    agent.image_encoder_arch="${IMAGE_ENCODER_ARCH:-resnet34}"
    agent.image_encoder_pretrained="${IMAGE_ENCODER_PRETRAINED:-true}"
    agent.image_height="${IMAGE_HEIGHT:-256}"
    agent.image_width="${IMAGE_WIDTH:-704}"
    agent.image_grid_h="${IMAGE_GRID_H:-8}"
    agent.image_grid_w="${IMAGE_GRID_W:-16}"
    agent.image_token_dropout="${IMAGE_TOKEN_DROPOUT:-0.0}"
    agent.image_view="${IMAGE_VIEW:-cam_f0}"
  )
fi

torchrun \
  --nnodes=1 \
  --node_rank="${MLP_ROLE_INDEX:-0}" \
  --master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}" \
  --nproc_per_node="${GPUS}" \
  --master_port="${MLP_WORKER_0_PORT:-${MASTER_PORT}}" \
  "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py" \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.vlm_path="${QWEN3_VL_PATH}" \
  agent.cam_type=single \
  agent.grpo=True \
  agent.cache_hidden_state=True \
  agent.use_vlm="${USE_VLM}" \
  agent.vlm_ablation=zero \
  agent.vlm_type=qwen3 \
  agent.checkpoint_path=\"${CHECKPOINT}\" \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  agent.use_hdmap_vector=true \
  agent.use_lane_boundaries=true \
  agent.ego_status_encoder_type="${EGO_STATUS_ENCODER}" \
  agent.ego_state_dropout="${EGO_STATE_DROPOUT:-0.75}" \
  agent.metric_cache_path="${METRIC_CACHE_PATH}" \
  agent.reference_policy_checkpoint=\"${CHECKPOINT}\" \
  trainer.params.max_epochs=20 \
  trainer.params.num_nodes=1 \
  trainer.params.devices="${GPUS}" \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  dataloader.params.batch_size=80 \
  experiment_name=v1.1_alldata_traffic_light${VLM_TAG}_RL \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  train_logs=null \
  val_logs=null \
  train_token_whitelist_path=navsim/local_config/data_selection/rl_grpo_merged_results_n13000_selected_tokens_v1.txt \
  "${CACHE_ARGS[@]}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  "${OBJ_ARGS[@]}" "${IMG_ARGS[@]}" > train_recogdrive_qwen3_agent_rl_2b.txt 2>&1

python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all

# agent.neighbor_sigma=0.5 \
# agent.quasi_norm_p=2.0 \
# agent.num_anchors=0 \
# agent.neighbor_adaptive_temp=True \
# agent.neighbor_leap_temperature=null \

# "agent.view_list=[cam_f0]" \
# agent.use_video_input=True \
# agent.num_frames=4 \
# '+log_split@_global_=mogo' \


