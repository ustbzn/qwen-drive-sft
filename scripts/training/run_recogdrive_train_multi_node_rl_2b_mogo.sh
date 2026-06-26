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

CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/recogdrive_qwen3_notrain_hdmap_cache}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v2}"
# Update CHECKPOINT to the HDMap IL run after training completes.
CHECKPOINT="${CHECKPOINT:-${NAVSIM_EXP_ROOT}/training_recogdrive_qwen3_hdmap_IL/2026.06.23.13.32.10/lightning_logs/version_0/checkpoints/epoch=151-step=16264.ckpt}"
GPUS="${GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-9696}"

echo "NAVSIM_EXP_ROOT:    ${NAVSIM_EXP_ROOT}"
echo "CACHE_PATH:         ${CACHE_PATH}"
echo "METRIC_CACHE_PATH:  ${METRIC_CACHE_PATH}"
echo "CHECKPOINT:         ${CHECKPOINT}"
echo "GPUS:               ${GPUS}"

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
  agent.vlm_type=qwen3 \
  agent.checkpoint_path=\"${CHECKPOINT}\" \
  agent.vl_hidden_dim=2048 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  agent.use_hdmap_vector=true \
  agent.metric_cache_path="${METRIC_CACHE_PATH}" \
  agent.reference_policy_checkpoint=\"${CHECKPOINT}\" \
  trainer.params.max_epochs=20 \
  trainer.params.num_nodes=1 \
  trainer.params.devices="${GPUS}" \
  dataloader.params.batch_size=16 \
  experiment_name=training_recogdrive_qwen3_hdmap_RL \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  train_logs=null \
  val_logs=null \
  train_token_whitelist_path=navsim/local_config/data_selection/rl_grpo_n4000_selected_tokens_v1.txt \
  cache_path="${CACHE_PATH}" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  "$@" > train_recogdrive_qwen3_hdmap_rl_2b.txt 2>&1

python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all
