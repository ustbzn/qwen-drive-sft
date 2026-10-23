#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/qwen-drive/bin/python}"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0731-0802}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/qwen_drive_runs/nav_station_mixed/qwen_drive_il_6cam/2026.10.19.09.29.48/lightning_logs/version_0/checkpoints/epoch=1-step=1784.ckpt}"
VLM_PATH="${VLM_PATH:-/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B}"
PLANNER_PATH="${PLANNER_PATH:-${VLM_PATH}/planner-sft}"
QWEN_VIEW_LIST="${QWEN_VIEW_LIST:-[cam_f0,cam_l0,cam_r0,cam_l1,cam_r1,cam_b0]}"
DATA_SPLIT="${DATA_SPLIT:-val}"
GPUS="${GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-9696}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_drive_il_6cam_eval}"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT NAVSIM_EXP_ROOT
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Qwen-Drive checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

exec "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${GPUS}" \
  --master_port="${MASTER_PORT}" \
  --module navsim.planning.script.run_pdm_score_qwen \
  train_test_split=mogo_qwen_drive \
  "train_test_split.data_split=${DATA_SPLIT}" \
  agent=qwen_drive_agent \
  "agent.vlm_path=${VLM_PATH}" \
  "agent.planner_path=${PLANNER_PATH}" \
  "agent.checkpoint_path=\"${CHECKPOINT}\"" \
  "agent.view_list=${QWEN_VIEW_LIST}" \
  "agent.num_inference_steps=${NUM_INFERENCE_STEPS:-10}" \
  "agent.num_samples=${NUM_SAMPLES:-1}" \
  "agent.noise_seed=${NOISE_SEED:-42}" \
  "metric_cache_path=${METRIC_CACHE_PATH}" \
  "experiment_name=${EXPERIMENT_NAME}" \
  "$@"
