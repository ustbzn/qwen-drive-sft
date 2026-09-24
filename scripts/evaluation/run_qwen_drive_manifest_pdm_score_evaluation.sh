#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/qwen-drive/bin/python}"
MOGO_MANIFEST="${MOGO_MANIFEST:-${REPO_ROOT}/navsim/planning/script/config/common/mogo_dataset/nav_station.yaml}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/qwen_drive_runs/nav_station/qwen_drive_il/2026.09.21.02.50.50/lightning_logs/version_0/checkpoints/epoch=4-step=4925.ckpt}"
VLM_PATH="${VLM_PATH:-/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B}"
PLANNER_PATH="${PLANNER_PATH:-${VLM_PATH}/planner-sft}"
GPUS="${GPUS:-8}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_drive_il_pdm_eval}"
DATA_SPLIT="${DATA_SPLIT:-val}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NOISE_SEED="${NOISE_SEED:-42}"
export VLM_PATH PLANNER_PATH DATA_SPLIT NUM_INFERENCE_STEPS NUM_SAMPLES NOISE_SEED

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Qwen-Drive checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

cmd=(
  "${PYTHON}" "${REPO_ROOT}/scripts/evaluation/run_mogo_manifest_evaluation.py"
  --manifest "${MOGO_MANIFEST}"
  --eval-script "${REPO_ROOT}/scripts/evaluation/run_qwen_drive_agent_pdm_score_evaluation.sh"
  --experiment-name "${EXPERIMENT_NAME}"
  --checkpoint "${CHECKPOINT}"
  --gpus "${GPUS}"
  --master-port-start "${MASTER_PORT_START:-9696}"
)
if [[ -n "${MOGO_DATASET_NAMES:-}" ]]; then
  dataset_names="${MOGO_DATASET_NAMES//,/ }"
  read -r -a selected_datasets <<< "${dataset_names}"
  for name in "${selected_datasets[@]}"; do
    cmd+=(--name "${name}")
  done
fi
if [[ "${RESUME_EVALUATION:-false}" == "true" ]]; then
  cmd+=(--resume)
fi
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  cmd+=(--dry-run)
fi

exec "${cmd[@]}"
