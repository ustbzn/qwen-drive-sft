#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/qwen-drive/bin/python}"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0731-0802}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
DATASET_NAME="${DATASET_NAME:-beijing_0731-0802}"
EVAL_EXPERIMENT="${EVAL_EXPERIMENT:-qwen_drive_il_pdm_eval_${DATASET_NAME}}"
EVAL_DIR="${NAVSIM_EXP_ROOT}/${EVAL_EXPERIMENT}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/qwen_drive_runs/nav_station/qwen_drive_il/2026.09.21.02.50.50/lightning_logs/version_0/checkpoints/epoch=4-step=4925.ckpt}"
VLM_PATH="${VLM_PATH:-/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B}"
PLANNER_PATH="${PLANNER_PATH:-${VLM_PATH}/planner-sft}"
DATA_SPLIT="${DATA_SPLIT:-val}"
SQLITE_PATH="${SQLITE_PATH:-${OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite}"
MOGO_UTM_ZONE="${MOGO_UTM_ZONE:-50}"

export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT NAVSIM_EXP_ROOT
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CSV_PATH="${CSV_PATH:-}"
if [[ -z "${CSV_PATH}" ]]; then
  latest_csv="$(find "${EVAL_DIR}" -type f -name '*.csv' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  CSV_PATH="${latest_csv#* }"
fi
if [[ -z "${CSV_PATH}" || ! -f "${CSV_PATH}" ]]; then
  echo "Evaluation CSV not found. Set CSV_PATH or evaluate first under ${EVAL_DIR}." >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Qwen-Drive checkpoint not found: ${CHECKPOINT}" >&2
  exit 2
fi

score_args=()
if [[ -n "${MIN_SCORE:-}" ]]; then score_args+=(--min_score "${MIN_SCORE}"); fi
if [[ -n "${MAX_SCORE:-}" ]]; then score_args+=(--max_score "${MAX_SCORE}"); fi

exec "${PYTHON}" -m navsim.planning.script.plt_vis_from_csv_all \
  --agent_type qwen_drive \
  --csv "${CSV_PATH}" \
  --out "${OUT_DIR:-vis_output/eval/qwen_drive_${DATASET_NAME}}" \
  --checkpoint "${CHECKPOINT}" \
  --vlm_path "${VLM_PATH}" \
  --planner_path "${PLANNER_PATH}" \
  --openscene_data_root "${OPENSCENE_DATA_ROOT}" \
  --split "${DATA_SPLIT}" \
  --scene_filter mogo_qwen_drive \
  --num_history_frames 16 \
  --qwen_view_list cam_f0 cam_l0 cam_r0 \
  --qwen_num_camera_frames 4 \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-10}" \
  --num_samples "${NUM_SAMPLES:-1}" \
  --noise_seed "${NOISE_SEED:-42}" \
  --mogo_map_sqlite_path "${SQLITE_PATH}" \
  --mogo_map_utm_zone "${MOGO_UTM_ZONE}" \
  "${score_args[@]}" \
  "$@"
