#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-mogo_trainval}"
DATA_SPLIT="${DATA_SPLIT:-val}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${DATA_ROOT}/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v3}"
GT_TOKEN_PATH="${GT_TOKEN_PATH:-null}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gt_pdm_score_val}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gt.py" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  train_test_split.data_split="${DATA_SPLIT}" \
  metric_cache_path="${METRIC_CACHE_PATH}" \
  gt_token_path="${GT_TOKEN_PATH}" \
  experiment_name="${EXPERIMENT_NAME}" \
  "$@"
