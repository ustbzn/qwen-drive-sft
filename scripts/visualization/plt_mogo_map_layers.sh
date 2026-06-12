#!/usr/bin/env bash
# Visualize Mogo SQLite map layers (LANE / ROADBLOCK / connectors / INTERSECTION).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/vis_output/map}"

mkdir -p "${OUT_DIR}"

cd "${REPO_ROOT}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate recogdrive 2>/dev/null || true
fi

python navsim/planning/script/plt_mogo_map_layers.py \
  --data-root "${DATA_ROOT}" \
  --pkl "${PKL:-JLBJA20073D_0317__2026-03-17-15-35-00__20260317153639.pkl}" \
  --frame-idx "${FRAME_IDX:--1}" \
  --show-baselines \
  --output-dir "${OUT_DIR}"

echo "Per-layer png files written to ${OUT_DIR}/"
