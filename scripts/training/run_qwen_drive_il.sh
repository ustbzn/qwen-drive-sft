#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/qwen-drive/bin/python}"
DATASET_MODE="${DATASET_MODE:-mixed}"
PHYSICALAI_ROOT="${PHYSICALAI_ROOT:-/pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip}"
PHYSICALAI_STORAGE="${PHYSICALAI_STORAGE:-files}"
PHYSICALAI_VIEW_LIST="${PHYSICALAI_VIEW_LIST:-[cam_f0,cam_l0,cam_r0,cam_b0]}"
PHYSICALAI_ANCHOR_TIME_S="${PHYSICALAI_ANCHOR_TIME_S:-10.0}"
PHYSICALAI_MAX_SAMPLES_PER_SPLIT="${PHYSICALAI_MAX_SAMPLES_PER_SPLIT:-null}"
PHYSICALAI_SCAN_WORKERS="${PHYSICALAI_SCAN_WORKERS:-48}"
PHYSICALAI_MIXTURE_RATIO="${PHYSICALAI_MIXTURE_RATIO:-proportional}"

case "${DATASET_MODE}" in
  mix|mixed)
    DATASET_MODE="mixed"
    ;;
esac

case "${DATASET_MODE}" in
  mogo|mixed)
    MOGO_MANIFEST="${MOGO_MANIFEST-${REPO_ROOT}/navsim/planning/script/config/common/mogo_dataset/nav_station.yaml}"
    MOGO_NAMES="${MOGO_NAMES:-}"
    if [[ -n "${MOGO_MANIFEST}" ]]; then
      LOG_SPLIT="${LOG_SPLIT:-$(basename "${MOGO_MANIFEST}" .yaml)}"
      DATA_ARGS=("mogo_manifest=${MOGO_MANIFEST}" "mogo_dataset_names=[${MOGO_NAMES}]")
    else
      DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset}"
      LOG_SPLIT="${LOG_SPLIT:-mogo}"
      DATA_ARGS=(
        "mogo_manifest=null"
        "navsim_log_path=${DATA_ROOT}/navsim_logs/train"
        "sensor_blobs_path=${DATA_ROOT}/sensor_blobs/train"
      )
    fi
    TRAINING_DATA_LABEL="${LOG_SPLIT}"
    if [[ "${DATASET_MODE}" == "mixed" ]]; then
      TRAINING_DATA_LABEL="${LOG_SPLIT}_mixed"
    fi
    ;;
  physicalai)
    # This config is still required by the Qwen IL Hydra defaults, but the
    # PhysicalAI builder uses its official clip_index split directly.
    LOG_SPLIT="${LOG_SPLIT:-nav_station}"
    DATA_ARGS=("mogo_manifest=null")
    TRAINING_DATA_LABEL="physicalai"
    ;;
  *)
    echo "ERROR: DATASET_MODE must be mogo, physicalai, mix, or mixed; got '${DATASET_MODE}'" >&2
    exit 2
    ;;
esac
# 训练说明与论文已公开/未公开的参数见 docs/qwen_drive_stage3_training.md。
# output_dir 是最终目录；未显式设置时，按 experiment_name / 时间戳隔离实验。
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_drive_il_4cam}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/qwen_drive_runs/${TRAINING_DATA_LABEL}}"
RUN_ID="${RUN_ID:-$(date +%Y.%m.%d.%H.%M.%S)}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${RUNS_ROOT}/"'${experiment_name}/${experiment_uid}'
fi

# 以下优化参数是本仓库的实验起点，论文未公开 Stage 3 的 batch size / LR。
# BATCH_SIZE 是每卡 micro-batch；有效全局 batch = BATCH_SIZE * GPUS * GRAD_ACCUM。
GPUS="${GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-80}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MAX_EPOCHS="${MAX_EPOCHS:-5}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
PRECISION="${PRECISION:-bf16-mixed}"
SEED="${SEED:-0}"

# 默认使用可复用的原始帧索引；只在 rank 0 转换未缓存的日志，不缓存模型特征。
SCENE_LOADING="${SCENE_LOADING:-indexed}"
# null 自动使用各数据集的 exp/qwen35_drive_cache；也可在 manifest 配置 cache_path。
SCENE_CACHE_DIR="${SCENE_CACHE_DIR:-null}"
SCENE_CACHE_WORKERS="${SCENE_CACHE_WORKERS:-2}"
SCENE_CACHE_ONLY="${SCENE_CACHE_ONLY:-false}"
# LOG_LOAD_WORKERS 仅用于旧 eager 模式；NUM_WORKERS 控制训练期按需读取。
# CACHE_INDEX_WORKERS 仅用于 feature cache-only 分支。
LOG_LOAD_WORKERS="${LOG_LOAD_WORKERS:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_INDEX_WORKERS="${CACHE_INDEX_WORKERS:-96}"

IMAGE_VIEW_LIST="${IMAGE_VIEW_LIST:-[cam_f0,cam_l1,cam_r1,cam_b0]}"

# Optional token-level rare-scene oversampling. The online Qwen dataset uses a
# token-aware repeated-index view, so this works for both one source and manifests.
if [[ "${DATASET_MODE}" == "mogo" ]]; then
  OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-20}"
else
  OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-1}"
fi
OVERSAMPLE_TOKENS="${OVERSAMPLE_TOKENS:-${REPO_ROOT}/navsim/local_config/data_selection/nudge_oversample_tokens_v1.txt}"
OVERSAMPLE_ARGS=()
if [[ ! "${OVERSAMPLE_FACTOR}" =~ ^[0-9]+$ ]] || (( OVERSAMPLE_FACTOR < 1 )); then
  echo "ERROR: OVERSAMPLE_FACTOR must be a positive integer; got '${OVERSAMPLE_FACTOR}'" >&2
  exit 2
fi
if (( OVERSAMPLE_FACTOR > 1 )); then
  if [[ "${DATASET_MODE}" != "mogo" ]]; then
    echo "ERROR: token oversampling currently requires DATASET_MODE=mogo; set OVERSAMPLE_FACTOR=1 for ${DATASET_MODE} training" >&2
    exit 2
  fi
  if [[ ! -f "${OVERSAMPLE_TOKENS}" ]]; then
    echo "ERROR: OVERSAMPLE_FACTOR=${OVERSAMPLE_FACTOR} but token list not found: ${OVERSAMPLE_TOKENS}" >&2
    echo "       generate it with scripts/cache_dataset/gen_nudge_oversample_tokens.py" >&2
    exit 2
  fi
  OVERSAMPLE_ARGS=(
    "train_oversample_tokens_path=${OVERSAMPLE_TOKENS}"
    "train_oversample_factor=${OVERSAMPLE_FACTOR}"
  )
fi

# 默认继续微调发布的 SFT expert；PLANNER_PATH=null 则随机初始化 PE。
VLM_PATH="${VLM_PATH:-/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B}"
PLANNER_PATH="${PLANNER_PATH:-${VLM_PATH}/planner-sft}"
STRATEGY=auto
if (( GPUS > 1 )); then
  STRATEGY=ddp
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node="${GPUS}" \
  --module navsim.planning.script.run_training_qwen_drive_il \
  "log_split@_global_=${LOG_SPLIT}" \
  "${DATA_ARGS[@]}" \
  "dataset_mode=${DATASET_MODE}" \
  "physicalai_root=${PHYSICALAI_ROOT}" \
  "physicalai_storage=${PHYSICALAI_STORAGE}" \
  "physicalai_view_list=${PHYSICALAI_VIEW_LIST}" \
  "physicalai_anchor_time_s=${PHYSICALAI_ANCHOR_TIME_S}" \
  "physicalai_max_samples_per_split=${PHYSICALAI_MAX_SAMPLES_PER_SPLIT}" \
  "physicalai_scan_workers=${PHYSICALAI_SCAN_WORKERS}" \
  "physicalai_mixture_ratio=${PHYSICALAI_MIXTURE_RATIO}" \
  "experiment_name=${EXPERIMENT_NAME}" \
  "experiment_uid=${RUN_ID}" \
  "output_dir=${OUTPUT_DIR}" \
  "agent.vlm_path=${VLM_PATH}" \
  "agent.planner_path=${PLANNER_PATH}" \
  "agent.lr=${LR}" \
  "agent.weight_decay=${WEIGHT_DECAY}" \
  "agent.view_list=${IMAGE_VIEW_LIST}" \
  "seed=${SEED}" \
  "trainer.params.max_epochs=${MAX_EPOCHS}" \
  "trainer.params.devices=${GPUS}" \
  "trainer.params.strategy=${STRATEGY}" \
  "trainer.params.accumulate_grad_batches=${GRAD_ACCUM}" \
  "trainer.params.gradient_clip_val=${GRAD_CLIP}" \
  "trainer.params.precision=${PRECISION}" \
  "dataloader.params.batch_size=${BATCH_SIZE}" \
  "dataloader.params.num_workers=${NUM_WORKERS}" \
  "scene_loading=${SCENE_LOADING}" \
  "scene_cache_dir=${SCENE_CACHE_DIR}" \
  "scene_cache_workers=${SCENE_CACHE_WORKERS}" \
  "prepare_scene_cache_only=${SCENE_CACHE_ONLY}" \
  "log_load_workers=${LOG_LOAD_WORKERS}" \
  "cache_index_workers=${CACHE_INDEX_WORKERS}" \
  "${OVERSAMPLE_ARGS[@]}" \
  "$@"



python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all
