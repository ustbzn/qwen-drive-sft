#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/miniconda3/envs/qwen-drive/bin/python}"
DATA_ROOT="${DATA_ROOT:-/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0731-0802}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${DATA_ROOT}/dataset}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${DATA_ROOT}/exp}"
VLM_PATH="${VLM_PATH:-/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/qwen_drive_runs/nav_station_mixed/qwen_drive_il_6cam/2026.10.19.09.29.48/lightning_logs/version_0/checkpoints/epoch=1-step=1784.ckpt}"
MOGO_MAP_SQLITE_PATH="${MOGO_MAP_SQLITE_PATH:-${OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite}"
MOGO_MAP_UTM_ZONE="${MOGO_MAP_UTM_ZONE:-50}"
DATA_SPLIT="${DATA_SPLIT:-val}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen_drive_il_6cam_pdm_eval}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT NAVSIM_EXP_ROOT
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ "${AGENT_TYPE:-qwen_drive}" == "qwen_drive" ]]; then
  CSV="${CSV:-}"
  if [[ -z "${CSV}" ]]; then
    eval_dir="${NAVSIM_EXP_ROOT}/${EXPERIMENT_NAME}"
    if [[ -d "${eval_dir}" ]]; then
      CSV="$(find "${eval_dir}" -type f -name '*.csv' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    fi
  fi
  if [[ -z "${CSV}" || ! -f "${CSV}" ]]; then
    echo "PDM CSV not found. Set CSV=/path/to/evaluation.csv or run the Qwen-Drive PDM evaluation first." >&2
    exit 2
  fi
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Qwen-Drive checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
  fi
  read -r -a QWEN_VIEWS <<< "${QWEN_VIEW_LIST:-cam_f0 cam_l0 cam_r0 cam_l1 cam_r1 cam_b0}"
  exec "${PYTHON}" -m navsim.planning.script.plt_vis_from_csv_all \
    --csv "${CSV}" \
    --out "${OUT:-vis_output/eval/qwen_drive_il_6cam}" \
    --agent_type qwen_drive \
    --checkpoint "${CHECKPOINT}" \
    --vlm_path "${VLM_PATH}" \
    --planner_path "${PLANNER_PATH:-${VLM_PATH}/planner-sft}" \
    --qwen_view_list "${QWEN_VIEWS[@]}" \
    --openscene_data_root "${OPENSCENE_DATA_ROOT}" \
    --split "${DATA_SPLIT}" \
    --scene_filter mogo_qwen_drive \
    --metric_cache_path "${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1}" \
    --mogo_map_sqlite_path "${MOGO_MAP_SQLITE_PATH}" \
    --mogo_map_utm_zone "${MOGO_MAP_UTM_ZONE}"
fi

# Legacy ReCogDrive visualization settings.
export SHANGHAI_NUPLAN_MAPS_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/maps"
export SHANGHAI_NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp"
export SHANGHAI_OPENSCENE_DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset"
export BEIJING_NUPLAN_MAPS_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/badcase_TSP/dataset/maps"
export BEIJING_NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/badcase_TSP/exp"
export BEIJING_OPENSCENE_DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/badcase_TSP/dataset"

# 站点与导航指令开关（默认开启；必须与 checkpoint 的训练配置保持一致）。
# 用法示例：USE_NAVIGATION_COMMAND=false USE_STATION_TOKEN=false bash scripts/visualization/plt_csv_all.sh
# MAX_STATION_TOKENS 当前仅支持 1。
export USE_NAVIGATION_COMMAND="${USE_NAVIGATION_COMMAND:-true}"
export USE_STATION_TOKEN="${USE_STATION_TOKEN:-true}"
export MAX_STATION_TOKENS="${MAX_STATION_TOKENS:-1}"
NAV_ARGS=()
if [[ "${USE_NAVIGATION_COMMAND}" == "true" ]]; then
  NAV_ARGS+=(--use_navigation_command)
fi
if [[ "${USE_STATION_TOKEN}" == "true" ]]; then
  NAV_ARGS+=(--use_station_token --max_station_tokens "${MAX_STATION_TOKENS}")
fi

# ===== 单进程 / 单卡版本（默认启用）=====
# 用单进程就保留这一段，注释掉下面的多卡版本。
# /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/2603_ego_SDE_obj_hdmap_novlm_IL_worldsim_eval/2026.07.16.03.41.18/2026.07.16.03.44.30.csv
python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/badcase_TSP/exp/v3_nav_station_4cam_novlm_RL_TSP_eval/2026.08.26.08.16.58/2026.08.26.08.17.35.csv" \
  --out "vis_output/eval/v3_nav_station_4cam_novlm_RL_TSP" \
  --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/nav_station/exp/v3_nav_station_4cam_novlm_nav_station_RL/2026.08.25.10.09.13/lightning_logs/version_0/checkpoints/epoch=15-step=1264.ckpt" \
  --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
  --no-use_vlm \
  --vlm_type qwen3 \
  --openscene_data_root "$BEIJING_OPENSCENE_DATA_ROOT" \
  --split trainval \
  --scene_filter mogo_10hz \
  --debug_failures \
  --metric_cache_path "$BEIJING_NAVSIM_EXP_ROOT/metric_cache_mogo_v1" \
  --mogo_map_sqlite_path "$BEIJING_OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite" \
  --mogo_map_utm_zone 50 \
  --vl_hidden_dim 2048 \
  --ego_status_encoder_type attention \
  "${NAV_ARGS[@]}" \
  --use_object_vector \
  --object_encoder_type temporal \
  --use_hdmap_vector \
  --use_lane_boundaries \
  --include_static_objects \
  --use_image_encoder \
  --image_encoder_arch resnet101 \
  --no-image_encoder_pretrained \
  --image_encoder_pretrained_path "" \
  --image_height 256 \
  --image_width 704 \
  --image_grid_h 8 \
  --image_grid_w 16 \
  --image_token_dropout 0.1 \
  --image_view_list cam_f0 cam_b0 cam_l0 cam_r0
  # CSV 若含 trajectory_path，将直接绘制已保存轨迹；缺失时才重新调用 Image Encoder 推理。
  # --token 00016f8b45c25a1d

# python -m navsim.planning.script.plt_vis_from_csv_all \
#   --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/v3_nav_station_4cam_novlm_RL_worldsim_eval/2026.08.26.08.14.46/2026.08.26.08.15.12.csv" \
#   --out "vis_output/eval/v3_nav_station_4cam_novlm_RL_worldsim" \
#   --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/nav_station/exp/v3_nav_station_4cam_novlm_nav_station_RL/2026.08.25.10.09.13/lightning_logs/version_0/checkpoints/epoch=15-step=1264.ckpt" \
#   --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
#   --no-use_vlm \
#   --vlm_type qwen3 \
#   --openscene_data_root "$SHANGHAI_OPENSCENE_DATA_ROOT" \
#   --split trainval \
#   --scene_filter mogo_10hz \
#   --debug_failures \
#   --metric_cache_path "$SHANGHAI_NAVSIM_EXP_ROOT/metric_cache_mogo_v1" \
#   --mogo_map_sqlite_path "$SHANGHAI_OPENSCENE_DATA_ROOT/maps/hm_shanghai_1.9.1_20260522.sqlite" \
#   --mogo_map_utm_zone 51 \
#   --vl_hidden_dim 2048 \
#   --ego_status_encoder_type attention \
#   "${NAV_ARGS[@]}" \
#   --use_object_vector \
#   --object_encoder_type temporal \
#   --use_hdmap_vector \
#   --use_lane_boundaries \
#   --include_static_objects \
#   --use_image_encoder \
#   --image_encoder_arch resnet101 \
#   --no-image_encoder_pretrained \
#   --image_encoder_pretrained_path "" \
#   --image_height 256 \
#   --image_width 704 \
#   --image_grid_h 8 \
#   --image_grid_w 16 \
#   --image_token_dropout 0.1 \
#   --image_view_list cam_f0 cam_b0 cam_l0 cam_r0

# python -m navsim.planning.script.plt_vis_from_csv_all \
#   --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/v3_alldata_traffic_light_Pretrain_lowBS_novlm_IL_worldsim_eval/2026.08.06.06.11.03/2026.08.06.06.11.27.csv" \
#   --out "vis_output/eval/v1_alldata_traffic_light_RL_ZHworldsim" \
#   --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/mogo_multi/exp/v1_alldata_SDE_obj_hdmap_novlm_RL/2026.07.22.03.44.31/lightning_logs/version_0/checkpoints/epoch=17-step=738.ckpt" \
#   --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
#   --vlm_type qwen3 \
#   --openscene_data_root "$OPENSCENE_DATA_ROOT" \
#   --split trainval \
#   --scene_filter mogo_10hz \
#   --debug_failures \
#   --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache_mogo_v1" \
#   --mogo_map_sqlite_path "$OPENSCENE_DATA_ROOT/maps/hm_zhuhai_1.4.2_20260604.sqlite" \
#   --mogo_map_utm_zone 49 \
#   --vl_hidden_dim 2048 \
#   --ego_status_encoder_type attention \
#   --use_object_vector \
#   --object_encoder_type temporal \
#   --use_hdmap_vector \
#   --use_lane_boundaries \

# ===== 多卡版本（用多卡就取消注释这一段，并注释掉上面的单进程版本）=====
# 各 rank 用 tokens[rank::world_size] 分摊渲染，跑完 rank0 自动把
# index_rank*.csv 合并成 index_all.csv。GPUS 改成要用的卡数。
#
# 模型架构 flag（须与 checkpoint 训练时一致，否则被 initialize() 的 strict=False 静默丢成
# 随机初始化）。下面这组对齐当前 --checkpoint 的 ego_SDE+obj(temporal)+hdmap(lane) 模型：
#   --ego_status_encoder_type attention   # 自车状态 SDE；mlp 则去掉
#   --use_object_vector --object_encoder_type temporal   # obj token；mlp 则改 mlp；不用则去掉
#   --use_hdmap_vector --use_lane_boundaries             # hdmap token + 车道边界几何
#   --use_image_encoder                                  # 前视图像 ResNet token（本 ckpt 未用）
#   --vlm_ablation zero                                  # 加载 VLM 但置零其 token（本 ckpt 用 none）
#   --no-use_vlm                                         # 完全不加载/不用 VLM（更小更省；本 ckpt 用 VLM）
# GPUS=8
# torchrun --nproc_per_node=${GPUS} --master_port=9696 \
#   -m navsim.planning.script.plt_vis_from_csv_all \
#   --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/badcase_TSP/exp/v3_imgencoder_4cam_novlm_RL_TSPnew_eval/2026.08.21.09.30.41/2026.08.21.09.32.05.csv" \
#   --out "vis_output/eval/v3_imgencoder_4cam_novlm_RL_TSPnew" \
#   --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/nav_station/exp/v3_nav_station_4cam_novlm_nav_station_IL/2026.08.25.07.23.13/lightning_logs/version_0/checkpoints/epoch=16-step=5032.ckpt" \
#   --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
#   --no-use_vlm \
#   --vlm_type qwen3 \
#   --openscene_data_root "$BEIJING_OPENSCENE_DATA_ROOT" \
#   --split trainval \
#   --scene_filter mogo_10hz \
#   --debug_failures \
#   --metric_cache_path "$BEIJING_NAVSIM_EXP_ROOT/metric_cache_mogo_v1" \
#   --mogo_map_sqlite_path "$BEIJING_OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite" \
#   --mogo_map_utm_zone 50 \
#   --vl_hidden_dim 2048 \
#   --ego_status_encoder_type attention \
#   "${NAV_ARGS[@]}" \
#   --use_object_vector \
#   --object_encoder_type temporal \
#   --use_hdmap_vector \
#   --use_lane_boundaries \
#   --include_static_objects \
#   --use_image_encoder \
#   --image_encoder_arch resnet101 \
#   --no-image_encoder_pretrained \
#   --image_encoder_pretrained_path "" \
#   --image_height 256 \
#   --image_width 704 \
#   --image_grid_h 8 \
#   --image_grid_w 16 \
#   --image_token_dropout 0.1 \
#   --image_view_list cam_f0 cam_b0 cam_l0 cam_r0

  # --csv "/NXT/lst/vla/navsim/exp/recogdrive_agent_eval/2026.06.03.05.04.20/2026.06.03.07.05.43.csv" \
  # --out "vis_output/eval/debug_nc_dac" \
  # --debug_failures \
  # --metric_cache_path "/NXT/lst/vla/navsim/exp/metric_cache2_v2" \
  # --checkpoint "checkpoints/RL/train_rl_hs0_hsL_5.ckpt" \
  # --vlm_path "/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
