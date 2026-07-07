export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/maps"
export NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp"
export NAVSIM_DEVKIT_ROOT="/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive"
export OPENSCENE_DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset"
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}

# ===== 单进程 / 单卡版本（默认启用）=====
# 用单进程就保留这一段，注释掉下面的多卡版本。
# python -m navsim.planning.script.plt_vis_from_csv_all \
#   --csv "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/exp/recogdrive_qwen3_RL_eval/0612QwenRL_new/2026.06.16.10.14.05.csv" \
#   --out "vis_output/eval/0612_qwen_RL_0616pdms_debug" \
#   --checkpoint "exp/training_recogdrive_qwen3_agent_RL/2026.06.12.03.28.29/lightning_logs/version_0/checkpoints/epoch=19-step=5000.ckpt" \
#   --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
#   --vlm_type qwen3 \
#   --openscene_data_root "$OPENSCENE_DATA_ROOT" \
#   --split val \
#   --scene_filter mogo_10hz \
#   --debug_failures \
#   --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache_mogo_v2" \
#   --mogo_map_sqlite_path "$OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite" \
  # --token 00016f8b45c25a1d

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
GPUS=4
torchrun --nproc_per_node=${GPUS} --master_port=9696 \
  -m navsim.planning.script.plt_vis_from_csv_all \
  --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/worldsim_ego_SDE_obj_hdmap_IL_eval/2026.07.06.12.08.47/2026.07.06.12.10.12.csv" \
  --out "vis_output/eval/2026.07.06.12.08.47_ego_SDE_obj_hdmap_IL" \
  --checkpoint "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/exp/training_recogdrive_ego_SDE_obj_hdmap_IL/2026.07.05.02.56.27/lightning_logs/version_0/checkpoints/epoch=196-step=5319.ckpt" \
  --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
  --vlm_type qwen3 \
  --openscene_data_root "$OPENSCENE_DATA_ROOT" \
  --split trainval \
  --scene_filter mogo_10hz \
  --debug_failures \
  --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache_mogo_v3" \
  --mogo_map_sqlite_path "$OPENSCENE_DATA_ROOT/maps/hm_shanghai_1.9.1_20260522.sqlite" \
  --mogo_map_utm_zone 51 \
  --vl_hidden_dim 2048 \
  --ego_status_encoder_type attention \
  --use_object_vector \
  --object_encoder_type temporal \
  --use_hdmap_vector \
  --use_lane_boundaries \

  # --csv "/NXT/lst/vla/navsim/exp/recogdrive_agent_eval/2026.06.03.05.04.20/2026.06.03.07.05.43.csv" \
  # --out "vis_output/eval/debug_nc_dac" \
  # --debug_failures \
  # --metric_cache_path "/NXT/lst/vla/navsim/exp/metric_cache2_v2" \
  # --checkpoint "checkpoints/RL/train_rl_hs0_hsL_5.ckpt" \
  # --vlm_path "/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
