export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export SHANGHAI_NUPLAN_MAPS_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/maps"
export SHANGHAI_NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp"
export SHANGHAI_OPENSCENE_DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset"
export NAVSIM_DEVKIT_ROOT="/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive"
export BEIJING_NUPLAN_MAPS_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset/maps"
export BEIJING_NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp"
export BEIJING_OPENSCENE_DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset"
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}

# ===== 单进程 / 单卡版本（默认启用）=====
# 用单进程就保留这一段，注释掉下面的多卡版本。
# /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/2603_ego_SDE_obj_hdmap_novlm_IL_worldsim_eval/2026.07.16.03.41.18/2026.07.16.03.44.30.csv
python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp/v3_alldata_traffic_light_Pretrain_lowBS_novlm_IL_badcase_eval/2026.08.05.06.47.42/2026.08.05.06.49.32.csv" \
  --out "vis_output/eval/v3_alldata_traffic_light_Pretrain_lowBS_novlm_IL_badcase" \
  --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/mogo_multi/exp/v3_alldata_traffic_light_Pretrain_lowBS_novlm_IL/2026.08.04.08.37.35/lightning_logs/version_0/checkpoints/epoch=3-step=160.ckpt" \
  --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
  --no-use_vlm \
  --vlm_type qwen3 \
  --openscene_data_root "$BEIJING_OPENSCENE_DATA_ROOT" \
  --split val \
  --scene_filter mogo_10hz \
  --debug_failures \
  --metric_cache_path "$BEIJING_NAVSIM_EXP_ROOT/metric_cache_mogo_v1" \
  --mogo_map_sqlite_path "$BEIJING_OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite" \
  --mogo_map_utm_zone 50 \
  --vl_hidden_dim 2048 \
  --ego_status_encoder_type attention \
  --use_object_vector \
  --object_encoder_type temporal \
  --use_hdmap_vector \
  --use_lane_boundaries \
  # --token 00016f8b45c25a1d

python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/v3_alldata_traffic_light_Pretrain_lowBS_novlm_IL_worldsim_eval/2026.08.06.06.11.03/2026.08.06.06.11.27.csv" \
  --out "vis_output/eval/v3_alldata_traffic_light_Pretrain_lowBS_novlm_IL_shworldsim" \
  --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/mogo_multi/exp/v3_alldata_traffic_light_Pretrain_lowBS_novlm_IL/2026.08.04.08.37.35/lightning_logs/version_0/checkpoints/epoch=3-step=160.ckpt" \
  --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
  --no-use_vlm \
  --vlm_type qwen3 \
  --openscene_data_root "$SHANGHAI_OPENSCENE_DATA_ROOT" \
  --split trainval \
  --scene_filter mogo_10hz \
  --debug_failures \
  --metric_cache_path "$SHANGHAI_NAVSIM_EXP_ROOT/metric_cache_mogo_v1" \
  --mogo_map_sqlite_path "$SHANGHAI_OPENSCENE_DATA_ROOT/maps/hm_shanghai_1.9.1_20260522.sqlite" \
  --mogo_map_utm_zone 51 \
  --vl_hidden_dim 2048 \
  --ego_status_encoder_type attention \
  --use_object_vector \
  --object_encoder_type temporal \
  --use_hdmap_vector \
  --use_lane_boundaries \

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
#   --csv "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/all_exp/worldsim/shanghai/exp/v1_alldata_SDE_obj_hdmap_novlm_IL_worldsim_eval/2026.07.20.06.44.56/2026.07.20.06.46.08.csv" \
#   --out "vis_output/eval/v1_alldata_SDE_obj_hdmap_novlm_IL_shworldsim" \
#   --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/mogo_multi/exp/v1_alldata_SDE_obj_hdmap_novlm_IL/2026.07.17.08.02.41/lightning_logs/version_0/checkpoints/epoch=84-step=2635.ckpt" \
#   --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
#   --vlm_type qwen3 \
#   --openscene_data_root "$OPENSCENE_DATA_ROOT" \
#   --split trainval \
#   --scene_filter mogo_10hz \
#   --debug_failures \
#   --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache_mogo_v1" \
#   --mogo_map_sqlite_path "$OPENSCENE_DATA_ROOT/maps/hm_shanghai_1.9.1_20260522.sqlite" \
#   --mogo_map_utm_zone 51 \
#   --vl_hidden_dim 2048 \
#   --ego_status_encoder_type attention \
#   --use_object_vector \
#   --object_encoder_type temporal \
#   --use_hdmap_vector \
#   --use_lane_boundaries \

  # --csv "/NXT/lst/vla/navsim/exp/recogdrive_agent_eval/2026.06.03.05.04.20/2026.06.03.07.05.43.csv" \
  # --out "vis_output/eval/debug_nc_dac" \
  # --debug_failures \
  # --metric_cache_path "/NXT/lst/vla/navsim/exp/metric_cache2_v2" \
  # --checkpoint "checkpoints/RL/train_rl_hs0_hsL_5.ckpt" \
  # --vlm_path "/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
