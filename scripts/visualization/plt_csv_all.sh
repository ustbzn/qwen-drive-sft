export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/dataset/maps"
export NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp"
export NAVSIM_DEVKIT_ROOT="/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive"
export OPENSCENE_DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/dataset"
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}

python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/recogdrive_qwen3_RL_eval/2026.06.15.07.27.18/2026.06.15.07.28.15.csv" \
  --out "vis_output/eval/0612_qwen_RL_bj_worldsim" \
  --checkpoint "exp/training_recogdrive_qwen3_agent_RL/2026.06.12.03.28.29/lightning_logs/version_0/checkpoints/epoch=19-step=5000.ckpt" \
  --vlm_path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct" \
  --vlm_type qwen3 \
  --openscene_data_root "$OPENSCENE_DATA_ROOT" \
  --split trainval \
  --scene_filter mogo_10hz \
  --debug_failures \
  --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache_mogo_v2" \
  --mogo_map_sqlite_path "$OPENSCENE_DATA_ROOT/maps/hm_beijing_3.4.9_20260521.sqlite" \
  # --token 00016f8b45c25a1d

  # --csv "/NXT/lst/vla/navsim/exp/recogdrive_agent_eval/2026.06.03.05.04.20/2026.06.03.07.05.43.csv" \
  # --out "vis_output/eval/debug_nc_dac" \
  # --debug_failures \
  # --metric_cache_path "/NXT/lst/vla/navsim/exp/metric_cache2_v2" \
  # --checkpoint "checkpoints/RL/train_rl_hs0_hsL_5.ckpt" \
  # --vlm_path "/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
