export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/D2/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/D2/NXT/lst/vla/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/D2/NXT/lst/vla/recog_drive"
export OPENSCENE_DATA_ROOT="/D2/NXT/lst/vla/navsim/download"

python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv "/D2/NXT/lst/vla/navsim/exp/recogdrive_agent_eval_il/2026.04.28.14.35.28/2026.04.28.16.31.22.csv" \
  --out "vis_output/eval/0428_163122" \
  --checkpoint "checkpoints/IL/train_il_hs0_hsL_2.ckpt" \
  --vlm_path "/D2/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
  --openscene_data_root "$OPENSCENE_DATA_ROOT" \
  --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache2" \
  --split test \
  --scene_filter navtest \
  # --token 00016f8b45c25a1d

  # --csv "/NXT/lst/vla/navsim/exp/recogdrive_agent_eval/2026.06.03.05.04.20/2026.06.03.07.05.43.csv" \
  # --out "vis_output/eval/debug_nc_dac" \
  # --debug_failures \
  # --metric_cache_path "/NXT/lst/vla/navsim/exp/metric_cache2_v2" \
  # --checkpoint "checkpoints/RL/train_rl_hs0_hsL_5.ckpt" \
  # --vlm_path "/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
