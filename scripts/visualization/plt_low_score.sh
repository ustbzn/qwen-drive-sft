export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/NXT/lst/vla/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/NXT/lst/vla/recog_drive"
export OPENSCENE_DATA_ROOT="/NXT/lst/vla/navsim/download"

python -m navsim.planning.script.plt_vis_low_score \
  --csv "/NXT/lst/vla/navsim/exp/recogdrive_agent_eval_il/2026.04.28.14.35.28/2026.04.28.16.31.22.csv" \
  --checkpoint "checkpoints/IL/train_il_hs0_hsL_2.ckpt" \
  --vlm_path "/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
  --out "vis_output" \
  --openscene_data_root "$OPENSCENE_DATA_ROOT" \
  --split test \
  --scene_filter navtest \
  # --token "000bf168dd5159af" \
