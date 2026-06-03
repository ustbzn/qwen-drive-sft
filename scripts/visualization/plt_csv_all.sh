export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/NXT/lst/vla/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/NXT/lst/vla/recog_drive"
export OPENSCENE_DATA_ROOT="/NXT/lst/vla/navsim/download"

python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv "/NXT/lst/vla/navsim/exp/recogdrive_agent_eval/2026.06.03.05.04.20/2026.06.03.07.05.43.csv" \
  --out "vis_output/eval/0603_070543/score_lt_0.5" \
  --max_score 0.5 \
  --checkpoint "checkpoints/RL/train_rl_hs0_hsL_5.ckpt" \
  --vlm_path "/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B" \
  --openscene_data_root "$OPENSCENE_DATA_ROOT" \
  --split test \
  --scene_filter navtest