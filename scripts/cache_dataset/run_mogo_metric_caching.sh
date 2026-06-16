TRAIN_TEST_SPLIT=trainval #navsim/planning/script/config/metric_caching/default_mogo_metric_caching.yaml
export NUPLAN_MAPS_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/maps/"
export NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp"
export NAVSIM_DEVKIT_ROOT="/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive"
export OPENSCENE_DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_mogo_v2
#CACHE_PATH=${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache}

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_mogo_metric_caching.py" \
train_test_split="$TRAIN_TEST_SPLIT" \
cache.cache_path="$CACHE_PATH" \
mogo_map.sqlite_path="$OPENSCENE_DATA_ROOT/maps/hm_shanghai_1.9.1_20260522.sqlite" \
mogo_map.utm_zone=51
