TRAIN_TEST_SPLIT=trainval
export NUPLAN_MAPS_ROOT="/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/maps/"
export NAVSIM_EXP_ROOT="/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp"
export NAVSIM_DEVKIT_ROOT="/NXT/lst/vla/recog_drive"
export OPENSCENE_DATA_ROOT="/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset"
CACHE_PATH=${CACHE_PATH:-$NAVSIM_EXP_ROOT/metric_cache}

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_mogo_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH \
