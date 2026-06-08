TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAPS_ROOT="/D2/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/D2/NXT/lst/vla/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/D2/NXT/lst/vla/recog_drive"
export OPENSCENE_DATA_ROOT="/D2/NXT/lst/vla/navsim/download"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache2_v2

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
cache.cache_path=$CACHE_PATH