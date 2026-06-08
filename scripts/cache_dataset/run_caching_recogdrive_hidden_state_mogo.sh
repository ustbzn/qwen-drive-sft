MOGO_ROOT=${MOGO_ROOT:-"/NXT/lst/vla/mogo_data/2026-03-13-15-22-20"}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-mogo_trainval}

export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"$MOGO_ROOT/dataset/maps"}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"$MOGO_ROOT/exp"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"/NXT/lst/vla/recog_drive"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"$MOGO_ROOT/dataset"}
CACHE_PATH=${CACHE_PATH:-"$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_mogo_hs0_hsL"}
VLM_PATH=${VLM_PATH:-"/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B"}

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "MOGO_ROOT: ${MOGO_ROOT}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "NAVSIM_EXP_ROOT: ${NAVSIM_EXP_ROOT}"
echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "GPUS: ${GPUS}"

torchrun \
    --nproc_per_node=${GPUS} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache_mogo_hs0_hsL \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="$VLM_PATH" \
    cache_path=$CACHE_PATH > caching_dataset_mogo_hs0_hsL.txt 2>&1
