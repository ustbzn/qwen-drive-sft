export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/NXT/lst/vla/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/NXT/lst/vla/recog_drive"
export OPENSCENE_DATA_ROOT="/NXT/lst/vla/navsim/download"
TRAIN_TEST_SPLIT=navtrain
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=0

CHECKPOINT="/NXT/lst/vla/recog_drive/checkpoints/IL/train_il_hs0_hsL_4.ckpt"

# Defaults for single-node runs (torchrun requires valid values).
# If your launcher provides MLP_* envs, they override these.
MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST:-127.0.0.1}
MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT:-$MASTER_PORT}

torchrun \
    --nnodes=1 \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.vlm_path='/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B' \
    agent.cam_type='single' \
    agent.grpo=True \
    agent.cache_hidden_state=True \
    agent.vlm_type="internvl" \
    agent.checkpoint_path="$CHECKPOINT" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path="/NXT/lst/vla/navsim/exp/metric_cache_train_v2" \
    agent.reference_policy_checkpoint="$CHECKPOINT" \
    trainer.params.max_epochs=20 \
    trainer.params.num_nodes=1 \
    trainer.params.devices=${GPUS} \
    dataloader.params.batch_size=8 \
    experiment_name=training_recogdrive_agent \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/NXT/lst/vla/navsim/exp/recogdrive_agent_cache_dir_train_v2" \
    use_cache_without_dataset=True \
    force_cache_computation=False > train_recogdrive_rl_2b.txt 2>&1
  # 2>&1 | tee -a "training_log.txt" &
