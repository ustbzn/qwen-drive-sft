MOGO_ROOT=${MOGO_ROOT:-"/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319"}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-mogo_trainval}

export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-"nuplan-maps-v1.0"}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"$MOGO_ROOT/dataset/maps"}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"$MOGO_ROOT/exp"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"$MOGO_ROOT/dataset"}
VLM_PATH=${VLM_PATH:-"/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct"}
CACHE_PATH=${CACHE_PATH:-"$NAVSIM_EXP_ROOT/recogdrive_qwen3_notrain_cache"}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-"$NAVSIM_EXP_ROOT/metric_cache_mogo_v2"}
CHECKPOINT=${CHECKPOINT:-"$NAVSIM_EXP_ROOT/training_recogdrive_qwen3_agent_IL/2026.06.11.13.28.50/lightning_logs/version_0/checkpoints/epoch=188-step=13986.ckpt"}

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}

MASTER_PORT=${MASTER_PORT:-9675}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}

# Defaults for single-node runs. Cluster launchers can override these.
MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST:-127.0.0.1}
MLP_WORKER_0_PORT=${MLP_WORKER_0_PORT:-$MASTER_PORT}

echo "MOGO_ROOT: ${MOGO_ROOT}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "NAVSIM_EXP_ROOT: ${NAVSIM_EXP_ROOT}"
echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "METRIC_CACHE_PATH: ${METRIC_CACHE_PATH}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "GPUS: ${GPUS}"

torchrun \
    --nnodes=1 \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --nproc_per_node=${GPUS} \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_rl.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.vlm_path="$VLM_PATH" \
    agent.cam_type='single' \
    agent.grpo=True \
    agent.cache_hidden_state=True \
    agent.vlm_type="qwen3" \
    agent.checkpoint_path=\"$CHECKPOINT\" \
    agent.vl_hidden_dim=2048 \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.metric_cache_path="$METRIC_CACHE_PATH" \
    agent.reference_policy_checkpoint=\"$CHECKPOINT\" \
    trainer.params.max_epochs=20 \
    trainer.params.num_nodes=1 \
    trainer.params.devices=${GPUS} \
    dataloader.params.batch_size=16 \
    experiment_name=training_recogdrive_qwen3_agent_RL \
    train_test_split=$TRAIN_TEST_SPLIT \
    '+log_split@_global_=mogo' \
    train_token_whitelist_path=\
    cache_path="$CACHE_PATH" \
    use_cache_without_dataset=True \
    force_cache_computation=False > train_recogdrive_qwen3_agent_rl_2b.txt 2>&1


python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all