# Qwen3-VL / InternVL 关键实验脚本与参数说明

日期：2026-06-04

本文档总结 2026-06-03 关键实验中，训练、评测、RL / GRPO 和可视化相关脚本的启动方式，以及复现实验时需要修改或覆盖的关键参数。

## 1. 通用环境变量

Mogo 数据统一使用以下根目录：

```bash
export DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=${DATA_ROOT}/dataset/maps
export NAVSIM_EXP_ROOT=${DATA_ROOT}/exp
export NAVSIM_DEVKIT_ROOT=/pcpt/pcpt/project/lijun/.code/qw3/recog_drive_mogo
export OPENSCENE_DATA_ROOT=${DATA_ROOT}/dataset
export PYTHONPATH=${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}
```

常用数据与缓存路径：

```bash
export TRAIN_TEST_SPLIT=trainval
export METRIC_CACHE_PATH=${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1
export MOGO_MAP_SQLITE=${OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite
```

注意：

- 本轮 Mogo 实验使用 `trainval`，对应 `navsim/planning/script/config/common/train_test_split/trainval.yaml`。
- Qwen3 相关脚本默认已经使用 Mogo 路径，并支持通过环境变量和命令行 Hydra 参数覆盖。
- 部分 InternVL 旧脚本仍写着通用 NAVSIM 路径，复现实验时应覆盖为上面的 Mogo 路径，或直接按本文档中的命令启动。

## 2. 关键参数含义

| 参数 | 作用 | Qwen3-VL 推荐值 | InternVL 推荐值 |
| --- | --- | --- | --- |
| `agent.vlm_type` | VLM 类型 | `qwen3` | `internvl` |
| `agent.vlm_path` | VLM 权重路径 | Qwen3 或 Qwen3 Mogo 微调路径 | `/D2/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B` |
| `agent.vl_hidden_dim` | VLM hidden state 维度 | `2048` | `1536` |
| `agent.cache_hidden_state` | 是否使用缓存 hidden state | 训练通常为 `True`；评测通常为 `False` | 同左 |
| `agent.cache_mode` | 是否在缓存阶段加载 VLM 计算 hidden state | 缓存阶段为 `True` | 缓存阶段为 `True` |
| `agent.normalize_vl_hidden_states` | planner 侧是否对 VLM hidden state 做 LayerNorm | 最优配置为 `true` | 默认 `false` |
| `agent.first_hidden_state_index` | 送入 planner 的第一路 hidden state 层号 | 最优配置为 `-8` | 默认 `0` |
| `agent.last_hidden_state_index` | 送入 planner 的第二路 hidden state 层号 | 最优配置为 `-2` | 默认 `-1` |
| `agent.grpo` | 是否开启 RL / GRPO 训练逻辑 | RL 为 `True`，IL 为 `False` | 同左 |
| `cache_path` | hidden-state cache 路径 | 每个 VLM / 层选择单独建目录 | 每个 VLM 单独建目录 |
| `metric_cache_path` | PDM / RL 使用的 metric cache 路径 | `${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1` | 同左 |

## 3. Metric Cache 生成

PDM 评测和 RL 奖励都依赖 metric cache。Mogo 对应脚本为：

```bash
scripts/cache_dataset/run_mogo_metric_caching.sh
```

推荐命令：

```bash
export NAVSIM_EXP_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp
export OPENSCENE_DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset
export NUPLAN_MAPS_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/maps

python navsim/planning/script/run_mogo_metric_caching.py \
  train_test_split=trainval \
  cache.cache_path=${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1
```

检查产物：

```bash
ls ${NAVSIM_EXP_ROOT}/metric_cache_mogo_v1
```

## 4. Qwen3-VL Hidden-State Cache

脚本：

```bash
scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh
```

### 4.1 原始 Qwen3-VL cache

```bash
DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \
QWEN3_VL_PATH=/pcpt/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct \
CACHE_PATH=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/qwen3_full_20260602_125622_cache \
GPUS=1 \
sh scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh
```

### 4.2 Mogo 微调 Qwen3-VL cache

微调模型路径：

```text
/D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924
```

推荐命令：

```bash
DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \
QWEN3_VL_PATH=/D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
CACHE_PATH=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/qwen3vl_mogo_ft_hs-8_hs-2_20260603_cache \
GPUS=1 \
sh scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh \
  agent.first_hidden_state_index=-8 \
  agent.last_hidden_state_index=-2
```

说明：

- cache 阶段只负责保存 `first_hidden_state` 和 `last_hidden_state`。
- 如果要做 hidden layer ablation，每组 `first_hidden_state_index` / `last_hidden_state_index` 应使用独立 `CACHE_PATH`，避免不同层的 hidden state 混用。
- `normalize_vl_hidden_states` 是 planner 内部 LayerNorm，不影响 cache 内容，可以只在训练和评测阶段开启。

## 5. Qwen3-VL IL 训练

脚本：

```bash
scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh
```

### 5.1 原始 Qwen3-VL IL

```bash
DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \
QWEN3_VL_PATH=/pcpt/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct \
CACHE_PATH=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/qwen3_full_20260602_125622_cache \
GPUS=1 \
BATCH_SIZE=8 \
MAX_EPOCHS=20 \
sh scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh \
  experiment_name=training_recogdrive_qwen3_agent
```

本轮对应 checkpoint：

```text
/tmp/qwen3_full_20260602_125622_epoch19_step320.ckpt
```

### 5.2 Mogo 微调 Qwen3-VL 默认配置

```bash
DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \
QWEN3_VL_PATH=/D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
CACHE_PATH=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/qwen3vl_mogo_ft_20260603_0718_cache \
GPUS=1 \
BATCH_SIZE=8 \
MAX_EPOCHS=20 \
sh scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh \
  experiment_name=training_recogdrive_qwen3_mogo_ft_agent
```

本轮对应 checkpoint：

```text
/tmp/qwen3vl_mogo_ft_20260603_0718_epoch17_step288.ckpt
```

### 5.3 Qwen3-VL 最优 IL 配置

本轮最优配置：

```yaml
agent.normalize_vl_hidden_states: true
agent.first_hidden_state_index: -8
agent.last_hidden_state_index: -2
trainer.params.max_epochs: 100
```

启动命令：

```bash
DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \
QWEN3_VL_PATH=/D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
CACHE_PATH=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/qwen3vl_mogo_ft_hs-8_hs-2_20260603_cache \
GPUS=1 \
BATCH_SIZE=8 \
MAX_EPOCHS=100 \
sh scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh \
  agent.normalize_vl_hidden_states=true \
  agent.first_hidden_state_index=-8 \
  agent.last_hidden_state_index=-2 \
  experiment_name=training_recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent
```

本轮最优 checkpoint：

```text
/tmp/qwen3vl_mogo_ft_hs-8_hs-2_norm_100ep_20260603_epoch65_step1056.ckpt
```

## 6. InternVL IL 训练

InternVL 2B 训练可基于脚本：

```bash
scripts/training/run_recogdrive_train_multi_node_2b.sh
```

该脚本中旧默认路径指向 NAVSIM，应在 Mogo 实验中改为或覆盖为：

```bash
export NAVSIM_EXP_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp
export NAVSIM_DEVKIT_ROOT=/pcpt/pcpt/project/lijun/.code/qw3/recog_drive_mogo
export OPENSCENE_DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset
export NUPLAN_MAPS_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/maps
```

推荐直接使用等价 `torchrun` 命令：

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=63669 \
  navsim/planning/script/run_training_recogdrive.py \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.grpo=False \
  agent.vlm_path=/D2/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.vlm_type=internvl \
  agent.vl_hidden_dim=1536 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  trainer.params.max_epochs=20 \
  trainer.params.num_nodes=1 \
  trainer.params.devices=1 \
  dataloader.params.batch_size=8 \
  experiment_name=training_recogdrive_internvl_agent \
  train_test_split=trainval \
  train_logs=null \
  val_logs=null \
  cache_path=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/internvl_full_20260603_034735_cache \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

本轮对应 checkpoint：

```text
/tmp/internvl_full_20260603_034735_epoch18_step304.ckpt
```

## 7. PDM 评测

Qwen3 评测脚本：

```bash
scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh
```

评测阶段建议：

- `agent.cache_hidden_state=False`，评测时用 checkpoint 和 VLM 现场推理 hidden state。
- Qwen3 必须保持训练时相同的 `agent.normalize_vl_hidden_states`、`agent.first_hidden_state_index`、`agent.last_hidden_state_index`。
- `metric_cache_path` 使用 Mogo 的 `metric_cache_mogo_v1`。

### 7.1 Qwen3-VL 最优 IL 评测

```bash
DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \
QWEN3_VL_PATH=/D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
CHECKPOINT=/tmp/qwen3vl_mogo_ft_hs-8_hs-2_norm_100ep_20260603_epoch65_step1056.ckpt \
METRIC_CACHE_PATH=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/metric_cache_mogo_v1 \
GPUS=1 \
sh scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh \
  agent.normalize_vl_hidden_states=true \
  agent.first_hidden_state_index=-8 \
  agent.last_hidden_state_index=-2 \
  experiment_name=recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent_eval
```

本轮评测 CSV：

```text
/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent_eval/2026.06.03.09.33.31/2026.06.03.09.34.22.csv
```

### 7.2 InternVL IL 评测

InternVL 可直接运行 `run_pdm_score_recogdrive.py`，关键是把 `agent.vlm_type`、`agent.vlm_path` 和 `agent.vl_hidden_dim` 改为 InternVL：

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=63669 \
  navsim/planning/script/run_pdm_score_recogdrive.py \
  train_test_split=trainval \
  agent=recogdrive_agent \
  agent.checkpoint_path=/tmp/internvl_full_20260603_034735_epoch18_step304.ckpt \
  agent.vlm_path=/D2/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B \
  metric_cache_path=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/metric_cache_mogo_v1 \
  agent.cam_type=single \
  agent.grpo=False \
  agent.cache_hidden_state=False \
  agent.vlm_type=internvl \
  agent.vl_hidden_dim=1536 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  experiment_name=recogdrive_internvl_agent_eval
```

本轮评测 CSV：

```text
/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_internvl_agent_eval/2026.06.03.03.59.59/2026.06.03.04.01.00.csv
```

## 8. RL / GRPO 训练

入口脚本：

```bash
navsim/planning/script/run_training_recogdrive_rl.py
```

可参考脚本：

```bash
scripts/training/run_recogdrive_train_multi_node_rl_2b.sh
```

RL 关键参数：

```yaml
agent.grpo: true
agent.checkpoint_path: <IL checkpoint>
agent.reference_policy_checkpoint: <同一个 IL checkpoint>
agent.metric_cache_path: /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/metric_cache_mogo_v1
trainer.params.max_epochs: 10
dataloader.params.batch_size: 2
```

### 8.1 Qwen3-VL RL

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=63669 \
  navsim/planning/script/run_training_recogdrive_rl.py \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.vlm_path=/D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
  agent.cam_type=single \
  agent.grpo=True \
  agent.cache_hidden_state=True \
  agent.vlm_type=qwen3 \
  agent.vl_hidden_dim=2048 \
  agent.checkpoint_path=/tmp/qwen3vl_mogo_ft_hs-8_hs-2_norm_100ep_20260603_epoch65_step1056.ckpt \
  agent.reference_policy_checkpoint=/tmp/qwen3vl_mogo_ft_hs-8_hs-2_norm_100ep_20260603_epoch65_step1056.ckpt \
  agent.normalize_vl_hidden_states=true \
  agent.first_hidden_state_index=-8 \
  agent.last_hidden_state_index=-2 \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  agent.metric_cache_path=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/metric_cache_mogo_v1 \
  trainer.params.max_epochs=10 \
  trainer.params.num_nodes=1 \
  trainer.params.devices=1 \
  dataloader.params.batch_size=2 \
  experiment_name=training_recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_grpo_10ep \
  train_test_split=trainval \
  cache_path=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/qwen3vl_mogo_ft_hs-8_hs-2_20260603_cache \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

本轮评测过的 RL checkpoint：

```text
/tmp/qwen3vl_100ep_grpo_10ep_20260603_epoch1_step126.ckpt
/tmp/qwen3vl_100ep_grpo_10ep_20260603_epoch9_step630.ckpt
```

### 8.2 InternVL RL

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=63669 \
  navsim/planning/script/run_training_recogdrive_rl.py \
  agent=recogdrive_agent \
  agent.lr=1e-4 \
  agent.vlm_path=/D2/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B \
  agent.cam_type=single \
  agent.grpo=True \
  agent.cache_hidden_state=True \
  agent.vlm_type=internvl \
  agent.vl_hidden_dim=1536 \
  agent.checkpoint_path=/tmp/internvl_full_20260603_034735_epoch18_step304.ckpt \
  agent.reference_policy_checkpoint=/tmp/internvl_full_20260603_034735_epoch18_step304.ckpt \
  agent.dit_type=small \
  agent.vlm_size=small \
  agent.sampling_method=ddim \
  agent.metric_cache_path=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/metric_cache_mogo_v1 \
  trainer.params.max_epochs=10 \
  trainer.params.num_nodes=1 \
  trainer.params.devices=1 \
  dataloader.params.batch_size=2 \
  experiment_name=training_recogdrive_internvl_grpo_10ep \
  train_test_split=trainval \
  cache_path=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/internvl_full_20260603_034735_cache \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

本轮评测过的 RL checkpoint：

```text
/tmp/internvl_grpo_10ep_20260603_epoch9_step630.ckpt
```

## 9. RL Checkpoint 评测

RL checkpoint 的评测仍使用 `run_pdm_score_recogdrive.py`。区别是：

- `agent.checkpoint_path` 换成 RL checkpoint。
- Qwen3 仍保持 `normalize_vl_hidden_states=true`、`first_hidden_state_index=-8`、`last_hidden_state_index=-2`。
- `agent.grpo=False`，因为评测只需要采样 planner，不走训练时 GRPO loss。

Qwen3 RL epoch 9 示例：

```bash
DATA_ROOT=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \
QWEN3_VL_PATH=/D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
CHECKPOINT=/tmp/qwen3vl_100ep_grpo_10ep_20260603_epoch9_step630.ckpt \
METRIC_CACHE_PATH=/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/metric_cache_mogo_v1 \
GPUS=1 \
sh scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh \
  agent.normalize_vl_hidden_states=true \
  agent.first_hidden_state_index=-8 \
  agent.last_hidden_state_index=-2 \
  experiment_name=recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_grpo_10ep_epoch9_eval
```

## 10. 可视化

主要脚本：

```bash
navsim/planning/script/plt_vis_from_csv_all.py
navsim/planning/script/plt_vis_low_score.py
```

脚本封装：

```bash
scripts/visualization/plt_csv_all.sh
scripts/visualization/plt_low_score.sh
```

推荐直接使用 Python 命令，便于传入 Qwen3 / InternVL 不同参数。

### 10.1 全量 CSV 可视化

Qwen3 100 epoch IL 示例：

```bash
python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent_eval/2026.06.03.09.33.31/2026.06.03.09.34.22.csv \
  --out /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_qwen3_100ep_compare_20260603/qwen3_100ep \
  --checkpoint /tmp/qwen3vl_mogo_ft_hs-8_hs-2_norm_100ep_20260603_epoch65_step1056.ckpt \
  --vlm_path /D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
  --openscene_data_root /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset \
  --split trainval \
  --scene_filter all_scenes \
  --vlm_type qwen3 \
  --vlm_size small \
  --dit_type small \
  --sampling_method ddim \
  --normalize_vl_hidden_states \
  --first_hidden_state_index -8 \
  --last_hidden_state_index -2 \
  --mogo_map_sqlite_path /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/maps/hm_beijing_3.4.9_20260521.sqlite
```

InternVL IL 示例：

```bash
python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_internvl_agent_eval/2026.06.03.03.59.59/2026.06.03.04.01.00.csv \
  --out /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_qwen3_100ep_compare_20260603/internvl \
  --checkpoint /tmp/internvl_full_20260603_034735_epoch18_step304.ckpt \
  --vlm_path /D2/NXT/lst/vla/recogdrive/ReCogDrive-VLM-2B \
  --openscene_data_root /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset \
  --split trainval \
  --scene_filter all_scenes \
  --vlm_type internvl \
  --vlm_size small \
  --dit_type small \
  --sampling_method ddim \
  --mogo_map_sqlite_path /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/maps/hm_beijing_3.4.9_20260521.sqlite
```

### 10.2 低分样例可视化

`plt_vis_low_score.py` 会从 CSV 中选择：

- `score` 最低的前 100 个 token。
- `score` 位于 1% 到 5% 分位之间的 token。

示例：

```bash
python -m navsim.planning.script.plt_vis_low_score \
  --csv /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent_eval/2026.06.03.09.33.31/2026.06.03.09.34.22.csv \
  --out /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_low_score_qwen3_100ep \
  --checkpoint /tmp/qwen3vl_mogo_ft_hs-8_hs-2_norm_100ep_20260603_epoch65_step1056.ckpt \
  --vlm_path /D2/NXT/vlm/miaobinqi/Qwen3-VL/output/qwen3vl_2b_mogo/20260409_1924 \
  --openscene_data_root /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset \
  --split trainval \
  --scene_filter all_scenes \
  --vlm_type qwen3 \
  --vlm_size small \
  --dit_type small \
  --sampling_method ddim \
  --normalize_vl_hidden_states \
  --first_hidden_state_index -8 \
  --last_hidden_state_index -2
```

单 token 快速检查：

```bash
python -m navsim.planning.script.plt_vis_from_csv_all \
  --csv <评测 CSV> \
  --out /tmp/vis_single_token \
  --token <token> \
  --checkpoint <planner checkpoint> \
  --vlm_path <VLM path> \
  --openscene_data_root /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset \
  --split trainval \
  --scene_filter all_scenes \
  --vlm_type qwen3 \
  --normalize_vl_hidden_states \
  --first_hidden_state_index -8 \
  --last_hidden_state_index -2 \
  --mogo_map_sqlite_path /D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/maps/hm_beijing_3.4.9_20260521.sqlite
```

### 10.3 Contact sheet 对比图

本轮已生成的关键对比图：

```text
/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_qwen3_100ep_compare_20260603/contact_sheet_qwen3_20ep_vs_100ep_vs_internvl.png
/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_centerline_qwen3_ft_20260603_0736/contact_sheet_centerline_worst_ft_vs_raw_vs_internvl.png
/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_rl_qwen3_internvl_compare_20260603/contact_sheet_qwen3_il_vs_rl.png
/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_rl_qwen3_internvl_compare_20260603/contact_sheet_internvl_il_vs_rl.png
```

如果需要新增 contact sheet，可以基于各可视化目录下的 `index_rank0.csv` 按相同 token 做横向拼图。推荐保留固定 token 集，避免每次对比样例不同导致结论不稳定。

## 11. 参数修改建议

### 11.1 优先使用命令行覆盖

Qwen3 脚本末尾保留了 `"$@"`，因此推荐在命令后追加 Hydra 参数：

```bash
sh scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh \
  agent.normalize_vl_hidden_states=true \
  agent.first_hidden_state_index=-8 \
  agent.last_hidden_state_index=-2 \
  experiment_name=<新实验名>
```

这样可以避免直接改脚本，降低影响已有功能的风险。

### 11.2 每组实验必须单独命名

建议统一命名规则：

```text
training_recogdrive_<vlm>_<backbone>_<hidden-state配置>_<norm>_<epoch>_agent
recogdrive_<vlm>_<backbone>_<hidden-state配置>_<norm>_<epoch>_agent_eval
```

示例：

```text
training_recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent
recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent_eval
```

### 11.3 Cache 路径必须和 hidden-state 层配置绑定

错误示例：

```text
同一个 cache_path 同时用于 hs0/hs-1 和 hs-8/hs-2
```

推荐示例：

```text
qwen3vl_mogo_ft_hs0_hs-2_20260603_cache
qwen3vl_mogo_ft_hs-8_hs-2_20260603_cache
internvl_full_20260603_034735_cache
```

### 11.4 训练和评测参数必须一致

Qwen3 训练时使用：

```yaml
agent.normalize_vl_hidden_states: true
agent.first_hidden_state_index: -8
agent.last_hidden_state_index: -2
```

评测和可视化也必须带上同样参数，否则 planner checkpoint 的 projector 输入分布会不一致。

## 12. 本轮关键产物索引

| 类型 | 路径 |
| --- | --- |
| Qwen3 100 epoch IL checkpoint | `/tmp/qwen3vl_mogo_ft_hs-8_hs-2_norm_100ep_20260603_epoch65_step1056.ckpt` |
| InternVL IL checkpoint | `/tmp/internvl_full_20260603_034735_epoch18_step304.ckpt` |
| Qwen3 RL epoch 1 checkpoint | `/tmp/qwen3vl_100ep_grpo_10ep_20260603_epoch1_step126.ckpt` |
| Qwen3 RL epoch 9 checkpoint | `/tmp/qwen3vl_100ep_grpo_10ep_20260603_epoch9_step630.ckpt` |
| InternVL RL epoch 9 checkpoint | `/tmp/internvl_grpo_10ep_20260603_epoch9_step630.ckpt` |
| Qwen3 100 epoch IL CSV | `/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_agent_eval/2026.06.03.09.33.31/2026.06.03.09.34.22.csv` |
| InternVL IL CSV | `/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_internvl_agent_eval/2026.06.03.03.59.59/2026.06.03.04.01.00.csv` |
| Qwen3 RL epoch 1 CSV | `/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_grpo_10ep_epoch1_eval/2026.06.03.12.12.24/2026.06.03.12.13.11.csv` |
| Qwen3 RL epoch 9 CSV | `/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_qwen3_mogo_ft_hs-8_hs-2_norm_100ep_grpo_10ep_epoch9_eval/2026.06.03.12.13.38/2026.06.03.12.14.27.csv` |
| InternVL RL epoch 9 CSV | `/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/recogdrive_internvl_grpo_10ep_epoch9_eval/2026.06.03.12.14.56/2026.06.03.12.15.52.csv` |
| Qwen3 / InternVL IL 可视化 | `/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_qwen3_100ep_compare_20260603` |
| RL 可视化 | `/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/exp/vis_rl_qwen3_internvl_compare_20260603` |

