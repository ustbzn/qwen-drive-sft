# Qwen-Drive 训练、评测与数据处理交接文档

> 面向：接手训练、评测、数据准备和结果复盘的工程师。  
> 维护原则：脚本和配置是当前实现的第一事实来源；本文解释运行顺序、输入输出和容易踩坑的约束。  
> 覆盖范围：仓库内的 Qwen-Drive Planning Expert IL 流程、PhysicalAI-AV 数据处理、MOGO 数据适配、PDM 评测和场景 GIF 可视化。

## 1. 先看结论

当前仓库接入的是 Qwen-Drive-1.0 的 **Stage 3 Planning Expert imitation learning**：

```text
MOGO / PhysicalAI-AV
        |
        | 16 个历史自车状态 @10Hz
        | 每路相机取 4 帧：t=-1.5、-1.0、-0.5、0s
        | 未来 50 个轨迹点 @10Hz
        v
Qwen-Drive 4B VLM（在线前向、冻结、no_grad）
        |
        | full-attention 层的 K/V + mRoPE position anchor
        v
Planning Expert（32 层 diffusion transformer）
        |
        | Flow Matching / clean-endpoint 预测
        v
50 点 @10Hz 轨迹
        |
        | NAVSIM 评测时重采样
        v
8 点 @2Hz、4 秒轨迹 -> PDM score
```

最小可用顺序：

1. 准备 MOGO 或 PhysicalAI-AV 数据。
2. PhysicalAI 用户先执行抽取，再执行图像缓存；MOGO 用户准备 `navsim_logs` 和 `sensor_blobs`。
3. IL 训练：`scripts/training/run_qwen_drive_il.sh`。
4. MOGO PDM 评测前准备 metric cache；再执行 `scripts/evaluation/run_qwen_drive_agent_pdm_score_evaluation.sh`。
5. 用 `scripts/vis_scenario_gif.py` 对指定 token 做预测、GT 和 LQR rollout 复盘。

当前流程**没有** Qwen-Drive VLM 端到端反向传播，也没有可直接接在这套脚本后的 RL/GRPO 入口。仓库中另有 ReCogDrive/Qwen3 RL 脚本，不能与本流程的 checkpoint、特征和配置混用。

## 2. 关键代码和职责

| 路径 | 职责 |
| --- | --- |
| `scripts/training/extract_physicalai_av.py` | 从 PhysicalAI-AV 的 ZIP 分片抽取 egomotion、时间戳和 MP4，并逐 clip 做完整性校验 |
| `scripts/training/cache_physicalai_images.py` | 将 4 帧图像按 Qwen patch grid 预缩放为 PNG，写入 uncompressed TAR shard 和随机读取索引 |
| `navsim/planning/training/physicalai_dataset.py` | PhysicalAI-AV 数据集适配；支持 ZIP 按需读取、已抽取文件读取和图像 cache |
| `navsim/planning/script/run_training_qwen_drive_il.py` | MOGO、PhysicalAI、mixed 三种训练数据模式；构建 DataLoader 和 Lightning Trainer |
| `navsim/agents/qwen_drive_planner/qwen_drive_agent.py` | 在线 VLM prefill、K/V padding/mask、Planning Expert flow matching、推理重采样 |
| `navsim/agents/qwen_drive_planner/qwen_drive_features.py` | 历史状态、导航命令、图像路径和未来轨迹 target 的构造 |
| `navsim/agents/qwen_drive_planner/qwen_drive_flow.py` | Beta 时间采样、直线插值、clean-endpoint loss、一阶/二阶 temporal Huber loss |
| `scripts/evaluation/run_qwen_drive_agent_pdm_score_evaluation.sh` | Qwen-Drive checkpoint 的多卡 PDM 评测包装脚本 |
| `navsim/planning/script/run_pdm_score_qwen.py` | 加载场景和 metric cache，执行 agent 推理与 PDM 打分，写 CSV |
| `scripts/vis_scenario_gif.py` | 对场景 token 渲染模型预测/GT 和 LQR rollout/GT GIF |
| `scripts/vis_scenario_video.py` | GIF 脚本复用的场景加载、agent 构造和相机/BEV 绘图工具 |
| `scripts/training/smoke_qwen_drive_il.py` | 单个 MOGO 样本验证数据、VLM、loss 和一次 Planning Expert 参数更新 |
| `scripts/training/physicalai_dataprocess.sh` | 依次执行 PhysicalAI 抽取和图像缓存的便捷入口 |

## 3. 环境和权重

### 3.1 Python 环境

火山云镜像：mogo-cn-beijing.cr.volces.com/pcpt/qwen-drive:v2

### 3.2 官方权重目录

默认路径在 `navsim/planning/script/config/common/agent/qwen_drive_agent.yaml` 和 shell 中：

```text
VLM_PATH=/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B
PLANNER_PATH=/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B/planner-sft
```

`QwenDriveAgent.initialize()` 会严格加载 VLM 和 `planner-sft`。`planner_path=null` 表示 Planning Expert 随机初始化，只适合调试或从头训练；通常应从发布的 `planner-sft` 继续微调。训练产出的 Lightning checkpoint 只覆盖 `agent._model.planning_expert.*`，评测时仍需同时提供原始 `VLM_PATH` 和 `PLANNER_PATH`。


### 3.3 轨迹和相机契约

这些值必须保持一致，改变后不能直接复用旧 checkpoint 或旧实验结果：

| 项目 | 当前值 |
| --- | --- |
| 历史状态 | 16 点，10Hz，覆盖 `-1.5s ... 0s` |
| 未来监督 | 50 点，10Hz，覆盖未来 5s |
| 每路相机帧数 | 4，取历史下标 `0/5/10/15` |
| 官方默认视角 | `cam_f0, cam_l0, cam_r0` |
| 当前训练 shell 默认视角 | `cam_f0, cam_l0, cam_r0, cam_l1, cam_r1, cam_b0` |
| 轨迹坐标 | 当前 ego frame，x 向前、y 向左、左转 heading 为正 |
| NAVSIM 输出 | 50 点 @10Hz 重采样为 8 点 @2Hz、4 秒 |
| 推理步数 | 10 步 Euler |
| 随机种子 | 默认 `42` |

官方三视角布局最接近公开模型的复现条件。当前 shell 显式传入六视角时，视角数量、prompt 长度和显存都会变化；六视角 checkpoint 不应与三视角配置交叉使用。

## 4. 数据目录约定

### 4.1 MOGO

训练/评测脚本需要 NAVSIM 风格目录：

```text
<DATA_ROOT>/dataset/
├── navsim_logs/
│   ├── train/*.pkl
│   ├── val/*.pkl                 # 评测 split 由脚本参数选择
│   └── trainval/*.pkl            # 视具体数据发布方式
├── sensor_blobs/
│   ├── train/...
│   └── val/...
└── maps/*.sqlite
```

MOGO 每个 pkl 需要能提供至少 66 个连续原始帧：16 个历史点加 50 个未来点。`mogo_qwen_drive` scene filter 使用 `num_history_frames=16`、`num_future_frames=50`、`frame_stride=1`，并用 `frame_interval=5` 让样本中心按约 0.5s 推进。

多数据源训练可以用 `mogo_manifest`。manifest 中每个 entry 独立指定 `data_root`、地图 sqlite、UTM zone 和可选的 raw scene cache；代码会按 train/val log split 构建各源数据，再检查 token 是否重复或 train/val 泄漏。

### 4.2 PhysicalAI-AV 原始目录

默认原始目录：

```text
/pcpt/pcpt/data/VLA/PhysicalAI-AV
```

脚本依赖以下元数据：

```text
clip_index.parquet
metadata/feature_presence.parquet
labels/egomotion/egomotion.chunk_XXXX.zip
camera/<camera_name>/<camera_name>.chunk_XXXX.zip
```

PhysicalAI 的语义视角映射：

| Qwen view | PhysicalAI camera |
| --- | --- |
| `cam_f0` | `camera_front_wide_120fov` |
| `cam_l0` | `camera_cross_left_120fov` |
| `cam_r0` | `camera_cross_right_120fov` |
| `cam_l1` | `camera_rear_left_70fov` |
| `cam_r1` | `camera_rear_right_70fov` |
| `cam_b0` | `camera_rear_tele_30fov` |

## 5. PhysicalAI-AV 数据处理

### 5.1 第一步：抽取和校验

推荐使用默认参数开始：

```bash
cd /pcpt/pcpt/project/jiangjunyuan/code/VLA/qwenDrive_recogdrive

python scripts/training/extract_physicalai_av.py \
  --source /pcpt/pcpt/data/VLA/PhysicalAI-AV \
  --dest /pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip \
  --views cam_f0 cam_l0 cam_r0 cam_l1 cam_r1 cam_b0 \
  --splits train val \
  --workers 48 \
  --anchor-time-s 10.0 \
  --video-check frames
```

抽取脚本会：

- 从 `clip_index.parquet` 和 `feature_presence.parquet` 选取 split 内、标记有效且包含所需模态的 clip；
- 对每个 clip 检查 egomotion 时间戳递增、覆盖 `anchor-1.5s ... anchor+5s`、数值有限、四元数有效；
- 对每路相机按最近时间选取 `anchor + [-1.5, -1.0, -0.5, 0]s` 的四帧，允许最大对齐误差 50ms；
- 用 decord 实际解码选中的 MP4 帧；
- 采用临时目录和原子替换，失败 clip 不会留下半成品。

输出目录：

```text
<DEST>/
├── clip_index.parquet
├── metadata/feature_presence.parquet
├── camera/<camera_name>/<clip>.<camera_name>.timestamps.parquet
├── camera/<camera_name>/<clip>.<camera_name>.mp4
├── labels/egomotion/<clip>.egomotion.parquet
└── .physicalai_extract/
    ├── chunks/chunk_XXXX.json
    ├── valid_clips.jsonl
    ├── bad_clips.jsonl
    └── summary.json
```

只有 `.physicalai_extract/summary.json` 中 `complete=true` 且 clip 出现在 `valid_clips.jsonl`，才算可用于 `physicalai_storage=files`。抽取支持断点复用；改变 source、views、split、anchor 或 `video_check` 后会自动失效对应状态。若要重新尝试上次失败的 clip，加 `--retry-failed`。

### 5.2 第二步：图像 TAR cache

```bash
python scripts/training/cache_physicalai_images.py \
  --root /pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip \
  --dest /pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip/.physicalai_image_cache \
  --views cam_f0 cam_l0 cam_r0 cam_l1 cam_r1 cam_b0 \
  --splits train val \
  --workers 96 \
  --clips-per-shard 128 \
  --history-image-pixels 174080 \
  --current-image-pixels 921600 \
  --patch-size 16 \
  --merge-size 2
```

cache 的特点：

- 每个 clip、每路相机保存 4 张无损 RGB PNG；历史帧使用 `174080` pixel budget，当前帧使用 `921600`；
- 图像已按 `patch_size * merge_size` 对齐，并使用 `torchvision`/PIL bicubic resize；
- TAR 使用 USTAR、**不压缩**，JSON index 记录 member offset 和 byte 数，因此训练 worker 可以随机读取单张图；
- `manifest.json` 记录 extraction plan、源文件签名、Pillow/torchvision 版本和 resize 参数；设置变化时应换新 `--dest`，不要覆盖成混合 cache；
- cache 只在抽取完成且 `video_check=frames` 时有效。数据源或抽取结果变化后，adapter 会做签名校验，失配时退回直接读取文件或判定 cache 不兼容。

输出：

```text
<CACHE>/
├── manifest.json
└── shards/
    ├── train-000000.tar
    ├── train-000000.json
    ├── val-000000.tar
    └── val-000000.json
```

解压、缓存脚本：

```bash
bash scripts/training/physicalai_dataprocess.sh
```

该入口使用脚本内默认路径，换机器时优先显式执行两个 Python 命令，避免误写默认目录。

## 6. Qwen-Drive IL 训练

### 6.1 默认训练命令

```bash
export PYTHON=/root/miniconda3/envs/qwen-drive/bin/python
bash scripts/training/run_qwen_drive_il.sh
```

当前 shell 默认值（以脚本实际内容为准）：

| 变量 | 默认值 |
| --- | --- |
| `DATASET_MODE` | `mixed` |
| `PHYSICALAI_ROOT` | `/pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip` |
| `PHYSICALAI_STORAGE` | `files` |
| `PHYSICALAI_IMAGE_CACHE_DIR` | `auto` |
| `PHYSICALAI_VIEW_LIST` | 六视角：`cam_f0 cam_l0 cam_r0 cam_b0 cam_l1 cam_r1` |
| `IMAGE_VIEW_LIST` | 六视角：`cam_f0 cam_l0 cam_r0 cam_l1 cam_r1 cam_b0` |
| `PHYSICALAI_ANCHOR_TIME_S` | `10.0` |
| `PHYSICALAI_SCAN_WORKERS` | `48` |
| `PHYSICALAI_MIXTURE_RATIO` | `proportional` |
| `GPUS` | `8` |
| `BATCH_SIZE` | `56`（每卡 micro-batch） |
| `GRAD_ACCUM` | `1` |
| `LR` | `2e-5` |
| `MAX_EPOCHS` | `5` |
| `PRECISION` | `bf16-mixed` |
| `NUM_WORKERS` | `8` |
| `SCENE_LOADING` | `indexed` |

脚本最终调用：

```bash
python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" \
  --module navsim.planning.script.run_training_qwen_drive_il \
  ...
```

训练日志重定向到仓库根目录的 `train_qwen_drive_il.txt`。训练结束后脚本还会调用：

```bash
python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all
```

如果新机器没有该工具，训练本身可能已经成功，但 shell 会在最后一步报错；查看日志末尾和 checkpoint 目录确认训练状态。

### 6.2 三种数据模式

训练修改scripts/training/run_qwen_drive_il.sh其中的配置

```bash
# 只训练 MOGO
DATASET_MODE=mogo bash scripts/training/run_qwen_drive_il.sh

# 只训练 PhysicalAI-AV
DATASET_MODE=physicalai bash scripts/training/run_qwen_drive_il.sh

# MOGO + PhysicalAI（shell 默认）
DATASET_MODE=mixed bash scripts/training/run_qwen_drive_il.sh
```

MOGO 数据默认通过 `navsim/planning/script/config/common/mogo_dataset/nav_station.yaml` manifest 选择；也可以关闭 manifest，显式指定：

```bash
DATASET_MODE=mogo \
MOGO_MANIFEST= \
DATA_ROOT=/path/to/data/dataset \
LOG_SPLIT=mogo \
bash scripts/training/run_qwen_drive_il.sh
```

mixed 模式的 `PHYSICALAI_MIXTURE_RATIO`：

- `proportional`：两个数据集按原始长度拼接；
- `1:1`、`2:1` 等整数比例：按周期重复较小数据源，使 MOGO:PhysicalAI 采样权重达到指定比例；
- 训练集按比例混合，验证集始终是 MOGO val 和 PhysicalAI val 的简单拼接，不重复验证样本。

MOGO 的 `OVERSAMPLE_FACTOR` 只支持 `DATASET_MODE=mogo`，默认是 20；mixed/physicalai 必须保持 `OVERSAMPLE_FACTOR=1`。

### 6.3 输出目录和 checkpoint

默认实验名是 `qwen_drive_il_6cam`。输出根目录：

```text
<RUNS_ROOT>/<experiment_name>/<experiment_uid>/
├── code/hydra/                         # Hydra 解析后的配置
├── lightning_logs/version_0/checkpoints/
│   ├── epoch=...-step=....ckpt
│   └── ...
└── ...
```

可显式指定：

```bash
EXPERIMENT_NAME=qwen_drive_il_debug \
RUNS_ROOT=/path/to/qwen_drive_runs \
RUN_ID=debug_001 \
GPUS=1 BATCH_SIZE=1 MAX_EPOCHS=1 \
PHYSICALAI_MAX_SAMPLES_PER_SPLIT=8 \
OVERSAMPLE_FACTOR=1 \
bash scripts/training/run_qwen_drive_il.sh \
  trainer.params.limit_train_batches=2 \
  trainer.params.limit_val_batches=1
```

## 8. PDM 评测

### 8.1 默认命令

修改以下脚本配置

```bash
bash scripts/evaluation/run_qwen_drive_agent_pdm_score_evaluation.sh
```

常用覆盖参数：

```bash
# 只跑少量场景（由 Hydra scene filter 覆盖）
bash scripts/evaluation/run_qwen_drive_agent_pdm_score_evaluation.sh \
  train_test_split.scene_filter.max_scenes=10

# 修改采样行为
bash scripts/evaluation/run_qwen_drive_agent_pdm_score_evaluation.sh \
  agent.num_inference_steps=10 \
  agent.num_samples=1 \
  agent.noise_seed=42
```

评测脚本默认 `QWEN_VIEW_LIST=[cam_f0,cam_l0,cam_r0,cam_l1,cam_r1,cam_b0]`，必须与 checkpoint 训练时的视角布局一致。`QwenDriveAgent.compute_trajectory()` 将模型的 50 点 @10Hz 结果转换为 NAVSIM 所需的 8 点 @2Hz；PDM 使用这 8 点做 4 秒闭环仿真。

### 8.2 输出

输出目录来自 `NAVSIM_EXP_ROOT/EXPERIMENT_NAME/experiment_uid`，CSV 文件名是时间戳，例如：

```text
<NAVSIM_EXP_ROOT>/qwen_drive_il_6cam_eval/<timestamp>.csv
```

### 8.3 评测一致性检查

| 检查项 | 要求 |
| --- | --- |
| `DATA_ROOT` | 与 checkpoint 训练所针对的数据版本一致 |
| `DATA_SPLIT` | `val`、`trainval` 等要和 metric cache 生成 split 对齐 |
| `METRIC_CACHE_PATH` | 由同一份 raw pkl、地图 sqlite 和 scene filter 生成 |
| `QWEN_VIEW_LIST` | 与训练 checkpoint 完全一致 |
| `VLM_PATH/PLANNER_PATH` | 与训练时相同的官方权重版本 |
| `agent.num_inference_steps` | 改变会改变轨迹和 score，应记录在实验名/配置中 |
| `agent.noise_seed` | 固定后结果可复现；多样本策略还需记录 `num_samples` |

## 9. 场景 GIF 复盘

### 9.1 Qwen-Drive GIF

```bash
python scripts/vis_scenario_gif.py \
  --agent-family qwen_drive \
  --model il \
  --checkpoint /path/to/epoch=...ckpt \
  --qwen-drive-vlm-path /D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B \
  --planner-path /D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B/planner-sft \
  --openscene-data-root /path/to/data/dataset \
  --split val \
  --token <scenario_token> \
  --output-dir vis_output/gifs/qwen_drive_IL_6cam \
  --window 30 \
  --gpu 0
```

`--token` 可以传多个；也可以把 token 放成位置参数。默认会生成两套 GIF：

```text
<token>_il.gif       # 模型原始预测 + GT
<token>_il_lqr.gif   # PDM 同款 LQR/自行车模型 rollout + GT
```

只看 BEV：

```bash
python scripts/vis_scenario_gif.py \
  --agent-family qwen_drive --model il --bev-only \
  --checkpoint /path/to/ckpt \
  --token <scenario_token> \
  --openscene-data-root /path/to/data/dataset \
  --split val
```

地图参数必须和 metric cache 一致：

```bash
--mogo-map-sqlite-path /path/to/maps/<city>.sqlite \
--mogo-map-utm-zone 50
```

如果只为复盘模型轨迹而不是做 PDM 打分，仍需要能加载场景和相机；MOGO 的 BEV 图层还依赖正确的 sqlite 地图。

## 相关文档

- `Qwen-Drive模型结构技术细节.md`：当前代码对模型结构、输入时间采样和 flow matching 的较详细说明。
- `Qwen-Drive-1.0-analysis.md`：论文/模型背景分析；其中部分内容描述完整论文流程，不等于本仓库当前脚本已经实现的全部能力。
- `CLAUDE.md`：仓库级数据、缓存、地图和 Qwen3/ReCogDrive 工作流约定；其中 ReCogDrive MOGO 流程与本文 Qwen-Drive IL 流程是两条不同链路。
- `scripts/AGENTS.md`：脚本目录的入口清单和本地绝对路径提醒。

