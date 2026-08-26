# ReCogDrive 进出站导航：navigation_command 编码 + 站点 token 实现方案

创建于 2026-08-24。本文档记录「让规划器学会进站/出站」这一能力的完整设计、代码改动、
兼容性约束、开关矩阵、参考依据与验证方法。数据字段语义详见
`docs/recogdrive_navigation_command_generation.md`（生成侧），本文档聚焦模型侧消费。

## 1. 背景与目标

### 1.1 为什么做

新版 Mogo pkl（样例
`/D2/NXT/lst/vla/mogo_data/centerline_data_pkl/dataset/navsim_logs/trainval/*.pkl`）
每帧新增两个字段，用于支持接驳/公交车的**进出站（靠边停车 / 驶离站点）**能力：

- **`navigation_command`**：恒为两个四元组 `[type, dist, a, b]`。
  - `type ∈ {None, straight, turn left, turn right, u turn, park}`。
  - `dist` = 到该动作/站点入口的沿 route 剩余距离（m）。
  - `(a, b)` = 车道范围；对 `park`，`(1,1)`=站点在**左侧**、`(-1,-1)`=站点在**右侧**。
  - 相比旧的 4 维 one-hot `driving_command`（只有 left/straight/right/u-turn），
    它多了**距离**与**车道/侧向**信息，且用 `park` 显式表达进出站意图。
  - 实测：`park` 的 `dist` 随车辆逼近站点从 ~70m 单调递减到 ~1m，是清晰的进站进度信号。
- **`route_stations`**：整条线路站点表（全局 UTM `point_xy` + role + route_progress），
  各帧相同，是静态目录。

**用户需求（5 条）→ 本方案的落点**：

| 需求 | 落点 |
| --- | --- |
| 1. 输出轨迹在站点处呈现明显进/出站动作 | station token（把当前活动站点投影到 ego 系作为 DiT context 目标点）+ navigation_command 条件，靠 IL 从 GT 学 |
| 2. 消费 navigation_command | `encode_nav_command_vector` 编码两个四元组的 type/dist(双尺度)/车道范围(a,b) + park 的 dist/侧向（P4 后；未编码 duration 等 meta 细节） |
| 3. navigation_command 编码进 ego state，替换 one-hot | `use_navigation_command=true` 时 status_feature 命令段换成 navigation_command 编码 |
| 4. 与旧数据兼容 | 统一编码 + 自动退化：无字段时从 driving_command 退化、station 全 padding |
| 5. 参考顶刊/开源 | goal-conditioned E2E（UniAD/VAD goal、DiffusionDrive anchor）、pluto SDE，见 §7 |
| 6. 输出文档 | 本文件 |

### 1.2 关键设计决策（已与需求方确认）

1. **实现深度 = B**：navigation_command 编码进 ego state（含 park dist/侧向）
   **+** station token 进 DiT cross-attention。不加显式 station 辅助 loss（KISS/YAGNI，
   靠 IL 从 GT 轨迹学，避免调权重/过拟合）。
2. **兼容 = 统一编码 + 自动退化**：单一 `encode_nav_command_vector` 函数——有
   navigation_command 用全通道，旧数据从 `driving_command` 映射 type、dist/side 置 0。
3. **启用 = 默认关的开关**（与 obj/hdmap/img/sde 完全同模式）：
   `agent.use_navigation_command` / `agent.use_station_token`，默认 `false` 时**字节级
   等价当前链路**。
4. **站点坐标来源 = nav_command 派生**：station token 由 `park` 的 dist/侧向 +
   `route_centerline` 弧长投影派生（无 park → station 全 padding）。

## 2. 数据流总览

```
新版 Mogo pkl 每帧
├─ navigation_command [[type,dist,a,b],[type,dist,a,b]]
├─ route_stations     [{point_xy(全局UTM), role, route_progress, ...}]
└─ route_centerline   [[x,y,heading](全局UTM), ...]
        │
        ▼  navsim/common/dataclasses.py（.get 兜底，旧数据 None）
   EgoStatus.navigation_command / .route_stations
   Frame.navigation_command / .route_stations / .route_centerline
        │
        ├────────────────────────────────┬───────────────────────────────┐
        ▼                                 ▼                               ▼
 build_recogdrive_ego_features    StationFeatureBuilder            (VLM prompt 仍用
 (use_navigation_command=true)    .compute_features_from_scene      4 维 one-hot,
        │                                 │                          不受影响)
        ▼                                 ▼
 status_feature 命令段 =          station_features [1,6] =
 encode_nav_command_vector        meta 精确站点坐标(优先)/centerline 投影到 ego 系
 (21维: 主命令type6+dist双尺度      + 侧向 + dist，无 park → 全 padding
  +车道范围a,b | 次命令type6+dist   │
  | has_park+park双尺度+park_side)   ▼
        │                          StationEncoder (LayerNorm+MLP → D)
        ▼                                 │
 ego_status_encoder (mlp/SDE)             │
        │                                 ▼
        └──────► DiT ◄── cross-attention context = [VLM | hdmap | obj | img | station]
                  │
                  ▼
           8×(x,y,heading) 轨迹（进/出站形状由 IL 从 GT 学）
```

## 3. 代码改动清单

| 文件 | 改动 |
| --- | --- |
| `navsim/common/dataclasses.py` | `EgoStatus`/`Frame` 加 `navigation_command`/`route_stations`/`navigation_command_meta`（Optional，`.get` 兜底）；`_build_ego_status`、`AgentInput.from_scene_dict_list`、`Scene.get_agent_input`、`Scene.from_scene_dict_list` 四处构造透传 |
| `navsim/common/command_utils.py` | 新增 `NAV_COMMAND_TYPES`/`NAV_COMMAND_DIM(=21)`/dist(200)+log+lane 归一化常量、`encode_nav_command_vector(nav, driving)`（两个四元组都编码）、`parse_park_command(nav)` |
| `navsim/agents/recogdrive/recogdrive_features.py` | `build_recogdrive_ego_features(..., use_navigation_command)` 按开关换命令段；`ego_status_feature_names`/`ego_status_dim` helper；两个 builder 加 `use_navigation_command`；`ReCogDriveEgoFeatureBuilder` 版本化缓存名 `recogdrive_ego_feature_v2_nav` |
| `navsim/agents/recogdrive/station_features.py` | **新建** `StationFeatureBuilder`（`STATION_FEATURE_DIM=6`，缓存名 `station_feature_v1`）：站点位置优先用 `navigation_command_meta` 精确全局坐标投影到 ego 系（P1），退化 route_centerline 弧长反投影，无 park 全 padding；新增 `collate_station_features` helper |
| **collate_fn（4 处）** `run_training_recogdrive.py` / `run_training_recogdrive_rl.py` / `run_recogdrive_accelerate.py` / `run_training_recogdrive_ema.py` | 各 import 并调用 `collate_station_features`，把 station_features/station_valid_mask stack 进 batch（**P0 修复**：否则训练 batch 丢弃站点字段、StationEncoder 无梯度） |
| `navsim/agents/recogdrive/recogdrive_diffusion_planner.py` | 新增 `StationEncoder`；config 加 `use_station_token`/`station_feature_dim`/`max_station_tokens`；`build_context_tokens` 与 `_build_context_tokens_no_vlm` 追加 station token+mask；`_extract_station_inputs`/`encode_station_context_tokens`；GRPO/frozen 时 station_encoder 切 eval |
| `navsim/agents/recogdrive/recogdrive_agent.py` | 加 `use_navigation_command`/`use_station_token`/`max_station_tokens` 入参；`requires_scene` 并入 station；`_ego_status_dim`/`_ego_num_command_states` 按开关计算并透传；`make_recogdrive_config` 加 `ego_status_dim`/station 参数；`get_feature_builders` 透传+append StationFeatureBuilder；`forward` 注入 station_data |
| `.../config/common/agent/recogdrive_agent.yaml` | 新增 `use_navigation_command`/`use_station_token`/`max_station_tokens`（默认 false/1）+ 中文注释 |
| `scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh` | `USE_NAVIGATION_COMMAND`（slim 也生效）/`USE_STATION_TOKEN` → override |
| `scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh` | 同上 + experiment_name 加 `${NAV_TAG}` |
| `scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh` | `NAV_ARGS` 数组展开 + experiment_name tag |
| `scripts/evaluation/run_recogdrive_qwen3_agent_pdm_score_evaluation_2b.sh` | 同训练脚本 |

### 3.1 navigation_command 编码布局（NAV_COMMAND_DIM = 21）

两个四元组**都**编码（P4 修复：不再只取第一条+丢 b+dist 截断）：

```
索引    含义
主命令（第一条非 park 动作，无则 park）
0-5     type one-hot: [none, straight, turn left, turn right, u turn, park]
6       dist_norm    dist/200，clamp[0,1]（近距离高分辨率）
7       dist_log     log1p(dist)/log1p(1000)（远距离不饱和，缓解 200~1000m 截断）
8       lane_a_norm  车道范围起点 a/10（保留符号；park 时为侧向 ±1/10）
9       lane_b_norm  车道范围终点 b/10（P4：b 不再丢弃）
次命令（第二条非 park 动作，覆盖下一路口动作/提前变道；无则全 0）
10-15   type one-hot
16      dist_norm
park（进出站关键信号，与路口动作并存）
17      has_park     (1/0)
18      park_dist_norm  park.dist/200 ← 进站进度信号
19      park_dist_log   log1p(park.dist)/log1p(1000)
20      park_side    park (a,b) 符号: +1 左侧站 / -1 右侧站（park 左右侧语义明确）
```

- **主命令**取第一条非 park 动作（无则 park）；**次命令**取第二条非 park（P4：第 2 条不再丢）。
- **车道范围 vs 侧向**：转向/直行的 `(a,b)` 是车道序号范围（如 [2,3]），故 `lane_a/lane_b`
  两端都归一化编码；只有 **park** 的 `(a,b)` 是明确左右侧向，单独用 `park_side` 表达（P4：
  修正原先把转向车道范围当侧向的语义错误）。
- **旧数据退化**：`navigation_command` 缺失时，仅按 `driving_command` 4 维 one-hot 映射主命令
  `type`，其余段全 0。开关开启后旧数据照常训练。
- 因 NAV_COMMAND_DIM=21，启用 nav 时 `ego_status_dim = 21 + 9 = 30`（SDE 永久可见命令段=21）。

### 3.2 station token 特征布局（STATION_FEATURE_DIM = 6）

```
[x_norm(x/100), y_norm(y/40), dist_norm(park.dist/200), sin_bearing, cos_bearing, side]
```

- 站点全局 (x,y) **三级来源，优先精确**（P1 修复）：
  1. `navigation_command_meta.station_target_buffer.current_target.point_xy`（生成侧算好的
     **精确活动站点全局坐标**，实测 100% 覆盖 park 帧，能处理站点偏离中心线/横向偏移）；
  2. 退化：`park.dist` 沿 `route_centerline` 累积弧长采点（站点近似在中心线上）；
  3. 再退化：`(dist, side*1.75)` 粗略占位。
  得到全局 (x,y) 后用当前帧**全局** `ego_pose` 逆变换到 ego 后轴系（复用
  `safety_targets._global_xy_to_local` 公式）。
- **有效性规则**（与代码一致，P8 对齐）：
  - **无 park / 旧数据（含无 navigation_command）** → `station_valid_mask=False`（全 padding），
    `StationEncoder` 按 mask 置零，DiT 忽略。
  - **有 park** → `station_valid_mask=True`（有效 token），坐标按上述三级来源取；即便 meta 与
    route_centerline 几何都缺失，也用第 3 级 `(dist, side*1.75)` 近似占位而**非 padding**，
    保证「已知要进站」时进站信号（dist/side）不丢。实践中新 pkl 每帧都带 route_centerline 且
    meta 100% 覆盖 park 帧，第 3 级近似几乎不触发。
- 已核实：偏离站点案例用 meta 精确坐标得 y_local=-50.6m（横向偏移），非硬编码车道半宽。
- STATION_FEATURE_DIM=6 的 `side` 取自 park `(a,b)` 符号（+1 左 / -1 右）。

> **为何不直接遍历 `route_stations`**：`route_stations` 是整条线路目录，远处站点不在当前
> ~200m centerline/感知范围内（实测某些 log waypoint 距 ego 1700m）。"当前活动站点"由生成侧
> 的缓冲逻辑（见生成文档）选出并写入 `meta.current_target`，故直接取 meta 比自行从
> `route_stations` 挑选活动站点更可靠。`route_stations` 的 role/route_progress 目前未进模型
> （出站阶段建模若启用会用到，见 §10）。

## 4. 开关矩阵与缓存文件

| 开关组合 | ego_status_dim | ego sidecar 缓存名 | VLM 缓存名(use_vlm=true) | 新增缓存 | DiT context |
| --- | --- | --- | --- | --- | --- |
| 全关（默认） | 13 | `recogdrive_ego_feature_v1` | `qwen3_feature` | — | VLM(+hdmap/obj/img) |
| `use_navigation_command=true` | 30 | `recogdrive_ego_feature_v2_nav` | `qwen3_feature_nav` | — | 同上（命令段编码变化，仅进 ego state） |
| `+ use_station_token=true` | 30 | `recogdrive_ego_feature_v2_nav` | `qwen3_feature_nav` | `station_feature_v1.gz` | +station token |

> **P5（缓存版本隔离）**：use_vlm=true 训练时 status_feature 来自 VLM 缓存文件（由
> `ReCogDriveFeatureBuilder` 写入，与 VLM hidden state 同存一个 .gz），非独立 ego sidecar。
> 开 nav 后 status_feature 从 13→30 维。缓存写入逐 builder 判 `exists()` 跳过（即便
> `force_cache_computation=true` 也跳过已存在文件），若沿用旧名 `qwen3_feature.gz` 会读到旧
> 13 维状态而与 planner 首层冲突。**修复**：开 nav 时 VLM 缓存名加 `_nav` 后缀
> （`qwen3_feature_nav`），新名不存在必然重算，与旧缓存彻底隔离（代价：VLM hidden state 随之
> 重算一次，正确性优先）。use_vlm=false 训练读独立版本化的 `recogdrive_ego_feature_v2_nav.gz`。

## 5. 兼容性与迁移约束

- **默认关 = 零影响**：两开关默认 `false` → `ego_status_dim=13`、无 station token、
  缓存名不变、status_feature 字节级不变、旧 ckpt 完全可用。**当前所有实验不受影响。**
- **旧数据即使开开关也不崩**：`encode_nav_command_vector` 从 `driving_command` 退化；
  station 全 padding。可在旧数据上做「开关开启」的消融而无需新字段。
- **ckpt 不可跨 use_navigation_command 移植**：命令段维度变 → `ego_status_encoder` 首层
  形状变（mlp 的 `in_features` 或 SDE 的 `linears`/`pos_embed` 长度），旧 ckpt 加载时被
  `initialize()` 的 `strict=False` 静默丢弃 → 等价这些层重新训练。开启后必须**重训 IL**，
  并让 RL/eval 用同源 ckpt。
- **station_encoder 层同理**：旧 ckpt 无该层，`use_station_token=true` 时新增，须重训。
- **全链路一致**：缓存 / IL / RL / eval / 可视化必须统一 `use_navigation_command` 与
  `use_station_token`，否则缓存缺 `station_feature_v1.gz`（`CacheOnlyDataset` 过滤样本）
  或 ego 缓存名对不上（读到旧 v1 → 维度冲突）。
- **单站点约束（P6）**：`max_station_tokens` 当前仅支持 1（只编码当前活动站点），
  `StationFeatureBuilder` 对 !=1 报错防呆。多站点为未来扩展（见 §10.2）。
- **通用 submission 入口不兼容（P7）**：启用 station token（或 obj/hdmap）会置
  `requires_scene=True`，而 `run_create_submission_pickle.py` 显式拒绝 `requires_scene` 的
  agent（NAVSIM 测试服务器只给 AgentInput、无标注 scene）。这是 obj/hdmap 已有的同源约束，
  **非 station 新引入**。station 评测走 PDM 路径（`run_pdm_score_recogdrive.py`，传 scene）与
  可视化脚本；通用 submission 导出需在导航字段搬入 AgentInput + 提供全局 pose 后才可能支持
  （未做，与 obj/hdmap 现状一致）。

## 6. 与现有可选模块的关系（正交性）

- 与 **vlm_ablation / use_vlm** 正交：station token 在 use_vlm=false 分支
  （`_build_context_tokens_no_vlm`）也已接入；vlm_ablation=zero 只置零 VLM token，
  station token 保留。
- 与 **SDE 状态编码器**（`ego_status_encoder_type=attention`）协同：nav 命令段（前
  `NAV_COMMAND_DIM=21` 维）作为 SDE 的**永久可见**段（`ego_num_command_states` 自动置为
  `NAV_COMMAND_DIM`），运动学量仍按 dropout 遮蔽——进出站意图不被当作可丢的捷径。
- 与 **obj / hdmap / image token** 并列拼接进 DiT，顺序为
  `[VLM | hdmap | obj | image | station]`，各自独立 mask。

## 7. 参考依据（顶刊 / 开源）

- **Goal-conditioned end-to-end driving**：UniAD（CVPR'23 best paper）、VAD/VADv2 用
  goal point / 驾驶命令条件化规划头。本方案的 station token 即「显式空间 goal」的实例化。
- **Anchor / proposal-based diffusion planning**：DiffusionDrive（2024）用锚点提议引导
  扩散轨迹解码。station token 作为进出站锚点，向 DiT cross-attention 注入目标点先验。
- **命令 / 语言条件扩散规划**：ReCogDrive 本身的 VLM + diffusion planner 架构；本方案
  把「离散命令 one-hot」升级为「带距离/侧向的结构化命令 + 空间目标 token」。
- **pluto 状态丢弃编码器（SDE）**：本仓 `StateAttentionEncoder` 已沿用，nav 命令段作为
  永久可见维接入其掩码布局。
- 站点全局→ego 坐标变换复用本仓 `safety_targets._global_xy_to_local`。

## 8. 验证

1. **零影响回归**（默认关）：
   ```bash
   python -m pytest tests/test_qwen3_recogdrive_integration.py -k "config or planner or select" -q
   python -m py_compile navsim/agents/recogdrive/recogdrive_agent.py \
     navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
     navsim/agents/recogdrive/recogdrive_features.py \
     navsim/agents/recogdrive/station_features.py \
     navsim/common/dataclasses.py navsim/common/command_utils.py
   ```
   确认默认 `ego_status_dim==13`、缓存名 `recogdrive_ego_feature_v1` 不变。
2. **编码函数单测**：`encode_nav_command_vector` 对（新数据带 park / park-only / 旧数据
   driving_command）三种输入返回正确 21 维向量（已通过：主命令 straight+车道范围[2,3]、
   park 左/右侧、第 2 条指令保留、远距离 dist_log 区分、旧数据退化）。
3. **station builder 单测**：park dist 递减时 `x_local` 递减、`valid=True`；无 park /
   旧数据 `valid=False`（已通过：46.8→25.5→15.1m 对应 x_local 精确递减）。
4. **planner 前向**：`use_station_token=true` 下 `get_action` / IL forward 输出 `[B,8,3]`，
   padding station token 置零（已通过 use_vlm true/false 两种）。
5. **端到端 fast_dev_run**（需先小规模缓存新数据）：
   ```bash
   USE_NAVIGATION_COMMAND=true USE_STATION_TOKEN=true \
   torchrun --nproc_per_node=1 navsim/planning/script/run_training_recogdrive.py \
     agent=recogdrive_agent agent.use_navigation_command=true agent.use_station_token=true \
     train_test_split=mogo_trainval trainer.params.fast_dev_run=true
   ```
6. **可视化确认进出站**：`scripts/vis_scenario_video.py` 在 park 场景 token 上渲染，肉眼
   确认输出轨迹在站点处呈现靠边/驶离形状。

## 9. 开启 checklist（生产）

1. 确认目标数据集 pkl 带 `navigation_command` + `route_stations`（旧数据仅退化，不学进出站）。
2. 重跑 hidden-state 缓存：`USE_NAVIGATION_COMMAND=true USE_STATION_TOKEN=true bash
   scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh`
   → 补出 `station_feature_v1.gz`；ego sidecar 写为 `recogdrive_ego_feature_v2_nav.gz`；
   use_vlm=true 时 VLM 缓存写为 `qwen3_feature_nav.gz`（P5，与旧 `qwen3_feature.gz` 隔离，
   旧文件可保留不影响）。
3. 用 `scripts/cache_dataset/check_recogdrive_cache_products.py` 核对缓存完整性。
4. 重训 IL（同两开关，**不要热启旧 ckpt 的 ego/station 层**）：
   `USE_NAVIGATION_COMMAND=true USE_STATION_TOKEN=true bash
   scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh`。
5. RL / eval 用新 IL ckpt，且脚本同设两开关。

## 10. 评审修复记录与已知限制

### 10.1 已修复

- **P0（致命）— 训练 collate 丢弃站点特征**：4 个训练入口的 `custom_collate_fn` 原本只
  显式 stack history/command/status + obj/hdmap/image，**未处理 station_features**，导致
  缓存的 `station_feature_v1.gz` 被读入后在 collate 阶段丢弃，`_extract_station_inputs`
  返回 None，StationEncoder 无梯度、站点功能实际不生效。**修复**：新增
  `station_features.collate_station_features` helper 并在 4 处 collate 调用。已验证：collate
  后 batch 含 station 字段，破除 adaLN-zero gate 后 station_encoder 与 obj_encoder 梯度同步
  流入（0.085/0.060）。注：随机初始化下所有 cross-attn context 路径（VLM/obj/hdmap/station）
  初始梯度为 0 是 adaLN-zero 的正常行为，非断路。
- **P1 — route_stations 精确坐标未被消费**：原实现仅用 park.dist 沿 route_centerline 反投影，
  实测不同 log 站点到中心线距离 0.4~1700m，方法脆弱且横向偏移靠硬编码 1.75m 猜。**修复**：
  改用 `navigation_command_meta.station_target_buffer.current_target.point_xy`（精确活动站点
  全局坐标，100% 覆盖 park 帧）投影到 ego 系，保留 centerline/dist 为退化兜底。已验证偏离站点
  案例得 y_local=-50.6m。
- **P4 — navigation_command 编码有损**：原 11 维编码只取第一条非 park + 第一条 park、`(a,b)`
  只用 a 的符号（丢 b、且把转向的车道范围误当侧向）、dist 统一 200m 截断丢远距离区分。**修复**：
  扩展为 21 维（见 §3.1）——两个四元组都编码（补第 2 条普通指令 type+dist）、车道范围 a/b 两端
  都编码、dist 用 norm+log 双尺度缓解截断、park_side 单独用 park 的 (a,b) 符号（语义明确）。
  已验证第 2 条指令保留、车道范围负号保留、远距离 dist_log 区分（300 vs 800m）。
- **P5 — VLM 缓存未按 nav 版本隔离**：`qwen3_feature.gz` 与 status_feature 同存一文件，开 nav
  后 status_feature 维度变但文件名不变，重跑缓存时 `exists()` 跳过旧文件（`force_cache_computation`
  也不覆盖已存在文件）→ 训练读旧 13 维状态与 planner 冲突。**修复**：`ReCogDriveFeatureBuilder.
  get_unique_name` 开 nav 时加 `_nav` 后缀（`qwen3_feature_nav`），强制与旧缓存隔离。
- **P6 — max_station_tokens>1 未实现**：`_build` 只写 slot 0，且缓存名不含 token 数，改成 >1
  会读到 shape [1,6] 旧缓存 mismatch。**修复**：`StationFeatureBuilder.__init__` assert
  `max_station_tokens==1`，防呆报错；多站点列为未来扩展（§10.2）。
- **P7 — 通用 submission 入口不兼容**：见 §5「通用 submission 入口不兼容」。station 与
  obj/hdmap 同为 `requires_scene=True`，`run_create_submission_pickle.py` 一律拒绝——既有约束，
  非 station 引入。station 评测走 PDM/可视化路径。**处置**：文档明示，不改代码（保持与
  obj/hdmap 现状一致）。

生成侧规则：车辆到站（dist≤1m）后 `park` 被清除。因此**车辆驶离站点的帧通常没有 park、
没有 station token**（实测 park 序列尾部多为空；`station_park.reached & station_passed` 出站
信号仅 24 帧，很稀疏）。当前实现下，"出站"（从路边加速汇入车流）只能靠历史运动 + BC 隐式学，
**不保证在站点处形成明显、稳定的驶离行为**；且方案选择了「不加显式 station loss」，可验证性有限。

若需显式建模出站，可选扩展（未实现，待需求方决策，因涉及监督语义扩展 / 偏离原 KISS 决策）：

1. **出站阶段 station token（推荐，最小扩展）**：用 `route_stations.route_progress_m` 与自车
   route 进度对比，判定「刚驶离最近站点 N 米内」，此时也发一个 station token，用**负纵向距离**
   （站点在 ego 后方）+ side + 一个 `phase` 标志位（进站/出站）表达。station 特征加 1~2 维
   phase。需消费 route_stations（呼应 P1 里 role/progress 目前未用）。
2. **出站状态位进 nav 命令段**：`encode_nav_command_vector` 增加「已到站/正在出站」通道
   （从 `meta.station_park.reached/reason` 派生），让 ego state 显式携带出站意图。
3. **显式 station 靠边/驶离 loss**：在 park 末端帧对轨迹终点横向位置加监督（进站靠 side 侧、
   出站回中心线）。最强约束但需调权重、有过拟合风险，偏离原 YAGNI 决策。

进站（靠边减速）方向本方案已通过 station token + park.dist 递减信号 + IL 覆盖；出站是否需要
上述显式扩展，取决于闭环评测中驶离行为是否达标。
