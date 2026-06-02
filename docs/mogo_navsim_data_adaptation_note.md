# Mogo 到 NAVSIM 数据适配笔记

这份笔记记录当前 Mogo 内部数据适配到 NAVSIM 数据流的设计和实现，覆盖 ROS bag 到 `.pkl`、SQLite 地图 adapter、以及 metric cache 接入。前半部分关注数据生产端：

```text
内部 ROS bag
  -> parseRosBag-2O-RL-RecogDrive.py
  -> NAVSIM-style List[frame_dict]
  -> dataset/navsim_logs/{split}/*.pkl
  -> dataset/sensor_blobs/{split}/...
```

它和 `navsim_data_flow_note.md` 的关系是：

```text
mogo_navsim_data_adaptation_note.md
  重点：内部数据如何生产成 NAVSIM 可读取的 pkl / image 目录，并继续接入 Mogo 地图和 metric cache

navsim_data_flow_note.md
  重点：pkl 进入 NAVSIM 后如何被 SceneLoader、AgentInput、MetricCacheProcessor 消费
```

## 1. 脚本位置

当前转换脚本放在：

```text
/home/zhongwei/Vla/src/Drive_pi/mogo-dataset-tools/parseRosBag-2O-RL-RecogDrive.py
```

这个脚本是一个独立的数据生产脚本，不直接侵入现有 NAVSIM 数据流。它的目标是尽量在生产端把内部数据整理成 NAVSIM 当前能消费的 `frame_dict` 格式。

## 2. 输入数据假设

当前内部原始数据是一段段 ROS bag。一个上层原始数据目录下包含多个 log 片段目录，例如：

```text
/home/zhongwei/Vla/src/recog_drive/navsim/data/2026-03-19-15-24-05/
  20260319152428/
    *.bag
  20260319152441/
    *.bag
  ...
  2026-03-19-15-24-05-routing_results.bag
```

其中：

| 概念 | 当前约定 |
|---|---|
| 上层目录 | 一次连续采集或一组 log 片段 |
| 子目录 | 一个 NAVSIM log 片段，会生成一个 `.pkl` 文件 |
| routing bag | 该组 log 共用的 route 结果，优先用 `*routing_results*.bag` 自动发现 |
| camera / localization / perception bag | 每个 log 片段目录内部自己的 bag 文件 |

当前逻辑会把每个包含 `.bag` 文件的子目录视为一个 log 片段。

## 3. 输出目录结构

用户通过 `--dataset-root` 指定输出根目录。脚本会确保目录结构存在：

```text
用户指定根目录/
├── dataset/
│   ├── maps/
│   ├── navsim_logs/
│   │   ├── test/
│   │   └── trainval/
│   └── sensor_blobs/
│       ├── test/
│       └── trainval/
└── exp/
```

目录检查只做 `mkdir -p` 类型的补齐：

```text
缺什么目录就创建什么目录
目录已经存在就直接使用
不会清空已有数据
不会删除已有 pkl 或图片
```

典型输出是：

```text
{dataset_root}/dataset/navsim_logs/trainval/{log_name}.pkl
{dataset_root}/dataset/sensor_blobs/trainval/{log_name}/...
```

如果 `--split test`，则写入 `test` 子目录。

## 4. 推荐执行命令

进入脚本目录：

```bash
cd /home/zhongwei/Vla/src/Drive_pi/mogo-dataset-tools
```

批量转换一个原始数据目录：

```bash
python3 parseRosBag-2O-RL-RecogDrive.py \
  --raw-data-root /home/zhongwei/Vla/src/recog_drive/navsim/data/2026-03-19-15-24-05 \
  --dataset-root /path/to/your/output_root \
  --split trainval
```

先检查输入、输出路径和 topics，不真正生产数据：

```bash
python3 parseRosBag-2O-RL-RecogDrive.py \
  --raw-data-root /home/zhongwei/Vla/src/recog_drive/navsim/data/2026-03-19-15-24-05 \
  --dataset-root /path/to/your/output_root \
  --split trainval \
  --dry-run
```

如果 route bag 不能自动发现，可以手动指定：

```bash
python3 parseRosBag-2O-RL-RecogDrive.py \
  --raw-data-root /home/zhongwei/Vla/src/recog_drive/navsim/data/2026-03-19-15-24-05 \
  --route-source-path /home/zhongwei/Vla/src/recog_drive/navsim/data/2026-03-19-15-24-05/2026-03-19-15-24-05-routing_results.bag \
  --dataset-root /path/to/your/output_root \
  --split trainval
```

转换测试集时：

```bash
python3 parseRosBag-2O-RL-RecogDrive.py \
  --raw-data-root /path/to/raw_test_root \
  --dataset-root /path/to/your/output_root \
  --split test
```

单个 log 目录转换也支持：

```bash
python3 parseRosBag-2O-RL-RecogDrive.py \
  --input-path /path/to/one_log_dir \
  --route-source-path /path/to/routing_results.bag \
  --dataset-root /path/to/your/output_root \
  --split trainval
```

## 5. 关键参数

| 参数 | 含义 | 当前建议 |
|---|---|---|
| `--raw-data-root` | 包含多个 log 片段目录的原始数据根目录 | 批量生产时使用 |
| `--input-path` | 单个 bag 或单个 log 目录 | 调试单段数据时使用 |
| `--dataset-root` | NAVSIM 风格数据输出根目录 | 推荐使用 |
| `--split` | 输出到 `trainval` 或 `test` | 按训练 / 测试集选择 |
| `--route-source-path` | routing bag 或目录 | 自动发现失败时手动指定 |
| `--dry-run` | 只检查，不写 pkl / 图片 | 正式生产前建议先跑 |
| `--no-extract-images` | 不导出图片 payload | 只调试 pkl 字段时可用 |
| `--write-debug-json` | 额外写 debug json | 排查字段时可用 |
| `--sample-interval` | 采样间隔 | 默认 `0.1s` |
| `--max-camera-delta` | 多视角相机与采样时间戳的最近帧对齐阈值 | 默认 `0.05s` |
| `--max-obstacle-delta` | 障碍物与采样时间戳的最近帧对齐阈值 | 默认 `0.10s` |
| `--max-local-map-delta` | 局部地图与采样时间戳的最近帧对齐阈值 | 默认 `0.10s` |
| `--max-sensor-delta` | 兼容旧命令的统一阈值，会覆盖上面三项 | 默认不启用 |
| `--driving-command-distance` | driving command 向前检查距离 | 默认 `30m` |
| `--max-lateral-match-distance` | ego 到 route lane 最大匹配距离 | 默认 `2.0m` |

## 6. 主要 topic 对应关系

当前脚本尽量复用已有参考脚本中的 topic：

| 数据 | Topic | Proto / 来源 | 当前用途 |
|---|---|---|---|
| 自车定位 | `/localization/global` | `Localization` | ego pose、速度、加速度、can_bus 主来源 |
| 底盘状态 | `/chassis/chassis2025` | `ChassisStates` | can_bus 运动量 fallback |
| 障碍物 | `/perception/fusion/obstacles` | `TrackedObjects` | annotations / gt boxes |
| 局部地图 | `/hadmap_engine/map_msg` | 地图 proto | 当前主要保留读取入口，后续扩展 |
| route 结果 | `/hadmap_engine/route_result_msg` | `RoutingResult` | roadblock_ids、driving_command |
| 前视相机 | `/sensor/camera/sensing/image_raw_120/resize/nvjpeg` | compressed image payload | 默认采样时间轴、CAM_F0 |
| 其他相机 | `/sensor/camera/sensing/*/resize/nvjpeg` | compressed image payload | 多视角扩展，映射到 NAVSIM camera slot |

当前使用前视相机时间轴作为默认采样时间轴，因此如果前视相机是 10Hz，就自然生成 0.1s 间隔的 frame。其他视角相机会在各自 topic 中寻找距离该时间戳最近的帧，默认要求时间差不超过 `0.05s`；超过阈值时，该相机 slot 保持空占位。障碍物和局部地图使用独立阈值，默认分别是 `0.10s` 和 `0.10s`。

## 7. log_name / token / timestamp

### log_name

当前 log name 由两部分组成：

```text
{vehicle_id}_{base_log_name}
```

其中：

```text
vehicle_id 优先从 bag 文件名中的 -rosmaster-{vehicle_id}.bag 提取
如果提取失败，则使用 --vehicle-id
如果仍没有，则使用 unknown_vehicle

base_log_name 默认使用 log 目录名
单 input 模式下可以用 --log-name 覆盖
```

例如：

```text
JLBJA20073D_20260319152428
```

### token

每个 frame 的 token 使用 md5 生成，md5 前的原始字符串包含：

```text
{vehicle_id}:{base_log_name}:{timestamp:.6f}
```

这样可以避免多车、多 log 之间的 token 冲突。

### timestamp

timestamp 来自采样帧对应的 ROS message 时间。当前优先使用前视相机 timeline，因此 timestamp 与前视图像帧对齐。

## 8. ego / can_bus 字段

`ego2global_translation` 来自 localization 的全局位置：

```text
[x, y, z]
```

`ego2global_rotation` 当前由 yaw 转 quaternion：

```text
[w, x, y, z]
```

`ego_dynamic_state` 当前写入：

```text
[vx, vy, ax, ay]
```

`can_bus` 当前填充 18 维数组，关键位置是：

```text
0:3   -> x, y, z
3:7   -> quaternion
7     -> vx
8     -> vy
10    -> ax
11    -> ay
15    -> yaw_rate
```

数据来源优先级是：

```text
Localization 优先
ChassisStates 作为 motion fields 的 fallback
```

## 9. 图像导出

相机 payload 当前认为已经是 compressed / nvjpeg 数据，脚本直接写成图片文件。

camera role 到 NAVSIM key 的映射是：

| role | NAVSIM key |
|---|---|
| front | `CAM_F0` |
| left | `CAM_L0` |
| left_back | `CAM_L1` |
| right | `CAM_R0` |
| right_back | `CAM_R1` |
| back | `CAM_B0` |

NAVSIM 里存在但当前没有真实图像的 slot 会保留空 camera 占位，例如 `CAM_L2 / CAM_R2`，这样 dataclass 加载结构更稳定。对于已经发现的多路相机，如果某一路最近帧与前视 timeline 的采样时间差超过 `max_camera_delta=0.05s`，也会保持该路相机为空，避免 10Hz 数据下跨视角错配过大。

## 10. Annotations

障碍物来自：

```text
/perception/fusion/obstacles
```

当前转换为 NAVSIM annotation 字段：

| NAVSIM 字段 | 当前来源 |
|---|---|
| `gt_boxes` | object center / size / yaw |
| `gt_names` | object type 映射到 vehicle / pedestrian / bicycle / generic_object |
| `gt_velocity_3d` | object velocity |
| `track_tokens` | 直接使用 object id 字符串 |
| `instance_tokens` | `{log_name}_{timestamp_us}_{object_id}` |

`gt_names` 的映射目前是初版，后续需要结合 nuPlan / NAVSIM 实际类别进一步校准。

## 11. roadblock_ids

roadblock ids 来自 routing result：

```text
/hadmap_engine/route_result_msg
```

生成规则是：

```text
roadblock_{route_road.pkid}
```

例如：

```text
roadblock_32189
```

读取策略：

```text
routing bag 中可能有多个 RoutingResult
有些 msg 可能为空
脚本会跳过无效 msg，读取第一个包含有效 route_roadlinks.route_road.pkid 的 RoutingResult
如果读不到有效 route，则直接报错，不继续生产 log
```

重复 roadblock 的处理：

```text
如果上一个 path 的最后一个 road 和下一个 path 的第一个 road pkid 相同，则去掉 path 边界处的重复
其他重复暂时保留，因为可能来自真实环线
```

当前环线导致的重复和 index disambiguation 问题先记录为后续风险，不阻断 pkl 生产。

## 12. driving_command

`driving_command` 仍保持 4 维字段，但已经不再严格等价于 one-hot，而是 4 维 command code。

当前编码是：

```text
left     = [1, 0, 0, 0]
straight = [0, 1, 0, 0]
right    = [0, 0, 1, 0]
unknown  = [0, 0, 0, 1]
u-turn   = [1, 1, 0, 0]
```

route link type 到 command 的自动映射采用保守规则：

| LaneType | command |
|---|---|
| `LANE_TYPE_Driving(1)` | straight |
| `LANE_TYPE_JuncStraight(26)` | straight |
| `LANE_TYPE_JuncTurnLeft(25)` | left |
| `LANE_TYPE_JuncTurnRight(27)` | right |
| `LANE_TYPE_Turn(28)` | u-turn |
| 其他类型 | 暂不自动映射 |

特别注意：

```text
LANE_TYPE_UTurnWaiting(23)
LANE_TYPE_LeftOrUturnWaite(34)
LANE_TYPE_TurnLeftWaiting(21)
LANE_TYPE_TurnRightWaiting(24)
```

当前都不自动映射。原因是等待区 / ambiguous 类型语义还需要结合更多 routing 样本确认。

## 13. driving_command 几何逻辑

当前计算思路是：

```text
1. 用 ego 当前全局坐标，在 routing 的 route lanes 中找最近 lane。
2. 如果 ego 到最近 lane 的距离超过 max_lateral_match_distance，则 command = unknown。
3. 如果 ego 更靠近当前 road 前后相关的非空 link，则优先使用 link.type 直接判断 command。
4. 否则计算 ego 投影到 lane centerline 后，到 lane 终点的剩余距离。
5. 如果 30m 内没有遇到有效 turn link，则返回 straight。
6. 如果 30m 内遇到有效 link.type，则按保守映射返回 left / straight / right / u-turn。
```

最近 lane 的距离计算使用点到折线段投影，不是简单点到 centerline 离散点距离，因此不直接依赖中心线点密度。

如果某段 log 和 route 空间关系明显不匹配，脚本会输出 warning，方便后续检查原始数据。

## 14. 当前已确认内容

已经比较明确的部分：

| 项目 | 状态 |
|---|---|
| NAVSIM 输出目录结构 | 已实现自动创建，不清空已有数据 |
| log_name / token 生成逻辑 | 已确认，包含 vehicle_id |
| 前视相机作为 0.1s 采样 timeline | 已确认 |
| 多视角图像提取入口 | 已保留 |
| Localization / ChassisStates 到 ego / can_bus | 已实现 |
| Obstacles 到 annotations | 已实现初版 |
| RoutingResult 到 roadblock_ids | 已实现 |
| route link 到 driving_command | 已实现保守映射 |
| U-turn 4 维 command code | 已保留生产端编码能力 |

## 15. 当前暂时忽略或后续再处理的问题

这些问题目前不阻断 pkl 生产，但后续接入完整训练 / 评估链路时要回头确认：

| 问题 | 当前处理 |
|---|---|
| 下游 prompt/status feature 仍把 command 当 one-hot | 已记录，后续 NAVSIM 数据流中适配 command code |
| `gt_names` 类别映射 | 初版，后续结合 nuPlan 类别细化 |
| 局部地图 msg 到 nuPlan map API | 当前未完整转换，地图对齐后另做 |
| roadblock_ids 与 map_api 的一致性 | 后续接入内部地图时重点确认 |
| 环线 roadblock 重复导致 route index 歧义 | 当前先记录，不阻断生产 |
| 个别 log 与 route 空间关系不匹配 | 脚本打印 warning，数据质量问题先不阻断 |
| 未被下游消费但 schema 需要的字段 | 先用空值或 dummy 占位 |

## 16. 和 NAVSIM 后续链路的接口边界

pkl 生产完成后，后续 NAVSIM 链路预期看到的是：

```text
{dataset_root}/dataset/navsim_logs/{split}/*.pkl
{dataset_root}/dataset/sensor_blobs/{split}/{log_name}/...
{dataset_root}/dataset/maps/...
```

后续还需要：

```text
1. 确保 SceneLoader 能读取新生成的 pkl。
2. 确保 AgentInput 能拿到图像、ego status、driving_command。
3. 如果训练需要 feature cache，需要重新跑 feature cache。
4. 如果 RL / evaluation 需要 PDM score，需要为内部数据生成 metric_cache.pkl。
5. 线上推理通常不需要 metric cache / PDM scorer，只需要模型输入链路。
```

最重要的接口提醒是：

```text
driving_command 已经是 command code，不再能简单假设为严格 one-hot。
```

因此后续在 NAVSIM 数据流中，需要统一适配 prompt 文本转换、status feature 解析，以及可能的 command debug 可视化。

## 17. Mogo SQLite 地图接入：从 pkl 到 map_api

Mogo 地图接入的核心不是在 pkl 里塞入完整地图，而是在构造 `Scene` 时注入一个 `MogoSQLiteMapAdapter`。这个 adapter 先提供 nuPlan 风格的 `map_api`，真正的 SQLite 读取和地图对象构建，则在下游调用 `map_api` 时按需触发。

### 17.1 主流程中的位置

```text
navsim_logs/*.pkl
  -> SceneLoader 读取 scene_dict
  -> cache_mogo_scenarios()
       -> _build_mogo_map_api()，构造 MogoSQLiteMapAdapter
       -> Scene.from_scene_dict_list(..., map_api=mogo_map_api)
       -> MogoNavSimScenario(scene)
  -> MetricCacheProcessor / PDM 调用 scenario.map_api
       -> 触发 adapter.load()
       -> Reader 读取 SQLite
       -> Builder 组织地图对象
  -> metric_cache.pkl
```

也就是说，主流程先完成的是 adapter 的创建和注入；Reader / Builder 的主逻辑不是在 adapter 构造时立刻执行，而是在 PDM 查询地图时由 adapter lazy load。

### 17.2 Reader / Builder / Adapter 的分工

| 组件 | 在流程中的位置 | 职责 |
| --- | --- | --- |
| `MogoSQLiteMapAdapter` | 先被构造并注入 `Scene` / `Scenario` | 对外提供 `get_map_object()`、`get_proximal_map_objects()` 等 nuPlan 风格接口，并负责在需要时触发 `load()`。 |
| `MogoSQLiteReader` | `adapter.load()` 内部执行 | 从 SQLite 读取 road / lane / link / junction 等原始要素，并完成坐标转换。 |
| `MogoSQLiteLayerBuilder` | Reader 之后执行 | 将原始要素组织成 PDM 可消费的 `LANE`、`ROADBLOCK`、`LANE_CONNECTOR`、`ROADBLOCK_CONNECTOR`、`INTERSECTION` 等地图对象。 |

这样的接入方式让原有 NAVSIM 数据流保持干净：`SceneLoader` 仍然只读 pkl，PDM 仍然只通过标准 `scenario.map_api` 消费地图，Mogo SQLite 的细节被集中封装在 adapter 内部。

### 17.3 推荐阅读顺序

```text
planning/script/run_mogo_metric_caching.py
  -> planning/metric_caching/mogo_caching.py
  -> common/dataclasses.py: Scene.from_scene_dict_list(..., map_api=...)
  -> planning/scenario_builder/mogo_navsim_scenario.py
  -> planning/scenario_builder/mogo_sqlite_map_adapter.py
```

其中 `mogo_caching.py` 负责把 adapter 接入主流程，`mogo_sqlite_map_adapter.py` 负责 Reader / Builder / Adapter 的具体实现。

## 18. MogoSQLiteMapAdapter 的职责

`MogoSQLiteMapAdapter` 的目标不是完整复刻 nuPlan map API，而是先实现 PDM / metric cache 实际会消费的地图接口。

PDM 当前主要使用这些能力：

| 消费位置 | 需要的 map_api 能力 | 用途 |
|---|---|---|
| `AbstractPDMPlanner._load_route_dicts()` | `get_map_object(id, ROADBLOCK / ROADBLOCK_CONNECTOR)` | 按 route roadblock ids 加载 route block 和 route lane graph |
| `route_utils.get_current_roadblock_candidates()` | `get_proximal_map_objects()` | 找 ego 当前所在或附近 roadblock / connector |
| `route_utils.get_current_roadblock_candidates()` | `get_distance_to_nearest_map_object()` | 近邻查找失败时 fallback |
| `BreadthFirstSearchRoadBlock` | `_get_roadblock()` / `_get_roadblock_connector()` | roadblock-level BFS route correction |
| `PDMDrivableMap.from_simulation()` | `get_proximal_map_objects()` | 构建 drivable area polygon map |
| `Dijkstra` centerline search | lane / connector 的 `outgoing_edges`、`baseline_path`、`get_roadblock_id()` | 生成中心线 |

因此 adapter 当前优先实现的是：

```text
ROADBLOCK
ROADBLOCK_CONNECTOR
LANE
LANE_CONNECTOR
INTERSECTION
CARPARK_AREA 空占位
BASELINE_PATHS
```

暂时没有实现完整 raster layer，也没有完整交通灯地图层。PDM 当前不依赖这些能力。

## 19. SQLite Reader 阶段

`MogoSQLiteReader` 负责从内部 SQLite 文件读取原始地图表，并转换成 adapter 后续能构造 object layer 的中间数据。

它目前做这些事情：

```text
1. 打开 SQLite 数据库。
2. 检查必要表是否存在，例如 TX_LANE、TX_LANE_LINK、TX_ROAD、TX_JUNCTION。
3. 读取 SQLite geometry blob。
4. 将 geometry 中的 WGS84 经纬度转换到 UTM 坐标。
5. 支持按 route road pkid 读取 route subgraph。
6. 支持按 ego UTM 位置和 radius 读取局部地图。
```

关键配置位于 `MogoSQLiteMapConfig`：

```python
@dataclass(frozen=True)
class MogoSQLiteMapConfig:
    sqlite_path: Path
    map_name: str = "mogo"
    query_radius_m: float = 150.0
    max_segment_length_m: float = 2.0
    enable_route_edge_filter: bool = True
    geometry_srid: int = 4326
    utm_zone: int = 50
    utm_south: bool = False
```

当前内部地图读取以如下假设为基础：

```text
SQLite geometry: WGS84 / SRID 4326
PDM / ego / pkl 中坐标: UTM
当前样例数据 UTM zone: 50
```

所以读取阶段会完成 WGS84 -> UTM 的坐标转换。这样后续 PDM 计算时，ego、障碍物、地图都在同一个 UTM 坐标系下。

## 20. LayerBuilder：从 raw map 到 PDM 可消费对象

`MogoSQLiteLayerBuilder` 负责把 reader 读出的 raw map 变成 PDM 能消费的 map object。

### LANE

每条 lane 会变成 `MogoLaneMapObject`。它提供：

```text
id
polygon
baseline_path
left_boundary
right_boundary
speed_limit_mps
incoming_edges / outgoing_edges
get_roadblock_id()
contains_point()
```

其中 lane id 的规则要特别注意：内部地图中 lane id 从左到右一般是 `-1, -2, ..., -N`。为了让 lane index 更符合 PDM / nuPlan 的正数顺序，当前在构造 lane object 时使用：

```python
index=abs(int(raw_lane.get("lane_id", 0)))
```

也就是说：

```text
lane_id = -1 -> index = 1
lane_id = -2 -> index = 2
```

### ROADBLOCK

一个内部 road 对应一个 `ROADBLOCK`。

`ROADBLOCK.interior_edges` 只包含真实 lane，不包含非空 link。原因是 PDM 原逻辑里 roadblock 的 interior edges 默认是 lane-like edge，route roadblock 顺序也依赖这个结构。

ROADBLOCK polygon 的构造优先级是：

```text
1. 使用最左 lane 的左边界 + 最右 lane 的右边界拼成 polygon。
2. 如果失败，fallback 到 lane polygon union。
3. 如果仍失败，fallback 到 road centerline buffer，并打印 warning。
```

### LANE_CONNECTOR

非空 link 会被构造成 `LANE_CONNECTOR`，也就是 lane-level graph 中的 connector edge。

当前只为 routing 上的 road 之间的非空 link 生成 connector：

```text
from_road_pkid in route_road_pkids
and
to_road_pkid in route_road_pkids
```

这样做的出发点是：

```text
第一版不希望规划 graph 脱离 routing
减少不必要分支
提升 centerline / proposal 搜索的稳定性
```

空 link 不生成 connector object，只作为拓扑连接：

```text
from_lane.outgoing_edges += to_lane
to_lane.incoming_edges += from_lane
```

### ROADBLOCK_CONNECTOR

非空 link 会进一步按前后 roadblock 组合成 `ROADBLOCK_CONNECTOR`，例如：

```text
roadblock_connector_roadblock_59801_roadblock_59804
```

这个 block 的 `interior_edges` 是对应的 `LANE_CONNECTOR` 列表，polygon 是这些 connector polygon 的 union。

它有两个主要用途：

```text
1. 给 PDMDrivableMap 提供路口 / connector 区域的 drivable polygon。
2. 给 route correction / BFS 提供 roadblock-level connector 节点。
```

### INTERSECTION

`TX_JUNCTION` 会被构造成 `INTERSECTION` object。它主要用于 drivable area 的补充覆盖，特别是 connector polygon 不够覆盖整个路口可行驶区域时。

当前也把 junction 里的 road 绑定信息保留到 metadata 中：

```text
road_pkids
enter_road_pkids
inside_road_pkids
exit_road_pkids
```

后续如果需要用 junction 和 road/link 的绑定关系，可以继续从这里扩展。

## 21. Connector polygon 的当前策略

非空 link 的 polygon 采用一个简单、稳定优先的策略生成：

```text
以 link centerline 为中心
结合前继 lane / 后继 lane 的左右边界估计宽度
再生成 connector polygon
```

普通 link：

```text
left_width  = max_half_width + 1.0
right_width = max_half_width + 1.0
```

U-turn link 做特殊处理：

```text
left_width  = min_half_width + 0.5
right_width = max_half_width + 1.0
```

这样做是为了避免 U-turn 曲率较大时，左侧 polygon 过宽导致明显扭曲。

这个 polygon 的质量目标不是完美精细建图，而是满足 PDM 第一版的实际消费：

```text
能在 drivable area 中覆盖主要 connector 区域
能让 ego 在 link 上时被 map_api 查询命中
能让 lane graph 在路口处连通
```

如果后续发现 connector polygon 对 drivable area 过窄，可以再结合 `INTERSECTION` polygon 做 fallback 或扩展。

## 22. PDM route lane graph 适配

为了让 PDM centerline search 能穿过内部地图的非空 link，`AbstractPDMPlanner._load_route_dicts()` 做了一个小扩展。

原逻辑是：

```text
route roadblock ids
  -> load ROADBLOCK
  -> collect ROADBLOCK.interior_edges into _route_lane_dict
```

但现在非空 link 不在 `ROADBLOCK.interior_edges` 里，而是独立的 `LANE_CONNECTOR`。因此新增了：

```python
def _load_route_connector_edges(self) -> None:
    ...
```

它会从 route lanes 的 `outgoing_edges` 中，把连接 route roadblock 的 connector edge 补进 `_route_lane_dict`。这样 Dijkstra 的候选边集合就包含：

```text
lane
lane_connector
lane
```

但 route roadblock 的主顺序不变，仍然只由原始 route roadblock ids 决定。

这点很重要：

```text
connector 参与 lane-level graph search
但不破坏 roadblock-level route sequence
```

## 23. Ego 位于 connector 上时的 centerline 修复

真实测试中发现一个关键边界：当 ego 正好在非空 link 上时，PDM 的 `_get_starting_lane()` 可能选中 `LANE_CONNECTOR` 作为 current lane。

这时：

```text
current_lane.get_roadblock_id()
= roadblock_connector_roadblock_59801_roadblock_59804
```

它并不在原始 route roadblock ids 里。如果继续使用旧逻辑：

```python
start_idx = np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
```

会有两个问题：

```text
1. 找不到 connector 对应的 roadblock index。
2. np.argmax 在全 False 时会返回 0，可能 silent 生成错误 centerline。
```

因此新增了：

```python
def _get_current_roadblock_index(self, current_lane, roadblock_ids) -> int:
    ...
```

解析策略是：

```text
1. 如果 current lane 自己的 roadblock id 在 route 中，直接使用。
2. 如果 current lane 是 connector，优先从 outgoing_edges 找下游 route roadblock。
3. 如果还找不到，再从 incoming_edges 找上游 route roadblock。
4. 都找不到时，显式抛出 ValueError，不再 silent fallback 到 0。
```

真实 log `20260319153048` 的 smoke test 中，ego 位于：

```text
link_85598
roadblock_connector_roadblock_59801_roadblock_59804
```

修正后可以解析到下游：

```text
roadblock_59804
```

并从 connector 本身作为 Dijkstra 起点继续生成 centerline。

## 24. Centerline 拼接去重

centerline 最终由 route plan 中每条 lane / connector 的 `baseline_path.discrete_path` 拼接得到：

```python
for lane in route_plan:
    for state in lane.baseline_path.discrete_path:
        ...
```

由于相邻 baseline 的首尾点可能重合，如果直接 `extend()`，会产生连续重复点。后续几何投影、progress、曲率等计算都可能被重复点干扰。

当前修复是在拼接阶段做相邻去重：

```python
if centerline_discrete_path and np.allclose(centerline_discrete_path[-1].array, state.array):
    continue
centerline_discrete_path.append(state)
```

这个去重只处理相邻重复点，不改变每条 baseline 内部的点序列，也不影响非相邻位置可能出现的真实重复路径。

## 25. MogoNavSimScenario：避免原始 nuPlan map_api assert

原始 `NavSimScenario.map_api` 会通过：

```python
get_maps_api(self._map_root, self._map_version, self._map_name)
```

加载 nuPlan 地图。如果内部数据的 `map_name` 不是 nuPlan 支持的 map location，就会被 `MAP_LOCATIONS` assert 卡住。

为此新增：

```text
planning/scenario_builder/mogo_navsim_scenario.py
```

它继承 `NavSimScenario`，复用原有的 ego、obstacle、trajectory、route ids 等逻辑，只 override：

```python
@property
def map_api(self) -> AbstractMap:
    return self._mogo_map_api
```

也就是说：

```text
Scenario 行为仍然像 NAVSIM Scenario
但地图来源换成 MogoSQLiteMapAdapter
```

同时，`Scene.from_scene_dict_list()` 新增了一个向后兼容参数：

```python
map_api: Optional[AbstractMap] = None
```

如果传入 `map_api`，就用传入的；否则保持原来的 `cls._build_map_api(scene_metadata.map_name)` 行为。

## 26. Mogo metric cache 生成入口

新增 Mogo 专用 cache 入口：

```text
planning/script/run_mogo_metric_caching.py
planning/metric_caching/mogo_caching.py
planning/script/config/metric_caching/default_mogo_metric_caching.yaml
```

它基本复用原有 metric cache 的结构，只在构造 scenario 时替换成 Mogo 版本。

核心代码逻辑是：

```python
route_roadblock_ids = scene_dict[current_frame_idx]["roadblock_ids"]
map_api = _build_mogo_map_api(cfg, route_roadblock_ids)
scene = Scene.from_scene_dict_list(..., map_api=map_api)
scenario = MogoNavSimScenario(scene)
processor.compute_metric_cache(scenario)
```

这样 `MetricCacheProcessor` 本身不需要区分 nuPlan map 还是 Mogo map。它看到的仍然是一个 `AbstractScenario`：

```text
scenario.initial_ego_state
scenario.initial_tracked_objects
scenario.get_route_roadblock_ids()
scenario.map_api
scenario.get_tracked_objects_at_iteration()
scenario.get_ego_future_trajectory()
```

最终仍然输出同一个 `MetricCache` dataclass：

```text
trajectory
ego_state
observation
centerline
route_lane_ids
drivable_area_map
history_states
```

## 27. Mogo metric cache 推荐执行方式

Mogo metric cache 的入口是：

```text
planning/script/run_mogo_metric_caching.py
```

对应默认配置是：

```text
planning/script/config/metric_caching/default_mogo_metric_caching.yaml
```

建议在 `/home/zhongwei/Vla/src/recog_drive` 目录下执行，保证 `navsim` 包可以被正常 import：

```bash
cd /home/zhongwei/Vla/src/recog_drive
```

### 27.1 环境变量

运行前至少需要设置两个根目录：

```bash
export OPENSCENE_DATA_ROOT=/path/to/output_root/dataset
export NAVSIM_EXP_ROOT=/path/to/output_root/exp
```

其中 `OPENSCENE_DATA_ROOT` 下应包含：

```text
${OPENSCENE_DATA_ROOT}/navsim_logs/trainval/*.pkl
${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval/...
${OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite
```

`NAVSIM_EXP_ROOT` 用来存放 metric cache 输出。Mogo 流程沿用原始 NAVSIM 的子目录名 `metric_cache`，这样后续训练、RL 和 PDM scoring 可以复用默认路径约定；Mogo / nuPlan 数据集之间通过不同的 `NAVSIM_EXP_ROOT` 区分。

```text
${NAVSIM_EXP_ROOT}/metric_cache/...
```

### 27.2 默认配置含义

`default_mogo_metric_caching.yaml` 中的关键配置是：

```yaml
cache:
  cache_path: ${oc.env:NAVSIM_EXP_ROOT}/metric_cache
  use_cache_without_dataset: false
  force_feature_computation: false

output_dir: ${cache.cache_path}/metadata
navsim_log_path: ${oc.env:OPENSCENE_DATA_ROOT}/navsim_logs/${train_test_split.data_split}

mogo_map:
  sqlite_path: ${oc.env:OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite
  map_name: mogo
  query_radius_m: 150.0
  max_segment_length_m: 2.0
  enable_route_edge_filter: true
  geometry_srid: 4326
  utm_zone: 50
  utm_south: false
```

这些字段的含义是：

| 配置 | 含义 |
|---|---|
| `cache.cache_path` | 生成的 `metric_cache.pkl` 根目录 |
| `cache.force_feature_computation` | 如果 cache 已存在，是否强制重新计算 |
| `navsim_log_path` | 输入 `.pkl` 所在目录，默认由 `OPENSCENE_DATA_ROOT` 和 split 拼出来 |
| `train_test_split` | 使用哪个数据 split 和 scene_filter，例如 `trainval` / `test` |
| `mogo_map.sqlite_path` | 内部 SQLite 地图文件路径 |
| `mogo_map.query_radius_m` | 按 ego 查询局部地图时的半径 |
| `mogo_map.enable_route_edge_filter` | 是否只保留 routing 相关 connector 分支 |
| `mogo_map.geometry_srid` | SQLite geometry 的原始坐标 SRID，当前按 WGS84 / 4326 |
| `mogo_map.utm_zone` | 转 UTM 时使用的 zone，当前样例为 50 |
| `mogo_map.utm_south` | 是否南半球 UTM，当前为 false |

### 27.3 批量生成命令

默认 trainval 批量生成：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  worker=sequential \
  cache.force_feature_computation=true
```

如果要显式指定 split：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  train_test_split=trainval \
  worker=sequential \
  cache.force_feature_computation=true
```

测试集可以改成：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  train_test_split=test \
  worker=sequential \
  cache.force_feature_computation=true
```

### 27.4 单 log / 小样本 smoke test

第一次跑建议只跑一个或少量 scene，避免完整数据集慢慢炸。

限制最多一个 scene：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  worker=sequential \
  train_test_split.scene_filter.max_scenes=1 \
  cache.force_feature_computation=true
```

如果要指定某个 log，可以覆盖 `log_names`。注意 shell 里中括号最好加引号：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  worker=sequential \
  'train_test_split.scene_filter.log_names=[JLBJA20073D_20260319153048]' \
  train_test_split.scene_filter.max_scenes=1 \
  cache.force_feature_computation=true
```

如果要指定某个 token，可以覆盖 `tokens`：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  worker=sequential \
  'train_test_split.scene_filter.tokens=[your_token]' \
  cache.force_feature_computation=true
```

### 27.5 常用覆盖参数

如果 SQLite 不在默认位置：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  worker=sequential \
  cache.force_feature_computation=true \
  mogo_map.sqlite_path=/path/to/hm_beijing_3.4.9_20260521.sqlite
```

如果 `.pkl` 不在默认 `OPENSCENE_DATA_ROOT/navsim_logs/{split}` 下，也可以直接覆盖：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  worker=sequential \
  navsim_log_path=/path/to/navsim_logs/trainval \
  cache.force_feature_computation=true
```

如果想把 cache 输出到临时目录：

```bash
python navsim/planning/script/run_mogo_metric_caching.py \
  worker=sequential \
  cache.cache_path=/tmp/metric_cache_smoke \
  cache.force_feature_computation=true
```

### 27.6 输出结果

默认输出是：

```text
${NAVSIM_EXP_ROOT}/metric_cache/{log_name}/unknown/{token}/metric_cache.pkl
${NAVSIM_EXP_ROOT}/metric_cache/metadata/*.csv
```

`metadata/*.csv` 会记录所有成功生成的 cache 文件路径，后续 `MetricCacheLoader` 就是通过这个 metadata 找到每个 token 对应的 `metric_cache.pkl`。

## 28. 真实 log smoke test 结论

使用真实 log：

```text
data/2026-03-19-15-24-05/20260319153048
```

它是一个比较适合的过路口样本：

```text
unknown command ratio = 0
route lane 匹配稳定
存在非空 link 命中
```

测试中选中的路口帧：

```text
sample_idx_10hz = 214
ego = (460579.612, 4439347.747)
nearest lane = road_pkid 59801, lane_id -2, distance 0.326m
nearest link = 59801 -> 59804, link_type 26, distance 0.213m
```

map adapter 加载结果：

```text
route roadblocks = 58
route lanes + connectors = 198
connector edges added = 37
missing route blocks = 0
```

connector-aware centerline 验证结果：

```text
start = link_85598
resolved downstream roadblock = roadblock_59804
target = roadblock_226100
Dijkstra path_found = True
edges = 30
states = 307
length ≈ 833.4m
```

说明当前地图对象、route graph、connector 起点解析、centerline 生成这一段逻辑已经能在真实过路口样本上闭环。

## 29. 当前运行环境问题

当前默认环境是：

```text
Python 3.8.10
```

但本机安装的 nuPlan 相关包里存在 Python 3.9+ 类型写法，例如：

```python
list[StateSE2]
```

这会导致完整导入 PDMClosedPlanner / MetricCacheProcessor 时出现：

```text
TypeError: 'type' object is not subscriptable
```

此外，当前环境还缺少：

```text
geopandas
```

所以目前能完成的是：

```text
SQLite reader / adapter 局部测试
route graph / centerline 局部 smoke test
py_compile 语法检查
```

但完整 `run_mogo_metric_caching.py` 需要在 Python 3.9+，更推荐 Python 3.10，并补齐 nuPlan 地图相关依赖后再运行。

建议环境路径：

```text
1. 新建 Python 3.10 环境。
2. 安装项目依赖和 nuPlan / geopandas / shapely / pyproj 等依赖。
3. 先跑 py_compile。
4. 再跑单 log / max_scenes=1 的 Mogo metric cache smoke test。
5. 最后批量生成 metric cache。
```

## 30. Mogo 数据流适配剩余事项

当前已经完成：

| 项目 | 状态 |
|---|---|
| ROS bag -> NAVSIM-style pkl | 已实现 |
| 多路相机导出 | 已实现初版 |
| obstacles -> annotations | 已实现初版 |
| route -> roadblock_ids / driving_command | 已实现初版 |
| Mogo SQLite reader | 已实现第一版 |
| Mogo map object adapter | 已实现第一版 |
| PDM route graph 支持 connector | 已实现 |
| connector 起点 centerline 修复 | 已实现 |
| centerline 拼接相邻重复点去重 | 已实现 |
| Mogo metric cache 专用入口 | 已实现 |

还需要继续做：

| 项目 | 为什么需要 |
|---|---|
| 在 Python 3.9+ 环境跑完整 metric cache | 当前环境无法完整导入 nuPlan runtime |
| 单 log `run_mogo_metric_caching.py` smoke test | 验证 cache 文件真实生成 |
| 用生成的 metric cache 跑 PDM score | 验证 scorer 下游消费完整性 |
| 检查 drivable area 覆盖 | 确认 ROADBLOCK / ROADBLOCK_CONNECTOR / INTERSECTION polygon 是否足够 |
| 检查 centerline 质量 | 尤其是路口、掉头、变道前后的 centerline 是否符合预期 |
| 检查 route correction | ego 在 road、connector、junction 边界时是否稳定 |
| 复核 traffic light | 当前 traffic_lights 为空，后续如需要红绿灯约束需补充 |
| 复核 command code 下游解析 | 生产端已有 u-turn code，但 prompt/status feature 需要同步适配 |
| 扩展地图 placeholder | 例如 `get_curvature_at_arc_length()`、CARPARK_AREA、raster layer 等按需补齐 |

整体上，目前链路已经从：

```text
rosbag -> pkl
```

推进到：

```text
rosbag -> pkl -> Mogo map_api -> PDM route graph / centerline -> Mogo metric cache 入口
```

下一步真正的验收点就是：在正确环境里跑通 `run_mogo_metric_caching.py`，并拿生成的 `metric_cache.pkl` 跑一次 PDM score。

