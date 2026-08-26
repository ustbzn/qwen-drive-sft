# MOGO 多批次与多城市数据使用方案

## 结论

当前 `0312_0319` 与 `0320-0325` 都是北京数据，并且地图文件均为：

```text
hm_beijing_3.4.9_20260521.sqlite
```

因此这两批数据可以按“同城同地图”处理：构建一个只读的联合数据根，或分别缓存后在 cache 层合并。考虑后续还会加入不同城市、不同高精地图、不同 UTM zone，推荐长期采用：

1. 原始数据按批次/城市保持独立，不做物理复制合并。
2. 缓存阶段按 `city + map_version + batch` 独立运行，确保每次只使用对应的 `mogo_map.sqlite_path` 和 `mogo_map.utm_zone`。
3. 训练阶段使用 `use_cache_without_dataset=true`，只合并已经计算好的 cache 目录。

这个方案对当前代码改动最小，也能避免跨城市时用错地图或投影参数。

## 当前代码约束

相关入口与约束如下：

- `scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh`
  - 默认 `DATA_ROOT=/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319`
  - 通过 `OPENSCENE_DATA_ROOT=${DATA_ROOT}/dataset` 指向 logs 和 sensor blobs。
  - 通过 `SQLITE_PATH=${OPENSCENE_DATA_ROOT}/maps/hm_beijing_3.4.9_20260521.sqlite` 指向高精地图。
  - 通过 `MOGO_UTM_ZONE=50` 配置投影。

- `navsim/planning/script/run_dataset_caching_multi_node.py`
  - 一次只构建一个 `SceneLoader`。
  - 一次只从 Hydra config 构建一个 `map_api_builder`。

- `navsim/planning/scenario_builder/mogo_map_api_builder.py`
  - `mogo_map.sqlite_path`、`mogo_map.utm_zone` 会被固化到同一个 `MogoSQLiteMapConfig`。
  - 该配置对当前 `SceneLoader` 里的所有 scene 生效。

- `navsim/planning/training/dataset.py`
  - `Dataset` 缓存路径结构为 `cache_path/log_name/token/*.gz`。
  - `CacheOnlyDataset` 只读取 cache，不再依赖原始 pkl、图片路径和 map api。
  - `CacheOnlyDataset` 用 token 作为样本 key，如果不同数据源存在 token 冲突，后读到的 cache 会覆盖先读到的 token。

这意味着：跨城市或跨地图时，不应该把原始 pkl 混到同一个 `SceneLoader` 中再只传一个 sqlite。否则 HDMap 特征和 metric/map 查询会使用错误地图。

## 当前两批北京数据的推荐做法

### 方案 A：分别缓存，训练时合并 cache

这是推荐方案，也最接近后续多城市流程。

分别为两批数据生成 cache：

```bash
DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319" \
CACHE_PATH="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0312_0319_qwen3_obj_hdmap" \
SQLITE_PATH="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset/maps/hm_beijing_3.4.9_20260521.sqlite" \
MOGO_UTM_ZONE="50" \
bash "scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh"
```

```bash
DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0320-0325" \
CACHE_PATH="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0320_0325_qwen3_obj_hdmap" \
SQLITE_PATH="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0320-0325/dataset/maps/hm_beijing_3.4.9_20260521.sqlite" \
MOGO_UTM_ZONE="50" \
bash "scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh"
```

然后建立一个联合 cache root，下面只放各 batch 的 log 目录软链接：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_all_qwen3_obj_hdmap/
  <log_name_from_0312_0319> -> ../beijing_0312_0319_qwen3_obj_hdmap/<log_name>
  <log_name_from_0320_0325> -> ../beijing_0320_0325_qwen3_obj_hdmap/<log_name>
```

训练时使用：

```bash
cache_path="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_all_qwen3_obj_hdmap" \
use_cache_without_dataset=true
```

优点：

- 原始数据不复制，cache 可追溯到具体 batch。
- 后续增加城市时复用同样流程。
- 训练阶段不需要 map api，避免跨城市地图配置问题。

需要额外做两项检查：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0312_0319_qwen3_obj_hdmap" -mindepth 1 -maxdepth 1 -type d | sort | wc -l
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0320_0325_qwen3_obj_hdmap" -mindepth 1 -maxdepth 1 -type d | sort | wc -l
```

如果 log 名称有冲突，不要直接软链接到同一层；应在生成 cache 前给 log 名称加 batch 前缀，或者扩展 `CacheOnlyDataset` 支持多 cache root。

### 方案 B：同城同地图下建立联合数据根

这只适用于当前两批北京数据，因为它们使用同一个 sqlite 和 UTM zone。

可以建立一个新的只读数据根，例如：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/beijing_0312_0325_union/dataset/
  maps/
  navsim_logs/train/
  navsim_logs/val/
  navsim_logs/trainval/
  sensor_blobs/train/
  sensor_blobs/val/
  sensor_blobs/trainval/
```

其中 `maps` 可以软链接北京 sqlite，`navsim_logs` 和 `sensor_blobs` 下的文件也使用软链接汇总。缓存时只需要：

```bash
DATA_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/beijing_0312_0325_union" \
SQLITE_PATH="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/beijing_0312_0325_union/dataset/maps/hm_beijing_3.4.9_20260521.sqlite" \
MOGO_UTM_ZONE="50" \
CACHE_PATH="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0312_0325_union_qwen3_obj_hdmap" \
bash "scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh"
```

这个方案的缺点是容易被误用于多城市。只要出现不同城市、不同 sqlite、不同 UTM zone，就必须切回方案 A 或做代码改造。

## 后续多城市数据方案

### 数据组织

建议统一保留如下结构：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/
  0312_0319/
  0320-0325/
  worldsim/shanghai/
  <city>/<batch>/
```

每个数据根都独立包含：

```text
dataset/maps/*.sqlite
dataset/navsim_logs/train/*.pkl
dataset/navsim_logs/val/*.pkl
dataset/navsim_logs/trainval/*.pkl
dataset/sensor_blobs/...
```

不要把不同城市的 `navsim_logs/trainval` 直接混在同一个原始目录里，除非代码已支持 per-scene map 配置。

### Manifest

新增一个人工维护或脚本生成的 manifest，记录每个数据源的地图参数：

```yaml
datasets:
  - name: beijing_0312_0319
    city: beijing
    data_root: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319
    sqlite_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset/maps/hm_beijing_3.4.9_20260521.sqlite
    utm_zone: 50
    utm_south: false
    cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0312_0319_qwen3_obj_hdmap

  - name: beijing_0320_0325
    city: beijing
    data_root: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0320-0325
    sqlite_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0320-0325/dataset/maps/hm_beijing_3.4.9_20260521.sqlite
    utm_zone: 50
    utm_south: false
    cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0320_0325_qwen3_obj_hdmap

  - name: shanghai_worldsim
    city: shanghai
    data_root: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai
    sqlite_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/maps/hm_shanghai_1.9.1_20260522.sqlite
    utm_zone: 51
    utm_south: false
    cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/shanghai_worldsim_qwen3_obj_hdmap
```

缓存脚本读取 manifest 后逐项调用现有 caching 脚本即可。这样遵循 KISS/YAGNI：先不改 `SceneLoader` 和 map adapter，只把易错参数集中管理。

### 训练合并

训练优先合并 cache，而不是合并原始数据。可以使用两种方式：

1. 软链接合并：生成一个 `cache/all_cities_qwen3_obj_hdmap`，内部展开所有 log 目录。
2. 小规模代码改造：新增 `MultiCacheOnlyDataset`，接受多个 cache root，内部返回统一样本列表。

如果预计 log 名称或 token 可能跨城市重复，推荐直接做 `MultiCacheOnlyDataset`，不要用单层软链接。原因是当前 `CacheOnlyDataset` 使用 token 作为 key，冲突会覆盖样本。

推荐接口：

```python
class MultiCacheOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, cache_paths, feature_builders, target_builders, log_names=None):
        ...
```

内部样本 key 应使用 `(cache_path, log_name, token)`，而不是只使用 `token`。

## 不推荐的做法

不要把不同城市的 pkl 直接放到同一个 `OPENSCENE_DATA_ROOT/navsim_logs/trainval`，然后只传一个：

```bash
mogo_map.sqlite_path="某一个城市.sqlite"
mogo_map.utm_zone="某一个城市 zone"
```

这会导致：

- HDMap vector 特征使用错误地图。
- route roadblock ids 在错误 sqlite 中查不到或查到错误对象。
- UTM 投影错误，空间位置偏移。
- metric cache 和训练 cache 难以追溯。

也不建议复制整份 sensor blobs 到一个大目录。数据体量大，且复制会破坏 batch/city 的来源边界。

## 推荐落地顺序

1. 当前北京数据先按方案 A 分别缓存 `0312_0319` 与 `0320-0325`。
2. 用软链接生成 `beijing_all_qwen3_obj_hdmap` 联合 cache。
3. 训练改为 `use_cache_without_dataset=true`，只读取联合 cache。
4. 新城市到来时，先在 manifest 中登记 `data_root/sqlite_path/utm_zone/cache_path`，再单独缓存。
5. 如果出现 log/token 冲突，新增 `MultiCacheOnlyDataset`，不要继续扩大软链接方案。

## 质量检查清单

缓存前检查：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0320-0325/dataset/maps" -maxdepth 1 -type f | sort
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0320-0325/dataset/navsim_logs" -maxdepth 2 -type f -name "*.pkl" | sort | wc -l
```

缓存后检查：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_0320_0325_qwen3_obj_hdmap" -mindepth 2 -maxdepth 2 -type d | sort | wc -l
```

训练前检查：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/cache/beijing_all_qwen3_obj_hdmap" -mindepth 1 -maxdepth 1 | sort | wc -l
```

如果使用 train/val split，需要为联合数据重新生成或合并 log split YAML。当前脚本 `scripts/cache_dataset/gen_mogo_log_split.py` 只能从一个 `logs-root` 生成 split；多数据源时建议扩展为读取 manifest 后输出一个全量：

```text
navsim/planning/script/config/training/log_split/mogo_beijing_0312_0325.yaml
```

## 原则说明

- KISS：当前不改核心 `SceneLoader`/map adapter，先用独立缓存和 cache 合并解决问题。
- YAGNI：只有在 token/log 冲突或必须 on-the-fly 多城市训练时，才引入 `MultiCacheOnlyDataset` 或 per-scene map builder。
- DRY：用 manifest 集中维护 `data_root/sqlite_path/utm_zone/cache_path`，避免脚本里散落硬编码路径。
- SOLID：地图配置、数据源登记、cache 读取各自独立；训练阶段依赖已缓存特征，不依赖具体城市地图实现。

## 已实现多城市方案全链路使用手册

本节对应当前代码中已经落地的多城市方案，适用于 `/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim` 下的 `beijing`、`shanghai`，也适用于后续继续增加城市或批次。

核心原则不变：

1. 原始数据不跨城市混放。
2. 每个城市独立使用自己的 `sqlite_path`、`utm_zone`、`utm_south` 生成 feature cache 和 metric cache。
3. 训练阶段只合并 cache root，不再依赖原始 pkl、sensor blobs 或 map api。
4. 评测阶段按城市分别跑，因为 PDM 评测仍然需要 SceneLoader、metric cache、地图和 UTM 参数；跑完后再合并 CSV。

### 已新增或改造的组件

#### 数据源 manifest

当前默认 manifest：

```text
navsim/planning/script/config/common/mogo_dataset/worldsim.yaml
```

当前内容按如下语义维护：

```yaml
datasets:
  - name: beijing_worldsim
    city: beijing
    data_root: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing
    sqlite_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/dataset/maps/hm_beijing_3.4.9_20260521.sqlite
    utm_zone: 50
    utm_south: false
    cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/recogdrive_qwen3_notrain_obj_hdmap_cache
    metric_cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/metric_cache_mogo_v3

  - name: shanghai_worldsim
    city: shanghai
    data_root: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai
    sqlite_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/maps/hm_shanghai_1.9.1_20260522.sqlite
    utm_zone: 51
    utm_south: false
    cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/recogdrive_qwen3_notrain_obj_hdmap_cache
    metric_cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/metric_cache_mogo_v3
```

字段说明：

- `name`：数据源唯一名，用于 `--name` 过滤、评测实验名后缀、日志输出。必须全局唯一。
- `city`：城市名，仅用于标注和合并 CSV。
- `data_root`：单个城市/批次数据根，下面必须包含 `dataset/maps`、`dataset/navsim_logs`、`dataset/sensor_blobs`。
- `sqlite_path`：当前数据源对应的高精地图 sqlite。不要跨城市复用。
- `utm_zone`：当前地图使用的 UTM zone。北京当前为 `50`，上海当前为 `51`。
- `utm_south`：南半球投影标记。当前中国城市为 `false`。
- `cache_path`：训练用 feature cache 目标目录。
- `metric_cache_path`：PDM 评测和 RL reward 用 metric cache 目标目录。

新增城市时，只需要在 `datasets` 下追加一项，不需要改训练和评测脚本。

#### 多 cache root 训练读取

已新增：

```text
navsim/planning/training/dataset.py
```

关键能力：

- `CacheOnlyDataset` 保持单 cache root 旧行为。
- `MultiCacheOnlyDataset` 支持多个 cache root。
- `build_cache_only_dataset(...)` 根据 `cache_paths` 数量自动选择单 root 或多 root。
- 多 root 样本 key 使用 `(cache_path, log_name, token)`，避免不同城市/批次 token 相同时互相覆盖。

训练配置已新增：

```yaml
cache_paths: null
```

支持两种写法：

```bash
cache_path=/single/cache/root
```

或：

```bash
'cache_paths=[/cache/root/a,/cache/root/b]'
```

当 `cache_paths` 非空时，训练会优先使用 `cache_paths`，`cache_path` 只作为兼容字段保留。

#### 多 metric cache root 读取

已改造：

```text
navsim/common/dataloader.py
```

关键能力：

- `MetricCacheLoader` 支持单个 metric cache root。
- `MetricCacheLoader` 支持多个 metric cache root。
- 多 root 可以传 Python list、Hydra list，或 `os.pathsep` 分隔的字符串。在 Linux 下 `os.pathsep` 是冒号 `:`。
- 如果多个 metric cache root 中存在相同 token，会直接报错。原因是 GRPO/RL reward 当前是 token-only 查表，不能安全区分同名 token 属于哪个城市。

#### Manifest 驱动 feature cache 脚本

新增：

```text
scripts/cache_dataset/run_mogo_manifest_feature_caching.py
```

用途：读取 manifest，逐个数据源调用现有：

```text
scripts/cache_dataset/run_caching_recogdrive_qwen3_hidden_state.sh
```

它负责为每个城市设置：

```text
DATA_ROOT
CACHE_PATH
SQLITE_PATH
MOGO_UTM_ZONE
MOGO_UTM_SOUTH
OPENSCENE_DATA_ROOT
NUPLAN_MAPS_ROOT
NAVSIM_EXP_ROOT
TRAIN_TEST_SPLIT
MASTER_PORT
```

因此不会出现“上海数据用了北京 sqlite”或“北京数据用了上海 UTM zone”的问题。

#### Manifest 驱动 metric cache 脚本

新增：

```text
scripts/cache_dataset/run_mogo_manifest_metric_caching.py
```

用途：读取 manifest，逐个数据源调用：

```text
navsim/planning/script/run_mogo_metric_caching.py
```

它负责为每个城市设置：

```text
OPENSCENE_DATA_ROOT
NUPLAN_MAPS_ROOT
NAVSIM_EXP_ROOT
cache.cache_path
mogo_map.sqlite_path
mogo_map.utm_zone
mogo_map.utm_south
```

metric cache 用于：

- PDM score 评测。
- GRPO/RL 训练 reward 计算。

#### 多数据源 log split 生成脚本

新增：

```text
scripts/cache_dataset/gen_mogo_multi_log_split.py
```

默认输出：

```text
navsim/planning/script/config/training/log_split/mogo_worldsim.yaml
```

当前默认行为是按数据源分层切分：

- 每个数据源内部从 `dataset/navsim_logs/train` 中按 `val_ratio` 切出一部分到 `val_logs`。
- 所有数据源的 `train_logs` 合并成全局 `train_logs`。
- 所有数据源的 `val_logs` 合并成全局 `val_logs`。
- 这样可以避免小数据集时验证集只落在一个城市。

如果确实想把所有城市日志先合在一起再全局随机切分，可以加：

```bash
--global-split
```

#### Manifest 驱动评测脚本

新增：

```text
scripts/evaluation/run_mogo_manifest_evaluation.py
```

用途：读取 manifest，逐个数据源调用：

```text
scripts/evaluation/worldsim_pdms_evaluation_2b.sh
```

每个城市评测时使用自己的：

```text
DATA_ROOT
METRIC_CACHE_PATH
SQLITE_PATH
MOGO_UTM_ZONE
MOGO_UTM_SOUTH
```

所有城市跑完后，脚本会把各城市 CSV 合并成一个总 CSV，并重新计算去掉各城市 `average` 行后的全局 `average` 行。

默认合并输出位置：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/merged_eval/<experiment_name>_<timestamp>.csv
```

## 全链路最短路径

下面是从原始 worldsim 多城市数据到训练、RL、评测的推荐顺序。

### 0. 确认原始数据仍按城市独立

先确认 worldsim 结构：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim" -maxdepth 3 -type d | sort
```

期望至少存在：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/dataset/maps
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/dataset/navsim_logs
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/dataset/sensor_blobs
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/maps
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/navsim_logs
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/sensor_blobs
```

检查地图：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim" -path "*/dataset/maps/*.sqlite" -type f | sort
```

当前应包含：

```text
.../worldsim/beijing/dataset/maps/hm_beijing_3.4.9_20260521.sqlite
.../worldsim/shanghai/dataset/maps/hm_shanghai_1.9.1_20260522.sqlite
```

检查 logs：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/dataset/navsim_logs" -maxdepth 2 -type f -name "*.pkl" | sort | wc -l
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/dataset/navsim_logs" -maxdepth 2 -type f -name "*.pkl" | sort | wc -l
```

不要把北京和上海的 pkl 复制到同一个 `navsim_logs/trainval` 下再跑一个 `SceneLoader`。

### 1. 检查 manifest

查看默认 manifest：

```bash
sed -n '1,200p' "navsim/planning/script/config/common/mogo_dataset/worldsim.yaml"
```

确认每个数据源的路径和地图参数：

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" --dry-run --gpus 1
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" --dry-run
python "scripts/evaluation/run_mogo_manifest_evaluation.py" --dry-run --gpus 1
```

这些 dry-run 只打印将要执行的命令，不会启动 `torchrun` 或真实缓存。

### 2. 生成训练用 feature cache

全量生成北京和上海 feature cache：

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" \
  --gpus 8 \
  --qwen3-vl-path "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct"
```

只生成单个城市：

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" \
  --name beijing_worldsim \
  --gpus 8
```

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" \
  --name shanghai_worldsim \
  --gpus 8
```

传递额外 Hydra override：

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" \
  --name shanghai_worldsim \
  --gpus 8 \
  agent.use_image_encoder=true \
  agent.image_height=256 \
  agent.image_width=704
```

常用参数：

- `--manifest`：指定其他 manifest。
- `--name`：只跑某个数据源，可重复传多次。
- `--cache-script`：替换底层 cache shell，默认是 `run_caching_recogdrive_qwen3_hidden_state.sh`。
- `--train-test-split`：默认 `mogo_trainval`。
- `--qwen3-vl-path`：覆盖 Qwen3-VL 模型路径。
- `--gpus`：覆盖底层 shell 的 `GPUS`。
- `--master-port-start`：多个数据源顺序运行时，给每个数据源分配不同端口。
- `--dry-run`：只打印命令，不执行。

feature cache 输出目录来自 manifest 的 `cache_path`。当前默认是：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/recogdrive_qwen3_notrain_obj_hdmap_cache
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/recogdrive_qwen3_notrain_obj_hdmap_cache
```

缓存后检查样本目录数：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/recogdrive_qwen3_notrain_obj_hdmap_cache" -mindepth 2 -maxdepth 2 -type d | sort | wc -l
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/recogdrive_qwen3_notrain_obj_hdmap_cache" -mindepth 2 -maxdepth 2 -type d | sort | wc -l
```

检查某个 token 是否具备训练需要的所有 `.gz`：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/recogdrive_qwen3_notrain_obj_hdmap_cache" -mindepth 2 -maxdepth 2 -type d | head -1
find "<上一步输出的token目录>" -maxdepth 1 -type f -name "*.gz" | sort
```

注意：`CacheOnlyDataset` / `MultiCacheOnlyDataset` 只会加载包含当前 agent 所有 feature builder 和 target builder 对应 `.gz` 的 token 目录。如果训练样本数为 0，优先检查缓存配置和 agent 配置是否一致，例如：

- `agent.use_hdmap_vector`
- `agent.use_object_vector`
- `agent.object_encoder_type`
- `agent.use_image_encoder`
- `agent.cache_hidden_state`
- 图像尺寸相关配置

### 3. 生成 PDM/RL 用 metric cache

全量生成北京和上海 metric cache：

```bash
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py"
```

只生成单个城市：

```bash
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" --name beijing_worldsim
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" --name shanghai_worldsim
```

传递额外 Hydra override：

```bash
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" \
  --name shanghai_worldsim \
  train_test_split.data_split=trainval \
  cache.force_feature_computation=false
```

常用参数：

- `--manifest`：指定其他 manifest。
- `--name`：只跑某个数据源，可重复传多次。
- `--train-test-split`：默认 `trainval`。
- `--dry-run`：只打印命令，不执行。

metric cache 输出目录来自 manifest 的 `metric_cache_path`。当前默认是：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/metric_cache_mogo_v3
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/metric_cache_mogo_v3
```

缓存后检查：

```bash
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/metric_cache_mogo_v3" -mindepth 2 -maxdepth 2 -type f -name "metric_cache.pkl" | sort | wc -l
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/metric_cache_mogo_v3" -mindepth 2 -maxdepth 2 -type f -name "metric_cache.pkl" | sort | wc -l
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/metric_cache_mogo_v3/metadata" -maxdepth 1 -type f -name "*.csv" | sort
find "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/metric_cache_mogo_v3/metadata" -maxdepth 1 -type f -name "*.csv" | sort
```

### 4. 生成多城市 train/val log split

默认生成 `mogo_worldsim.yaml`：

```bash
python "scripts/cache_dataset/gen_mogo_multi_log_split.py"
```

当前输出位置：

```text
navsim/planning/script/config/training/log_split/mogo_worldsim.yaml
```

当前 worldsim 数据默认输出类似：

```text
train_logs: 7
val_logs: 2
test_logs: 0
```

默认是按数据源分层切分。对于当前北京 2 个 train log、上海 7 个 train log，会保证北京和上海都进入验证集。

只包含某些数据源：

```bash
python "scripts/cache_dataset/gen_mogo_multi_log_split.py" \
  --name beijing_worldsim \
  --name shanghai_worldsim
```

调整验证比例：

```bash
python "scripts/cache_dataset/gen_mogo_multi_log_split.py" --val-ratio 0.2 --seed 0
```

全局混合切分，而不是按数据源分层：

```bash
python "scripts/cache_dataset/gen_mogo_multi_log_split.py" --global-split
```

输出到自定义 config 名：

```bash
python "scripts/cache_dataset/gen_mogo_multi_log_split.py" \
  --out "navsim/planning/script/config/training/log_split/mogo_worldsim_v2.yaml"
```

训练时使用 `mogo_worldsim_v2`：

```bash
LOG_SPLIT=mogo_worldsim_v2 bash "scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh"
```

如果不同数据源存在重复 log name，脚本会直接报错。原因是当前训练入口用 `train_logs` / `val_logs` 过滤 log name，而不是 `(dataset, log_name)`，重复 log name 会导致过滤语义不明确。

### 5. 多城市 IL 训练

推荐先显式定义 cache root：

```bash
BEIJING_CACHE="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/recogdrive_qwen3_notrain_obj_hdmap_cache"
SHANGHAI_CACHE="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/recogdrive_qwen3_notrain_obj_hdmap_cache"
```

使用 shell 入口训练：

```bash
CACHE_PATH="${BEIJING_CACHE}" \
CACHE_PATHS="${BEIJING_CACHE}:${SHANGHAI_CACHE}" \
LOG_SPLIT="mogo_worldsim" \
NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/exp" \
GPUS=8 \
BATCH_SIZE=160 \
MAX_EPOCHS=200 \
bash "scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh"
```

这里的关键点：

- `CACHE_PATHS` 使用冒号 `:` 分隔多个 cache root。
- `CACHE_PATH` 仍传第一个 cache root，用于兼容旧配置；实际训练优先使用 `cache_paths`。
- `LOG_SPLIT=mogo_worldsim` 会注入 `+log_split@_global_=mogo_worldsim`。
- `NAVSIM_EXP_ROOT` 建议显式指定到 worldsim 总目录下，避免输出落到某个单城市 `exp` 下。
- `use_cache_without_dataset=True` 时训练不再构建 `SceneLoader`，不会读取 `OPENSCENE_DATA_ROOT`、sqlite 或 sensor blobs。

如果不用 shell，直接走 Hydra 也可以：

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port=9676 \
  "navsim/planning/script/run_training_recogdrive.py" \
  agent=recogdrive_agent \
  train_test_split=mogo_trainval \
  '+log_split@_global_=mogo_worldsim' \
  cache_path="${BEIJING_CACHE}" \
  "cache_paths=[${BEIJING_CACHE},${SHANGHAI_CACHE}]" \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

Hydra list 必须作为一个 shell 参数传入，因此建议用引号包住：

```bash
"cache_paths=[/path/a,/path/b]"
```

训练样本加载逻辑：

- `train_logs` 会在每个 cache root 下过滤同名 log 目录。
- `val_logs` 同理。
- token 相同但来自不同 cache root 时，IL 数据集会保留两个样本，并打印 warning。
- 下游如果只把 token 当唯一 ID 使用，需要额外确认不会产生歧义。当前 IL loss 不依赖 token-only 查外部表，通常可接受。

### 6. 多城市 RL / GRPO 训练

RL 训练除了 feature cache，还需要 metric cache。先定义路径：

```bash
BEIJING_CACHE="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/recogdrive_qwen3_notrain_obj_hdmap_cache"
SHANGHAI_CACHE="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/recogdrive_qwen3_notrain_obj_hdmap_cache"
BEIJING_METRIC="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/metric_cache_mogo_v3"
SHANGHAI_METRIC="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/metric_cache_mogo_v3"
```

使用 RL shell：

```bash
CACHE_PATH="${BEIJING_CACHE}" \
CACHE_PATHS="${BEIJING_CACHE}:${SHANGHAI_CACHE}" \
METRIC_CACHE_PATHS="${BEIJING_METRIC}:${SHANGHAI_METRIC}" \
NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/exp" \
CHECKPOINT="/path/to/il/checkpoint.ckpt" \
GPUS=8 \
bash "scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh"
```

说明：

- `CACHE_PATHS` 用于训练样本合并。
- `METRIC_CACHE_PATHS` 用于 reward 的 metric cache 合并。
- `METRIC_CACHE_PATHS` 同样使用冒号 `:` 分隔。
- `MetricCacheLoader` 如果发现多个 metric cache root 中有重复 token，会直接报错。
- 如果 feature cache 中存在重复 token，但 metric cache 中没有重复，仍需要确认业务上是否合理；RL reward 是 token-only 查表，最好保证 token 全局唯一。
- RL 脚本当前默认有 `train_token_whitelist_path=navsim/local_config/data_selection/rl_grpo_20260706_n4000_selected_tokens_v1.txt`。多城市 RL 时需要确认这个 token 白名单覆盖所有目标城市，否则训练可能只采到部分城市。

直接 Hydra 写法示例：

```bash
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master_port=9675 \
  "navsim/planning/script/run_training_recogdrive_rl.py" \
  agent=recogdrive_agent \
  agent.grpo=True \
  agent.checkpoint_path="/path/to/il/checkpoint.ckpt" \
  agent.reference_policy_checkpoint="/path/to/il/checkpoint.ckpt" \
  agent.metric_cache_path="${BEIJING_METRIC}:${SHANGHAI_METRIC}" \
  cache_path="${BEIJING_CACHE}" \
  "cache_paths=[${BEIJING_CACHE},${SHANGHAI_CACHE}]" \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

### 7. 多城市 PDM 评测

评测不能把多个城市放进一次 `run_pdm_score_recogdrive.py`，因为评测阶段仍需要一个 `SceneLoader` 和一个 `mogo_map` 配置。正确方式是按 manifest 每个城市独立评测，最后合并 CSV。

全量评测北京和上海：

```bash
python "scripts/evaluation/run_mogo_manifest_evaluation.py" \
  --gpus 8 \
  --checkpoint "/path/to/checkpoint.ckpt"
```

指定输出合并 CSV：

```bash
python "scripts/evaluation/run_mogo_manifest_evaluation.py" \
  --gpus 8 \
  --checkpoint "/path/to/checkpoint.ckpt" \
  --merged-csv "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/merged_eval/my_eval.csv"
```

只评测上海：

```bash
python "scripts/evaluation/run_mogo_manifest_evaluation.py" \
  --name shanghai_worldsim \
  --gpus 8 \
  --checkpoint "/path/to/checkpoint.ckpt"
```

传递额外 Hydra override：

```bash
python "scripts/evaluation/run_mogo_manifest_evaluation.py" \
  --gpus 8 \
  --checkpoint "/path/to/checkpoint.ckpt" \
  agent.vlm_ablation=zero \
  agent.sampling_method=ddim
```

评测脚本会为每个数据源设置：

```text
DATA_ROOT=<dataset.data_root>
METRIC_CACHE_PATH=<dataset.metric_cache_path>
SQLITE_PATH=<dataset.sqlite_path>
MOGO_UTM_ZONE=<dataset.utm_zone>
MOGO_UTM_SOUTH=<dataset.utm_south>
EXPERIMENT_NAME=<experiment_name>_<dataset.name>
```

每个城市的原始 CSV 会保留在各自城市的 `exp/<experiment_name>_<dataset.name>/<timestamp>/` 下。合并 CSV 默认保存在：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/merged_eval/
```

合并逻辑：

- 读取每个城市输出的 CSV。
- 添加 `dataset_name`、`city`、`source_csv` 列。
- 删除每个城市已有的 `average` 行。
- 对所有 token 级结果重新计算一个全局 `average` 行。

### 8. 添加新城市或新批次

假设新增广州数据：

```text
/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/guangzhou/
  dataset/maps/hm_guangzhou_xxx.sqlite
  dataset/navsim_logs/train/*.pkl
  dataset/navsim_logs/val/*.pkl
  dataset/navsim_logs/trainval/*.pkl
  dataset/sensor_blobs/...
```

只需要在 manifest 追加：

```yaml
  - name: guangzhou_worldsim
    city: guangzhou
    data_root: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/guangzhou
    sqlite_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/guangzhou/dataset/maps/hm_guangzhou_xxx.sqlite
    utm_zone: <正确zone>
    utm_south: false
    cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/guangzhou/exp/recogdrive_qwen3_notrain_obj_hdmap_cache
    metric_cache_path: /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/guangzhou/exp/metric_cache_mogo_v3
```

然后依次执行：

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" --name guangzhou_worldsim --gpus 8
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" --name guangzhou_worldsim
python "scripts/cache_dataset/gen_mogo_multi_log_split.py"
```

训练时把新 cache root 加到 `CACHE_PATHS`：

```bash
CACHE_PATHS="${BEIJING_CACHE}:${SHANGHAI_CACHE}:${GUANGZHOU_CACHE}" \
LOG_SPLIT="mogo_worldsim" \
bash "scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh"
```

RL 时同时把新 metric cache 加到 `METRIC_CACHE_PATHS`：

```bash
METRIC_CACHE_PATHS="${BEIJING_METRIC}:${SHANGHAI_METRIC}:${GUANGZHOU_METRIC}" \
bash "scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh"
```

评测无需改命令，manifest 中新增项会自动被执行：

```bash
python "scripts/evaluation/run_mogo_manifest_evaluation.py" --gpus 8 --checkpoint "/path/to/checkpoint.ckpt"
```

### 9. 常见问题与排查

#### 问题：训练样本数为 0

优先检查：

```bash
find "${BEIJING_CACHE}" -mindepth 2 -maxdepth 2 -type d | sort | wc -l
find "${SHANGHAI_CACHE}" -mindepth 2 -maxdepth 2 -type d | sort | wc -l
sed -n '1,120p' "navsim/planning/script/config/training/log_split/mogo_worldsim.yaml"
```

如果 cache 目录有样本，但训练为 0，通常是：

- `LOG_SPLIT` 里的 log name 与 cache root 下的 log 目录不一致。
- 缓存时的 agent 配置与训练时 agent 配置不一致，导致某些 `.gz` 缺失。
- `cache_paths` Hydra list 没有正确传入。建议打印命令或用引号包住：`"cache_paths=[/a,/b]"`。

#### 问题：feature cache 生成时地图查询异常

检查 manifest 里的：

```text
sqlite_path
utm_zone
utm_south
```

并确认底层命令 dry-run 输出符合预期：

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" --dry-run
```

不要把不同城市 pkl 放进一个数据根后只传一个 sqlite。

#### 问题：PDM 评测分数异常或 route 查不到

优先检查当前城市的 metric cache 是否用同一个城市的 sqlite/UTM 生成：

```bash
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" --dry-run
```

再检查评测 dry-run：

```bash
python "scripts/evaluation/run_mogo_manifest_evaluation.py" --dry-run --checkpoint "/path/to/checkpoint.ckpt"
```

评测必须按城市分别跑；不要让一次 `run_pdm_score_recogdrive.py` 同时读取北京和上海原始 logs。

#### 问题：RL 报 duplicate token

这是预期保护。RL reward 当前通过 token 查 metric cache，如果多个 metric cache root 有相同 token，无法确定该 token 应该用哪个城市的 metric cache。

解决方式：

1. 在数据生成阶段保证 token 全局唯一。
2. 或扩展 RL reward 查表 key，从 `token` 改为 `(cache_path, log_name, token)` 或 `(dataset_name, token)`。
3. 不要通过覆盖后读 cache 的方式绕过；那会静默污染 reward。

#### 问题：端口冲突

feature cache 和评测脚本支持：

```bash
--master-port-start 9700
```

底层 shell 也支持：

```bash
MASTER_PORT=9700 bash "..."
```

#### 问题：只想看命令，不想启动训练/缓存/评测

使用 dry-run：

```bash
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" --dry-run
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" --dry-run
python "scripts/evaluation/run_mogo_manifest_evaluation.py" --dry-run --checkpoint "/path/to/checkpoint.ckpt"
```

### 10. 推荐执行顺序清单

首次完整跑 worldsim 北京 + 上海：

```bash
# 1. 检查 manifest 命令是否正确。
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" --dry-run --gpus 1
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py" --dry-run
python "scripts/evaluation/run_mogo_manifest_evaluation.py" --dry-run --gpus 1 --checkpoint "/path/to/checkpoint.ckpt"

# 2. 生成训练 feature cache。
python "scripts/cache_dataset/run_mogo_manifest_feature_caching.py" --gpus 8

# 3. 生成 PDM/RL metric cache。
python "scripts/cache_dataset/run_mogo_manifest_metric_caching.py"

# 4. 生成多城市 log split。
python "scripts/cache_dataset/gen_mogo_multi_log_split.py"

# 5. IL 训练。
BEIJING_CACHE="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/recogdrive_qwen3_notrain_obj_hdmap_cache"
SHANGHAI_CACHE="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/recogdrive_qwen3_notrain_obj_hdmap_cache"
CACHE_PATH="${BEIJING_CACHE}" \
CACHE_PATHS="${BEIJING_CACHE}:${SHANGHAI_CACHE}" \
LOG_SPLIT="mogo_worldsim" \
NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/exp" \
GPUS=8 \
bash "scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh"

# 6. 多城市评测。
python "scripts/evaluation/run_mogo_manifest_evaluation.py" \
  --gpus 8 \
  --checkpoint "/path/to/il/checkpoint.ckpt"
```

如果继续做 RL：

```bash
BEIJING_METRIC="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/beijing/exp/metric_cache_mogo_v3"
SHANGHAI_METRIC="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/shanghai/exp/metric_cache_mogo_v3"
CACHE_PATH="${BEIJING_CACHE}" \
CACHE_PATHS="${BEIJING_CACHE}:${SHANGHAI_CACHE}" \
METRIC_CACHE_PATHS="${BEIJING_METRIC}:${SHANGHAI_METRIC}" \
NAVSIM_EXP_ROOT="/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/worldsim/exp" \
CHECKPOINT="/path/to/il/checkpoint.ckpt" \
GPUS=8 \
bash "scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh"
```

### 11. 代码层设计边界

这次实现刻意没有做下面两件事：

1. 没有让一个 `SceneLoader` 同时支持 per-scene map config。
2. 没有把不同城市原始 pkl/sensor blobs 物理合并成一个数据根。

原因：

- 当前 `mogo_map.sqlite_path` 和 `mogo_map.utm_zone` 是一次 Hydra run 的全局配置。
- 跨城市原始数据混读时，HDMap feature、metric cache、route roadblock 查询都会存在用错地图的风险。
- 训练从 cache 读取时不需要 map api，天然适合作为合并边界。

当前方案的责任划分：

- manifest 负责登记数据源和地图参数。
- feature cache 脚本负责按数据源独立生成训练缓存。
- metric cache 脚本负责按数据源独立生成评测/RL 缓存。
- `MultiCacheOnlyDataset` 负责训练阶段多 cache root 合并。
- `MetricCacheLoader` 负责 RL 阶段多 metric cache root 合并，并保护重复 token。
- evaluation manifest runner 负责按城市独立评测并合并 CSV。

这样保持 KISS/YAGNI：只解决当前明确需要的多城市训练和评测合并，不引入 per-scene map adapter 的复杂度。

## 本项目 4 数据集落地（mogo_multi manifest）

本节是把上面的多城市机制落到当前项目实际要合并的 4 个批次数据上。原始数据仍按批次独立，
只在训练/评测阶段合并 cache。新增 manifest：

```text
navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml
```

登记的 4 个数据源（3 北京 + 1 上海）：

| name | city | data_root | 地图 sqlite | utm_zone |
|---|---|---|---|---|
| `beijing_0312_0319` | beijing | `…/recogdrive_dataset/0312_0319` | `hm_beijing_3.4.9_20260521.sqlite` | 50 |
| `beijing_0320_0325new` | beijing | `…/recogdrive_dataset/0320-0325new` | `hm_beijing_3.4.9_20260521.sqlite` | 50 |
| `shanghai_predata` | shanghai | `…/recogdrive_dataset/predata` | `hm_shanghai_1.9.1_20260522.sqlite` | 51 |
| `beijing_predata` | beijing | `…/recogdrive_dataset/predata_beijing` | `hm_beijing_3.4.9_20260521.sqlite` | 50 |

每个 `cache_path` 指向该数据集 `exp/recogdrive_qwen3_notrain_obj_hdmap_cache`，
`metric_cache_path` 指向 `exp/metric_cache_mogo_v3`。manifest 顶层新增
`merged_eval_root`，合并 CSV 落到 `…/recogdrive_dataset/mogo_multi/merged_eval`，不再默认落到 worldsim 下。

### 选择 manifest（一次性）

4 个 manifest runner 现在支持用环境变量选默认 manifest（不填则回退 worldsim）：

```bash
export MOGO_MANIFEST=navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml
```

设置后 `run_mogo_manifest_feature_caching.py` / `run_mogo_manifest_metric_caching.py` /
`gen_mogo_multi_log_split.py` / `run_mogo_manifest_evaluation.py` 都默认读 `mogo_multi.yaml`，
不必每条命令都写 `--manifest`。也可继续用 `--manifest` 显式覆盖。

### 当前缓存现状（决定还要补什么）

- **feature cache：4 个都已生成**，且 builder 集合完全一致：每个 token 目录含
  `{qwen3_feature, object_temporal_feature_v1, hdmap_vector_feature, trajectory_target}.gz`，
  与当前 IL agent 配置（temporal obj + hdmap + lane boundary + ego SDE + use_vlm=false）匹配。
  合并 IL **现在即可训练**，无需重跑 feature cache。样本量约
  36.7K + 9.6K + 245.5K + 153.6K ≈ **445K**。
- **metric cache（仅 RL/eval 需要）：** `beijing_0312_0319`、`beijing_predata` 有可用
  `metric_cache_mogo_v3`（含 metadata CSV）；`beijing_0320_0325new` 无 v3，manifest 已指向其
  现有 `metric_cache_mogo_v1`（已验证可加载，约 5.7 万 token）；`shanghai_predata` 的 v3 有
  token 目录但 **缺 metadata CSV**（`MetricCacheLoader` 无法索引），跑 RL/eval 前需补（见下）。
  metric caching 只读地图+pkl，不需要 sensor blobs。

> **caveat（重要）：`shanghai_predata` 的 pkl 无相机图像**（`cams.*.data_path=None`），
> 其 `qwen3_feature.gz` 是退化的 VLM 特征（约 771 字节）。当前活动链路 IL `use_vlm=false`、
> RL/eval `vlm_ablation=zero` 已屏蔽 VLM 语义，故 `predata` 兼容；但**一旦开启 VLM 语义
> （use_vlm=true 且非 zero 消融），`predata` 会注入垃圾特征**。届时应把 `shanghai_predata`
> 从合并集合中剔除，或先为其补齐图像并重跑 feature cache。

### 全链路命令

先生成合并 log split（纯 CPU；数据集变动后重跑）——脚本已默认 `mogo_multi` manifest，`--out`
自动派生为 `log_split/<manifest stem>.yaml`（即 `mogo_multi.yaml`，与训练 `LOG_SPLIT` 对齐），一条命令即可：

```bash
python scripts/cache_dataset/gen_mogo_multi_log_split.py
```

等价的完整显式写法（换其他 manifest / 自定义输出 / 调比例时用）：

```bash
python scripts/cache_dataset/gen_mogo_multi_log_split.py \
  --manifest navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml \
  --out      navsim/planning/script/config/training/log_split/mogo_multi.yaml \
  --train-subdir train --test-subdir val --val-ratio 0.1 --seed 0
```

语义：`train/` 按 `--val-ratio`(0.1) 切成 train_logs/val_logs，`val/` 整体进 test_logs（held-out
测试集，训练完全不参与）。当前输出：`train_logs=6192 val_logs=690 test_logs=53`（test 全来自
0312_0319 的 `val/`，其余 3 个数据集 `val/` 为空故不贡献 test）。可选参数：`--name <dataset>` 只含
某些数据源（可重复）、`--global-split` 全局混合切分而非按数据源分层。要给其他数据集也留 test，
把对应 pkl 放进其 `dataset/navsim_logs/val/` 后重跑即可。

> test_logs 只是从训练中排除这批 log；训练入口只用 train_logs/val_logs 过滤缓存目录，
> test_logs 里的 log 目录虽在 feature cache 中（缓存按 trainval 生成）但不会被加载。若要对
> 这批 test 单独跑 PDM 评测，用评测脚本按 token/ log 过滤，或另建只含 test log 的评测 split。


合并 IL 训练（现在即可跑）——IL 脚本已**内置默认 manifest（mogo_multi），直接 bash 启动即多城市**：

```bash
GPUS=8 BATCH_SIZE=160 MAX_EPOCHS=200 \
bash scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh
```

脚本内部把 `MOGO_MANIFEST` 默认指向 `mogo_multi.yaml`，并自动派生（均可被显式环境变量覆盖）：
- `CACHE_PATHS` ← manifest 全部 `cache_path`（冒号连接，4 个数据集）；
- `LOG_SPLIT` ← manifest 文件名 stem（`mogo_multi`）；
- `NAVSIM_EXP_ROOT` ← manifest 顶层 `train_exp_root`（`mogo_multi/exp`）。

只训子集用 `MOGO_NAMES`（逗号分隔 name），例如只训北京 3 批：

```bash
MOGO_NAMES="beijing_0312_0319,beijing_0320_0325new,beijing_predata" \
GPUS=8 bash scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh
```

回到旧的单城市行为：`MOGO_MANIFEST= bash …`（置空）→ `LOG_SPLIT=mogo`、单 `cache_path`；换其他数据集：
`MOGO_MANIFEST=<其他 manifest> bash …`。要点：`CACHE_PATHS` → hydra `cache_paths=[...]` →
`MultiCacheOnlyDataset` 以 `(cache_index, log_name, token)` 为 key，跨数据集 token 相同不互相覆盖；
`use_cache_without_dataset=True` 时不建 `SceneLoader`，不读原始 pkl / sensor blobs / 地图。

补齐 RL/eval 所需 metric cache（重活，按需触发）：

```bash
# shanghai_predata：v3 有 token 目录但缺 metadata CSV，重跑补齐
python scripts/cache_dataset/run_mogo_manifest_metric_caching.py --name shanghai_predata
# beijing_0320_0325new 已用现有 v1（manifest 指向 v1），无需再生成；
# 如要统一成 v3，把 manifest 该项 metric_cache_path 改回 v3 后再单独跑：
# python scripts/cache_dataset/run_mogo_manifest_metric_caching.py --name beijing_0320_0325new
```

合并 RL / GRPO——RL 脚本同样**内置默认 manifest**，自动派生 `CACHE_PATHS` + `METRIC_CACHE_PATHS` +
`NAVSIM_EXP_ROOT`（`METRIC_CACHE_PATHS` 取 manifest 各 `metric_cache_path`，含 0320-0325new 的 v1）：

```bash
CHECKPOINT=/path/to/multi_city_il_checkpoint.ckpt \
GPUS=8 bash scripts/training/run_recogdrive_train_multi_node_rl_2b_mogo.sh
```

（跑 RL 前需先补齐 `shanghai_predata` 的 metric metadata，见上。`MOGO_MANIFEST=` 置空回单城市、
`MOGO_NAMES` 选子集、手动传 `METRIC_CACHE_PATHS` 覆盖，方式同 IL。）

RL 注意事项：
- `METRIC_CACHE_PATHS` 冒号分隔；`MetricCacheLoader` 若发现多 root 有重复 token 会直接报错
  （4 个数据集 log 名两两不冲突，token 预期也不冲突，但补完 metric cache 后应确认）。
- RL 的 agent 结构开关（`USE_VLM` / `USE_OBJECT_VECTOR` / `use_hdmap_vector` /
  `EGO_STATUS_ENCODER`）必须与 `CHECKPOINT` 对应的 IL 训练一致，否则 `_init_grpo` 的
  `strict=True` 加载会报缺键或静默错配。当前 IL 默认 `USE_VLM=false`，RL 也需 `USE_VLM=false`
  且 CHECKPOINT 来自 novlm IL。
- RL 脚本默认带 `train_token_whitelist_path=…rl_grpo_20260707_n4000_selected_tokens_v1.txt`。
  多数据集 RL 时确认该白名单覆盖目标数据集 token，否则只会采到部分数据。

合并评测（按城市分别评，再合并 CSV）：

```bash
python scripts/evaluation/run_mogo_manifest_evaluation.py \
  --gpus 8 \
  --experiment-name mogo_multi_eval \
  --checkpoint /path/to/checkpoint.ckpt
# 每城市原始 CSV 落在各自 exp/mogo_multi_eval_<name>/<ts>/；
# 合并 CSV 落在 merged_eval_root：…/recogdrive_dataset/mogo_multi/merged_eval/
```

### 加数据集配方（后续只改一处）

1. 在 `mogo_multi.yaml` 的 `datasets:` 下追加一条
   `{name, city, data_root, sqlite_path, utm_zone, utm_south, cache_path, metric_cache_path}`
   （`name` 全局唯一；北京 zone 50、上海 zone 51；`sqlite_path` 用该城市自己的地图）。
2. 生成该数据源的 cache：
   ```bash
   python scripts/cache_dataset/run_mogo_manifest_feature_caching.py --name <new> --gpus 8
   python scripts/cache_dataset/run_mogo_manifest_metric_caching.py --name <new>   # 需要 RL/eval 才跑
   ```
3. 重跑 split：`python scripts/cache_dataset/gen_mogo_multi_log_split.py`（默认即 mogo_multi，
   输出 `log_split/mogo_multi.yaml` 覆盖全部数据源含新增；deterministic，可反复重跑）。
   `python "scripts/cache_dataset/remove_logs_from_split.py" --write`
   `python "scripts/cache_dataset/gen_mogo_multi_log_split.py" --manifest "navsim/planning/script/config/common/mogo_dataset/traffic_light.yaml" `
   `python "scripts/cache_dataset/gen_mogo_multi_log_split.py" --manifest "navsim/planning/script/config/common/mogo_dataset/nav_station.yaml" `
4. **训练/RL/评测命令都不用改**（脚本与 manifest 工具已默认 `mogo_multi`）：IL/RL 直接
   `bash scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh`（RL 换对应脚本）会自动把
   manifest 新增项的 `cache_path`/`metric_cache_path` 并入 `CACHE_PATHS`/`METRIC_CACHE_PATHS`；评测
   `python scripts/evaluation/run_mogo_manifest_evaluation.py --checkpoint <ckpt>` 自动逐个跑并入合并 CSV。
   只想临时排除某项用 `MOGO_NAMES` 选子集。
