"""Station (bus-stop / pull-over) token feature builder for ReCogDrive.

进出站导航：新版 Mogo pkl 每帧的 `navigation_command` 含一个 `park` 四元组
`[park, dist, a, b]`，dist=到站沿线剩余距离(m)，(a,b)=(1,1)左侧站 / (-1,-1)右侧站
（语义见 docs/recogdrive_navigation_command_generation.md）。本 builder 把「当前活动站点」
投影到当前 ego 系，编成 1 个固定 station token，作为 DiT cross-attention 的 context
token（与 obj / hdmap / VLM token 并列），为规划器提供**显式空间目标点**，让输出轨迹在
站点处呈现进/出站（靠边、驶入/驶离）形状。这是 goal-conditioned planning 的做法
（参考 UniAD/VAD 的 goal point、DiffusionDrive 的 anchor 提议）。

站点位置来源（nav_command 派生，见方案文档决策）：
- park.dist 沿当前帧 `route_centerline`（全局 UTM 折线，首点≈当前 ego）累积弧长采点，
  得到站点全局近似 (x, y)，再用当前帧全局 ego_pose 逆变换到 ego 系。
- 侧向 side 取自 park 的 (a,b) 符号。

兼容性：
- 无 park / 无 route_centerline / 旧数据（无 navigation_command）→ 返回全 padding
  （station_valid_mask=False），编码器按 mask 置零，DiT 忽略，等价「无站点约束」。
- 结构完全对标 object_vector_features.py（compute_features_from_scene + 无 scene 兜底）。

坐标系：站点 (x,y) 与 TrajectoryTargetBuilder 的未来轨迹、obj token 同为当前 ego 后轴系，
一致（ego 前为 +x、左为 +y）。
"""

from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from navsim.agents.abstract_agent import AgentInput
from navsim.common.command_utils import NAV_COMMAND_DIST_NORM, parse_park_command
from navsim.common.dataclasses import Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


# --- 单一来源：站点特征维度 / token 数，agent 与 planner config 都从这里取 ---
# 布局: [x_norm, y_norm, dist_norm, sin_bearing, cos_bearing, side]
STATION_FEATURE_DIM: int = 6
STATION_FEATURE_NAME: str = "station_feature_v1"

# 归一化尺度：站点纵向可能较远（弧长采点，最远 ~centerline 长度），用较大尺度避免饱和。
_NORM_X: float = 100.0
_NORM_Y: float = 40.0


def _global_xy_to_local(global_xy: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """把全局 XY 点转到当前后轴（ego）系。复用 safety_targets._global_xy_to_local 的公式。

    :param global_xy: (N, 2) 全局 UTM 点。
    :param ego_pose: (>=3,) 全局 [x, y, heading]。
    :return: (N, 2) ego 系 [x(前), y(左)]。
    """
    delta = global_xy - np.asarray(ego_pose[:2], dtype=np.float64)
    yaw = float(ego_pose[2])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    return np.stack([local_x, local_y], axis=-1)


def _station_xy_along_centerline(
    route_centerline: List, dist_m: float
) -> Optional[np.ndarray]:
    """沿 route_centerline（全局折线）累积弧长到 dist_m 处取点，返回全局 (x, y)。

    首点≈当前 ego 位置，故弧长即到站沿线剩余距离，与 park.dist 语义一致。
    dist 超出折线总长时钳到末点；折线点数不足时返回 None。
    """
    if route_centerline is None:
        return None
    try:
        pts = np.asarray(route_centerline, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] < 2:
        return None
    xy = pts[:, :2]
    if xy.shape[0] == 1:
        return xy[0].copy()

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    target = float(min(max(dist_m, 0.0), total))

    # 找到 target 落在的分段，线性插值。
    idx = int(np.searchsorted(cum, target, side="right")) - 1
    idx = max(0, min(idx, len(seg) - 1))
    seg_len = seg[idx]
    if seg_len < 1e-6:
        return xy[idx].copy()
    ratio = (target - cum[idx]) / seg_len
    return xy[idx] + ratio * (xy[idx + 1] - xy[idx])


def collate_station_features(features: Dict[str, torch.Tensor], features_list) -> None:
    """把固定形状的站点张量 stack 进 batch：station_features (B,N,D) / station_valid_mask (B,N)。

    与 obj / front_image 同为固定形状，直接 stack，无需 pad_sequence。IL / RL / accelerate /
    ema 四处 collate_fn 共用本函数。仅当缓存里带 station_features 时补（use_station_token=False
    或旧缓存无该键时零影响）。成对存在（builder 一次写两键），故只判 station_features。
    """
    if not features_list:
        return
    if "station_features" in features_list[0]:
        features["station_features"] = torch.stack(
            [f["station_features"] for f in features_list], dim=0
        ).cpu()
        features["station_valid_mask"] = torch.stack(
            [f["station_valid_mask"] for f in features_list], dim=0
        ).cpu()


def _active_station_xy_from_meta(navigation_command_meta) -> Optional[np.ndarray]:
    """从 navigation_command_meta 取**精确**活动站点全局 UTM 坐标 (x, y)。

    优先 station_target_buffer.current_target（当前目标站点，实测 100% 覆盖 park 帧且投影
    与 park.dist 精确吻合）；退化 station_park.target。缺失返回 None。

    这是 P1 的正解：直接用生成侧算好的精确站点坐标，避免「park.dist 沿中心线反投影」在
    站点偏离中心线 / 中心线未覆盖站点 road 时的误差（实测不同 log 站点到中心线距离 0.4~1700m）。
    """
    if not isinstance(navigation_command_meta, dict):
        return None
    for key, sub in (("station_target_buffer", "current_target"), ("station_park", "target")):
        block = navigation_command_meta.get(key)
        if isinstance(block, dict):
            target = block.get(sub)
            if isinstance(target, dict):
                pxy = target.get("point_xy")
                if pxy is not None and len(pxy) >= 2:
                    try:
                        return np.array([float(pxy[0]), float(pxy[1])], dtype=np.float64)
                    except (TypeError, ValueError):
                        pass
    return None


class StationFeatureBuilder(AbstractFeatureBuilder):
    """从当前帧 navigation_command 的 park + 精确站点坐标（meta）派生站点 token 特征。

    站点位置优先用 navigation_command_meta 的精确全局坐标（P1 修复），退化到 park.dist 沿
    route_centerline 反投影，再退化到 dist/side 粗略占位。
    """

    def __init__(self, max_station_tokens: int = 1):
        """:param max_station_tokens: 站点 token 数。**当前仅支持 1**（只编码当前活动站点）。

        P6：_build 只写 slot 0，>1 的 slot 恒为 padding；且缓存名不含 token 数，改成 >1 会
        读到 shape 为 [1,6] 的旧缓存导致 mismatch。故此处 assert 强制为 1，防呆。多站点
        （从 route_stations 选最近 N 个）是未来扩展，需同时实现多站点选择/排序/坐标变换与
        缓存版本化，见 docs/recogdrive_navigation_and_station_plan.md §10.2。
        """
        super().__init__()
        if int(max_station_tokens) != 1:
            raise ValueError(
                f"StationFeatureBuilder 当前仅支持 max_station_tokens==1（只编码当前活动站点），"
                f"收到 {max_station_tokens}。多站点为未来扩展，见方案文档 §10.2。"
            )
        self.max_station_tokens = 1

    def get_unique_name(self) -> str:
        return STATION_FEATURE_NAME

    # ------------------------------------------------------------------ #
    # 离线缓存 / 在线 eval 主路径：scene-aware（有 route_centerline + 全局 ego_pose）
    # ------------------------------------------------------------------ #
    def compute_features_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        frame_idx = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[frame_idx]
        navigation_command = getattr(frame, "navigation_command", None)
        navigation_command_meta = getattr(frame, "navigation_command_meta", None)
        route_centerline = getattr(frame, "route_centerline", None)
        ego_pose = np.asarray(frame.ego_status.ego_pose, dtype=np.float64)
        return self._build(navigation_command, navigation_command_meta, route_centerline, ego_pose)

    # ------------------------------------------------------------------ #
    # 无 scene 兜底：AgentInput 无 route_centerline / 全局 ego_pose，返回全 padding。
    # 真正评测启用 station token 时由 agent 在 requires_scene=True 下传 scene。
    # ------------------------------------------------------------------ #
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        return self._empty_feature_dict()

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _empty_feature_dict(self) -> Dict[str, torch.Tensor]:
        n = self.max_station_tokens
        return {
            "station_features": torch.zeros((n, STATION_FEATURE_DIM), dtype=torch.float32),
            "station_valid_mask": torch.zeros((n,), dtype=torch.bool),
        }

    def _build(
        self,
        navigation_command,
        navigation_command_meta,
        route_centerline,
        ego_pose: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        out = self._empty_feature_dict()

        park = parse_park_command(navigation_command)
        if park is None:
            return out  # 无 park（含旧数据）→ 全 padding
        dist_m, side = park

        # P1 修复：站点位置三级来源，优先精确。
        # 1) meta 精确全局坐标（生成侧算好，处理站点偏离中心线/横向偏移，实测 100% 覆盖 park 帧）；
        # 2) 退化：park.dist 沿 route_centerline 弧长反投影（站点近似在中心线上）；
        # 3) 再退化：dist/side 粗略占位（前方 dist、横向 side*车道半宽）。
        station_global = _active_station_xy_from_meta(navigation_command_meta)
        if station_global is None:
            station_global = _station_xy_along_centerline(route_centerline, dist_m)
        if station_global is None:
            x_local, y_local = float(dist_m), float(side) * 1.75
        else:
            local = _global_xy_to_local(station_global[None, :], ego_pose)[0]
            x_local, y_local = float(local[0]), float(local[1])

        bearing = math.atan2(y_local, x_local)
        feature_vec = np.array(
            [
                x_local / _NORM_X,
                y_local / _NORM_Y,
                float(min(max(dist_m / NAV_COMMAND_DIST_NORM, 0.0), 1.0)),
                math.sin(bearing),
                math.cos(bearing),
                float(side),
            ],
            dtype=np.float32,
        )
        assert feature_vec.shape[0] == STATION_FEATURE_DIM

        out["station_features"][0] = torch.from_numpy(feature_vec)
        out["station_valid_mask"][0] = True
        return out
