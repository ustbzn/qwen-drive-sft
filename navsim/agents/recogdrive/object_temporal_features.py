"""Temporal object (obstacle) vector feature builder for ReCogDrive.

在 `object_vector_features.py`（单帧 15-D）基础上，为每个当前帧障碍物额外提取
「历史+当前」多帧轨迹，做**差分编码**供规划侧的时序 FPN 编码器（planner 的
`TemporalObjectEncoder`）捕捉运动意图。动/静按当前帧地速阈值划分：

- 动态（speed >= static_speed_threshold）：abs 几何 token + 时序差分 motion token。
- 静态（speed <  static_speed_threshold）：仅 abs 几何 token（纯约束）。

设计要点（见 docs/obstacle_token_fusion_plan.md 与本次时序改造方案）：
- 时序窗口 = 历史帧 + 当前帧，共 `num_history_frames` 帧（mogo_10hz=4，@2Hz≈1.5s），
  **不使用未来帧**（train/eval 一致、可部署）。
- 历史帧障碍物 box 在「各自帧的 ego 系」（Mogo 生成脚本 + fix_mogo_obstacle_frame.py
  已把 heading/velocity 修到各帧 ego 系）。ego 0.5s 移动数米，做时序位置差分前
  **必须先把历史帧对象统一到当前帧 ego 系**，否则差分被 ego 自身运动污染。
- `scene.frames[i].ego_status.ego_pose` 是全局系 [X,Y,yaw]（in_global_frame=True），
  用它做「帧 f local -> global -> 当前帧 c local」的两段变换。
- 跨帧对象用 `track_tokens` 关联（mogo track 跨帧一致性好）。
- builder 只造数据；模型侧编码在 planner 的 TemporalObjectEncoder。
- get_unique_name() bump 为 object_temporal_feature_v1，与旧 object_vector_feature_v1
  不同文件，开启时必须重跑 obj 缓存。

坐标系：当前帧障碍物 (x,y,heading)/velocity 与 TrajectoryTargetBuilder 未来轨迹同系
（当前 ego 系），历史帧经变换后落到同一当前 ego 系。
"""

from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.agents.abstract_agent import AgentInput
from navsim.common.dataclasses import Scene
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder

# 复用单帧 builder 的类别映射、类型数与归一化常量，保证 abs 分支的 15-D 布局一致。
from .object_vector_features import (
    OBJECT_FEATURE_DIM,
    OBJECT_NUM_TYPES,
    OBJ_TYPE_TO_ID,
    _OBJ_TYPE_UNKNOWN_ID,
    _category_of,
    _NORM_X,
    _NORM_Y,
    _NORM_Z,
    _NORM_LENGTH,
    _NORM_WIDTH,
    _NORM_HEIGHT,
    _NORM_VEL,
    _NORM_DIST,
    _TTC_MAX,
)


# --- 差分特征维度（与 pluto to_vector 的 9 维对齐）与归一化常量 ---
OBJECT_DIFF_DIM: int = 9
# ego 0.5s 约移动 4~5m，物体相对位移可能更大；用 10m 作 delta 尺度，LayerNorm/GroupNorm 再规整。
_NORM_DELTA_XY: float = 10.0

# 单帧间隔(约0.5s)内物体相对位移的物理上限(米)。超过即判为异常帧并置零该 delta。
# 已知触发源(上海 predata 批次)：原始 pkl 某帧 gt_boxes 漏做 global->ego 变换，该帧 object
# 坐标是 UTM 大数(3.4e6)，object 在该中间帧"瞬移"产生 1e5~1e6 的 delta。根因在 object 坐标、
# 不在 ego_pose（ego 变换成功抵消了 UTM，见 memory 与回溯 pkl 核实）。正常最快相对运动
# (对向~200km/h≈55m/s * 0.5s ≈ 28m)远低于 100m；此阈值只兜底物理不可能的坏值。
_MAX_DELTA_POS: float = 100.0

# 时序窗口默认帧数（含当前帧），与 mogo_10hz 的 num_history_frames 对齐。
DEFAULT_NUM_HISTORY_FRAMES: int = 4


def _wrap(angle):
    """把角度归一化到 [-pi, pi]。支持标量或 ndarray。"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _local_to_global_se2(xytheta: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """把某帧 ego-local 的 SE2 状态 [K,3] 变换到全局系。

    ego_pose = [X, Y, yaw]（该帧 ego 在全局系）。逆于
    convert_absolute_to_relative_se2_array（后者是 global->relative）。
    """
    X, Y, yaw = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    xy_global = xytheta[:, :2] @ R.T + np.array([X, Y], dtype=np.float64)
    theta_global = _wrap(xytheta[:, 2] + yaw)
    return np.concatenate([xy_global, theta_global[:, None]], axis=1)


def _rotate_velocity(vel_xy: np.ndarray, yaw_from: float, yaw_to: float) -> np.ndarray:
    """把 2D 速度矢量 [K,2] 从 yaw_from 帧 ego 系旋转到 yaw_to 帧 ego 系（只转不平移）。

    v_to = R(yaw_from - yaw_to) @ v_from。
    """
    d = float(yaw_from) - float(yaw_to)
    c, s = math.cos(d), math.sin(d)
    vx = c * vel_xy[:, 0] - s * vel_xy[:, 1]
    vy = s * vel_xy[:, 0] + c * vel_xy[:, 1]
    return np.stack([vx, vy], axis=1)


class ObjectTemporalFeatureBuilder(AbstractFeatureBuilder):
    """从 Scene 历史+当前帧提取障碍物时序轨迹，输出固定形状的时序 obj token 特征。"""

    def __init__(
        self,
        max_obj_tokens: int = 64,
        obj_range_x: Tuple[float, float] = (-20.0, 100.0),
        obj_range_y: Tuple[float, float] = (-40.0, 40.0),
        include_static: bool = False,
        static_speed_threshold: float = 0.5,
        num_history_frames: int = DEFAULT_NUM_HISTORY_FRAMES,
    ):
        """
        :param max_obj_tokens: 固定 token 数量，按风险排序后截断 / padding 到该长度
        :param obj_range_x: 纵向保留范围 (min, max)，单位 m，ego 前为 +x
        :param obj_range_y: 横向保留范围 (min, max)，单位 m
        :param include_static: 是否保留静态类别（当前 mogo 数据无静态类别，保留兼容）
        :param static_speed_threshold: 动/静划分阈值，当前帧地速 < 阈值判为静态（纯 abs 约束）
        :param num_history_frames: 时序窗口帧数（含当前帧），决定 diff 序列长度 T-1
        """
        super().__init__()
        self.max_obj_tokens = int(max_obj_tokens)
        self.obj_range_x = (float(obj_range_x[0]), float(obj_range_x[1]))
        self.obj_range_y = (float(obj_range_y[0]), float(obj_range_y[1]))
        self.include_static = bool(include_static)
        self.static_speed_threshold = float(static_speed_threshold)
        self.num_history_frames = int(num_history_frames)
        # diff 序列长度固定，保证 batch collate 形状一致（mogo_10hz num_history_frames=4 -> 3）。
        self.num_deltas = max(self.num_history_frames - 1, 1)

    def get_unique_name(self) -> str:
        # 时序特征缓存文件名。与单帧 object_vector_feature_v1 区分，避免脏读旧缓存。
        return "object_temporal_feature_v1"

    # ------------------------------------------------------------------ #
    # 离线缓存路径：训练 cache 走这里（Dataset._cache_scene_with_token 已 scene-aware）
    # ------------------------------------------------------------------ #
    def compute_features_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        return self._build_from_scene(scene)

    # ------------------------------------------------------------------ #
    # 无 scene 推理兜底：返回全 padding。真正评测启用时应由 agent 在 requires_scene=True
    # 下传 scene；否则只代表训练侧 privileged 条件。
    # ------------------------------------------------------------------ #
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        return self._empty_feature_dict()

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _empty_feature_dict(self) -> Dict[str, torch.Tensor]:
        n = self.max_obj_tokens
        return {
            "obj_features": torch.zeros((n, OBJECT_FEATURE_DIM), dtype=torch.float32),
            "obj_diff_features": torch.zeros((n, self.num_deltas, OBJECT_DIFF_DIM), dtype=torch.float32),
            "obj_type": torch.full((n,), _OBJ_TYPE_UNKNOWN_ID, dtype=torch.long),
            "obj_valid_mask": torch.zeros((n,), dtype=torch.bool),
            "obj_is_dynamic": torch.zeros((n,), dtype=torch.bool),
            "obj_pose": torch.zeros((n, 3), dtype=torch.float32),
        }

    def _in_range(self, x: float, y: float) -> bool:
        return (
            self.obj_range_x[0] <= x <= self.obj_range_x[1]
            and self.obj_range_y[0] <= y <= self.obj_range_y[1]
        )

    @staticmethod
    def _ego_yaw(ego_pose: np.ndarray) -> float:
        return float(ego_pose[2])

    def _build_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        num_history = scene.scene_metadata.num_history_frames
        curr_idx = num_history - 1
        # 时序窗口 [start_idx .. curr_idx]，长度 self.num_history_frames（不足则从 0 起，前补无效）。
        start_idx = max(0, curr_idx - (self.num_history_frames - 1))
        frame_indices = list(range(start_idx, curr_idx + 1))  # 末尾是当前帧

        curr_frame = scene.frames[curr_idx]
        ego_pose_c = np.asarray(curr_frame.ego_status.ego_pose, dtype=np.float64)
        yaw_c = self._ego_yaw(ego_pose_c)

        # 1) 当前帧：range/category 过滤 + 15-D 特征 + risk 排序 + 截断，得到 anchor slot。
        records = self._collect_current_records(curr_frame.annotations)
        records = self._sort_obstacles(records)[: self.max_obj_tokens]

        # 2) 预变换每个历史帧的全部对象到当前 ego 系，建 track_token -> (xyheading, vxy) 映射。
        per_frame_lookup: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = [
            self._transform_frame_to_current(scene.frames[f], ego_pose_c, yaw_c)
            for f in frame_indices
        ]

        # 3) 组装输出。
        out = self._empty_feature_dict()
        for slot, rec in enumerate(records):
            out["obj_features"][slot] = torch.from_numpy(rec["feature"])
            out["obj_type"][slot] = int(rec["type_id"])
            out["obj_valid_mask"][slot] = True
            out["obj_is_dynamic"][slot] = bool(rec["is_dynamic"])
            out["obj_pose"][slot] = torch.from_numpy(rec["pose"])

            # 该 anchor 在时序窗口各帧的轨迹（当前 ego 系）。
            track_token = rec["track_token"]
            traj = self._gather_track_trajectory(track_token, per_frame_lookup)
            diff = self._build_diff_features(traj)
            out["obj_diff_features"][slot] = torch.from_numpy(diff)
        return out

    def _collect_current_records(self, annotations) -> List[Dict]:
        """当前帧障碍物 -> 15-D 特征 record（复用单帧布局 + 动静标记 + track_token）。"""
        boxes = np.asarray(annotations.boxes, dtype=np.float32)
        names = list(annotations.names)
        velocities = np.asarray(annotations.velocity_3d, dtype=np.float32)
        track_tokens = list(annotations.track_tokens)

        records: List[Dict] = []
        if boxes.ndim != 2 or boxes.shape[0] == 0:
            return records

        for i in range(boxes.shape[0]):
            category = _category_of(names[i])
            if category is None:
                continue
            if category == "static" and not self.include_static:
                continue

            box = boxes[i]
            x = float(box[BoundingBoxIndex.X])
            y = float(box[BoundingBoxIndex.Y])
            z = float(box[BoundingBoxIndex.Z])
            length = float(box[BoundingBoxIndex.LENGTH])
            width = float(box[BoundingBoxIndex.WIDTH])
            height = float(box[BoundingBoxIndex.HEIGHT])
            heading = float(box[BoundingBoxIndex.HEADING])

            if not self._in_range(x, y):
                continue

            vel = velocities[i] if i < len(velocities) else np.zeros(3, dtype=np.float32)
            vx = float(vel[0])
            vy = float(vel[1])
            speed = math.hypot(vx, vy)
            distance = math.hypot(x, y)
            bearing = math.atan2(y, x)
            ttc = distance / max(speed, 1e-3)
            ttc = float(min(max(ttc, 0.0), _TTC_MAX))

            # 15-D 当前帧几何特征，布局与 object_vector_features._build_from_annotations 一致。
            feature_vec = np.array(
                [
                    x / _NORM_X,
                    y / _NORM_Y,
                    z / _NORM_Z,
                    length / _NORM_LENGTH,
                    width / _NORM_WIDTH,
                    height / _NORM_HEIGHT,
                    math.sin(heading),
                    math.cos(heading),
                    vx / _NORM_VEL,
                    vy / _NORM_VEL,
                    speed / _NORM_VEL,
                    distance / _NORM_DIST,
                    math.sin(bearing),
                    math.cos(bearing),
                    ttc / _TTC_MAX,
                ],
                dtype=np.float32,
            )
            assert feature_vec.shape[0] == OBJECT_FEATURE_DIM

            records.append(
                {
                    "feature": feature_vec,
                    "type_id": OBJ_TYPE_TO_ID.get(category, _OBJ_TYPE_UNKNOWN_ID),
                    "pose": np.array([x, y, heading], dtype=np.float32),
                    "track_token": str(track_tokens[i]) if i < len(track_tokens) else "",
                    "is_dynamic": speed >= self.static_speed_threshold,
                    "norm_distance": distance / _NORM_DIST,
                    "norm_abs_lateral": abs(y) / _NORM_Y,
                    "norm_ttc": ttc / _TTC_MAX,
                }
            )
        return records

    def _transform_frame_to_current(
        self, frame, ego_pose_c: np.ndarray, yaw_c: float
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """把某历史/当前帧的全部对象统一到当前 ego 系，返回 track_token -> (xyheading[3], vxy[2])。"""
        annotations = frame.annotations
        boxes = np.asarray(annotations.boxes, dtype=np.float64)
        velocities = np.asarray(annotations.velocity_3d, dtype=np.float64)
        track_tokens = list(annotations.track_tokens)
        lookup: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if boxes.ndim != 2 or boxes.shape[0] == 0:
            return lookup

        ego_pose_f = np.asarray(frame.ego_status.ego_pose, dtype=np.float64)
        yaw_f = self._ego_yaw(ego_pose_f)

        # 该帧 local (x,y,heading) -> global -> 当前帧 ego-local。
        local_xyt = np.stack(
            [
                boxes[:, BoundingBoxIndex.X],
                boxes[:, BoundingBoxIndex.Y],
                boxes[:, BoundingBoxIndex.HEADING],
            ],
            axis=1,
        )
        global_xyt = _local_to_global_se2(local_xyt, ego_pose_f)
        curr_xyt = convert_absolute_to_relative_se2_array(StateSE2(*ego_pose_c), global_xyt)

        # velocity 只旋转到当前 ego 系。
        if velocities.ndim == 2 and velocities.shape[0] == boxes.shape[0]:
            vel_curr = _rotate_velocity(velocities[:, :2], yaw_f, yaw_c)
        else:
            vel_curr = np.zeros((boxes.shape[0], 2), dtype=np.float64)

        for i in range(boxes.shape[0]):
            tk = str(track_tokens[i]) if i < len(track_tokens) else ""
            if tk:
                lookup[tk] = (curr_xyt[i].astype(np.float32), vel_curr[i].astype(np.float32))
        return lookup

    def _gather_track_trajectory(
        self,
        track_token: str,
        per_frame_lookup: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    ) -> Dict[str, np.ndarray]:
        """按时序窗口顺序收集某 track 的 (x,y,heading,vx,vy) + valid。缺失帧置零 valid=False。

        输出长度 = self.num_history_frames（末尾对齐当前帧）。若实际窗口更短（scene 边界），
        在头部补无效帧，保证 diff 序列长度固定。
        """
        T = self.num_history_frames
        xyt = np.zeros((T, 3), dtype=np.float32)
        vel = np.zeros((T, 2), dtype=np.float32)
        valid = np.zeros((T,), dtype=bool)

        # per_frame_lookup 末尾对齐当前帧；右对齐填入。
        n = len(per_frame_lookup)
        for k in range(n):
            slot_t = T - n + k  # 右对齐
            if slot_t < 0:
                continue
            hit = per_frame_lookup[k].get(track_token)
            if hit is not None:
                xyt[slot_t] = hit[0]
                vel[slot_t] = hit[1]
                valid[slot_t] = True
        return {"xyt": xyt, "vel": vel, "valid": valid}

    def _build_diff_features(self, traj: Dict[str, np.ndarray]) -> np.ndarray:
        """相邻帧差分 -> [T-1, 9]。两帧都 valid 才算 delta，否则整行置零（pluto 同款）。

        9 维: [dx, dy, dvx, dvy, cos(dheading), sin(dheading), speed_t1, |dpos|, pair_valid]。
        """
        xyt = traj["xyt"]      # [T,3]
        vel = traj["vel"]      # [T,2]
        valid = traj["valid"]  # [T]
        T = xyt.shape[0]
        out = np.zeros((self.num_deltas, OBJECT_DIFF_DIM), dtype=np.float32)

        # 与末尾（当前帧）对齐：delta t 连接 (t, t+1)。取最后 num_deltas 个 delta。
        deltas = []
        for t in range(T - 1):
            pair_valid = bool(valid[t] and valid[t + 1])
            if not pair_valid:
                deltas.append(np.zeros((OBJECT_DIFF_DIM,), dtype=np.float32))
                continue
            dx = (xyt[t + 1, 0] - xyt[t, 0])
            dy = (xyt[t + 1, 1] - xyt[t, 1])
            dvx = (vel[t + 1, 0] - vel[t, 0])
            dvy = (vel[t + 1, 1] - vel[t, 1])
            dheading = _wrap(xyt[t + 1, 2] - xyt[t, 2])
            speed_t1 = math.hypot(float(vel[t + 1, 0]), float(vel[t + 1, 1]))
            dpos = math.hypot(float(dx), float(dy))
            # 异常帧防护：dpos 超过物理上限说明某帧坐标被 ego_pose 异常污染（残留 UTM 大偏移），
            # 该 delta 不可信，整行置零并标 pair_valid=0（与"帧无效"同处理）。否则 1e5~1e6 坏值
            # 会写进缓存，下游 fp16 Conv1d 溢出成 nan。这是根治：让新生成的缓存不再带毒。
            if not math.isfinite(dpos) or dpos > _MAX_DELTA_POS:
                deltas.append(np.zeros((OBJECT_DIFF_DIM,), dtype=np.float32))
                continue
            deltas.append(
                np.array(
                    [
                        dx / _NORM_DELTA_XY,
                        dy / _NORM_DELTA_XY,
                        dvx / _NORM_VEL,
                        dvy / _NORM_VEL,
                        math.cos(float(dheading)),
                        math.sin(float(dheading)),
                        speed_t1 / _NORM_VEL,
                        dpos / _NORM_DELTA_XY,
                        1.0,
                    ],
                    dtype=np.float32,
                )
            )
        # 右对齐取最后 num_deltas 个（当前帧侧的运动最重要）。
        deltas = deltas[-self.num_deltas :]
        for k, d in enumerate(deltas):
            out[self.num_deltas - len(deltas) + k] = d
        return out

    @staticmethod
    def _sort_obstacles(records: List[Dict]) -> List[Dict]:
        """按风险分升序（越小越关键）排序，截断时优先保留关键障碍物。与单帧 builder 一致。"""
        return sorted(
            records,
            key=lambda r: 0.5 * r["norm_distance"] + 0.3 * r["norm_abs_lateral"] + 0.2 * r["norm_ttc"],
        )
