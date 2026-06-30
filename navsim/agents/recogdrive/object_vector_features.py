"""Object (obstacle) vector feature builder for ReCogDrive.

从 Scene 当前帧的 GT annotations 提取障碍物，编码成固定形状的结构化 token，
作为 ReCogDriveDiffusionPlanner 规划侧的 context token（与 VLM hidden state 并列
进 DiT cross-attention）。

设计原则（见 docs/obstacle_token_fusion_plan.md）：
- 只用当前帧障碍物（KISS），不引入 future GT，避免 train/eval 不一致的越界信息。
- 连续几何特征用归一化 + LayerNorm/MLP 编码，类别用 embedding，不写进 VLM prompt。
- builder 只负责造数据，不碰 VLM / DiT；模型侧编码在 recogdrive_diffusion_planner.ObjectEncoder。
- compute_features(agent_input) 返回全 padding，保证无 scene 的在线推理不崩；真正
  评测时由 agent 在 requires_scene=True 下走 compute_features_from_scene(scene)。

坐标系：annotations.boxes 的 (x,y,z) 与 heading 均为当前 ego 系（heading 已在 Mogo
生成脚本修正为 yaw-ego_yaw），与 TrajectoryTargetBuilder 的未来轨迹同系，一致。
"""

from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import numpy.typing as npt
import torch

from navsim.agents.abstract_agent import AgentInput
from navsim.common.dataclasses import Scene
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


# --- 单一来源：特征维度 / 类别数，agent 与 planner config 都从这里取，避免维度对不上 ---
OBJECT_FEATURE_DIM: int = 15
OBJECT_NUM_TYPES: int = 5

# 类别 -> type id。padding / 未知类别用 unknown(4)，编码时会被 valid_mask 置零。
OBJ_TYPE_TO_ID: Dict[str, int] = {
    "vehicle": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "static": 3,
}
_OBJ_TYPE_UNKNOWN_ID: int = 4

# 动态 / 静态类别集合。include_static=False 时只保留动态三类（与 Dataset_For_Pipeline
# 的 _compute_agent_targets 保持一致）。
_DYNAMIC_CLASSES = {"vehicle", "pedestrian", "bicycle"}
_STATIC_CLASSES = {"traffic_cone", "barrier", "czone_sign", "generic_object", "static"}

# --- 归一化常量（让原始特征落到 O(1)，LayerNorm 会再做一次归一化）---
_NORM_X = 80.0
_NORM_Y = 40.0
_NORM_Z = 5.0
_NORM_LENGTH = 20.0
_NORM_WIDTH = 5.0
_NORM_HEIGHT = 5.0
_NORM_VEL = 20.0
_NORM_DIST = 80.0
_TTC_MAX = 10.0


def _category_of(name: str) -> Optional[str]:
    """把原始类别名映射到 {vehicle,pedestrian,bicycle,static}；不识别返回 None。"""
    name = str(name).lower()
    if name in _DYNAMIC_CLASSES:
        return name
    if name in _STATIC_CLASSES:
        return "static"
    return None


class ObjectVectorFeatureBuilder(AbstractFeatureBuilder):
    """从 Scene 当前帧提取障碍物，输出固定形状的 obj token 特征。"""

    def __init__(
        self,
        max_obj_tokens: int = 64,
        obj_range_x: Tuple[float, float] = (-20.0, 100.0),
        obj_range_y: Tuple[float, float] = (-40.0, 40.0),
        include_static: bool = False,
    ):
        """
        :param max_obj_tokens: 固定 token 数量，按风险排序后截断 / padding 到该长度
        :param obj_range_x: 纵向保留范围 (min, max)，单位 m，ego 前为 +x
        :param obj_range_y: 横向保留范围 (min, max)，单位 m
        :param include_static: 是否保留静态类别（traffic_cone/barrier/...）
        """
        super().__init__()
        self.max_obj_tokens = int(max_obj_tokens)
        self.obj_range_x = (float(obj_range_x[0]), float(obj_range_x[1]))
        self.obj_range_y = (float(obj_range_y[0]), float(obj_range_y[1]))
        self.include_static = bool(include_static)

    def get_unique_name(self) -> str:
        # 缓存文件名。第一版固定 v1；若以后改特征布局 / 排序逻辑需 bump 版本，避免脏读旧缓存。
        return "object_vector_feature_v1"

    # ------------------------------------------------------------------ #
    # 离线缓存路径：训练 cache 走这里（Dataset._cache_scene_with_token 已 scene-aware）
    # ------------------------------------------------------------------ #
    def compute_features_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        return self._build_from_annotations(annotations)

    # ------------------------------------------------------------------ #
    # 无 scene 推理兜底：返回全 padding（valid_mask 全 False）。真正评测启用 obj token
    # 时应由 agent 在 requires_scene=True 下传 scene；否则只代表训练侧 privileged 条件。
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
            "obj_type": torch.full((n,), _OBJ_TYPE_UNKNOWN_ID, dtype=torch.long),
            "obj_valid_mask": torch.zeros((n,), dtype=torch.bool),
            "obj_pose": torch.zeros((n, 3), dtype=torch.float32),
        }

    def _in_range(self, x: float, y: float) -> bool:
        return (
            self.obj_range_x[0] <= x <= self.obj_range_x[1]
            and self.obj_range_y[0] <= y <= self.obj_range_y[1]
        )

    def _build_from_annotations(self, annotations) -> Dict[str, torch.Tensor]:
        boxes = np.asarray(annotations.boxes, dtype=np.float32)
        names = list(annotations.names)
        velocities = np.asarray(annotations.velocity_3d, dtype=np.float32)

        records: List[Dict[str, float]] = []
        if boxes.ndim == 2 and boxes.shape[0] > 0:
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
                # ttc_approx：距离 / 物体地速，clamp 到 [0, TTC_MAX]。这是 frame-robust 的
                # 近似（地速是模长，不依赖速度坐标系），不解析径向闭合速度，仅作风险序信号。
                ttc = distance / max(speed, 1e-3)
                ttc = float(min(max(ttc, 0.0), _TTC_MAX))

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
                        # 排序用的归一化中间量
                        "norm_distance": distance / _NORM_DIST,
                        "norm_abs_lateral": abs(y) / _NORM_Y,
                        "norm_ttc": ttc / _TTC_MAX,
                    }
                )

        records = self._sort_obstacles(records)[: self.max_obj_tokens]

        out = self._empty_feature_dict()
        for slot, rec in enumerate(records):
            out["obj_features"][slot] = torch.from_numpy(rec["feature"])
            out["obj_type"][slot] = int(rec["type_id"])
            out["obj_valid_mask"][slot] = True
            out["obj_pose"][slot] = torch.from_numpy(rec["pose"])
        return out

    @staticmethod
    def _sort_obstacles(records: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """按风险分升序（越小越关键）排序，用于截断时优先保留关键障碍物。

        risk = 0.5*norm_distance + 0.3*norm_abs_lateral + 0.2*norm_ttc
        预留为独立函数，便于后续替换为 TTC / 计划相关性等更复杂的排序。
        """
        return sorted(
            records,
            key=lambda r: 0.5 * r["norm_distance"] + 0.3 * r["norm_abs_lateral"] + 0.2 * r["norm_ttc"],
        )
