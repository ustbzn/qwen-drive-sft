"""将离线可视化轨迹平滑地吸附到自车左侧相邻车道中心线。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


@dataclass(frozen=True)
class LaneCenterPolishConfig:
    """中心线查询及轨迹混合参数；局部坐标 ``y > 0`` 表示自车左侧。"""

    temporal_radius: int = 2
    lift_start_frame: int = 0
    lift_transition_frames: int = 0
    horizon_blend_end: float = 0.625
    map_query_radius_m: float = 100.0
    left_lateral_min_m: float = 2.5
    left_lateral_max_m: float = 6.0
    heading_tolerance_deg: float = 60.0
    anchor_longitudinal_max_m: float = 15.0
    join_lateral_tolerance_m: float = 1.25
    merge_resolution_m: float = 0.05

    def __post_init__(self) -> None:
        if self.temporal_radius < 0:
            raise ValueError("temporal_radius 不能为负数")
        if self.lift_start_frame < 0:
            raise ValueError("lift_start_frame 不能为负数")
        if self.lift_transition_frames < 0:
            raise ValueError("lift_transition_frames 不能为负数")
        if not 0.0 < self.horizon_blend_end <= 1.0:
            raise ValueError("horizon_blend_end 必须在 (0, 1] 范围内")
        if self.map_query_radius_m <= 0.0:
            raise ValueError("map_query_radius_m 必须大于 0")
        if not 0.0 < self.left_lateral_min_m < self.left_lateral_max_m:
            raise ValueError("左侧车道横向范围无效")
        if not 0.0 < self.heading_tolerance_deg < 90.0:
            raise ValueError("heading_tolerance_deg 必须在 (0, 90) 范围内")
        if self.anchor_longitudinal_max_m < 0.0:
            raise ValueError("anchor_longitudinal_max_m 不能为负数")
        if self.join_lateral_tolerance_m <= 0.0:
            raise ValueError("join_lateral_tolerance_m 必须大于 0")
        if self.merge_resolution_m <= 0.0:
            raise ValueError("merge_resolution_m 必须大于 0")


@dataclass(frozen=True)
class LeftLaneCenterline:
    """当前自车坐标系下拼接后的左侧相邻车道中心线。"""

    xy: npt.NDArray[np.float64]
    object_ids: tuple[str, ...]
    reference_lateral_m: float


@dataclass(frozen=True)
class _CandidatePath:
    xy: npt.NDArray[np.float64]
    object_id: str
    lateral_m: float
    nearest_longitudinal_m: float


def _smootherstep(value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


def _temporal_smooth(
    values: npt.NDArray[np.float64], radius: int
) -> npt.NDArray[np.float64]:
    """先用中值滤波移除单帧尖峰，再用三点核消除折变。"""
    if radius == 0 or values.shape[0] <= 1:
        return values.copy()

    pad_width = [(radius, radius)] + [(0, 0)] * (values.ndim - 1)
    padded = np.pad(values, pad_width, mode="edge")
    window_size = radius * 2 + 1
    median = np.stack(
        [
            np.median(padded[index : index + window_size], axis=0)
            for index in range(values.shape[0])
        ]
    )
    padded_median = np.pad(
        median,
        [(1, 1)] + [(0, 0)] * (median.ndim - 1),
        mode="edge",
    )
    return (
        padded_median[:-2]
        + 2.0 * padded_median[1:-1]
        + padded_median[2:]
    ) / 4.0


def _global_to_local_xy(
    global_xy: npt.NDArray[np.float64],
    ego_pose: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    delta = global_xy - ego_pose[None, :2]
    cosine = math.cos(float(ego_pose[2]))
    sine = math.sin(float(ego_pose[2]))
    return np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
        )
    )


def _baseline_xy(map_object: Any) -> npt.NDArray[np.float64] | None:
    baseline_path = getattr(map_object, "baseline_path", None)
    if baseline_path is None:
        return None

    discrete_path = getattr(baseline_path, "discrete_path", None)
    if discrete_path:
        points = np.asarray(
            [[float(state.x), float(state.y)] for state in discrete_path],
            dtype=np.float64,
        )
    else:
        linestring = getattr(baseline_path, "linestring", None)
        if linestring is None or linestring.is_empty:
            return None
        points = np.asarray(linestring.coords, dtype=np.float64)[:, :2]

    if points.ndim != 2 or points.shape[0] < 2 or not np.isfinite(points).all():
        return None
    return points


def _longitudinal_alignment(local_xy: npt.NDArray[np.float64]) -> float:
    segments = np.diff(local_xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-6
    if not valid.any():
        return 0.0
    return float(np.abs(segments[valid, 0]).sum() / lengths[valid].sum())


def _lateral_at_zero(local_xy: npt.NDArray[np.float64]) -> float:
    order = np.argsort(local_xy[:, 0])
    sorted_xy = local_xy[order]
    if sorted_xy[0, 0] <= 0.0 <= sorted_xy[-1, 0]:
        return float(np.interp(0.0, sorted_xy[:, 0], sorted_xy[:, 1]))
    nearest_index = int(np.argmin(np.abs(sorted_xy[:, 0])))
    return float(sorted_xy[nearest_index, 1])


def _merge_points_by_x(
    paths: Sequence[npt.NDArray[np.float64]], resolution_m: float
) -> npt.NDArray[np.float64]:
    points = np.concatenate(paths, axis=0)
    order = np.argsort(points[:, 0])
    points = points[order]
    bin_ids = np.rint(points[:, 0] / resolution_m).astype(np.int64)

    merged_points: list[tuple[float, float]] = []
    for bin_id in np.unique(bin_ids):
        bin_points = points[bin_ids == bin_id]
        merged_points.append(
            (float(np.mean(bin_points[:, 0])), float(np.median(bin_points[:, 1])))
        )
    return np.asarray(merged_points, dtype=np.float64)


def extract_left_lane_centerline(
    map_api: Any,
    ego_pose: npt.ArrayLike,
    config: LaneCenterPolishConfig,
) -> LeftLaneCenterline | None:
    """查询并拼接当前自车左侧最近的平行车道中心线。"""
    if map_api is None:
        return None
    ego_pose_array = np.asarray(ego_pose, dtype=np.float64)
    if ego_pose_array.shape != (3,) or not np.isfinite(ego_pose_array).all():
        raise ValueError("ego_pose 必须是有限的 [x, y, heading]")

    layers = (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR)
    proximal = map_api.get_proximal_map_objects(
        point=Point2D(float(ego_pose_array[0]), float(ego_pose_array[1])),
        radius=config.map_query_radius_m,
        layers=layers,
    )
    min_alignment = math.cos(math.radians(config.heading_tolerance_deg))
    candidates: list[_CandidatePath] = []

    for layer in layers:
        for map_object in proximal.get(layer, []):
            global_xy = _baseline_xy(map_object)
            if global_xy is None:
                continue
            local_xy = _global_to_local_xy(global_xy, ego_pose_array)
            if _longitudinal_alignment(local_xy) < min_alignment:
                continue
            lateral_m = _lateral_at_zero(local_xy)
            if not config.left_lateral_min_m <= lateral_m <= config.left_lateral_max_m:
                continue
            candidates.append(
                _CandidatePath(
                    xy=local_xy,
                    object_id=str(getattr(map_object, "id", "<unknown>")),
                    lateral_m=lateral_m,
                    nearest_longitudinal_m=float(np.min(np.abs(local_xy[:, 0]))),
                )
            )

    if not candidates:
        return None

    anchor_candidates = [
        candidate
        for candidate in candidates
        if candidate.nearest_longitudinal_m <= config.anchor_longitudinal_max_m
    ]
    reference_pool = anchor_candidates or candidates
    reference = min(reference_pool, key=lambda candidate: candidate.lateral_m)
    selected = [
        candidate
        for candidate in candidates
        if abs(candidate.lateral_m - reference.lateral_m)
        <= config.join_lateral_tolerance_m
    ]
    merged_xy = _merge_points_by_x(
        [candidate.xy for candidate in selected], config.merge_resolution_m
    )
    if merged_xy.shape[0] < 2 or np.ptp(merged_xy[:, 0]) <= 1e-3:
        return None

    return LeftLaneCenterline(
        xy=merged_xy,
        object_ids=tuple(candidate.object_id for candidate in selected),
        reference_lateral_m=reference.lateral_m,
    )


def _interpolate_centerline_y(
    centerline_xy: npt.ArrayLike,
    query_x: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    centerline = np.asarray(centerline_xy, dtype=np.float64)
    if centerline.ndim != 2 or centerline.shape[0] < 2 or centerline.shape[1] != 2:
        raise ValueError("centerline 必须具有 [point, 2] 形状且至少包含两个点")
    if not np.isfinite(centerline).all():
        raise ValueError("centerline 包含非有限数值")

    order = np.argsort(centerline[:, 0])
    sorted_centerline = centerline[order]
    unique_x, inverse = np.unique(sorted_centerline[:, 0], return_inverse=True)
    if unique_x.size < 2:
        raise ValueError("centerline 的纵向坐标必须覆盖至少两个不同位置")
    unique_y = np.asarray(
        [
            np.median(sorted_centerline[inverse == index, 1])
            for index in range(unique_x.size)
        ],
        dtype=np.float64,
    )
    return np.interp(query_x, unique_x, unique_y)


def _fill_missing_centerlines(
    target_y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    frame_indices = np.arange(target_y.shape[0], dtype=np.float64)
    filled = target_y.copy()
    for horizon_index in range(target_y.shape[1]):
        valid = np.isfinite(target_y[:, horizon_index])
        if not valid.any():
            raise ValueError("所有帧都缺少可用的左侧车道中心线")
        filled[:, horizon_index] = np.interp(
            frame_indices,
            frame_indices[valid],
            target_y[valid, horizon_index],
        )
    return filled


def _trajectory_headings(
    xy: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    frame_count, horizon, _ = xy.shape
    origins = np.zeros((frame_count, 1, 2), dtype=np.float64)
    extended = np.concatenate((origins, xy), axis=1)
    tangents = np.empty_like(xy)
    if horizon == 1:
        tangents[:, 0] = extended[:, 1] - extended[:, 0]
    else:
        tangents[:, :-1] = extended[:, 2:] - extended[:, :-2]
        tangents[:, -1] = extended[:, -1] - extended[:, -2]
    return np.arctan2(tangents[..., 1], tangents[..., 0])


def polish_trajectories_to_left_lane_center(
    poses: npt.ArrayLike,
    centerlines: Sequence[LeftLaneCenterline | npt.ArrayLike | None],
    config: LaneCenterPolishConfig,
) -> npt.NDArray[np.float32]:
    """平滑轨迹序列，并在指定帧之后渐进贴合左侧车道中心线。"""
    pose_array = np.asarray(poses, dtype=np.float64)
    if pose_array.ndim != 3 or pose_array.shape[-1] != 3:
        raise ValueError("poses 必须具有 [frame, horizon, 3] 形状")
    if pose_array.shape[0] == 0 or pose_array.shape[1] == 0:
        raise ValueError("poses 的 frame 和 horizon 维度不能为空")
    if not np.isfinite(pose_array).all():
        raise ValueError("poses 包含非有限数值")
    if len(centerlines) != pose_array.shape[0]:
        raise ValueError("centerlines 数量必须与 poses 帧数一致")
    if config.lift_start_frame >= pose_array.shape[0]:
        raise ValueError("lift_start_frame 超出轨迹帧范围")

    smoothed_xy = _temporal_smooth(
        pose_array[..., :2], config.temporal_radius
    )
    target_y = np.full(smoothed_xy.shape[:2], np.nan, dtype=np.float64)
    for frame_index, centerline in enumerate(centerlines):
        if centerline is None:
            continue
        centerline_xy = (
            centerline.xy
            if isinstance(centerline, LeftLaneCenterline)
            else centerline
        )
        target_y[frame_index] = _interpolate_centerline_y(
            centerline_xy, smoothed_xy[frame_index, :, 0]
        )

    target_y = _fill_missing_centerlines(target_y)
    target_y = _temporal_smooth(target_y, config.temporal_radius)

    frame_indices = np.arange(pose_array.shape[0], dtype=np.float64)
    if config.lift_transition_frames == 0:
        frame_progress = (frame_indices >= config.lift_start_frame).astype(
            np.float64
        )
    else:
        frame_progress = (
            frame_indices - config.lift_start_frame
        ) / config.lift_transition_frames
    frame_weights = _smootherstep(frame_progress)

    horizon_progress = (
        np.arange(1, pose_array.shape[1] + 1, dtype=np.float64)
        / pose_array.shape[1]
        / config.horizon_blend_end
    )
    horizon_weights = _smootherstep(horizon_progress)
    blend_weights = frame_weights[:, None] * horizon_weights[None, :]

    polished_xy = smoothed_xy.copy()
    polished_xy[..., 1] = (
        (1.0 - blend_weights) * smoothed_xy[..., 1]
        + blend_weights * target_y
    )
    polished = np.empty_like(pose_array)
    polished[..., :2] = polished_xy
    polished[..., 2] = _trajectory_headings(polished_xy)
    return polished.astype(np.float32)
