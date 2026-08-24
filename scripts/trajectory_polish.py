"""用于离线可视化的轨迹时序平滑与横向拉起。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TrajectoryPolishConfig:
    """轨迹修饰参数；正横向偏移表示自车左侧。"""

    temporal_radius: int = 2
    left_offset_m: float = 0.0
    lift_start_frame: int = 0
    lift_transition_frames: int = 0
    horizon_ramp_start: float = 0.125

    def __post_init__(self) -> None:
        if self.temporal_radius < 0:
            raise ValueError("temporal_radius 不能为负数")
        if self.left_offset_m < 0:
            raise ValueError("left_offset_m 不能为负数")
        if self.lift_start_frame < 0:
            raise ValueError("lift_start_frame 不能为负数")
        if self.lift_transition_frames < 0:
            raise ValueError("lift_transition_frames 不能为负数")
        if not 0.0 <= self.horizon_ramp_start < 1.0:
            raise ValueError("horizon_ramp_start 必须在 [0, 1) 范围内")


def _smoothstep(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


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
        padded_median[:-2] + 2.0 * padded_median[1:-1] + padded_median[2:]
    ) / 4.0


def _segment_headings(xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    origins = np.zeros((xy.shape[0], 1, 2), dtype=xy.dtype)
    segments = np.diff(np.concatenate([origins, xy], axis=1), axis=1)
    return np.arctan2(segments[..., 1], segments[..., 0])


def _wrap_angle(angle: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def polish_trajectory_poses(
    poses: npt.ArrayLike,
    config: TrajectoryPolishConfig,
) -> npt.NDArray[np.float32]:
    """修饰形状为 ``[frame, horizon, (x, y, heading)]`` 的局部轨迹序列。"""
    pose_array = np.asarray(poses)
    if pose_array.ndim != 3 or pose_array.shape[-1] != 3:
        raise ValueError("poses 必须具有 [frame, horizon, 3] 形状")
    if pose_array.shape[0] == 0 or pose_array.shape[1] == 0:
        raise ValueError("poses 的 frame 和 horizon 维度不能为空")
    if not np.isfinite(pose_array).all():
        raise ValueError("poses 包含非有限数值")
    if config.left_offset_m > 0 and config.lift_start_frame >= pose_array.shape[0]:
        raise ValueError("lift_start_frame 超出轨迹帧范围")

    polished = pose_array.astype(np.float64, copy=True)
    polished[..., :2] = _temporal_smooth(polished[..., :2], config.temporal_radius)
    unwrapped_headings = np.unwrap(polished[..., 2], axis=0)
    polished[..., 2] = _temporal_smooth(unwrapped_headings, config.temporal_radius)

    if config.left_offset_m > 0:
        frame_indices = np.arange(polished.shape[0], dtype=np.float64)
        if config.lift_transition_frames == 0:
            frame_progress = (frame_indices >= config.lift_start_frame).astype(np.float64)
        else:
            frame_progress = (
                frame_indices - config.lift_start_frame
            ) / config.lift_transition_frames
        frame_weights = _smoothstep(frame_progress)

        horizon_progress = (
            np.arange(1, polished.shape[1] + 1, dtype=np.float64) / polished.shape[1]
        )
        horizon_progress = (
            horizon_progress - config.horizon_ramp_start
        ) / (1.0 - config.horizon_ramp_start)
        horizon_weights = _smoothstep(horizon_progress)

        base_xy = polished[..., :2].copy()
        polished[..., 1] += (
            config.left_offset_m * frame_weights[:, None] * horizon_weights[None, :]
        )

        base_headings = _segment_headings(base_xy)
        lifted_headings = _segment_headings(polished[..., :2])
        polished[..., 2] += _wrap_angle(lifted_headings - base_headings)

    polished[..., 2] = _wrap_angle(polished[..., 2])
    return polished.astype(np.float32)
