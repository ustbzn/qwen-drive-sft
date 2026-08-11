from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from navsim.common.dataclasses import NAVSIM_INTERVAL_LENGTH, Scene
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


SAFETY_TARGET_NAME = "recogdrive_safety_target_v1"
SAFETY_TARGET_KEYS = (
    "future_obj_boxes",
    "future_obj_valid_mask",
    "drivable_sdf",
    "drivable_sdf_origin",
    "drivable_sdf_resolution",
    "drivable_sdf_valid",
)

DEFAULT_SDF_X_RANGE = (-10.0, 80.0)
DEFAULT_SDF_Y_RANGE = (-40.0, 40.0)
DEFAULT_SDF_RESOLUTION = 0.5
DEFAULT_SDF_MAX_DISTANCE = 20.0

_DRIVABLE_LAYER_NAMES = (
    "ROADBLOCK",
    "ROADBLOCK_CONNECTOR",
    "INTERSECTION",
    "CARPARK_AREA",
    "LANE",
    "LANE_CONNECTOR",
)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _local_to_global_se2(local_states: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """Transform ego-local [x, y, heading] states into the global frame."""
    x, y, yaw = (float(value) for value in ego_pose[:3])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64
    )
    global_xy = local_states[:, :2] @ rotation.T + np.array([x, y])
    global_heading = _wrap_angle(local_states[:, 2] + yaw)
    return np.concatenate([global_xy, global_heading[:, None]], axis=1)


def _global_xy_to_local(global_xy: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """Transform global XY points into the current rear-axle frame."""
    delta = global_xy - np.asarray(ego_pose[:2], dtype=np.float64)
    yaw = float(ego_pose[2])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    local_x = cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1]
    local_y = -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1]
    return np.stack([local_x, local_y], axis=-1)


def _iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
        return
    for child in getattr(geometry, "geoms", ()):  # MultiPolygon/GeometryCollection
        yield from _iter_polygons(child)


class ReCogDriveSafetyTargetBuilder(AbstractTargetBuilder):
    """Build privileged future-object and drivable-SDF training targets.

    These tensors are target-only. They must never be registered as feature builders or
    concatenated into the planner's cross-attention context.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        max_future_objects: int = 64,
        sdf_x_range: Sequence[float] = DEFAULT_SDF_X_RANGE,
        sdf_y_range: Sequence[float] = DEFAULT_SDF_Y_RANGE,
        sdf_resolution: float = DEFAULT_SDF_RESOLUTION,
        sdf_max_distance: float = DEFAULT_SDF_MAX_DISTANCE,
        map_radius: float = 120.0,
    ) -> None:
        super().__init__()
        self.trajectory_sampling = trajectory_sampling
        self.max_future_objects = int(max_future_objects)
        self.sdf_x_range = self._validate_range(sdf_x_range, "sdf_x_range")
        self.sdf_y_range = self._validate_range(sdf_y_range, "sdf_y_range")
        self.sdf_resolution = float(sdf_resolution)
        self.sdf_max_distance = float(sdf_max_distance)
        self.map_radius = float(map_radius)

        if self.max_future_objects <= 0:
            raise ValueError("max_future_objects must be positive")
        if self.sdf_resolution <= 0.0:
            raise ValueError("sdf_resolution must be positive")
        if self.sdf_max_distance <= 0.0:
            raise ValueError("sdf_max_distance must be positive")

        self.sdf_width = self._grid_size(self.sdf_x_range)
        self.sdf_height = self._grid_size(self.sdf_y_range)

    @staticmethod
    def _validate_range(value: Sequence[float], name: str) -> Tuple[float, float]:
        values = tuple(float(item) for item in value)
        if len(values) != 2 or values[0] >= values[1]:
            raise ValueError(f"{name} must contain two increasing values")
        return values

    def _grid_size(self, value_range: Tuple[float, float]) -> int:
        span = (value_range[1] - value_range[0]) / self.sdf_resolution
        rounded_span = round(span)
        if not math.isclose(span, rounded_span, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("SDF ranges must be divisible by sdf_resolution")
        return int(rounded_span) + 1

    @staticmethod
    def _cache_float(value: float) -> str:
        return f"{float(value):g}".replace("-", "m").replace(".", "p")

    def get_unique_name(self) -> str:
        # Shape/range parameters are part of the cache identity to prevent stale SDF reads.
        x_range = "_".join(self._cache_float(value) for value in self.sdf_x_range)
        y_range = "_".join(self._cache_float(value) for value in self.sdf_y_range)
        resolution = self._cache_float(self.sdf_resolution)
        return (
            f"{SAFETY_TARGET_NAME}_n{self.max_future_objects}"
            f"_x{x_range}_y{y_range}_r{resolution}"
        )

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_boxes, future_valid = self._build_future_object_targets(scene)
        sdf, sdf_valid = self._build_drivable_sdf(scene)
        return {
            "future_obj_boxes": future_boxes,
            "future_obj_valid_mask": future_valid,
            # fp16 is sufficient at 0.5 m raster resolution and materially reduces cache size.
            "drivable_sdf": torch.from_numpy(sdf).unsqueeze(0).to(torch.float16),
            "drivable_sdf_origin": torch.tensor(
                [self.sdf_x_range[0], self.sdf_y_range[0]], dtype=torch.float32
            ),
            "drivable_sdf_resolution": torch.tensor(
                self.sdf_resolution, dtype=torch.float32
            ),
            "drivable_sdf_valid": torch.tensor(sdf_valid, dtype=torch.bool),
        }

    def _build_future_object_targets(
        self, scene: Scene
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        horizon = int(self.trajectory_sampling.num_poses)
        boxes_out = torch.zeros(
            (horizon, self.max_future_objects, 5), dtype=torch.float32
        )
        valid_out = torch.zeros(
            (horizon, self.max_future_objects), dtype=torch.bool
        )

        current_idx = int(scene.scene_metadata.num_history_frames) - 1
        current_pose = np.asarray(
            scene.frames[current_idx].ego_status.ego_pose, dtype=np.float64
        )

        for step in range(horizon):
            future_time = (step + 1) * float(self.trajectory_sampling.interval_length)
            frame_offset = int(round(future_time / NAVSIM_INTERVAL_LENGTH))
            frame_idx = current_idx + frame_offset
            if frame_idx >= len(scene.frames):
                continue

            future_frame = scene.frames[frame_idx]
            raw_boxes = np.asarray(future_frame.annotations.boxes, dtype=np.float64)
            if raw_boxes.ndim != 2 or raw_boxes.shape[0] == 0:
                continue

            local_states = raw_boxes[
                :,
                [
                    BoundingBoxIndex.X,
                    BoundingBoxIndex.Y,
                    BoundingBoxIndex.HEADING,
                ],
            ]
            lengths = raw_boxes[:, BoundingBoxIndex.LENGTH]
            widths = raw_boxes[:, BoundingBoxIndex.WIDTH]
            finite = np.isfinite(local_states).all(axis=1)
            finite &= np.isfinite(lengths) & np.isfinite(widths)
            finite &= (lengths > 0.0) & (widths > 0.0)
            if not finite.any():
                continue

            raw_boxes = raw_boxes[finite]
            local_states = local_states[finite]
            # Prioritize objects near the future ego pose before fixed-size truncation.
            order = np.argsort(np.linalg.norm(local_states[:, :2], axis=1))
            order = order[: self.max_future_objects]
            raw_boxes = raw_boxes[order]
            local_states = local_states[order]

            future_pose = np.asarray(future_frame.ego_status.ego_pose, dtype=np.float64)
            global_states = _local_to_global_se2(local_states, future_pose)
            current_local_states = convert_absolute_to_relative_se2_array(
                StateSE2(*current_pose), global_states
            )
            target = np.stack(
                [
                    current_local_states[:, 0],
                    current_local_states[:, 1],
                    current_local_states[:, 2],
                    raw_boxes[:, BoundingBoxIndex.LENGTH],
                    raw_boxes[:, BoundingBoxIndex.WIDTH],
                ],
                axis=-1,
            ).astype(np.float32)

            count = target.shape[0]
            boxes_out[step, :count] = torch.from_numpy(target)
            valid_out[step, :count] = True

        return boxes_out, valid_out

    def _build_drivable_sdf(self, scene: Scene) -> Tuple[np.ndarray, bool]:
        empty = np.full(
            (self.sdf_height, self.sdf_width),
            -self.sdf_max_distance,
            dtype=np.float32,
        )
        current_idx = int(scene.scene_metadata.num_history_frames) - 1
        current_frame = scene.frames[current_idx]
        current_pose = np.asarray(current_frame.ego_status.ego_pose, dtype=np.float64)
        polygons = self._query_drivable_polygons(scene, StateSE2(*current_pose))
        if not polygons:
            return empty, False

        try:
            geometry = unary_union(polygons)
        except Exception:
            return empty, False
        if geometry.is_empty:
            return empty, False

        mask = np.zeros((self.sdf_height, self.sdf_width), dtype=np.uint8)
        for polygon in _iter_polygons(geometry):
            self._fill_polygon(mask, polygon, current_pose, value=1)
        if not mask.any():
            return empty, False

        # A zero border makes positions beyond the cached grid explicitly non-drivable.
        padded = np.pad(mask, 1, mode="constant", constant_values=0)
        inside = distance_transform_edt(
            padded, sampling=self.sdf_resolution
        )[1:-1, 1:-1]
        outside = distance_transform_edt(
            1 - padded, sampling=self.sdf_resolution
        )[1:-1, 1:-1]
        raw_sdf = inside - outside
        raw_sdf -= np.sign(raw_sdf) * (0.5 * self.sdf_resolution)
        sdf = np.clip(
            raw_sdf,
            -self.sdf_max_distance,
            self.sdf_max_distance,
        ).astype(np.float32)
        return sdf, True

    def _query_drivable_polygons(
        self, scene: Scene, origin: StateSE2
    ) -> List[Polygon]:
        map_api = scene.map_api
        if map_api is None:
            return []
        layers = [
            getattr(SemanticMapLayer, name)
            for name in _DRIVABLE_LAYER_NAMES
            if hasattr(SemanticMapLayer, name)
        ]
        try:
            layer_objects = map_api.get_proximal_map_objects(
                origin.point, self.map_radius, layers
            )
        except Exception:
            layer_objects = {}
            for layer in layers:
                try:
                    result = map_api.get_proximal_map_objects(
                        origin.point, self.map_radius, [layer]
                    )
                    layer_objects[layer] = result.get(layer, [])
                except Exception:
                    layer_objects[layer] = []

        polygons: List[Polygon] = []
        for layer in layers:
            for map_object in layer_objects.get(layer, []):
                polygon = getattr(map_object, "polygon", None)
                if polygon is None or polygon.is_empty:
                    continue
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not polygon.is_empty:
                    polygons.extend(_iter_polygons(polygon))
        return polygons

    def _fill_polygon(
        self,
        mask: np.ndarray,
        polygon: Polygon,
        current_pose: np.ndarray,
        value: int,
    ) -> None:
        exterior = self._ring_to_pixels(polygon.exterior.coords, current_pose)
        if exterior.shape[0] >= 3:
            cv2.fillPoly(mask, [exterior], color=value)
        for interior in polygon.interiors:
            hole = self._ring_to_pixels(interior.coords, current_pose)
            if hole.shape[0] >= 3:
                cv2.fillPoly(mask, [hole], color=0)

    def _ring_to_pixels(
        self, coords, current_pose: np.ndarray
    ) -> np.ndarray:
        global_xy = np.asarray(coords, dtype=np.float64)[:, :2]
        local_xy = _global_xy_to_local(global_xy, current_pose)
        pixels = np.empty_like(local_xy)
        pixels[:, 0] = (
            local_xy[:, 0] - self.sdf_x_range[0]
        ) / self.sdf_resolution
        pixels[:, 1] = (
            local_xy[:, 1] - self.sdf_y_range[0]
        ) / self.sdf_resolution
        return np.rint(pixels).astype(np.int32)
