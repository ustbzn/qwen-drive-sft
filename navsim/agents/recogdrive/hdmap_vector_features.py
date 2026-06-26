from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


logger = logging.getLogger(__name__)


HDMAP_VECTOR_FEATURE_NAME = "hdmap_vector_feature"
HDMAP_VECTOR_FLOAT_KEYS = (
    "map_point_position",
    "map_point_vector",
    "map_point_orientation",
    "map_polygon_center",
    "map_polygon_speed_limit",
    "map_point_valid_mask",
    "map_valid_mask",
)
HDMAP_VECTOR_LONG_KEYS = (
    "map_polygon_type",
    "map_polygon_on_route",
    "map_polygon_tl_status",
)
HDMAP_VECTOR_BOOL_KEYS = ("map_polygon_has_speed_limit",)
HDMAP_VECTOR_BATCH_KEYS = HDMAP_VECTOR_FLOAT_KEYS + HDMAP_VECTOR_LONG_KEYS + HDMAP_VECTOR_BOOL_KEYS


def collate_hdmap_vector_features(features: Dict[str, torch.Tensor], features_list: Sequence[Dict[str, torch.Tensor]]) -> None:
    """Append optional HDMap vector tensors to a collated feature batch."""
    if not features_list:
        return
    for key in HDMAP_VECTOR_BATCH_KEYS:
        if all(key in sample for sample in features_list):
            features[key] = torch.stack([sample[key] for sample in features_list], dim=0).cpu()



class HDMapPolygonType:
    LANE = 0
    LANE_CONNECTOR = 1
    UNKNOWN = 2


class HDMapTrafficLightStatus:
    GREEN = 0
    RED = 1
    YELLOW = 2
    UNKNOWN = 3


class HDMapVectorFeatureBuilder(AbstractFeatureBuilder):
    """Build structured HDMap vectors from scene.map_api for ReCogDrive action side."""

    def __init__(
        self,
        max_map_tokens: int = 128,
        map_sample_points: int = 20,
        radius: float = 120.0,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.max_map_tokens = int(max_map_tokens)
        self.map_sample_points = int(map_sample_points)
        self.radius = float(radius)
        self.strict = bool(strict)

    def get_unique_name(self) -> str:
        return HDMAP_VECTOR_FEATURE_NAME

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:  # noqa: ARG002
        """Return no-map fallback when only AgentInput is available."""
        del agent_input  # intentionally unused: this path has no map API access
        if self.strict:
            raise RuntimeError("HDMapVectorFeatureBuilder requires Scene.map_api; got AgentInput-only path.")
        return self._empty_feature_dict()

    def compute_features_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        map_api = scene.map_api
        if map_api is None:
            if self.strict:
                raise RuntimeError("HDMapVectorFeatureBuilder requires scene.map_api, but it is None.")
            logger.debug("scene.map_api is None; returning zero HDMap vector features.")
            return self._empty_feature_dict()

        frame_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[frame_idx]
        origin = StateSE2(*current_frame.ego_status.ego_pose)

        try:
            if hasattr(map_api, "load"):
                map_api.load(origin.point)
            proximal_objects = map_api.get_proximal_map_objects(
                origin.point,
                self.radius,
                [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
            )
        except Exception as exc:
            if self.strict:
                raise RuntimeError("Failed to query HDMap vector features from scene.map_api.") from exc
            logger.warning("Failed to query HDMap vector features: %s", exc)
            return self._empty_feature_dict()

        route_ids = set(current_frame.roadblock_ids or [])
        traffic_light_status = self._traffic_light_status_by_lane(current_frame.traffic_lights)
        candidates = []
        candidates.extend(
            self._build_map_object_candidate(
                map_object=obj,
                origin=origin,
                route_ids=route_ids,
                traffic_light_status=traffic_light_status,
                polygon_type=HDMapPolygonType.LANE,
            )
            for obj in proximal_objects.get(SemanticMapLayer.LANE, [])
        )
        candidates.extend(
            self._build_map_object_candidate(
                map_object=obj,
                origin=origin,
                route_ids=route_ids,
                traffic_light_status=traffic_light_status,
                polygon_type=HDMapPolygonType.LANE_CONNECTOR,
            )
            for obj in proximal_objects.get(SemanticMapLayer.LANE_CONNECTOR, [])
        )
        candidates = [candidate for candidate in candidates if candidate is not None]
        candidates.sort(key=lambda item: float(item["map_polygon_center"][:2].norm().item()))

        features = self._empty_feature_dict()
        for index, candidate in enumerate(candidates[: self.max_map_tokens]):
            for key, value in candidate.items():
                features[key][index] = value
            features["map_valid_mask"][index] = 1.0
        return features

    def _empty_feature_dict(self) -> Dict[str, torch.Tensor]:
        m = self.max_map_tokens
        p = self.map_sample_points
        return {
            "map_point_position": torch.zeros((m, 3, p, 2), dtype=torch.float32),
            "map_point_vector": torch.zeros((m, 3, p, 2), dtype=torch.float32),
            "map_point_orientation": torch.zeros((m, 3, p), dtype=torch.float32),
            "map_polygon_center": torch.zeros((m, 3), dtype=torch.float32),
            "map_polygon_type": torch.full((m,), HDMapPolygonType.UNKNOWN, dtype=torch.long),
            "map_polygon_on_route": torch.zeros((m,), dtype=torch.long),
            "map_polygon_tl_status": torch.full((m,), HDMapTrafficLightStatus.UNKNOWN, dtype=torch.long),
            "map_polygon_has_speed_limit": torch.zeros((m,), dtype=torch.bool),
            "map_polygon_speed_limit": torch.zeros((m,), dtype=torch.float32),
            "map_point_valid_mask": torch.zeros((m, p), dtype=torch.float32),
            "map_valid_mask": torch.zeros((m,), dtype=torch.float32),
        }

    def _build_map_object_candidate(
        self,
        map_object: Any,
        origin: StateSE2,
        route_ids: set[str],
        traffic_light_status: Dict[str, int],
        polygon_type: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        centerline = self._extract_path_states(getattr(map_object, "baseline_path", None))
        if len(centerline) < 2:
            return None

        left_boundary = self._extract_path_states(getattr(map_object, "left_boundary", None))
        right_boundary = self._extract_path_states(getattr(map_object, "right_boundary", None))
        if (
            len(left_boundary) < 2
            or len(right_boundary) < 2
            or self._paths_are_same(centerline, left_boundary)
            or self._paths_are_same(centerline, right_boundary)
        ):
            left_boundary, right_boundary = self._fallback_boundaries(centerline)

        paths = (centerline, left_boundary, right_boundary)
        point_position = torch.zeros((3, self.map_sample_points, 2), dtype=torch.float32)
        point_vector = torch.zeros((3, self.map_sample_points, 2), dtype=torch.float32)
        point_orientation = torch.zeros((3, self.map_sample_points), dtype=torch.float32)
        path_valid_mask = torch.zeros((self.map_sample_points,), dtype=torch.float32)

        for path_index, path in enumerate(paths):
            sampled_global = self._resample_states(path, self.map_sample_points + 1)
            local_states = self._to_local_states(origin, sampled_global)
            xy = local_states[:, :2]
            vectors = xy[1:] - xy[:-1]
            point_position[path_index] = torch.from_numpy(xy[:-1].astype(np.float32))
            point_vector[path_index] = torch.from_numpy(vectors.astype(np.float32))
            point_orientation[path_index] = torch.from_numpy(
                np.arctan2(vectors[:, 1], vectors[:, 0]).astype(np.float32)
            )
            if path_index == 0:
                path_valid_mask[:] = 1.0

        sampled_center_global = self._resample_states(centerline, self.map_sample_points + 1)
        local_center = self._to_local_states(origin, sampled_center_global)
        mid_index = min(len(local_center) - 2, max(0, len(local_center) // 2))
        center_xy = local_center[mid_index, :2]
        center_next_xy = local_center[mid_index + 1, :2]
        center_heading = math.atan2(
            float(center_next_xy[1] - center_xy[1]),
            float(center_next_xy[0] - center_xy[0]),
        )

        roadblock_id = self._roadblock_id(map_object)
        on_route = self._is_on_route(map_object, roadblock_id, route_ids)
        speed_limit = getattr(map_object, "speed_limit_mps", None)
        has_speed_limit = speed_limit is not None and float(speed_limit) > 0.0

        return {
            "map_point_position": point_position,
            "map_point_vector": point_vector,
            "map_point_orientation": point_orientation,
            "map_polygon_center": torch.tensor(
                [float(center_xy[0]), float(center_xy[1]), center_heading],
                dtype=torch.float32,
            ),
            "map_polygon_type": torch.tensor(int(polygon_type), dtype=torch.long),
            "map_polygon_on_route": torch.tensor(int(on_route), dtype=torch.long),
            "map_polygon_tl_status": torch.tensor(
                traffic_light_status.get(str(getattr(map_object, "id", "")), HDMapTrafficLightStatus.UNKNOWN),
                dtype=torch.long,
            ),
            "map_polygon_has_speed_limit": torch.tensor(bool(has_speed_limit), dtype=torch.bool),
            "map_polygon_speed_limit": torch.tensor(float(speed_limit or 0.0), dtype=torch.float32),
            "map_point_valid_mask": path_valid_mask,
        }

    @staticmethod
    def _extract_path_states(path_object: Any) -> List[StateSE2]:
        if path_object is None:
            return []
        discrete_path = getattr(path_object, "discrete_path", None)
        if discrete_path is None:
            return []
        states = []
        for state in discrete_path:
            if hasattr(state, "x") and hasattr(state, "y"):
                states.append(StateSE2(float(state.x), float(state.y), float(getattr(state, "heading", 0.0))))
        return states

    @staticmethod
    def _fallback_boundaries(centerline: Sequence[StateSE2], half_width: float = 1.75) -> Tuple[List[StateSE2], List[StateSE2]]:
        left: List[StateSE2] = []
        right: List[StateSE2] = []
        for state in centerline:
            normal_x = -math.sin(float(state.heading))
            normal_y = math.cos(float(state.heading))
            left.append(StateSE2(state.x + normal_x * half_width, state.y + normal_y * half_width, state.heading))
            right.append(StateSE2(state.x - normal_x * half_width, state.y - normal_y * half_width, state.heading))
        return left, right

    @staticmethod
    def _paths_are_same(first_path: Sequence[StateSE2], second_path: Sequence[StateSE2], tolerance: float = 1e-3) -> bool:
        if len(first_path) != len(second_path) or not first_path:
            return False
        for first, second in zip(first_path, second_path):
            if math.hypot(first.x - second.x, first.y - second.y) > tolerance:
                return False
        return True

    @staticmethod
    def _path_lengths(states: Sequence[StateSE2]) -> np.ndarray:
        lengths = [0.0]
        for prev, curr in zip(states[:-1], states[1:]):
            lengths.append(lengths[-1] + math.hypot(curr.x - prev.x, curr.y - prev.y))
        return np.asarray(lengths, dtype=np.float64)

    def _resample_states(self, states: Sequence[StateSE2], num_points: int) -> np.ndarray:
        if not states:
            return np.zeros((num_points, 3), dtype=np.float64)
        if len(states) == 1:
            state = states[0]
            return np.repeat(np.asarray([[state.x, state.y, state.heading]], dtype=np.float64), num_points, axis=0)

        arc_lengths = self._path_lengths(states)
        total_length = float(arc_lengths[-1])
        if total_length <= 1e-6:
            state = states[0]
            return np.repeat(np.asarray([[state.x, state.y, state.heading]], dtype=np.float64), num_points, axis=0)

        target_lengths = np.linspace(0.0, total_length, num_points, dtype=np.float64)
        source_xy = np.asarray([[state.x, state.y] for state in states], dtype=np.float64)
        sampled_x = np.interp(target_lengths, arc_lengths, source_xy[:, 0])
        sampled_y = np.interp(target_lengths, arc_lengths, source_xy[:, 1])
        headings = np.zeros((num_points,), dtype=np.float64)
        if num_points > 1:
            dx = np.gradient(sampled_x)
            dy = np.gradient(sampled_y)
            headings = np.arctan2(dy, dx)
        return np.stack([sampled_x, sampled_y, headings], axis=-1)

    @staticmethod
    def _to_local_states(origin: StateSE2, states: np.ndarray) -> np.ndarray:
        return convert_absolute_to_relative_se2_array(origin, states.astype(np.float64))

    @staticmethod
    def _roadblock_id(map_object: Any) -> str:
        get_roadblock_id = getattr(map_object, "get_roadblock_id", None)
        if callable(get_roadblock_id):
            return str(get_roadblock_id())
        parent = getattr(map_object, "parent", None)
        if callable(parent):
            parent_object = parent()
            if parent_object is not None and hasattr(parent_object, "id"):
                return str(parent_object.id)
        return ""

    @classmethod
    def _is_on_route(cls, map_object: Any, roadblock_id: str, route_ids: set[str]) -> bool:
        if roadblock_id in route_ids:
            return True
        for edge_group_name in ("incoming_edges", "outgoing_edges"):
            edge_group = getattr(map_object, edge_group_name, [])
            for edge in edge_group:
                edge_roadblock_id = cls._roadblock_id(edge)
                if edge_roadblock_id in route_ids:
                    return True
        return False

    @staticmethod
    def _traffic_light_status_by_lane(traffic_lights: Iterable[Any]) -> Dict[str, int]:
        status: Dict[str, int] = {}
        for item in traffic_lights or []:
            lane_id: Optional[str] = None
            raw_value: Any = None
            if isinstance(item, dict):
                lane_id = str(item.get("lane_connector_id") or item.get("lane_id") or item.get("id") or "")
                raw_value = item.get("status", item.get("is_green", None))
            elif isinstance(item, (tuple, list)) and item:
                lane_id = str(item[0])
                raw_value = item[1] if len(item) > 1 else None
            if not lane_id:
                continue
            if isinstance(raw_value, bool):
                status[lane_id] = HDMapTrafficLightStatus.GREEN if raw_value else HDMapTrafficLightStatus.RED
            elif isinstance(raw_value, str):
                status_value = raw_value.lower()
                if "green" in status_value:
                    status[lane_id] = HDMapTrafficLightStatus.GREEN
                elif "yellow" in status_value:
                    status[lane_id] = HDMapTrafficLightStatus.YELLOW
                elif "red" in status_value:
                    status[lane_id] = HDMapTrafficLightStatus.RED
                else:
                    status[lane_id] = HDMapTrafficLightStatus.UNKNOWN
            else:
                status[lane_id] = HDMapTrafficLightStatus.UNKNOWN
        return status
