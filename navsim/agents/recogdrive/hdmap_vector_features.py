from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from shapely.geometry import LineString

from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


logger = logging.getLogger(__name__)


HDMAP_VECTOR_FEATURE_NAME = "hdmap_vector_feature_v2"
INVALID_STOP_LINE_IDS = {0, (1 << 64) - 1}
STOP_LINE_ROUTE_BUFFER_M = 1.0

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
    "map_polygon_control_type",
    "map_polygon_tl_direction",
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
    STOP_LINE = 3


class HDMapRouteRelation:
    OFF_ROUTE = 0
    ON_ROUTE = 1
    UNKNOWN = 2


class HDMapTrafficLightStatus:
    GREEN = 0
    RED = 1
    YELLOW = 2
    UNKNOWN = 3
    NONE = 4


class HDMapControlType:
    NONE = 0
    TRAFFIC_LIGHT = 1
    UNKNOWN = 2


class HDMapTrafficLightDirection:
    NONE = 0
    STRAIGHT = 1
    LEFT = 2
    RIGHT = 3
    U_TURN = 4
    UNKNOWN = 5


class HDMapVectorFeatureBuilder(AbstractFeatureBuilder):
    """Build lane and traffic-control map tokens for ReCogDrive."""

    def __init__(
        self,
        max_map_tokens: int = 128,
        map_sample_points: int = 20,
        radius: float = 120.0,
        strict: bool = False,
        max_control_tokens: int = 8,
        control_range_x: Sequence[float] = (-10.0, 160.0),
        control_range_y: Sequence[float] = (-60.0, 60.0),
    ) -> None:
        super().__init__()
        self.max_map_tokens = int(max_map_tokens)
        self.map_sample_points = int(map_sample_points)
        self.radius = float(radius)
        self.max_control_tokens = int(max_control_tokens)
        self.control_range_x = self._validate_range(control_range_x, "control_range_x")
        self.control_range_y = self._validate_range(control_range_y, "control_range_y")
        self.strict = bool(strict)
        if self.max_map_tokens <= 0:
            raise ValueError("max_map_tokens must be positive")
        if self.map_sample_points <= 0:
            raise ValueError("map_sample_points must be positive")
        if self.max_control_tokens < 0:
            raise ValueError("max_control_tokens must be non-negative")

    @staticmethod
    def _validate_range(value: Sequence[float], name: str) -> Tuple[float, float]:
        values = tuple(float(item) for item in value)
        if len(values) != 2 or values[0] >= values[1]:
            raise ValueError(f"{name} must contain two increasing values")
        return values

    def get_unique_name(self) -> str:
        return HDMAP_VECTOR_FEATURE_NAME

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:  # noqa: ARG002
        """Return no-map fallback when only AgentInput is available."""
        del agent_input  # intentionally unused: this path has no map API access
        if self.strict:
            raise RuntimeError("HDMapVectorFeatureBuilder requires Scene.map_api; got AgentInput-only path.")
        return self._empty_feature_dict()

    def compute_features_from_scene(self, scene: Scene) -> Dict[str, torch.Tensor]:
        frame_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[frame_idx]
        origin = StateSE2(*current_frame.ego_status.ego_pose)

        map_api = scene.map_api
        proximal_objects: Dict[SemanticMapLayer, Sequence[Any]] = {
            SemanticMapLayer.LANE: [],
            SemanticMapLayer.LANE_CONNECTOR: [],
        }
        if map_api is None:
            if self.strict:
                raise RuntimeError("HDMapVectorFeatureBuilder requires scene.map_api, but it is None.")
            logger.debug(
                "scene.map_api is None; lane tokens are unavailable, but PKL control tokens remain usable."
            )
        else:
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
                logger.warning(
                    "Failed to query HDMap lane features; continuing with PKL traffic controls only: %s",
                    exc,
                )

        route_ids = {str(route_id) for route_id in (current_frame.roadblock_ids or [])}
        lane_candidates = []
        lane_candidates.extend(
            self._build_map_object_candidate(
                map_object=obj,
                origin=origin,
                route_ids=route_ids,
                polygon_type=HDMapPolygonType.LANE,
            )
            for obj in proximal_objects.get(SemanticMapLayer.LANE, [])
        )
        lane_candidates.extend(
            self._build_map_object_candidate(
                map_object=obj,
                origin=origin,
                route_ids=route_ids,
                polygon_type=HDMapPolygonType.LANE_CONNECTOR,
            )
            for obj in proximal_objects.get(SemanticMapLayer.LANE_CONNECTOR, [])
        )
        lane_candidates = [candidate for candidate in lane_candidates if candidate is not None]
        lane_candidates.sort(key=lambda item: float(item["map_polygon_center"][:2].norm().item()))

        route_map_objects = list(proximal_objects.get(SemanticMapLayer.LANE, []))
        route_map_objects.extend(proximal_objects.get(SemanticMapLayer.LANE_CONNECTOR, []))
        control_candidates = self._build_traffic_control_candidates(
            traffic_light_states=getattr(current_frame, "traffic_light_states", None) or [],
            origin=origin,
            route_ids=route_ids,
            route_map_objects=route_map_objects,
        )
        control_candidates.sort(key=lambda item: float(item["map_polygon_center"][:2].norm().item()))
        control_limit = min(self.max_control_tokens, self.max_map_tokens)
        selected_candidates = list(control_candidates[:control_limit])
        remaining_slots = self.max_map_tokens - len(selected_candidates)
        selected_candidates.extend(lane_candidates[:remaining_slots])

        features = self._empty_feature_dict()
        for index, candidate in enumerate(selected_candidates):
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
            "map_polygon_on_route": torch.full(
                (m,), HDMapRouteRelation.UNKNOWN, dtype=torch.long
            ),
            "map_polygon_tl_status": torch.full((m,), HDMapTrafficLightStatus.NONE, dtype=torch.long),
            "map_polygon_control_type": torch.full((m,), HDMapControlType.NONE, dtype=torch.long),
            "map_polygon_tl_direction": torch.full((m,), HDMapTrafficLightDirection.NONE, dtype=torch.long),
            "map_polygon_has_speed_limit": torch.zeros((m,), dtype=torch.bool),
            "map_polygon_speed_limit": torch.zeros((m,), dtype=torch.float32),
            "map_point_valid_mask": torch.zeros((m, p), dtype=torch.float32),
            "map_valid_mask": torch.zeros((m,), dtype=torch.float32),
        }

    def _build_traffic_control_candidates(
        self,
        traffic_light_states: Sequence[Dict[str, Any]],
        origin: StateSE2,
        route_ids: set[str],
        route_map_objects: Sequence[Any],
    ) -> List[Dict[str, torch.Tensor]]:
        candidates: List[Dict[str, torch.Tensor]] = []
        seen_keys = set()
        for state_entry in traffic_light_states:
            if not isinstance(state_entry, dict):
                continue
            stop_line_id = self._valid_stop_line_id(state_entry.get("stop_line_id"))
            if stop_line_id is None:
                continue
            direction = self._traffic_light_direction(state_entry.get("direction"))
            dedupe_key = (stop_line_id, direction)
            if dedupe_key in seen_keys:
                continue
            points = self._valid_stop_line_points(state_entry.get("position"))
            if points is None:
                continue

            sampled_global = self._resample_states(self._states_from_points(points), self.map_sample_points + 1)
            local_states = self._to_local_states(origin, sampled_global)
            local_xy = local_states[:, :2]
            mid_index = min(len(local_xy) - 2, max(0, len(local_xy) // 2))
            center_xy = local_xy[mid_index]
            if not self._is_control_in_range(center_xy):
                continue

            seen_keys.add(dedupe_key)
            route_relation = self._control_route_relation(points, route_ids, route_map_objects)
            candidates.append(
                self._build_traffic_control_candidate(
                    local_states=local_states,
                    direction=direction,
                    state_name=str(state_entry.get("state", "unknown")),
                    route_relation=route_relation,
                )
            )
        return candidates

    def _build_traffic_control_candidate(
        self,
        local_states: np.ndarray,
        direction: int,
        state_name: str,
        route_relation: int,
    ) -> Dict[str, torch.Tensor]:
        xy = local_states[:, :2]
        vectors = xy[1:] - xy[:-1]
        orientations = np.arctan2(vectors[:, 1], vectors[:, 0]).astype(np.float32)
        point_position = torch.from_numpy(np.repeat(xy[:-1][None, ...], 3, axis=0).astype(np.float32))
        point_vector = torch.from_numpy(np.repeat(vectors[None, ...], 3, axis=0).astype(np.float32))
        point_orientation = torch.from_numpy(np.repeat(orientations[None, ...], 3, axis=0))
        mid_index = min(len(local_states) - 2, max(0, len(local_states) // 2))
        center_xy = local_states[mid_index, :2]
        center_next_xy = local_states[mid_index + 1, :2]
        center_heading = math.atan2(
            float(center_next_xy[1] - center_xy[1]),
            float(center_next_xy[0] - center_xy[0]),
        )
        return {
            "map_point_position": point_position,
            "map_point_vector": point_vector,
            "map_point_orientation": point_orientation,
            "map_polygon_center": torch.tensor(
                [float(center_xy[0]), float(center_xy[1]), center_heading], dtype=torch.float32
            ),
            "map_polygon_type": torch.tensor(HDMapPolygonType.STOP_LINE, dtype=torch.long),
            "map_polygon_on_route": torch.tensor(route_relation, dtype=torch.long),
            "map_polygon_tl_status": torch.tensor(self._traffic_light_status(state_name), dtype=torch.long),
            "map_polygon_control_type": torch.tensor(HDMapControlType.TRAFFIC_LIGHT, dtype=torch.long),
            "map_polygon_tl_direction": torch.tensor(direction, dtype=torch.long),
            "map_polygon_has_speed_limit": torch.tensor(False, dtype=torch.bool),
            "map_polygon_speed_limit": torch.tensor(0.0, dtype=torch.float32),
            "map_point_valid_mask": torch.ones((self.map_sample_points,), dtype=torch.float32),
        }

    def _build_map_object_candidate(
        self,
        map_object: Any,
        origin: StateSE2,
        route_ids: set[str],
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
        route_relation = self._route_relation(map_object, roadblock_id, route_ids)
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
            "map_polygon_on_route": torch.tensor(route_relation, dtype=torch.long),
            "map_polygon_tl_status": torch.tensor(
                HDMapTrafficLightStatus.NONE, dtype=torch.long
            ),
            "map_polygon_control_type": torch.tensor(HDMapControlType.NONE, dtype=torch.long),
            "map_polygon_tl_direction": torch.tensor(HDMapTrafficLightDirection.NONE, dtype=torch.long),
            "map_polygon_has_speed_limit": torch.tensor(bool(has_speed_limit), dtype=torch.bool),
            "map_polygon_speed_limit": torch.tensor(float(speed_limit or 0.0), dtype=torch.float32),
            "map_point_valid_mask": path_valid_mask,
        }

    @staticmethod
    def _valid_stop_line_id(value: Any) -> Optional[int]:
        try:
            stop_line_id = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if stop_line_id in INVALID_STOP_LINE_IDS or stop_line_id < 0:
            return None
        return stop_line_id

    @staticmethod
    def _valid_stop_line_points(value: Any) -> Optional[List[List[float]]]:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return None
        points: List[List[float]] = []
        for raw_point in value:
            try:
                x = float(raw_point[0])
                y = float(raw_point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            points.append([x, y, 0.0])
        if len(points) < 2:
            return None
        if sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])) <= 1e-3:
            return None
        return points

    @staticmethod
    def _states_from_points(points: Sequence[Sequence[float]]) -> List[StateSE2]:
        states: List[StateSE2] = []
        for index, point in enumerate(points):
            previous = points[max(0, index - 1)]
            following = points[min(len(points) - 1, index + 1)]
            heading = math.atan2(following[1] - previous[1], following[0] - previous[0])
            states.append(StateSE2(float(point[0]), float(point[1]), float(heading)))
        return states

    @staticmethod
    def _traffic_light_status(value: Any) -> int:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if "green" in normalized or normalized == "gflash":
            return HDMapTrafficLightStatus.GREEN
        if "yellow" in normalized:
            return HDMapTrafficLightStatus.YELLOW
        if "red" in normalized or normalized in {"flash", "flashing"}:
            return HDMapTrafficLightStatus.RED
        return HDMapTrafficLightStatus.UNKNOWN

    @staticmethod
    def _traffic_light_direction(value: Any) -> int:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        direction_by_name = {
            "straight": HDMapTrafficLightDirection.STRAIGHT,
            "forward": HDMapTrafficLightDirection.STRAIGHT,
            "through": HDMapTrafficLightDirection.STRAIGHT,
            "left": HDMapTrafficLightDirection.LEFT,
            "left_turn": HDMapTrafficLightDirection.LEFT,
            "right": HDMapTrafficLightDirection.RIGHT,
            "right_turn": HDMapTrafficLightDirection.RIGHT,
            "u_turn": HDMapTrafficLightDirection.U_TURN,
            "uturn": HDMapTrafficLightDirection.U_TURN,
        }
        return direction_by_name.get(normalized, HDMapTrafficLightDirection.UNKNOWN)

    def _is_control_in_range(self, center_xy: Sequence[float]) -> bool:
        x, y = float(center_xy[0]), float(center_xy[1])
        return (
            math.isfinite(x)
            and math.isfinite(y)
            and self.control_range_x[0] <= x <= self.control_range_x[1]
            and self.control_range_y[0] <= y <= self.control_range_y[1]
        )

    def _control_route_relation(
        self,
        stop_line_points: Sequence[Sequence[float]],
        route_ids: set[str],
        route_map_objects: Sequence[Any],
    ) -> int:
        if not route_ids:
            return HDMapRouteRelation.UNKNOWN

        try:
            stop_line = LineString([(float(point[0]), float(point[1])) for point in stop_line_points])
        except Exception:
            return HDMapRouteRelation.UNKNOWN

        found_route_geometry = False
        for map_object in route_map_objects:
            roadblock_id = self._roadblock_id(map_object)
            if not self._is_on_route(map_object, roadblock_id, route_ids):
                continue
            centerline = self._extract_path_states(getattr(map_object, "baseline_path", None))
            if len(centerline) < 2:
                continue
            try:
                route_line = LineString([(state.x, state.y) for state in centerline])
            except Exception:
                continue
            if route_line.length <= 1e-3:
                continue
            found_route_geometry = True
            if stop_line.distance(route_line) <= STOP_LINE_ROUTE_BUFFER_M:
                return HDMapRouteRelation.ON_ROUTE

        if found_route_geometry:
            return HDMapRouteRelation.OFF_ROUTE
        return HDMapRouteRelation.UNKNOWN

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
    def _route_relation(cls, map_object: Any, roadblock_id: str, route_ids: set[str]) -> int:
        if not route_ids:
            return HDMapRouteRelation.UNKNOWN
        if cls._is_on_route(map_object, roadblock_id, route_ids):
            return HDMapRouteRelation.ON_ROUTE
        return HDMapRouteRelation.OFF_ROUTE

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

    @classmethod
    def _traffic_light_status_by_lane(cls, traffic_lights: Iterable[Any]) -> Dict[str, int]:
        """Parse the legacy NAVSIM lane-linked field whose tuple boolean means is_red."""
        status: Dict[str, int] = {}
        for item in traffic_lights or []:
            lane_id = ""
            raw_value: Any = None
            boolean_means_red = False
            if isinstance(item, dict):
                lane_id = str(
                    item.get("lane_connector_id") or item.get("lane_id") or item.get("id") or ""
                )
                if "is_red" in item:
                    raw_value = item["is_red"]
                    boolean_means_red = True
                elif "is_green" in item:
                    raw_value = item["is_green"]
                else:
                    raw_value = item.get("status")
            elif isinstance(item, (tuple, list)) and item:
                lane_id = str(item[0])
                raw_value = item[1] if len(item) > 1 else None
                boolean_means_red = True
            if not lane_id:
                continue

            if isinstance(raw_value, (bool, np.bool_)):
                is_active = bool(raw_value)
                if boolean_means_red:
                    status[lane_id] = (
                        HDMapTrafficLightStatus.RED
                        if is_active
                        else HDMapTrafficLightStatus.GREEN
                    )
                else:
                    status[lane_id] = (
                        HDMapTrafficLightStatus.GREEN
                        if is_active
                        else HDMapTrafficLightStatus.RED
                    )
            else:
                status[lane_id] = cls._traffic_light_status(raw_value)
        return status
