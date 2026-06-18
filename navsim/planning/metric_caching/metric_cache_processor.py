import logging
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Point

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer

from navsim.planning.scenario_builder.mogo_vehicle_parameters import MogoDynamicParameters
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import (
    PDMPath,
    PDMPathBoundaryPoint,
    PDMPathBoundaryProfile,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.metric_caching.metric_caching_utils import StateInterpolator, build_history_states_from_scenario


logger = logging.getLogger(__name__)

MOGO_BOUNDARY_ENDPOINT_EPS = 1e-2  # [m]
MOGO_LINK_DEFAULT_POINT_MIN_DISTANCE = 0.5  # [m]
MOGO_BOUNDARY_TYPE_DEFAULT = 0


def _extract_route_centerline_point(point: Any) -> Optional[Tuple[float, float, Optional[float]]]:
    """Extract x/y/optional-heading from a serialized route centerline point."""
    if isinstance(point, dict):
        x = point.get("x")
        y = point.get("y")
        heading = point.get("heading", point.get("yaw"))
    else:
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
            return None
        x = point[0]
        y = point[1]
        heading = point[2] if len(point) >= 3 else None

    if x is None or y is None:
        return None
    return float(x), float(y), None if heading is None else float(heading)


def _unwrap_route_centerline_points(route_centerline: Any) -> Any:
    """Unwrap common pkl serialization envelopes for route centerline points."""
    if isinstance(route_centerline, dict):
        for key in ("points", "states", "centerline", "route_centerline"):
            if key in route_centerline:
                return route_centerline[key]
    return route_centerline


def _route_centerline_to_pdm_path(route_centerline: Any) -> Optional[PDMPath]:
    """Convert serialized route centerline points to PDMPath."""
    if route_centerline is None:
        return None
    if isinstance(route_centerline, PDMPath):
        return route_centerline

    route_centerline = _unwrap_route_centerline_points(route_centerline)
    if not isinstance(route_centerline, (list, tuple, np.ndarray)) or len(route_centerline) < 2:
        return None

    parsed_points = [_extract_route_centerline_point(point) for point in route_centerline]
    parsed_points = [point for point in parsed_points if point is not None]
    deduped_points: List[Tuple[float, float, Optional[float]]] = []
    for point in parsed_points:
        if deduped_points and math.hypot(point[0] - deduped_points[-1][0], point[1] - deduped_points[-1][1]) <= 1e-6:
            continue
        deduped_points.append(point)
    parsed_points = deduped_points
    if len(parsed_points) < 2:
        return None

    states: List[StateSE2] = []
    for idx, (x, y, heading) in enumerate(parsed_points):
        if heading is None:
            if idx < len(parsed_points) - 1:
                next_x, next_y, _ = parsed_points[idx + 1]
                heading = math.atan2(next_y - y, next_x - x)
            else:
                prev_x, prev_y, _ = parsed_points[idx - 1]
                heading = math.atan2(y - prev_y, x - prev_x)
        states.append(StateSE2(x, y, heading))

    return PDMPath(states)


def _route_segment_get(segment: Any, key: str, default: Any = None) -> Any:
    """Read a route centerline segment field from dict or dataclass-like object."""
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


def _route_segment_type(segment: Any) -> str:
    """Return normalized route centerline segment type."""
    segment_type = _route_segment_get(segment, "segment_type", "")
    return str(getattr(segment_type, "value", segment_type))


def _mogo_route_lane_id(road_pkid: Any, section_id: Any, lane_id: Any) -> str:
    """Build the Mogo lane id used by MogoSQLiteMapAdapter."""
    return f"lane_{int(road_pkid)}_{int(section_id)}_{int(lane_id)}"


def _mogo_route_link_id(link_pkid: Any) -> str:
    """Build the Mogo link id used by MogoSQLiteMapAdapter."""
    return f"link_{int(link_pkid)}"


class MetricCacheProcessor:
    """Class for creating metric cache in NAVSIM."""

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
        use_dense_annotations: bool = False,
        history_interval_length: Optional[float] = None,
        ego_vehicle_parameters: Optional[VehicleParameters] = None,
        ego_dynamic_parameters: Optional[MogoDynamicParameters] = None,
    ):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation
        self._use_dense_annotations = use_dense_annotations
        self._history_interval_length = history_interval_length
        self._ego_vehicle_parameters = ego_vehicle_parameters
        self._ego_dynamic_parameters = ego_dynamic_parameters

        # TODO: Add to some config
        self._future_sampling = TrajectorySampling(num_poses=50, interval_length=0.1)
        self._proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self._map_radius = 100

        accel_max = ego_dynamic_parameters.accel_max if ego_dynamic_parameters else 1.5
        decel_max = ego_dynamic_parameters.decel_max if ego_dynamic_parameters else 3.0
        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=self._future_sampling,
            proposal_sampling=self._proposal_sampling,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
                fallback_target_velocity=15.0,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=accel_max,
                decel_max=decel_max,
            ),
            lateral_offsets=[-1.0, 1.0],
            map_radius=self._map_radius,
            vehicle_parameters=self._ego_vehicle_parameters,
        )

    def _get_metric_cache_centerline(self, scenario: AbstractScenario) -> PDMPath:
        """Prefer Mogo pkl route centerline when available, otherwise use PDM-Closed centerline."""
        route_centerline = getattr(scenario, "route_centerline", None)
        if route_centerline is None:
            return self._pdm_closed._centerline

        try:
            centerline = _route_centerline_to_pdm_path(route_centerline)
        except Exception as error:  # pragma: no cover - defensive fallback for data adaptation.
            logger.warning("Failed to parse pkl route_centerline, falling back to PDM-Closed centerline: %s", error)
            return self._pdm_closed._centerline

        if centerline is None:
            logger.warning("Invalid pkl route_centerline, falling back to PDM-Closed centerline.")
            return self._pdm_closed._centerline

        return centerline

    @staticmethod
    def _build_route_plan_from_centerline_segments(
        route_centerline_segments: Optional[Any],
        route_lane_dict: Optional[Dict[str, Any]],
    ) -> Optional[List[Any]]:
        """Map pkl route centerline segments back to Mogo map edges when possible."""
        if not route_centerline_segments or not route_lane_dict:
            return None

        route_plan: List[Any] = []
        for segment in route_centerline_segments:
            segment_type = _route_segment_type(segment)
            edge_id: Optional[str] = None
            if segment_type == "lane":
                road_pkid = _route_segment_get(segment, "road_pkid")
                section_id = _route_segment_get(segment, "section_id")
                lane_id = _route_segment_get(segment, "lane_id")
                if road_pkid is not None and section_id is not None and lane_id is not None:
                    edge_id = _mogo_route_lane_id(road_pkid, section_id, lane_id)
            elif segment_type == "link":
                link_pkid = _route_segment_get(segment, "link_pkid")
                if link_pkid is not None:
                    edge_id = _mogo_route_link_id(link_pkid)

            if edge_id is None:
                continue

            edge = route_lane_dict.get(edge_id)
            if edge is not None:
                route_plan.append(edge)

        return route_plan or None

    def _get_metric_cache_boundary_route_plan(
        self,
        scenario: AbstractScenario,
        centerline: PDMPath,
    ) -> Optional[List[Any]]:
        """Select the route plan that is aligned with the centerline saved in metric cache."""
        if centerline is self._pdm_closed._centerline:
            return getattr(self._pdm_closed, "_centerline_route_plan", None)

        route_plan = self._build_route_plan_from_centerline_segments(
            getattr(scenario, "route_centerline_segments", None),
            getattr(self._pdm_closed, "_route_lane_dict", None),
        )
        if route_plan is None:
            logger.warning("Missing or invalid pkl route_centerline_segments; skip boundary profile for pkl centerline.")
        return route_plan

    def _get_planner_inputs(self, scenario: AbstractScenario) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,
            ego_states=[scenario.initial_ego_state],
            observations=[scenario.initial_tracked_objects],
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
        )

        return planner_input, planner_initialization

    def _interpolate_gt_observation(self, scenario: AbstractScenario) -> PDMObservation:
        """
        Helper function to interpolate detections tracks to higher temporal resolution.
        :param scenario: scenario interface of nuPlan framework
        :return: observation object of PDM-Closed
        """

        # TODO: add to config
        state_size = 6  # (time, x, y, heading, velo_x, velo_y)

        time_horizon = 5.0  # [s]
        resolution_step = 0.5  # [s]
        interpolate_step = 0.1  # [s]

        scenario_step = scenario.database_interval  # [s]

        # sample detection tracks a 2Hz
        relative_time_s = np.arange(0, (time_horizon * 1 / resolution_step) + 1, 1, dtype=float) * resolution_step

        gt_indices = np.arange(0, int(time_horizon / scenario_step) + 1, int(resolution_step / scenario_step))
        gt_detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration) for iteration in gt_indices
        ]

        detection_tracks_states: Dict[str, Any] = {}
        unique_detection_tracks: Dict[str, Any] = {}

        for time_s, detection_track in zip(relative_time_s, gt_detection_tracks):

            for tracked_object in detection_track.tracked_objects:
                # log detection track
                token = tracked_object.track_token

                # extract states for dynamic and static objects
                tracked_state = np.zeros(state_size, dtype=np.float64)
                tracked_state[:4] = (
                    time_s,
                    tracked_object.center.x,
                    tracked_object.center.y,
                    tracked_object.center.heading,
                )

                if tracked_object.tracked_object_type in AGENT_TYPES:
                    # extract additional states for dynamic objects
                    tracked_state[4:] = (
                        tracked_object.velocity.x,
                        tracked_object.velocity.y,
                    )

                # found new object
                if token not in detection_tracks_states.keys():
                    detection_tracks_states[token] = [tracked_state]
                    unique_detection_tracks[token] = tracked_object

                # object already existed
                else:
                    detection_tracks_states[token].append(tracked_state)

        # create time interpolators
        detection_interpolators: Dict[str, StateInterpolator] = {}
        for token, states_list in detection_tracks_states.items():
            states = np.array(states_list, dtype=np.float64)
            detection_interpolators[token] = StateInterpolator(states)

        # interpolate at 10Hz
        interpolated_time_s = np.arange(0, int(time_horizon / interpolate_step) + 1, 1, dtype=float) * interpolate_step

        interpolated_detection_tracks = []
        for time_s in interpolated_time_s:
            interpolated_tracks = []
            for token, interpolator in detection_interpolators.items():
                initial_detection_track = unique_detection_tracks[token]
                interpolated_state = interpolator.interpolate(time_s)

                if interpolator.start_time == interpolator.end_time:
                    interpolated_tracks.append(initial_detection_track)

                elif interpolated_state is not None:

                    tracked_type = initial_detection_track.tracked_object_type
                    metadata = initial_detection_track.metadata  # copied since time stamp is ignored

                    oriented_box = OrientedBox(
                        StateSE2(*interpolated_state[:3]),
                        initial_detection_track.box.length,
                        initial_detection_track.box.width,
                        initial_detection_track.box.height,
                    )

                    if tracked_type in AGENT_TYPES:
                        velocity = StateVector2D(*interpolated_state[3:])

                        detection_track = Agent(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            velocity=velocity,
                            metadata=initial_detection_track.metadata,  # simply copy
                        )
                    else:
                        detection_track = StaticObject(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            metadata=metadata,
                        )

                    interpolated_tracks.append(detection_track)
            interpolated_detection_tracks.append(DetectionsTracks(TrackedObjects(interpolated_tracks)))

        # convert to pdm observation
        pdm_observation = PDMObservation(
            self._future_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
        )
        pdm_observation.update_detections_tracks(interpolated_detection_tracks)
        return pdm_observation

    def _build_dense_gt_observation(self, scenario: AbstractScenario) -> PDMObservation:
        """Build PDM observation directly from dense 10Hz annotations without interpolation."""
        expected_interval = self._future_sampling.interval_length
        if not np.isclose(scenario.database_interval, expected_interval):
            raise ValueError(
                "Dense annotations require scenario.database_interval to match PDM interval: "
                f"scenario={scenario.database_interval}, pdm={expected_interval}"
            )

        observation_samples = max(
            self._proposal_sampling.num_poses + int(1 / self._future_sampling.interval_length),
            self._future_sampling.num_poses,
        )
        required_iterations = observation_samples + 1
        if scenario.get_number_of_iterations() < required_iterations:
            raise ValueError(
                "Dense annotations require enough scene frames for PDM observation: "
                f"required={required_iterations}, available={scenario.get_number_of_iterations()}"
            )

        detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration)
            for iteration in range(required_iterations)
        ]

        pdm_observation = PDMObservation(
            self._future_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
        )
        pdm_observation.update_detections_tracks(detection_tracks)
        return pdm_observation

    def _build_gt_observation(self, scenario: AbstractScenario) -> PDMObservation:
        """Build observation using dense annotations when available, otherwise interpolate GT tracks."""
        if self._use_dense_annotations:
            return self._build_dense_gt_observation(scenario)
        return self._interpolate_gt_observation(scenario)

    @staticmethod
    def _edge_is_mogo_connector(edge: Any) -> bool:
        """Return whether a Mogo route-plan edge is a non-empty connector edge."""
        is_connector = getattr(edge, "is_connector", False)
        return bool(is_connector() if callable(is_connector) else is_connector)

    @staticmethod
    def _edge_boundary_xy(edge: Any, side: str) -> npt.NDArray[np.float64]:
        """Extract raw boundary xy points from a Mogo lane edge."""
        boundary = getattr(edge, f"{side}_boundary", None)
        if boundary is None:
            return np.empty((0, 2), dtype=np.float64)

        discrete_path = getattr(boundary, "discrete_path", [])
        if not discrete_path:
            return np.empty((0, 2), dtype=np.float64)

        return np.asarray([state.array[:2] for state in discrete_path], dtype=np.float64)

    @staticmethod
    def _edge_boundary_type(edge: Any, side: str) -> int:
        """Extract compact boundary type from a Mogo lane edge."""
        return int(getattr(edge, f"{side}_boundary_type", MOGO_BOUNDARY_TYPE_DEFAULT))

    @staticmethod
    def _edge_start_xy(edge: Any) -> Optional[npt.NDArray[np.float64]]:
        """Extract the start xy of an edge baseline path."""
        baseline_path = getattr(edge, "baseline_path", None)
        discrete_path = getattr(baseline_path, "discrete_path", []) if baseline_path is not None else []
        if not discrete_path:
            return None
        return np.asarray(discrete_path[0].array[:2], dtype=np.float64)

    @staticmethod
    def _project_boundary_xy(
        centerline: PDMPath,
        xy: npt.NDArray[np.float64],
        line_type: int,
    ) -> PDMPathBoundaryPoint:
        """Project a raw boundary point into the centerline s-l frame."""
        progress_s = float(centerline.project(Point(float(xy[0]), float(xy[1]))))
        pose = centerline.interpolate([progress_s], as_array=True)[0]
        dx = float(xy[0]) - float(pose[0])
        dy = float(xy[1]) - float(pose[1])
        lateral_offset = -np.sin(float(pose[2])) * dx + np.cos(float(pose[2])) * dy
        return PDMPathBoundaryPoint(progress_s, float(lateral_offset), int(line_type))

    @staticmethod
    def _append_lane_boundary_points(
        centerline: PDMPath,
        boundary_points: List[PDMPathBoundaryPoint],
        raw_xy: npt.NDArray[np.float64],
        line_type: int,
        skip_first: bool,
    ) -> None:
        """Append lane boundary points while keeping progress monotonic."""
        start_idx = 1 if skip_first and len(raw_xy) > 1 else 0
        for xy in raw_xy[start_idx:]:
            boundary_point = MetricCacheProcessor._project_boundary_xy(centerline, xy, line_type)
            if boundary_points and boundary_point.s <= boundary_points[-1].s + 1e-6:
                continue
            boundary_points.append(boundary_point)

    @staticmethod
    def _append_link_default_boundary_point(
        centerline: PDMPath,
        boundary_points: List[PDMPathBoundaryPoint],
        link_start_xy: npt.NDArray[np.float64],
    ) -> None:
        """Insert a default boundary-type point when entering a non-empty connector."""
        if not boundary_points:
            return

        link_start_s = float(centerline.project(Point(float(link_start_xy[0]), float(link_start_xy[1]))))
        previous_point = boundary_points[-1]
        if link_start_s - previous_point.s >= MOGO_LINK_DEFAULT_POINT_MIN_DISTANCE:
            insert_s = link_start_s
        else:
            insert_s = previous_point.s + MOGO_LINK_DEFAULT_POINT_MIN_DISTANCE

        insert_s = min(insert_s, float(centerline.length))
        if insert_s <= previous_point.s + 1e-6:
            return

        boundary_points.append(
            PDMPathBoundaryPoint(insert_s, float(previous_point.l), MOGO_BOUNDARY_TYPE_DEFAULT)
        )

    @staticmethod
    def _build_centerline_boundary_profile(
        centerline: Optional[PDMPath],
        route_plan: Optional[List[Any]],
    ) -> Optional[PDMPathBoundaryProfile]:
        """Build a Mogo-only s-l boundary profile for metric cache centerline."""
        if centerline is None or not route_plan:
            return None
        if not all(hasattr(edge, "left_boundary_type") and hasattr(edge, "right_boundary_type") for edge in route_plan):
            return None

        left_boundary: List[PDMPathBoundaryPoint] = []
        right_boundary: List[PDMPathBoundaryPoint] = []
        last_raw_xy: Dict[str, Optional[npt.NDArray[np.float64]]] = {"left": None, "right": None}
        previous_edge_was_lane = False

        for edge in route_plan:
            if MetricCacheProcessor._edge_is_mogo_connector(edge):
                link_start_xy = MetricCacheProcessor._edge_start_xy(edge)
                if link_start_xy is not None:
                    MetricCacheProcessor._append_link_default_boundary_point(centerline, left_boundary, link_start_xy)
                    MetricCacheProcessor._append_link_default_boundary_point(centerline, right_boundary, link_start_xy)
                previous_edge_was_lane = False
                continue

            for side, profile_points in (("left", left_boundary), ("right", right_boundary)):
                raw_xy = MetricCacheProcessor._edge_boundary_xy(edge, side)
                if len(raw_xy) == 0:
                    continue

                previous_raw_xy = last_raw_xy[side]
                skip_first = bool(
                    previous_edge_was_lane
                    and previous_raw_xy is not None
                    and np.linalg.norm(raw_xy[0] - previous_raw_xy) < MOGO_BOUNDARY_ENDPOINT_EPS
                )
                MetricCacheProcessor._append_lane_boundary_points(
                    centerline,
                    profile_points,
                    raw_xy,
                    MetricCacheProcessor._edge_boundary_type(edge, side),
                    skip_first,
                )
                last_raw_xy[side] = raw_xy[-1]

            previous_edge_was_lane = True

        if not left_boundary or not right_boundary:
            return None
        return PDMPathBoundaryProfile(left_boundary=left_boundary, right_boundary=right_boundary)

    def compute_metric_cache(self, scenario: AbstractScenario) -> Optional[CacheMetadataEntry]:

        file_name = self._cache_path / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl"

        if file_name.exists() and not self._force_feature_computation:
            return CacheMetadataEntry(file_name)

        # init and run PDM-Closed
        planner_input, planner_initialization = self._get_planner_inputs(scenario)
        self._pdm_closed.initialize(planner_initialization)
        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(planner_input)

        observation = self._build_gt_observation(scenario)
        history_states = build_history_states_from_scenario(
            scenario,
            history_horizon=1.5,
            interval_length=self._history_interval_length or scenario.database_interval,
        )
        centerline = self._get_metric_cache_centerline(scenario)

        boundary_profile = self._build_centerline_boundary_profile(
            centerline,
            self._get_metric_cache_boundary_route_plan(scenario, centerline),
        )
        if boundary_profile is not None:
            centerline.set_boundary_profile(boundary_profile)

        boundary_profile = self._build_centerline_boundary_profile(
            self._pdm_closed._centerline,
            getattr(self._pdm_closed, "_centerline_route_plan", None),
        )
        if boundary_profile is not None:
            self._pdm_closed._centerline.set_boundary_profile(boundary_profile)

        # save and dump features
        MetricCache(
            file_name,
            pdm_closed_trajectory,
            scenario.initial_ego_state,
            observation,
            centerline,
            list(self._pdm_closed._route_lane_dict.keys()),
            self._pdm_closed._drivable_area_map,
            history_states,
        ).dump()

        # return metadata
        return CacheMetadataEntry(file_name)
