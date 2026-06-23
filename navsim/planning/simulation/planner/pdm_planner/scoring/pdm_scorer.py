from typing import List, Optional, Tuple
from dataclasses import dataclass
import copy

import numpy as np
import numpy.typing as npt
from shapely import Point, creation

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_comfortable,
    ego_is_comfortable_v2,
    ego_is_start_anchor_comfortable,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    coords_array_to_polygon_array,
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)


@dataclass
class PDMScorerConfig:

    # weighted metric weights
    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    comfortable_weight: float = 2.0
    driving_direction_weight: float = 0.0
    obstacle_clearance_weight: float = 2.0
    following_safety_weight: float = 2.0
    centerline_weight: float = 1.0
    start_anchor_comfort_weight: float = 0.0
    comfort_metric_version: str = "v2"

    # thresholds
    driving_direction_horizon: float = 1.0  # [s] (driving direction)
    driving_direction_compliance_threshold: float = 2.0  # [m] (driving direction)
    driving_direction_violation_threshold: float = 6.0  # [m] (driving direction)
    stopped_speed_threshold: float = 5e-03  # [m/s] (ttc)
    at_fault_collision_score_floor: float = 0.05  # (no at-fault collision)
    ttc_horizon: float = 2.0  # [s] (ttc)
    ttc_full_weight_time: float = 1.0  # [s] (ttc)
    ttc_transition_center: float = 1.5  # [s] (ttc)
    ttc_transition_width: float = 0.35  # [s] (ttc)
    ttc_min_time_weight: float = 0.0  # (ttc)
    progress_distance_threshold: float = 5.0  # [m] (progress)
    drivable_area_soft_margin: float = 0.5  # [m] (drivable area)
    drivable_area_soft_score: float = 0.618  # score when only the safety margin leaves drivable area
    drivable_area_hard_score: float = 0.05  # score when the ego footprint leaves drivable area
    obstacle_clearance_base_margin: float = 0.3  # [m] (obstacle clearance)
    obstacle_clearance_time_headway: float = 0.3  # [s] (obstacle clearance)
    obstacle_clearance_max_margin: float = 1.0  # [m] (obstacle clearance)
    obstacle_clearance_score_exponent: float = 0.5  # < 1 makes near-obstacle clearance change faster
    following_safety_start_time: float = 0.5  # [s] (following safety)
    following_safety_horizon: float = 4.0  # [s] (following safety)
    following_safety_transition_center: float = 1.5  # [s] (following safety)
    following_safety_transition_width: float = 0.5  # [s] (following safety)
    following_base_gap: float = 1.0  # [m] (following safety)
    following_time_headway: float = 0.35  # [s] (following safety)
    following_closing_speed_gain: float = 0.25  # [s] (following safety)
    following_max_required_gap: float = 7.0  # [m] (following safety)
    following_lateral_margin: float = 0.8  # [m] (following safety)
    following_score_scale: float = 1.5  # [m] (following safety)
    centerline_target_distance: float = 0.7  # [m] (centerline convergence)
    centerline_max_distance: float = 1.5  # [m] (centerline convergence)
    centerline_horizon: float = 1.0  # [s] (centerline convergence, kept for config compatibility)
    centerline_start_weight: float = 0.05  # sigmoid weight at start_anchor_comfort_horizon
    start_anchor_comfort_horizon: float = 1.0  # [s] (start anchor comfort)
    start_anchor_comfort_interval_length: float = 0.5  # [s] (start anchor comfort)
    start_anchor_history_horizon: float = 1.5  # [s] (start anchor comfort)
    start_anchor_comfort_strict_ratio: float = 0.618  # stricter near current state, relaxed over anchor horizon
    start_anchor_jerk_penalty: float = 1e-4  # same default as BatchLQRTracker
    start_anchor_curvature_rate_penalty: float = 1e-2  # same default as BatchLQRTracker

    @property
    def weighted_metrics_array(self) -> npt.NDArray[np.float64]:
        weighted_metrics = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
        weighted_metrics[WeightedMetricIndex.PROGRESS] = self.progress_weight
        weighted_metrics[WeightedMetricIndex.TTC] = self.ttc_weight
        weighted_metrics[WeightedMetricIndex.COMFORTABLE] = self.comfortable_weight
        weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION] = self.driving_direction_weight
        weighted_metrics[WeightedMetricIndex.OBSTACLE_CLEARANCE] = self.obstacle_clearance_weight
        weighted_metrics[WeightedMetricIndex.FOLLOWING_SAFETY] = self.following_safety_weight
        weighted_metrics[WeightedMetricIndex.CENTERLINE_CONVERGENCE] = self.centerline_weight
        weighted_metrics[WeightedMetricIndex.START_ANCHOR_COMFORT] = self.start_anchor_comfort_weight
        return weighted_metrics


class PDMScorer:
    """Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics."""

    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        config: PDMScorerConfig = PDMScorerConfig(),
        vehicle_parameters: VehicleParameters | None = None,
    ):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self.proposal_sampling = proposal_sampling
        self._config = config
        self._vehicle_parameters = vehicle_parameters or get_pacifica_parameters()

        # lazy loaded
        self._observation: Optional[PDMObservation] = None
        self._centerline: Optional[PDMPath] = None
        self._route_lane_ids: Optional[List[str]] = None
        self._drivable_area_map: Optional[PDMDrivableMap] = None

        self._num_proposals: Optional[int] = None
        self._states: Optional[npt.NDArray[np.float64]] = None
        self._reference_signal: Optional[npt.NDArray[np.float64]] = None
        self._history_states: Optional[npt.NDArray[np.float64]] = None
        self._ego_coords: Optional[npt.NDArray[np.float64]] = None
        self._ego_polygons: Optional[npt.NDArray[np.object_]] = None

        self._ego_areas: Optional[npt.NDArray[np.bool_]] = None

        self._multi_metrics: Optional[npt.NDArray[np.float64]] = None
        self._weighted_metrics: Optional[npt.NDArray[np.float64]] = None
        self._progress_raw: Optional[npt.NDArray[np.float64]] = None

        self._collision_time_idcs: Optional[npt.NDArray[np.float64]] = None
        self._ttc_time_idcs: Optional[npt.NDArray[np.float64]] = None
        self._start_anchor_comfort_available: bool = False

    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return self._collision_time_idcs[proposal_idx] * self.proposal_sampling.interval_length

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return self._ttc_time_idcs[proposal_idx] * self.proposal_sampling.interval_length

    def score_proposals(
        self,
        states: npt.NDArray[np.float64],
        reference_signal: Optional[npt.NDArray[np.float64]],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        history_states: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_ids: list containing on-route lane ids
        :param drivable_area_map: Occupancy map of drivable are polygons
        :return: array containing score of each proposal
        """

        # initialize & lazy load class values
        self._reset(
            states,
            reference_signal,
            observation,
            centerline,
            route_lane_ids,
            drivable_area_map,
            history_states,
        )

        # fill value ego-area array (used in multiple metrics)
        self._calculate_ego_area()

        # 1. multiplicative metrics
        self._calculate_no_at_fault_collision()
        self._calculate_drivable_area_compliance()

        # 2. weighted metrics
        self._calculate_progress()
        self._calculate_centerline_convergence()
        self._calculate_ttc()
        self._calculate_following_safety()
        self._calculate_obstacle_clearance()
        self._calculate_driving_direction_compliance()
        self._calculate_start_anchor_comfort()
        self._calculate_is_comfortable()

        return self._aggregate_scores()

    def _aggregate_scores(self) -> npt.NDArray[np.float64]:
        """
        Aggregates metrics with multiplicative and weighted average.
        :return: array containing score of each proposal
        """

        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize and fill progress values
        raw_progress = self._progress_raw * multiplicate_metric_scores
        max_raw_progress = np.max(raw_progress)
        if max_raw_progress > self._config.progress_distance_threshold:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float64)
            normalized_progress[multiplicate_metric_scores == 0.0] = 0.0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        # accumulate weighted metrics
        weighted_metrics_array = self._config.weighted_metrics_array.copy()
        if not self._start_anchor_comfort_available:
            weighted_metrics_array[WeightedMetricIndex.START_ANCHOR_COMFORT] = 0.0
        weighted_metric_scores = (self._weighted_metrics * weighted_metrics_array[..., None]).sum(axis=0)
        weighted_metric_scores /= weighted_metrics_array.sum()

        # calculate final scores
        final_scores = self._multi_metrics.prod(axis=0) * weighted_metric_scores

        return final_scores

    def _reset(
        self,
        states: npt.NDArray[np.float64],
        reference_signal: Optional[npt.NDArray[np.float64]],
        observation: PDMObservation,
        centerline: PDMPath,
        route_lane_ids: List[str],
        drivable_area_map: PDMDrivableMap,
        history_states: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """
        Resets metric values and lazy loads input classes.
        :param states: array representation of simulated proposals
        :param observation: PDM's observation class
        :param centerline: path of the centerline
        :param route_lane_ids: list containing on-route lane ids
        :param drivable_area_map: Occupancy map of drivable are polygons
        """
        assert states.ndim == 3
        assert states.shape[1] == self.proposal_sampling.num_poses + 1
        assert states.shape[2] == StateIndex.size()
        if history_states is not None:
            assert history_states.ndim in (2, 3)
            if history_states.ndim == 2:
                assert history_states.shape[1] == StateIndex.size()
                history_states = np.repeat(history_states[None, ...], states.shape[0], axis=0)
            else:
                assert history_states.shape[0] == states.shape[0]
                assert history_states.shape[2] == StateIndex.size()

        self._observation = observation
        self._centerline = centerline
        self._route_lane_ids = route_lane_ids
        self._drivable_area_map = drivable_area_map

        self._num_proposals = states.shape[0]

        # save ego state values
        self._states = states
        self._reference_signal = reference_signal
        self._history_states = history_states
        self._start_anchor_comfort_available = False

        # calculate coordinates of ego corners and center
        self._ego_coords = state_array_to_coords_array(states, self._vehicle_parameters)

        # initialize all ego polygons from corners
        self._ego_polygons = coords_array_to_polygon_array(self._ego_coords)

        # zero initialize all remaining arrays.
        self._ego_areas = np.zeros(
            (
                self._num_proposals,
                self.proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        self._multi_metrics = np.zeros((len(MultiMetricIndex), self._num_proposals), dtype=np.float64)
        self._weighted_metrics = np.zeros((len(WeightedMetricIndex), self._num_proposals), dtype=np.float64)
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float64)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        self._collision_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._ttc_time_idcs = np.zeros(self._num_proposals, dtype=np.float64)
        self._collision_time_idcs.fill(np.inf)
        self._ttc_time_idcs.fill(np.inf)

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """

        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape

        in_polygons = self._drivable_area_map.points_in_polygons(self._ego_coords)
        in_polygons = in_polygons.transpose(1, 2, 0, 3)  # shape: n_proposals, n_horizon, n_polygons, n_points

        drivable_area_idcs = self._drivable_area_map.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.DRIVABLE_AREA,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )

        drivable_lane_idcs = self._drivable_area_map.get_indices_of_map_type(
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )

        drivable_on_route_idcs: List[int] = [
            idx for idx in drivable_lane_idcs if self._drivable_area_map.tokens[idx] in self._route_lane_ids
        ]  # index mask for on-route lanes

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        center_in_polygon = in_polygons[..., -1]  # only center

        # in_multiple_lanes: if
        # - more than one drivable polygon contains at least one corner
        # - no polygon contains all corners
        batch_multiple_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_multiple_lanes_mask = (corners_in_polygon[:, :, drivable_lane_idcs].sum(axis=-1) > 0).sum(axis=-1) > 1

        batch_not_single_lanes_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_not_single_lanes_mask = np.all(corners_in_polygon[:, :, drivable_lane_idcs].sum(axis=-1) != 4, axis=-1)

        multiple_lanes_mask = np.logical_and(batch_multiple_lanes_mask, batch_not_single_lanes_mask)
        self._ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_nondrivable_area: if at least one corner is not within any drivable polygon
        batch_nondrivable_area_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_nondrivable_area_mask = (corners_in_polygon[:, :, drivable_area_idcs].sum(axis=-2) > 0).sum(axis=-1) < 4
        self._ego_areas[batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA] = True

        # in_oncoming_traffic: if center not in any drivable polygon that is on-route
        batch_oncoming_traffic_mask = np.zeros((n_proposals, n_horizon), dtype=np.bool_)
        batch_oncoming_traffic_mask = center_in_polygon[..., drivable_on_route_idcs].sum(axis=-1) == 0
        self._ego_areas[batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC] = True

    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        no_collision_scores = np.ones(self._num_proposals, dtype=np.float64)

        proposal_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        for time_idx in range(self.proposal_sampling.num_poses + 1):
            ego_polygons = self._ego_polygons[:, time_idx]
            intersecting = self._observation[time_idx].query(ego_polygons, predicate="intersects")

            if len(intersecting) == 0:
                continue

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = self._observation[time_idx].tokens[geometry_idx]
                if (self._observation.red_light_token in token) or (token in proposal_collided_track_ids[proposal_idx]):
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                )

                tracked_object = self._observation.unique_objects[token]

                # classify collision
                collision_type: CollisionType = get_collision_type(
                    self._states[proposal_idx, time_idx],
                    self._ego_polygons[proposal_idx, time_idx],
                    tracked_object,
                    self._observation[time_idx][token],
                )
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

                # 1. at fault collision
                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = (
                        np.clip(self._config.at_fault_collision_score_floor, 0.0, 1.0)
                        if tracked_object.tracked_object_type in AGENT_TYPES
                        else 0.5
                    )
                    no_collision_scores[proposal_idx] = np.minimum(
                        no_collision_scores[proposal_idx], no_at_fault_collision_score
                    )
                    self._collision_time_idcs[proposal_idx] = min(time_idx, self._collision_time_idcs[proposal_idx])

                else:  # 2. no at fault collision
                    proposal_collided_track_ids[proposal_idx].append(token)

        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_collision_scores

    def _calculate_soft_drivable_area_violation(self) -> npt.NDArray[np.bool_]:
        """
        Checks whether the ego footprint with safety margin leaves the drivable area.
        """
        margin = self._config.drivable_area_soft_margin
        if margin <= 0.0:
            return np.zeros((self._num_proposals, self.proposal_sampling.num_poses + 1), dtype=np.bool_)

        soft_ego_coords = state_array_to_coords_array(
            self._states,
            self._vehicle_parameters,
            lon_margin=margin,
            lat_margin=margin,
        )
        in_polygons = self._drivable_area_map.points_in_polygons(soft_ego_coords)
        in_polygons = in_polygons.transpose(1, 2, 0, 3)

        drivable_area_idcs = self._drivable_area_map.get_indices_of_map_type(
            [
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.DRIVABLE_AREA,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )

        corners_in_polygon = in_polygons[..., :-1]
        return (corners_in_polygon[:, :, drivable_area_idcs].sum(axis=-2) > 0).sum(axis=-1) < 4

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric.
        The hard check keeps a strong floor score, while the soft margin check is time-weighted.
        """
        hard_off_road_mask = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(axis=-1)
        soft_off_road_mask = self._calculate_soft_drivable_area_violation()
        start_idx, time_weights = self._get_sigmoid_time_weights(
            self.proposal_sampling.num_poses + 1,
            self._config.centerline_start_weight,
        )

        soft_penalty = 1.0 - self._config.drivable_area_soft_score
        weighted_soft_penalties = soft_off_road_mask[:, start_idx:] * time_weights[None, :] * soft_penalty
        drivable_area_compliance_scores = 1.0 - weighted_soft_penalties.max(axis=-1)
        drivable_area_compliance_scores[hard_off_road_mask] = self._config.drivable_area_hard_score
        self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA] = drivable_area_compliance_scores

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        oncoming_progress = np.zeros(
            (self._num_proposals, self.proposal_sampling.num_poses + 1),
            dtype=np.float64,
        )
        oncoming_progress[:, 1:] = np.linalg.norm(center_coordinates[:, 1:] - center_coordinates[:, :-1], axis=-1)

        # mask out progress along the driving direction
        oncoming_traffic_masks = self._ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[~oncoming_traffic_masks] = 0.0

        # aggregate
        driving_direction_compliance_scores = np.ones(self._num_proposals, dtype=np.float64)
        horizon = int(self._config.driving_direction_horizon / self.proposal_sampling.interval_length)

        oncoming_progress_over_horizon = np.concatenate(
            [
                oncoming_progress[:, max(0, time_idx - horizon) : time_idx + 1].sum(axis=-1)[..., None]
                for time_idx in range(oncoming_progress.shape[-1])
            ],
            dtype=np.float64,
            axis=-1,
        )

        for proposal_idx, progress in enumerate(oncoming_progress_over_horizon.max(axis=-1)):
            if progress < self._config.driving_direction_compliance_threshold:
                driving_direction_compliance_scores[proposal_idx] = 1.0
            elif progress < self._config.driving_direction_violation_threshold:
                driving_direction_compliance_scores[proposal_idx] = 0.5
            else:
                driving_direction_compliance_scores[proposal_idx] = 0.0

        self._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION] = driving_direction_compliance_scores

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """

        # calculate raw progress in meter
        progress_in_meter = np.zeros(self._num_proposals, dtype=np.float64)
        for proposal_idx in range(self._num_proposals):
            start_point = Point(*self._ego_coords[proposal_idx, 0, BBCoordsIndex.CENTER])
            end_point = Point(*self._ego_coords[proposal_idx, -1, BBCoordsIndex.CENTER])
            progress = self._centerline.project([start_point, end_point])
            progress_in_meter[proposal_idx] = progress[1] - progress[0]

        self._progress_raw = np.clip(progress_in_meter, a_min=0, a_max=None)

    @staticmethod
    def _smoothstep(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Smoothly maps values from [0, 1] to [0, 1] with flatter slopes near both ends.
        """
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _logit(value: float) -> float:
        """Maps probability from (0, 1) to log-odds."""
        value = np.clip(value, 1e-6, 1.0 - 1e-6)
        return float(np.log(value / (1.0 - value)))

    def _get_sigmoid_time_weights(
        self,
        n_time: int,
        start_weight: float,
    ) -> Tuple[int, npt.NDArray[np.float64]]:
        """Builds a temporal weight ramp from the start anchor horizon to the proposal end."""
        time_points_s = np.arange(n_time, dtype=np.float64) * self.proposal_sampling.interval_length
        anchor_end_time = min(self._config.start_anchor_comfort_horizon, time_points_s[-1])
        start_idx = int(np.searchsorted(time_points_s, anchor_end_time, side="left"))
        active_time_points_s = time_points_s[start_idx:]

        if len(active_time_points_s) <= 1:
            return start_idx, np.ones(len(active_time_points_s), dtype=np.float64)

        t_start = active_time_points_s[0]
        t_end = active_time_points_s[-1]
        t0 = 0.5 * (t_start + t_end)

        start_weight = np.clip(start_weight, 1e-6, 0.5 - 1e-6)
        k = self._logit(start_weight) / (t_start - t0)
        return start_idx, 1.0 / (1.0 + np.exp(-k * (active_time_points_s - t0)))

    @staticmethod
    def _sigmoid(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Numerically stable enough sigmoid for metric shaping."""
        value = np.clip(value, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-value))

    def _get_increasing_time_weights(
        self,
        n_time: int,
        start_time: float,
        transition_center: float,
        transition_width: float,
    ) -> Tuple[int, npt.NDArray[np.float64]]:
        """Builds an increasing sigmoid time weight after start_time."""
        time_points_s = np.arange(n_time, dtype=np.float64) * self.proposal_sampling.interval_length
        start_idx = int(np.searchsorted(time_points_s, max(start_time, 0.0), side="left"))
        active_time_points_s = time_points_s[start_idx:]
        if len(active_time_points_s) == 0:
            return start_idx, np.zeros(0, dtype=np.float64)

        width = max(transition_width, 1e-6)
        weights = self._sigmoid((active_time_points_s - transition_center) / width)
        return start_idx, weights.astype(np.float64)

    def _get_ttc_time_weights(self, future_time_idcs: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Builds a decreasing TTC time weight from full strength to the configured minimum."""
        future_times_s = future_time_idcs.astype(np.float64) * self.proposal_sampling.interval_length
        horizon = max(self._config.ttc_horizon, self._config.ttc_full_weight_time)
        full_weight_time = min(self._config.ttc_full_weight_time, horizon)
        min_weight = np.clip(self._config.ttc_min_time_weight, 0.0, 1.0)

        if horizon <= full_weight_time + 1e-6:
            return np.ones_like(future_times_s, dtype=np.float64)

        width = max(self._config.ttc_transition_width, 1e-6)
        raw_weights = 1.0 / (1.0 + np.exp((future_times_s - self._config.ttc_transition_center) / width))
        raw_full = 1.0 / (1.0 + np.exp((full_weight_time - self._config.ttc_transition_center) / width))
        raw_end = 1.0 / (1.0 + np.exp((horizon - self._config.ttc_transition_center) / width))
        normalized = (raw_weights - raw_end) / max(raw_full - raw_end, 1e-6)
        normalized[future_times_s <= full_weight_time] = 1.0
        normalized[future_times_s >= horizon] = 0.0
        normalized = np.clip(normalized, 0.0, 1.0)
        return min_weight + (1.0 - min_weight) * normalized

    def _get_ttc_future_time_idcs(self) -> npt.NDArray[np.int64]:
        """Keeps the original dense near-term TTC samples and adds sparse mid-term samples when needed."""
        horizon_step = max(0, int(round(self._config.ttc_horizon / self.proposal_sampling.interval_length)))
        near_steps = np.array([0, 1, 3, 6, 10], dtype=np.int64)
        if horizon_step <= near_steps[-1]:
            future_time_idcs = near_steps[near_steps <= horizon_step]
        else:
            mid_steps = np.rint(np.linspace(near_steps[-1], horizon_step, 3)).astype(np.int64)
            future_time_idcs = np.unique(np.concatenate([near_steps, mid_steps]))

        if len(future_time_idcs) == 0 or future_time_idcs[-1] != horizon_step:
            future_time_idcs = np.unique(np.append(future_time_idcs, horizon_step))
        return future_time_idcs.astype(np.int64)

    def _calculate_centerline_convergence(self) -> None:
        """
        Scores whether the proposal converges toward the route centerline after the start anchor horizon.
        """
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        n_time = self.proposal_sampling.num_poses + 1
        distances = np.zeros((self._num_proposals, n_time), dtype=np.float64)

        for proposal_idx in range(self._num_proposals):
            points = [Point(*xy) for xy in center_coordinates[proposal_idx]]
            projected_progress = self._centerline.project(points)
            projected_states = self._centerline.interpolate(projected_progress, as_array=True)
            projected_xy = projected_states[:, :2]
            distances[proposal_idx] = np.linalg.norm(center_coordinates[proposal_idx] - projected_xy, axis=-1)

        start_idx, time_weights = self._get_sigmoid_time_weights(n_time, self._config.centerline_start_weight)
        centerline_distances = distances[:, start_idx:]

        if len(time_weights) <= 1:
            weighted_distance = centerline_distances[:, -1]
        else:
            weighted_distance = np.average(centerline_distances, axis=-1, weights=time_weights)

        target_distance = self._config.centerline_target_distance
        distance_range = max(self._config.centerline_max_distance - target_distance, 1e-6)
        normalized_distance = (weighted_distance - target_distance) / distance_range
        centerline_scores = 1.0 - self._smoothstep(normalized_distance)

        self._weighted_metrics[WeightedMetricIndex.CENTERLINE_CONVERGENCE] = centerline_scores

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        """

        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        temp_collided_track_ids = {
            proposal_idx: copy.deepcopy(self._observation.collided_track_ids)
            for proposal_idx in range(self._num_proposals)
        }

        # Calculate TTC with dense near-term and sparse mid-term projections.
        future_time_idcs = self._get_ttc_future_time_idcs()
        future_scores = np.linspace(0.0, 1.0, len(future_time_idcs))
        future_time_weights = self._get_ttc_time_weights(future_time_idcs)
        n_future_steps = len(future_time_idcs)

        # create polygons for each ego position and 1s future projection
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]
        coords_exterior_time_steps = np.repeat(coords_exterior[:, :, None], n_future_steps, axis=2)

        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )

        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        for idx, future_time_idx in enumerate(future_time_idcs):
            delta_t = float(future_time_idx) * self.proposal_sampling.interval_length
            coords_exterior_time_steps[:, :, idx] = (
                coords_exterior_time_steps[:, :, idx] + dxy_per_s[:, :, None] * delta_t
            )

        polygons = creation.polygons(coords_exterior_time_steps)

        # check collision for each proposal and projection
        max_observation_time_idx = len(self._observation._global_to_local_idcs) - 1
        for time_idx in range(self.proposal_sampling.num_poses + 1):
            for step_idx, future_time_idx in enumerate(future_time_idcs):
                current_time_idx = time_idx + future_time_idx
                if current_time_idx > max_observation_time_idx:
                    continue

                polygons_at_time_step = polygons[:, time_idx, step_idx]
                intersecting = self._observation[current_time_idx].query(polygons_at_time_step, predicate="intersects")

                if len(intersecting) == 0:
                    continue

                for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    token = self._observation[current_time_idx].tokens[geometry_idx]
                    if (
                        (self._observation.red_light_token in token)
                        or (token in temp_collided_track_ids[proposal_idx])
                        or (speeds[proposal_idx, time_idx] < self._config.stopped_speed_threshold)
                    ):
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                        or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                    )
                    ego_rear_axle: StateSE2 = StateSE2(*self._states[proposal_idx, time_idx, StateIndex.STATE_SE2])

                    centroid = self._observation[current_time_idx][token].centroid
                    track_heading = self._observation.unique_objects[token].box.center.heading
                    track_state = StateSE2(centroid.x, centroid.y, track_heading)
                    # TODO: fix ego_area for intersection
                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or self._drivable_area_map.is_in_layer(
                                ego_rear_axle.point, layer=SemanticMapLayer.INTERSECTION
                            )
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        ttc_risk = 1.0 - future_scores[step_idx]
                        weighted_ttc_score = 1.0 - future_time_weights[step_idx] * ttc_risk
                        ttc_scores[proposal_idx] = np.minimum(ttc_scores[proposal_idx], weighted_ttc_score)
                        self._ttc_time_idcs[proposal_idx] = min(time_idx, self._ttc_time_idcs[proposal_idx])
                    else:
                        temp_collided_track_ids[proposal_idx].append(token)

        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def _get_obstacle_velocity(self, tracked_object) -> npt.NDArray[np.float64]:
        """
        Returns obstacle velocity when available, otherwise treats it as static.
        """
        velocity = getattr(tracked_object, "velocity", None)
        if callable(velocity):
            velocity = velocity()
        if velocity is None:
            return np.zeros(2, dtype=np.float64)

        if hasattr(velocity, "array"):
            velocity_array = np.asarray(velocity.array, dtype=np.float64)
        elif hasattr(velocity, "x") and hasattr(velocity, "y"):
            velocity_array = np.array([velocity.x, velocity.y], dtype=np.float64)
        else:
            velocity_array = np.asarray(velocity, dtype=np.float64)

        if velocity_array.size < 2:
            return np.zeros(2, dtype=np.float64)
        return np.nan_to_num(velocity_array[:2], nan=0.0)

    def _compute_closing_speed(
        self,
        ego_state: npt.NDArray[np.float64],
        ego_polygon,
        obstacle_polygon,
        tracked_object,
    ) -> float:
        """
        Computes the relative speed projected onto the ego-to-obstacle direction.
        """
        ego_xy = np.array([ego_polygon.centroid.x, ego_polygon.centroid.y], dtype=np.float64)
        obstacle_xy = np.array([obstacle_polygon.centroid.x, obstacle_polygon.centroid.y], dtype=np.float64)

        direction = obstacle_xy - ego_xy
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return 0.0
        direction /= distance

        ego_velocity = ego_state[StateIndex.VELOCITY_2D]
        obstacle_velocity = self._get_obstacle_velocity(tracked_object)
        relative_velocity = ego_velocity - obstacle_velocity
        return max(0.0, float(np.dot(relative_velocity, direction)))

    def _compute_following_gap(
        self,
        ego_state: npt.NDArray[np.float64],
        obstacle_polygon,
    ) -> Optional[float]:
        """
        Computes longitudinal gap from ego front bumper to the rear edge of an ahead obstacle.
        Returns None when the obstacle is not a longitudinal following candidate.
        """
        if not hasattr(obstacle_polygon, "exterior"):
            return None

        obstacle_coords = np.asarray(obstacle_polygon.exterior.coords, dtype=np.float64)
        if obstacle_coords.ndim != 2 or obstacle_coords.shape[0] < 2:
            return None

        heading = ego_state[StateIndex.HEADING]
        longitudinal_axis = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        lateral_axis = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float64)
        ego_xy = ego_state[StateIndex.POINT]
        relative_coords = obstacle_coords[:, :2] - ego_xy

        obstacle_s = relative_coords @ longitudinal_axis
        obstacle_l = relative_coords @ lateral_axis
        ego_front_s = self._vehicle_parameters.front_length

        # Ignore objects fully behind the ego front bumper.
        if float(np.max(obstacle_s)) <= ego_front_s:
            return None

        lateral_margin = self._config.following_lateral_margin
        ego_left_l = self._vehicle_parameters.half_width
        ego_right_l = -self._vehicle_parameters.half_width
        if (float(np.max(obstacle_l)) < ego_right_l - lateral_margin) or (
            float(np.min(obstacle_l)) > ego_left_l + lateral_margin
        ):
            return None

        return float(np.min(obstacle_s) - ego_front_s)

    def _calculate_following_safety(self) -> None:
        """
        Penalizes insufficient longitudinal following gap in the mid-to-long horizon.
        """
        following_scores = np.ones(self._num_proposals, dtype=np.float64)
        ignored_tokens = set(self._observation.collided_track_ids)
        n_time = self.proposal_sampling.num_poses + 1
        start_idx, time_weights = self._get_increasing_time_weights(
            n_time,
            self._config.following_safety_start_time,
            self._config.following_safety_transition_center,
            self._config.following_safety_transition_width,
        )
        end_idx = min(
            n_time,
            int(round(self._config.following_safety_horizon / self.proposal_sampling.interval_length)) + 1,
        )
        if start_idx >= end_idx or len(time_weights) == 0:
            self._weighted_metrics[WeightedMetricIndex.FOLLOWING_SAFETY] = following_scores
            return

        time_weights = time_weights[: end_idx - start_idx]
        max_query_margin = self._config.following_max_required_gap + self._config.following_lateral_margin
        score_scale = max(self._config.following_score_scale, 1e-6)

        for relative_time_idx, time_idx in enumerate(range(start_idx, end_idx)):
            obstacle_map = self._observation[time_idx]
            time_weight = time_weights[relative_time_idx]

            for proposal_idx in range(self._num_proposals):
                ego_state = self._states[proposal_idx, time_idx]
                ego_polygon = self._ego_polygons[proposal_idx, time_idx]
                candidate_idcs = obstacle_map.query(ego_polygon.buffer(max_query_margin), predicate="intersects")

                heading = ego_state[StateIndex.HEADING]
                longitudinal_axis = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
                ego_velocity = ego_state[StateIndex.VELOCITY_2D]
                ego_speed = max(0.0, float(np.dot(ego_velocity, longitudinal_axis)))

                for geometry_idx in candidate_idcs:
                    token = obstacle_map.tokens[geometry_idx]
                    if (self._observation.red_light_token in token) or (token in ignored_tokens):
                        continue

                    tracked_object = self._observation.unique_objects.get(token)
                    if tracked_object is None:
                        continue

                    obstacle_polygon = obstacle_map[token]
                    gap = self._compute_following_gap(ego_state, obstacle_polygon)
                    if gap is None:
                        continue

                    obstacle_velocity = self._get_obstacle_velocity(tracked_object)
                    obstacle_speed = float(np.dot(obstacle_velocity, longitudinal_axis))
                    closing_speed = max(0.0, ego_speed - obstacle_speed)
                    required_gap = (
                        self._config.following_base_gap
                        + self._config.following_time_headway * ego_speed
                        + self._config.following_closing_speed_gain * closing_speed
                    )
                    required_gap = np.clip(
                        required_gap,
                        self._config.following_base_gap,
                        self._config.following_max_required_gap,
                    )
                    raw_score = float(self._sigmoid(np.array([(gap - required_gap) / score_scale]))[0])
                    weighted_score = 1.0 - time_weight * (1.0 - raw_score)
                    following_scores[proposal_idx] = np.minimum(following_scores[proposal_idx], weighted_score)

        self._weighted_metrics[WeightedMetricIndex.FOLLOWING_SAFETY] = following_scores

    def _calculate_obstacle_clearance(self) -> None:
        """
        Penalizes proposals that pass too close to obstacles without changing collision semantics.
        """
        obstacle_clearance_scores = np.ones(self._num_proposals, dtype=np.float64)
        ignored_tokens = set(self._observation.collided_track_ids)
        max_margin = self._config.obstacle_clearance_max_margin

        for time_idx in range(self.proposal_sampling.num_poses + 1):
            obstacle_map = self._observation[time_idx]

            for proposal_idx in range(self._num_proposals):
                ego_polygon = self._ego_polygons[proposal_idx, time_idx]
                candidate_polygon = ego_polygon.buffer(max_margin)
                candidate_idcs = obstacle_map.query(candidate_polygon, predicate="intersects")

                for geometry_idx in candidate_idcs:
                    token = obstacle_map.tokens[geometry_idx]
                    if (self._observation.red_light_token in token) or (token in ignored_tokens):
                        continue

                    tracked_object = self._observation.unique_objects.get(token)
                    if tracked_object is None:
                        continue

                    obstacle_polygon = obstacle_map[token]
                    if ego_polygon.intersects(obstacle_polygon):
                        continue

                    closing_speed = self._compute_closing_speed(
                        self._states[proposal_idx, time_idx],
                        ego_polygon,
                        obstacle_polygon,
                        tracked_object,
                    )
                    clearance_margin = min(
                        self._config.obstacle_clearance_base_margin
                        + self._config.obstacle_clearance_time_headway * closing_speed,
                        max_margin,
                    )

                    obstacle_distance = ego_polygon.distance(obstacle_polygon)
                    clearance_ratio = np.clip(obstacle_distance / max(clearance_margin, 1e-6), 0.0, 1.0)
                    clearance_score = clearance_ratio ** self._config.obstacle_clearance_score_exponent
                    obstacle_clearance_scores[proposal_idx] = np.minimum(
                        obstacle_clearance_scores[proposal_idx],
                        clearance_score,
                    )

        self._weighted_metrics[WeightedMetricIndex.OBSTACLE_CLEARANCE] = obstacle_clearance_scores

    def _calculate_start_anchor_comfort(self) -> None:
        """
        Checks start-window comfort with historical left context.
        """
        start_anchor_comfort_scores = np.ones(self._num_proposals, dtype=np.float64)
        if self._history_states is None:
            self._weighted_metrics[WeightedMetricIndex.START_ANCHOR_COMFORT] = start_anchor_comfort_scores
            return

        anchor_interval = self._config.start_anchor_comfort_interval_length
        sample_interval = self.proposal_sampling.interval_length
        sample_stride = int(round(anchor_interval / sample_interval))
        if sample_stride <= 0 or not np.isclose(sample_stride * sample_interval, anchor_interval):
            self._weighted_metrics[WeightedMetricIndex.START_ANCHOR_COMFORT] = start_anchor_comfort_scores
            return

        history_steps = int(round(self._config.start_anchor_history_horizon / anchor_interval))
        if history_steps <= 0 or self._history_states.shape[1] < history_steps:
            self._weighted_metrics[WeightedMetricIndex.START_ANCHOR_COMFORT] = start_anchor_comfort_scores
            return

        check_steps = max(1, int(round(self._config.start_anchor_comfort_horizon / anchor_interval)) + 1)
        max_future_idx = (check_steps - 1) * sample_stride
        if max_future_idx >= self._states.shape[1]:
            self._weighted_metrics[WeightedMetricIndex.START_ANCHOR_COMFORT] = start_anchor_comfort_scores
            return

        history_anchor_states = self._history_states[:, -history_steps:]
        sampled_state_idcs = np.arange(0, check_steps * sample_stride, sample_stride)
        sampled_start_states = self._states[:, sampled_state_idcs]

        anchor_states = np.concatenate([history_anchor_states, sampled_start_states], axis=1)
        history_time_point_s = -anchor_interval * np.arange(history_steps, 0, -1, dtype=np.float64)
        start_time_point_s = anchor_interval * np.arange(check_steps, dtype=np.float64)
        anchor_time_point_s = np.concatenate([history_time_point_s, start_time_point_s])

        is_start_anchor_comfortable = ego_is_start_anchor_comfortable(
            anchor_states,
            anchor_time_point_s,
            check_start_idx=history_steps,
            check_steps=check_steps,
            strict_ratio=self._config.start_anchor_comfort_strict_ratio,
            jerk_penalty=self._config.start_anchor_jerk_penalty,
            curvature_rate_penalty=self._config.start_anchor_curvature_rate_penalty,
        )
        self._weighted_metrics[WeightedMetricIndex.START_ANCHOR_COMFORT] = np.all(
            is_start_anchor_comfortable,
            axis=-1,
        ).astype(np.float64)
        self._start_anchor_comfort_available = True

    def _calculate_is_comfortable(self) -> None:
        """
        Re-implementation of nuPlan's comfortability metric.
        """
        time_point_s: npt.NDArray[np.float64] = (
            np.arange(0, self.proposal_sampling.num_poses + 1).astype(np.float64)
            * self.proposal_sampling.interval_length
        )
        if self._config.comfort_metric_version == "v1":
            is_comfortable = ego_is_comfortable(self._states, time_point_s)
        elif self._config.comfort_metric_version == "v2":
            if self._reference_signal is None:
                raise ValueError("reference_signal must be provided when comfort_metric_version is 'v2'")
            is_comfortable = ego_is_comfortable_v2(self._states, self._reference_signal, time_point_s)
        else:
            raise ValueError(f"Unknown comfort_metric_version: {self._config.comfort_metric_version}")
        self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = np.all(is_comfortable, axis=-1)
