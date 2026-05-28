from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from shapely.geometry import Point

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap


class AbstractPDMPlanner(AbstractPlanner, ABC):
    """
    Interface for planners incorporating PDM-* variants.
    """

    def __init__(
        self,
        map_radius: float,
    ):
        """
        Constructor of AbstractPDMPlanner.
        :param map_radius: radius around ego to consider
        """

        self._map_radius: int = map_radius  # [m]
        self._iteration: int = 0

        # lazy loaded
        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_dict: Optional[Dict[str, RoadBlockGraphEdgeMapObject]] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None

        self._centerline: Optional[PDMPath] = None
        self._drivable_area_map: Optional[PDMDrivableMap] = None

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

        self._load_route_connector_edges()

    def _load_route_connector_edges(self) -> None:
        """Add connector edges linking loaded route lanes without changing the roadblock route sequence."""
        route_roadblock_ids = set(self._route_roadblock_dict.keys())
        connector_edges = {}

        for lane in list(self._route_lane_dict.values()):
            for outgoing_edge in lane.outgoing_edges:
                if outgoing_edge.id in self._route_lane_dict:
                    continue

                outgoing_targets = getattr(outgoing_edge, "outgoing_edges", [])
                target_roadblock_ids = {
                    target_edge.get_roadblock_id()
                    for target_edge in outgoing_targets
                    if hasattr(target_edge, "get_roadblock_id")
                }
                edge_roadblock_id = outgoing_edge.get_roadblock_id()
                connects_route = bool(target_roadblock_ids.intersection(route_roadblock_ids))
                if edge_roadblock_id in route_roadblock_ids or connects_route:
                    connector_edges[outgoing_edge.id] = outgoing_edge

        self._route_lane_dict.update(connector_edges)

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        route_roadblock_ids = route_roadblock_correction(ego_state.rear_axle, self._map_api, self._route_roadblock_dict)
        self._load_route_dicts(route_roadblock_ids)

    def _get_current_roadblock_index(self, current_lane: LaneGraphEdgeMapObject, roadblock_ids: List[str]) -> int:
        """
        Resolve the route roadblock index for a lane or lane-connector edge.
        :param current_lane: current lane or connector edge.
        :param roadblock_ids: ordered route roadblock ids.
        :return: index in ordered route roadblocks.
        """
        roadblock_idcs = np.where(np.array(roadblock_ids) == current_lane.get_roadblock_id())[0]
        if len(roadblock_idcs) > 0:
            return int(roadblock_idcs[0])

        # Connector edges are not part of the route roadblock sequence. When ego is
        # already on a connector, continue the centerline search from its downstream
        # route roadblock while keeping the connector itself as the Dijkstra start.
        for adjacent_edges in (current_lane.outgoing_edges, current_lane.incoming_edges):
            for adjacent_edge in adjacent_edges:
                if not hasattr(adjacent_edge, "get_roadblock_id"):
                    continue

                adjacent_roadblock_idcs = np.where(np.array(roadblock_ids) == adjacent_edge.get_roadblock_id())[0]
                if len(adjacent_roadblock_idcs) > 0:
                    return int(adjacent_roadblock_idcs[0])

        raise ValueError(
            f"Unable to resolve current lane {current_lane.id} with roadblock "
            f"{current_lane.get_roadblock_id()} in route roadblocks."
        )

    def _get_discrete_centerline(self, current_lane: LaneGraphEdgeMapObject, search_depth: int = 30) -> List[StateSE2]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of discrete states on centerline (x,y,θ)
        """

        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = self._get_current_roadblock_index(current_lane, roadblock_ids)
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            for state in lane.baseline_path.discrete_path:
                if centerline_discrete_path and np.allclose(centerline_discrete_path[-1].array, state.array):
                    continue
                centerline_discrete_path.append(state)

        return centerline_discrete_path

    def _get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in ego's vicinity.
        :param ego_state: state of ego-vehicle
        :return: lane object (on-route)
        """
        starting_lane: LaneGraphEdgeMapObject = None
        on_route_lanes, heading_error = self._get_intersecting_lanes(ego_state)

        if on_route_lanes:
            # 1. Option: find lanes from lane occupancy-map
            # select lane with lowest heading error
            starting_lane = on_route_lanes[np.argmin(np.abs(heading_error))]
            return starting_lane

        else:
            # 2. Option: find any intersecting or close lane on-route
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane

    def _get_intersecting_lanes(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
        """
        Returns on-route lanes and heading errors where ego-vehicle intersects.
        :param ego_state: state of ego-vehicle
        :return: tuple of lists with lane objects and heading errors [rad].
        """
        assert self._drivable_area_map, "AbstractPDMPlanner: Drivable area map must be initialized first!"

        ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
        ego_rear_axle_point: Point = Point(*ego_position_array)
        ego_heading: float = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                # collect baseline path as array
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[StateSE2] = lane_object.baseline_path.discrete_path
                lane_state_se2_array = np.array([state.array for state in lane_discrete_path], dtype=np.float64)
                # calculate nearest state on baseline
                lane_distances = (ego_position_array[None, ...] - lane_state_se2_array) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5

                # calculate heading error
                heading_error = lane_discrete_path[np.argmin(lane_distances)].heading - ego_heading
                heading_error = np.abs(normalize_angle(heading_error))

                # add lane to candidates
                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)

        return on_route_lanes, on_route_heading_errors
