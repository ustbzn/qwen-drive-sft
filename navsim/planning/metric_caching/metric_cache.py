from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

import lzma
import pickle

import numpy as np
import numpy.typing as npt

from nuplan.common.utils.io_utils import save_buffer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap


@dataclass
class MetricCache:
    """Dataclass for storing metric computation information."""

    file_path: Path
    trajectory: InterpolatedTrajectory
    ego_state: EgoState

    observation: PDMObservation
    centerline: PDMPath
    route_lane_ids: List[str]
    drivable_area_map: PDMDrivableMap
    history_states: Optional[npt.NDArray[np.float64]] = None
    traffic_light_states: Optional[List[Dict[str, Any]]] = None
    driving_command: Optional[Any] = None

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Backward-compatible unpickle for caches created before traffic-light fields."""
        self.__dict__.update(state)
        self.traffic_light_states = self.__dict__.get("traffic_light_states")
        self.driving_command = self.__dict__.get("driving_command")

    def dump(self) -> None:
        """Dump metric cache to pickle with lzma compression."""
        # TODO: check if file_path must really be pickled
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        save_buffer(self.file_path, lzma.compress(pickle_object, preset=0))
