from __future__ import annotations

from typing import Any, Optional, Tuple, Type

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Scene
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.scenario_builder.mogo_vehicle_parameters import get_mogo_vehicle_parameters


class MogoNavSimScenario(NavSimScenario):
    """NAVSIM scenario variant that uses an injected Mogo map API."""

    def __init__(
        self,
        scene: Scene,
        map_root: str = "",
        map_version: str = "mogo-sqlite",
        ego_vehicle_parameters: VehicleParameters | None = None,
        database_interval: float = 0.1,
    ) -> None:
        """Initialize scenario with map api already attached to the Scene."""
        ego_vehicle_parameters = ego_vehicle_parameters or get_mogo_vehicle_parameters()
        super().__init__(
            scene=scene,
            map_root=map_root,
            map_version=map_version,
            ego_vehicle_parameters=ego_vehicle_parameters,
        )
        self._mogo_map_api = scene.map_api
        self._database_interval = database_interval
        self._future_sampling = TrajectorySampling(
            num_poses=len(self._time_points) + 1,
            interval_length=database_interval,
        )

    def __reduce__(self) -> Tuple[Type[MogoNavSimScenario], Tuple[Any, ...]]:
        """Hints on how to reconstruct the object when pickling."""
        return (
            self.__class__,
            (
                self._scene,
                self._map_root,
                self._map_version,
                self._ego_vehicle_parameters,
                self._database_interval,
            ),
        )

    @property
    def map_api(self) -> AbstractMap:
        """Return the injected Mogo SQLite map adapter."""
        return self._mogo_map_api

    @property
    def route_centerline(self) -> Optional[Any]:
        """Return pkl-provided route centerline for the current frame, if available."""
        current_frame_idx = self._scene.scene_metadata.num_history_frames - 1
        return getattr(self._scene.frames[current_frame_idx], "route_centerline", None)

    @property
    def route_centerline_segments(self) -> Optional[Any]:
        """Return pkl-provided route centerline segments for the current frame, if available."""
        current_frame_idx = self._scene.scene_metadata.num_history_frames - 1
        return getattr(self._scene.frames[current_frame_idx], "route_centerline_segments", None)
