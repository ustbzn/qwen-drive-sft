from __future__ import annotations

from typing import Any, Tuple, Type

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.abstract_map import AbstractMap

from navsim.common.dataclasses import Scene
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario


class MogoNavSimScenario(NavSimScenario):
    """NAVSIM scenario variant that uses an injected Mogo map API."""

    def __init__(
        self,
        scene: Scene,
        map_root: str = "",
        map_version: str = "mogo-sqlite",
        ego_vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    ) -> None:
        """Initialize scenario with map api already attached to the Scene."""
        super().__init__(
            scene=scene,
            map_root=map_root,
            map_version=map_version,
            ego_vehicle_parameters=ego_vehicle_parameters,
        )
        self._mogo_map_api = scene.map_api

    def __reduce__(self) -> Tuple[Type[MogoNavSimScenario], Tuple[Any, ...]]:
        """Hints on how to reconstruct the object when pickling."""
        return (
            self.__class__,
            (
                self._scene,
                self._map_root,
                self._map_version,
                self._ego_vehicle_parameters,
            ),
        )

    @property
    def map_api(self) -> AbstractMap:
        """Return the injected Mogo SQLite map adapter."""
        return self._mogo_map_api
