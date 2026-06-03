from __future__ import annotations

from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_states_to_state_array


def build_history_states_from_scenario(
    scenario: AbstractScenario,
    history_horizon: float,
    interval_length: float,
) -> Optional[npt.NDArray[np.float64]]:
    """
    Builds right-aligned historical ego states before the current scenario iteration.
    """
    if history_horizon <= 0.0 or interval_length <= 0.0:
        return None

    num_samples = int(round(history_horizon / interval_length))
    if num_samples <= 0:
        return None

    scenario_interval = scenario.database_interval
    if scenario_interval <= 0.0:
        return None

    sample_stride = max(1, int(round(interval_length / scenario_interval)))
    if not np.isclose(sample_stride * scenario_interval, interval_length):
        raise ValueError(
            "History interval must be an integer multiple of scenario.database_interval: "
            f"history_interval={interval_length}, scenario_interval={scenario_interval}"
        )

    source_num_samples = num_samples * sample_stride
    history_ego_states = list(
        scenario.get_ego_past_trajectory(
            iteration=0,
            time_horizon=history_horizon,
            num_samples=source_num_samples,
        )
    )
    if len(history_ego_states) < source_num_samples:
        return None

    sampled_history_ego_states = history_ego_states[-source_num_samples::sample_stride]
    if len(sampled_history_ego_states) < num_samples:
        return None

    return ego_states_to_state_array(sampled_history_ego_states[-num_samples:])


class StateInterpolator:
    """Helper function to interpolate detections for metric caching."""

    def __init__(self, state_array: npt.NDArray[np.float64]):
        """
        Initializes the state interpolator.
        :param state_array: numpy array with detection state
        """

        # attribute
        self._state_array = state_array

        # loaded during initialization
        self._time = state_array[:, 0]
        self._states = state_array[:, 1:]

        # unwrap heading angle
        self._states[:, 2] = np.unwrap(self._states[:, 2], axis=0)
        self._interpolator = interp1d(self._time, self._states, axis=0)

    def __reduce__(self) -> Tuple[Type[StateInterpolator], Tuple[Any, ...]]:
        """Helper for pickling."""
        return self.__class__, (self.state_array,)

    @property
    def start_time(self) -> float:
        """
        :return: start time for interpolation
        """
        return self._time[0]

    @property
    def end_time(self) -> float:
        """
        :return: end time for interpolation
        """
        return self._time[-1]

    def interpolate(self, time: float) -> Union[npt.NDArray[np.object_], npt.NDArray[np.float64]]:
        """
        Temporally interpolate state array
        :param time: time step to retrieve state
        :return: interpolated state
        """
        if self.start_time <= time <= self.end_time:

            interpolated_state = self._interpolator(time)
            interpolated_state[2] = normalize_angle(interpolated_state[2])
            return interpolated_state

        return None
