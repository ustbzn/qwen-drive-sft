from typing import Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import BatchKinematicBicycleModel
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import ReferenceSignalIndex


class PDMSimulator:
    """
    Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation.
    """

    def __init__(self, proposal_sampling: TrajectorySampling):
        """
        Constructor of PDMSimulator.
        :param proposal_sampling: Sampling parameters for proposals
        """
        if proposal_sampling.num_poses <= 0:
            raise ValueError("PDMSimulator requires proposal_sampling.num_poses > 0.")

        # time parameters
        self.proposal_sampling = proposal_sampling

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()
        self._tracker = BatchLQRTracker()

    def simulate_proposals(
        self, states: npt.NDArray[np.float64], initial_ego_state: EgoState
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Simulate all proposals over batch-dim
        :param initial_ego_state: ego-vehicle state at current iteration
        :param states: proposal states as array
        :return: simulated proposal states and reference signals over the rollout horizon
        """

        # TODO: find cleaner way to load parameters
        # set parameters of motion model and tracker
        self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        self._tracker._discretization_time = self.proposal_sampling.interval_length

        proposal_states = states[:, : self.proposal_sampling.num_poses + 1]
        self._tracker.update(proposal_states)

        # state array representation for simulated vehicle states
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = ego_state_to_state_array(initial_ego_state)
        simulated_reference_signals = np.zeros(
            (len(proposal_states), self.proposal_sampling.num_poses + 1, len(ReferenceSignalIndex)),
            dtype=np.float64,
        )

        # timing objects
        current_time_point = initial_ego_state.time_point
        delta_time_point = TimeDuration.from_s(self.proposal_sampling.interval_length)

        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + delta_time_point, 1)

        for time_idx in range(1, self.proposal_sampling.num_poses + 1):
            sampling_time: TimePoint = next_iteration.time_point - current_iteration.time_point

            command_states, reference_signals = self._tracker.track_trajectory(
                current_iteration,
                next_iteration,
                simulated_states[:, time_idx - 1],
            )
            simulated_reference_signals[:, time_idx - 1] = reference_signals

            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,
            )

            current_iteration = next_iteration
            next_iteration = SimulationIteration(current_iteration.time_point + delta_time_point, 1 + time_idx)

        # Repeat the last valid reference value so reference signals align with the node-aligned state time axis.
        simulated_reference_signals[:, -1] = simulated_reference_signals[:, -2]

        return simulated_states, simulated_reference_signals
