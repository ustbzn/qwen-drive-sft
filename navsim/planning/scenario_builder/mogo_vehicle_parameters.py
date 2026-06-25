import math
from dataclasses import dataclass

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters


@dataclass(frozen=True)
class MogoDynamicParameters:
    """Dynamic limits for the MOGO ego vehicle."""

    weight: float = 4600.0
    accel_min: float = -8.0
    accel_max: float = 2.0
    steering_min_deg: float = -900.0
    steering_max_deg: float = 900.0
    steer_ratio: float = 21.0
    max_abs_speed_when_stopped: float = 0.15
    throttle_deadzone: float = 0.0
    brake_deadzone: float = 0.1

    @property
    def decel_max(self) -> float:
        return abs(self.accel_min)

    @property
    def max_wheel_steering_angle(self) -> float:
        return math.radians(max(abs(self.steering_min_deg), abs(self.steering_max_deg)) / self.steer_ratio)


def get_mogo_vehicle_parameters() -> VehicleParameters:
    """Return the ego vehicle parameters used by the MOGO dataset."""
    return VehicleParameters(
        vehicle_name="mogo",
        vehicle_type="bus",
        width=2.100,
        front_length=4.868,
        rear_length=1.127,
        wheel_base=3.790,
        cog_position_from_rear_axle=(5.995 / 2.0) - 1.127,
        height=2.830,
    )


def get_mogo_dynamic_parameters() -> MogoDynamicParameters:
    """Return non-geometric dynamic limits for the MOGO ego vehicle."""
    return MogoDynamicParameters()
