from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import ReferenceSignalIndex, StateIndex
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr_utils import (
    get_velocity_curvature_profiles_with_derivatives_from_poses,
)

# TODO: Refactor & add to config

# (1) ego_jerk_metric,
max_abs_mag_jerk = 6.0  # [m/s^3]

# (2) ego_lat_acceleration_metric
max_abs_lat_accel = 2.1246  # [m/s^2]

# (3) ego_lon_acceleration_metric
max_lon_accel = 1.5  # [m/s^2]
min_lon_accel = -4.05

# (4) ego_yaw_acceleration_metric
max_abs_yaw_accel = 1.93  # [rad/s^2]

# (5) ego_lon_jerk_metric
max_abs_lon_jerk = 4.13  # [m/s^3]

# (6) ego_yaw_rate_metric
max_abs_yaw_rate = 0.95  # [rad/s]

# (7) ego_centripetal_acceleration_metric
max_abs_centripetal_accel = 2.1246  # [m/s^2]

# (8) ego_centripetal_jerk_metric
max_abs_centripetal_jerk = 2.5578  # [m/s^3]

# (9) start_anchor_comfort_metric
# Anchor bounds start stricter near the current state and relax to comfort v2 bounds at the end of the anchor horizon.
start_anchor_min_lon_accel = min_lon_accel
start_anchor_max_lon_accel = max_lon_accel
start_anchor_max_abs_lon_jerk = max_abs_lon_jerk  # [m/s^3]


def _get_centripetal_acceleration_bounds(
    velocity: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute velocity-dependent bounds for centripetal acceleration.
    :param velocity: longitudinal velocity [m/s]
    :return: minimum and maximum centripetal acceleration bounds
    """
    abs_velocity = np.abs(velocity)
    max_bound = np.clip(
        max_abs_centripetal_accel - 0.01 * abs_velocity - 0.0005 * abs_velocity**2,
        1.5,
        max_abs_centripetal_accel,
    )
    min_bound = -max_bound
    return min_bound, max_bound


def _get_centripetal_jerk_bounds(
    velocity: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute velocity-dependent bounds for centripetal jerk.
    :param velocity: longitudinal velocity [m/s]
    :return: minimum and maximum centripetal jerk bounds
    """
    abs_velocity = np.abs(velocity)
    max_bound = np.clip(
        max_abs_centripetal_jerk - 0.03 * abs_velocity - 0.0005 * abs_velocity**2,
        1.8,
        max_abs_centripetal_jerk,
    )
    min_bound = -max_bound
    return min_bound, max_bound


def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float32]:
    """
    Extract acceleration of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param decimals: decimal precision, defaults to 8
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 8
    :raises ValueError: when coordinate not available
    :return: array containing acceleration values
    """

    n_batch, n_time, n_states = states.shape
    if acceleration_coordinate == "x":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_X]

    elif acceleration_coordinate == "y":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_Y]

    elif acceleration_coordinate == "magnitude":
        acceleration: npt.NDArray[np.float64] = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(
            f"acceleration_coordinate option: {acceleration_coordinate} not available. "
            f"Available options are: x, y or magnitude"
        )

    acceleration = savgol_filter(
        acceleration,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract jerk of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param time_steps_s: time steps [s] of time dim
    :param decimals: decimal precision, defaults to 8
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 15
    :return: array containing jerk values
    """
    n_batch, n_time, n_states = states.shape
    ego_acceleration = _extract_ego_acceleration(states, acceleration_coordinate=acceleration_coordinate)
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    jerk = np.round(jerk, decimals=decimals)
    return jerk


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract yaw-rate of simulation history over batch-dim
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param decimals:  decimal precision, defaults to 8
    :param window_length: window size for extraction, defaults to 15
    :return: array containing ego's yaw rate
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
    )  # convert to seconds
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate


def _extract_ego_curvature(
    reference_signal: npt.NDArray[np.float64],
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract curvature of reference signal over batch-dim
    :param reference_signal: array representation of reference signals
    :param decimals: decimal precision, defaults to 8
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 15
    :return: array containing curvature values
    """
    n_batch, n_time, n_signals = reference_signal.shape
    curvature: npt.NDArray[np.float64] = reference_signal[..., ReferenceSignalIndex.CURVATURE]

    curvature = savgol_filter(
        curvature,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    curvature = np.round(curvature, decimals=decimals)
    return curvature


def _extract_ego_curvature_rate(
    reference_signal: npt.NDArray[np.float64],
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract curvature rate of reference signal over batch-dim
    :param reference_signal: array representation of reference signals
    :param decimals: decimal precision, defaults to 8
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 15
    :return: array containing curvature rate values
    """
    n_batch, n_time, n_signals = reference_signal.shape
    curvature_rate: npt.NDArray[np.float64] = reference_signal[..., ReferenceSignalIndex.CURVATURE_RATE]

    curvature_rate = savgol_filter(
        curvature_rate,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    curvature_rate = np.round(curvature_rate, decimals=decimals)
    return curvature_rate


def _phase_unwrap(headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative


def _within_bound(
    metric: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    """
    Determines wether values in batch-dim are within bounds.
    :param metric: metric values
    :param min_bound: minimum bound, defaults to None
    :param max_bound: maximum bound, defaults to None
    :return: array of booleans wether metric values are within bounds
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    metric_within_bound = (metric_values > min_bound) & (metric_values < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _within_time_varying_bound(
    metric: npt.NDArray[np.float64],
    min_bound: npt.NDArray[np.float64],
    max_bound: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Determines whether time-varying metric values in batch-dim are within bounds.
    """
    metric_within_bound = (metric > min_bound) & (metric < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _interpolate_anchor_bounds(
    min_bound: npt.NDArray[np.float64],
    max_bound: npt.NDArray[np.float64],
    num_steps: int,
    strict_ratio: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Interpolates anchor bounds from strict near the current state to relaxed over the anchor horizon.
    """
    alpha = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    if min_bound.ndim > 0:
        alpha = alpha[None, :]

    strict_min_bound = min_bound * strict_ratio
    strict_max_bound = max_bound * strict_ratio
    anchor_min_bound = strict_min_bound * (1.0 - alpha) + min_bound * alpha
    anchor_max_bound = strict_max_bound * (1.0 - alpha) + max_bound * alpha
    return anchor_min_bound, anchor_max_bound


def _extract_lon_acceleration_metric(
    states: npt.NDArray[np.float64],
    reference_signal: Optional[npt.NDArray[np.float64]],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Extract longitudinal acceleration metric values over batch-dim.
    """
    n_batch, n_time, n_states = states.shape
    return _extract_ego_acceleration(states, acceleration_coordinate="x", window_length=n_time)


def _extract_lon_jerk_metric(
    states: npt.NDArray[np.float64],
    reference_signal: Optional[npt.NDArray[np.float64]],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Extract longitudinal jerk metric values over batch-dim.
    """
    n_batch, n_time, n_states = states.shape
    return _extract_ego_jerk(
        states,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )


def _extract_centripetal_acceleration_metric(
    states: npt.NDArray[np.float64],
    reference_signal: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Extract centripetal acceleration metric values over batch-dim.
    """
    velocity = states[:, :, StateIndex.VELOCITY_X]
    curvature = _extract_ego_curvature(reference_signal, window_length=reference_signal.shape[1])
    return velocity**2 * curvature


def _extract_centripetal_jerk_metric(
    states: npt.NDArray[np.float64],
    reference_signal: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Extract centripetal jerk metric values over batch-dim.
    """
    centripetal_jerk_mode = "simplified"  # available modes: "simplified" and "full"

    velocity = states[:, :, StateIndex.VELOCITY_X]
    acceleration = states[:, :, StateIndex.ACCELERATION_X]
    curvature = _extract_ego_curvature(reference_signal, window_length=reference_signal.shape[1])
    curvature_rate = _extract_ego_curvature_rate(reference_signal, window_length=reference_signal.shape[1])

    if centripetal_jerk_mode == "simplified":
        return velocity**2 * curvature_rate
    if centripetal_jerk_mode == "full":
        return 2 * velocity * acceleration * curvature + velocity**2 * curvature_rate
    raise ValueError(f"Unknown centripetal_jerk_mode: {centripetal_jerk_mode}")


def _build_reference_signal_from_states(
    states: npt.NDArray[np.float64],
    discretization_time: float,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> npt.NDArray[np.float64]:
    """
    Builds node-aligned reference signals from anchor states using the same profile utility as LQR tracking.
    """
    _, _, curvature_profile, curvature_rate_profile = get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=discretization_time,
        poses=states[..., StateIndex.STATE_SE2],
        jerk_penalty=jerk_penalty,
        curvature_rate_penalty=curvature_rate_penalty,
    )

    reference_signal = np.zeros((states.shape[0], states.shape[1], len(ReferenceSignalIndex)), dtype=np.float64)

    valid_curvature_len = min(reference_signal.shape[1], curvature_profile.shape[1])
    if valid_curvature_len > 0:
        reference_signal[:, :valid_curvature_len, ReferenceSignalIndex.CURVATURE] = curvature_profile[
            :, :valid_curvature_len
        ]
        reference_signal[:, valid_curvature_len:, ReferenceSignalIndex.CURVATURE] = curvature_profile[
            :, valid_curvature_len - 1 : valid_curvature_len
        ]

    valid_rate_len = min(reference_signal.shape[1], curvature_rate_profile.shape[1])
    if valid_rate_len > 0:
        reference_signal[:, :valid_rate_len, ReferenceSignalIndex.CURVATURE_RATE] = curvature_rate_profile[
            :, :valid_rate_len
        ]
        reference_signal[:, valid_rate_len:, ReferenceSignalIndex.CURVATURE_RATE] = curvature_rate_profile[
            :, valid_rate_len - 1 : valid_rate_len
        ]

    return reference_signal


def _compute_lon_acceleration(
    states: npt.NDArray[np.float64],
    reference_signal: Optional[npt.NDArray[np.float64]],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Compute longitudinal acceleration over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param reference_signal: array representation of reference signals
    :param time_steps_s: time steps [s] of time dim
    :return: longitudinal acceleration within bound
    """
    lon_acceleration = _extract_lon_acceleration_metric(states, reference_signal, time_steps_s)
    return _within_bound(lon_acceleration, min_bound=min_lon_accel, max_bound=max_lon_accel)


def _compute_lat_acceleration(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute lateral acceleration over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: lateral acceleration within bound
    """
    n_batch, n_time, n_states = states.shape
    lat_acceleration = _extract_ego_acceleration(states, acceleration_coordinate="y", window_length=n_time)
    return _within_bound(lat_acceleration, min_bound=-max_abs_lat_accel, max_bound=max_abs_lat_accel)


def _compute_centripetal_acceleration(
    states: npt.NDArray[np.float64],
    reference_signal: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Compute centripetal acceleration over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param reference_signal: array representation of reference signals
    :param time_steps_s: time steps [s] of time dim
    :return: centripetal acceleration within bound
    """
    velocity = states[:, :, StateIndex.VELOCITY_X]
    centripetal_acceleration = _extract_centripetal_acceleration_metric(states, reference_signal, time_steps_s)
    min_bound, max_bound = _get_centripetal_acceleration_bounds(velocity)
    metric_within_bound = (centripetal_acceleration > min_bound) & (centripetal_acceleration < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _compute_jerk_metric(
    states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """
    Compute absolute jerk over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: absolute jerk within bound
    """
    n_batch, n_time, n_states = states.shape
    jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    return _within_bound(jerk_metric, min_bound=-max_abs_mag_jerk, max_bound=max_abs_mag_jerk)


def _compute_lon_jerk_metric(
    states: npt.NDArray[np.float64],
    reference_signal: Optional[npt.NDArray[np.float64]],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Compute longitudinal jerk over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param reference_signal: array representation of reference signals
    :param time_steps_s: time steps [s] of time dim
    :return: longitudinal jerk within bound
    """
    lon_jerk_metric = _extract_lon_jerk_metric(states, reference_signal, time_steps_s)
    return _within_bound(lon_jerk_metric, min_bound=-max_abs_lon_jerk, max_bound=max_abs_lon_jerk)


def _compute_centripetal_jerk_metric(
    states: npt.NDArray[np.float64],
    reference_signal: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Compute centripetal jerk over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param reference_signal: array representation of reference signals
    :param time_steps_s: time steps [s] of time dim
    :return: centripetal jerk within bound
    """
    velocity = states[:, :, StateIndex.VELOCITY_X]
    centripetal_jerk = _extract_centripetal_jerk_metric(states, reference_signal, time_steps_s)
    min_bound, max_bound = _get_centripetal_jerk_bounds(velocity)
    metric_within_bound = (centripetal_jerk > min_bound) & (centripetal_jerk < max_bound)
    return np.all(metric_within_bound, axis=-1)


def _compute_yaw_accel(states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Compute acceleration of yaw-angle over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: acceleration of yaw-angle within bound
    """
    n_batch, n_time, n_states = states.shape
    yaw_accel_metric = _extract_ego_yaw_rate(states, time_steps_s, deriv_order=2, poly_order=3, window_length=n_time)
    return _within_bound(yaw_accel_metric, min_bound=-max_abs_yaw_accel, max_bound=max_abs_yaw_accel)


def _compute_yaw_rate(states: npt.NDArray[np.float64], time_steps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Compute velocity of yaw-angle over batch-dim of simulated proposals
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :return: velocity of yaw-angle within bound
    """
    n_batch, n_time, n_states = states.shape
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s, window_length=n_time)
    return _within_bound(yaw_rate_metric, min_bound=-max_abs_yaw_rate, max_bound=max_abs_yaw_rate)


def ego_is_comfortable(states: npt.NDArray[np.float64], time_point_s: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """
    Accumulates all within-bound comfortability metrics
    :param states: array representation of ego state values
    :param time_point_s: time steps [s] of time dim
    :return: _description_
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == StateIndex.size()

    results: npt.NDArray[np.bool_] = np.zeros((n_batch, 6), dtype=np.bool_)
    results[:, 0] = _compute_lon_acceleration(states, None, time_point_s)
    results[:, 1] = _compute_lat_acceleration(states, time_point_s)
    results[:, 2] = _compute_jerk_metric(states, time_point_s)
    results[:, 3] = _compute_lon_jerk_metric(states, None, time_point_s)
    results[:, 4] = _compute_yaw_accel(states, time_point_s)
    results[:, 5] = _compute_yaw_rate(states, time_point_s)

    return results


def ego_is_start_anchor_comfortable(
    anchor_states: npt.NDArray[np.float64],
    anchor_time_point_s: npt.NDArray[np.float64],
    check_start_idx: int,
    check_steps: int,
    strict_ratio: float,
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> npt.NDArray[np.bool_]:
    """
    Checks start-window comfort using historical left context.

    Metrics are computed on the full history + start window, then checked at the
    right endpoint nodes of the start window. This keeps edge-like metrics aligned
    with their right endpoint while reusing the same extraction logic as comfort v2.
    """
    n_batch, n_time, n_states = anchor_states.shape
    assert n_time == len(anchor_time_point_s)
    assert n_states == StateIndex.size()
    assert 0 <= check_start_idx < n_time

    check_end_idx = min(check_start_idx + check_steps, n_time)
    check_slice = slice(check_start_idx, check_end_idx)
    num_check_steps = check_end_idx - check_start_idx

    if num_check_steps <= 0 or n_time < 4:
        return np.ones((n_batch, 4), dtype=np.bool_)

    dt = float(np.mean(np.diff(anchor_time_point_s)))
    anchor_reference_signal = _build_reference_signal_from_states(
        anchor_states,
        dt,
        jerk_penalty,
        curvature_rate_penalty,
    )

    results = np.zeros((n_batch, 4), dtype=np.bool_)

    lon_acceleration = _extract_lon_acceleration_metric(
        anchor_states,
        anchor_reference_signal,
        anchor_time_point_s,
    )[:, check_slice]
    min_bound, max_bound = _interpolate_anchor_bounds(
        np.array(start_anchor_min_lon_accel, dtype=np.float64),
        np.array(start_anchor_max_lon_accel, dtype=np.float64),
        num_check_steps,
        strict_ratio,
    )
    results[:, 0] = _within_time_varying_bound(lon_acceleration, min_bound, max_bound)

    lon_jerk = _extract_lon_jerk_metric(
        anchor_states,
        anchor_reference_signal,
        anchor_time_point_s,
    )[:, check_slice]
    min_bound, max_bound = _interpolate_anchor_bounds(
        np.array(-start_anchor_max_abs_lon_jerk, dtype=np.float64),
        np.array(start_anchor_max_abs_lon_jerk, dtype=np.float64),
        num_check_steps,
        strict_ratio,
    )
    results[:, 1] = _within_time_varying_bound(lon_jerk, min_bound, max_bound)

    centripetal_acceleration = _extract_centripetal_acceleration_metric(
        anchor_states,
        anchor_reference_signal,
        anchor_time_point_s,
    )[:, check_slice]
    velocity = anchor_states[:, check_slice, StateIndex.VELOCITY_X]
    min_bound, max_bound = _get_centripetal_acceleration_bounds(velocity)
    min_bound, max_bound = _interpolate_anchor_bounds(
        min_bound,
        max_bound,
        num_check_steps,
        strict_ratio,
    )
    results[:, 2] = _within_time_varying_bound(centripetal_acceleration, min_bound, max_bound)

    centripetal_jerk = _extract_centripetal_jerk_metric(
        anchor_states,
        anchor_reference_signal,
        anchor_time_point_s,
    )[:, check_slice]
    min_bound, max_bound = _get_centripetal_jerk_bounds(velocity)
    min_bound, max_bound = _interpolate_anchor_bounds(
        min_bound,
        max_bound,
        num_check_steps,
        strict_ratio,
    )
    results[:, 3] = _within_time_varying_bound(centripetal_jerk, min_bound, max_bound)

    return results


def ego_is_comfortable_v2(
    states: npt.NDArray[np.float64],
    reference_signal: npt.NDArray[np.float64],
    time_point_s: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """
    Accumulates all within-bound comfortability metrics
    :param states: array representation of ego state values
    :param reference_signal: array representation of reference signals
    :param time_point_s: time steps [s] of time dim
    :return: _description_
    """
    n_batch, n_time, n_states = states.shape
    assert n_time == len(time_point_s)
    assert n_states == StateIndex.size()
    assert reference_signal.shape[0] == n_batch
    assert reference_signal.shape[1] == n_time
    assert reference_signal.shape[2] == len(ReferenceSignalIndex)

    comfort_metric_functions = [
        _compute_lon_acceleration,
        _compute_lon_jerk_metric,
        _compute_centripetal_acceleration,
        _compute_centripetal_jerk_metric,
    ]
    results: npt.NDArray[np.bool_] = np.zeros((n_batch, len(comfort_metric_functions)), dtype=np.bool_)
    for idx, metric_function in enumerate(comfort_metric_functions):
        results[:, idx] = metric_function(states, reference_signal, time_point_s)

    return results
