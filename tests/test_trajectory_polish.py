import numpy as np
import pytest

from scripts.trajectory_polish import TrajectoryPolishConfig, polish_trajectory_poses


def _straight_poses(endpoint_x: list[float], horizon: int = 4) -> np.ndarray:
    fractions = np.linspace(1.0 / horizon, 1.0, horizon, dtype=np.float32)
    poses = np.zeros((len(endpoint_x), horizon, 3), dtype=np.float32)
    poses[..., 0] = np.asarray(endpoint_x, dtype=np.float32)[:, None] * fractions
    return poses


def test_temporal_smoothing_removes_single_frame_length_drop() -> None:
    poses = _straight_poses([12.0, 12.0, 12.0, 5.0, 12.0, 12.0, 12.0])

    polished = polish_trajectory_poses(
        poses,
        TrajectoryPolishConfig(temporal_radius=2),
    )

    np.testing.assert_allclose(polished[:, -1, 0], 12.0)
    np.testing.assert_allclose(polished[..., 1:], 0.0)


def test_left_lift_is_gradual_in_frame_and_horizon() -> None:
    poses = _straight_poses([10.0] * 7)

    polished = polish_trajectory_poses(
        poses,
        TrajectoryPolishConfig(
            temporal_radius=0,
            left_offset_m=1.2,
            lift_start_frame=2,
            lift_transition_frames=2,
            horizon_ramp_start=0.25,
        ),
    )

    np.testing.assert_allclose(polished[:3, :, 1], 0.0)
    np.testing.assert_allclose(polished[3, -1, 1], 0.6)
    np.testing.assert_allclose(polished[4:, -1, 1], 1.2)
    np.testing.assert_allclose(polished[4:, 0, 1], 0.0)
    assert np.all(np.diff(polished[4, :, 1]) >= 0.0)
    assert polished[4, -1, 2] > 0.0
    np.testing.assert_allclose(polished[..., 0], poses[..., 0])


def test_rejects_invalid_pose_shape() -> None:
    with pytest.raises(ValueError, match=r"\[frame, horizon, 3\]"):
        polish_trajectory_poses(
            np.zeros((8, 3), dtype=np.float32),
            TrajectoryPolishConfig(),
        )
