from types import SimpleNamespace

import numpy as np

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from scripts.lane_center_trajectory_polish import (
    LaneCenterPolishConfig,
    extract_left_lane_centerline,
    polish_trajectories_to_left_lane_center,
)


class _MapObject:
    def __init__(self, object_id: str, points: list[tuple[float, float]]) -> None:
        self.id = object_id
        states = [StateSE2(x, y, 0.0) for x, y in points]
        self.baseline_path = SimpleNamespace(discrete_path=states)


class _MapAPI:
    def __init__(self, lanes: list[_MapObject]) -> None:
        self._lanes = lanes

    def get_proximal_map_objects(self, point, radius, layers):
        del point, radius
        return {
            layer: self._lanes if layer == SemanticMapLayer.LANE else []
            for layer in layers
        }


def _straight_poses(endpoint_x: list[float], horizon: int = 8) -> np.ndarray:
    fractions = np.arange(1, horizon + 1, dtype=np.float32) / horizon
    poses = np.zeros((len(endpoint_x), horizon, 3), dtype=np.float32)
    poses[..., 0] = np.asarray(endpoint_x, dtype=np.float32)[:, None] * fractions
    return poses


def test_extracts_nearest_left_centerline_with_reverse_map_direction() -> None:
    map_api = _MapAPI(
        [
            _MapObject("current", [(-20.0, 0.0), (40.0, 0.0)]),
            _MapObject("left-a", [(10.0, 4.0), (-20.0, 4.0)]),
            _MapObject("left-b", [(40.0, 4.1), (10.0, 4.0)]),
            _MapObject("second-left", [(-20.0, 5.8), (40.0, 5.8)]),
            _MapObject("right", [(-20.0, -4.0), (40.0, -4.0)]),
            _MapObject("crossing", [(0.0, 3.5), (0.0, 20.0)]),
        ]
    )
    config = LaneCenterPolishConfig(join_lateral_tolerance_m=1.0)

    centerline = extract_left_lane_centerline(
        map_api, np.array([0.0, 0.0, 0.0]), config
    )

    assert centerline is not None
    assert centerline.object_ids == ("left-a", "left-b")
    assert centerline.xy[0, 0] == -20.0
    assert centerline.xy[-1, 0] == 40.0
    np.testing.assert_allclose(centerline.xy[:, 1], 4.0, atol=0.1)


def test_lift_blends_in_time_then_follows_curved_centerline() -> None:
    poses = _straight_poses([16.0] * 7)
    centerline = np.array([[-5.0, 4.0], [20.0, 6.5]], dtype=np.float64)
    config = LaneCenterPolishConfig(
        temporal_radius=0,
        lift_start_frame=2,
        lift_transition_frames=2,
        horizon_blend_end=0.625,
    )

    polished = polish_trajectories_to_left_lane_center(
        poses, [centerline] * len(poses), config
    )

    np.testing.assert_allclose(polished[..., 0], poses[..., 0])
    np.testing.assert_allclose(polished[:3, :, 1], 0.0)
    np.testing.assert_allclose(polished[3, -1, 1], 3.05, atol=1e-6)
    np.testing.assert_allclose(polished[4:, -1, 1], 6.1, atol=1e-6)
    assert 0.0 < polished[4, 0, 1] < polished[4, 4, 1]
    np.testing.assert_allclose(polished[4:, 4, 1], 5.5, atol=1e-6)
    assert polished[4, -1, 2] > 0.0


def test_temporal_filter_removes_short_length_spike() -> None:
    poses = _straight_poses([12.0, 12.0, 5.0, 12.0, 12.0])
    centerline = np.array([[-5.0, 4.0], [20.0, 4.0]], dtype=np.float64)

    polished = polish_trajectories_to_left_lane_center(
        poses,
        [centerline] * len(poses),
        LaneCenterPolishConfig(
            temporal_radius=2,
            lift_start_frame=0,
            lift_transition_frames=0,
        ),
    )

    np.testing.assert_allclose(polished[:, -1, 0], 12.0)
    np.testing.assert_allclose(polished[:, -1, 1], 4.0)
