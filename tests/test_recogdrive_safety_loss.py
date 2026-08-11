import math
from types import SimpleNamespace

import numpy as np
import torch
from shapely.geometry import Polygon

from navsim.agents.recogdrive.safety_loss import (
    collision_clearance_loss,
    drivable_sdf_loss,
    oriented_box_signed_clearance,
    route_alignment_loss,
    _weighted_valid_mean,
)
from navsim.agents.recogdrive.safety_targets import ReCogDriveSafetyTargetBuilder
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


EGO_FRONT_LENGTH = 4.868
EGO_REAR_LENGTH = 1.127
EGO_WIDTH = 2.1


def test_snr_weight_reduces_high_noise_safety_contribution():
    value = torch.tensor([10.0])
    weighted = _weighted_valid_mean(value, torch.tensor([True]), torch.tensor([0.1]))
    assert weighted.item() == 1.0


def test_oriented_box_clearance_has_correct_sign_and_finite_gradient():
    trajectory = torch.zeros((1, 1, 3), requires_grad=True)
    separated = torch.tensor([[[[8.0, 0.0, 0.0, 4.0, 2.0]]]])
    overlapping = torch.tensor([[[[4.0, 0.0, 0.0, 4.0, 2.0]]]])

    separated_clearance = oriented_box_signed_clearance(
        trajectory,
        separated,
        EGO_FRONT_LENGTH,
        EGO_REAR_LENGTH,
        EGO_WIDTH,
    )
    overlapping_clearance = oriented_box_signed_clearance(
        trajectory,
        overlapping,
        EGO_FRONT_LENGTH,
        EGO_REAR_LENGTH,
        EGO_WIDTH,
    )

    assert separated_clearance.item() > 0.5
    assert overlapping_clearance.item() < 0.0

    loss = collision_clearance_loss(
        trajectory,
        overlapping,
        torch.ones((1, 1, 1), dtype=torch.bool),
        EGO_FRONT_LENGTH,
        EGO_REAR_LENGTH,
        EGO_WIDTH,
        safety_margin=0.5,
        smooth_beta=10.0,
    )
    loss.backward()
    assert trajectory.grad is not None
    assert torch.isfinite(trajectory.grad).all()


def _annotations(box_x):
    return SimpleNamespace(
        boxes=np.array([[box_x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]], dtype=np.float32),
        names=["vehicle"],
        velocity_3d=np.zeros((1, 3), dtype=np.float32),
        instance_tokens=["instance"],
        track_tokens=["track"],
    )


def _scene(map_api=None):
    frames = []
    for frame_index in range(9):
        frames.append(
            SimpleNamespace(
                ego_status=SimpleNamespace(
                    ego_pose=np.array([float(frame_index), 0.0, 0.0])
                ),
                annotations=_annotations(5.0),
            )
        )
    return SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=1, num_future_frames=8),
        frames=frames,
        map_api=map_api,
    )


def test_future_object_boxes_are_expressed_in_current_ego_frame():
    builder = ReCogDriveSafetyTargetBuilder(
        TrajectorySampling(time_horizon=4.0, interval_length=0.5),
        max_future_objects=4,
        sdf_x_range=(-10.0, 10.0),
        sdf_y_range=(-10.0, 10.0),
        sdf_resolution=1.0,
    )

    targets = builder.compute_targets(_scene(map_api=None))

    # Future ego is at global x=1 and the object is 5 m ahead in that frame.
    assert targets["future_obj_boxes"][0, 0, 0].item() == 6.0
    assert targets["future_obj_boxes"][-1, 0, 0].item() == 13.0
    assert targets["future_obj_valid_mask"][:, 0].all()
    assert not targets["drivable_sdf_valid"].item()


class _MapObject:
    def __init__(self, polygon):
        self.polygon = polygon


class _MapAPI:
    def __init__(self, polygon):
        self._object = _MapObject(polygon)

    def load(self, point):
        del point

    def get_proximal_map_objects(self, point, radius, layers):
        del point, radius
        return {
            layer: [self._object] if layer == SemanticMapLayer.LANE else []
            for layer in layers
        }


def test_drivable_target_sdf_is_positive_inside_and_negative_outside():
    builder = ReCogDriveSafetyTargetBuilder(
        TrajectorySampling(time_horizon=4.0, interval_length=0.5),
        max_future_objects=4,
        sdf_x_range=(-10.0, 10.0),
        sdf_y_range=(-10.0, 10.0),
        sdf_resolution=1.0,
    )
    polygon = Polygon([(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)])

    targets = builder.compute_targets(_scene(map_api=_MapAPI(polygon)))
    sdf = targets["drivable_sdf"][0].float()

    assert targets["drivable_sdf_valid"].item()
    assert sdf[10, 10].item() > 0.0
    assert sdf[0, 0].item() < 0.0


def _straight_road_sdf():
    x = torch.arange(-20.0, 21.0)
    y = torch.arange(-10.0, 11.0)
    del x
    sdf = (5.0 - y.abs())[:, None].expand(-1, 41)
    return sdf[None, None]


def test_drivable_loss_samples_all_ego_corners_and_backpropagates():
    sdf = _straight_road_sdf()
    origin = torch.tensor([[-20.0, -10.0]])
    resolution = torch.tensor([1.0])
    valid = torch.tensor([True])
    inside = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    outside = torch.tensor([[[0.0, 7.0, 0.0]]], requires_grad=True)

    inside_loss = drivable_sdf_loss(
        inside,
        sdf,
        origin,
        resolution,
        valid,
        EGO_FRONT_LENGTH,
        EGO_REAR_LENGTH,
        EGO_WIDTH,
        safety_margin=0.5,
        smooth_beta=10.0,
    )
    outside_loss = drivable_sdf_loss(
        outside,
        sdf,
        origin,
        resolution,
        valid,
        EGO_FRONT_LENGTH,
        EGO_REAR_LENGTH,
        EGO_WIDTH,
        safety_margin=0.5,
        smooth_beta=10.0,
    )

    assert outside_loss.item() > inside_loss.item() + 1.0
    outside_loss.backward()
    assert outside.grad is not None
    assert torch.isfinite(outside.grad).all()


def _route_map_inputs(polygon_type=0):
    points = torch.zeros((1, 1, 3, 4, 2))
    vectors = torch.zeros_like(points)
    points[0, 0, 0, :, 0] = torch.arange(4.0)
    vectors[0, 0, 0, :, 0] = 1.0
    return dict(
        map_point_position=points,
        map_point_vector=vectors,
        map_polygon_type=torch.tensor([[polygon_type]]),
        map_polygon_on_route=torch.tensor([[1]]),
        map_point_valid_mask=torch.ones((1, 1, 4), dtype=torch.float32),
        map_valid_mask=torch.ones((1, 1), dtype=torch.float32),
    )


def test_route_loss_accepts_float_masks_and_uses_local_lane_tangent():
    route = _route_map_inputs()
    aligned = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    misaligned = torch.tensor(
        [[[0.0, 5.0, math.pi / 2], [1.0, 5.0, math.pi / 2], [2.0, 5.0, math.pi / 2]]],
        requires_grad=True,
    )

    aligned_loss = route_alignment_loss(
        aligned, **route, lateral_margin=1.0, heading_weight=0.5, smooth_beta=10.0
    )
    misaligned_loss = route_alignment_loss(
        misaligned, **route, lateral_margin=1.0, heading_weight=0.5, smooth_beta=10.0
    )

    assert misaligned_loss.item() > aligned_loss.item() + 3.0
    misaligned_loss.backward()
    assert misaligned.grad is not None
    assert torch.isfinite(misaligned.grad).all()


def test_route_loss_ignores_on_route_stop_line_tokens():
    stop_line = _route_map_inputs(polygon_type=3)
    trajectory = torch.zeros((1, 3, 3), requires_grad=True)

    loss = route_alignment_loss(
        trajectory,
        **stop_line,
        lateral_margin=1.0,
        heading_weight=0.5,
        smooth_beta=10.0,
    )

    assert loss.item() == 0.0
