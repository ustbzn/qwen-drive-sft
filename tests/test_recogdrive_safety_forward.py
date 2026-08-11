import torch
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
)


def _map_batch():
    batch_size, map_tokens, points = 1, 2, 4
    point_position = torch.zeros((batch_size, map_tokens, 3, points, 2))
    point_vector = torch.zeros_like(point_position)
    point_position[0, 0, 0, :, 0] = torch.arange(float(points))
    point_vector[0, 0, 0, :, 0] = 1.0
    return {
        "map_point_position": point_position,
        "map_point_vector": point_vector,
        "map_point_orientation": torch.zeros(
            (batch_size, map_tokens, 3, points)
        ),
        "map_polygon_center": torch.zeros((batch_size, map_tokens, 3)),
        "map_polygon_speed_limit": torch.zeros((batch_size, map_tokens)),
        "map_point_valid_mask": torch.tensor([[[1.0] * points, [0.0] * points]]),
        "map_valid_mask": torch.tensor([[1.0, 0.0]]),
        "map_polygon_type": torch.tensor([[0, 2]]),
        "map_polygon_on_route": torch.tensor([[1, 2]]),
        "map_polygon_tl_status": torch.tensor([[4, 4]]),
        "map_polygon_control_type": torch.tensor([[0, 0]]),
        "map_polygon_tl_direction": torch.tensor([[0, 0]]),
        "map_polygon_has_speed_limit": torch.tensor([[False, False]]),
    }


def test_planner_forward_backpropagates_total_explicit_safety_loss():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        use_hdmap_vector=True,
        use_explicit_safety_loss=True,
    )
    config.use_vlm = False
    planner = ReCogDriveDiffusionPlanner(config).train()

    action_data = {
        "status_feature": torch.zeros((1, 13)),
        "action": torch.zeros((1, 8, 3)),
        **_map_batch(),
    }
    future_boxes = torch.zeros((1, 8, 2, 5))
    future_boxes[..., 0, :] = torch.tensor([20.0, 0.0, 0.0, 4.0, 2.0])
    safety_targets = BatchFeature(
        data={
            "future_obj_boxes": future_boxes,
            "future_obj_valid_mask": torch.tensor(
                [[[True, False]] * 8], dtype=torch.bool
            ),
            "drivable_sdf": torch.full((1, 1, 41, 81), 10.0),
            "drivable_sdf_origin": torch.tensor([[-20.0, -20.0]]),
            "drivable_sdf_resolution": torch.tensor([1.0]),
            "drivable_sdf_valid": torch.tensor([True]),
        }
    )

    output = planner(
        None,
        None,
        BatchFeature(data=action_data),
        safety_targets=safety_targets,
    )

    assert set(output) == {
        "loss",
        "diffusion_loss",
        "collision_loss",
        "offroad_loss",
        "route_loss",
    }
    assert all(torch.isfinite(value) for value in output.values())
    output.loss.backward()
    assert planner.action_decoder.fc2.weight.grad is not None
    assert torch.isfinite(planner.action_decoder.fc2.weight.grad).all()
