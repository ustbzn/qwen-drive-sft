from types import SimpleNamespace

import torch

from navsim.agents.recogdrive.recogdrive_agent import (
    ReCogDriveAgent,
    make_recogdrive_config,
)
from navsim.agents.recogdrive.safety_targets import (
    SAFETY_TARGET_KEYS,
    ReCogDriveSafetyTargetBuilder,
)
from navsim.planning.script.run_training_recogdrive import custom_collate_fn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def _agent_shell(use_explicit_safety_loss):
    return SimpleNamespace(
        _trajectory_sampling=TrajectorySampling(
            time_horizon=4.0, interval_length=0.5
        ),
        use_explicit_safety_loss=use_explicit_safety_loss,
        max_future_objects=64,
        safety_sdf_x_range=(-10.0, 80.0),
        safety_sdf_y_range=(-40.0, 40.0),
        safety_sdf_resolution=0.5,
        hdmap_radius=120.0,
    )


def test_agent_registers_safety_builder_only_when_enabled():
    disabled = ReCogDriveAgent.get_target_builders(_agent_shell(False))
    enabled = ReCogDriveAgent.get_target_builders(_agent_shell(True))

    assert [builder.get_unique_name() for builder in disabled] == ["trajectory_target"]
    assert any(isinstance(builder, ReCogDriveSafetyTargetBuilder) for builder in enabled)


def _sample(token):
    features = {
        "history_trajectory": torch.zeros((4, 3)),
        "high_command_one_hot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "status_feature": torch.zeros(13),
    }
    targets = {
        "trajectory": torch.zeros((8, 3)),
        "future_obj_boxes": torch.zeros((8, 4, 5)),
        "future_obj_valid_mask": torch.zeros((8, 4), dtype=torch.bool),
        "drivable_sdf": torch.zeros((1, 5, 7), dtype=torch.float16),
        "drivable_sdf_origin": torch.tensor([-1.0, -1.0]),
        "drivable_sdf_resolution": torch.tensor(0.5),
        "drivable_sdf_valid": torch.tensor(True),
    }
    return features, targets, token


def test_collate_keeps_safety_supervision_out_of_model_features():
    features, targets, tokens = custom_collate_fn(
        [_sample("token-a"), _sample("token-b")]
    )

    assert set(SAFETY_TARGET_KEYS).issubset(targets)
    assert set(SAFETY_TARGET_KEYS).isdisjoint(features)
    assert targets["future_obj_boxes"].shape == (2, 8, 4, 5)
    assert targets["drivable_sdf"].shape == (2, 1, 5, 7)
    assert tokens == ("token-a", "token-b")


def test_config_exposes_explicit_safety_weights():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        use_explicit_safety_loss=True,
        lambda_collision=0.2,
        lambda_offroad=0.3,
        lambda_route=0.4,
    )

    assert config.use_explicit_safety_loss is True
    assert config.lambda_collision == 0.2
    assert config.lambda_offroad == 0.3
    assert config.lambda_route == 0.4
