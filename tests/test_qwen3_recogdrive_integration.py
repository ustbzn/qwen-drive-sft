import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.agents.recogdrive import recogdrive_features
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent, make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_backbone import load_qwen3_processor, system_message
from navsim.agents.recogdrive.hdmap_vector_encoder import MapEncoder
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_features import (
    RECOGDRIVE_EGO_FEATURE_KEYS,
    ReCogDriveEgoFeatureBuilder,
    ReCogDriveFeatureBuilder,
    select_hidden_state,
)
from navsim.planning.training.dataset import (
    CacheOnlyDataset,
    Dataset,
    dump_feature_target_to_pickle,
)


def test_make_recogdrive_config_accepts_explicit_vl_hidden_dim():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        vl_hidden_dim=2048,
    )

    assert config.vl_hidden_dim == 2048


def test_make_recogdrive_config_accepts_vl_hidden_normalization_toggle():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        vl_hidden_dim=2048,
        normalize_vl_hidden_states=True,
    )

    assert config.normalize_vl_hidden_states is True


def test_planner_uses_identity_vl_norm_by_default():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        vl_hidden_dim=2048,
    )
    planner = ReCogDriveDiffusionPlanner(config)

    assert isinstance(planner.vl_summary_norm, nn.Identity)
    assert isinstance(planner.vl_context_first_norm, nn.Identity)
    assert isinstance(planner.vl_context_last_norm, nn.Identity)


def test_planner_can_layer_norm_vl_hidden_states():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        vl_hidden_dim=4,
        normalize_vl_hidden_states=True,
    )
    planner = ReCogDriveDiffusionPlanner(config)
    hidden_states = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])

    normalized = planner.vl_summary_norm(hidden_states)

    assert isinstance(planner.vl_summary_norm, nn.LayerNorm)
    assert torch.allclose(normalized.mean(dim=-1), torch.zeros(1, 1), atol=1e-6)
    assert torch.allclose(normalized.std(dim=-1, unbiased=False), torch.ones(1, 1), atol=1e-5)


def test_select_hidden_state_supports_default_and_custom_layer_indices():
    hidden_states = [torch.full((1, 2, 3), float(i)) for i in range(5)]

    assert torch.equal(select_hidden_state(hidden_states, 0), hidden_states[0])
    assert torch.equal(select_hidden_state(hidden_states, -1), hidden_states[-1])
    assert torch.equal(select_hidden_state(hidden_states, -4), hidden_states[1])


def test_qwen3_feature_cache_name_is_model_specific_for_cache_read_and_write(monkeypatch):
    class FakeBackbone:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(recogdrive_features, "RecogDriveBackbone", FakeBackbone)

    write_builder = ReCogDriveFeatureBuilder(
        cache_hidden_state=True,
        model_type="qwen3",
        checkpoint_path="/fake/qwen3",
        cache_mode=True,
    )
    read_builder = ReCogDriveFeatureBuilder(
        cache_hidden_state=True,
        model_type="qwen3",
        checkpoint_path=None,
        cache_mode=False,
    )

    assert write_builder.get_unique_name() == "qwen3_feature"
    assert read_builder.get_unique_name() == "qwen3_feature"


def test_load_qwen3_processor_from_local_model_path():
    processor = load_qwen3_processor("/pcpt/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct")

    assert type(processor).__name__ == "Qwen3VLProcessor"
    assert type(processor.tokenizer).__name__ == "Qwen2Tokenizer"


def test_qwen3_processor_accepts_mogo_image_message():
    processor = load_qwen3_processor("/pcpt/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct")
    image = (
        "/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/sensor_blobs/trainval/"
        "JLBJA20073D_20260313152238/front/1773386553936696.jpg"
    )
    messages = [[
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "<image>\nDescribe this image."},
            ],
        },
    ]]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    assert "pixel_values" in inputs
    assert "image_grid_thw" in inputs
    assert tuple(inputs["input_ids"].shape)[0] == 1


def test_mogo_scene_can_be_built_without_nuplan_map_for_training_cache():
    pkl_path = (
        "/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/navsim_logs/trainval/"
        "JLBJA20073D_20260313152238.pkl"
    )
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)

    scene = Scene.from_scene_dict_list(
        frames[:12],
        sensor_blobs_path=Path("/D2/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/sensor_blobs/trainval"),
        num_history_frames=4,
        num_future_frames=8,
        sensor_config=SensorConfig.build_all_sensors(include=[0, 1, 2, 3]),
        load_image_path=True,
    )

    assert scene.scene_metadata.map_name == "las_vegas"
    assert scene.map_api is None
    assert scene.frames[0].lidar.lidar_pc is None


def _random_map_inputs(batch_size=2, max_map_tokens=4, sample_points=10, num_valid=3):
    """构造 MapEncoder.forward 所需的随机 HDMap 张量；前 num_valid 个 polygon 有效，其余为 padding（三线置零）。"""
    B, M, P = batch_size, max_map_tokens, sample_points
    point_position = torch.randn(B, M, 3, P, 2)
    point_vector = torch.randn(B, M, 3, P, 2)
    point_orientation = torch.randn(B, M, 3, P)
    point_valid_mask = torch.zeros(B, M, P)
    point_valid_mask[:, :num_valid] = 1.0
    map_valid_mask = torch.zeros(B, M)
    map_valid_mask[:, :num_valid] = 1.0
    # padding polygon 三线几何置零，贴近 HDMapVectorFeatureBuilder._empty_feature_dict 的行为
    point_position[:, num_valid:] = 0.0
    point_vector[:, num_valid:] = 0.0
    point_orientation[:, num_valid:] = 0.0
    return dict(
        point_position=point_position,
        point_vector=point_vector,
        point_orientation=point_orientation,
        polygon_type=torch.randint(0, 4, (B, M)),
        polygon_on_route=torch.randint(0, 3, (B, M)),
        polygon_tl_status=torch.randint(0, 5, (B, M)),
        polygon_control_type=torch.randint(0, 3, (B, M)),
        polygon_tl_direction=torch.randint(0, 6, (B, M)),
        polygon_has_speed_limit=torch.randint(0, 2, (B, M)).bool(),
        polygon_speed_limit=torch.randn(B, M).abs(),
        point_valid_mask=point_valid_mask,
        map_valid_mask=map_valid_mask,
    )


def test_map_encoder_boundaries_disabled_keeps_centerline_only():
    encoder = MapEncoder(hidden_size=32, use_lane_boundaries=False)
    assert encoder.centerline_encoder.point_mlp[0].in_features == 6

    inputs = _random_map_inputs()
    tokens = encoder(**inputs)

    B, M = inputs["map_valid_mask"].shape
    assert tokens.shape == (B, M, 32)
    # 无效（padding）polygon 的 token 必须严格为 0
    assert torch.count_nonzero(tokens[inputs["map_valid_mask"] == 0]) == 0


def test_map_encoder_boundaries_enabled_widens_input_and_runs():
    encoder = MapEncoder(hidden_size=32, use_lane_boundaries=True)
    # 中心线 6 维 + 左右边界几何 7 维 = 13
    assert encoder.centerline_encoder.point_mlp[0].in_features == 13

    inputs = _random_map_inputs()
    tokens = encoder(**inputs)

    B, M = inputs["map_valid_mask"].shape
    assert tokens.shape == (B, M, 32)
    assert torch.isfinite(tokens).all()
    # 边界特征不改变 mask 语义：padding polygon 仍严格置零
    assert torch.count_nonzero(tokens[inputs["map_valid_mask"] == 0]) == 0


def test_make_recogdrive_config_accepts_use_lane_boundaries():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        vl_hidden_dim=2048,
        use_hdmap_vector=True,
        use_lane_boundaries=True,
    )
    assert config.use_lane_boundaries is True

    planner = ReCogDriveDiffusionPlanner(config)
    assert planner.map_encoder is not None
    assert planner.map_encoder.centerline_encoder.point_mlp[0].in_features == 13


def _make_fake_agent_input():
    ego_statuses = []
    for index in range(4):
        ego_statuses.append(
            SimpleNamespace(
                ego_pose=[float(index), float(index + 1), float(index) * 0.1],
                driving_command=[1.0, 0.0, 0.0, 0.0],
                ego_velocity=[2.0, 0.5],
                ego_acceleration=[0.1, -0.2],
                ego_yaw_rate=0.01,
                ego_curvature=0.02,
                ego_curvature_rate=0.03,
                ego_steering_wheel_angle=0.04,
                ego_steering_wheel_rate=0.05,
            )
        )
    return SimpleNamespace(ego_statuses=ego_statuses, cameras=object())


def _make_agent_shell(use_vlm: bool, cache_mode: bool = False):
    return SimpleNamespace(
        use_vlm=use_vlm,
        cache_mode=cache_mode,
        cache_hidden_state=False,
        vlm_type="qwen3",
        vlm_path=None,
        device="cpu",
        first_hidden_state_index=0,
        last_hidden_state_index=-1,
        view_list=["cam_f0"],
        num_frames=1,
        use_video_input=False,
        video_fps=2.0,
        use_hdmap_vector=False,
        use_object_vector=False,
        use_image_encoder=False,
    )


def test_slim_ego_builder_matches_full_cache_fields(monkeypatch):
    class FakeBackbone:
        def __init__(self, *args, **kwargs):
            pass

        def forward_from_paths(self, *args, **kwargs):
            return SimpleNamespace(
                hidden_states=(
                    torch.ones(1, 3, 5),
                    torch.full((1, 3, 5), 2.0),
                )
            )

    monkeypatch.setattr(recogdrive_features, "RecogDriveBackbone", FakeBackbone)
    monkeypatch.setattr(
        recogdrive_features,
        "collect_view_frame_paths_with_status",
        lambda *args, **kwargs: ([["fake.jpg"]], False),
    )

    agent_input = _make_fake_agent_input()
    slim_builder = ReCogDriveEgoFeatureBuilder()
    full_builder = ReCogDriveFeatureBuilder(
        cache_hidden_state=True,
        model_type="qwen3",
        checkpoint_path="/fake/qwen3",
        cache_mode=True,
    )
    slim_features = slim_builder.compute_features(agent_input)
    full_features = full_builder.compute_features(agent_input)

    assert slim_builder.get_unique_name() == "recogdrive_ego_feature_v1"
    assert tuple(slim_features) == RECOGDRIVE_EGO_FEATURE_KEYS
    for key in RECOGDRIVE_EGO_FEATURE_KEYS:
        assert torch.equal(slim_features[key], full_features[key])
    assert "first_hidden_state" in full_features
    assert "last_hidden_state" in full_features


def test_agent_selects_full_and_slim_builders_by_mode():
    no_vlm_builders = ReCogDriveAgent.get_feature_builders(_make_agent_shell(use_vlm=False))
    vlm_read_builders = ReCogDriveAgent.get_feature_builders(_make_agent_shell(use_vlm=True))
    cache_write_builders = ReCogDriveAgent.get_feature_builders(
        _make_agent_shell(use_vlm=True, cache_mode=True)
    )

    assert [type(builder) for builder in no_vlm_builders] == [ReCogDriveEgoFeatureBuilder]
    assert [type(builder) for builder in vlm_read_builders] == [ReCogDriveFeatureBuilder]
    assert [type(builder) for builder in cache_write_builders] == [
        ReCogDriveFeatureBuilder,
        ReCogDriveEgoFeatureBuilder,
    ]

    no_vlm_online_agent = _make_agent_shell(use_vlm=False)
    no_vlm_online_agent.view_list = ["cam_f0", "cam_l0"]
    no_vlm_online_agent._is_online_backbone = lambda: ReCogDriveAgent._is_online_backbone(
        no_vlm_online_agent
    )
    assert ReCogDriveAgent._needs_view_encoding(no_vlm_online_agent) is False


class _TestTargetBuilder:
    def __init__(self):
        self.compute_calls = 0

    def get_unique_name(self):
        return "trajectory_target"

    def compute_targets(self, scene):
        self.compute_calls += 1
        return {"trajectory": torch.zeros(8, 3)}


def test_cache_only_no_vlm_reads_slim_without_opening_qwen(tmp_path):
    token_path = tmp_path / "log" / "token"
    token_path.mkdir(parents=True)
    slim_builder = ReCogDriveEgoFeatureBuilder()
    target_builder = _TestTargetBuilder()
    dump_feature_target_to_pickle(
        token_path / "recogdrive_ego_feature_v1.gz",
        slim_builder.compute_features(_make_fake_agent_input()),
    )
    dump_feature_target_to_pickle(
        token_path / "trajectory_target.gz",
        {"trajectory": torch.zeros(8, 3)},
    )
    (token_path / "qwen3_feature.gz").write_bytes(b"intentionally invalid gzip")

    dataset = CacheOnlyDataset(
        cache_path=str(tmp_path),
        feature_builders=[slim_builder],
        target_builders=[target_builder],
    )
    features, targets, token = dataset[0]

    assert token == "token"
    assert tuple(features) == RECOGDRIVE_EGO_FEATURE_KEYS
    assert tuple(targets["trajectory"].shape) == (8, 3)


def test_cache_only_vlm_keeps_legacy_full_cache_compatible(tmp_path):
    token_path = tmp_path / "log" / "token"
    token_path.mkdir(parents=True)
    full_builder = ReCogDriveFeatureBuilder(
        cache_hidden_state=True,
        model_type="qwen3",
        cache_mode=False,
    )
    full_features = ReCogDriveEgoFeatureBuilder().compute_features(_make_fake_agent_input())
    full_features.update(
        {
            "first_hidden_state": torch.zeros(2, 3),
            "last_hidden_state": torch.ones(2, 3),
        }
    )
    dump_feature_target_to_pickle(token_path / "qwen3_feature.gz", full_features)
    dump_feature_target_to_pickle(
        token_path / "trajectory_target.gz",
        {"trajectory": torch.zeros(8, 3)},
    )

    dataset = CacheOnlyDataset(
        cache_path=str(tmp_path),
        feature_builders=[full_builder],
        target_builders=[_TestTargetBuilder()],
    )
    features, _, token = dataset[0]

    assert token == "token"
    assert "first_hidden_state" in features
    assert not (token_path / "recogdrive_ego_feature_v1.gz").exists()


def test_cache_only_fails_clearly_when_slim_sidecar_is_missing(tmp_path):
    token_path = tmp_path / "log" / "token"
    token_path.mkdir(parents=True)
    dump_feature_target_to_pickle(
        token_path / "trajectory_target.gz",
        {"trajectory": torch.zeros(8, 3)},
    )
    dump_feature_target_to_pickle(
        token_path / "qwen3_feature.gz",
        {"first_hidden_state": torch.zeros(2, 3), "last_hidden_state": torch.zeros(2, 3)},
    )

    with pytest.raises(RuntimeError, match="recogdrive_ego_feature_v1.gz=1"):
        CacheOnlyDataset(
            cache_path=str(tmp_path),
            feature_builders=[ReCogDriveEgoFeatureBuilder()],
            target_builders=[_TestTargetBuilder()],
        )


def test_cache_backfill_does_not_recompute_existing_target(tmp_path):
    target_builder = _TestTargetBuilder()
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(log_name="log", initial_token="token"),
        get_agent_input=lambda: _make_fake_agent_input(),
    )
    scene_loader = SimpleNamespace(get_scene_from_token=lambda token: scene)
    dataset = Dataset.__new__(Dataset)
    dataset._scene_loader = scene_loader
    dataset._cache_path = tmp_path
    dataset._feature_builders = []
    dataset._target_builders = [target_builder]
    dataset._valid_cache_paths = {}

    token_path = tmp_path / "log" / "token"
    token_path.mkdir(parents=True)
    dump_feature_target_to_pickle(
        token_path / "trajectory_target.gz",
        {"trajectory": torch.ones(8, 3)},
    )

    dataset._cache_scene_with_token("token")

    assert target_builder.compute_calls == 0
