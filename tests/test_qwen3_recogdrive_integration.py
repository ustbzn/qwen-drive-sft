import pickle
from pathlib import Path

import torch
from torch import nn

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.agents.recogdrive import recogdrive_features
from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_backbone import load_qwen3_processor, system_message
from navsim.agents.recogdrive.hdmap_vector_encoder import MapEncoder
from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder, select_hidden_state


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
        polygon_type=torch.randint(0, 3, (B, M)),
        polygon_on_route=torch.randint(0, 2, (B, M)),
        polygon_tl_status=torch.randint(0, 4, (B, M)),
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
