import pickle
from pathlib import Path

from navsim.common.dataclasses import Scene, SensorConfig
from navsim.agents.recogdrive import recogdrive_features
from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
from navsim.agents.recogdrive.recogdrive_backbone import load_qwen3_processor, system_message
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder


def test_make_recogdrive_config_accepts_explicit_vl_hidden_dim():
    config = make_recogdrive_config(
        "small",
        action_dim=3,
        action_horizon=8,
        input_embedding_dim=384,
        vl_hidden_dim=2048,
    )

    assert config.vl_hidden_dim == 2048


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
    processor = load_qwen3_processor("/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct")

    assert type(processor).__name__ == "Qwen3VLProcessor"
    assert type(processor.tokenizer).__name__ == "Qwen2Tokenizer"


def test_qwen3_processor_accepts_mogo_image_message():
    processor = load_qwen3_processor("/pcpt/project/lijun/.code/Qwen3-VL-2B-Instruct")
    image = (
        "/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/sensor_blobs/trainval/"
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
        "/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/navsim_logs/trainval/"
        "JLBJA20073D_20260313152238.pkl"
    )
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)

    scene = Scene.from_scene_dict_list(
        frames[:12],
        sensor_blobs_path=Path("/NXT/lst/vla/mogo_data/2026-03-13-15-22-20/dataset/sensor_blobs/trainval"),
        num_history_frames=4,
        num_future_frames=8,
        sensor_config=SensorConfig.build_all_sensors(include=[0, 1, 2, 3]),
        load_image_path=True,
    )

    assert scene.scene_metadata.map_name == "las_vegas"
    assert scene.map_api is None
    assert scene.frames[0].lidar.lidar_pc is None
