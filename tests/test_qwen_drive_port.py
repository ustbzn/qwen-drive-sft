"""Qwen-Drive 适配契约：论文损失、官方前向、在线 VLM 和训练 checkpoint。"""

from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from PIL import Image
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer, Qwen3_5Config

from navsim.agents.qwen_drive_planner import modeling
from navsim.agents.qwen_drive_planner.qwen_drive_agent import QwenDriveAgent
from navsim.agents.qwen_drive_planner.qwen_drive_config import (
    DEFAULT_VLM_PATH, FULL_VIEW_LIST, QwenDrivePlannerConfig,
)
from navsim.agents.qwen_drive_planner.qwen_drive_features import (
    build_qwen_drive_ego_features, collect_qwen_drive_view_frames,
    format_navigation_instruction, nav_command_from_driving_command,
    QwenDriveTrajectoryTargetBuilder,
)
from navsim.agents.qwen_drive_planner.qwen_drive_flow import (
    endpoint_loss, interpolate_noisy_trajectory, sample_flow_time, temporal_smoothness_loss,
)
from navsim.agents.qwen_drive_planner.qwen_drive_lightning import QwenDriveLightningModule
from navsim.agents.qwen_drive_planner.qwen_drive_official import (
    make_multi_view_processor, predict_endpoint_masked, sample_masked,
)
from navsim.planning.script.run_training_qwen_drive_il import custom_collate_fn


@pytest.fixture(scope="module")
def official():
    torch.set_num_threads(2)
    return modeling


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(DEFAULT_VLM_PATH, local_files_only=True)


@pytest.fixture
def tiny_model(official, tokenizer, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for name in ("causal_conv1d_fn", "chunk_gated_delta_rule", "FusedRMSNormGated"):
        monkeypatch.setattr(modeling_qwen3_5, name, None)
    expert_config = official.PlanningExpertConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layers_per_kv=4, time_embed_dim=8, fourier_num_features=2,
        partial_rotary_factor=0.5, mrope_section=(1, 1, 0),
    )
    vlm_config = Qwen3_5Config(
        text_config={
            "vocab_size": 248320, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4,
            "num_key_value_heads": 1, "head_dim": 8,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 1, "linear_num_value_heads": 2,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 4, "out_hidden_size": 32, "num_position_embeddings": 16,
        },
    )
    config = official.QwenDriveConfig(
        vlm_config=vlm_config, expert_config=expert_config,
        history_image_pixels=4096, current_image_pixels=4096,
    )
    torch.manual_seed(12)
    model = official.QwenDriveForPlanning(config)
    model.processor = official.QwenDriveProcessor(tokenizer, config)
    return model


def make_features(offset=0.0):
    history = torch.zeros(16, 3)
    history[:, 0] = torch.linspace(-15.0 - offset, 0.0, 16)
    return {
        "history": history,
        "history_velocity": torch.ones(16, 2),
        "history_acceleration": torch.zeros(16, 2),
        "ego_status": torch.tensor([1., 0., 0., 0., 0., 1., 0., 0.]),
        "nav_command": torch.tensor(0),
        "navigation_instruction": None,
        "view_frame_paths": [
            [np.full((64, 64, 3), 30 + index, dtype=np.uint8) for index in range(4)]
            for _ in range(3)
        ],
    }


def make_agent(model):
    agent = QwenDriveAgent()
    model.vlm.requires_grad_(False)
    agent._model = model
    agent._processor = make_multi_view_processor(model.processor)
    agent._initialized = True
    agent.train()
    return agent


def test_velocity_objective_matches_paper():
    clean = torch.randn(2, 5, 3)
    noise = torch.randn_like(clean)
    predicted = torch.randn_like(clean, requires_grad=True)
    times = torch.tensor([0.2, 0.9])
    noisy = interpolate_noisy_trajectory(clean, noise, times)
    expected = functional.mse_loss(
        (predicted - noisy) / (1 - times[:, None, None]), clean - noise,
    )
    torch.testing.assert_close(endpoint_loss(predicted, clean, times), expected)
    loss = endpoint_loss(torch.ones(2, 1, 3), torch.zeros(2, 1, 3), torch.tensor([0., 0.9]))
    torch.testing.assert_close(loss, torch.tensor(50.5))


def test_beta_is_clipped_not_rescaled():
    generator = torch.Generator().manual_seed(31)
    times = sample_flow_time(40000, torch.device("cpu"), generator=generator)
    assert times.max() == torch.tensor(0.9)
    assert 0.13 < (times == 0.9).float().mean() < 0.16
    assert 0.58 < times.mean() < 0.60


def test_temporal_huber_coefficients_and_mask():
    predicted = torch.tensor([[[0., 0., 0.], [2., 2., 2.], [0., 0., 0.]]])
    clean = torch.zeros_like(predicted)
    expected = 2e-4 * 1.5 + 2e-5 * 3.5
    torch.testing.assert_close(temporal_smoothness_loss(predicted, clean), torch.tensor(expected))
    mask = torch.tensor([[True, True, False]])
    torch.testing.assert_close(
        temporal_smoothness_loss(predicted, clean, valid_mask=mask),
        torch.tensor(2e-4 * 1.5),
    )


@pytest.mark.parametrize("code, expected", [
    ([1, 0, 0, 0], 1), ([0, 1, 0, 0], 0), ([0, 0, 1, 0], 2),
    ([1, 1, 0, 0], -1), ([0, 0, 0, 1], -1),
])
def test_navigation_mapping(code, expected):
    assert nav_command_from_driving_command(code) == expected


def test_full_navigation_instruction_formats_actions_distance_and_station_side():
    text = format_navigation_instruction([
        ["turn left", 46.0, 2, 3],
        ["park", 8.5, 1, 1],
    ])

    assert "entry 1: action=turn left" in text
    assert "distance=46.0 m" in text
    assert "lane_range=[2, 3]" in text
    assert "entry 2: action=park at the station" in text
    assert "distance=8.5 m" in text
    assert "station_side=left" in text


def test_qwen_feature_builder_emits_structured_navigation_text():
    statuses = [SimpleNamespace(
        ego_pose=[0.0, 0.0, 0.0], ego_velocity=[1.0, 0.0],
        ego_acceleration=[0.0, 0.0], driving_command=[0, 1, 0, 0],
        navigation_command=None,
    ) for _ in range(16)]
    statuses[-1].navigation_command = [["turn right", 31.0, -1, -2]]

    features = build_qwen_drive_ego_features(SimpleNamespace(ego_statuses=statuses))

    assert "action=turn right" in features["navigation_instruction"]
    assert "distance=31.0 m" in features["navigation_instruction"]


def test_navigation_text_is_collated_and_added_to_qwen_prompt(tiny_model):
    navigation_text = format_navigation_instruction([
        ["turn left", 28.0, 1, 2],
    ])
    sample = make_features()
    sample["navigation_instruction"] = navigation_text
    target = {"trajectory": torch.zeros(50, 3)}
    features, _, _ = custom_collate_fn([(sample, target)])

    agent = make_agent(tiny_model)
    scene = agent._build_scene(
        features["view_frame_paths"][0], features["history"][0],
        features["history_velocity"][0], features["history_acceleration"][0],
        features["ego_status"][0], int(features["nav_command"][0]),
        features["navigation_instruction"][0],
    )

    prompt = scene.instruction()
    assert "Active navigation command: [GO STRAIGHT]" in prompt
    assert "Detailed navigation instructions" in prompt
    assert "action=turn left" in prompt
    assert "distance=28.0 m" in prompt


def test_eight_views_and_sparse_camera_loading():
    assert len(FULL_VIEW_LIST) == 8
    config = QwenDrivePlannerConfig(view_list=list(FULL_VIEW_LIST))
    sensors = QwenDriveAgent(config).get_sensor_config()
    for view in FULL_VIEW_LIST:
        assert getattr(sensors, view) == [0, 5, 10, 15]


def test_camera_order_and_missing_view():
    cameras = [SimpleNamespace(cam_f0=SimpleNamespace(image=f"front-{index}.jpg")) for index in range(16)]
    data = SimpleNamespace(cameras=cameras)
    assert collect_qwen_drive_view_frames(data, ["cam_f0"], 4) == [
        ["front-0.jpg", "front-5.jpg", "front-10.jpg", "front-15.jpg"]
    ]
    with pytest.raises(ValueError, match="cam_l2"):
        collect_qwen_drive_view_frames(data, ["cam_f0", "cam_l2"], 4)


def test_rejects_2hz_history_and_target_timestamps():
    with pytest.raises(ValueError, match="10Hz"):
        build_qwen_drive_ego_features(SimpleNamespace(ego_statuses=[None] * 4))
    scene = SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=16, num_future_frames=50),
        frames=[SimpleNamespace(timestamp=index * 500000) for index in range(66)],
    )
    with pytest.raises(ValueError, match="10Hz"):
        QwenDriveTrajectoryTargetBuilder(QwenDrivePlannerConfig()).compute_targets(scene)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_attention_matches_unpadded_official(tiny_model, official, dtype):
    expert = tiny_model.planning_expert.to(dtype)
    batch, length = 2, 5
    cache = [(torch.randn(batch, length, 1, 8).to(dtype), torch.randn(batch, length, 1, 8).to(dtype))]
    valid = torch.tensor([[True, True, True, False, False], [True] * length])
    anchor = torch.tensor([[500, 700]] * 3)
    noisy = torch.randn(batch, 50, 3)
    times = torch.tensor([0.2, 0.9])
    navigation = torch.tensor([0, 1])
    ego = torch.randn(batch, 8)
    history = torch.randn(batch, 15, 3)
    velocity, acceleration = torch.randn(batch, 16, 2), torch.randn(batch, 16, 2)
    queries = expert.encode_history(history, navigation, velocity, acceleration)
    masked = predict_endpoint_masked(
        expert, noisy, times, queries, cache, anchor, navigation, ego,
        scene_valid=valid, gradient_checkpointing=True,
    )
    expected = []
    for index, count in enumerate([3, 5]):
        expected.append(expert.predict_endpoint(
            noisy[index:index+1], times[index:index+1],
            tuple(query[index:index+1] for query in queries),
            [(key[index:index+1, :count], value[index:index+1, :count]) for key, value in cache],
            anchor[:, index:index+1], navigation[index:index+1], ego[index:index+1],
        ))
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(masked, torch.cat(expected), atol=tolerance, rtol=tolerance)
    masked.square().mean().backward()
    assert expert.layers[0].qkv_proj.weight.grad.abs().sum() > 0


def test_prompt_and_pixels_equal_official(tiny_model, official):
    agent = make_agent(tiny_model)
    features = make_features()
    scene = agent._build_scene(
        features["view_frame_paths"], features["history"], features["history_velocity"],
        features["history_acceleration"], features["ego_status"], 0,
    )
    standard_scene = official.DrivingScene(**{
        key: value for key, value in vars(scene).items()
        if key not in {"view_order", "navigation_instruction"}
    })
    base = tiny_model.processor
    for reasoning in (False, True):
        reference = base(standard_scene, with_reasoning=reasoning)
        adapted = agent._processor(scene, with_reasoning=reasoning)
        for key in reference:
            torch.testing.assert_close(adapted[key], reference[key], rtol=0, atol=0)
    scene.nav_command = -1
    assert "Active navigation command: [NONE]" in scene.instruction()


def test_online_vlm_backward_optimizer_and_checkpoint(tiny_model, official, tmp_path):
    agent = make_agent(tiny_model)
    target = {"trajectory": torch.zeros(50, 3)}
    features, targets, tokens = custom_collate_fn([
        (make_features(), target), (make_features(100.125), target),
    ])
    observed_lengths = []
    hook = tiny_model.vlm.register_forward_pre_hook(
        lambda module, args, kwargs: observed_lengths.append(kwargs["input_ids"].shape[1]),
        with_kwargs=True,
    )
    optimizer = agent.get_optimizers()
    before = tiny_model.planning_expert.out_proj.weight.detach().clone()
    prediction = agent(features, targets, tokens)
    hook.remove()
    assert len(set(observed_lengths)) == 2
    assert torch.isfinite(prediction["loss"])
    prediction["loss"].backward()
    assert all(parameter.grad is None for parameter in tiny_model.vlm.parameters())
    assert tiny_model.planning_expert.layers[0].qkv_proj.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, tiny_model.planning_expert.out_proj.weight)
    assert not tiny_model.vlm.training

    wrapper = QwenDriveLightningModule(agent)
    checkpoint = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(checkpoint)
    assert checkpoint["state_dict"]
    assert not any("vlm." in key for key in checkpoint["state_dict"])
    checkpoint_path = tmp_path / "planner.ckpt"
    torch.save(checkpoint, checkpoint_path)
    wrapper.on_load_checkpoint(checkpoint)
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)


def test_qwen_lightning_transfer_preserves_frozen_lazy_image_frames():
    @dataclass(frozen=True)
    class FrozenFrame:
        value: int

        def __call__(self):
            return self.value

    frame = FrozenFrame(7)
    features = {
        "history": torch.zeros(1, 16, 3),
        "view_frame_paths": [[frame]],
    }
    targets = {"trajectory": torch.zeros(1, 50, 3)}
    tokens = (None,)
    transferred = QwenDriveLightningModule(SimpleNamespace()).transfer_batch_to_device(
        (features, targets, tokens), torch.device("cpu"), 0,
    )

    moved_features, moved_targets, moved_tokens = transferred
    assert moved_features["history"].device.type == "cpu"
    assert moved_targets["trajectory"].device.type == "cpu"
    assert moved_features["view_frame_paths"][0][0] is frame
    assert moved_tokens is tokens


def test_mixed_source_views_use_per_sample_prompt_labels(monkeypatch):
    agent = QwenDriveAgent(QwenDrivePlannerConfig(view_list=["cam_f0", "cam_l1", "cam_r1"]))
    mogo = make_features()
    public = make_features()
    public["view_list"] = ("cam_f0", "cam_l0", "cam_r0")
    features, _, _ = custom_collate_fn([
        (mogo, {"trajectory": torch.zeros(50, 3)}),
        (public, {"trajectory": torch.zeros(50, 3)}),
    ])

    scenes = []
    agent._processor = lambda scene, **kwargs: scenes.append(scene) or {}
    model = SimpleNamespace(
        parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
        _prefill=lambda _: (
            [(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))],
            torch.zeros(3, 1),
        ),
    )
    monkeypatch.setattr(agent, "_require_model", lambda: model)
    _, anchor, valid = agent._encode_scenes(features)

    assert anchor.shape == (3, 2)
    assert valid.shape == (2, 1)
    assert scenes[0].view_order == ("<FRONT VIEW>", "<BACK LEFT VIEW>", "<BACK RIGHT VIEW>")
    assert scenes[1].view_order == ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")


def test_hydra_defaults_match_agent_config():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "navsim/planning/script/config/training"), version_base=None):
        config = compose(config_name="qwen_drive_il")
    defaults = asdict(QwenDrivePlannerConfig())
    assert {key: config.agent[key] for key in defaults if key != "view_list"} == {
        key: value for key, value in defaults.items() if key != "view_list"
    }
    assert config.agent.view_list == ["cam_f0", "cam_l1", "cam_r1"]
    assert config.physicalai_view_list == defaults["view_list"]
    assert config.train_test_split.scene_filter.num_history_frames == 16
    assert config.train_test_split.scene_filter.num_future_frames == 50
    assert config.trainer.params.precision == "bf16-mixed"
    assert config.cache_path is None

def test_sampler_matches_official(tiny_model):
    expert = tiny_model.planning_expert.eval()
    inputs = dict(
        scene_cache=[(torch.randn(1, 7, 1, 8), torch.randn(1, 7, 1, 8))],
        position_anchor=torch.full((3, 1), 500),
        history=torch.randn(1, 15, 3),
        history_velocity=torch.randn(1, 16, 2),
        history_acceleration=torch.randn(1, 16, 2),
        nav_command=torch.tensor([1]),
        ego_status=torch.randn(1, 8),
        noise=torch.randn(1, 50, 3),
        num_steps=10,
        min_one_minus_t=0.1,
    )
    expected = expert.sample(**inputs)
    actual = sample_masked(expert, **inputs, scene_valid=torch.ones(1, 7, dtype=torch.bool))
    torch.testing.assert_close(actual, expected)


def test_lightning_fit_save_resume(tiny_model, official, tmp_path):
    import pytorch_lightning as lightning
    from torch.utils.data import DataLoader

    agent = make_agent(tiny_model)
    wrapper = QwenDriveLightningModule(agent)
    data = DataLoader(
        [(make_features(), {"trajectory": torch.zeros(50, 3)})],
        batch_size=1, collate_fn=custom_collate_fn,
    )
    settings = dict(
        accelerator="cpu", devices=1, precision="32-true", logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, limit_train_batches=1, limit_val_batches=1,
        default_root_dir=str(tmp_path),
    )
    trainer = lightning.Trainer(max_epochs=1, **settings)
    trainer.fit(wrapper, train_dataloaders=data, val_dataloaders=data)
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["val/loss_epoch"])
    checkpoint_path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert not any("vlm." in key for key in checkpoint["state_dict"])
    resumed = lightning.Trainer(max_epochs=2, **settings)
    resumed.fit(wrapper, train_dataloaders=data, val_dataloaders=data, ckpt_path=str(checkpoint_path))
    assert resumed.global_step == 2


def test_local_model_save_and_reload(tiny_model, tmp_path):
    from safetensors.torch import save_file

    model_path = tmp_path / "model"
    planner_path = tmp_path / "planner"
    planner_path.mkdir()
    tiny_model.save_pretrained(model_path)
    save_file(tiny_model.planning_expert.state_dict(), str(planner_path / "model.safetensors"))
    restored = modeling.QwenDriveForPlanning.from_pretrained(
        model_path, planner=planner_path, local_files_only=True,
    )
    original_state = tiny_model.state_dict()
    restored_state = restored.state_dict()
    assert original_state.keys() == restored_state.keys()
    for key in original_state:
        torch.testing.assert_close(original_state[key], restored_state[key], rtol=0, atol=0)
    assert restored.__class__.__module__.startswith("navsim.agents.qwen_drive_planner.modeling.")


def test_imports_work_when_external_package_is_unavailable():
    root = Path(__file__).resolve().parents[1]
    program = textwrap.dedent("""
        import importlib.abc
        import inspect
        import sys
        from pathlib import Path

        class BlockExternalQwenDrive(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "qwen_drive" or fullname.startswith("qwen_drive."):
                    raise ModuleNotFoundError("External qwen_drive deliberately unavailable")
                return None

        sys.meta_path.insert(0, BlockExternalQwenDrive())
        from navsim.agents.qwen_drive_planner import modeling
        from navsim.agents.qwen_drive_planner.qwen_drive_agent import QwenDriveAgent
        from navsim.planning.script.run_training_qwen_drive_il import custom_collate_fn
        from transformers import AutoConfig

        root = Path.cwd().resolve()
        for symbol in (modeling.QwenDriveForPlanning, modeling.PlanningExpert,
                       modeling.QwenDriveProcessor, modeling.QwenDriveConfig):
            assert Path(inspect.getfile(symbol)).resolve().is_relative_to(root)
        assert isinstance(AutoConfig.for_model("qwen_drive"), modeling.QwenDriveConfig)
        assert not hasattr(QwenDriveAgent().config, "official_repo_path")
        assert not any(name == "qwen_drive" or name.startswith("qwen_drive.") for name in sys.modules)
    """)
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=root, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_eight_view_processor_encodes_all_frames(tiny_model):
    config = QwenDrivePlannerConfig(view_list=list(FULL_VIEW_LIST))
    agent = make_agent(tiny_model)
    agent._config = config
    features = make_features()
    images = [
        [np.full((64, 64, 3), view_index * 8 + frame_index, dtype=np.uint8) for frame_index in range(4)]
        for view_index in range(8)
    ]
    scene = agent._build_scene(
        images, features["history"], features["history_velocity"],
        features["history_acceleration"], features["ego_status"], 0,
    )
    patches, grids, counts = agent._processor.encode_images(scene)
    inputs = agent._processor(scene)
    assert grids.shape == (32, 3)
    assert len(counts) == 32
    torch.testing.assert_close(inputs["pixel_values"], patches)
    assert (inputs["input_ids"] == tiny_model.config.vlm_config.image_token_id).sum() == sum(counts)
    prompt = agent._processor.tokenizer.decode(inputs["input_ids"][0])
    positions = [prompt.index(label) for label in config.view_labels]
    assert positions == sorted(positions)
    assert len(set(positions)) == 8


def test_navsim_inference_with_local_model(tiny_model):
    agent = make_agent(tiny_model)
    features = make_features()
    statuses = [
        SimpleNamespace(
            ego_pose=features["history"][index].numpy(),
            ego_velocity=features["history_velocity"][index].numpy(),
            ego_acceleration=features["history_acceleration"][index].numpy(),
            driving_command=np.array([0, 1, 0, 0], dtype=np.float32),
        )
        for index in range(16)
    ]
    cameras = [
        SimpleNamespace(**{
            view: SimpleNamespace(image=np.full((64, 64, 3), index, dtype=np.uint8))
            for view in agent.config.view_list
        })
        for index in range(16)
    ]
    trajectory = agent.compute_trajectory(SimpleNamespace(ego_statuses=statuses, cameras=cameras))
    assert trajectory.poses.shape == (8, 3)
    assert np.isfinite(trajectory.poses).all()
    assert trajectory.trajectory_sampling.interval_length == 0.5
