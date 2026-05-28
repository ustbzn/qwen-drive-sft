#!/usr/bin/env python3
"""
单 token 轨迹调试：对比 IL/RL checkpoint、随机/确定性采样、去噪链与 PDM 指标。

示例:
  python -m navsim.planning.script.debug_token_trajectory \\
    --token 1bdfacbfcff75c27 \\
    --checkpoint_rl /path/to/RL.ckpt \\
    --checkpoint_il /path/to/IL.ckpt \\
    --vlm_path /path/to/ReCogDrive-VLM-2B \\
    --metric_cache_path /path/to/metric_cache2 \\
    --out vis_output/debug/1bdfacbfcff75c27
"""
from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import pickle
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import initialize_config_dir
from hydra.utils import instantiate
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from transformers.feature_extraction_utils import BatchFeature

_REPO_ROOT = Path(__file__).resolve().parents[3]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.agents.recogdrive.recogdrive_features import EGO_STATUS_FEATURE_NAMES
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import Camera, Scene, SceneFilter, SensorConfig, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.script.run_pdm_score_recogdrive import compute_trajectory_l2_loss
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.training.dataset import load_feature_target_from_pickle
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_trajectory_to_camera_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_bev_ax


NAV_COMMANDS = ["turn left", "go straight", "turn right"]
# Diffusion planner 预测的未来路点数（与 action_horizon 一致）
MODEL_NUM_POSES = 8


def build_scene_filter_simple(token: str, scene_filter_name: str = "navtest") -> SceneFilter:
    config_dir = (
        Path(__file__).resolve().parent / "config" / "common" / "train_test_split" / "scene_filter"
    ).as_posix()
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=scene_filter_name)
    scene_filter: SceneFilter = instantiate(cfg)
    scene_filter.tokens = [token]
    scene_filter.log_names = None
    return scene_filter


def build_pdm_tools() -> Tuple[PDMSimulator, PDMScorer]:
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(
        proposal_sampling=proposal_sampling,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
            driving_direction_weight=0.0,
        ),
    )
    return simulator, scorer


def build_agent(args: argparse.Namespace, checkpoint_path: str) -> ReCogDriveAgent:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trajectory_sampling = TrajectorySampling(
        time_horizon=float(args.time_horizon),
        interval_length=float(args.interval_length),
    )
    agent = ReCogDriveAgent(
        trajectory_sampling=trajectory_sampling,
        checkpoint_path=checkpoint_path,
        vlm_path=str(args.vlm_path),
        cam_type=str(args.cam_type),
        vlm_type=str(args.vlm_type),
        dit_type=str(args.dit_type),
        sampling_method=str(args.sampling_method),
        cache_mode=False,
        cache_hidden_state=bool(args.cache_hidden_state),
        vlm_size=str(args.vlm_size),
        grpo=False,
    )
    os.environ["LOCAL_RANK"] = "0"
    agent.device = device
    if agent.backbone is not None:
        agent.backbone = agent.backbone.to(device)
    agent.action_head = agent.action_head.to(device)
    agent.initialize()
    agent.eval()
    return agent


def load_action_head_weights(agent: ReCogDriveAgent, checkpoint_path: str) -> int:
    """仅重载 Diffusion Planner 权重，便于在共享 VLM 特征下对比 IL/RL。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    model_dict = agent.action_head.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if k.startswith("agent.action_head."):
            k2 = k[len("agent.action_head.") :]
        elif k.startswith("action_head."):
            k2 = k[len("action_head.") :]
        else:
            continue
        if k2 in model_dict and v.shape == model_dict[k2].shape:
            filtered[k2] = v
    agent.action_head.load_state_dict(filtered, strict=False)
    return len(filtered)


def find_feature_cache_path(cache_root: Path, token: str) -> Optional[Path]:
    feature_file = "internvl_feature.gz"
    if not cache_root.is_dir():
        return None
    for log_path in cache_root.iterdir():
        if not log_path.is_dir():
            continue
        candidate = log_path / token
        if (candidate / feature_file).is_file():
            return candidate
    return None


def load_features_from_cache(cache_root: Path, token: str) -> Dict[str, torch.Tensor]:
    token_path = find_feature_cache_path(cache_root, token)
    if token_path is None:
        raise FileNotFoundError(
            f"未在 {cache_root} 下找到 token={token} 的 internvl_feature.gz；"
            "请去掉 --feature_cache_path 或先跑特征缓存。"
        )
    return load_feature_target_from_pickle(token_path / "internvl_feature.gz")


def compute_features_from_scene(agent: ReCogDriveAgent, agent_input) -> Dict[str, torch.Tensor]:
    features: Dict[str, torch.Tensor] = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    return features


def prepare_planner_tensors(
    agent: ReCogDriveAgent, features: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """与 ReCogDriveAgent.forward 推理分支一致，返回 planner 所需张量。"""
    model_dtype = next(agent.action_head.parameters()).dtype
    dev = agent.device

    for key, tensor in list(features.items()):
        if isinstance(tensor, torch.Tensor):
            features[key] = tensor.to(dev)

    history_trajectory = features["history_trajectory"]
    high_command_one_hot = features["high_command_one_hot"]
    if history_trajectory.ndim == 2:
        history_trajectory = history_trajectory.unsqueeze(0)
    if high_command_one_hot.ndim == 1:
        high_command_one_hot = high_command_one_hot.unsqueeze(0)

    if agent.cache_hidden_state:
        first_hidden_state = features["first_hidden_state"]
        last_hidden_state = features["last_hidden_state"]
    else:
        if agent.backbone is None:
            raise RuntimeError("cache_hidden_state=False 但 backbone 未初始化")
        image_path_tensor = features["image_path_tensor"]
        if image_path_tensor.ndim == 1:
            image_path_tensor = image_path_tensor.unsqueeze(0)
        image_paths = agent._decode_paths_from_tensor(image_path_tensor)
        from navsim.agents.recogdrive.utils.internvl_preprocess import load_image

        pixel_values_list = [load_image(path) for path in image_paths]
        num_patches_list = [p.shape[0] for p in pixel_values_list]
        pixel_values_cat = torch.cat(pixel_values_list, dim=0).to(dev)

        command_indices = torch.argmax(high_command_one_hot, dim=-1)
        command_str_list = [NAV_COMMANDS[idx.item()] for idx in command_indices]
        questions = []
        batch_size = high_command_one_hot.shape[0]
        for i in range(batch_size):
            hist = history_trajectory[i]
            cmd = command_str_list[i]
            from navsim.agents.recogdrive.utils.utils import format_number

            history_str = " ".join(
                [
                    f"   - t-{3-j}: ({format_number(hist[j, 0].item())}, "
                    f"{format_number(hist[j, 1].item())}, {format_number(hist[j, 2].item())})"
                    for j in range(hist.shape[0])
                ]
            )
            prompt = (
                "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                "1. Visual perception from front camera view\n"
                f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                f"3. Active navigation command: [{cmd.upper()}]"
            )
            output_requirements = (
                "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                "- Each point format: (x:float, y:float, heading:float)\n"
                "- Use [PT, ...] to encapsulate the trajectory\n"
                "- Maintain numerical precision to 2 decimal places"
            )
            questions.append(f"{prompt}{output_requirements}")

        outputs = agent.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
        first_hidden_state = outputs.hidden_states[0]
        last_hidden_state = outputs.hidden_states[-1]

    status_feature = features["status_feature"]
    if status_feature.ndim == 1:
        status_feature = status_feature.unsqueeze(0)
    if first_hidden_state.ndim == 2:
        first_hidden_state = first_hidden_state.unsqueeze(0)
    if last_hidden_state.ndim == 2:
        last_hidden_state = last_hidden_state.unsqueeze(0)

    first_hidden_state = first_hidden_state.to(model_dtype)
    last_hidden_state = last_hidden_state.to(model_dtype)
    status_feature = status_feature.to(model_dtype)
    return first_hidden_state, last_hidden_state, status_feature


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_trajectory(
    agent: ReCogDriveAgent,
    first_hs: torch.Tensor,
    last_hs: torch.Tensor,
    status_feature: torch.Tensor,
    *,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> Trajectory:
    if seed is not None:
        set_seed(seed)
    action_inputs = BatchFeature({"status_feature": status_feature})
    with torch.no_grad():
        out = agent.action_head.get_action(
            first_hs, last_hs, action_inputs, deterministic=deterministic
        )
    poses = out["pred_traj"].float().cpu().squeeze(0)
    return Trajectory(poses)


def sample_denoising_chain(
    agent: ReCogDriveAgent,
    first_hs: torch.Tensor,
    last_hs: torch.Tensor,
    status_feature: torch.Tensor,
    *,
    deterministic: bool = False,
    seed: Optional[int] = None,
) -> Tuple[List[Trajectory], Trajectory]:
    if seed is not None:
        set_seed(seed)
    with torch.no_grad():
        chain_norm, final = agent.action_head.sample_chain(
            first_hs, last_hs, status_feature, deterministic=deterministic
        )
    steps: List[Trajectory] = []
    for i in range(chain_norm.shape[1]):
        denormed = agent.action_head.denorm_odo(chain_norm[:, i])
        steps.append(Trajectory(denormed.float().cpu().squeeze(0)))
    final_traj = Trajectory(final.float().cpu().squeeze(0))
    return steps, final_traj


def evaluate_trajectory(
    trajectory: Trajectory,
    metric_cache: MetricCache,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    gt_trajectory: Optional[Trajectory] = None,
) -> Dict[str, Any]:
    pdm_result = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    row = asdict(pdm_result)
    row["l2_xy"] = compute_trajectory_l2_loss(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
    )
    if gt_trajectory is not None:
        row["l2_xy_openloop"] = compute_openloop_l2_xy(trajectory, gt_trajectory)
    return row


def compute_openloop_l2_xy(pred: Trajectory, gt: Trajectory) -> float:
    """在 ego 坐标系下比较 pred/GT，按较短长度对齐（模型为 8 点，scene 默认 10 点）。"""
    n = min(len(pred.poses), len(gt.poses))
    if n == 0:
        return float("nan")
    pred_xy = pred.poses[:n, :2]
    gt_xy = gt.poses[:n, :2]
    return float(np.linalg.norm(pred_xy - gt_xy, axis=1).mean())


def decode_status(status_feature: torch.Tensor) -> Dict[str, float]:
    vals = status_feature.detach().float().cpu().reshape(-1).tolist()
    return {name: float(v) for name, v in zip(EGO_STATUS_FEATURE_NAMES, vals)}


def decode_nav_command(status_feature: torch.Tensor) -> str:
    cmd = status_feature.detach().float().cpu()[:4]
    idx = int(torch.argmax(cmd).item())
    return NAV_COMMANDS[idx] if idx < len(NAV_COMMANDS) else "unknown"


def camera_with_loaded_image(camera: Camera) -> Camera:
    """load_image_path=True 时 camera.image 为路径；绘图/投影需 numpy 数组。"""
    if isinstance(camera.image, np.ndarray):
        return camera
    from PIL import Image

    return replace(camera, image=np.array(Image.open(str(camera.image))))


def _traj_config(color: str, alpha: float = 1.0, width: float = 2.0, style: str = "-") -> Dict[str, Any]:
    cfg = dict(TRAJECTORY_CONFIG["agent"])
    cfg.update(
        {
            "line_color": color,
            "fill_color": color,
            "line_color_alpha": alpha,
            "fill_color_alpha": alpha,
            "line_width": width,
            "line_style": style,
            "arrow_color": color,
            "arrow_edge_color": color,
        }
    )
    return cfg


def _camera_traj_config(color: str, alpha: float = 0.85, width: float = 4.0) -> Dict[str, Any]:
    """add_trajectory_to_camera_ax 所需字段（与 plots.py 一致）。"""
    return {
        "line_color": color,
        "line_color_alpha": alpha,
        "line_width": width,
        "line_style": "-",
        "marker": None,
        "marker_size": 0,
        "marker_edge_color": color,
        "zorder": 3,
        "arrow_color": color,
        "arrow_edge_color": color,
        "arrow_alpha": alpha,
        "arrow_line_width": 1.5,
    }


def plot_bev_camera_panel(
    scene: Scene,
    frame_idx: int,
    trajectories: List[Tuple[str, Trajectory, str]],
    title: str,
    save_path: Path,
) -> None:
    frame = scene.frames[frame_idx]
    cam_f0 = camera_with_loaded_image(frame.cameras.cam_f0)
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.05)
    camera_ax = fig.add_subplot(gs[0])
    bev_ax = fig.add_subplot(gs[1])

    camera_ax.imshow(cam_f0.image)
    camera_ax.axis("off")
    add_configured_bev_on_ax(bev_ax, scene.map_api, frame)
    configure_bev_ax(bev_ax)
    bev_ax.set_aspect("equal")
    bev_ax.set_xticks([])
    bev_ax.set_yticks([])

    for label, traj, color in trajectories:
        add_trajectory_to_bev_ax(bev_ax, traj, _traj_config(color))
        add_trajectory_to_camera_ax(camera_ax, cam_f0, traj, _camera_traj_config(color))

    fig.suptitle(title, fontsize=12)
    legend_lines = [f"{label}" for label, _, color in trajectories]
    fig.text(0.5, 0.02, " | ".join(legend_lines), ha="center", fontsize=10)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_denoising_chain(
    scene: Scene,
    frame_idx: int,
    chain_steps: List[Trajectory],
    final_traj: Trajectory,
    gt_traj: Trajectory,
    save_path: Path,
    label: str,
) -> None:
    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(figsize=(9, 9))
    add_configured_bev_on_ax(ax, scene.map_api, frame)
    configure_bev_ax(ax)
    ax.set_aspect("equal")

    add_trajectory_to_bev_ax(ax, gt_traj, _traj_config("green", width=3.0))
    n = len(chain_steps)
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, max(n, 1)))
    for i, step_traj in enumerate(chain_steps):
        alpha = 0.25 + 0.75 * (i / max(n - 1, 1))
        add_trajectory_to_bev_ax(
            ax,
            step_traj,
            _traj_config(plt.matplotlib.colors.to_hex(cmap[i]), alpha=alpha, width=1.5, style="--"),
        )
    add_trajectory_to_bev_ax(ax, final_traj, _traj_config("red", width=3.0))
    ax.set_title(f"{label} | denoise steps={n} (purple→yellow) | final=red | GT=green")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def run_variant_suite(
    agent: ReCogDriveAgent,
    first_hs: torch.Tensor,
    last_hs: torch.Tensor,
    status_feature: torch.Tensor,
    metric_cache: MetricCache,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    gt_traj: Trajectory,
    label_prefix: str,
    num_stochastic: int,
    base_seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    variants: List[Tuple[str, bool, Optional[int]]] = [
        ("deterministic", True, base_seed),
    ]
    for i in range(num_stochastic):
        variants.append((f"stochastic_{i}", False, base_seed + i))

    for name, deterministic, seed in variants:
        traj = predict_trajectory(
            agent,
            first_hs,
            last_hs,
            status_feature,
            deterministic=deterministic,
            seed=seed,
        )
        metrics = evaluate_trajectory(traj, metric_cache, simulator, scorer, gt_traj)
        rows.append(
            {
                "checkpoint_label": label_prefix,
                "variant": name,
                "deterministic": deterministic,
                "seed": seed if seed is not None else "",
                **metrics,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单 token ReCogDrive 轨迹调试")
    parser.add_argument("--token", required=True, help="场景 token")
    parser.add_argument("--checkpoint_rl", required=True, help="RL（或当前）checkpoint")
    parser.add_argument(
        "--checkpoint_il",
        default="",
        help="可选 IL checkpoint，用于对比",
    )
    parser.add_argument("--vlm_path", required=True)
    parser.add_argument("--out", default="", help="输出目录，默认 vis_output/debug/<token>")
    parser.add_argument("--metric_cache_path", default=os.getenv("NAVSIM_EXP_ROOT", "") + "/metric_cache2")
    parser.add_argument(
        "--feature_cache_path",
        default="",
        help="训练特征缓存根目录（含 internvl_feature.gz）；设置后与 --cache_hidden_state 联用",
    )
    parser.add_argument("--openscene_data_root", default=os.getenv("OPENSCENE_DATA_ROOT", ""))
    parser.add_argument("--split", default="test", choices=["mini", "test", "trainval"])
    parser.add_argument("--scene_filter", default="navtest")
    parser.add_argument("--cam_type", default="single")
    parser.add_argument("--vlm_type", default="internvl")
    parser.add_argument("--dit_type", default="small")
    parser.add_argument("--vlm_size", default="small")
    parser.add_argument("--sampling_method", default="ddim")
    parser.add_argument(
        "--cache_hidden_state",
        action="store_true",
        help="从 feature cache 读 hidden state（需 --feature_cache_path）",
    )
    parser.add_argument("--num_stochastic", type=int, default=5, help="额外随机采样次数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_horizon", type=float, default=4.0)
    parser.add_argument("--interval_length", type=float, default=0.5)
    parser.add_argument("--skip_plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = str(args.token)
    out_dir = Path(args.out) if args.out else Path("vis_output/debug") / token
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.openscene_data_root:
        raise ValueError("请设置 --openscene_data_root 或环境变量 OPENSCENE_DATA_ROOT")
    if not args.metric_cache_path:
        raise ValueError("请设置 --metric_cache_path")

    if args.cache_hidden_state and not args.feature_cache_path:
        raise ValueError("使用 --cache_hidden_state 时必须提供 --feature_cache_path")

    scene_filter = build_scene_filter_simple(token, args.scene_filter)
    # 推理只需 cam_f0 路径；与 plt_vis_from_csv_all 一致
    sensor_config = SensorConfig.build_all_sensors(include=False)
    sensor_config.cam_f0 = True
    data_root = Path(args.openscene_data_root)
    scene_loader = SceneLoader(
        data_root / f"navsim_logs/{args.split}",
        data_root / f"sensor_blobs/{args.split}",
        scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )
    if token not in scene_loader.tokens:
        raise ValueError(f"token {token} 不在 split={args.split} scene_filter={args.scene_filter} 中")

    scene = scene_loader.get_scene_from_token(token)
    agent_input = scene_loader.get_agent_input_from_token(token)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    # 与 planner 的 8 个未来点（4s @ 0.5s）对齐；默认 get_future_trajectory() 为 10 点
    gt_traj = scene.get_future_trajectory(num_trajectory_frames=MODEL_NUM_POSES)

    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))
    if token not in metric_cache_loader.tokens:
        raise ValueError(f"token {token} 不在 metric_cache_path={args.metric_cache_path}")
    with lzma.open(metric_cache_loader.metric_cache_paths[token], "rb") as f:
        metric_cache: MetricCache = pickle.load(f)

    simulator, scorer = build_pdm_tools()

    # 构建 agent 并提取特征（VLM 只跑一次）
    agent = build_agent(args, args.checkpoint_rl)
    if args.cache_hidden_state:
        features = load_features_from_cache(Path(args.feature_cache_path), token)
        print(f"[info] 从特征缓存加载 hidden state: {args.feature_cache_path}")
    else:
        features = compute_features_from_scene(agent, agent_input)
        print("[info] 在线 VLM 提取特征（与 eval 脚本一致）")

    first_hs, last_hs, status_feature = prepare_planner_tensors(agent, features)
    status_dict = decode_status(status_feature.squeeze(0))
    status_dict["nav_command"] = decode_nav_command(status_feature.squeeze(0))
    with (out_dir / "ego_status.json").open("w") as f:
        json.dump(status_dict, f, indent=2)
    print(f"[info] ego status: {json.dumps(status_dict)}")

    all_rows: List[Dict[str, Any]] = []
    plot_trajectories: List[Tuple[str, Trajectory, str]] = [("GT", gt_traj, "green")]

    checkpoints: List[Tuple[str, str]] = [("RL", args.checkpoint_rl)]
    if args.checkpoint_il:
        checkpoints.append(("IL", args.checkpoint_il))

    for ckpt_label, ckpt_path in checkpoints:
        n_loaded = load_action_head_weights(agent, ckpt_path)
        print(f"[info] 已加载 {ckpt_label} action_head 权重 {n_loaded} 个 tensor <- {ckpt_path}")

        rows = run_variant_suite(
            agent,
            first_hs,
            last_hs,
            status_feature,
            metric_cache,
            simulator,
            scorer,
            gt_traj,
            ckpt_label,
            int(args.num_stochastic),
            int(args.seed),
        )
        all_rows.extend(rows)

        # 代表轨迹：确定性采样
        set_seed(int(args.seed))
        rep_traj = predict_trajectory(
            agent, first_hs, last_hs, status_feature, deterministic=True, seed=int(args.seed)
        )
        color = "red" if ckpt_label == "RL" else "orange"
        plot_trajectories.append((f"{ckpt_label} (det)", rep_traj, color))

        if not args.skip_plots:
            chain_steps, final_traj = sample_denoising_chain(
                agent,
                first_hs,
                last_hs,
                status_feature,
                deterministic=False,
                seed=int(args.seed),
            )
            plot_denoising_chain(
                scene,
                frame_idx,
                chain_steps,
                final_traj,
                gt_traj,
                out_dir / f"denoise_chain_{ckpt_label.lower()}.png",
                ckpt_label,
            )

            stoch_trajs = []
            for i in range(int(args.num_stochastic)):
                t = predict_trajectory(
                    agent,
                    first_hs,
                    last_hs,
                    status_feature,
                    deterministic=False,
                    seed=int(args.seed) + i,
                )
                color = plt.matplotlib.colors.to_hex(plt.cm.cool(i / max(int(args.num_stochastic) - 1, 1)))
                stoch_trajs.append((f"s{i}", t, color))
            multi = [("GT", gt_traj, "green")] + [(name, t, color) for name, t, color in stoch_trajs]
            plot_bev_camera_panel(
                scene,
                frame_idx,
                multi,
                f"{ckpt_label}: {args.num_stochastic} stochastic samples",
                out_dir / f"multi_sample_{ckpt_label.lower()}.png",
            )

    # 写 metrics.csv
    csv_path = out_dir / "metrics.csv"
    fieldnames = sorted({k for row in all_rows for k in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[info] 指标表已写入 {csv_path}")

    if not args.skip_plots:
        plot_bev_camera_panel(
            scene,
            frame_idx,
            plot_trajectories,
            f"token={token}",
            out_dir / "compare_checkpoints.png",
        )
        print(f"[info] 对比图已写入 {out_dir / 'compare_checkpoints.png'}")

    # 文本摘要
    summary_lines = [
        f"token: {token}",
        f"nav_command: {status_dict.get('nav_command')}",
        f"metric_cache: {metric_cache_loader.metric_cache_paths[token]}",
        f"checkpoints: RL={args.checkpoint_rl}" + (f", IL={args.checkpoint_il}" if args.checkpoint_il else ""),
        "",
        "variant metrics (score / DAC / NC / l2_xy):",
    ]
    for row in all_rows:
        summary_lines.append(
            f"  [{row['checkpoint_label']}] {row['variant']}: "
            f"score={row.get('score', 'NA'):.3f} "
            f"DAC={row.get('drivable_area_compliance', 'NA')} "
            f"NC={row.get('no_at_fault_collisions', 'NA')} "
            f"l2_xy={row.get('l2_xy', float('nan')):.3f}"
        )
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"[info] 摘要已写入 {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
