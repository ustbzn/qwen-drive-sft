#!/usr/bin/env python3
"""生成帧间稳定、渐进贴合左侧相邻车道中心线的专用场景 GIF。

该脚本仅用于需要离线修饰预测轨迹的场景，不改变 ``vis_scenario_gif.py`` 的原始输出。
默认参数针对 7a8525b11fea08e4187b4afe77ef9117：第 10 帧开始拉起，第 16 帧完全生效。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from tqdm import tqdm

from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from scripts import vis_scenario_gif as base_gif
from scripts import vis_scenario_video as video
from scripts.lane_center_trajectory_polish import (
    LaneCenterPolishConfig,
    LeftLaneCenterline,
    extract_left_lane_centerline,
    polish_trajectories_to_left_lane_center,
)

DEFAULT_TOKEN = "7a8525b11fea08e4187b4afe77ef9117"


def _reset_inference_seed(seed: int) -> None:
    """每帧重置同一扩散噪声，避免随机采样造成可视化抖动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_and_polish(
    tokens: list[str],
    agent: video.ReCogDriveAgent,
    scene_loader: video.SceneLoader,
    scene_loader_traj: video.SceneLoader,
    config: LaneCenterPolishConfig,
    inference_seed: int,
) -> list[Trajectory]:
    center_idx = scene_loader._scene_filter.num_history_frames - 1
    raw_trajectories: list[Trajectory] = []
    centerlines: list[LeftLaneCenterline | None] = []

    for token in tqdm(tokens, desc="预测轨迹并查询左侧车道中心线"):
        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        _reset_inference_seed(inference_seed)
        trajectory = agent.compute_trajectory_vis(
            scene_traj.get_agent_input(), scene_traj
        )
        raw_trajectories.append(trajectory)

        ego_pose = scene.frames[center_idx].ego_status.ego_pose
        centerlines.append(
            extract_left_lane_centerline(scene.map_api, ego_pose, config)
        )

    available_count = sum(centerline is not None for centerline in centerlines)
    if available_count == 0:
        raise RuntimeError("全部帧均未查询到符合横向范围的左侧车道中心线")

    raw_poses = np.stack(
        [
            np.asarray(trajectory.poses, dtype=np.float32)
            for trajectory in raw_trajectories
        ]
    )
    polished_poses = polish_trajectories_to_left_lane_center(
        raw_poses, centerlines, config
    )
    reference_laterals = [
        centerline.reference_lateral_m
        for centerline in centerlines
        if centerline is not None
    ]
    path_lengths = np.linalg.norm(
        np.diff(
            np.concatenate(
                (
                    np.zeros(
                        (polished_poses.shape[0], 1, 2), dtype=np.float32
                    ),
                    polished_poses[..., :2],
                ),
                axis=1,
            ),
            axis=1,
        ),
        axis=2,
    ).sum(axis=1)
    max_length_second_diff = (
        float(np.max(np.abs(np.diff(path_lengths, n=2))))
        if path_lengths.size >= 3
        else 0.0
    )
    print(
        f"中心线可用帧: {available_count}/{len(tokens)}；"
        f"目标横向中心: {np.median(reference_laterals):.2f} m；"
        f"轨迹长度最大二阶波动: {max_length_second_diff:.2f} m"
    )

    return [
        Trajectory(
            poses=polished_poses[index],
            trajectory_sampling=raw_trajectories[index].trajectory_sampling,
        )
        for index in range(len(raw_trajectories))
    ]


def _render_polished_frames(
    tokens: list[str],
    trajectories: list[Trajectory],
    agent: video.ReCogDriveAgent,
    scene_loader: video.SceneLoader,
    scene_loader_traj: video.SceneLoader,
    prediction_frame_dir: Path,
    lqr_frame_dir: Path,
    render_dpi: int,
    bev_only: bool,
    simulator: PDMSimulator,
) -> base_gif.RenderedFramePaths:
    prediction_frame_dir.mkdir(parents=True, exist_ok=True)
    lqr_frame_dir.mkdir(parents=True, exist_ok=True)
    center_idx = scene_loader._scene_filter.num_history_frames - 1
    prediction_paths: list[Path] = []
    lqr_paths: list[Path] = []

    jobs = zip(tokens, trajectories)
    for index, (token, trajectory) in enumerate(
        tqdm(jobs, total=len(tokens), desc="渲染中心线修饰轨迹")
    ):
        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        lqr_trajectory = base_gif.build_lqr_rollout_trajectory(
            scene, center_idx, trajectory, simulator
        )

        prediction_path = prediction_frame_dir / f"{index:05d}_{token}.png"
        lqr_path = lqr_frame_dir / f"{index:05d}_{token}.png"
        base_gif._save_trajectory_frame(
            scene=scene,
            scene_traj=scene_traj,
            center_idx=center_idx,
            agent=agent,
            trajectory=trajectory,
            trajectory_name="Lane-center prediction",
            frame_path=prediction_path,
            render_dpi=render_dpi,
            bev_only=bev_only,
        )
        base_gif._save_trajectory_frame(
            scene=scene,
            scene_traj=scene_traj,
            center_idx=center_idx,
            agent=agent,
            trajectory=lqr_trajectory,
            trajectory_name="Lane-center LQR rollout",
            frame_path=lqr_path,
            render_dpi=render_dpi,
            bev_only=bev_only,
            trajectory_config=base_gif.LQR_TRAJECTORY_CONFIG,
        )
        prediction_paths.append(prediction_path)
        lqr_paths.append(lqr_path)

    return base_gif.RenderedFramePaths(
        prediction=prediction_paths,
        lqr_rollout=lqr_paths,
    )


def _encode_job(
    frame_paths: list[Path], output_path: Path, args: argparse.Namespace
) -> None:
    result = base_gif.frames_to_gif(
        frame_paths,
        output_path,
        args.fps,
        args.max_size_mb,
        args.max_width,
        args.colors,
        preserve_colors=args.bev_only,
    )
    print(
        f"已保存 GIF: {output_path.resolve()} "
        f"({result.size_bytes / base_gif.BYTES_PER_MB:.2f} MB, "
        f"{result.frame_count} 帧, {result.width}x{result.height}, "
        f"抽帧步长 {result.frame_step})"
    )


def generate_gifs_for_tokens(
    anchor_tokens: list[str],
    models: list[str],
    scene_filter: video.SceneFilter,
    args: argparse.Namespace,
    config: LaneCenterPolishConfig,
) -> None:
    output_dir = Path(args.output_dir)
    frames_root = Path(args.frames_dir)
    openscene_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    map_path = (
        Path(args.mogo_map_sqlite_path)
        if args.mogo_map_sqlite_path
        else video._default_mogo_map_sqlite_path(openscene_root)
    )
    image_views = video.normalize_image_view_list(
        args.image_view, args.image_view_list
    )
    token_to_log = video.resolve_log_names(
        anchor_tokens, scene_filter, openscene_root, image_views
    )
    log_names = sorted(set(token_to_log.values()))
    scene_loader, scene_loader_traj = video.load_scene_loaders(
        log_names,
        scene_filter,
        openscene_root,
        map_path,
        args.mogo_map_utm_zone,
        args.mogo_map_utm_south,
        image_views,
    )
    log_to_tokens = video.build_log_token_sequences(scene_loader)
    frames_root.mkdir(parents=True, exist_ok=True)
    simulator = PDMSimulator(proposal_sampling=base_gif.LQR_ROLLOUT_SAMPLING)

    for model in models:
        agent = base_gif._build_agent(model, args)
        for anchor in anchor_tokens:
            log_name = token_to_log[anchor]
            tokens = video.slice_tokens(
                log_to_tokens[log_name], anchor, args.window, args.full_log
            )
            if config.lift_start_frame >= len(tokens):
                raise ValueError(
                    f"{anchor} 仅渲染 {len(tokens)} 帧，"
                    f"无法从第 {config.lift_start_frame} 帧开始拉起"
                )

            suffix = "_bev" if args.bev_only else ""
            prediction_name = f"{anchor}_{model}_lane_center{suffix}.gif"
            lqr_name = f"{anchor}_{model}_lane_center_lqr{suffix}.gif"
            frame_name = f"{anchor}_lane_center{suffix}"

            temporary_frames = None
            if args.keep_frames:
                frame_root = frames_root / model / frame_name
            else:
                temporary_frames = tempfile.TemporaryDirectory(
                    prefix=f"{model}_{frame_name}_", dir=frames_root
                )
                frame_root = Path(temporary_frames.name)

            try:
                trajectories = _predict_and_polish(
                    tokens,
                    agent,
                    scene_loader,
                    scene_loader_traj,
                    config,
                    args.inference_seed,
                )
                rendered = _render_polished_frames(
                    tokens,
                    trajectories,
                    agent,
                    scene_loader,
                    scene_loader_traj,
                    frame_root / "prediction",
                    frame_root / "lqr_rollout",
                    args.render_dpi,
                    args.bev_only,
                    simulator,
                )
                _encode_job(
                    rendered.prediction, output_dir / prediction_name, args
                )
                _encode_job(rendered.lqr_rollout, output_dir / lqr_name, args)
            finally:
                if temporary_frames is not None:
                    temporary_frames.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成稳定并贴合左侧相邻车道中心线的专用轨迹 GIF"
    )
    parser.add_argument("tokens", nargs="*", help="anchor scenario token")
    parser.add_argument(
        "--token", nargs="+", default=None, help="anchor scenario token"
    )
    parser.add_argument(
        "--model", choices=["il", "rl", "both"], default="il", help="模型类型"
    )
    parser.add_argument(
        "--output-dir",
        default="vis_output/gifs/v3_imgencoder_4cam_novlm_IL_worldsim",
        help="GIF 输出目录",
    )
    parser.add_argument(
        "--frames-dir", default="vis_output/gif_frames", help="中间帧目录"
    )
    parser.add_argument(
        "--window", type=int, default=30, help="anchor 前后各取多少帧"
    )
    parser.add_argument("--full-log", action="store_true", help="渲染整条 log")
    parser.add_argument("--fps", type=float, default=2.0, help="GIF 帧率")
    parser.add_argument(
        "--render-dpi", type=int, default=base_gif.DEFAULT_RENDER_DPI
    )
    parser.add_argument(
        "--max-size-mb", type=float, default=base_gif.DEFAULT_MAX_SIZE_MB
    )
    parser.add_argument(
        "--max-width", type=int, default=base_gif.DEFAULT_MAX_WIDTH
    )
    parser.add_argument("--colors", type=int, default=base_gif.DEFAULT_COLORS)
    parser.add_argument(
        "--bev-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="只输出 BEV；使用 --no-bev-only 可保留相机画面",
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--lift-start-frame", type=int, default=10)
    parser.add_argument("--lift-transition-frames", type=int, default=6)
    parser.add_argument("--horizon-blend-end", type=float, default=0.625)
    parser.add_argument("--map-query-radius-m", type=float, default=100.0)
    parser.add_argument("--target-left-min-y", type=float, default=2.5)
    parser.add_argument("--target-left-max-y", type=float, default=6.0)
    parser.add_argument("--heading-tolerance-deg", type=float, default=60.0)
    parser.add_argument(
        "--centerline-anchor-max-x-m", type=float, default=15.0
    )
    parser.add_argument(
        "--centerline-join-tolerance-m", type=float, default=1.25
    )
    parser.add_argument("--mogo-map-sqlite-path", default="")
    parser.add_argument("--mogo-map-utm-zone", type=int, default=51)
    parser.add_argument("--mogo-map-utm-south", action="store_true")
    video._add_model_arch_args(parser)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.window < 0:
        parser.error("--window 不能为负数")
    if args.fps <= 0.0 or args.render_dpi <= 0 or args.max_size_mb <= 0.0:
        parser.error("fps、render-dpi 和 max-size-mb 必须大于 0")
    if args.max_width < base_gif.MIN_FALLBACK_WIDTH:
        parser.error(f"--max-width 不能小于 {base_gif.MIN_FALLBACK_WIDTH}")
    if not base_gif.MIN_FALLBACK_COLORS <= args.colors <= 256:
        parser.error(
            f"--colors 必须在 {base_gif.MIN_FALLBACK_COLORS}-256 之间"
        )

    try:
        config = LaneCenterPolishConfig(
            temporal_radius=args.temporal_radius,
            lift_start_frame=args.lift_start_frame,
            lift_transition_frames=args.lift_transition_frames,
            horizon_blend_end=args.horizon_blend_end,
            map_query_radius_m=args.map_query_radius_m,
            left_lateral_min_m=args.target_left_min_y,
            left_lateral_max_m=args.target_left_max_y,
            heading_tolerance_deg=args.heading_tolerance_deg,
            anchor_longitudinal_max_m=args.centerline_anchor_max_x_m,
            join_lateral_tolerance_m=args.centerline_join_tolerance_m,
        )
    except ValueError as error:
        parser.error(str(error))

    anchor_tokens = args.token or args.tokens or [DEFAULT_TOKEN]
    models = ["il", "rl"] if args.model == "both" else [args.model]
    hydra.initialize(config_path=video.HYDRA_CONFIG_PATH, version_base=None)
    cfg = hydra.compose(config_name="mogo_trainval")
    scene_filter: video.SceneFilter = instantiate(cfg.scene_filter)
    generate_gifs_for_tokens(anchor_tokens, models, scene_filter, args, config)


if __name__ == "__main__":
    main()
