#!/usr/bin/env python3
"""将 scenario token 所在 log 的连续采样帧渲染为小体积 GIF。

用法（在项目根目录）：
  python scripts/vis_scenario_gif.py 002e450c6cba5895
  python scripts/vis_scenario_gif.py --model both --window 30 --token token1 token2
  python scripts/vis_scenario_gif.py --bev-only --token token1

默认保证每个 GIF 不超过 5 MB（5,000,000 字节）。超过限制时，脚本会依次降低分辨率、色数和帧率，
直到输出满足大小限制；可通过 --max-size-mb 调整上限。

每个场景分别输出原始模型预测与 GT、LQR rollout 与 GT 两个 GIF。LQR rollout 使用与 PDM
评测一致的 10 Hz BatchLQRTracker 和运动学自行车模型，覆盖 4 秒预测时域。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import hydra
import numpy as np
from hydra.utils import instantiate
from PIL import Image
from tqdm import tqdm

from scripts import vis_scenario_video as video
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.scenario_builder.mogo_vehicle_parameters import get_mogo_vehicle_parameters
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

DEFAULT_MAX_SIZE_MB = 5.0
BYTES_PER_MB = 1_000_000
DEFAULT_MAX_WIDTH = 960
DEFAULT_COLORS = 256
DEFAULT_RENDER_DPI = 100
MIN_PRIMARY_WIDTH = 480
MIN_FALLBACK_WIDTH = 160
MIN_PRIMARY_COLORS = 32
MIN_FALLBACK_COLORS = 8
LQR_ROLLOUT_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)
LQR_TRAJECTORY_CONFIG = {"line_style": "-", "marker": None}


@dataclass(frozen=True)
class GifEncodingResult:
    size_bytes: int
    width: int
    height: int
    colors: int
    frame_step: int
    frame_count: int


@dataclass(frozen=True)
class RenderedFramePaths:
    prediction: list[Path]
    lqr_rollout: list[Path]


def build_lqr_rollout_trajectory(
    scene,
    frame_idx: int,
    model_trajectory: Trajectory,
    simulator: PDMSimulator,
) -> Trajectory:
    """用当前真实自车状态对模型轨迹执行 LQR + 运动学自行车模型 rollout。"""
    frame = scene.frames[frame_idx]
    ego_status = frame.ego_status
    initial_ego_state = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(*ego_status.ego_pose),
        tire_steering_angle=0.0,
        vehicle_parameters=get_mogo_vehicle_parameters(),
        time_point=TimePoint(int(frame.timestamp)),
        rear_axle_velocity_2d=StateVector2D(*ego_status.ego_velocity),
        rear_axle_acceleration_2d=StateVector2D(*ego_status.ego_acceleration),
    )

    global_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    reference_states = get_trajectory_as_array(
        global_trajectory,
        simulator.proposal_sampling,
        initial_ego_state.time_point,
    )
    simulated_states, _ = simulator.simulate_proposals(
        reference_states[None, ...],
        initial_ego_state,
    )

    origin = StateSE2(*initial_ego_state.rear_axle.serialize())
    simulated_local_poses = convert_absolute_to_relative_se2_array(
        origin,
        simulated_states[0, :, StateIndex.STATE_SE2],
    )
    return Trajectory(
        poses=simulated_local_poses[1:].astype(np.float32),
        trajectory_sampling=simulator.proposal_sampling,
    )


def _save_trajectory_frame(
    *,
    scene,
    scene_traj,
    center_idx: int,
    agent: video.ReCogDriveAgent,
    trajectory: Trajectory,
    trajectory_name: str,
    frame_path: Path,
    render_dpi: int,
    bev_only: bool,
    trajectory_config: dict[str, object] | None = None,
) -> None:
    fig, bev_ax, camera_ax = video.plot_bev_and_camera_with_agent(
        scene,
        scene_traj,
        center_idx,
        agent,
        agent_trajectory=trajectory,
        agent_trajectory_config=trajectory_config,
    )
    label_ax = bev_ax if bev_only else camera_ax
    label_ax.text(
        0.99,
        0.99,
        f"GT: green | {trajectory_name}: red",
        transform=label_ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
        zorder=10,
    )
    if bev_only:
        camera_ax.remove()
        fig.set_size_inches(9, 9, forward=True)
        bev_ax.set_position([0.03, 0.03, 0.94, 0.94])
    fig.savefig(frame_path, bbox_inches="tight", dpi=render_dpi)
    video.plt.close(fig)


def render_frames(
    tokens: list[str],
    agent: video.ReCogDriveAgent,
    scene_loader: video.SceneLoader,
    scene_loader_traj: video.SceneLoader,
    prediction_frame_dir: Path,
    lqr_frame_dir: Path,
    render_dpi: int,
    bev_only: bool,
    simulator: PDMSimulator,
) -> RenderedFramePaths:
    """一次模型推理同时渲染原始预测和 LQR rollout 两套 GIF 帧。"""
    prediction_frame_dir.mkdir(parents=True, exist_ok=True)
    lqr_frame_dir.mkdir(parents=True, exist_ok=True)
    center_idx = scene_loader._scene_filter.num_history_frames - 1
    prediction_frame_paths: list[Path] = []
    lqr_frame_paths: list[Path] = []

    description = f"渲染预测与 LQR 帧 -> {prediction_frame_dir.parent.name}"
    for i, token in enumerate(tqdm(tokens, desc=description)):
        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input(), scene_traj)
        lqr_trajectory = build_lqr_rollout_trajectory(
            scene,
            center_idx,
            agent_trajectory,
            simulator,
        )

        prediction_frame_path = prediction_frame_dir / f"{i:05d}_{token}.png"
        lqr_frame_path = lqr_frame_dir / f"{i:05d}_{token}.png"
        _save_trajectory_frame(
            scene=scene,
            scene_traj=scene_traj,
            center_idx=center_idx,
            agent=agent,
            trajectory=agent_trajectory,
            trajectory_name="Prediction",
            frame_path=prediction_frame_path,
            render_dpi=render_dpi,
            bev_only=bev_only,
        )
        _save_trajectory_frame(
            scene=scene,
            scene_traj=scene_traj,
            center_idx=center_idx,
            agent=agent,
            trajectory=lqr_trajectory,
            trajectory_name="LQR rollout",
            frame_path=lqr_frame_path,
            render_dpi=render_dpi,
            bev_only=bev_only,
            trajectory_config=LQR_TRAJECTORY_CONFIG,
        )
        prediction_frame_paths.append(prediction_frame_path)
        lqr_frame_paths.append(lqr_frame_path)

    return RenderedFramePaths(
        prediction=prediction_frame_paths,
        lqr_rollout=lqr_frame_paths,
    )


def _resample_filter():
    return Image.Resampling.LANCZOS


def _dither_mode():
    return Image.Dither.FLOYDSTEINBERG


def _load_frames(frame_paths: list[Path], max_width: int, frame_step: int) -> list[Image.Image]:
    selected_paths = frame_paths[::frame_step]
    with Image.open(selected_paths[0]) as first:
        first_width, first_height = first.size

    scale = min(1.0, max_width / first_width)
    target_size = (
        max(1, int(round(first_width * scale))),
        max(1, int(round(first_height * scale))),
    )
    frames: list[Image.Image] = []
    for frame_path in selected_paths:
        with Image.open(frame_path) as source:
            frame = source.convert("RGB")
        if frame.size != target_size:
            frame.thumbnail(target_size, _resample_filter())
            if frame.size != target_size:
                canvas = Image.new("RGB", target_size, "white")
                offset = ((target_size[0] - frame.width) // 2, (target_size[1] - frame.height) // 2)
                canvas.paste(frame, offset)
                frame.close()
                frame = canvas
        frames.append(frame)
    return frames


def _build_global_palette(frames: list[Image.Image], colors: int) -> Image.Image:
    sample_count = min(12, len(frames))
    sample_indices = [round(i * (len(frames) - 1) / max(1, sample_count - 1)) for i in range(sample_count)]
    sample_width = min(240, frames[0].width)
    sample_height = max(1, round(frames[0].height * sample_width / frames[0].width))
    palette_source = Image.new("RGB", (sample_width, sample_height * sample_count))

    for row, frame_index in enumerate(sample_indices):
        sample = frames[frame_index].resize((sample_width, sample_height), _resample_filter())
        palette_source.paste(sample, (0, row * sample_height))
        sample.close()

    palette = palette_source.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    palette_source.close()
    return palette


def _encode_gif_candidate(
    frame_paths: list[Path],
    fps: float,
    max_width: int,
    colors: int,
    frame_step: int,
    preserve_colors: bool,
) -> tuple[bytes, tuple[int, int], int]:
    frames = _load_frames(frame_paths, max_width, frame_step)
    palette = None
    if preserve_colors:
        gif_frames = [
            frame.quantize(
                colors=256,
                method=Image.Quantize.MAXCOVERAGE,
                dither=Image.Dither.NONE,
            )
            for frame in frames
        ]
    else:
        palette = _build_global_palette(frames, colors)
        gif_frames = [frame.quantize(palette=palette, dither=_dither_mode()) for frame in frames]
    duration_ms = max(1, int(round(1000.0 * frame_step / fps)))

    try:
        with BytesIO() as buffer:
            gif_frames[0].save(
                buffer,
                format="GIF",
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration_ms,
                loop=0,
                disposal=2,
                optimize=not preserve_colors,
            )
            data = buffer.getvalue()
        return data, frames[0].size, len(frames)
    finally:
        if palette is not None:
            palette.close()
        for frame in frames:
            frame.close()
        for frame in gif_frames:
            frame.close()


def frames_to_gif(
    frame_paths: list[Path],
    out_path: Path,
    fps: float,
    max_size_mb: float,
    max_width: int,
    colors: int,
    preserve_colors: bool = False,
) -> GifEncodingResult:
    """自适应编码 GIF，并在写入前校验实际大小。"""
    if not frame_paths:
        raise RuntimeError("没有可合成的帧")
    if fps <= 0:
        raise ValueError("--fps 必须大于 0")
    if max_size_mb <= 0:
        raise ValueError("--max-size-mb 必须大于 0")

    max_size_bytes = int(max_size_mb * BYTES_PER_MB)
    current_width = max_width
    current_colors = 256 if preserve_colors else colors
    frame_step = 1

    while True:
        data, (width, height), frame_count = _encode_gif_candidate(
            frame_paths,
            fps,
            current_width,
            current_colors,
            frame_step,
            preserve_colors,
        )
        size_bytes = len(data)
        if size_bytes <= max_size_bytes:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            if out_path.stat().st_size > max_size_bytes:
                raise RuntimeError(f"GIF 写入后超过大小限制: {out_path}")
            return GifEncodingResult(
                size_bytes=size_bytes,
                width=width,
                height=height,
                colors=current_colors,
                frame_step=frame_step,
                frame_count=frame_count,
            )

        size_ratio = max_size_bytes / size_bytes
        if current_width > MIN_PRIMARY_WIDTH:
            estimated_width = int(current_width * math.sqrt(size_ratio) * 0.95)
            current_width = max(MIN_PRIMARY_WIDTH, min(current_width - 1, estimated_width))
        elif not preserve_colors and current_colors > MIN_PRIMARY_COLORS:
            current_colors = max(MIN_PRIMARY_COLORS, current_colors // 2)
        elif frame_step < len(frame_paths):
            estimated_step = math.ceil(frame_step / size_ratio * 1.05)
            frame_step = min(len(frame_paths), max(frame_step + 1, estimated_step))
        elif current_width > MIN_FALLBACK_WIDTH:
            current_width = max(MIN_FALLBACK_WIDTH, int(current_width * 0.7))
        elif not preserve_colors and current_colors > MIN_FALLBACK_COLORS:
            current_colors = max(MIN_FALLBACK_COLORS, current_colors // 2)
        else:
            raise RuntimeError(
                f"无法在最低质量参数下将 GIF 压缩到 {max_size_mb:g} MB；"
                "请提高 --max-size-mb"
            )


def _build_agent(model: str, args) -> video.ReCogDriveAgent:
    agent = video.ReCogDriveAgent(
        video.TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path=video.CHECKPOINTS[model],
        vlm_path=video.VLM_PATH,
        cam_type="single",
        vlm_type=str(args.vlm_type),
        vl_hidden_dim=int(args.vl_hidden_dim),
        dit_type=str(args.dit_type),
        sampling_method=str(args.sampling_method),
        cache_mode=False,
        cache_hidden_state=False,
        vlm_size="small",
        grpo=False,
        use_vlm=bool(args.use_vlm),
        vlm_ablation=str(args.vlm_ablation),
        ego_status_encoder_type=str(args.ego_status_encoder_type),
        use_hdmap_vector=bool(args.use_hdmap_vector),
        use_lane_boundaries=bool(args.use_lane_boundaries),
        use_object_vector=bool(args.use_object_vector),
        object_encoder_type=str(args.object_encoder_type),
        static_speed_threshold=float(args.static_speed_threshold),
        max_obj_tokens=int(args.max_obj_tokens),
        include_static_objects=bool(args.include_static_objects),
        object_token_dropout=float(args.object_token_dropout),
        use_image_encoder=bool(args.use_image_encoder),
    )
    agent.initialize()
    return agent


def generate_gifs_for_tokens(
    anchor_tokens: list[str],
    models: list[str],
    scene_filter: video.SceneFilter,
    args,
    output_dir: Path,
    frames_root: Path,
    openscene_root: Path,
    mogo_map_sqlite_path: Path,
) -> None:
    token_to_log = video.resolve_log_names(anchor_tokens, scene_filter, openscene_root)
    log_names = sorted(set(token_to_log.values()))
    scene_loader, scene_loader_traj = video.load_scene_loaders(
        log_names,
        scene_filter,
        openscene_root,
        mogo_map_sqlite_path,
        args.mogo_map_utm_zone,
        args.mogo_map_utm_south,
    )
    log_to_tokens = video.build_log_token_sequences(scene_loader)
    frames_root.mkdir(parents=True, exist_ok=True)
    simulator = PDMSimulator(proposal_sampling=LQR_ROLLOUT_SAMPLING)

    for model in models:
        agent = _build_agent(model, args)
        for anchor in anchor_tokens:
            log_name = token_to_log[anchor]
            tokens = video.slice_tokens(log_to_tokens[log_name], anchor, args.window, args.full_log)
            if args.bev_only:
                prediction_output_name = f"{anchor}_{model}_bev.gif"
                lqr_output_name = f"{anchor}_{model}_lqr_bev.gif"
                frame_name = f"{anchor}_bev"
            else:
                prediction_output_name = f"{anchor}_{model}.gif"
                lqr_output_name = f"{anchor}_{model}_lqr.gif"
                frame_name = anchor

            temporary_frames = None
            if args.keep_frames:
                frame_root = frames_root / model / frame_name
            else:
                temporary_frames = tempfile.TemporaryDirectory(
                    prefix=f"{model}_{frame_name}_",
                    dir=frames_root,
                )
                frame_root = Path(temporary_frames.name)
            prediction_frame_dir = frame_root / "prediction"
            lqr_frame_dir = frame_root / "lqr_rollout"

            try:
                rendered_frames = render_frames(
                    tokens,
                    agent,
                    scene_loader,
                    scene_loader_traj,
                    prediction_frame_dir,
                    lqr_frame_dir,
                    args.render_dpi,
                    args.bev_only,
                    simulator,
                )
                gif_jobs = (
                    (rendered_frames.prediction, prediction_output_name),
                    (rendered_frames.lqr_rollout, lqr_output_name),
                )
                for frame_paths, output_name in gif_jobs:
                    gif_path = output_dir / output_name
                    result = frames_to_gif(
                        frame_paths,
                        gif_path,
                        args.fps,
                        args.max_size_mb,
                        args.max_width,
                        args.colors,
                        preserve_colors=args.bev_only,
                    )
                    size_mb = result.size_bytes / BYTES_PER_MB
                    print(
                        f"已保存 GIF: {gif_path.resolve()} "
                        f"({size_mb:.2f} MB, {result.frame_count} 帧, "
                        f"{result.width}x{result.height}, {result.colors} 色, "
                        f"抽帧步长 {result.frame_step})"
                    )
            finally:
                if temporary_frames is not None:
                    temporary_frames.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="将 scenario token 所在 log 渲染为不超过指定大小的 GIF")
    parser.add_argument("tokens", nargs="*", help="anchor scenario token（兼容位置参数，可传多个）")
    parser.add_argument("--token", nargs="+", default=None, help="anchor scenario token（可传多个）")
    parser.add_argument("--model", choices=["il", "rl", "both"], default="il", help="IL / RL / 两者")
    parser.add_argument(
        "--output-dir",
        default="vis_output/gifs/v1.1_alldata_traffic_light_novlm_il_worldsim",
        help="GIF 输出目录",
    )
    parser.add_argument("--frames-dir", default="vis_output/gif_frames", help="中间帧目录")
    parser.add_argument("--window", type=int, default=30, help="anchor 前后各取多少帧")
    parser.add_argument("--full-log", action="store_true", help="渲染整条 log（较慢且可能自动抽帧）")
    parser.add_argument("--fps", type=float, default=2.0, help="GIF 初始帧率")
    parser.add_argument("--render-dpi", type=int, default=DEFAULT_RENDER_DPI, help="单帧渲染 DPI")
    parser.add_argument("--max-size-mb", type=float, default=DEFAULT_MAX_SIZE_MB, help="单个 GIF 大小上限，单位 MB")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH, help="GIF 初始最大宽度")
    parser.add_argument("--colors", type=int, default=DEFAULT_COLORS, help=f"GIF 初始色数，范围 {MIN_FALLBACK_COLORS}-256")
    parser.add_argument(
        "--bev-only",
        action="store_true",
        help="移除左侧相机图，仅输出右侧 BEV，并固定使用 GIF 最大 256 色",
    )
    parser.add_argument("--keep-frames", action="store_true", help="保留中间 PNG 帧")
    parser.add_argument("--gpu", type=int, default=None, help="指定 GPU")
    parser.add_argument(
        "--mogo-map-sqlite-path",
        default="",
        help="MOGO sqlite 地图路径；默认使用 OPENSCENE_DATA_ROOT/maps 下第一个 .sqlite",
    )
    parser.add_argument(
        "--mogo-map-utm-zone",
        type=int,
        default=51,
        help="MOGO 地图经纬度转 UTM 的 zone，需和 metric caching 一致",
    )
    parser.add_argument("--mogo-map-utm-south", action="store_true", help="使用 UTM 南半球坐标系")
    video._add_model_arch_args(parser)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.window < 0:
        parser.error("--window 不能为负数")
    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if args.render_dpi <= 0:
        parser.error("--render-dpi 必须大于 0")
    if args.max_size_mb <= 0:
        parser.error("--max-size-mb 必须大于 0")
    if args.max_width < MIN_FALLBACK_WIDTH:
        parser.error(f"--max-width 不能小于 {MIN_FALLBACK_WIDTH}")
    if not MIN_FALLBACK_COLORS <= args.colors <= 256:
        parser.error(f"--colors 必须在 {MIN_FALLBACK_COLORS}-256 之间")

    default_token_list = [
        "6778f89b398dee3ddde0156b2079e356",
        "8ec9b77adeaa5c6aae500334f321f1c2",
        "ec6788598cc3fa0b2ea7a667f03f7346",
        "7a8525b11fea08e4187b4afe77ef9117",
        "1e223bdaeed2f9c46bb351de8e712168",
        "99bba5e74abb2261fa0c4a6328c1c611",
        "f7dc542d3e1bf2c84c429f859b70ecbf",
        # val case
        # "506ef5edffa5ddd9acf8e72a9c78f86c",
        # "9fdf107997923ef7bd7207e4618b6a6b",
        # "efc7652bed3fd9c35799cc12dc5f183b",
        # "b48a61ac392670f1a2296c5ef118d58a",
        # "c545657ea85751ad9a6c35bc2515bd4e",
        # "37227caebfdda3257c46f3314816c52a",
        # "45b0c353a00f3faa12ce7d46bef5dcf6",
        # "26a0a7d4da89cd5b6175b4bb6a3dbbd7",
        # "15b734feb8f2b64e0032fafb34e91c6d",
        # "a3879c14d1c1a07f70cdb7bb5c07e9f1",
        # "1d9efa9752fff8fe4c01d170964f635f",
        # "37ad798b2faabb0e60426aef42ffcd3a",
        # "32e1c3140db74ec058c785f77193712a",
        # "1dd6b4cb40661650a0e62a81fe5506fc",
        # "5afd8ce7ff7defc319ef16920788f801",
        # "6dee9f3ef67a08bee0ae2c211f1383e4",
    ]
    anchor_tokens = args.token or args.tokens
    if not anchor_tokens:
        print(f"未指定 token，使用默认 token 列表: {default_token_list}")
        anchor_tokens = default_token_list

    models = ["il", "rl"] if args.model == "both" else [args.model]
    output_dir = Path(args.output_dir)
    frames_root = Path(args.frames_dir)
    openscene_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    mogo_map_sqlite_path = (
        Path(args.mogo_map_sqlite_path)
        if args.mogo_map_sqlite_path
        else video._default_mogo_map_sqlite_path(openscene_root)
    )

    hydra.initialize(config_path=video.HYDRA_CONFIG_PATH, version_base=None)
    cfg = hydra.compose(config_name="mogo_trainval")
    scene_filter: video.SceneFilter = instantiate(cfg.scene_filter)
    generate_gifs_for_tokens(
        anchor_tokens,
        models,
        scene_filter,
        args,
        output_dir,
        frames_root,
        openscene_root,
        mogo_map_sqlite_path,
    )


if __name__ == "__main__":
    main()
