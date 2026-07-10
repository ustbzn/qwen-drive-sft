#!/usr/bin/env python3
"""
将指定 token 所在 log 的连续采样帧渲染为视频（参考 plt_all_log.py 逻辑）。

用法（在项目根目录）：
  python scripts/vis_scenario_video.py 002e450c6cba5895 0033fc4b187a5f1f
  python scripts/vis_scenario_video.py --model both --window 30 --token token1 token2

默认以 anchor token 为中心截取前后各 30 帧；加 --full-log 可渲染整条 log。
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

os.environ.setdefault("OPENSCENE_DATA_ROOT", "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import hydra
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from tqdm import tqdm

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.planning.scenario_builder.mogo_sqlite_map_adapter import (
    MogoRouteContext,
    MogoSQLiteMapAdapter,
    MogoSQLiteMapConfig,
)
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

CHECKPOINTS = {
    "il": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/exp/training_recogdrive_ego_SDE_obj_hdmap_IL/2026.07.05.02.56.27/lightning_logs/version_0/checkpoints/epoch=196-step=5319.ckpt",
    "rl": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/exp/training_recogdrive_ego_SDE_obj_hdmap_novlm_RL/2026.07.07.13.53.42/lightning_logs/version_0/checkpoints/epoch=19-step=640.ckpt",
}
VLM_PATH = "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct"
DATA_SPLIT = "val"
HYDRA_CONFIG_PATH = "../navsim/planning/script/config/common/train_test_split"


def _default_mogo_map_sqlite_path(openscene_root: Path) -> Path:
    maps_dir = openscene_root / "maps"
    candidates = sorted(maps_dir.glob("*.sqlite"))
    if not candidates:
        raise FileNotFoundError(f"No MOGO sqlite map found under {maps_dir}")
    return candidates[0]


def _front_camera_sensor_config() -> SensorConfig:
    sensor_config = SensorConfig.build_all_sensors(include=False)
    sensor_config.cam_f0 = True
    return sensor_config


def _build_mogo_map_api_builder(sqlite_path: Path, num_history_frames: int, utm_zone: int, utm_south: bool):
    map_config = MogoSQLiteMapConfig(
        sqlite_path=sqlite_path,
        map_name="mogo",
        utm_zone=utm_zone,
        utm_south=utm_south,
    )

    def _builder(scene_dict_list):
        current_frame_idx = num_history_frames - 1
        route_roadblock_ids = list(scene_dict_list[current_frame_idx]["roadblock_ids"])
        route_context = MogoRouteContext(roadblock_ids=route_roadblock_ids)
        return MogoSQLiteMapAdapter(config=map_config, route_context=route_context)

    return _builder


def build_log_token_sequences(scene_loader: SceneLoader) -> dict[str, list[str]]:
    log_to_frames: dict[str, list[dict]] = {}
    for token, scene_frame_list in scene_loader.scene_frames_dicts.items():
        log_name = scene_loader.token_to_log_file[token]
        center_idx = scene_loader._scene_filter.num_history_frames - 1
        log_to_frames.setdefault(log_name, []).append(scene_frame_list[center_idx])

    log_to_tokens: dict[str, list[str]] = {}
    for log_name, frames in log_to_frames.items():
        frames.sort(key=lambda x: x["timestamp"])
        log_to_tokens[log_name] = [frame["token"] for frame in frames]
    return log_to_tokens


def slice_tokens(sorted_tokens: list[str], anchor: str, window: int | None, full_log: bool) -> list[str]:
    if full_log:
        return sorted_tokens
    anchor_idx = sorted_tokens.index(anchor)
    start = max(0, anchor_idx - window)
    end = min(len(sorted_tokens), anchor_idx + window + 1)
    return sorted_tokens[start:end]


def render_frames(
    tokens: list[str],
    agent: ReCogDriveAgent,
    scene_loader: SceneLoader,
    scene_loader_traj: SceneLoader,
    frame_dir: Path,
) -> list[Path]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    center_idx = scene_loader._scene_filter.num_history_frames - 1
    frame_paths: list[Path] = []

    for i, token in enumerate(tqdm(tokens, desc=f"渲染帧 -> {frame_dir.name}")):
        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        agent_trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input(), scene_traj)
        fig, _, _ = plot_bev_and_camera_with_agent(
            scene,
            scene_traj,
            center_idx,
            agent,
            agent_trajectory=agent_trajectory,
        )
        frame_path = frame_dir / f"{i:05d}_{token}.png"
        fig.savefig(frame_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        frame_paths.append(frame_path)
    return frame_paths


def frames_to_video(frame_paths: list[Path], out_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("没有可合成的帧")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for i, frame_path in enumerate(frame_paths):
            shutil.copy(frame_path, tmp / f"{i:05d}.png")

        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(tmp / "%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 编码失败:\n{result.stderr}")


def resolve_log_names(anchor_tokens: list[str], scene_filter: SceneFilter, openscene_root: Path) -> dict[str, str]:
    scene_filter.tokens = anchor_tokens
    navsim_logs = openscene_root / "navsim_logs" / DATA_SPLIT
    sensor_blobs = openscene_root / "sensor_blobs" / DATA_SPLIT
    sensor_config = _front_camera_sensor_config()
    scene_loader = SceneLoader(navsim_logs, sensor_blobs, scene_filter, sensor_config=sensor_config)

    token_to_log = {}
    for token in anchor_tokens:
        if token not in scene_loader.tokens:
            raise ValueError(f"Token {token} 不在 navtest 数据中")
        token_to_log[token] = scene_loader.token_to_log_file[token]
    return token_to_log


def load_scene_loaders(
    log_names: list[str],
    scene_filter: SceneFilter,
    openscene_root: Path,
    mogo_map_sqlite_path: Path,
    mogo_map_utm_zone: int,
    mogo_map_utm_south: bool,
) -> tuple[SceneLoader, SceneLoader]:
    scene_filter.tokens = None
    scene_filter.log_names = log_names

    navsim_logs = openscene_root / "navsim_logs" / DATA_SPLIT
    sensor_blobs = openscene_root / "sensor_blobs" / DATA_SPLIT
    sensor_config = _front_camera_sensor_config()
    map_api_builder = _build_mogo_map_api_builder(
        mogo_map_sqlite_path,
        scene_filter.num_history_frames,
        mogo_map_utm_zone,
        mogo_map_utm_south,
    )

    scene_loader = SceneLoader(
        navsim_logs,
        sensor_blobs,
        scene_filter,
        sensor_config=sensor_config,
        map_api_builder=map_api_builder,
    )
    scene_loader_traj = SceneLoader(
        navsim_logs,
        sensor_blobs,
        scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
        map_api_builder=map_api_builder,
    )
    return scene_loader, scene_loader_traj

def generate_videos_for_tokens(
    anchor_tokens: list[str],
    models: list[str],
    scene_filter: SceneFilter,
    args,
    output_dir: Path,
    frames_root: Path,
    openscene_root: Path,
    mogo_map_sqlite_path: Path,
):
    token_to_log = resolve_log_names(anchor_tokens, scene_filter, openscene_root)
    log_names = sorted(set(token_to_log.values()))
    scene_loader, scene_loader_traj = load_scene_loaders(
        log_names,
        scene_filter,
        openscene_root,
        mogo_map_sqlite_path,
        args.mogo_map_utm_zone,
        args.mogo_map_utm_south,
    )
    log_to_tokens = build_log_token_sequences(scene_loader)

    for model in models:
        agent = ReCogDriveAgent(
            TrajectorySampling(time_horizon=4, interval_length=0.5),
            checkpoint_path=CHECKPOINTS[model],
            vlm_path=VLM_PATH,
            cam_type="single",
            vlm_type=str(args.vlm_type),
            vl_hidden_dim=int(args.vl_hidden_dim),
            dit_type=str(args.dit_type),
            sampling_method=str(args.sampling_method),
            cache_mode=False,
            cache_hidden_state=False,
            vlm_size="small",
            grpo=False,
            # 模型架构开关（来自 CLI，默认对齐历史 qwen3 ego-SDE+obj+hdmap 配置）
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

        for anchor in anchor_tokens:
            log_name = token_to_log[anchor]
            tokens = slice_tokens(log_to_tokens[log_name], anchor, args.window, args.full_log)
            frame_dir = frames_root / model / anchor
            if frame_dir.exists() and not args.keep_frames:
                shutil.rmtree(frame_dir)

            frame_paths = render_frames(tokens, agent, scene_loader, scene_loader_traj, frame_dir)
            video_path = output_dir / f"{anchor}_{model}.mp4"
            frames_to_video(frame_paths, video_path, args.fps)
            print(f"已保存视频: {video_path.resolve()} ({len(frame_paths)} 帧)")

            if not args.keep_frames:
                shutil.rmtree(frame_dir)


def _add_model_arch_args(parser: argparse.ArgumentParser) -> None:
    """模型架构开关：让可视化脚本按训练时的配置构造 agent。

    默认值对齐本脚本历史硬编码的 qwen3 ego-SDE + obj(temporal) + hdmap 模型；换其它模型
    （obj/hdmap/obj SDE(ego attention)/img/vlm 置零或不用）时用这些 flag override。所有
    影响权重形状的开关都必须和 checkpoint 训练时一致，否则会被 initialize() 的 strict=False
    静默丢成随机初始化。
    """
    group = parser.add_argument_group("model architecture (align with training)")
    group.add_argument("--vlm-type", default="qwen3")
    group.add_argument("--vl-hidden-dim", type=int, default=2048)
    group.add_argument("--dit-type", default="small")
    group.add_argument("--sampling-method", default="ddim")
    # VLM 总开关 / 消融
    group.add_argument("--use-vlm", action=argparse.BooleanOptionalAction, default=False,
                       help="False=不加载/不前向 VLM，DiT 只用 obj/hdmap/img token（更小更省）")
    group.add_argument("--vlm-ablation", default="zero", choices=["none", "zero"],
                       help="zero=加载 VLM 但投影后置零（区别于 --no-use-vlm 的完全不用）")
    # ego 状态编码器（attention=SDE 状态丢弃编码器）
    group.add_argument("--ego-status-encoder-type", default="attention", choices=["mlp", "attention"])
    # 障碍物 obj token
    group.add_argument("--use-object-vector", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--object-encoder-type", default="temporal", choices=["mlp", "temporal"])
    group.add_argument("--max-obj-tokens", type=int, default=64)
    group.add_argument("--include-static-objects", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--object-token-dropout", type=float, default=0.1)
    group.add_argument("--static-speed-threshold", type=float, default=0.5)
    # hdmap 向量 token
    group.add_argument("--use-hdmap-vector", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--use-lane-boundaries", action=argparse.BooleanOptionalAction, default=True)
    # 前视图像 ResNet encoder token
    group.add_argument("--use-image-encoder", action=argparse.BooleanOptionalAction, default=False)


def main():
    parser = argparse.ArgumentParser(description="将 scenario token 所在 log 渲染为视频")
    parser.add_argument("tokens", nargs="*", help="anchor scenario token（兼容位置参数，可传多个）")
    parser.add_argument("--token", nargs="+", default=None, help="anchor scenario token（可传多个）")
    parser.add_argument("--model", choices=["il", "rl", "both"], default="rl", help="IL / RL / 两者")
    parser.add_argument("--output-dir", default="vis_output/videos/ego_SDE_obj_hdmap_10case", help="视频输出目录")
    parser.add_argument("--frames-dir", default="vis_output/video_frames", help="中间帧目录")
    parser.add_argument("--window", type=int, default=30, help="anchor 前后各取多少帧")
    parser.add_argument("--full-log", action="store_true", help="渲染整条 log（较慢）")
    parser.add_argument("--fps", type=float, default=2.0, help="输出视频帧率")
    parser.add_argument("--keep-frames", action="store_true", help="保留中间 PNG 帧")
    parser.add_argument("--gpu", type=int, default=None, help="指定 GPU")
    parser.add_argument("--mogo-map-sqlite-path", default="", help="MOGO sqlite 地图路径；默认使用 OPENSCENE_DATA_ROOT/maps 下第一个 .sqlite")
    parser.add_argument("--mogo-map-utm-zone", type=int, default=50, help="MOGO 地图经纬度转 UTM 的 zone，需和 metric caching 一致")
    parser.add_argument("--mogo-map-utm-south", action="store_true", help="使用 UTM 南半球坐标系")
    _add_model_arch_args(parser)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    models = ["il", "rl"] if args.model == "both" else [args.model]
    output_dir = Path(args.output_dir)
    frames_root = Path(args.frames_dir)
    openscene_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    mogo_map_sqlite_path = (
        Path(args.mogo_map_sqlite_path)
        if args.mogo_map_sqlite_path
        else _default_mogo_map_sqlite_path(openscene_root)
    )

    hydra.initialize(config_path=HYDRA_CONFIG_PATH, version_base=None)
    cfg = hydra.compose(config_name="mogo_trainval")
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    
    default_token_list = [
        # "6778f89b398dee3ddde0156b2079e356",
        # "8ec9b77adeaa5c6aae500334f321f1c2",
        # "ec6788598cc3fa0b2ea7a667f03f7346",
        "506ef5edffa5ddd9acf8e72a9c78f86c",
        "9fdf107997923ef7bd7207e4618b6a6b",
        "efc7652bed3fd9c35799cc12dc5f183b",
        "b48a61ac392670f1a2296c5ef118d58a",
        "c545657ea85751ad9a6c35bc2515bd4e",
        "37227caebfdda3257c46f3314816c52a",
        "45b0c353a00f3faa12ce7d46bef5dcf6",
        "26a0a7d4da89cd5b6175b4bb6a3dbbd7",
        "15b734feb8f2b64e0032fafb34e91c6d",
        "a3879c14d1c1a07f70cdb7bb5c07e9f1",
    ]
    anchor_tokens = args.token or args.tokens
    if not anchor_tokens:
        print(f"未指定 token，使用默认 token 列表: {default_token_list}")
        anchor_tokens = default_token_list
    args.token = anchor_tokens
    generate_videos_for_tokens(
        anchor_tokens,
        models,
        scene_filter,
        args,
        output_dir,
        frames_root,
        openscene_root,
        mogo_map_sqlite_path,
    )

    # token_to_log = resolve_log_names(args.token, scene_filter, openscene_root)
    # log_names = sorted(set(token_to_log.values()))
    # scene_loader, scene_loader_traj = load_scene_loaders(
    #     log_names,
    #     scene_filter,
    #     openscene_root,
    #     mogo_map_sqlite_path,
    #     args.mogo_map_utm_zone,
    #     args.mogo_map_utm_south,
    # )
    # log_to_tokens = build_log_token_sequences(scene_loader)

    # for model in models:
    #     agent = ReCogDriveAgent(
    #         TrajectorySampling(time_horizon=4, interval_length=0.5),
    #         checkpoint_path=CHECKPOINTS[model],
    #         vlm_path=VLM_PATH,
    #         cam_type="single",
    #         vlm_type="qwen3",
    #         vl_hidden_dim=2048,
    #         dit_type="small",
    #         sampling_method="ddim",
    #         cache_mode=False,
    #         cache_hidden_state=False,
    #         vlm_size="small",
    #         grpo=False,
    #     )
    #     agent.initialize()

    #     for anchor in args.token:
    #         log_name = token_to_log[anchor]
    #         tokens = slice_tokens(log_to_tokens[log_name], anchor, args.window, args.full_log)
    #         frame_dir = frames_root / model / anchor
    #         if frame_dir.exists() and not args.keep_frames:
    #             shutil.rmtree(frame_dir)

    #         frame_paths = render_frames(tokens, agent, scene_loader, scene_loader_traj, frame_dir)
    #         video_path = output_dir / f"{anchor}_{model}.mp4"
    #         frames_to_video(frame_paths, video_path, args.fps)
    #         print(f"已保存视频: {video_path.resolve()} ({len(frame_paths)} 帧)")

    #         if not args.keep_frames:
    #             shutil.rmtree(frame_dir)


if __name__ == "__main__":
    main()
