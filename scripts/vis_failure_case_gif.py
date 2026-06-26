#!/usr/bin/env python3
"""
根据 failure_case_token_report.txt 自动为每个 token 生成 GIF。

用法（在项目根目录）：
  python scripts/vis_failure_case_gif.py
  python scripts/vis_failure_case_gif.py --model rl
  python scripts/vis_failure_case_gif.py --full-log --keep-frames
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from io import BytesIO
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

os.environ.setdefault("OPENSCENE_DATA_ROOT", "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from PIL import Image
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
    "il": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive_hdmap/exp/training_recogdrive_qwen3_hdmap_IL/2026.06.23.13.32.10/lightning_logs/version_0/checkpoints/epoch=151-step=16264.ckpt",
    "rl": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive_hdmap/exp/training_recogdrive_qwen3_hdmap_RL/2026.06.24.08.21.53/lightning_logs/version_0/checkpoints/epoch=19-step=10680.ckpt",
}
VLM_PATH = "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct"
DATA_SPLIT = "trainval"
HYDRA_CONFIG_PATH = "../navsim/planning/script/config/common/train_test_split"
DEFAULT_TOKEN_REPORT = Path(
    "vis_output/eval/2026.06.25.10.01.35_qwen3_hdmap_RL_low_score/failure_case_token_report.txt"
)
TOKEN_RE = re.compile(r"[0-9a-fA-F]{32}")
DEFAULT_RENDER_DPI = 60


def _default_mogo_map_sqlite_path(openscene_root: Path) -> Path:
    maps_dir = openscene_root / "maps"
    candidates = sorted(maps_dir.glob("*.sqlite"))
    if not candidates:
        raise FileNotFoundError(f"No MOGO sqlite map found under {maps_dir}")
    return candidates[0]


def _normalize_log_name(log_name: str) -> str:
    return Path(log_name).stem if Path(log_name).suffix == ".pkl" else log_name


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


def load_report_tokens(report_path: Path) -> tuple[list[str], dict[str, str]]:
    tokens: list[str] = []
    seen_tokens: set[str] = set()
    token_to_log: dict[str, str] = {}
    current_log: str | None = None

    content = report_path.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "Unknown pkl":
            current_log = None
            continue
        if line.endswith(".pkl"):
            current_log = _normalize_log_name(line)
            continue

        for token in TOKEN_RE.findall(line):
            token = token.lower()
            if token not in seen_tokens:
                seen_tokens.add(token)
                tokens.append(token)
            if current_log is not None and token not in token_to_log:
                token_to_log[token] = _normalize_log_name(current_log)

    return tokens, token_to_log


def resolve_missing_token_logs(
    tokens: list[str],
    token_to_log: dict[str, str],
    scene_filter: SceneFilter,
    openscene_root: Path,
) -> dict[str, str]:
    missing_tokens = [token for token in tokens if token not in token_to_log]
    if not missing_tokens:
        return token_to_log

    scene_filter.tokens = missing_tokens
    navsim_logs = openscene_root / "navsim_logs" / DATA_SPLIT
    sensor_blobs = openscene_root / "sensor_blobs" / DATA_SPLIT
    sensor_config = _front_camera_sensor_config()
    scene_loader = SceneLoader(navsim_logs, sensor_blobs, scene_filter, sensor_config=sensor_config)

    for token in missing_tokens:
        if token in scene_loader.tokens:
            token_to_log[token] = _normalize_log_name(scene_loader.token_to_log_file[token])

    return token_to_log


def build_log_token_sequences(scene_loader: SceneLoader) -> dict[str, list[str]]:
    log_to_frames: dict[str, list[dict]] = {}
    for token, scene_frame_list in scene_loader.scene_frames_dicts.items():
        log_name = _normalize_log_name(scene_loader.token_to_log_file[token])
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


def _figure_to_png_bytes(fig, dpi: int) -> bytes:
    with BytesIO() as buffer:
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=dpi)
        return buffer.getvalue()


def render_frames(
    tokens: list[str],
    agent: ReCogDriveAgent,
    scene_loader: SceneLoader,
    scene_loader_traj: SceneLoader,
    frame_dir: Path | None,
    render_dpi: int,
) -> list[Image.Image]:
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    center_idx = scene_loader._scene_filter.num_history_frames - 1
    images: list[Image.Image] = []

    for i, token in enumerate(tqdm(tokens, desc=f"渲染帧 -> {frame_dir.name if frame_dir else 'gif'}")):
        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, center_idx, agent)

        fig.set_size_inches(14, 7, forward=True)
        png_bytes = _figure_to_png_bytes(fig, render_dpi)
        if frame_dir is not None:
            (frame_dir / f"{i:05d}_{token}.png").write_bytes(png_bytes)

        images.append(Image.open(BytesIO(png_bytes)).convert("RGBA"))
        plt.close(fig)

    return images


def images_to_gif(images: list[Image.Image], out_path: Path, fps: float) -> None:
    if not images:
        raise RuntimeError("没有可合成的帧")
    if fps <= 0:
        raise ValueError("--fps 必须大于 0")

    duration_ms = max(1, int(round(1000.0 / fps)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gif_images = [image.convert("P", palette=Image.ADAPTIVE, colors=256) for image in images]
    gif_images[0].save(
        out_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=True,
    )


def load_scene_loaders(
    log_names: list[str],
    scene_filter: SceneFilter,
    openscene_root: Path,
    mogo_map_sqlite_path: Path,
    mogo_map_utm_zone: int,
    mogo_map_utm_south: bool,
) -> tuple[SceneLoader, SceneLoader]:
    scene_filter.tokens = None
    scene_filter.log_names = [_normalize_log_name(log_name) for log_name in log_names]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 failure_case_token_report.txt 生成 GIF")
    parser.add_argument(
        "--token-report",
        type=Path,
        default=DEFAULT_TOKEN_REPORT,
        help=f"token 报告路径。默认: {DEFAULT_TOKEN_REPORT}",
    )
    parser.add_argument("--model", choices=["il", "rl", "both"], default="rl", help="IL / RL / 两者")
    parser.add_argument("--output-dir", type=Path, default=None, help="GIF 输出目录")
    parser.add_argument("--frames-dir", type=Path, default=None, help="中间帧目录")
    parser.add_argument("--window", type=int, default=30, help="anchor 前后各取多少帧")
    parser.add_argument("--full-log", action="store_true", help="渲染整条 log（较慢）")
    parser.add_argument("--fps", type=float, default=2.0, help="GIF 帧率")
    parser.add_argument("--render-dpi", type=int, default=DEFAULT_RENDER_DPI, help="单帧渲染 DPI，越小文件越小")
    parser.add_argument("--keep-frames", action="store_true", help="保留中间 PNG 帧")
    parser.add_argument("--gpu", type=int, default=None, help="指定 GPU")
    parser.add_argument(
        "--mogo-map-sqlite-path",
        default="",
        help="MOGO sqlite 地图路径；默认使用 OPENSCENE_DATA_ROOT/maps 下第一个 .sqlite",
    )
    parser.add_argument("--mogo-map-utm-zone", type=int, default=50, help="MOGO 地图经纬度转 UTM 的 zone")
    parser.add_argument("--mogo-map-utm-south", action="store_true", help="使用 UTM 南半球坐标系")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.window < 0:
        raise SystemExit("--window 不能为负数")
    if args.render_dpi <= 0:
        raise SystemExit("--render-dpi 必须大于 0")

    report_path = args.token_report
    if not report_path.is_file():
        raise SystemExit(f"Token report not found: {report_path}")

    output_dir = args.output_dir or report_path.parent / "gifs"
    frames_root = args.frames_dir or output_dir / "_frames"
    openscene_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    mogo_map_sqlite_path = (
        Path(args.mogo_map_sqlite_path)
        if args.mogo_map_sqlite_path
        else _default_mogo_map_sqlite_path(openscene_root)
    )

    hydra.initialize(config_path=HYDRA_CONFIG_PATH, version_base=None)
    cfg = hydra.compose(config_name="mogo_trainval")
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)

    report_tokens, token_to_log = load_report_tokens(report_path)
    token_to_log = resolve_missing_token_logs(report_tokens, token_to_log, scene_filter, openscene_root)
    if not token_to_log:
        raise SystemExit(f"未能从报告中解析到有效 token: {report_path}")

    log_names = sorted({_normalize_log_name(name) for name in token_to_log.values()})
    scene_loader, scene_loader_traj = load_scene_loaders(
        log_names,
        scene_filter,
        openscene_root,
        mogo_map_sqlite_path,
        args.mogo_map_utm_zone,
        args.mogo_map_utm_south,
    )
    log_to_tokens = build_log_token_sequences(scene_loader)

    models = ["il", "rl"] if args.model == "both" else [args.model]
    print(f"Token report: {report_path}")
    print(f"Unique tokens: {len(report_tokens)}, logs: {len(log_names)}, models: {models}")

    for model in models:
        agent = ReCogDriveAgent(
            TrajectorySampling(time_horizon=4, interval_length=0.5),
            checkpoint_path=CHECKPOINTS[model],
            vlm_path=VLM_PATH,
            cam_type="single",
            vlm_type="qwen3",
            vl_hidden_dim=2048,
            dit_type="small",
            sampling_method="ddim",
            cache_mode=False,
            cache_hidden_state=False,
            vlm_size="small",
            grpo=False,
        )
        agent.initialize()

        for anchor in tqdm(report_tokens, desc=f"{model} tokens"):
            log_name = token_to_log.get(anchor)
            if log_name is None:
                print(f"Token {anchor} 缺少 log 信息，跳过")
                continue
            if log_name not in log_to_tokens:
                print(f"Token {anchor} 所属 log {log_name} 未加载，跳过")
                continue

            tokens = slice_tokens(log_to_tokens[log_name], anchor, args.window, args.full_log)
            frame_dir = frames_root / model / anchor if args.keep_frames else None
            images = render_frames(tokens, agent, scene_loader, scene_loader_traj, frame_dir, args.render_dpi)

            gif_path = output_dir / f"{anchor}_{model}.gif"
            images_to_gif(images, gif_path, args.fps)
            print(f"已保存 GIF: {gif_path.resolve()} ({len(images)} 帧)")


if __name__ == "__main__":
    main()
