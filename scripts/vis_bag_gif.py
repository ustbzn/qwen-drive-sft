#!/usr/bin/env python3
"""按车牌、日期和 case 查找 MOGO PKL，一个 bag 生成一个 GIF。

示例（在项目根目录、recog_drive 环境中运行）：
  python scripts/vis_bag_gif.py --plate JLBJA31588D --date 0828 --case 145200
  python scripts/vis_bag_gif.py --pkl /path/to/bag.pkl --bev-only
  python scripts/vis_bag_gif.py --date 0828 --case 145200 --dry-run
  
  python "scripts/vis_bag_gif.py" \
    --plate JLBJA31588D \
    --date 0828 \
    --case 145200 \
    --model il \
    --checkpoint "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/traffic_light_multi/exp/v3_imgencoder_4cam_novlm_IL/2026.08.12.13.42.27/lightning_logs/version_0/checkpoints/epoch=12-step=3315.ckpt" \
    --gpu 0

默认沿用 vis_scenario_gif.py 的 RL 模型、相机 + BEV + 预测/GT 样式，
只生成预测 GIF。模型需要完整历史/未来窗口，因此首尾不够窗口的原始帧
不会作为 GIF 中心帧；默认每 5 个原始帧采样一次，与参考脚本一致。
同名 JSON 保存全部原始 token 及实际渲染 token。--dry-run 只检查 PKL。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.generate_timestamped_token_list import _load_frames, _select_anchor_records

DEFAULT_DATA_ROOT = Path("/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset")


def find_bag(logs_dir: Path, plate: str, date: str, case: str) -> Path:
    """严格匹配车牌及末尾时间戳；多个匹配时要求显式指定 PKL。"""
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"PKL 搜索目录不存在: {logs_dir}")
    pattern = re.compile(
        rf"{re.escape(plate)}_.+__[0-9]{{4}}{re.escape(date)}{re.escape(case)}\.pkl"
    )
    matches = sorted(path for path in logs_dir.rglob("*.pkl") if pattern.fullmatch(path.name))
    if not matches:
        raise FileNotFoundError(f"未找到 PKL: 车牌={plate}, 日期={date}, case={case}, 目录={logs_dir}")
    if len(matches) > 1:
        raise ValueError("找到多个 PKL，请用 --pkl 指定其中一个:\n" + "\n".join(map(str, matches)))
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-dir", type=Path, help="递归搜索目录；默认 badcase_<date>/dataset/navsim_logs/train")
    parser.add_argument("--pkl", type=Path, help="直接指定 PKL，跳过车牌/日期/case 查找")
    parser.add_argument("--plate", default="JLBJA31588D", help="车牌，默认 JLBJA31588D")
    parser.add_argument("--date", default="0828", help="日期 MMDD，默认 0828")
    parser.add_argument("--case", default="145200", help="case 号 HHMMSS，默认 145200")
    parser.add_argument("--dataset-root", type=Path, help="dataset 根目录；默认从 PKL 所在 navsim_logs 目录推导")
    parser.add_argument("--sensor-blobs-dir", type=Path, help="传感器根目录；默认 dataset/sensor_blobs/<PKL split>")
    parser.add_argument("--output-dir", type=Path, default=Path("vis_output/bag_gifs"))
    parser.add_argument("--model", choices=["il", "rl"], default="rl", help="只选一个模型，默认 rl")
    parser.add_argument("--checkpoint", type=Path, help="覆盖参考脚本的模型权重；架构需与参考脚本默认值一致")
    parser.add_argument("--bev-only", action="store_true", help="只显示 BEV")
    parser.add_argument("--frame-interval", type=int, default=5, help="中心帧之间的原始帧间隔，默认 5（2 Hz）")
    parser.add_argument("--fps", type=float, help="播放帧率；默认按 PKL 时间戳和采样间隔推导")
    parser.add_argument("--render-dpi", type=int, default=100)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--max-size-mb", type=float, default=5.0, help="大小上限；超限时沿用参考脚本自动降质/抽帧")
    parser.add_argument("--keep-frames", action="store_true", help="保留中间 PNG")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--mogo-map-sqlite-path", type=Path)
    parser.add_argument("--mogo-map-utm-zone", type=int, default=51)
    parser.add_argument("--mogo-map-utm-south", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="仅定位 PKL 并检查 token，不加载模型或写文件")
    return parser


def generate_bag_gif(pkl_path: Path, frames: list, records: list, args) -> Path:
    """直接从找到的 PKL 所在目录加载场景，避免参考脚本固定 trainval 的限制。"""
    # 推迟导入模型依赖，--help / --dry-run 无需加载 torch、地图或模型。
    from scripts import vis_scenario_gif as gif

    video = gif.video
    dataset_root = args.dataset_root
    if dataset_root is None:
        navsim_dir = next((parent for parent in pkl_path.parents if parent.name == "navsim_logs"), None)
        if navsim_dir is None:
            raise ValueError("无法从 PKL 路径推导 dataset 根目录，请指定 --dataset-root")
        dataset_root = navsim_dir.parent
    sensor_blobs = args.sensor_blobs_dir or dataset_root / "sensor_blobs" / pkl_path.parent.name
    if not sensor_blobs.is_dir():
        raise FileNotFoundError(f"传感器目录不存在，请设置 --sensor-blobs-dir: {sensor_blobs}")
    map_path = args.mogo_map_sqlite_path or video._default_mogo_map_sqlite_path(dataset_root)
    if not map_path.is_file():
        raise FileNotFoundError(f"地图不存在: {map_path}")

    # 模型参数和渲染实现复用参考脚本，保持默认架构、图像视角一致。
    model_parser = argparse.ArgumentParser()
    video._add_model_arch_args(model_parser)
    model_args = model_parser.parse_args([])
    if args.checkpoint:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"模型权重不存在: {args.checkpoint}")
        video.CHECKPOINTS[args.model] = str(args.checkpoint)
    scene_filter = video.SceneFilter(
        num_history_frames=4, num_future_frames=10, frame_stride=5,
        frame_interval=args.frame_interval, has_route=True, log_names=[pkl_path.stem],
    )
    sensor_config = video._front_camera_sensor_config(model_args.image_view_list)
    map_builder = video._build_mogo_map_api_builder(
        map_path, scene_filter.num_history_frames, args.mogo_map_utm_zone, args.mogo_map_utm_south,
    )
    loader_kwargs = dict(
        data_path=pkl_path.parent, sensor_blobs_path=sensor_blobs, scene_filter=scene_filter,
        sensor_config=sensor_config, map_api_builder=map_builder,
    )
    scene_loader = video.SceneLoader(**loader_kwargs)
    scene_loader_traj = video.SceneLoader(**loader_kwargs, load_image_path=True)
    tokens = [record.token for record in records]
    if set(tokens) != set(scene_loader.tokens):
        raise RuntimeError("PKL token 采样与 SceneLoader 不一致，请检查数据或采样配置")
    agent = gif._build_agent(args.model, model_args)
    suffix = "_bev" if args.bev_only else ""
    output = args.output_dir / f"{pkl_path.stem}_{args.model}{suffix}.gif"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{pkl_path.stem}_", dir=args.output_dir) as tmp:
        frame_dir = Path(tmp)
        if args.keep_frames:
            frame_dir = args.output_dir / "frames" / output.stem
            frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for index, token in enumerate(video.tqdm(tokens, desc=f"渲染 bag {pkl_path.stem}")):
            scene = scene_loader.get_scene_from_token(token)
            scene_traj = scene_loader_traj.get_scene_from_token(token)
            trajectory = agent.compute_trajectory_vis(scene_traj.get_agent_input(), scene_traj)
            frame_path = frame_dir / f"{index:05d}_{token}.png"
            gif._save_trajectory_frame(
                scene=scene, scene_traj=scene_traj, center_idx=scene_filter.num_history_frames - 1,
                agent=agent, trajectory=trajectory, trajectory_name="Prediction",
                frame_path=frame_path, render_dpi=args.render_dpi, bev_only=args.bev_only,
            )
            frame_paths.append(frame_path)
        result = gif.frames_to_gif(
            frame_paths, output, args.fps, args.max_size_mb, args.max_width, 256,
            preserve_colors=args.bev_only,
        )

    manifest = {
        "pkl": str(pkl_path), "gif": str(output.resolve()),
        "raw_tokens": [frame["token"] for frame in frames],
        "rendered_tokens": tokens, "gif_tokens": tokens[::result.frame_step],
        "frame_interval": args.frame_interval, "gif_frame_step": result.frame_step,
        "fps": args.fps,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成一个 GIF: {output.resolve()} ({result.frame_count} 帧, {result.size_bytes / 1_000_000:.2f} MB)")
    print(f"token 清单: {output.with_suffix('.json').resolve()}")
    return output


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.pkl and (not re.fullmatch(r"[0-9]{4}", args.date) or not re.fullmatch(r"[0-9]{6}", args.case)):
        parser.error("--date 必须是四位 MMDD，--case 必须是六位 HHMMSS")
    if args.frame_interval < 1 or args.render_dpi < 1 or args.max_width < 160:
        parser.error("--frame-interval / --render-dpi 必须为正整数，--max-width 至少为 160")
    if not math.isfinite(args.max_size_mb) or args.max_size_mb <= 0:
        parser.error("--max-size-mb 必须是有限正数")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps 必须是有限正数")
    try:
        logs_dir = args.logs_dir or DEFAULT_DATA_ROOT / f"badcase_{args.date}" / "dataset/navsim_logs/train"
        pkl_path = (args.pkl or find_bag(logs_dir, args.plate, args.date, args.case)).absolute()
        frames = _load_frames(pkl_path)
        timestamps = [int(frame["timestamp"]) for frame in frames]
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError("PKL 时间戳必须严格递增，无法按时间顺序渲染")
        records = _select_anchor_records(
            frames, num_history_frames=4, num_future_frames=10,
            frame_stride=5, frame_interval=args.frame_interval, require_route=True,
        )
        if not records:
            raise ValueError("没有满足完整历史/未来窗口且含路线的可渲染帧")
        if args.fps is None:
            import statistics
            args.fps = 1_000_000 / statistics.median(
                right - left for left, right in zip(timestamps, timestamps[1:])
            ) / args.frame_interval
        print(f"PKL: {pkl_path}\n原始帧/token: {len(frames)}；可渲染帧/token: {len(records)}；FPS: {args.fps:.3f}")
        print(f"首个渲染 token: {records[0].token}\n最后渲染 token: {records[-1].token}")
        if args.dry_run:
            return
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        generate_bag_gif(pkl_path, frames, records, args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"错误: {exc}\n")


if __name__ == "__main__":
    main()
