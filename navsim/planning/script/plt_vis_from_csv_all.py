import os
import argparse
import sys
import math
from pathlib import Path

import hydra
from hydra import initialize_config_dir
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.distributed as dist
from tqdm import tqdm
import csv

# 确保从任意工作目录运行都能 import `navsim`
_REPO_ROOT = Path(__file__).resolve().parents[3]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


def init_distributed():
    """可选分布式初始化：支持 torchrun 多卡分摊渲染任务。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"Distributed init: rank {rank}/{world_size}, local_rank {local_rank}")
        return rank, world_size, local_rank
    return 0, 1, 0


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fmt_metric(v: object) -> str:
    if v is None:
        return "NA"
    try:
        f = float(v)
        if not math.isfinite(f):
            return "NA"
        return f"{f:.3f}"
    except Exception:
        return str(v)


def _build_camera_overlay_text(token: str, metrics: dict[str, object]) -> str:
    # 兼容 CSV 里不存在的列：不存在就显示 NA
    parts = [f"token: {token}"]
    for k in [
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
        "score",
    ]:
        if k in metrics or k in {
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
            "score",
        }:
            short = {
                "no_at_fault_collisions": "NC",
                "drivable_area_compliance": "DAC",
                "ego_progress": "EP",
                "time_to_collision_within_bound": "TTC",
                "comfort": "C",
                "driving_direction_compliance": "DDC",
                "score": "score",
            }[k]
            parts.append(f"{short}: {_fmt_metric(metrics.get(k))}")
    return " | ".join(parts)


def build_scene_filter(scene_filter_name: str) -> SceneFilter:
    config_dir = (
        Path(__file__).resolve().parent / "config" / "common" / "train_test_split" / "scene_filter"
    ).as_posix()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name=scene_filter_name)
    return instantiate(cfg)


def build_agent(args, device: torch.device) -> ReCogDriveAgent:
    agent = ReCogDriveAgent(
        TrajectorySampling(time_horizon=float(args.time_horizon), interval_length=float(args.interval_length)),
        checkpoint_path=str(args.checkpoint),
        vlm_path=str(args.vlm_path),
        cam_type=str(args.cam_type),
        vlm_type=str(args.vlm_type),
        dit_type=str(args.dit_type),
        sampling_method=str(args.sampling_method),
        cache_mode=bool(args.cache_mode),
        cache_hidden_state=bool(args.cache_hidden_state),
        vlm_size=str(args.vlm_size),
        grpo=bool(args.grpo),
    ).to(device)
    agent.initialize()
    return agent


def build_scene_loaders(
    *,
    openscene_data_root: Path,
    split: str,
    scene_filter: SceneFilter,
    sensor_config: SensorConfig,
) -> tuple[SceneLoader, SceneLoader]:
    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{split}",
        openscene_data_root / f"sensor_blobs/{split}",
        scene_filter,
        sensor_config=sensor_config,
    )
    scene_loader_traj = SceneLoader(
        openscene_data_root / f"navsim_logs/{split}",
        openscene_data_root / f"sensor_blobs/{split}",
        scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )
    return scene_loader, scene_loader_traj


def tokens_from_csv(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    # 强制 token 为字符串，保留前导 0
    df = pd.read_csv(csv_path, dtype={"token": "string"})
    if "token" not in df.columns:
        raise ValueError(f"CSV 缺少 token 列: {csv_path}")
    tokens = [str(t).strip() for t in df["token"].dropna().astype(str).tolist()]
    # 去重但保持顺序
    seen: set[str] = set()
    uniq: list[str] = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return df, uniq


def save_vis_for_tokens(
    *,
    tokens: list[str],
    df: pd.DataFrame,
    out_dir: Path,
    scene_loader: SceneLoader,
    scene_loader_traj: SceneLoader,
    agent: ReCogDriveAgent,
    rank: int,
    world_size: int,
) -> int:
    _ensure_dir(out_dir)
    df_index = df.set_index("token", drop=False) if "token" in df.columns else None
    local_tokens = tokens[rank::world_size]
    saved = 0
    index_rows: list[dict[str, object]] = []

    for token in tqdm(local_tokens, desc=f"Rank {rank} 渲染 all"):
        try:
            scene = scene_loader.get_scene_from_token(token)
            scene_traj = scene_loader_traj.get_scene_from_token(token)
            frame_idx = scene.scene_metadata.num_history_frames - 1

            metrics: dict[str, object] = {}
            score = None
            comfort = None
            if df_index is not None and token in df_index.index:
                row = df_index.loc[token]
                for c in df.columns:
                    if c == "token":
                        continue
                    metrics[c] = row[c]
                if "score" in metrics:
                    try:
                        score = float(metrics["score"])
                    except Exception:
                        score = None
                if "comfort" in metrics:
                    try:
                        comfort = float(metrics["comfort"])
                    except Exception:
                        comfort = None

            fig, _bev_ax, camera_ax = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)
            if camera_ax is not None:
                overlay = _build_camera_overlay_text(token, metrics)
                camera_ax.text(
                    0.01,
                    0.99,
                    overlay,
                    transform=camera_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
                    zorder=10,
                )
                camera_ax.text(
                    0.01,
                    0.01,
                    "GT trajectory: green | Planner trajectory: red",
                    transform=camera_ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
                    zorder=10,
                )

            # 用户要求：文件名为 token_scoreX.XXX.png
            score_s = f"{score:.3f}" if isinstance(score, float) and math.isfinite(score) else "NA"
            file_name = f"{token}_score{score_s}.png"
            save_path = out_dir / file_name
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            saved += 1

            index_row: dict[str, object] = {"token": token, "image_path": str(save_path)}
            if score is not None:
                index_row["score"] = score
            for k, v in metrics.items():
                # 避免重复覆盖 token/image_path
                if k in {"token", "image_path"}:
                    continue
                index_row[k] = v
            index_rows.append(index_row)
        except Exception as e:
            print(f"[WARN] token={token} 渲染失败: {e}")

    # 每个 rank 写索引，方便检索
    index_path = out_dir / f"index_rank{rank}.csv"
    # 字段：token/score + 指标列 + image_path（尽量覆盖）
    fieldnames = ["token"]
    if "score" in df.columns:
        fieldnames.append("score")
    for c in df.columns:
        if c in {"token", "score"}:
            continue
        fieldnames.append(c)
    fieldnames.append("image_path")
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="从指定 CSV 渲染所有 token 的轨迹图")
    parser.add_argument(
        "--csv",
        required=True,
        help="推理结果 CSV 路径（至少包含 token 列）",
    )
    parser.add_argument("--out", default="vis_output/eval/0428_163122", help="输出目录")
    parser.add_argument("--token", default="", help="可选：只渲染单个 token（快速自测）")

    parser.add_argument("--split", default="test", choices=["mini", "test", "trainval"], help="NAVSIM split")
    parser.add_argument("--scene_filter", default="navtest", help="scene_filter 配置名（Hydra）")
    parser.add_argument(
        "--openscene_data_root", default=os.getenv("OPENSCENE_DATA_ROOT", ""), help="OPENSCENE_DATA_ROOT"
    )

    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径（用于生成 agent 轨迹）")
    parser.add_argument("--vlm_path", required=True, help="VLM checkpoint/路径（用于生成 backbone 特征）")
    parser.add_argument("--cam_type", default="single")
    parser.add_argument("--vlm_type", default="internvl")
    parser.add_argument("--dit_type", default="small")
    parser.add_argument("--sampling_method", default="ddim")
    parser.add_argument("--cache_mode", action="store_true")
    parser.add_argument("--cache_hidden_state", action="store_true")
    parser.add_argument("--vlm_size", default="small")
    parser.add_argument("--grpo", action="store_true")

    parser.add_argument("--time_horizon", type=float, default=4.0)
    parser.add_argument("--interval_length", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if not args.openscene_data_root:
        raise ValueError("需要设置 --openscene_data_root 或环境变量 OPENSCENE_DATA_ROOT")

    df, tokens = tokens_from_csv(args.csv)
    if args.token:
        tokens = [str(args.token)]

    scene_filter = build_scene_filter(args.scene_filter)
    agent = build_agent(args, device)
    # 只渲染 + 推理需要前视 cam_f0；为避免某些 history iteration 未加载导致 cam_f0.image=None，
    # 这里显式只加载 cam_f0（所有 history iteration）。
    sensor_cfg = SensorConfig.build_all_sensors(include=False)
    sensor_cfg.cam_f0 = True
    scene_loader, scene_loader_traj = build_scene_loaders(
        openscene_data_root=Path(args.openscene_data_root),
        split=args.split,
        scene_filter=scene_filter,
        sensor_config=sensor_cfg,
    )

    out_dir = Path(args.out)
    saved = save_vis_for_tokens(
        tokens=tokens,
        df=df,
        out_dir=out_dir,
        scene_loader=scene_loader,
        scene_loader_traj=scene_loader_traj,
        agent=agent,
        rank=rank,
        world_size=world_size,
    )
    print(f"Rank {rank}: 保存 {saved} 张 -> {out_dir}")


if __name__ == "__main__":
    main()

