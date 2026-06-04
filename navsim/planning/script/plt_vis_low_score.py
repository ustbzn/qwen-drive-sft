import os
import argparse
import math
from pathlib import Path

import csv
import sys
import hydra
from hydra.utils import instantiate
from hydra import initialize_config_dir
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

# 确保从任意工作目录运行都能 import `navsim`
_REPO_ROOT = Path(__file__).resolve().parents[3]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


METRIC_COLS_DEFAULT = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
]


def _pick_metric_cols(df: pd.DataFrame) -> list[str]:
    """优先输出常用 PDM 指标；若缺失则输出除 token 外的全部列。"""
    cols = [c for c in METRIC_COLS_DEFAULT if c in df.columns]
    if cols:
        return cols
    return [c for c in df.columns if c != "token"]


def _fmt_metric(v: object) -> str:
    try:
        fv = float(v)  # type: ignore[arg-type]
        if math.isfinite(fv):
            return f"{fv:.3f}"
    except Exception:
        pass
    return "NA"


def _build_camera_overlay_text(token: str, metrics: dict[str, object]) -> str:
    # 指标缩写映射（按你指定）
    # NC: no_at_fault_collisions, DAC: drivable_area_compliance, EP: ego_progress, TTC: time_to_collision_within_bound, C: comfort
    parts = [
        f"token: {token}",
        f"NC: {_fmt_metric(metrics.get('no_at_fault_collisions'))}",
        f"DAC: {_fmt_metric(metrics.get('drivable_area_compliance'))}",
        f"EP: {_fmt_metric(metrics.get('ego_progress'))}",
        f"TTC: {_fmt_metric(metrics.get('time_to_collision_within_bound'))}",
        f"C: {_fmt_metric(metrics.get('comfort'))}",
        f"score: {_fmt_metric(metrics.get('score'))}",
    ]
    return " | ".join(parts)


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


def _validate_csv_columns(df: pd.DataFrame) -> None:
    required = {"token", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")


def select_top100_and_p1_p5(df: pd.DataFrame) -> tuple[list[str], list[str], float, float]:
    """
    选择两组 token:
    - top100: score 最低的 100 条
    - p1_p5: score 介于 p1 和 p5（含边界）的所有样本
    返回: (top100_tokens, p1_p5_tokens, q01, q05)
    """
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score", "token"])
    df["token"] = df["token"].astype(str)

    q01 = float(df["score"].quantile(0.01))
    q05 = float(df["score"].quantile(0.05))

    df_sorted = df.sort_values("score", ascending=True)
    top100_tokens = df_sorted.head(100)["token"].tolist()

    p1_p5_mask = (df["score"] >= q01) & (df["score"] <= q05)
    p1_p5_tokens = df.loc[p1_p5_mask, "token"].tolist()
    return top100_tokens, p1_p5_tokens, q01, q05


def build_agent(args, device: torch.device) -> ReCogDriveAgent:
    agent = ReCogDriveAgent(
        TrajectorySampling(time_horizon=args.time_horizon, interval_length=args.interval_length),
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm_path,
        cam_type=args.cam_type,
        vlm_type=args.vlm_type,
        dit_type=args.dit_type,
        sampling_method=args.sampling_method,
        cache_mode=bool(args.cache_mode),
        cache_hidden_state=bool(args.cache_hidden_state),
        vlm_size=args.vlm_size,
        grpo=bool(args.grpo),
        normalize_vl_hidden_states=bool(args.normalize_vl_hidden_states),
        first_hidden_state_index=int(args.first_hidden_state_index),
        last_hidden_state_index=int(args.last_hidden_state_index),
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
    max_items: int | None = None,
) -> int:
    _ensure_dir(out_dir)
    df_index = df.set_index("token", drop=False)

    if max_items is not None:
        tokens = tokens[:max_items]

    local_tokens = tokens[rank::world_size]
    saved = 0

    metric_cols = _pick_metric_cols(df)
    index_rows: list[dict] = []

    for token in tqdm(local_tokens, desc=f"Rank {rank} 渲染 {out_dir.name}"):
        try:
            scene = scene_loader.get_scene_from_token(token)
            scene_traj = scene_loader_traj.get_scene_from_token(token)
            frame_idx = scene.scene_metadata.num_history_frames - 1

            row = df_index.loc[token] if token in df_index.index else None
            metrics: dict[str, object] = {}
            score = None
            comfort = None
            if row is not None:
                for c in metric_cols:
                    try:
                        metrics[c] = float(row[c])
                    except Exception:
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
            # 将指标写在“相机视角”那张图上（不是 BEV）
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
                # 轨迹颜色图例（相机视角左下角）
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

            score_s = f"{score:.3f}" if isinstance(score, float) and math.isfinite(score) else "NA"
            comfort_s = f"{comfort:.3f}" if isinstance(comfort, float) and math.isfinite(comfort) else "NA"
            file_name = f"score{score_s}_comfort{comfort_s}_token{token}.png"
            save_path = out_dir / file_name
            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close(fig)
            saved += 1

            index_row = {"token": token}
            index_row.update(metrics)
            index_row["image_path"] = str(save_path)

            index_rows.append(index_row)
        except Exception as e:
            print(f"[WARN] token={token} 渲染失败: {e}")

    # 每个 rank 写自己的索引，避免多进程竞争写同一文件
    index_path = out_dir / f"index_rank{rank}.csv"
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["token", *metric_cols, "image_path"])
        writer.writeheader()
        writer.writerows(index_rows)

    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="根据推理 CSV 可视化低分轨迹（Top100 + p1-p5）")
    parser.add_argument("--csv", required=True, help="推理结果 CSV 路径（含 token, score, comfort 等列）")
    parser.add_argument("--out", default="vis_output", help="输出目录（已创建也没关系）")
    parser.add_argument(
        "--token",
        default="",
        help="只渲染单个 token（用于快速测试）；提供后将忽略 top100/p1-p5 选样逻辑",
    )

    parser.add_argument("--split", default="test", choices=["mini", "test", "trainval"], help="NAVSIM split")
    parser.add_argument("--scene_filter", default="navtest", help="scene_filter 配置名（Hydra）")
    parser.add_argument(
        "--openscene_data_root", default=os.getenv("OPENSCENE_DATA_ROOT", ""), help="OPENSCENE_DATA_ROOT"
    )

    parser.add_argument("--checkpoint", required=True, help="模型 checkpoint 路径（用于重跑 agent 轨迹）")
    parser.add_argument("--vlm_path", default="", help="VLM 权重/路径（如需要）")

    parser.add_argument("--cam_type", default="single")
    parser.add_argument("--vlm_type", default="internvl")
    parser.add_argument("--dit_type", default="small")
    parser.add_argument("--sampling_method", default="ddim")
    parser.add_argument("--cache_mode", action="store_true", help="开启特征侧 backbone（需要 --vlm_path）")
    parser.add_argument("--cache_hidden_state", action="store_true", help="特征中包含 hidden state（通常和 --cache_mode 配合）")
    parser.add_argument("--vlm_size", default="small")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--normalize_vl_hidden_states", action="store_true")
    parser.add_argument("--first_hidden_state_index", type=int, default=0)
    parser.add_argument("--last_hidden_state_index", type=int, default=-1)

    parser.add_argument("--time_horizon", type=float, default=4.0)
    parser.add_argument("--interval_length", type=float, default=0.5)

    parser.add_argument("--max_p1_p5", type=int, default=-1, help="限制 p1-p5 导出数量；-1 表示不限制")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if not args.openscene_data_root:
        raise ValueError("需要设置 --openscene_data_root 或环境变量 OPENSCENE_DATA_ROOT")

    # cache / no-cache 模式下对 vlm_path 的基本约束，避免只打印 warning 却一直不出图
    if (args.cache_mode or args.cache_hidden_state) and not args.vlm_path:
        raise ValueError("你开启了 --cache_mode/--cache_hidden_state，但没有提供 --vlm_path（VLM checkpoint 路径）。")
    if (not args.cache_mode and not args.cache_hidden_state) and not args.vlm_path:
        raise ValueError("当前脚本需要 --vlm_path 来计算 hidden state（否则 agent 无法生成轨迹，无法出图）。")

    out_root = Path(args.out)
    _ensure_dir(out_root)

    df = pd.read_csv(args.csv)
    _validate_csv_columns(df)

    # 用绝对路径初始化，避免从不同 cwd 运行时找不到配置目录
    config_dir = (Path(__file__).resolve().parent / "config" / "common" / "train_test_split" / "scene_filter").as_posix()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.scene_filter)
    scene_filter: SceneFilter = instantiate(cfg)

    agent = build_agent(args, device)
    sensor_cfg = agent.module.get_sensor_config() if hasattr(agent, "module") else agent.get_sensor_config()
    scene_loader, scene_loader_traj = build_scene_loaders(
        openscene_data_root=Path(args.openscene_data_root),
        split=args.split,
        scene_filter=scene_filter,
        sensor_config=sensor_cfg,
    )

    if args.token:
        tokens = [str(args.token)]
        saved_single = save_vis_for_tokens(
            tokens=tokens,
            df=df,
            out_dir=out_root / "single",
            scene_loader=scene_loader,
            scene_loader_traj=scene_loader_traj,
            agent=agent,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            print(f"完成: single 保存 {saved_single} 张")
        return

    top100_tokens, p1_p5_tokens, q01, q05 = select_top100_and_p1_p5(df)
    print(f"p1(score)={q01:.6f}, p5(score)={q05:.6f}")
    print(f"Top100 tokens: {len(top100_tokens)}")
    print(f"p1-p5 tokens: {len(p1_p5_tokens)}")

    max_p1_p5 = None if args.max_p1_p5 < 0 else int(args.max_p1_p5)

    saved_top100 = save_vis_for_tokens(
        tokens=top100_tokens,
        df=df,
        out_dir=out_root / "top100",
        scene_loader=scene_loader,
        scene_loader_traj=scene_loader_traj,
        agent=agent,
        rank=rank,
        world_size=world_size,
    )
    saved_p1_p5 = save_vis_for_tokens(
        tokens=p1_p5_tokens,
        df=df,
        out_dir=out_root / "p1_p5",
        scene_loader=scene_loader,
        scene_loader_traj=scene_loader_traj,
        agent=agent,
        rank=rank,
        world_size=world_size,
        max_items=max_p1_p5,
    )

    if rank == 0:
        print(f"完成: top100 保存 {saved_top100} 张；p1_p5 保存 {saved_p1_p5} 张")


if __name__ == "__main__":
    main()
