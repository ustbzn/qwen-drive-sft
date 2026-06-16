#!/usr/bin/env python3
"""
单 scenario 轨迹可视化脚本。
用法：
  export OPENSCENE_DATA_ROOT=/MOGO_VEPFS/D2/NXT/lst/vla/navsim/download
  export NAVSIM_EXP_ROOT=/MOGO_VEPFS/D2/NXT/lst/vla/navsim/exp
  python scripts/vis_one_scenario.py 00016f8b45c25a1d
  python scripts/vis_one_scenario.py token1 token2 token3
  python scripts/vis_one_scenario.py --model rl --output-dir vis_output/il_rl_compare token1 token2

可选：使用 --gpu N 指定其他空闲 GPU；--model il|rl 选择 checkpoint。
"""
import os
import sys
from pathlib import Path

# 确保项目根在 path 中
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

os.environ.setdefault("OPENSCENE_DATA_ROOT", "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/dataset")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/0312_0319/exp")
os.environ.setdefault("NUPLAN_MAPS_ROOT", "/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

import argparse
import hydra
from hydra.utils import instantiate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig
from navsim.planning.scenario_builder.mogo_sqlite_map_adapter import (
    MogoRouteContext,
    MogoSQLiteMapAdapter,
    MogoSQLiteMapConfig,
)
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

CHECKPOINTS = {
    "il": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/exp/training_recogdrive_qwen3_agent_IL/2026.06.11.13.28.50/lightning_logs/version_0/checkpoints/epoch=188-step=13986.ckpt",
    "rl": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/recog_drive/exp/training_recogdrive_qwen3_agent_RL/2026.06.12.03.28.29/lightning_logs/version_0/checkpoints/epoch=19-step=5000.ckpt",
}
VLM_PATH = "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct"
OUTPUT_DIR = "vis_output"
DATA_SPLIT = "val"


def _default_mogo_map_sqlite_path(openscene_root: Path) -> Path:
    maps_dir = openscene_root / "maps"
    candidates = sorted(maps_dir.glob("*.sqlite"))
    if not candidates:
        raise FileNotFoundError(f"No MOGO sqlite map found under {maps_dir}")
    return candidates[0]


def _build_mogo_map_api_builder(sqlite_path: Path, num_history_frames: int):
    map_config = MogoSQLiteMapConfig(sqlite_path=sqlite_path, map_name="mogo")

    def _builder(scene_dict_list):
        current_frame_idx = num_history_frames - 1
        route_roadblock_ids = list(scene_dict_list[current_frame_idx]["roadblock_ids"])
        route_context = MogoRouteContext(roadblock_ids=route_roadblock_ids)
        return MogoSQLiteMapAdapter(config=map_config, route_context=route_context)

    return _builder


def main():
    parser = argparse.ArgumentParser(description="单 scenario 轨迹可视化")
    parser.add_argument("token", nargs="+", help="scenario token，可传多个")
    parser.add_argument("--model", choices=["il", "rl"], default="il", help="使用 IL 或 RL checkpoint")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--gpu", type=int, default=None, help="指定 GPU 编号，如 1")
    parser.add_argument("--mogo-map-sqlite-path", default="", help="MOGO sqlite 地图路径；默认使用 OPENSCENE_DATA_ROOT/maps 下第一个 .sqlite")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"使用 GPU {args.gpu}")

    tokens = args.token
    checkpoint = CHECKPOINTS[args.model]
    output_dir = Path(args.output_dir)
    openscene_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    navsim_logs = openscene_root / "navsim_logs" / DATA_SPLIT
    sensor_blobs = openscene_root / "sensor_blobs" / DATA_SPLIT
    mogo_map_sqlite_path = Path(args.mogo_map_sqlite_path) if args.mogo_map_sqlite_path else _default_mogo_map_sqlite_path(openscene_root)

    # scene filter（Hydra 相对路径相对于运行时的 cwd，即项目根）
    hydra.initialize(config_path="../navsim/planning/script/config/common/train_test_split", version_base=None)
    cfg = hydra.compose(config_name="mogo_trainval")
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    scene_filter.tokens = tokens

    # agent
    agent = ReCogDriveAgent(
        TrajectorySampling(time_horizon=4, interval_length=0.5),
        checkpoint_path=checkpoint,
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

    sensor_config = SensorConfig.build_all_sensors(include=False)
    sensor_config.cam_f0 = True
    map_api_builder = _build_mogo_map_api_builder(mogo_map_sqlite_path, scene_filter.num_history_frames)

    # loaders
    scene_loader = SceneLoader(
        data_path=navsim_logs,
        sensor_blobs_path=sensor_blobs,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        map_api_builder=map_api_builder,
    )
    scene_loader_traj = SceneLoader(
        data_path=navsim_logs,
        sensor_blobs_path=sensor_blobs,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
        map_api_builder=map_api_builder,
    )

    for token in tokens:
        if token not in scene_loader.tokens:
            print(f"Token {token} 不在 scene_loader 中，跳过")
            continue

        scene = scene_loader.get_scene_from_token(token)
        scene_traj = scene_loader_traj.get_scene_from_token(token)
        frame_idx = scene.scene_metadata.num_history_frames - 1

        fig, _, _ = plot_bev_and_camera_with_agent(scene, scene_traj, frame_idx, agent)
        os.makedirs(output_dir, exist_ok=True)
        out_path = output_dir / f"{token}_{args.model}_vis_traj.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"已保存: {out_path.absolute()}")


if __name__ == "__main__":
    main()
