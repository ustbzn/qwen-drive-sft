import copy
import lzma
import os
import argparse
import sys
import math
import pickle
from datetime import timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
from hydra import initialize_config_dir
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

import torch
import torch.distributed as dist
from tqdm import tqdm
import csv
import traceback

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from typing import Callable

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from shapely.geometry import Polygon

# 确保从任意工作目录运行都能 import `navsim`
_REPO_ROOT = Path(__file__).resolve().parents[3]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, EgoAreaIndex, StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.visualization.plots import plot_bev_and_camera_with_agent
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.bev import add_trajectory_to_bev_ax
from navsim.visualization.camera import add_trajectory_to_camera_ax
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent

from navsim.evaluate.pdm_score import get_trajectory_as_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.scenario_builder.mogo_sqlite_map_adapter import (
    MogoRouteContext,
    MogoSQLiteMapConfig,
    MogoSQLiteMapAdapter,
)

def init_distributed():
    """可选分布式初始化：支持 torchrun 多卡分摊渲染任务。"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        # 渲染耗时长且各 rank 结束时间差大，收尾 barrier 用大超时避免误杀。
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )
        print(f"Distributed init: rank {rank}/{world_size}, local_rank {local_rank}")
        return rank, world_size, local_rank
    return 0, 1, 0


def merge_rank_indices(out_dir: Path, world_size: int) -> Path | None:
    """多卡收尾：把各 rank 的 index_rank{rank}.csv 合并成总表 index_all.csv。

    单进程（world_size==1）下各 rank 索引即全量，无需合并，返回 None。
    """
    if world_size <= 1:
        return None

    rank_csvs = sorted(out_dir.glob("index_rank*.csv"))
    if not rank_csvs:
        print(f"[WARN] {out_dir} 下未找到 index_rank*.csv，跳过合并")
        return None

    frames: list[pd.DataFrame] = []
    for p in rank_csvs:
        try:
            frames.append(pd.read_csv(p, dtype={"token": "string"}))
        except Exception as e:
            print(f"[WARN] 读取 {p} 失败，跳过: {type(e).__name__}: {e}")

    if not frames:
        print(f"[WARN] {out_dir} 下 index_rank*.csv 均无法读取，跳过合并")
        return None

    merged = pd.concat(frames, ignore_index=True)
    if "token" in merged.columns:
        merged = merged.drop_duplicates(subset=["token"], keep="first")
    if "score" in merged.columns:
        merged = merged.sort_values("score", ascending=True, kind="stable")

    merged_path = out_dir / "index_all.csv"
    merged.to_csv(merged_path, index=False)
    print(f"合并 {len(rank_csvs)} 个 rank 索引 -> {merged_path}（共 {len(merged)} 行）")
    return merged_path


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


def _metric_as_float(value: object) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if math.isfinite(metric) else None


def load_saved_trajectory(path_value: object) -> Trajectory | None:
    if path_value is None:
        return None

    path_text = str(path_value).strip()
    if not path_text or path_text.lower() in {"nan", "none", "<na>"}:
        return None

    trajectory_path = Path(path_text)
    if not trajectory_path.is_file():
        return None

    with np.load(trajectory_path) as data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        interval_length = float(np.asarray(data["interval_length"]).item())

    return Trajectory(
        poses=poses,
        trajectory_sampling=TrajectorySampling(
            num_poses=int(poses.shape[0]),
            interval_length=interval_length,
        ),
    )


PRIMARY_METRIC_FIELDS = [
    ("no_at_fault_collisions", "NC"),
    ("drivable_area_compliance", "DAC"),
    ("no_reverse_driving", "NRD"),
    ("traffic_light_compliance", "TLC"),
]

OTHER_METRIC_FIELDS = [
    ("ego_progress", "EP"),
    ("time_to_collision_within_bound", "TTC"),
    ("comfort", "C"),
    ("driving_direction_compliance", "DDC"),
    ("obstacle_clearance", "OC"),
    ("centerline_convergence", "CC"),
    ("start_anchor_comfort", "SAC"),
]

TRAFFIC_LIGHT_STATE_COLORS = {
    "red": "#e53935",
    "yellow": "#fdd835",
    "green": "#43a047",
    "green_flash": "#76ff03",
    "off": "#9e9e9e",
    "unknown": "#29b6f6",
}


def _build_overlay_text(token: str, metrics: dict[str, object], debug_notes: list[str] | None = None) -> str:
    """相机图左上角分行显示：token/score → NC/DAC → 其余指标 → 失败诊断。"""
    lines: list[str] = []

    # 第一行：token + 总 score
    lines.append(f"token: {token} | score: {_fmt_metric(metrics.get('score'))}")

    # 第二行：乘性指标（NC / DAC / NRD）
    multi_metrics = " | ".join(f"{short}: {_fmt_metric(metrics.get(key))}" for key, short in PRIMARY_METRIC_FIELDS)
    lines.append(multi_metrics)

    # 第三行：其余指标
    other = " | ".join(f"{short}: {_fmt_metric(metrics.get(key))}" for key, short in OTHER_METRIC_FIELDS)
    lines.append(other)

    # 诊断笔记：失败 debug + 红绿灯摘要
    if debug_notes:
        lines.append(" | ".join(debug_notes))

    return "\n".join(lines)


DEBUG_SIM_TRAJ_CONFIG = {
    "line_color": "#FFA500",
    "line_color_alpha": 0.9,
    "line_width": 2.5,
    "line_style": "--",
    "zorder": 6,
}
DEBUG_EGO_EVENT_BOX_CONFIG = {
    "fill_color": "#FFD700",
    "fill_color_alpha": 0.35,
    "line_color": "#FFD700",
    "line_color_alpha": 1.0,
    "line_width": 2.5,
    "line_style": "-",
    "zorder": 9,
}
DEBUG_OBJ_EVENT_BOX_CONFIG = {
    "fill_color": "#FF00FF",
    "fill_color_alpha": 0.30,
    "line_color": "#FF00FF",
    "line_color_alpha": 1.0,
    "line_width": 2.5,
    "line_style": "-",
    "zorder": 8,
}
DEBUG_OFFROAD_EGO_BOX_CONFIG = {
    "fill_color": "#FF4500",
    "fill_color_alpha": 0.35,
    "line_color": "#FF4500",
    "line_color_alpha": 1.0,
    "line_width": 2.5,
    "line_style": "-",
    "zorder": 9,
}
DEBUG_REVERSE_EGO_BOX_CONFIG = {
    "fill_color": "#00CED1",
    "fill_color_alpha": 0.35,
    "line_color": "#00CED1",
    "line_color_alpha": 1.0,
    "line_width": 4,
    "line_style": "-",
    "zorder": 9,
}


@dataclass
class PDMFailureDiagnostics:
    debug_notes: list[str] = field(default_factory=list)
    simulated_xy_local: np.ndarray | None = None
    collision_time_s: float | None = None
    collision_type: str | None = None
    collision_object_type: str | None = None
    collision_object_token: str | None = None
    collision_ego_polygon_global: Polygon | None = None
    collision_object_polygon_global: Polygon | None = None
    offroad_time_s: float | None = None
    offroad_ego_polygon_global: Polygon | None = None
    reverse_driving_time_s: float | None = None
    reverse_driving_ego_polygon_global: Polygon | None = None



def build_pdm_simulator_scorer() -> tuple[PDMSimulator, PDMScorer]:
    config_dir = (Path(__file__).resolve().parent / "config" / "pdm_scoring").as_posix()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="default_run_pdm_score")
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    return simulator, scorer

_METRIC_PERFECT = 1.0
_METRIC_EPS = 1e-6
_DAC_HARD_SCORE = 0.05  # 与 pdm_scorer drivable_area_hard_score 一致


def _metric_float(value: object) -> float | None:
    try:
        f = float(value)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return None


def _metric_is_nc_failed(value: object) -> bool:
    """NC 低于 1 表示发生了 at-fault 碰撞（agent→0.05，非 agent→0.5）。"""
    f = _metric_float(value)
    return f is not None and f < _METRIC_PERFECT - _METRIC_EPS


def _metric_is_dac_hard_failed(value: object) -> bool:
    """DAC 等于 hard 地板分 0.05，表示 ego footprint 离开 drivable area。"""
    f = _metric_float(value)
    return f is not None and f <= _DAC_HARD_SCORE + _METRIC_EPS


def _metric_is_nrd_failed(value: object) -> bool:
    """NRD 低于 1 表示检测到倒车（score floor = 0.05）。"""
    f = _metric_float(value)
    return f is not None and f < _METRIC_PERFECT - _METRIC_EPS


# Aligns with PDMScorerConfig.at_fault_collision_score_floor / drivable_area_hard_score.
HARD_MULTI_METRIC_SCORE_FLOOR = 0.05


def _metric_is_hard_collision(value: object) -> bool:
    metric = _metric_as_float(value)
    return metric is not None and metric <= HARD_MULTI_METRIC_SCORE_FLOOR + 1e-9


def _metric_is_hard_offroad(value: object) -> bool:
    metric = _metric_as_float(value)
    return metric is not None and metric <= HARD_MULTI_METRIC_SCORE_FLOOR + 1e-9


def _should_debug_failures(metrics: dict[str, object], debug_failures: bool) -> bool:
    if not debug_failures:
        return False
    nc = metrics.get("no_at_fault_collisions")
    dac = metrics.get("drivable_area_compliance")
    return (
        _metric_is_nc_failed(metrics.get("no_at_fault_collisions"))
        or _metric_is_dac_hard_failed(metrics.get("drivable_area_compliance"))
        or _metric_is_nrd_failed(metrics.get("no_reverse_driving"))
    )


def _global_polygon_to_ego_local(polygon: Polygon, origin: StateSE2) -> np.ndarray:
    x, y = polygon.exterior.xy
    global_pts = np.column_stack([x, y])
    se2 = np.column_stack([global_pts, np.zeros(len(global_pts), dtype=np.float64)])
    local = convert_absolute_to_relative_se2_array(origin, se2)
    return local[:, :2]


def _plot_local_polygon_on_bev(ax: plt.Axes, local_xy: np.ndarray, config: dict[str, Any]) -> None:
    closed = np.vstack([local_xy, local_xy[:1]])
    ax.fill(
        closed[:, 1],
        closed[:, 0],
        color=config["fill_color"],
        alpha=config["fill_color_alpha"],
        zorder=config["zorder"],
    )
    ax.plot(
        closed[:, 1],
        closed[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
    )



def _plot_local_xy_path_on_bev(ax: plt.Axes, local_xy: np.ndarray, config: dict[str, Any]) -> None:
    if local_xy is None or len(local_xy) == 0:
        return
    ax.plot(
        local_xy[:, 1],
        local_xy[:, 0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        zorder=config["zorder"],
        label="Simulated (PDM)",
    )


def _normalize_traffic_light_state_label(raw_state: object) -> str:
    state = str(raw_state or "unknown").strip().lower()
    aliases = {
        "g": "green",
        "r": "red",
        "y": "yellow",
        "greenflash": "green_flash",
        "green_flashing": "green_flash",
        "flash_green": "green_flash",
    }
    return aliases.get(state, state or "unknown")


def _traffic_light_states_from_sources(
    *,
    scene_frame_traffic_lights: object,
    metric_cache: MetricCache | None,
) -> list[dict[str, Any]]:
    """Prefer metric-cache lights (same as scoring); fall back to scene frame."""
    for source in (
        getattr(metric_cache, "traffic_light_states", None) if metric_cache is not None else None,
        scene_frame_traffic_lights,
    ):
        if not source:
            continue
        lights = [item for item in source if isinstance(item, dict)]
        if lights:
            return lights
    return []


def add_traffic_lights_to_bev_ax(
    ax: plt.Axes,
    origin: StateSE2,
    traffic_light_states: list[dict[str, Any]],
) -> list[str]:
    """Draw stop-line segments + state labels in ego-local BEV. Returns text notes."""
    notes: list[str] = []
    for idx, light in enumerate(traffic_light_states):
        position = light.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        try:
            global_xy = np.asarray(position, dtype=np.float64)[:, :2]
        except Exception:
            continue
        if global_xy.shape[0] < 2 or not np.isfinite(global_xy).all():
            continue

        se2 = np.column_stack([global_xy, np.zeros(len(global_xy), dtype=np.float64)])
        local_xy = convert_absolute_to_relative_se2_array(origin, se2)[:, :2]
        state = _normalize_traffic_light_state_label(light.get("state"))
        color = TRAFFIC_LIGHT_STATE_COLORS.get(state, TRAFFIC_LIGHT_STATE_COLORS["unknown"])
        direction = str(light.get("direction", "?")).strip() or "?"
        stop_line_id = light.get("stop_line_id", "?")

        ax.plot(
            local_xy[:, 1],
            local_xy[:, 0],
            color=color,
            linewidth=4.0,
            linestyle="-",
            alpha=0.95,
            zorder=6,
        )
        mid = local_xy.mean(axis=0)
        label = f"{direction}:{state}"
        ax.scatter([mid[1]], [mid[0]], c=color, s=36, zorder=7, edgecolors="black", linewidths=0.4)
        ax.text(
            mid[1],
            mid[0],
            f" {label}",
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=8,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1.5),
        )
        notes.append(f"TL{idx}: id={stop_line_id} {direction}/{state}")
    return notes


def _find_first_at_fault_collision(scorer: PDMScorer, pred_idx: int) -> dict[str, Any] | None:
    proposal_collided_track_ids = copy.deepcopy(scorer._observation.collided_track_ids)
    observation = scorer._observation
    interval = scorer.proposal_sampling.interval_length

    for time_idx in range(scorer.proposal_sampling.num_poses + 1):
        ego_polygon = scorer._ego_polygons[pred_idx, time_idx]
        intersecting = observation[time_idx].query(
            np.asarray([ego_polygon], dtype=object),
            predicate="intersects",
        )
        if len(intersecting) == 0:
            continue

        tree_indices = intersecting[1] if len(intersecting) > 1 else intersecting[0]
        if len(tree_indices) == 0:
            continue

        for geometry_idx in np.atleast_1d(tree_indices):
            token = observation[time_idx].tokens[int(geometry_idx)]
            if observation.red_light_token in token or token in proposal_collided_track_ids:
                continue

            ego_in_bad_area = (
                scorer._ego_areas[pred_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                or scorer._ego_areas[pred_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
            )
            tracked_object = observation.unique_objects[token]
            collision_type = get_collision_type(
                scorer._states[pred_idx, time_idx],
                ego_polygon,
                tracked_object,
                observation[time_idx][token],
            )
            at_fault = collision_type in [
                CollisionType.ACTIVE_FRONT_COLLISION,
                CollisionType.STOPPED_TRACK_COLLISION,
            ] or (ego_in_bad_area and collision_type == CollisionType.ACTIVE_LATERAL_COLLISION)

            if at_fault:
                return {
                    "time_idx": time_idx,
                    "time_s": time_idx * interval,
                    "collision_type": collision_type.name,
                    "object_type": str(tracked_object.tracked_object_type),
                    "object_token": token,
                    "ego_polygon": ego_polygon,
                    "object_polygon": observation[time_idx][token],
                }
            proposal_collided_track_ids.append(token)
    return None


def _find_first_hard_offroad(scorer: PDMScorer, pred_idx: int) -> dict[str, Any] | None:
    mask = scorer._ego_areas[pred_idx, :, EgoAreaIndex.NON_DRIVABLE_AREA]
    if not mask.any():
        return None
    time_idx = int(np.where(mask)[0][0])
    return {
        "time_idx": time_idx,
        "time_s": time_idx * scorer.proposal_sampling.interval_length,
        "ego_polygon": scorer._ego_polygons[pred_idx, time_idx],
    }
    
def _find_first_reverse_driving(scorer: PDMScorer, pred_idx: int) -> dict[str, Any] | None:
    from shapely import Point

    center_coordinates = scorer._ego_coords[pred_idx, :, BBCoordsIndex.CENTER]
    points = [Point(*xy) for xy in center_coordinates]
    centerline_progress = np.asarray(scorer._centerline.project(points), dtype=np.float64)
    progress_delta = centerline_progress[1:] - centerline_progress[:-1]
    min_delta = getattr(scorer._config, "reverse_driving_min_delta", 0.0)
    interval = scorer.proposal_sampling.interval_length

    for step_idx, delta in enumerate(progress_delta):
        if delta < -min_delta:
            time_idx = step_idx + 1
            return {
                "time_idx": time_idx,
                "time_s": time_idx * interval,
                "centerline_progress_delta_m": float(delta),
                "ego_polygon": scorer._ego_polygons[pred_idx, time_idx],
                "mode": "centerline",
            }
    return None


def run_pdm_failure_diagnostics(
    *,
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    metrics: dict[str, object],
) -> PDMFailureDiagnostics:
    pred_idx = 1
    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    pdm_states = get_trajectory_as_array(
        pdm_trajectory,
        simulator.proposal_sampling,
        initial_ego_state.time_point,
    )
    pred_states = get_trajectory_as_array(
        pred_trajectory,
        simulator.proposal_sampling,
        initial_ego_state.time_point,
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    history_states = getattr(metric_cache, "history_states", None)
    if history_states is not None and history_states.ndim == 2:
        history_states = np.repeat(history_states[None, ...], len(trajectory_states), axis=0)

    simulated_states, simulated_reference_signals = simulator.simulate_proposals(
        trajectory_states,
        initial_ego_state,
    )
    scorer.score_proposals(
        simulated_states,
        simulated_reference_signals,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        history_states=history_states,
    )

    origin = StateSE2(*initial_ego_state.rear_axle.serialize())
    global_xy = simulated_states[pred_idx, :, StateIndex.POINT]
    global_se2 = np.column_stack([global_xy, simulated_states[pred_idx, :, StateIndex.HEADING]])
    simulated_xy_local = convert_absolute_to_relative_se2_array(origin, global_se2)[:, :2]

    diagnostics = PDMFailureDiagnostics(simulated_xy_local=simulated_xy_local)

    if _metric_is_hard_collision(metrics.get("no_at_fault_collisions")):
        collision = _find_first_at_fault_collision(scorer, pred_idx)
        if collision is not None:
            diagnostics.collision_time_s = collision["time_s"]
            diagnostics.collision_type = collision["collision_type"]
            diagnostics.collision_object_type = collision["object_type"]
            diagnostics.collision_object_token = collision["object_token"]
            diagnostics.collision_ego_polygon_global = collision["ego_polygon"]
            diagnostics.collision_object_polygon_global = collision["object_polygon"]
            diagnostics.debug_notes.append(
                f"NC fail @ {collision['time_s']:.1f}s | {collision['collision_type']} | "
                f"{collision['object_type']} | obj={collision['object_token'][:8]}"
            )

    if _metric_is_hard_offroad(metrics.get("drivable_area_compliance")):
        offroad = _find_first_hard_offroad(scorer, pred_idx)
        if offroad is not None:
            diagnostics.offroad_time_s = offroad["time_s"]
            diagnostics.offroad_ego_polygon_global = offroad["ego_polygon"]
            diagnostics.debug_notes.append(f"DAC hard off-road @ {offroad['time_s']:.1f}s")
    
    if _metric_is_nrd_failed(metrics.get("no_reverse_driving")):
        reverse = _find_first_reverse_driving(scorer, pred_idx)
        if reverse is not None:
            diagnostics.reverse_driving_time_s = reverse["time_s"]
            diagnostics.reverse_driving_ego_polygon_global = reverse["ego_polygon"]
            diagnostics.debug_notes.append(
                f"NRD reverse @ {reverse['time_s']:.1f}s | "
                f"ds={reverse['centerline_progress_delta_m']:.3f}m"
            )
        else:
            diagnostics.debug_notes.append(
                f"NRD failed ({_fmt_metric(metrics.get('no_reverse_driving'))}), "
                "reverse event not found in re-sim"
            )

    return diagnostics


def add_failure_debug_overlays(
    bev_ax: plt.Axes,
    origin: StateSE2,
    diagnostics: PDMFailureDiagnostics,
) -> None:
    _plot_local_xy_path_on_bev(bev_ax, diagnostics.simulated_xy_local, DEBUG_SIM_TRAJ_CONFIG)

    if diagnostics.collision_ego_polygon_global is not None:
        ego_local = _global_polygon_to_ego_local(diagnostics.collision_ego_polygon_global, origin)
        _plot_local_polygon_on_bev(bev_ax, ego_local, DEBUG_EGO_EVENT_BOX_CONFIG)
        center = ego_local[:-1].mean(axis=0)
        bev_ax.scatter(
            [center[1]],
            [center[0]],
            s=80,
            c=DEBUG_EGO_EVENT_BOX_CONFIG["line_color"],
            marker="x",
            zorder=10,
        )

    if diagnostics.collision_object_polygon_global is not None:
        obj_local = _global_polygon_to_ego_local(diagnostics.collision_object_polygon_global, origin)
        _plot_local_polygon_on_bev(bev_ax, obj_local, DEBUG_OBJ_EVENT_BOX_CONFIG)

    if diagnostics.offroad_ego_polygon_global is not None:
        offroad_local = _global_polygon_to_ego_local(diagnostics.offroad_ego_polygon_global, origin)
        _plot_local_polygon_on_bev(bev_ax, offroad_local, DEBUG_OFFROAD_EGO_BOX_CONFIG)
        center = offroad_local[:-1].mean(axis=0)
        bev_ax.scatter(
            [center[1]],
            [center[0]],
            s=80,
            c=DEBUG_OFFROAD_EGO_BOX_CONFIG["line_color"],
            marker="x",
            zorder=10,
        )
    
    if diagnostics.reverse_driving_ego_polygon_global is not None:
        reverse_local = _global_polygon_to_ego_local(diagnostics.reverse_driving_ego_polygon_global, origin)
        _plot_local_polygon_on_bev(bev_ax, reverse_local, DEBUG_REVERSE_EGO_BOX_CONFIG)
        center = reverse_local[:-1].mean(axis=0)
        bev_ax.scatter(
            [center[1]],
            [center[0]],
            s=80,
            c=DEBUG_REVERSE_EGO_BOX_CONFIG["line_color"],
            marker="x",
            zorder=10,
        )



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
        vl_hidden_dim=args.vl_hidden_dim,
        normalize_vl_hidden_states=bool(args.normalize_vl_hidden_states),
        first_hidden_state_index=int(args.first_hidden_state_index),
        last_hidden_state_index=int(args.last_hidden_state_index),
        # --- 模型架构开关（须与 checkpoint 训练时一致，否则被 strict=False 丢成随机初始化）---
        use_vlm=bool(args.use_vlm),
        vlm_ablation=str(args.vlm_ablation),
        ego_status_encoder_type=str(args.ego_status_encoder_type),
        use_object_vector=bool(args.use_object_vector),
        object_encoder_type=str(args.object_encoder_type),
        max_obj_tokens=int(args.max_obj_tokens),
        include_static_objects=bool(args.include_static_objects),
        object_token_dropout=float(args.object_token_dropout),
        static_speed_threshold=float(args.static_speed_threshold),
        use_hdmap_vector=bool(args.use_hdmap_vector),
        use_lane_boundaries=bool(args.use_lane_boundaries),
        strict_hdmap_vector=bool(args.strict_hdmap_vector),
        max_map_tokens=int(args.max_map_tokens),
        map_sample_points=int(args.map_sample_points),
        hdmap_radius=float(args.hdmap_radius),
        use_image_encoder=bool(args.use_image_encoder),
        image_encoder_arch=str(args.image_encoder_arch),
        image_view=str(args.image_view),
    ).to(device)
    agent.initialize()
    return agent


def build_scene_loaders(
    *,
    openscene_data_root: Path,
    split: str,
    scene_filter: SceneFilter,
    sensor_config: SensorConfig,
    map_api_builder: Callable | None = None,
) -> tuple[SceneLoader, SceneLoader]:
    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{split}",
        openscene_data_root / f"sensor_blobs/{split}",
        scene_filter,
        sensor_config=sensor_config,
        map_api_builder=map_api_builder,
    )
    scene_loader_traj = SceneLoader(
        openscene_data_root / f"navsim_logs/{split}",
        openscene_data_root / f"sensor_blobs/{split}",
        scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
        map_api_builder=map_api_builder,
    )
    return scene_loader, scene_loader_traj


def build_mogo_map_api_builder(args) -> Callable | None:
    if not args.mogo_map_sqlite_path:
        return None

    map_config = MogoSQLiteMapConfig(
        sqlite_path=Path(args.mogo_map_sqlite_path),
        map_name=str(args.mogo_map_name),
        query_radius_m=float(args.mogo_map_query_radius_m),
        max_segment_length_m=float(args.mogo_map_max_segment_length_m),
        enable_route_edge_filter=bool(args.mogo_map_enable_route_edge_filter),
        geometry_srid=int(args.mogo_map_geometry_srid),
        utm_zone=int(args.mogo_map_utm_zone),
        utm_south=bool(args.mogo_map_utm_south),
    )

    def _builder(scene_dict_list):
        frame_idx = int(args.num_history_frames) - 1
        route_roadblock_ids = list(scene_dict_list[frame_idx]["roadblock_ids"])
        route_context = MogoRouteContext(roadblock_ids=route_roadblock_ids)
        return MogoSQLiteMapAdapter(config=map_config, route_context=route_context)

    return _builder


def tokens_from_csv(
    csv_path: str,
    *,
    min_score: float | None = None,
    max_score: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    # 强制 token 为字符串，保留前导 0
    df = pd.read_csv(csv_path, dtype={"token": "string"})
    if "token" not in df.columns:
        raise ValueError(f"CSV 缺少 token 列: {csv_path}")

    df["token"] = df["token"].astype(str).str.strip()
    df = df[df["token"].notna() & (df["token"] != "") & (df["token"].str.lower() != "average")]

    if "score" in df.columns and (min_score is not None or max_score is not None):
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        if min_score is not None:
            df = df[df["score"].notna() & (df["score"] >= min_score)]
        if max_score is not None:
            df = df[df["score"].notna() & (df["score"] < max_score)]

    tokens = df["token"].tolist()
    # 去重但保持顺序
    seen: set[str] = set()
    uniq: list[str] = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return df, uniq


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    uniq: list[str] = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def select_tokens(df: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    """按 CLI 筛选要渲染的 token 列表。"""
    if args.tokens_csv:
        sub_df, tokens = tokens_from_csv(str(args.tokens_csv))
        if "score" in df.columns and "score" in sub_df.columns:
            return tokens
        token_set = set(tokens)
        return [t for t in tokens if t in set(df["token"].astype(str))]

    work = df.copy()
    if "score" in work.columns:
        work["score"] = pd.to_numeric(work["score"], errors="coerce")
        work = work.dropna(subset=["score", "token"])
    work["token"] = work["token"].astype(str)

    if args.selection == "top100":
        if "score" not in work.columns:
            raise ValueError("--selection top100 需要 CSV 含 score 列")
        return work.sort_values("score", ascending=True).head(100)["token"].tolist()

    if args.selection == "p1_p5":
        if "score" not in work.columns:
            raise ValueError("--selection p1_p5 需要 CSV 含 score 列")
        q_low = float(work["score"].quantile(float(args.percentile_low)))
        q_high = float(work["score"].quantile(float(args.percentile_high)))
        mask = (work["score"] >= q_low) & (work["score"] <= q_high)
        print(f"p{args.percentile_low:.0%}-p{args.percentile_high:.0%}: score in [{q_low:.6f}, {q_high:.6f}]")
        return work.loc[mask, "token"].tolist()

    needs_score = (
        args.score_max is not None
        or args.score_min is not None
        or args.score_zero
        or args.top_k_lowest is not None
    )
    if needs_score and "score" not in work.columns:
        raise ValueError("score 区间/零分/top_k 筛选需要 CSV 含 score 列")

    mask = pd.Series(True, index=work.index)
    if args.score_max is not None:
        mask &= work["score"] <= float(args.score_max)
    if args.score_min is not None:
        mask &= work["score"] >= float(args.score_min)
    if args.score_zero:
        mask &= work["score"].abs() <= 1e-9
    filtered = work.loc[mask]

    if args.top_k_lowest is not None and int(args.top_k_lowest) > 0:
        filtered = filtered.sort_values("score", ascending=True).head(int(args.top_k_lowest))

    if not needs_score and not args.selection:
        return _dedupe_tokens(work["token"].tolist())
    return _dedupe_tokens(filtered["token"].tolist())


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
    debug_failures: bool = False,
    metric_cache_loader: MetricCacheLoader | None = None,
    simulator: PDMSimulator | None = None,
    scorer: PDMScorer | None = None,
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

            model_trajectory = load_saved_trajectory(metrics.get("trajectory_path"))
            if model_trajectory is None:
                # 传入 scene_traj，使 obj/hdmap builder 能走 compute_features_from_scene；
                # 对 VLM-only agent（无 scene-aware builder）该参数被忽略，行为不变。
                model_trajectory = agent.compute_trajectory_vis(
                    scene_traj.get_agent_input(), scene=scene_traj
                )

            metric_cache: MetricCache | None = None
            if metric_cache_loader is not None and token in metric_cache_loader.tokens:
                try:
                    with lzma.open(metric_cache_loader.metric_cache_paths[token], "rb") as f:
                        metric_cache = pickle.load(f)
                except Exception as cache_exc:
                    print(f"[WARN] token={token} 读取 metric cache 失败: {cache_exc}")

            diagnostics: PDMFailureDiagnostics | None = None
            debug_notes: list[str] = []
            if _should_debug_failures(metrics, debug_failures):
                if metric_cache is None or simulator is None or scorer is None:
                    print(f"[WARN] token={token} debug_failures 需要 metric_cache/simulator/scorer，已跳过")
                else:
                    try:
                        diagnostics = run_pdm_failure_diagnostics(
                            metric_cache=metric_cache,
                            model_trajectory=model_trajectory,
                            simulator=simulator,
                            scorer=scorer,
                            metrics=metrics,
                        )
                        debug_notes = diagnostics.debug_notes
                    except Exception as debug_exc:
                        print(f"[WARN] token={token} debug_failures 失败: {debug_exc}")

            fig, bev_ax, camera_ax = plot_bev_and_camera_with_agent(
                scene,
                scene_traj,
                frame_idx,
                agent,
                agent_trajectory=model_trajectory,
            )
            origin = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
            if diagnostics is not None:
                add_failure_debug_overlays(bev_ax, origin, diagnostics)

            traffic_light_states = _traffic_light_states_from_sources(
                scene_frame_traffic_lights=getattr(scene.frames[frame_idx], "traffic_light_states", None),
                metric_cache=metric_cache,
            )
            tl_notes = add_traffic_lights_to_bev_ax(bev_ax, origin, traffic_light_states)
            if tl_notes:
                debug_notes = list(debug_notes) + tl_notes

            if camera_ax is not None:
                overlay = _build_overlay_text(token, metrics, debug_notes)
                camera_ax.text(
                    0.01,
                    0.99,
                    overlay,
                    transform=camera_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=13 if debug_notes else 14,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3),
                    zorder=10,
                )

                legend_lines = ["GT: green | Planner: red | TL stop-line: state color"]
                if diagnostics is not None:
                    legend_lines.append(
                        "Sim: orange dash | Collision ego: gold | Collision obj: magenta | Off-road ego: orange-red"
                    )
                camera_ax.text(
                    0.01,
                    0.01,
                    " | ".join(legend_lines),
                    transform=camera_ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=13,
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
            print(f"[WARN] token={token} 渲染失败: {type(e).__name__}: {e}")

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


def resolve_tokens(args) -> tuple[pd.DataFrame, list[str]]:
    """从 CSV、--token 或 --tokens 解析待渲染 token 列表。"""
    if args.tokens:
        tokens = [t.strip() for t in str(args.tokens).split(",") if t.strip()]
        df = pd.DataFrame({"token": tokens})
        return df, tokens

    if args.token:
        token = str(args.token).strip()
        if args.csv:
            df, _ = tokens_from_csv(args.csv)
            if token not in df["token"].astype(str).values:
                print(f"[INFO] --token {token} 不在 CSV 中，仍将渲染该 token")
            df_token = df[df["token"].astype(str) == token]
            if len(df_token) == 0:
                df = pd.concat([df, pd.DataFrame({"token": [token]})], ignore_index=True)
        else:
            df = pd.DataFrame({"token": [token]})
        return df, [token]

    if not args.csv:
        raise ValueError("需要 --csv，或使用 --token / --tokens 指定场景")
    df, tokens = tokens_from_csv(args.csv)
    return df, tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 CSV 或指定 token 渲染 BEV+前视轨迹图（可选叠加 PDM-Closed 轨迹）"
    )
    parser.add_argument(
        "--csv",
        default="",
        help="推理结果 CSV（含 token 列）；与 --token/--tokens 二选一或组合使用",
    )
    parser.add_argument("--out", default="vis_output/eval/0428_163122", help="输出目录")
    parser.add_argument("--token", default="", help="可选：只渲染单个 token（快速自测）")
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="只渲染 score >= 该值的样本（含边界）",
    )
    parser.add_argument(
        "--max_score",
        type=float,
        default=None,
        help="只渲染 score < 该值的样本，例如 0.5 表示低于 0.5 分",
    )

    parser.add_argument("--split", default="test", choices=["mini", "test", "trainval", "val"], help="NAVSIM split")
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
    parser.add_argument("--normalize_vl_hidden_states", action="store_true")
    parser.add_argument("--first_hidden_state_index", type=int, default=0)
    parser.add_argument("--last_hidden_state_index", type=int, default=-1)

    parser.add_argument("--time_horizon", type=float, default=4.0)
    parser.add_argument("--interval_length", type=float, default=0.5)
    parser.add_argument(
        "--debug_failures",
        action="store_true",
        help="对 NC<=0.05 或 DAC<=0.05 的样本叠加 PDM 仿真轨迹及碰撞/越界 box",
    )
    parser.add_argument(
        "--metric_cache_path",
        default="",
        help="metric cache 目录（debug_failures 必需，默认 $NAVSIM_EXP_ROOT/metric_cache2_v2）",
    )
    parser.add_argument("--num_history_frames", type=int, default=4)
    parser.add_argument("--mogo_map_sqlite_path", default="")
    parser.add_argument("--mogo_map_name", default="mogo")
    parser.add_argument("--mogo_map_query_radius_m", type=float, default=150.0)
    parser.add_argument("--mogo_map_max_segment_length_m", type=float, default=2.0)
    parser.add_argument("--mogo_map_enable_route_edge_filter", action="store_true", default=True)
    parser.add_argument("--mogo_map_geometry_srid", type=int, default=4326)
    parser.add_argument("--mogo_map_utm_zone", type=int, default=50)
    parser.add_argument("--mogo_map_utm_south", action="store_true")
    # --- 模型架构开关（须与 checkpoint 训练时一致；默认对齐 agent yaml 的"全关"VLM-only 配置，
    #     故不加任何 flag 时行为与旧版一致。换 obj/hdmap/obj-SDE/img/vlm 模型时显式 override）---
    parser.add_argument("--vl_hidden_dim", type=int, default=None,
                        help="VLM hidden dim；留空则按 vlm_type 自动推断（qwen3→2048）")
    parser.add_argument("--use_vlm", action=argparse.BooleanOptionalAction, default=True,
                        help="--no-use_vlm=完全不加载/不前向 VLM，DiT 只用 obj/hdmap/img token")
    parser.add_argument("--vlm_ablation", default="none", choices=["none", "zero"])
    parser.add_argument("--ego_status_encoder_type", default="mlp", choices=["mlp", "attention"],
                        help="attention=状态丢弃编码器 SDE")
    # obj token
    parser.add_argument("--use_object_vector", action="store_true")
    parser.add_argument("--object_encoder_type", default="mlp", choices=["mlp", "temporal"])
    parser.add_argument("--max_obj_tokens", type=int, default=64)
    parser.add_argument("--include_static_objects", action="store_true")
    parser.add_argument("--object_token_dropout", type=float, default=0.1)
    parser.add_argument("--static_speed_threshold", type=float, default=0.5)
    # hdmap token
    parser.add_argument("--use_hdmap_vector", action="store_true")
    parser.add_argument("--use_lane_boundaries", action="store_true")
    parser.add_argument("--strict_hdmap_vector", action="store_true")
    parser.add_argument("--max_map_tokens", type=int, default=128)
    parser.add_argument("--map_sample_points", type=int, default=20)
    parser.add_argument("--hdmap_radius", type=float, default=120.0)
    # 前视图像 ResNet encoder token
    parser.add_argument("--use_image_encoder", action="store_true")
    parser.add_argument("--image_encoder_arch", default="resnet34")
    parser.add_argument("--image_view", default="cam_f0")
    return parser.parse_args()


def _resolve_data_root(cli_value: str, env_key: str) -> str:
    """CLI 显式传空字符串时，回退到环境变量。"""
    return (cli_value or os.getenv(env_key, "")).strip()


def _require_env_paths(openscene_data_root: str) -> None:
    if not openscene_data_root:
        raise ValueError(
            "需要设置 --openscene_data_root 或环境变量 OPENSCENE_DATA_ROOT。\n"
            "示例: --openscene_data_root /NXT/lst/vla/navsim/download"
        )
    nuplan_maps_root = os.getenv("NUPLAN_MAPS_ROOT", "").strip()
    if not nuplan_maps_root:
        raise ValueError(
            "需要设置环境变量 NUPLAN_MAPS_ROOT（BEV 地图加载依赖）。\n"
            "示例:\n"
            "  export NUPLAN_MAPS_ROOT=/NXT/lst/vla/navsim/download/nuplan-maps-v1.1/nuplan-maps-v1.0\n"
            "  export NUPLAN_MAP_VERSION=nuplan-maps-v1.0"
        )
    if not Path(nuplan_maps_root).exists():
        raise ValueError(f"NUPLAN_MAPS_ROOT 路径不存在: {nuplan_maps_root}")


def main():
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    openscene_data_root = _resolve_data_root(args.openscene_data_root, "OPENSCENE_DATA_ROOT")
    if not args.openscene_data_root:
        _require_env_paths(openscene_data_root)
    args.openscene_data_root = openscene_data_root

    df, tokens = tokens_from_csv(
        args.csv,
        min_score=args.min_score,
        max_score=args.max_score,
    )
    if args.token:
        tokens = [str(args.token)]
    elif rank == 0:
        filter_desc = []
        if args.min_score is not None:
            filter_desc.append(f"score>={args.min_score}")
        if args.max_score is not None:
            filter_desc.append(f"score<{args.max_score}")
        if filter_desc:
            print(f"Score filter: {' & '.join(filter_desc)} -> {len(tokens)} tokens")
        else:
            print(f"No score filter -> {len(tokens)} tokens")

    scene_filter = build_scene_filter(args.scene_filter)
    print("use_vlm:", args.use_vlm)
    agent = build_agent(args, device)
    # 只渲染 + 推理需要前视 cam_f0；为避免某些 history iteration 未加载导致 cam_f0.image=None，
    # 这里显式只加载 cam_f0（所有 history iteration）。
    sensor_cfg = SensorConfig.build_all_sensors(include=False)
    sensor_cfg.cam_f0 = True
    map_api_builder = build_mogo_map_api_builder(args)
    scene_loader, scene_loader_traj = build_scene_loaders(
        openscene_data_root=Path(args.openscene_data_root),
        split=args.split,
        scene_filter=scene_filter,
        sensor_config=sensor_cfg,
        map_api_builder=map_api_builder,
    )

    out_dir = Path(args.out)

    metric_cache_loader = None
    simulator = None
    scorer = None
    # 传入 --metric_cache_path 时加载 cache：用于画红绿灯；debug_failures 额外建 simulator/scorer
    if args.metric_cache_path or args.debug_failures:
        metric_cache_path = args.metric_cache_path or str(
            Path(os.getenv("NAVSIM_EXP_ROOT", "/NXT/lst/vla/navsim/exp")) / "metric_cache2"
        )
        if rank == 0:
            extras = []
            if args.metric_cache_path:
                extras.append("traffic-light overlay")
            if args.debug_failures:
                extras.append("debug_failures")
            print(f"metric_cache_path={metric_cache_path} ({', '.join(extras) or 'enabled'})")
        metric_cache_loader = MetricCacheLoader(Path(metric_cache_path))
        if args.debug_failures:
            simulator, scorer = build_pdm_simulator_scorer()

    saved = save_vis_for_tokens(
        tokens=tokens,
        df=df,
        out_dir=out_dir,
        scene_loader=scene_loader,
        scene_loader_traj=scene_loader_traj,
        agent=agent,
        rank=rank,
        world_size=world_size,
        debug_failures=bool(args.debug_failures),
        metric_cache_loader=metric_cache_loader,
        simulator=simulator,
        scorer=scorer,
    )
    print(f"Rank {rank}: 保存 {saved} 张 -> {out_dir}")

    # 多卡收尾：等所有 rank 写完各自 index_rank{rank}.csv，再由 rank0 合并成总表。
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        if rank == 0:
            merge_rank_indices(out_dir, world_size)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
