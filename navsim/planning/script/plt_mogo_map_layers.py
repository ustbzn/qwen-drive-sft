#!/usr/bin/env python3
"""Visualize Mogo SQLite map layers and PDM drivable area / route centerline.

Map layers: LANE, ROADBLOCK, LANE_CONNECTOR, ROADBLOCK_CONNECTOR, INTERSECTION.
Combined layers: ROADBLOCK_INTERSECTION, LANE_LANE_CONNECTOR.
PDM layers: DRIVABLE_AREA (PDMDrivableMap), CENTERLINE (route Dijkstra centerline).

Example:
  python navsim/planning/script/plt_mogo_map_layers.py \\
    --data-root /NXT/lst/vla/mogo_data/2026-03-13-15-22-20 \\
    --pkl JLBJA20073D_20260313152238.pkl \\
    --frame-idx 150 \\
    --query-radius-m 200 --map-radius-m 200 --margin-m 200

  # Saves one png per layer under vis_output/map/ by default.
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from shapely.geometry import LineString, MultiPolygon, Polygon

_REPO_ROOT = Path(__file__).resolve().parents[3]
if _REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, _REPO_ROOT.as_posix())


def _patch_nuplan_map_imports() -> None:
    """Allow mogo_sqlite_map_adapter import when nuPlan map symbols differ by version."""
    try:
        import nuplan.common.maps.abstract_map_objects as map_objects  # type: ignore

        if not hasattr(map_objects, "MapObject"):
            map_objects.MapObject = object  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        pass


_patch_nuplan_map_imports()

from navsim.planning.scenario_builder.mogo_sqlite_map_adapter import (
    MogoRouteContext,
    MogoSQLiteLayerBuilder,
    MogoSQLiteMapAdapter,
    MogoSQLiteMapConfig,
    _semantic_layer,
)
from navsim.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

logger = logging.getLogger(__name__)

MAP_LAYER_ORDER: Tuple[str, ...] = (
    "INTERSECTION",
    "ROADBLOCK",
    "ROADBLOCK_CONNECTOR",
    "LANE",
    "LANE_CONNECTOR",
)

# Extra combined views: (output suffix, layer names in draw order).
COMBINED_LAYER_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("ROADBLOCK_INTERSECTION", ("ROADBLOCK", "INTERSECTION")),
    ("LANE_LANE_CONNECTOR", ("LANE", "LANE_CONNECTOR")),
)

PDM_LAYER_ORDER: Tuple[str, ...] = (
    "DRIVABLE_AREA",
    "CENTERLINE",
)

LAYER_ORDER: Tuple[str, ...] = MAP_LAYER_ORDER + PDM_LAYER_ORDER

LAYER_STYLES: Dict[str, Dict[str, Any]] = {
    "INTERSECTION": {"facecolor": "#9b59b6", "edgecolor": "#6c3483", "alpha": 0.30, "linewidth": 0.6},
    "ROADBLOCK": {"facecolor": "#3498db", "edgecolor": "#1f618d", "alpha": 0.18, "linewidth": 0.5},
    "ROADBLOCK_CONNECTOR": {"facecolor": "#e67e22", "edgecolor": "#ba4a00", "alpha": 0.35, "linewidth": 0.7},
    "LANE": {"facecolor": "#2ecc71", "edgecolor": "#1e8449", "alpha": 0.22, "linewidth": 0.4},
    "LANE_CONNECTOR": {"facecolor": "#e74c3c", "edgecolor": "#922b21", "alpha": 0.40, "linewidth": 0.8},
}

DRIVABLE_TYPE_STYLES: Dict[Any, Dict[str, Any]] = {
    SemanticMapLayer.ROADBLOCK: {"facecolor": "#3498db", "edgecolor": "#1f618d", "alpha": 0.20, "linewidth": 0.4},
    SemanticMapLayer.LANE: {"facecolor": "#2ecc71", "edgecolor": "#1e8449", "alpha": 0.25, "linewidth": 0.35},
    SemanticMapLayer.LANE_CONNECTOR: {"facecolor": "#e74c3c", "edgecolor": "#922b21", "alpha": 0.30, "linewidth": 0.5},
    SemanticMapLayer.INTERSECTION: {"facecolor": "#9b59b6", "edgecolor": "#6c3483", "alpha": 0.28, "linewidth": 0.5},
    SemanticMapLayer.CARPARK_AREA: {"facecolor": "#95a5a6", "edgecolor": "#566573", "alpha": 0.25, "linewidth": 0.4},
}

CENTERLINE_STYLE = {"color": "#c0392b", "linewidth": 2.5, "alpha": 0.95}


def _default_sqlite_path(data_root: Path) -> Path:
    maps_dir = data_root / "dataset" / "maps"
    candidates = sorted(maps_dir.glob("*.sqlite"))
    if not candidates:
        raise FileNotFoundError(f"No .sqlite file under {maps_dir}")
    return candidates[0]


def _default_pkl_path(data_root: Path, pkl_name: Optional[str]) -> Path:
    logs_dir = data_root / "dataset" / "navsim_logs" / "trainval"
    if pkl_name:
        path = Path(pkl_name)
        if path.is_file():
            return path
        return logs_dir / pkl_name
    pkls = sorted(logs_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl under {logs_dir}")
    return pkls[0]


def _load_frame_context(pkl_path: Path, frame_idx: int) -> Tuple[np.ndarray, List[str], str, int, Dict[str, Any]]:
    with pkl_path.open("rb") as handle:
        scene = pickle.load(handle)
    if not scene:
        raise ValueError(f"Empty scene in {pkl_path}")
    idx = frame_idx if frame_idx >= 0 else len(scene) + frame_idx
    if idx < 0 or idx >= len(scene):
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {len(scene) - 1}]")
    frame = scene[idx]
    ego = np.asarray(frame["ego2global_translation"][:2], dtype=np.float64)
    roadblocks = list(frame.get("roadblock_ids") or [])
    token = str(frame.get("token", f"frame_{idx}"))
    return ego, roadblocks, token, idx, frame


def _quaternion_to_yaw(quaternion_wxyz: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in quaternion_wxyz[:4]]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ego_state_from_frame(frame: Dict[str, Any]) -> EgoState:
    translation = frame["ego2global_translation"]
    yaw = _quaternion_to_yaw(frame["ego2global_rotation"])
    dynamics = list(frame.get("ego_dynamic_state") or [0.0, 0.0, 0.0, 0.0])
    while len(dynamics) < 4:
        dynamics.append(0.0)
    timestamp_us = int(frame.get("timestamp", 0))
    return EgoState.build_from_rear_axle(
        StateSE2(float(translation[0]), float(translation[1]), yaw),
        tire_steering_angle=0.0,
        vehicle_parameters=get_pacifica_parameters(),
        time_point=TimePoint(timestamp_us),
        rear_axle_velocity_2d=StateVector2D(float(dynamics[0]), float(dynamics[1])),
        rear_axle_acceleration_2d=StateVector2D(float(dynamics[2]), float(dynamics[3])),
    )


def _build_map_api(
    sqlite_path: Path,
    ego_xy: np.ndarray,
    roadblock_ids: Sequence[str],
    query_radius_m: float,
    utm_zone: int,
    utm_south: bool,
) -> MogoSQLiteMapAdapter:
    config = MogoSQLiteMapConfig(
        sqlite_path=sqlite_path,
        query_radius_m=query_radius_m,
        utm_zone=utm_zone,
        utm_south=utm_south,
    )
    route_context = MogoRouteContext(roadblock_ids=list(roadblock_ids)) if roadblock_ids else None
    map_api = MogoSQLiteMapAdapter(config=config, route_context=route_context)
    map_api.load(Point2D(float(ego_xy[0]), float(ego_xy[1])))
    return map_api


class _PDMCenterlineHelper(AbstractPDMPlanner):
    """Minimal PDM planner surface for route centerline extraction."""

    def initialize(self, initialization: Any) -> None:
        pass

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Any:
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        raise NotImplementedError("Visualization helper only.")

    def compute_route_centerline(
        self,
        map_api: MogoSQLiteMapAdapter,
        route_roadblock_ids: Sequence[str],
        ego_state: EgoState,
    ) -> Tuple[List[StateSE2], str]:
        self._map_api = map_api
        self._iteration = 0
        self._load_route_dicts(list(route_roadblock_ids))
        self._route_roadblock_correction(ego_state)
        self._drivable_area_map = PDMDrivableMap.from_simulation(
            self._map_api,
            ego_state,
            float(self._map_radius),
        )
        starting_lane = self._get_starting_lane(ego_state)
        centerline = self._get_discrete_centerline(starting_lane)
        return centerline, starting_lane.id


def _compute_pdm_artifacts(
    map_api: MogoSQLiteMapAdapter,
    roadblock_ids: Sequence[str],
    ego_state: EgoState,
    map_radius_m: float,
) -> Tuple[PDMDrivableMap, List[StateSE2], str]:
    if not roadblock_ids:
        raise ValueError("CENTERLINE and DRIVABLE_AREA require route roadblock_ids from pkl (do not use --no-route).")
    helper = _PDMCenterlineHelper(map_radius=int(map_radius_m))
    centerline, start_lane_id = helper.compute_route_centerline(map_api, roadblock_ids, ego_state)
    assert helper._drivable_area_map is not None
    return helper._drivable_area_map, centerline, start_lane_id


class _Point2D:
    """Minimal Point2D when nuPlan is not installed."""

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)


def _iter_polygons(geometry: Any) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, MultiPolygon):
        for part in geometry.geoms:
            if not part.is_empty and part.geom_type == "Polygon":
                yield part
        return
    if geometry.geom_type == "Polygon":
        yield geometry


def _plot_polygon(ax: plt.Axes, polygon: Polygon, style: Dict[str, Any], zorder: int) -> None:
    facecolor = style["facecolor"]
    edgecolor = style["edgecolor"]
    alpha = style["alpha"]
    linewidth = style.get("linewidth", 0.5)
    for part in _iter_polygons(polygon):
        x, y = part.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, zorder=zorder)
        for interior in part.interiors:
            ix, iy = interior.xy
            ax.fill(
                ix,
                iy,
                facecolor="white",
                edgecolor=edgecolor,
                alpha=1.0,
                linewidth=0.3,
                zorder=zorder + 1,
            )


def _plot_baseline(ax: plt.Axes, linestring: LineString, color: str, zorder: int) -> None:
    if linestring is None or linestring.is_empty:
        return
    x, y = linestring.xy
    ax.plot(x, y, color=color, linewidth=0.6, alpha=0.7, zorder=zorder)


def _build_layers(
    sqlite_path: Path,
    ego_xy: np.ndarray,
    roadblock_ids: Sequence[str],
    query_radius_m: float,
    utm_zone: int,
    utm_south: bool,
) -> Dict[Any, List[Any]]:
    config = MogoSQLiteMapConfig(
        sqlite_path=sqlite_path,
        query_radius_m=query_radius_m,
        utm_zone=utm_zone,
        utm_south=utm_south,
    )
    builder = MogoSQLiteLayerBuilder(config)
    route_context = MogoRouteContext(roadblock_ids=list(roadblock_ids)) if roadblock_ids else None
    return builder.build_layers(
        ego_position=_Point2D(float(ego_xy[0]), float(ego_xy[1])),
        route_context=route_context,
    )


def _layer_key(name: str) -> Any:
    return _semantic_layer(name)


def _count_objects(layers: Dict[Any, List[Any]], name: str) -> int:
    return len(layers.get(_layer_key(name), []))


def _apply_bev_axes(ax: plt.Axes, ego_xy: np.ndarray, margin_m: float) -> None:
    half = float(margin_m)
    ax.set_xlim(ego_xy[0] - half, ego_xy[0] + half)
    ax.set_ylim(ego_xy[1] - half, ego_xy[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("UTM x (m)")
    ax.set_ylabel("UTM y (m)")


def _plot_ego_marker(ax: plt.Axes, ego_xy: np.ndarray) -> None:
    ax.plot(
        ego_xy[0],
        ego_xy[1],
        marker="*",
        markersize=16,
        color="#f1c40f",
        markeredgecolor="#7d6608",
        markeredgewidth=1.0,
        zorder=100,
        label="ego",
    )


def _plot_shapely_geometry(ax: plt.Axes, geometry: Any, style: Dict[str, Any], zorder: int) -> None:
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        _plot_polygon(ax, geometry, style, zorder=zorder)
    elif geometry.geom_type == "MultiPolygon":
        for part in geometry.geoms:
            _plot_polygon(ax, part, style, zorder=zorder)


def plot_drivable_area(
    drivable_map: PDMDrivableMap,
    ego_xy: np.ndarray,
    *,
    title: str,
    margin_m: float,
) -> plt.Figure:
    """Plot PDMDrivableMap polygons (same composition as metric cache)."""
    fig, ax = plt.subplots(figsize=(14, 14))
    type_counts: Dict[str, int] = {}

    for index, geometry in enumerate(drivable_map._geometries):
        map_type = drivable_map._map_types[index]
        style = DRIVABLE_TYPE_STYLES.get(
            map_type,
            {"facecolor": "#bdc3c7", "edgecolor": "#7f8c8d", "alpha": 0.25, "linewidth": 0.4},
        )
        _plot_shapely_geometry(ax, geometry, style, zorder=10)
        type_name = map_type.name if hasattr(map_type, "name") else str(map_type)
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    _plot_ego_marker(ax, ego_xy)
    _apply_bev_axes(ax, ego_xy, margin_m)

    legend_handles = [
        Patch(
            facecolor=DRIVABLE_TYPE_STYLES[t]["facecolor"],
            edgecolor=DRIVABLE_TYPE_STYLES[t]["edgecolor"],
            label=f"{t.name} ({type_counts.get(t.name, 0)})",
        )
        for t in DRIVABLE_TYPE_STYLES
        if type_counts.get(t.name, 0) > 0
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#f1c40f",
            markeredgecolor="#7d6608",
            markersize=14,
            label="ego",
        )
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    total = len(drivable_map._geometries)
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    ax.set_title(f"{title}\nDRIVABLE_AREA: {total} polygons ({counts_str})", fontsize=11)
    fig.tight_layout()
    return fig


def plot_centerline(
    centerline: Sequence[StateSE2],
    ego_xy: np.ndarray,
    *,
    title: str,
    margin_m: float,
    start_lane_id: str,
    drivable_map: Optional[PDMDrivableMap] = None,
) -> plt.Figure:
    """Plot PDM route centerline; optional faint drivable area background."""
    fig, ax = plt.subplots(figsize=(14, 14))

    if drivable_map is not None:
        bg_style = {"facecolor": "#ecf0f1", "edgecolor": "#bdc3c7", "alpha": 0.35, "linewidth": 0.2}
        for geometry in drivable_map._geometries:
            _plot_shapely_geometry(ax, geometry, bg_style, zorder=5)

    if centerline:
        xs = [state.x for state in centerline]
        ys = [state.y for state in centerline]
        ax.plot(xs, ys, **CENTERLINE_STYLE, zorder=30, label=f"centerline ({len(centerline)} pts)")
        step = max(1, len(centerline) // 25)
        for idx in range(0, len(centerline), step):
            state = centerline[idx]
            dx = math.cos(state.heading) * 3.0
            dy = math.sin(state.heading) * 3.0
            ax.arrow(
                state.x,
                state.y,
                dx,
                dy,
                head_width=1.5,
                head_length=2.0,
                fc=CENTERLINE_STYLE["color"],
                ec=CENTERLINE_STYLE["color"],
                alpha=0.6,
                zorder=31,
                length_includes_head=True,
            )

    _plot_ego_marker(ax, ego_xy)
    _apply_bev_axes(ax, ego_xy, margin_m)
    length_m = 0.0
    if len(centerline) >= 2:
        diffs = np.diff(np.asarray([[s.x, s.y] for s in centerline], dtype=np.float64), axis=0)
        length_m = float(np.sum(np.linalg.norm(diffs, axis=1)))

    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"{title}\nCENTERLINE: {len(centerline)} states, ~{length_m:.1f}m, start_lane={start_lane_id}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_single_layer(
    layers: Dict[Any, List[Any]],
    ego_xy: np.ndarray,
    layer_name: str,
    *,
    title: str,
    show_baselines: bool,
    margin_m: float,
) -> plt.Figure:
    """Plot one semantic layer on its own figure."""
    fig, ax = plt.subplots(figsize=(14, 14))
    style = LAYER_STYLES[layer_name]
    objects = layers.get(_layer_key(layer_name), [])

    for obj in objects:
        _plot_polygon(ax, obj.polygon, style, zorder=10)
        if show_baselines and hasattr(obj, "baseline_path"):
            _plot_baseline(ax, obj.baseline_path.linestring, style["edgecolor"], zorder=20)

    _plot_ego_marker(ax, ego_xy)
    _apply_bev_axes(ax, ego_xy, margin_m)

    count = len(objects)
    ax.legend(
        handles=[
            Patch(facecolor=style["facecolor"], edgecolor=style["edgecolor"], label=f"{layer_name} (n={count})"),
            plt.Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#f1c40f",
                markeredgecolor="#7d6608",
                markersize=14,
                label="ego",
            ),
        ],
        loc="upper right",
        fontsize=9,
    )
    ax.set_title(f"{title}\n{layer_name}: {count} objects", fontsize=11)
    fig.tight_layout()
    return fig


def plot_combined_layers(
    layers: Dict[Any, List[Any]],
    ego_xy: np.ndarray,
    layer_names: Sequence[str],
    *,
    title: str,
    show_baselines: bool,
    margin_m: float,
    combined_label: str,
) -> plt.Figure:
    """Plot multiple semantic layers on one figure."""
    fig, ax = plt.subplots(figsize=(14, 14))
    counts: Dict[str, int] = {}

    for layer_name in layer_names:
        style = LAYER_STYLES[layer_name]
        objects = layers.get(_layer_key(layer_name), [])
        counts[layer_name] = len(objects)
        for obj in objects:
            _plot_polygon(ax, obj.polygon, style, zorder=10)
            if show_baselines and hasattr(obj, "baseline_path"):
                _plot_baseline(ax, obj.baseline_path.linestring, style["edgecolor"], zorder=20)

    _plot_ego_marker(ax, ego_xy)
    _apply_bev_axes(ax, ego_xy, margin_m)

    legend_handles = [
        Patch(
            facecolor=LAYER_STYLES[name]["facecolor"],
            edgecolor=LAYER_STYLES[name]["edgecolor"],
            label=f"{name} (n={counts.get(name, 0)})",
        )
        for name in layer_names
    ]
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#f1c40f",
            markeredgecolor="#7d6608",
            markersize=14,
            label="ego",
        )
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    total = sum(counts.values())
    counts_str = ", ".join(f"{name}={counts.get(name, 0)}" for name in layer_names)
    ax.set_title(f"{title}\n{combined_label}: {total} objects ({counts_str})", fontsize=11)
    fig.tight_layout()
    return fig


def save_layers_separately(
    layers: Dict[Any, List[Any]],
    ego_xy: np.ndarray,
    output_dir: Path,
    *,
    file_prefix: str,
    title: str,
    show_baselines: bool,
    margin_m: float,
    dpi: int,
    drivable_map: Optional[PDMDrivableMap] = None,
    centerline: Optional[Sequence[StateSE2]] = None,
    start_lane_id: str = "",
) -> List[Path]:
    """Save one PNG per layer; returns written file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for layer_name in MAP_LAYER_ORDER:
        fig = plot_single_layer(
            layers,
            ego_xy,
            layer_name,
            title=title,
            show_baselines=show_baselines,
            margin_m=margin_m,
        )
        out_path = output_dir / f"{file_prefix}_{layer_name}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        logger.info("Saved %s (%d objects)", out_path.name, _count_objects(layers, layer_name))

    for combined_label, layer_names in COMBINED_LAYER_GROUPS:
        fig = plot_combined_layers(
            layers,
            ego_xy,
            layer_names,
            title=title,
            show_baselines=show_baselines,
            margin_m=margin_m,
            combined_label=combined_label,
        )
        out_path = output_dir / f"{file_prefix}_{combined_label}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        total = sum(_count_objects(layers, name) for name in layer_names)
        logger.info("Saved %s (%d objects)", out_path.name, total)

    if drivable_map is not None:
        fig = plot_drivable_area(drivable_map, ego_xy, title=title, margin_m=margin_m)
        out_path = output_dir / f"{file_prefix}_DRIVABLE_AREA.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        logger.info("Saved %s (%d polygons)", out_path.name, len(drivable_map._geometries))

    if centerline is not None:
        fig = plot_centerline(
            centerline,
            ego_xy,
            title=title,
            margin_m=margin_m,
            start_lane_id=start_lane_id,
            drivable_map=drivable_map,
        )
        out_path = output_dir / f"{file_prefix}_CENTERLINE.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        logger.info("Saved %s (%d states)", out_path.name, len(centerline))

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Mogo SQLite map semantic layers.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/NXT/lst/vla/mogo_data/2026-03-13-15-22-20"),
        help="Dataset root containing dataset/maps and dataset/navsim_logs.",
    )
    parser.add_argument("--sqlite-path", type=Path, default=None, help="SQLite map file (default: first under dataset/maps).")
    parser.add_argument("--pkl", type=str, default=None, help="Navsim log pkl name or path under trainval.")
    parser.add_argument("--frame-idx", type=int, default=-1, help="Frame index in pkl (-1 = last frame).")
    parser.add_argument("--query-radius-m", type=float, default=200.0, help="Local map query radius around ego.")
    parser.add_argument("--utm-zone", type=int, default=50)
    parser.add_argument("--utm-south", action="store_true")
    parser.add_argument("--margin-m", type=float, default=200.0, help="Half-width of BEV plot around ego.")
    parser.add_argument(
        "--map-radius-m",
        type=float,
        default=200.0,
        help="Radius for PDMDrivableMap query.",
    )
    parser.add_argument("--no-route", action="store_true", help="Do not pass roadblock_ids from pkl (local map only).")
    parser.add_argument(
        "--skip-pdm-layers",
        action="store_true",
        help="Skip DRIVABLE_AREA and CENTERLINE (map semantic layers only).",
    )
    parser.add_argument("--show-baselines", action="store_true", help="Overlay lane / connector centerlines.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "vis_output" / "map",
        help="Directory for per-layer png files (default: vis_output/map/).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    data_root = args.data_root.expanduser().resolve()
    sqlite_path = (args.sqlite_path or _default_sqlite_path(data_root)).resolve()
    pkl_path = _default_pkl_path(data_root, args.pkl).resolve()

    ego_xy, roadblock_ids, token, resolved_frame_idx, frame = _load_frame_context(pkl_path, args.frame_idx)
    if args.no_route:
        roadblock_ids = []

    logger.info("sqlite=%s", sqlite_path)
    logger.info(
        "pkl=%s frame=%d token=%s ego=(%.2f, %.2f) route_roadblocks=%d",
        pkl_path.name,
        resolved_frame_idx,
        token,
        ego_xy[0],
        ego_xy[1],
        len(roadblock_ids),
    )

    layers = _build_layers(
        sqlite_path=sqlite_path,
        ego_xy=ego_xy,
        roadblock_ids=roadblock_ids,
        query_radius_m=args.query_radius_m,
        utm_zone=args.utm_zone,
        utm_south=args.utm_south,
    )

    drivable_map: Optional[PDMDrivableMap] = None
    centerline: Optional[List[StateSE2]] = None
    start_lane_id = ""

    if not args.skip_pdm_layers and roadblock_ids:
        map_api = _build_map_api(
            sqlite_path=sqlite_path,
            ego_xy=ego_xy,
            roadblock_ids=roadblock_ids,
            query_radius_m=args.query_radius_m,
            utm_zone=args.utm_zone,
            utm_south=args.utm_south,
        )
        ego_state = _ego_state_from_frame(frame)
        drivable_map, centerline, start_lane_id = _compute_pdm_artifacts(
            map_api,
            roadblock_ids,
            ego_state,
            map_radius_m=args.map_radius_m,
        )
        logger.info(
            "PDM artifacts: drivable_polygons=%d, centerline_states=%d, start_lane=%s",
            len(drivable_map._geometries),
            len(centerline),
            start_lane_id,
        )
    elif not args.skip_pdm_layers:
        logger.warning("Skipping DRIVABLE_AREA / CENTERLINE: no route roadblock_ids.")

    title = f"{pkl_path.stem} | frame={resolved_frame_idx} | {token}"
    file_prefix = f"{pkl_path.stem}_f{resolved_frame_idx}"

    output_dir = (args.output_dir.expanduser().resolve() / file_prefix)
    saved = save_layers_separately(
        layers,
        ego_xy,
        output_dir,
        file_prefix=file_prefix,
        title=title,
        show_baselines=args.show_baselines,
        margin_m=args.margin_m,
        dpi=args.dpi,
        drivable_map=drivable_map,
        centerline=centerline,
        start_lane_id=start_lane_id,
    )
    logger.info("Wrote %d layer images to %s", len(saved), output_dir)


if __name__ == "__main__":
    main()
