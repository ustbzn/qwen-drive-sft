from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig

from navsim.planning.scenario_builder.mogo_sqlite_map_adapter import (
    MogoRouteContext,
    MogoSQLiteMapAdapter,
    MogoSQLiteMapConfig,
)


def build_mogo_map_api_builder_from_cfg(cfg: DictConfig) -> Optional[Callable[[List[Dict[str, Any]]], Any]]:
    """Create a SceneLoader map_api_builder when Mogo sqlite config is available."""
    mogo_cfg = cfg.get("mogo_map", None)
    if mogo_cfg is None:
        return None

    sqlite_path = str(mogo_cfg.get("sqlite_path", "") or "")
    if not sqlite_path:
        return None

    map_config = MogoSQLiteMapConfig(
        sqlite_path=Path(sqlite_path),
        map_name=str(mogo_cfg.get("map_name", "mogo")),
        query_radius_m=float(mogo_cfg.get("query_radius_m", 150.0)),
        max_segment_length_m=float(mogo_cfg.get("max_segment_length_m", 2.0)),
        enable_route_edge_filter=bool(mogo_cfg.get("enable_route_edge_filter", True)),
        geometry_srid=int(mogo_cfg.get("geometry_srid", 4326)),
        utm_zone=int(mogo_cfg.get("utm_zone", 50)),
        utm_south=bool(mogo_cfg.get("utm_south", False)),
    )
    num_history_frames = int(cfg.train_test_split.scene_filter.num_history_frames)

    def build_map_api(scene_dict_list: List[Dict[str, Any]]) -> MogoSQLiteMapAdapter:
        current_frame_idx = num_history_frames - 1
        route_roadblock_ids = list(scene_dict_list[current_frame_idx].get("roadblock_ids") or [])
        route_context = MogoRouteContext(roadblock_ids=route_roadblock_ids)
        return MogoSQLiteMapAdapter(config=map_config, route_context=route_context)

    return build_map_api
