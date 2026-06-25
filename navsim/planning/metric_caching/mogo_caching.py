from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import gc
import logging
import os
import uuid

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.training.experiments.cache_metadata_entry import (
    CacheMetadataEntry,
    CacheResult,
    save_cache_metadata,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.mogo_navsim_scenario import MogoNavSimScenario
from navsim.planning.scenario_builder.mogo_vehicle_parameters import (
    get_mogo_dynamic_parameters,
    get_mogo_vehicle_parameters,
)
from navsim.planning.scenario_builder.mogo_sqlite_map_adapter import (
    MogoRouteContext,
    MogoSQLiteMapAdapter,
    MogoSQLiteMapConfig,
)

logger = logging.getLogger(__name__)


def _build_mogo_map_api(cfg: DictConfig, route_roadblock_ids: List[str]) -> MogoSQLiteMapAdapter:
    """Build a Mogo SQLite map api for one scenario route."""
    mogo_map_cfg = cfg.mogo_map
    map_config = MogoSQLiteMapConfig(
        sqlite_path=Path(mogo_map_cfg.sqlite_path),
        map_name=mogo_map_cfg.get("map_name", "mogo"),
        query_radius_m=float(mogo_map_cfg.get("query_radius_m", 150.0)),
        max_segment_length_m=float(mogo_map_cfg.get("max_segment_length_m", 2.0)),
        enable_route_edge_filter=bool(mogo_map_cfg.get("enable_route_edge_filter", True)),
        geometry_srid=int(mogo_map_cfg.get("geometry_srid", 4326)),
        utm_zone=int(mogo_map_cfg.get("utm_zone", 50)),
        utm_south=bool(mogo_map_cfg.get("utm_south", False)),
    )
    route_context = MogoRouteContext(roadblock_ids=route_roadblock_ids)
    return MogoSQLiteMapAdapter(config=map_config, route_context=route_context)


def cache_mogo_scenarios(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
    """Performs metric caching for Mogo/internal NAVSIM-format scenes."""

    def cache_mogo_scenarios_internal(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[CacheResult]:
        def cache_single_scenario(
            scene_dict: List[Dict[str, Any]],
            processor: MetricCacheProcessor,
        ) -> Optional[CacheMetadataEntry]:
            current_frame_idx = cfg.train_test_split.scene_filter.num_history_frames - 1
            route_roadblock_ids = list(scene_dict[current_frame_idx]["roadblock_ids"])
            map_api = _build_mogo_map_api(cfg, route_roadblock_ids)
            scene = Scene.from_scene_dict_list(
                scene_dict,
                None,
                num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,
                num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,
                sensor_config=SensorConfig.build_no_sensors(),
                map_api=map_api,
            )
            scenario = MogoNavSimScenario(
                scene,
                ego_vehicle_parameters=get_mogo_vehicle_parameters(),
                database_interval=float(cfg.mogo_observation.annotation_interval_length),
            )
            return processor.compute_metric_cache(scenario)

        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        log_names = [a["log_file"] for a in args]
        tokens = [token for a in args for token in a["tokens"]]
        cfg: DictConfig = args[0]["cfg"]

        scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
        scene_filter.log_names = log_names
        scene_filter.tokens = tokens
        scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_no_sensors(),
        )

        assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"
        processor = MetricCacheProcessor(
            cache_path=cfg.cache.cache_path,
            force_feature_computation=cfg.cache.force_feature_computation,
            use_dense_annotations=bool(cfg.mogo_observation.use_dense_annotations),
            history_interval_length=float(cfg.mogo_observation.history_interval_length),
            ego_vehicle_parameters=get_mogo_vehicle_parameters(),
            ego_dynamic_parameters=get_mogo_dynamic_parameters(),
        )

        logger.info(f"Extracted {len(scene_loader)} Mogo scenarios for thread_id={thread_id}, node_id={node_id}.")
        num_failures = 0
        num_successes = 0
        all_file_cache_metadata: List[Optional[CacheMetadataEntry]] = []
        for idx, scene_dict in enumerate(scene_loader.scene_frames_dicts.values()):
            logger.info(
                f"Processing Mogo scenario {idx + 1} / {len(scene_loader)} "
                f"in thread_id={thread_id}, node_id={node_id}"
            )
            try:
                file_cache_metadata = cache_single_scenario(scene_dict, processor)
            except Exception as exc:
                current_frame_idx = cfg.train_test_split.scene_filter.num_history_frames - 1
                frame = scene_dict[current_frame_idx]
                logger.warning(
                    "Failed to cache Mogo scenario token=%s log=%s error=%s: %s; skipping.",
                    frame.get("token"),
                    frame.get("log_name"),
                    type(exc).__name__,
                    exc,
                )
                file_cache_metadata = None
            gc.collect()

            num_failures += 0 if file_cache_metadata else 1
            num_successes += 1 if file_cache_metadata else 0
            all_file_cache_metadata += [file_cache_metadata]

        logger.info(f"Finished processing Mogo scenarios for thread_id={thread_id}, node_id={node_id}")
        return [
            CacheResult(
                failures=num_failures,
                successes=num_successes,
                cache_metadata=all_file_cache_metadata,
            )
        ]

    result = cache_mogo_scenarios_internal(args)
    gc.collect()
    return result


def cache_mogo_data(cfg: DictConfig, worker: WorkerPool) -> None:
    """Build Mogo metric caches from NAVSIM-format internal logs."""
    assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    logger.info("Starting Mogo metric caching of %s files...", str(len(data_points)))

    cache_results = worker_map(worker, cache_mogo_scenarios, data_points)

    num_success = sum(result.successes for result in cache_results)
    num_fail = sum(result.failures for result in cache_results)
    num_total = num_success + num_fail
    if num_fail == 0:
        logger.info("Completed Mogo metric caching! All %s caches were generated successfully.", str(num_total))
    else:
        logger.info("Completed Mogo metric caching! Failed caches: %s out of %s", str(num_fail), str(num_total))

    cached_metadata = [
        cache_metadata_entry
        for cache_result in cache_results
        for cache_metadata_entry in cache_result.cache_metadata
        if cache_metadata_entry is not None
    ]

    node_id = int(os.environ.get("NODE_RANK", 0))
    logger.info(f"Node {node_id}: Storing Mogo metric cache metadata csv...")
    save_cache_metadata(cached_metadata, Path(cfg.cache.cache_path), node_id)
    logger.info("Done storing Mogo metric cache metadata.")
