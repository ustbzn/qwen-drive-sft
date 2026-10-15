"""
Qwen-Drive-1.0 IL Training Entry (Stage 3: Planning Expert Pretraining).

Based on run_training_recogdrive.py. Trains the Qwen-Drive Planning Expert via
x-prediction flow matching, conditioned on the frozen VLM's online-computed
attention K/V (the 8 full-attention/GQA layers named by the official config's
`full_attention_layers`). The VLM is frozen and only the Planning Expert
(32-layer joint-attention transformer) receives gradients.

Key differences from run_training_recogdrive.py:
- agent=qwen_drive_agent (instead of recogdrive_agent)
- trajectory format: 50 waypoints @ 10Hz (5s) instead of 8 @ 2Hz (4s)
- No offline VLM feature cache. QwenDriveFeatureBuilder only emits small ego
  tensors (history/history_velocity/history_acceleration/ego_status/nav_command)
  plus a [view][frame] image *path* matrix; QwenDriveAgent runs the VLM online
  under no_grad and only backprops into the Planning Expert. collate_fn stacks
  the small tensors and passes the path matrices through as a plain list.
- train_test_split uses mogo_qwen_drive (16 history + 50 future at 10Hz).
  It selects the same 66-raw-frame windows and current tokens as mogo_10hz
  (4/10 frames with stride 5), while retaining dense poses for the expert.
- Raw logs are packed once by rank zero; all workers read scene blocks on demand.
- MOGO manifest entries supply independent raw log and image roots. Their
  datasets are concatenated after applying the shared train/validation log split.

Usage:
    python -m navsim.planning.script.run_training_qwen_drive_il \
        agent=qwen_drive_agent \
        train_test_split=mogo_qwen_drive \
        trainer.params.max_epochs=200
"""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import logging
import os
import traceback
from time import perf_counter

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.qwen_drive_planner.qwen_drive_features import QWEN_DRIVE_IMAGE_KEY
from navsim.common.dataclasses import SceneFilter
from navsim.common.scene_cameras import camera_requirements
from navsim.common.dataloader import SceneLoader
from navsim.common.indexed_scene_loader import IndexedSceneLoader, prepare_scene_index
from navsim.planning.training.dataset import Dataset, apply_qwen_train_oversample
from navsim.planning.training.distributed_cache_dataset import (
    build_distributed_cache_only_datasets,
    cache_index_sync_group,
)
from navsim.planning.training.physicalai_dataset import (
    MixedSourceDataset,
    PhysicalAIDataset,
    parse_mixed_ratio,
)
from navsim.agents.qwen_drive_planner.qwen_drive_lightning import QwenDriveLightningModule
from navsim.planning.scenario_builder.mogo_map_api_builder import build_mogo_map_api_builder_from_cfg
import torch

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "qwen_drive_il"


def custom_collate_fn(
    batch: Union[
        Sequence[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]],
        Sequence[Tuple[Dict[str, Any], Dict[str, torch.Tensor], str]],
    ],
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Tuple[Optional[str], ...]]:
    """Collate a batch of Qwen-Drive IL samples produced by QwenDriveFeatureBuilder.

    Feature keys and shapes (see qwen_drive_features.build_qwen_drive_ego_features):
      - history:               [num_history_points, 3]      -> stacked [B, H, 3]
      - history_velocity:      [num_history_points, 2]       -> stacked [B, H, 2]
      - history_acceleration:  [num_history_points, 2]       -> stacked [B, H, 2]
      - ego_status:            [8]                           -> stacked [B, 8]
      - nav_command:           scalar long                   -> stacked [B]
      - view_frame_paths ("QWEN_DRIVE_IMAGE_KEY"): nested [view][frame] path
        matrix, a plain list (not a tensor) -> kept as a list of length B.

    Target key: trajectory [num_future_points, 3] -> stacked [B, T, 3].

    `Dataset.__getitem__` (the online, non-cache-only dataset actually used here
    since this agent never caches VLM K/V) returns a 2-tuple (features, targets)
    without a token. `MultiCacheOnlyDataset`/distributed cache datasets return a
    3-tuple with the token. Both are accepted; the token list is all-None when
    unavailable (QwenDriveAgent.forward never reads tokens_list).
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    features_list, targets_list, tokens = [], [], []
    for sample in batch:
        if len(sample) == 3:
            features, targets, token = sample
        elif len(sample) == 2:
            features, targets = sample
            token = None
        else:
            raise ValueError(f"Unexpected sample tuple length {len(sample)}; expected 2 or 3")
        features_list.append(features)
        targets_list.append(targets)
        tokens.append(token)
    tokens_list = tuple(tokens)

    tensor_feature_keys = ("history", "history_velocity", "history_acceleration", "ego_status", "nav_command")
    f0 = features_list[0]

    features: Dict[str, Any] = {}
    required_keys = set(tensor_feature_keys) | {QWEN_DRIVE_IMAGE_KEY, "navigation_instruction"}
    for sample in features_list:
        if not required_keys.issubset(sample):
            raise ValueError(f"Missing Qwen-Drive features: {sorted(required_keys - set(sample))}")
    for key in tensor_feature_keys:
        features[key] = torch.stack([sample[key] for sample in features_list], dim=0).cpu()

    if QWEN_DRIVE_IMAGE_KEY in f0:
        # 图像路径矩阵是嵌套 list，不是 tensor；QwenDriveAgent._encode_scenes 按 batch
        # 维逐样本取用，这里只需保持 list 顺序与其余 tensor 特征的 batch 维一致。
        features[QWEN_DRIVE_IMAGE_KEY] = [f[QWEN_DRIVE_IMAGE_KEY] for f in features_list]

    # MOGO and PhysicalAI use different camera fields for the same three
    # semantic prompt slots. Keep the source view list per sample so mixed
    # batches can generate the correct labels without changing the model.
    features["view_lists"] = [sample.get("view_list") for sample in features_list]

    # Per-sample natural-language commands have varying lengths and stay as Python
    # strings; QwenDriveAgent adds each one to the matching scene prompt before VLM prefill.
    features["navigation_instruction"] = [f["navigation_instruction"] for f in features_list]

    # Targets: stack generically (trajectory [B, 50, 3] included).
    target_keys = set(targets_list[0])
    if any(set(sample) != target_keys for sample in targets_list[1:]):
        raise ValueError("All samples in a batch must expose identical target keys")
    targets = {
        key: torch.stack([sample[key] for sample in targets_list], dim=0).cpu()
        for key in sorted(target_keys)
    }

    return features, targets, tokens_list


def _dataset_source_configs(cfg: DictConfig) -> Sequence[Tuple[str, DictConfig]]:
    """Resolve raw data roots and the manifest cache_path for each source."""
    manifest_path = cfg.get("mogo_manifest")
    if not manifest_path:
        source_cfg = deepcopy(cfg)
        if not source_cfg.get("scene_cache_dir"):
            log_path = Path(str(cfg.navsim_log_path)).expanduser().resolve()
            if log_path.parent.name != "navsim_logs":
                raise ValueError("Set scene_cache_dir explicitly for nonstandard navsim_log_path layouts")
            source_cfg.scene_cache_dir = str(log_path.parents[2] / "exp" / "qwen35_drive_cache")
        return [("single", source_cfg)]
    if cfg.use_cache_without_dataset or cfg.cache_path or cfg.get("cache_paths"):
        raise ValueError("MOGO manifest training requires raw data: disable cache-only mode and cache paths")
    manifest_path = Path(str(manifest_path)).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[3] / manifest_path
    manifest = OmegaConf.load(manifest_path)
    entries = list(manifest.datasets)
    names = [str(entry.name) for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("MOGO manifest dataset names must be unique")
    selected_names = set(cfg.get("mogo_dataset_names") or [])
    if selected_names - set(names):
        raise ValueError(f"Unknown MOGO dataset names: {sorted(selected_names - set(names))}")
    if selected_names:
        entries = [entry for entry in entries if str(entry.name) in selected_names]
    if not entries:
        raise ValueError("MOGO manifest selects no datasets")

    sources = []
    seen_roots = set()
    for entry in entries:
        dataset_root = Path(str(entry.data_root)).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = manifest_path.parent / dataset_root
        dataset_root = (dataset_root / "dataset").resolve()
        if dataset_root in seen_roots:
            raise ValueError(f"Duplicate MOGO data root: {dataset_root}")
        seen_roots.add(dataset_root)
        log_path = dataset_root / "navsim_logs" / "train"
        sensor_path = dataset_root / "sensor_blobs" / "train"
        for directory in (log_path, sensor_path):
            if not directory.is_dir():
                raise FileNotFoundError(f"{entry.name}: missing data directory {directory}")
        available_logs = {path.stem for path in log_path.glob("*.pkl")}
        source_cfg = deepcopy(cfg)
        source_cfg.navsim_log_path = str(log_path)
        source_cfg.sensor_blobs_path = str(sensor_path)
        # An explicit global directory remains available for compatibility. By
        # default each manifest entry's cache_path stores its raw frame cache.
        # Keep cfg.cache_path unset: Dataset uses that separate top-level field
        # for precomputed model features, not raw frame packs.
        cache_dir = cfg.get("scene_cache_dir") or entry.get("cache_path")
        if cache_dir:
            cache_dir = Path(str(cache_dir)).expanduser()
            if not cache_dir.is_absolute() and not cfg.get("scene_cache_dir"):
                cache_dir = manifest_path.parent / cache_dir
        else:
            cache_dir = dataset_root.parent / "exp" / "qwen35_drive_cache"
        source_cfg.scene_cache_dir = str(cache_dir.resolve())
        source_cfg.train_logs = sorted(available_logs & set(cfg.train_logs))
        source_cfg.val_logs = sorted(available_logs & set(cfg.val_logs))
        if not source_cfg.train_logs and not source_cfg.val_logs:
            raise ValueError(f"{entry.name}: no logs match train_logs/val_logs; check log_split")
        if cfg.get("mogo_map") is not None:
            source_cfg.mogo_map = OmegaConf.merge(cfg.mogo_map, {
                "sqlite_path": entry.sqlite_path,
                "utm_zone": entry.utm_zone,
                "utm_south": entry.get("utm_south", False),
            })
        sources.append((str(entry.name), source_cfg))
    return sources


def _build_source_datasets(
    cfg: DictConfig, agent: AbstractAgent, index_paths=None,
) -> Tuple[Dataset, Dataset]:
    """Build the two log selections for one source, preserving its image root."""
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    # Bind preflight to the actual agent sensor selection, including view_list overrides.
    sensor_config = agent.get_sensor_config()
    for scene_filter in (train_scene_filter, val_scene_filter):
        scene_filter.required_camera_frames = camera_requirements(
            sensor_config, scene_filter.num_history_frames,
        )
    excluded_logs = set(cfg.get("excluded_logs") or [])
    for split, scene_filter in (("train", train_scene_filter), ("val", val_scene_filter)):
        excluded = sorted(set(scene_filter.log_names) & excluded_logs)
        if excluded:
            logger.warning("Excluding %d %s logs from %s: %s",
                           len(excluded), split, cfg.navsim_log_path, excluded)
            scene_filter.log_names = [name for name in scene_filter.log_names if name not in excluded_logs]

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    map_api_builder = build_mogo_map_api_builder_from_cfg(cfg)

    loading_mode = str(cfg.get("scene_loading", "eager"))
    if loading_mode not in ("eager", "indexed"):
        raise ValueError("scene_loading must be indexed or eager")
    if loading_mode == "indexed" and cfg.cache_path:
        raise ValueError("Indexed raw scenes require cache_path=null; model features are computed online")
    loaders = []
    for split_index, scene_filter in enumerate((train_scene_filter, val_scene_filter)):
        common = dict(
            sensor_blobs_path=sensor_blobs_path, data_path=data_path,
            scene_filter=scene_filter, sensor_config=sensor_config,
            map_api_builder=map_api_builder, load_image_path=True,
        )
        if loading_mode == "indexed":
            index_path = (Path(index_paths[split_index]) if index_paths is not None else
                          prepare_scene_index(
                              data_path, scene_filter, Path(cfg.scene_cache_dir),
                              workers=int(cfg.get("scene_cache_workers", 2)),
                          ))
            loaders.append(IndexedSceneLoader(**common, index_path=index_path))
        else:
            loaders.append(SceneLoader(**common, log_load_workers=int(cfg.get("log_load_workers", 1))))
    train_scene_loader, val_scene_loader = loaders

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


def _build_mogo_datasets(
    cfg: DictConfig, agent: AbstractAgent, *, prepared_indexes=None, read_only_indexes=False,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Combine online datasets with per-source paths and disjoint train/val tokens."""
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if (scene_filter.num_history_frames, scene_filter.num_future_frames, scene_filter.frame_stride) != (16, 50, 1):
        raise ValueError("Qwen-Drive IL requires mogo_qwen_drive: history=16, future=50, stride=1")
    if not cfg.train_logs or not cfg.val_logs:
        raise ValueError("train_logs and val_logs must explicitly select nonempty, separate log sets")
    if set(cfg.train_logs) & set(cfg.val_logs):
        raise ValueError("train_logs and val_logs overlap")
    train_parts, val_parts = [], []
    token_owners = {}
    for name, source_cfg in _dataset_source_configs(cfg):
        started = perf_counter()
        logger.info(
            "Loading MOGO source %s: train_logs=%d val_logs=%d mode=%s scene_cache_dir=%s",
            name, len(source_cfg.train_logs), len(source_cfg.val_logs),
            str(source_cfg.get("scene_loading", "eager")), source_cfg.scene_cache_dir,
        )
        if read_only_indexes and (prepared_indexes is None or name not in prepared_indexes):
            raise ValueError(f"Rank 0 did not publish raw scene indexes for {name}")
        source_indexes = prepared_indexes.get(name) if prepared_indexes is not None else None
        train_data, val_data = _build_source_datasets(source_cfg, agent, source_indexes)
        if prepared_indexes is not None and str(cfg.get("scene_loading", "eager")) == "indexed":
            prepared_indexes[name] = [str(data._scene_loader.index_path) for data in (train_data, val_data)]
        for split, dataset, parts in (("train", train_data, train_parts), ("val", val_data, val_parts)):
            for token in dataset._scene_loader.tokens:
                if token in token_owners:
                    raise ValueError(
                        f"Duplicate scene token {token}: {token_owners[token]} and {name}/{split}; "
                        "fix overlapping sources or train/val leakage in log_split"
                    )
                token_owners[token] = f"{name}/{split}"
            if len(dataset):
                parts.append(dataset)
        logger.info(
            "MOGO source %s: train_logs=%d val_logs=%d train_scenes=%d val_scenes=%d images=%s load_seconds=%.3f",
            name, len(source_cfg.train_logs), len(source_cfg.val_logs),
            len(train_data), len(val_data), source_cfg.sensor_blobs_path, perf_counter() - started,
        )
    if not train_parts or not val_parts:
        raise ValueError("No train/val scenes: check log split, data roots and 66-frame coverage")
    train_data = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_data = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    return train_data, val_data


def _physicalai_view_list(cfg: DictConfig) -> Sequence[str]:
    views = cfg.get("physicalai_view_list")
    if views is None:
        return ("cam_f0", "cam_l0", "cam_r0")
    views = [str(view) for view in views]
    if not views:
        raise ValueError("physicalai_view_list must contain at least one camera view")
    return views


def _build_physicalai_datasets(
    cfg: DictConfig, agent: AbstractAgent,
) -> Tuple[PhysicalAIDataset, PhysicalAIDataset]:
    root = cfg.get("physicalai_root")
    if not root:
        raise ValueError("physicalai_root must point to the PhysicalAI-AV dataset directory")
    max_samples = cfg.get("physicalai_max_samples_per_split")
    if max_samples in (None, "", "null"):
        max_samples = None
    else:
        max_samples = int(max_samples)
    requested_scan_workers = int(cfg.get("physicalai_scan_workers", 48))
    if requested_scan_workers < 1:
        raise ValueError("physicalai_scan_workers must be positive")
    world_size = max(1, int(os.getenv("WORLD_SIZE", "1")))
    scan_workers_per_rank = max(1, requested_scan_workers // world_size)
    common = dict(
        root=str(root),
        view_list=_physicalai_view_list(cfg),
        anchor_time_s=float(cfg.get("physicalai_anchor_time_s", 10.0)),
        max_samples=max_samples,
        shard_check_workers=scan_workers_per_rank,
        storage=str(cfg.get("physicalai_storage", "auto")),
    )
    return (
        PhysicalAIDataset(split="train", **common),
        PhysicalAIDataset(split="val", **common),
    )


def _normalize_dataset_mode(mode: Any) -> str:
    normalized = str(mode).lower()
    return "mixed" if normalized == "mix" else normalized


def build_datasets(
    cfg: DictConfig, agent: AbstractAgent, *, prepared_indexes=None, read_only_indexes=False,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Build MOGO, PhysicalAI-AV, or mixed Qwen datasets behind one batch contract."""
    mode = _normalize_dataset_mode(cfg.get("dataset_mode", "mogo"))
    if mode == "mogo":
        return _build_mogo_datasets(
            cfg, agent, prepared_indexes=prepared_indexes, read_only_indexes=read_only_indexes,
        )
    if mode == "physicalai":
        return _build_physicalai_datasets(cfg, agent)
    if mode != "mixed":
        raise ValueError("dataset_mode must be one of: mogo, physicalai, mixed")

    mogo_train, mogo_val = _build_mogo_datasets(
        cfg, agent, prepared_indexes=prepared_indexes, read_only_indexes=read_only_indexes,
    )
    public_train, public_val = _build_physicalai_datasets(cfg, agent)
    ratio = parse_mixed_ratio(str(cfg.get("physicalai_mixture_ratio", "proportional")))
    if ratio is None:
        return ConcatDataset((mogo_train, public_train)), ConcatDataset((mogo_val, public_val))
    return (
        MixedSourceDataset(mogo_train, public_train, ratio),
        # Keep validation representative of each source's official split; a
        # training mixture ratio must not duplicate either validation set.
        ConcatDataset((mogo_val, public_val)),
    )


def build_distributed_indexed_datasets(cfg, agent, rank, world_size, sync_group=None):
    """Only rank zero reads raw pickles; publish paths, never raw data objects."""
    status = [None]
    datasets = None
    if rank == 0:
        try:
            indexes = {}
            datasets = build_datasets(cfg, agent, prepared_indexes=indexes)
            status[0] = {"ok": True, "indexes": indexes}
        except Exception:
            status[0] = {"ok": False, "error": traceback.format_exc()}
    if world_size > 1:
        dist.broadcast_object_list(status, src=0, group=sync_group)
    if not status[0] or not status[0]["ok"]:
        raise RuntimeError(f"Rank 0 failed to prepare raw scene indexes: {status[0]}")
    local_error = None
    if rank != 0:
        try:
            datasets = build_datasets(
                cfg, agent, prepared_indexes=status[0]["indexes"], read_only_indexes=True,
            )
        except Exception:
            local_error = f"rank {rank}: {traceback.format_exc()}"
    errors = [local_error]
    if world_size > 1:
        errors = [None] * world_size
        dist.all_gather_object(errors, local_error, group=sync_group)
    if any(errors):
        raise RuntimeError("Failed to open shared raw scene indexes: " + str(errors))
    return datasets


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entrypoint for training the Qwen-Drive Planning Expert (IL)."""
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=int(cfg.distributed_timeout_seconds)),
        )
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info(
        "Training config: experiment=%s batch_per_device=%d world_size=%d "
        "grad_accum=%d effective_batch=%d lr=%g weight_decay=%g epochs=%d precision=%s",
        cfg.experiment_name, cfg.dataloader.params.batch_size, world_size,
        cfg.trainer.params.accumulate_grad_batches,
        cfg.dataloader.params.batch_size * world_size * cfg.trainer.params.accumulate_grad_batches,
        cfg.agent.lr, cfg.agent.weight_decay, cfg.trainer.params.max_epochs,
        cfg.trainer.params.precision,
    )
    logger.info(
        "Planning Expert initialization: %s; resume checkpoint: %s",
        cfg.agent.planner_path or "random", cfg.ckpt_path,
    )
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    dataset_mode = _normalize_dataset_mode(cfg.get("dataset_mode", "mogo"))
    if cfg.use_cache_without_dataset:
        if dataset_mode != "mogo" or cfg.get("mogo_manifest"):
            raise ValueError("MOGO manifest training requires use_cache_without_dataset=false")
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None or cfg.get("cache_paths", None)
        ), "cache_path or cache_paths must be provided when using cached data without building SceneLoader"
        with cache_index_sync_group(cfg, world_size) as cache_sync_group:
            train_data, val_data = build_distributed_cache_only_datasets(
                cfg=cfg,
                agent=agent,
                rank=rank,
                world_size=world_size,
                cache_sync_group=cache_sync_group,
            )
    elif (
        str(cfg.get("scene_loading", "eager")) == "indexed"
        and dataset_mode in ("mogo", "mixed")
    ):
        logger.info(
            "Using shared raw scene packs: directory=%s; rank 0 builds with %d workers; "
            "log_load_workers is inactive. Other ranks load only scene indexes.",
            cfg.scene_cache_dir or "per-source exp/qwen35_drive_cache", cfg.scene_cache_workers,
        )
        with cache_index_sync_group(cfg, world_size) as sync_group:
            train_data, val_data = build_distributed_indexed_datasets(
                cfg, agent, rank, world_size, sync_group,
            )
    else:
        logger.info(
            "Building %s datasets; DataLoader num_workers=%d.",
            dataset_mode, cfg.dataloader.params.num_workers,
        )
        train_data, val_data = build_datasets(cfg, agent)

    if cfg.get("prepare_scene_cache_only", False):
        if dataset_mode != "mogo":
            raise ValueError("prepare_scene_cache_only is available only for dataset_mode=mogo")
        if str(cfg.get("scene_loading", "eager")) != "indexed":
            raise ValueError("prepare_scene_cache_only requires scene_loading=indexed")
        logger.info("Raw scene cache ready: train=%d val=%d. Skipping model loading/training.",
                    len(train_data), len(val_data))
        return

    # Rare-scene oversampling applies only to the training set; val stays intact
    # so val/loss_epoch and checkpoint selection are not biased. Applied after the
    # cache index is built so the index spec signature is unchanged.
    if dataset_mode != "mogo" and int(cfg.get("train_oversample_factor", 1)) > 1:
        raise ValueError(
            "train_oversample_factor currently supports dataset_mode=mogo only; "
            "set it to 1 for PhysicalAI or mixed training"
        )
    train_data, added_oversample_samples = apply_qwen_train_oversample(
        dataset=train_data,
        tokens_path=str(cfg.get("train_oversample_tokens_path", "") or ""),
        factor=int(cfg.get("train_oversample_factor", 1)),
    )
    if added_oversample_samples:
        logger.info("Qwen training set includes %d added oversampled sample(s)", added_oversample_samples)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    agent.initialize()
    logger.info("Trainable parameters: %d", sum(p.numel() for p in agent.parameters() if p.requires_grad))
    lightning_module = QwenDriveLightningModule(agent=agent)
    logger.info("Building Trainer")
    trainer = pl.Trainer(
        **cfg.trainer.params,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss_epoch",
                mode="min",
                save_top_k=5,
                every_n_epochs=1,
            )
        ],
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    main()
