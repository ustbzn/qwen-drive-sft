from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import os
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.cache_dataset_index import (
    get_cache_dataset_index_path,
    load_cache_dataset_index,
)
from navsim.planning.training.dataset import (
    Dataset,
    build_cache_only_dataset,
    load_or_build_cache_only_dataset_index,
    make_cache_only_dataset_index_spec,
)
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.agents.recogdrive.hdmap_vector_features import collate_hdmap_vector_features
from navsim.agents.recogdrive.front_image_features import collate_front_image_features
from navsim.planning.scenario_builder.mogo_map_api_builder import build_mogo_map_api_builder_from_cfg
import torch
import torch.nn.utils.rnn as rnn_utils

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"




def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature
    }

    # VLM 隐状态（变长，需 pad）。use_vlm=False 时 agent 只注册 slim builder，dataset 不会加载这两个键，
    # planner 也不引用，故此处成对判定：仅当 first 与 last 同时存在才 pad；缺失（或脏 cache 半缺失）
    # 一律整体跳过，避免 KeyError。
    f0 = features_list[0]
    if 'first_hidden_state' in f0 and 'last_hidden_state' in f0:
        features['first_hidden_state'] = rnn_utils.pad_sequence(
            [features['first_hidden_state'] for features in features_list],
            batch_first=True,
            padding_value=0.0
        ).clone().detach()
        features['last_hidden_state'] = rnn_utils.pad_sequence(
            [features['last_hidden_state'] for features in features_list],
            batch_first=True,
            padding_value=0.0
        ).clone().detach()

    # hdmap vector 字段（变长，需要 collate）
    collate_hdmap_vector_features(features, features_list)

    # 障碍物 obj 字段（固定形状，直接 stack，无需 pad_sequence）。仅当缓存里有 obj 时才补。
    if "obj_features" in features_list[0]:
        features["obj_features"] = torch.stack([f["obj_features"] for f in features_list], dim=0).cpu()
        features["obj_type"] = torch.stack([f["obj_type"] for f in features_list], dim=0).cpu()
        features["obj_valid_mask"] = torch.stack([f["obj_valid_mask"] for f in features_list], dim=0).cpu()
        # temporal 编码器的差分序列与动静标记（mlp 缓存无这两个键，用内层 if 保护）
        if "obj_diff_features" in features_list[0]:
            features["obj_diff_features"] = torch.stack([f["obj_diff_features"] for f in features_list], dim=0).cpu()
            features["obj_is_dynamic"] = torch.stack([f["obj_is_dynamic"] for f in features_list], dim=0).cpu()

    # 前视图像字段（固定形状 uint8，直接 stack）。仅当缓存里带 front_image 时补（零影响保护）。
    collate_front_image_features(features, features_list)

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list


def build_distributed_cache_only_datasets(
    cfg: DictConfig,
    agent: AbstractAgent,
    rank: int,
    world_size: int,
    cache_sync_group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Build one full cache index on rank 0 and load it on every rank."""
    feature_builders = agent.get_feature_builders()
    target_builders = agent.get_target_builders()
    common_kwargs = {
        "cache_path": cfg.cache_path,
        "cache_paths": cfg.get("cache_paths", None),
        "feature_builders": feature_builders,
        "target_builders": target_builders,
        "drop_hidden_state": not getattr(agent, "use_vlm", True),
    }
    split_kwargs = {
        "train": {**common_kwargs, "log_names": cfg.train_logs},
        "val": {**common_kwargs, "log_names": cfg.val_logs},
    }

    index_dir_value = str(cfg.get("cache_index_dir", "") or "")
    if not index_dir_value:
        logger.warning("Cache index sharing is disabled; every rank will scan the cache")
        return (
            build_cache_only_dataset(**split_kwargs["train"]),
            build_cache_only_dataset(**split_kwargs["val"]),
        )

    cache_index_workers = int(cfg.get("cache_index_workers", 1))
    if cache_index_workers <= 0:
        raise ValueError("cache_index_workers must be positive")
    if rank == 0:
        logger.info(
            "Cache-index scans are configured to use up to %d workers",
            cache_index_workers,
        )

    requests = {}
    for split_name, dataset_kwargs in split_kwargs.items():
        spec = make_cache_only_dataset_index_spec(
            cache_path=dataset_kwargs["cache_path"],
            cache_paths=dataset_kwargs["cache_paths"],
            feature_builders=feature_builders,
            target_builders=target_builders,
            log_names=dataset_kwargs["log_names"],
        )
        requests[split_name] = {
            "dataset_kwargs": dataset_kwargs,
            "spec": spec,
            "index_path": get_cache_dataset_index_path(
                index_dir=Path(index_dir_value),
                split_name=split_name,
                spec=spec,
            ),
        }

    sync_payload = [None]
    if rank == 0:
        try:
            published_indexes = {}
            for split_name, request in requests.items():
                index, rebuilt = load_or_build_cache_only_dataset_index(
                    index_path=request["index_path"],
                    spec=request["spec"],
                    feature_builders=feature_builders,
                    target_builders=target_builders,
                    force_rebuild=bool(cfg.get("rebuild_cache_index", False)),
                    cache_index_workers=cache_index_workers,
                )
                published_indexes[split_name] = {
                    "path": str(request["index_path"]),
                    "signature": request["spec"].signature,
                    "sample_count": len(index.samples),
                }
                logger.info(
                    "%s %s cache index with %d samples: %s",
                    "Built" if rebuilt else "Loaded",
                    split_name,
                    len(index.samples),
                    request["index_path"],
                )
            del index
            sync_payload[0] = {"ok": True, "indexes": published_indexes}
        except Exception:
            sync_payload[0] = {"ok": False, "error": traceback.format_exc()}

    if world_size > 1:
        # Only small status metadata is broadcast; sample records stay on shared storage.
        dist.broadcast_object_list(sync_payload, src=0, group=cache_sync_group)

    status = sync_payload[0]
    if not status or not status.get("ok", False):
        error = "missing rank-0 status" if not status else status.get("error", "unknown error")
        raise RuntimeError(f"Rank 0 failed to prepare cache dataset indexes:\n{error}")

    datasets = {}
    local_error = None
    try:
        for split_name, request in requests.items():
            published = status["indexes"][split_name]
            if published["signature"] != request["spec"].signature:
                raise RuntimeError(
                    f"{split_name} cache index signature differs between rank 0 and rank {rank}"
                )
            index = load_cache_dataset_index(
                path=Path(published["path"]),
                expected_spec=request["spec"],
            )
            if len(index.samples) != published["sample_count"]:
                raise RuntimeError(
                    f"{split_name} cache index sample count changed after publication"
                )
            datasets[split_name] = build_cache_only_dataset(
                **request["dataset_kwargs"],
                cache_index=index,
            )
            del index
    except Exception:
        local_error = f"rank {rank}:\n{traceback.format_exc()}"

    if world_size > 1:
        rank_errors = [None] * world_size
        dist.all_gather_object(rank_errors, local_error, group=cache_sync_group)
    else:
        rank_errors = [local_error]
    errors = [error for error in rank_errors if error]
    if errors:
        raise RuntimeError("Failed to load shared cache indexes:\n" + "\n".join(errors))

    return datasets["train"], datasets["val"]


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
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

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    map_api_builder = build_mogo_map_api_builder_from_cfg(cfg)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        map_api_builder=map_api_builder,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        map_api_builder=map_api_builder,
    )

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


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=int(cfg.distributed_timeout_seconds)),
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None or cfg.get("cache_paths", None)
        ), "cache_path or cache_paths must be provided when using cached data without building SceneLoader"
        cache_sync_group = None
        if world_size > 1 and str(cfg.get("cache_index_dir", "") or ""):
            cache_index_timeout_seconds = int(
                cfg.get("cache_index_timeout_seconds", cfg.distributed_timeout_seconds)
            )
            if cache_index_timeout_seconds <= 0:
                raise ValueError("cache_index_timeout_seconds must be positive")
            logger.info(
                "Creating cache-index synchronization group with %d-second timeout",
                cache_index_timeout_seconds,
            )
            cache_sync_group = dist.new_group(
                backend="gloo",
                timeout=timedelta(seconds=cache_index_timeout_seconds),
            )
        try:
            train_data, val_data = build_distributed_cache_only_datasets(
                cfg=cfg,
                agent=agent,
                rank=rank,
                world_size=world_size,
                cache_sync_group=cache_sync_group,
            )
        finally:
            if cache_sync_group is not None:
                dist.destroy_process_group(cache_sync_group)
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, collate_fn=custom_collate_fn,  **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, collate_fn=custom_collate_fn, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/loss_epoch",mode='min', save_top_k=5,every_n_epochs=1)])

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
