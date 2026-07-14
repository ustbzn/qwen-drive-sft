from typing import Tuple
from pathlib import Path
import logging
import os
import warnings
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.distributed as dist
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset, build_cache_only_dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule, AgentLightningDiT
from navsim.agents.recogdrive.hdmap_vector_features import collate_hdmap_vector_features
from navsim.agents.recogdrive.front_image_features import collate_front_image_features
from navsim.planning.scenario_builder.mogo_map_api_builder import build_mogo_map_api_builder_from_cfg
import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List, Dict

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"



def custom_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    features_list, targets_list, tokens_list = zip(*batch)

    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()

    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()


    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
    }

    # VLM 隐状态（变长，需 pad）。use_vlm=False 时 agent 只注册 slim builder，dataset 不会加载这两个键，
    # planner/forward_grpo 也不引用，故此处成对判定：仅当 first 与 last 同时存在才 pad；
    # 缺失（或脏 cache 半缺失）一律整体跳过，避免 KeyError。
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

    # 障碍物 obj 字段（固定形状，直接 stack）。GRPO 时 forward_grpo 会按 G 复制这些字段。
    if "obj_features" in features_list[0]:
        features["obj_features"] = torch.stack([f["obj_features"] for f in features_list], dim=0).cpu()
        features["obj_type"] = torch.stack([f["obj_type"] for f in features_list], dim=0).cpu()
        features["obj_valid_mask"] = torch.stack([f["obj_valid_mask"] for f in features_list], dim=0).cpu()
        # temporal 编码器的差分序列与动静标记（mlp 缓存无这两个键，用内层 if 保护）
        if "obj_diff_features" in features_list[0]:
            features["obj_diff_features"] = torch.stack([f["obj_diff_features"] for f in features_list], dim=0).cpu()
            features["obj_is_dynamic"] = torch.stack([f["obj_is_dynamic"] for f in features_list], dim=0).cpu()

    # 前视图像字段（固定形状 uint8，直接 stack）。GRPO 时 forward_grpo 会按 G 复制这些字段。
    collate_front_image_features(features, features_list)

    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list


def load_token_whitelist(path_str: str) -> List[str]:
    """
    Loads a token whitelist from a text file.
    Each non-empty line is treated as one token.
    """
    if not path_str:
        return []

    whitelist_path = Path(path_str)
    if not whitelist_path.is_file():
        raise FileNotFoundError(f"Token whitelist file not found: {whitelist_path}")

    with whitelist_path.open("r") as fp:
        tokens = [line.strip() for line in fp if line.strip()]
    return tokens


def build_datasets(
    cfg: DictConfig, agent: AbstractAgent, train_tokens: List[str] | None
) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :param train_tokens: optional token whitelist for training data only
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs
    if train_tokens is not None:
        train_scene_filter.tokens = train_tokens

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

    # PyTorch 2.4+: avoids noisy barrier() UserWarning; keeps logs readable when stderr is redirected.
    warnings.filterwarnings(
        "ignore",
        message=r".*barrier\(\): using the device under current context\..*",
        category=UserWarning,
    )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        device_id=local_rank,
    )
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    train_token_whitelist_path = str(cfg.get("train_token_whitelist_path", "") or "")
    train_tokens = load_token_whitelist(train_token_whitelist_path) if train_token_whitelist_path else None
    if train_token_whitelist_path:
        logger.info("Loaded %d training whitelist tokens from %s", len(train_tokens), train_token_whitelist_path)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningDiT(
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
        train_data = build_cache_only_dataset(
            cache_path=cfg.cache_path,
            cache_paths=cfg.get("cache_paths", None),
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            tokens=train_tokens,
            drop_hidden_state=not getattr(agent, "use_vlm", True),
        )
        val_data = build_cache_only_dataset(
            cache_path=cfg.cache_path,
            cache_paths=cfg.get("cache_paths", None),
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            drop_hidden_state=not getattr(agent, "use_vlm", True),
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent, train_tokens)

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
