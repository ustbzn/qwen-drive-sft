from contextlib import contextmanager
from datetime import timedelta
import logging
from pathlib import Path
import traceback
from typing import Iterator, Optional, Sequence, Tuple

from omegaconf import DictConfig
import torch
import torch.distributed as dist

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.cache_dataset_index import (
    get_cache_dataset_index_path,
    load_cache_dataset_index,
)
from navsim.planning.training.dataset import (
    build_cache_only_dataset,
    load_or_build_cache_only_dataset_index,
    make_cache_only_dataset_index_spec,
)


logger = logging.getLogger(__name__)


@contextmanager
def cache_index_sync_group(
    cfg: DictConfig,
    world_size: int,
) -> Iterator[Optional[dist.ProcessGroup]]:
    """Create the long-timeout Gloo group used while rank 0 prepares cache indexes."""
    group = None
    if world_size > 1 and str(cfg.get("cache_index_dir", "") or ""):
        timeout_seconds = int(
            cfg.get("cache_index_timeout_seconds", cfg.distributed_timeout_seconds)
        )
        if timeout_seconds <= 0:
            raise ValueError("cache_index_timeout_seconds must be positive")
        logger.info(
            "Creating cache-index synchronization group with %d-second timeout",
            timeout_seconds,
        )
        group = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=timeout_seconds),
        )

    try:
        yield group
    finally:
        if group is not None:
            dist.destroy_process_group(group)


def build_distributed_cache_only_datasets(
    cfg: DictConfig,
    agent: AbstractAgent,
    rank: int,
    world_size: int,
    cache_sync_group: Optional[dist.ProcessGroup] = None,
    train_tokens: Optional[Sequence[str]] = None,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Build cache indexes on rank 0, then load the same sample order on every rank."""
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
        "train": {
            **common_kwargs,
            "log_names": cfg.train_logs,
            "tokens": None if train_tokens is None else list(train_tokens),
        },
        "val": {
            **common_kwargs,
            "log_names": cfg.val_logs,
            "tokens": None,
        },
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
            tokens=dataset_kwargs["tokens"],
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
