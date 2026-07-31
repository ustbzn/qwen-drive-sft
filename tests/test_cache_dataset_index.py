from datetime import timedelta
import logging
from pathlib import Path
import threading
import time

from omegaconf import OmegaConf
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import navsim.planning.training.dataset as dataset_module

from navsim.planning.script.run_training_recogdrive import (
    build_distributed_cache_only_datasets,
)
from navsim.planning.training.cache_dataset_index import (
    CacheDatasetIndexError,
    get_cache_dataset_index_path,
    load_cache_dataset_index,
)
from navsim.planning.training.dataset import (
    CacheOnlyDataset,
    MultiCacheOnlyDataset,
    build_cache_only_dataset,
    dump_feature_target_to_pickle,
    load_or_build_cache_only_dataset_index,
    make_cache_only_dataset_index_spec,
)


class _TestBuilder:
    def __init__(self, name: str):
        self._name = name

    def get_unique_name(self) -> str:
        return self._name


class _TestAgent:
    use_vlm = False

    def __init__(self):
        self._feature_builders = [_TestBuilder("feature")]
        self._target_builders = [_TestBuilder("target")]

    def get_feature_builders(self):
        return self._feature_builders

    def get_target_builders(self):
        return self._target_builders


def _write_sample(cache_root: Path, log_name: str, token: str, value: int) -> None:
    token_path = cache_root / log_name / token
    token_path.mkdir(parents=True)
    dump_feature_target_to_pickle(
        token_path / "feature.gz",
        {"feature": torch.tensor([value])},
    )
    dump_feature_target_to_pickle(
        token_path / "target.gz",
        {"target": torch.tensor([value])},
    )


def _dataset_inputs(cache_roots, log_names):
    feature_builders = [_TestBuilder("feature")]
    target_builders = [_TestBuilder("target")]
    return {
        "cache_path": str(cache_roots[0]),
        "cache_paths": [str(path) for path in cache_roots] if len(cache_roots) > 1 else None,
        "feature_builders": feature_builders,
        "target_builders": target_builders,
        "log_names": list(log_names),
    }


def _fail_cache_scan(*args, **kwargs):
    raise AssertionError("nonzero rank must not scan cache directories")


def _distributed_index_worker(rank, world_size, init_file, cache_roots, index_dir):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    cache_sync_group = None
    try:
        cache_sync_group = dist.new_group(
            backend="gloo",
            timeout=timedelta(seconds=60),
        )
        if rank != 0:
            MultiCacheOnlyDataset._load_valid_cache_samples = staticmethod(_fail_cache_scan)

        cfg = OmegaConf.create(
            {
                "cache_path": cache_roots[0],
                "cache_paths": list(cache_roots),
                "train_logs": ["train_log"],
                "val_logs": ["val_log"],
                "cache_index_dir": index_dir,
                "cache_index_workers": 4,
                "rebuild_cache_index": False,
            }
        )
        train_data, val_data = build_distributed_cache_only_datasets(
            cfg=cfg,
            agent=_TestAgent(),
            rank=rank,
            world_size=world_size,
            cache_sync_group=cache_sync_group,
        )
        dist.destroy_process_group(cache_sync_group)
        cache_sync_group = None

        assert len(train_data) == 6
        assert len(val_data) == 2
        ordered_tokens = list(train_data.tokens)
        gathered_orders = [None] * world_size
        dist.all_gather_object(gathered_orders, ordered_tokens)
        if rank == 0:
            assert gathered_orders[0] == gathered_orders[1]
        sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        local_indices = list(sampler)
        gathered_indices = [None] * world_size
        dist.all_gather_object(gathered_indices, local_indices)
        local_tokens = [ordered_tokens[index] for index in local_indices]
        gathered_tokens = [None] * world_size
        dist.all_gather_object(gathered_tokens, local_tokens)
        if rank == 0:
            assert set(gathered_indices[0]).isdisjoint(gathered_indices[1])
            assert set(gathered_indices[0] + gathered_indices[1]) == set(range(6))
            assert set(gathered_tokens[0]).isdisjoint(gathered_tokens[1])
            assert set(gathered_tokens[0] + gathered_tokens[1]) == set(ordered_tokens)
    finally:
        if cache_sync_group is not None:
            dist.destroy_process_group(cache_sync_group)
        dist.destroy_process_group()


def test_multi_cache_index_round_trip_skips_rescan(tmp_path, monkeypatch):
    cache_roots = [tmp_path / "cache_0", tmp_path / "cache_1"]
    _write_sample(cache_roots[0], "log", "token_0", 0)
    _write_sample(cache_roots[0], "log", "token_1", 1)
    _write_sample(cache_roots[1], "log", "token_2", 2)
    inputs = _dataset_inputs(cache_roots, ["log"])
    spec = make_cache_only_dataset_index_spec(**inputs)
    index_path = get_cache_dataset_index_path(tmp_path / "indexes", "train", spec)

    index, rebuilt = load_or_build_cache_only_dataset_index(
        index_path=index_path,
        spec=spec,
        feature_builders=inputs["feature_builders"],
        target_builders=inputs["target_builders"],
    )
    assert rebuilt is True
    assert index_path.is_file()

    loaded_index = load_cache_dataset_index(index_path, expected_spec=spec)
    assert loaded_index.samples == index.samples
    monkeypatch.setattr(
        MultiCacheOnlyDataset,
        "_load_valid_cache_samples",
        staticmethod(_fail_cache_scan),
    )
    dataset = build_cache_only_dataset(**inputs, cache_index=loaded_index)

    assert len(dataset) == 3
    assert dataset.samples == loaded_index.samples
    features, targets, token = dataset[2]
    assert token == "token_2"
    assert features["feature"].item() == 2
    assert targets["target"].item() == 2


def test_single_cache_index_round_trip_skips_rescan(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    _write_sample(cache_root, "log", "token", 3)
    inputs = _dataset_inputs([cache_root], ["log"])
    spec = make_cache_only_dataset_index_spec(**inputs)
    index_path = get_cache_dataset_index_path(tmp_path / "indexes", "train", spec)
    index, _ = load_or_build_cache_only_dataset_index(
        index_path=index_path,
        spec=spec,
        feature_builders=inputs["feature_builders"],
        target_builders=inputs["target_builders"],
    )

    monkeypatch.setattr(CacheOnlyDataset, "_load_valid_caches", staticmethod(_fail_cache_scan))
    dataset = build_cache_only_dataset(**inputs, cache_index=index)
    assert len(dataset) == 1
    assert dataset[0][2] == "token"


def test_parallel_cache_scan_preserves_order_and_coverage(
    tmp_path, monkeypatch, caplog
):
    cache_root = tmp_path / "cache"
    log_names = [f"log_{index}" for index in range(8)]
    expected_tokens = []

    for index, log_name in enumerate(log_names):
        valid_token = f"valid_{index}"
        expected_tokens.append(valid_token)
        _write_sample(cache_root, log_name, valid_token, index)

        invalid_path = cache_root / log_name / f"invalid_{index}"
        invalid_path.mkdir(parents=True)
        dump_feature_target_to_pickle(
            invalid_path / "feature.gz",
            {"feature": torch.tensor([index])},
        )

    original_scan = dataset_module._scan_valid_cache_log
    worker_thread_ids = set()
    thread_ids_lock = threading.Lock()

    def delayed_scan(log_path, required_names, tokens):
        log_index = int(log_path.name.rsplit("_", 1)[1])
        with thread_ids_lock:
            worker_thread_ids.add(threading.get_ident())
        time.sleep(0.005 * (len(log_names) - log_index))
        return original_scan(log_path, required_names, tokens)

    monkeypatch.setattr(dataset_module, "_scan_valid_cache_log", delayed_scan)
    caplog.set_level(logging.WARNING)
    inputs = _dataset_inputs([cache_root], log_names)
    spec = make_cache_only_dataset_index_spec(**inputs)
    index_path = get_cache_dataset_index_path(tmp_path / "indexes", "train", spec)

    index, rebuilt = load_or_build_cache_only_dataset_index(
        index_path=index_path,
        spec=spec,
        feature_builders=inputs["feature_builders"],
        target_builders=inputs["target_builders"],
        cache_index_workers=4,
    )

    assert rebuilt is True
    assert len(worker_thread_ids) > 1
    assert [sample.log_name for sample in index.samples] == log_names
    assert [sample.token for sample in index.samples] == expected_tokens
    assert "valid=8/16" in caplog.text
    assert "target.gz=8" in caplog.text


def test_cache_index_signature_covers_split_and_required_features(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    train_inputs = _dataset_inputs([cache_root], ["train_log"])
    val_inputs = _dataset_inputs([cache_root], ["val_log"])
    train_spec = make_cache_only_dataset_index_spec(**train_inputs)
    val_spec = make_cache_only_dataset_index_spec(**val_inputs)
    feature_spec = make_cache_only_dataset_index_spec(
        **{**train_inputs, "feature_builders": [_TestBuilder("other_feature")]}
    )

    assert train_spec.signature != val_spec.signature
    assert train_spec.signature != feature_spec.signature
    assert get_cache_dataset_index_path(tmp_path, "train", train_spec) != get_cache_dataset_index_path(
        tmp_path, "train", feature_spec
    )


def test_corrupt_cache_index_is_rebuilt(tmp_path):
    cache_root = tmp_path / "cache"
    _write_sample(cache_root, "log", "token", 4)
    inputs = _dataset_inputs([cache_root], ["log"])
    spec = make_cache_only_dataset_index_spec(**inputs)
    index_path = get_cache_dataset_index_path(tmp_path / "indexes", "train", spec)
    index_path.parent.mkdir()
    index_path.write_bytes(b"not a valid gzip index")

    index, rebuilt = load_or_build_cache_only_dataset_index(
        index_path=index_path,
        spec=spec,
        feature_builders=inputs["feature_builders"],
        target_builders=inputs["target_builders"],
    )

    assert rebuilt is True
    assert len(index.samples) == 1
    assert load_cache_dataset_index(index_path, expected_spec=spec).samples == index.samples


def test_cache_index_rejects_different_spec(tmp_path):
    cache_root = tmp_path / "cache"
    _write_sample(cache_root, "train_log", "token", 5)
    train_inputs = _dataset_inputs([cache_root], ["train_log"])
    val_inputs = _dataset_inputs([cache_root], ["val_log"])
    train_spec = make_cache_only_dataset_index_spec(**train_inputs)
    val_spec = make_cache_only_dataset_index_spec(**val_inputs)
    index_path = get_cache_dataset_index_path(tmp_path / "indexes", "train", train_spec)
    load_or_build_cache_only_dataset_index(
        index_path=index_path,
        spec=train_spec,
        feature_builders=train_inputs["feature_builders"],
        target_builders=train_inputs["target_builders"],
    )

    with pytest.raises(CacheDatasetIndexError, match="signature mismatch"):
        load_cache_dataset_index(index_path, expected_spec=val_spec)


def test_two_ranks_share_full_index_before_sampler_partition(tmp_path):
    cache_roots = [tmp_path / "cache_0", tmp_path / "cache_1"]
    for cache_index, cache_root in enumerate(cache_roots):
        for sample_index in range(3):
            _write_sample(
                cache_root,
                "train_log",
                f"train_{cache_index}_{sample_index}",
                cache_index * 10 + sample_index,
            )
        _write_sample(cache_root, "val_log", f"val_{cache_index}", cache_index)

    init_file = tmp_path / "distributed_init"
    mp.spawn(
        _distributed_index_worker,
        args=(
            2,
            str(init_file),
            tuple(str(path) for path in cache_roots),
            str(tmp_path / "shared_indexes"),
        ),
        nprocs=2,
        join=True,
    )
