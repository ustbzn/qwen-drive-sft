"""MOGO 多数据源在线加载、场景中心与 Qwen-Drive 轨迹时序契约。"""

import os
import pickle
import subprocess
from contextlib import closing
from threading import Event
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import ConcatDataset

from navsim.agents.qwen_drive_planner.qwen_drive_agent import QwenDriveAgent
from navsim.agents.qwen_drive_planner.qwen_drive_config import FULL_VIEW_LIST, QwenDrivePlannerConfig
from navsim.agents.qwen_drive_planner.qwen_drive_features import QwenDriveTrajectoryTargetBuilder
from navsim.common.dataclasses import SceneFilter
from navsim.common import dataloader as scene_dataloader
from navsim.common.dataloader import SceneLoader, filter_scenes
from navsim.planning.script.run_training_qwen_drive_il import (
    _dataset_source_configs, build_datasets, custom_collate_fn,
)


ROOT = Path(__file__).resolve().parents[1]


def write_log(data_root, log_name, token_prefix=None, count=76):
    log_dir = data_root / "dataset/navsim_logs/train"
    sensor_dir = data_root / "dataset/sensor_blobs/train"
    log_dir.mkdir(parents=True, exist_ok=True)
    sensor_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(sensor_dir / "image.png")
    frames = [{
        "token": f"{token_prefix or log_name}-{index}",
        "log_name": log_name,
        "scene_token": log_name,
        "map_location": "las_vegas",
        "roadblock_ids": ["route"],
        "timestamp": 1000000 + index * 100000,
        "ego2global_translation": [index * 0.1, 0.0, 0.0],
        "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        "ego_dynamic_state": [1.0, 0.0, 0.0, 0.0],
        "driving_command": [0, 1, 0, 0],
        "lidar_path": "",
        "anns": {
            "gt_boxes": np.empty((0, 7)), "gt_names": [], "gt_velocity_3d": [],
            "instance_tokens": [], "track_tokens": [],
        },
        "cams": {view.upper(): {
            "data_path": "image.png", "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": np.zeros(3), "cam_intrinsic": np.eye(3),
            "distortion": np.zeros(5),
        } for view in FULL_VIEW_LIST},
    } for index in range(count)]
    with (log_dir / f"{log_name}.pkl").open("wb") as stream:
        pickle.dump(frames, stream)
    return frames


@pytest.fixture(params=["eager", "indexed"])
def multi_config(tmp_path, request):
    entries = []
    for name in ("city_a", "city_b"):
        data_root = tmp_path / name
        write_log(data_root, f"{name}_train")
        write_log(data_root, f"{name}_val")
        entries.append({
            "name": name, "data_root": str(data_root),
            "cache_path": str(data_root / "exp/qwen35_drive_cache"),
        })
    manifest_path = tmp_path / "manifest.yaml"
    OmegaConf.save(OmegaConf.create({"datasets": entries}), manifest_path)
    return OmegaConf.create({
        "mogo_manifest": str(manifest_path), "mogo_dataset_names": None,
        "train_logs": ["city_a_train", "city_b_train"],
        "val_logs": ["city_a_val", "city_b_val"],
        "navsim_log_path": None, "sensor_blobs_path": None,
        "cache_path": None, "cache_paths": None, "force_cache_computation": False,
        "use_cache_without_dataset": False, "log_load_workers": 2,
        "scene_loading": request.param, "scene_cache_dir": None,
        "scene_cache_workers": 2,
        "train_test_split": {"scene_filter": {
            "_target_": "navsim.common.dataclasses.SceneFilter", "_convert_": "all",
            "num_history_frames": 16, "num_future_frames": 50,
            "frame_stride": 1, "frame_interval": 5, "has_route": True,
            "timestamp_interval_s": 0.1, "max_timestamp_error_s": 0.075,
            "max_scenes": None, "log_names": None, "tokens": None,
        }},
    })


@pytest.mark.parametrize("count", [66, 76, 301])
def test_dense_and_sparse_filters_select_same_centers(tmp_path, count):
    write_log(tmp_path, "log", count=count)
    shared = dict(frame_interval=5, has_route=True)
    dense = SceneFilter(num_history_frames=16, num_future_frames=50, frame_stride=1, **shared)
    sparse = SceneFilter(num_history_frames=4, num_future_frames=10, frame_stride=5, **shared)
    assert dense.num_raw_frames == sparse.num_raw_frames == 66
    directory = tmp_path / "dataset/navsim_logs/train"
    dense_scenes = filter_scenes(directory, dense)
    sparse_scenes = filter_scenes(directory, sparse)
    assert dense_scenes.keys() == sparse_scenes.keys()
    for token, dense_frames in dense_scenes.items():
        sparse_frames = sparse_scenes[token]
        assert dense_frames[15]["token"] == sparse_frames[3]["token"] == token
        assert [frame["token"] for frame in dense_frames[:16:5]] == [
            frame["token"] for frame in sparse_frames[:4]
        ]
        assert [frame["token"] for frame in dense_frames[20::5]] == [
            frame["token"] for frame in sparse_frames[4:]
        ]


def test_manifest_combines_raw_data_without_cache_or_image_root_mixup(multi_config):
    train, val = build_datasets(multi_config, QwenDriveAgent())
    assert isinstance(train, ConcatDataset) and isinstance(val, ConcatDataset)
    assert len(train) == len(val) == 6
    first, second = train[0], train[3]
    assert "city_a/dataset" in first[0]["view_frame_paths"][0][0]
    assert "city_b/dataset" in second[0]["view_frame_paths"][0][0]
    features, targets, _ = custom_collate_fn([first, second])
    assert features["history"].shape == (2, 16, 3)
    assert targets["trajectory"].shape == (2, 50, 3)
    assert len(features["view_frame_paths"]) == 2
    assert all(len(views) == 3 and all(len(images) == 4 for images in views)
               for views in features["view_frame_paths"])
    torch.testing.assert_close(targets["trajectory"][0, :, 0], torch.arange(1, 51) * 0.1)
    if multi_config.scene_loading == "indexed":
        manifest = OmegaConf.load(multi_config.mogo_manifest)
        for part, entry in zip(train.datasets, manifest.datasets):
            assert part._scene_loader.index_path.is_relative_to(Path(entry.cache_path))


def test_manifest_subset_and_per_source_scene_limit(multi_config):
    multi_config.mogo_dataset_names = ["city_b"]
    multi_config.train_test_split.scene_filter.max_scenes = 1
    train, val = build_datasets(multi_config, QwenDriveAgent())
    assert len(train) == len(val) == 1
    assert "city_b/dataset" in train[0][0]["view_frame_paths"][0][0]


@pytest.mark.parametrize("single_source", [False, True])
def test_excluded_logs_are_not_read_or_sampled(multi_config, single_source, caplog):
    config = _dataset_source_configs(multi_config)[0][1] if single_source else multi_config
    if single_source:
        config.mogo_manifest = None
    root = Path(multi_config.mogo_manifest).parent / "city_a"
    excluded = ["broken_train", "broken_val"]
    for name in excluded:
        # Exclusion must happen before raw reads, in eager and indexed modes alike.
        (root / "dataset/navsim_logs/train" / f"{name}.pkl").write_bytes(b"invalid pickle")
    config.train_logs.append(excluded[0])
    config.val_logs.append(excluded[1])
    config.excluded_logs = excluded

    train, val = build_datasets(config, QwenDriveAgent())
    expected = 3 if single_source else 6
    assert len(train) == len(val) == expected
    for dataset in (train, val):
        parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
        for part in parts:
            assert not set(excluded) & set(part._scene_loader.token_to_log_file.values())
        assert len(dataset[0][0]["view_frame_paths"]) == 3
    assert all(name in caplog.text for name in excluded)


def test_source_may_have_no_validation_logs(multi_config):
    multi_config.val_logs = ["city_a_val"]
    train, val = build_datasets(multi_config, QwenDriveAgent())
    assert len(train) == 6 and len(val) == 3


def test_single_source_mode_remains_available(multi_config):
    source = _dataset_source_configs(multi_config)[0][1]
    source.mogo_manifest = None
    train, val = build_datasets(source, QwenDriveAgent())
    assert len(train) == len(val) == 3
    assert not isinstance(train, ConcatDataset)


@pytest.mark.parametrize("setting, value, message", [
    ("mogo_dataset_names", ["missing"], "Unknown MOGO"),
    ("cache_path", "/tmp/old-cache", "requires raw data"),
    ("use_cache_without_dataset", True, "requires raw data"),
    ("train_logs", ["missing"], "no logs match"),
])
def test_manifest_rejects_invalid_selection(multi_config, setting, value, message):
    multi_config[setting] = value
    if setting == "train_logs":
        multi_config.val_logs = ["also_missing"]
    with pytest.raises(ValueError, match=message):
        build_datasets(multi_config, QwenDriveAgent())


def test_rejects_train_val_token_leakage(multi_config):
    data_root = Path(multi_config.mogo_manifest).parent / "city_b"
    write_log(data_root, "city_b_val", token_prefix="city_a_train")
    with pytest.raises(ValueError, match="Duplicate scene token"):
        build_datasets(multi_config, QwenDriveAgent())


def test_rejects_wrong_filter_and_log_overlap(multi_config):
    multi_config.train_logs.append("city_a_val")
    with pytest.raises(ValueError, match="train_logs and val_logs overlap"):
        build_datasets(multi_config, QwenDriveAgent())
    multi_config.train_test_split.scene_filter.num_history_frames = 4
    with pytest.raises(ValueError, match="history=16, future=50, stride=1"):
        build_datasets(multi_config, QwenDriveAgent())


def timestamp_scene(timestamps):
    return SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=16, num_future_frames=50),
        frames=[SimpleNamespace(timestamp=int(stamp)) for stamp in timestamps],
        get_future_trajectory=lambda **kwargs: SimpleNamespace(poses=np.zeros((50, 3))),
    )


def test_accepts_bounded_10hz_timestamp_jitter():
    timestamps = np.arange(66) * 100000
    timestamps[8] += 66667
    timestamps[24] += 33333
    target = QwenDriveTrajectoryTargetBuilder(QwenDrivePlannerConfig()).compute_targets(timestamp_scene(timestamps))
    assert target["trajectory"].shape == (50, 3)


@pytest.mark.parametrize("mode", ["2hz", "drift", "missing", "duplicate", "reverse"])
def test_rejects_non_10hz_or_discontinuous_timestamps(mode):
    timestamps = np.arange(66) * 100000
    if mode == "2hz":
        timestamps *= 5
    elif mode == "drift":
        timestamps += np.arange(66) * 20000
    elif mode == "missing":
        timestamps[30:] += 100000
    elif mode == "duplicate":
        timestamps[30] = timestamps[29]
    else:
        timestamps[30] = timestamps[29] - 1
    with pytest.raises(ValueError, match="10Hz"):
        QwenDriveTrajectoryTargetBuilder(QwenDrivePlannerConfig()).compute_targets(timestamp_scene(timestamps))


def test_hydra_uses_nav_station_split_and_manifest():
    with initialize_config_dir(config_dir=str(ROOT / "navsim/planning/script/config/training"), version_base=None):
        config = compose(config_name="qwen_drive_il")
    expected = OmegaConf.load(ROOT / "navsim/planning/script/config/training/log_split/nav_station.yaml")
    assert config.train_logs == expected.train_logs
    assert config.val_logs == expected.val_logs
    assert Path(config.mogo_manifest).name == "nav_station.yaml"


@pytest.mark.parametrize("single_source", [False, True])
def test_launcher_selects_manifest_or_explicit_single_source(single_source):
    environment = {key: value for key, value in os.environ.items()
                   if key not in ("MOGO_MANIFEST", "MOGO_NAMES", "LOG_SPLIT", "DATA_ROOT")}
    environment["PYTHON"] = "/bin/echo"
    if single_source:
        environment.update(MOGO_MANIFEST="", DATA_ROOT="/tmp/single/dataset")
    else:
        environment["MOGO_NAMES"] = "beijing_0714-0716,zhuhai_road_nudge"
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/training/run_qwen_drive_il.sh"), "--cfg", "job"],
        env=environment, capture_output=True, text=True, check=True,
    )
    if single_source:
        assert "mogo_manifest=null" in result.stdout
        assert "navsim_log_path=/tmp/single/dataset/navsim_logs/train" in result.stdout
        assert "log_split@_global_=mogo" in result.stdout
    else:
        assert "log_split@_global_=nav_station" in result.stdout
        assert "mogo_dataset_names=[beijing_0714-0716,zhuhai_road_nudge]" in result.stdout
        assert "nav_station.yaml" in result.stdout
        assert "0312_0319" not in result.stdout


@pytest.mark.parametrize("selection", ["all", "limit", "tokens", "no_logs"])
def test_parallel_scene_loading_preserves_serial_samples_and_order(tmp_path, selection):
    for index in range(4):
        frames = write_log(tmp_path, f"log_{index}")
        frames[15]["roadblock_ids"] = []
        with (tmp_path / f"dataset/navsim_logs/train/log_{index}.pkl").open("wb") as stream:
            pickle.dump(frames, stream)
    scene_filter = SceneFilter(
        num_history_frames=16, num_future_frames=50, frame_stride=1,
        frame_interval=5, has_route=True,
        max_scenes=3 if selection == "limit" else None,
        tokens=["log_0-20", "log_2-25"] if selection == "tokens" else None,
        log_names=[] if selection == "no_logs" else None,
    )
    kwargs = dict(
        data_path=tmp_path / "dataset/navsim_logs/train",
        sensor_blobs_path=tmp_path / "dataset/sensor_blobs/train", scene_filter=scene_filter,
    )
    serial = SceneLoader(**kwargs, log_load_workers=1)
    parallel = SceneLoader(**kwargs, log_load_workers=3)
    assert parallel.tokens == serial.tokens
    assert parallel.token_to_log_file == serial.token_to_log_file
    for token in serial.tokens:
        assert [frame["token"] for frame in parallel.scene_frames_dicts[token]] == [
            frame["token"] for frame in serial.scene_frames_dicts[token]
        ]
        np.testing.assert_array_equal(
            parallel.scene_frames_dicts[token][0]["cams"]["CAM_F0"]["cam_intrinsic"],
            serial.scene_frames_dicts[token][0]["cams"]["CAM_F0"]["cam_intrinsic"],
        )


def test_log_prefetch_is_concurrent_bounded_and_ordered(monkeypatch):
    second_finished = Event()
    visited = []

    def load(path):
        visited.append(path.name)
        if path.name == "first":
            assert second_finished.wait(timeout=5), "log reads did not run concurrently"
        else:
            second_finished.set()
        return [{"token": path.name}]

    monkeypatch.setattr(scene_dataloader, "_load_log_frames", load)
    # Stop after one result: only the two prefetched logs may have been opened.
    with closing(scene_dataloader._iter_loaded_logs(
        [Path(name) for name in ("first", "second", "third", "fourth")], 2,
    )) as logs:
        assert next(logs) == [{"token": "first"}]
    assert set(visited) == {"first", "second"}


@pytest.mark.parametrize("workers", [1, 3])
def test_log_loading_propagates_corrupt_selected_log(tmp_path, workers):
    (tmp_path / "bad.pkl").write_bytes(b"invalid pickle")
    with pytest.raises(pickle.UnpicklingError):
        filter_scenes(tmp_path, SceneFilter(16, 50), num_workers=workers)


def test_log_loading_rejects_nonpositive_workers(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        filter_scenes(tmp_path, SceneFilter(16, 50), num_workers=0)



def test_indexed_warm_start_does_not_unpickle_logs(multi_config, monkeypatch):
    from navsim.common import indexed_scene_loader as indexed
    multi_config.scene_loading = "indexed"
    agent = QwenDriveAgent()
    train, val = build_datasets(multi_config, agent)
    expected = custom_collate_fn([train[0], train[3]])
    def fail(path):
        raise AssertionError(f"Warm startup reopened raw pickle {path}")
    monkeypatch.setattr(indexed, "_load_log_frames", fail)
    warm_train, warm_val = build_datasets(multi_config, agent)
    actual = custom_collate_fn([warm_train[0], warm_train[3]])
    assert len(warm_val) == len(val)
    assert actual[0]["view_frame_paths"] == expected[0]["view_frame_paths"]
    for key in expected[1]:
        torch.testing.assert_close(actual[1][key], expected[1][key])
    for part in warm_train.datasets:
        assert not isinstance(part._scene_loader.scene_frames_dicts, dict)


def test_indexed_features_equal_eager_and_spawn_worker(multi_config):
    agent = QwenDriveAgent()
    multi_config.scene_loading = "eager"
    eager, _ = build_datasets(multi_config, agent)
    multi_config.scene_loading = "indexed"
    indexed, _ = build_datasets(multi_config, agent)
    for idx in range(len(eager)):
        original, lazy = eager[idx], indexed[idx]
        assert original[0]["view_frame_paths"] == lazy[0]["view_frame_paths"]
        for group in (0, 1):
            for key in original[group]:
                if key != "view_frame_paths":
                    torch.testing.assert_close(original[group][key], lazy[group][key])
    # Open mappings above, then pickle/spawn: handles must reopen in the child.
    restored = pickle.loads(pickle.dumps(indexed))
    loader = torch.utils.data.DataLoader(
        restored, batch_size=2, num_workers=1, multiprocessing_context="spawn",
        collate_fn=custom_collate_fn,
    )
    batch = next(iter(loader))
    torch.testing.assert_close(batch[1]["trajectory"], custom_collate_fn([eager[0], eager[1]])[1]["trajectory"])


def test_indexed_source_changes_invalidate_only_changed_log(multi_config, monkeypatch):
    from navsim.common import indexed_scene_loader as indexed
    multi_config.scene_loading = "indexed"
    agent = QwenDriveAgent()
    old_train, _ = build_datasets(multi_config, agent)
    calls = []
    real_load = indexed._load_log_frames
    def load(path):
        calls.append(path.name)
        return real_load(path)
    monkeypatch.setattr(indexed, "_load_log_frames", load)
    root = Path(multi_config.mogo_manifest).parent / "city_a"
    write_log(root, "city_a_train", token_prefix="replacement", count=81)
    new_train, _ = build_datasets(multi_config, agent)
    assert calls == ["city_a_train.pkl"]
    assert len(new_train) == len(old_train) + 1
    assert new_train.datasets[0]._scene_loader.tokens[0].startswith("replacement")
    # Immutable packs keep an existing reader on its original snapshot.
    assert old_train.datasets[0]._scene_loader.tokens[0].startswith("city_a_train")
    assert old_train[0][1]["trajectory"].shape == (50, 3)


def test_indexed_filters_match_eager_raw_frames(tmp_path):
    from navsim.common.indexed_scene_loader import IndexedSceneLoader, prepare_scene_index
    for name in ("b", "a", "c"):
        write_log(tmp_path, name, count=81)
    raw = tmp_path / "dataset/navsim_logs/train"
    for stride, history, future in ((1, 16, 50), (5, 4, 10)):
        for limit, tokens in ((None, None), (2, None), (None, ["a-15", "b-20"]), (None, [])):
            scene_filter = SceneFilter(history, future, frame_stride=stride, frame_interval=5,
                                       max_scenes=limit, tokens=tokens)
            eager = filter_scenes(raw, scene_filter)
            index = prepare_scene_index(raw, scene_filter, tmp_path / "cache")
            lazy = IndexedSceneLoader(raw, tmp_path, scene_filter, index_path=index)
            assert lazy.tokens == list(eager)
            for token in eager:
                actual = lazy.scene_frames_dicts[token]
                assert [frame["token"] for frame in actual] == [frame["token"] for frame in eager[token]]
                np.testing.assert_array_equal(actual[0]["cams"]["CAM_F0"]["cam_intrinsic"],
                                              eager[token][0]["cams"]["CAM_F0"]["cam_intrinsic"])


def test_raw_pack_resume_after_partial_failure(tmp_path, monkeypatch):
    from navsim.common import indexed_scene_loader as indexed
    write_log(tmp_path, "first")
    write_log(tmp_path, "second")
    raw = tmp_path / "dataset/navsim_logs/train"
    real_load = indexed._load_log_frames
    calls = []
    def failing_load(path):
        calls.append(path.name)
        if len(calls) == 2:
            raise OSError("simulated interrupted read")
        return real_load(path)
    monkeypatch.setattr(indexed, "_load_log_frames", failing_load)
    with pytest.raises(OSError, match="interrupted"):
        indexed.prepare_scene_index(raw, SceneFilter(16, 50), tmp_path / "cache", workers=1)
    finished = calls[0]
    calls.clear()
    def load(path):
        calls.append(path.name)
        return real_load(path)
    monkeypatch.setattr(indexed, "_load_log_frames", load)
    index = indexed.prepare_scene_index(raw, SceneFilter(16, 50), tmp_path / "cache", workers=1)
    assert index.exists()
    assert len(calls) == 1 and calls[0] != finished


def test_raw_pack_build_bounds_live_decoded_logs(tmp_path, monkeypatch):
    import weakref
    from navsim.common import indexed_scene_loader as indexed
    for i in range(12):
        write_log(tmp_path, f"log_{i}")
    live = weakref.WeakValueDictionary()
    peak = []
    real_load = indexed._load_log_frames
    class Frames(list):
        pass
    def load(path):
        frames = Frames(real_load(path))
        live[id(frames)] = frames
        peak.append(len(live))
        return frames
    monkeypatch.setattr(indexed, "_load_log_frames", load)
    indexed.prepare_scene_index(tmp_path / "dataset/navsim_logs/train", SceneFilter(16, 50),
                                tmp_path / "cache", workers=2)
    assert max(peak) <= 2
    assert not live


def _distributed_raw_worker(rank, config, directory, failure):
    import json
    from datetime import timedelta
    import torch.distributed as dist
    from navsim.common import indexed_scene_loader as indexed
    from navsim.planning.script.run_training_qwen_drive_il import build_distributed_indexed_datasets
    from navsim.planning.training.distributed_cache_dataset import cache_index_sync_group
    config = OmegaConf.create(config)
    config.cache_index_dir = str(Path(directory) / "sync_index")
    config.cache_index_timeout_seconds = 60
    config.distributed_timeout_seconds = 60
    calls = []
    real_load = indexed._load_log_frames
    def load(path):
        calls.append(str(path))
        if failure:
            raise OSError("simulated rank-zero read error")
        return real_load(path)
    indexed._load_log_frames = load
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous",
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        with cache_index_sync_group(config, 2) as group:
            if failure:
                try:
                    build_distributed_indexed_datasets(config, QwenDriveAgent(), rank, 2, group)
                except RuntimeError as error:
                    assert "rank-zero read error" in str(error)
                    result = {"error": True}
                else:
                    raise AssertionError("rank-zero failure was not propagated")
            else:
                train, val = build_distributed_indexed_datasets(config, QwenDriveAgent(), rank, 2, group)
                result = {"tokens": [part._scene_loader.tokens for part in train.datasets],
                          "val_count": len(val), "sample_shape": list(train[rank][1]["trajectory"].shape)}
        result["raw_reads"] = len(calls)
        (Path(directory) / f"rank_{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", [False, True])
def test_distributed_raw_index_only_rank_zero_reads(multi_config, tmp_path, failure):
    import json
    import torch.multiprocessing as mp
    multi_config.scene_loading = "indexed"
    root = Path(multi_config.mogo_manifest).parent / "city_a"
    frames = write_log(root, "city_a_train")
    frames[0]["timestamp"] -= 107902
    for index in (5, 10, 15, 20):
        for view in ("CAM_F0", "CAM_L0", "CAM_R0"):
            frames[index]["cams"][view]["data_path"] = None
    with (root / "dataset/navsim_logs/train/city_a_train.pkl").open("wb") as stream:
        pickle.dump(frames, stream)
    mp.spawn(_distributed_raw_worker,
             args=(OmegaConf.to_container(multi_config), str(tmp_path), failure),
             nprocs=2, join=True)
    rank0 = json.loads((tmp_path / "rank_0.json").read_text())
    rank1 = json.loads((tmp_path / "rank_1.json").read_text())
    assert rank0.pop("raw_reads") > 0
    assert rank1.pop("raw_reads") == 0
    assert rank0 == rank1
    if not failure:
        assert rank0["tokens"][0] == ["city_a_train-25"]



def test_manifest_defaults_to_each_dataset_cache_directory(multi_config):
    root = Path(multi_config.mogo_manifest).parent
    manifest = OmegaConf.load(multi_config.mogo_manifest)
    for entry in manifest.datasets:
        del entry["cache_path"]
    OmegaConf.save(manifest, multi_config.mogo_manifest)
    sources = _dataset_source_configs(multi_config)
    for name, source in sources:
        assert Path(source.scene_cache_dir) == root / name / "exp/qwen35_drive_cache"
        assert source.cache_path is None
    assert multi_config.scene_cache_dir is None
    # Single-source layout uses the same directory even without a manifest.
    single = sources[0][1]
    single.mogo_manifest = None
    single.scene_cache_dir = None
    assert Path(_dataset_source_configs(single)[0][1].scene_cache_dir) == root / "city_a/exp/qwen35_drive_cache"


@pytest.mark.parametrize("override", ["manifest_relative", "manifest_absolute", "global"])
def test_cache_directory_overrides(multi_config, override):
    root = Path(multi_config.mogo_manifest).parent
    manifest = OmegaConf.load(multi_config.mogo_manifest)
    manifest.datasets[0].cache_path = (
        str(root / "explicit") if override == "manifest_absolute" else "custom/qwen35_drive_cache"
    )
    # Relative data roots and cache paths resolve against the manifest directory.
    manifest.datasets[0].data_root = "city_a"
    OmegaConf.save(manifest, multi_config.mogo_manifest)
    if override == "global":
        multi_config.scene_cache_dir = str(root / "global_override")
    sources = dict(_dataset_source_configs(multi_config))
    if override == "global":
        expected = root / "global_override"
        assert Path(sources["city_b"].scene_cache_dir) == expected
    else:
        expected = root / ("explicit" if override == "manifest_absolute" else "custom/qwen35_drive_cache")
        assert Path(sources["city_b"].scene_cache_dir) == root / "city_b/exp/qwen35_drive_cache"
    assert Path(sources["city_a"].scene_cache_dir) == expected
    if multi_config.scene_loading == "indexed":
        train, _ = build_datasets(multi_config, QwenDriveAgent())
        assert train.datasets[0]._scene_loader.index_path.is_relative_to(expected)
        assert train[0][1]["trajectory"].shape == (50, 3)


def test_new_manifest_dataset_is_cached_without_rebuilding_old_sources(multi_config, monkeypatch):
    from navsim.common import indexed_scene_loader as indexed
    multi_config.scene_loading = "indexed"
    agent = QwenDriveAgent()
    multi_config.mogo_dataset_names = ["city_a"]
    build_datasets(multi_config, agent)
    root = Path(multi_config.mogo_manifest).parent
    assert (root / "city_a/exp/qwen35_drive_cache").is_dir()
    assert not (root / "city_b/exp/qwen35_drive_cache").exists()
    reads = []
    load = indexed._load_log_frames
    def record(path):
        reads.append(path.name)
        return load(path)
    monkeypatch.setattr(indexed, "_load_log_frames", record)
    # Enable the uncached dataset from the manifest; reuse existing city_a.
    multi_config.mogo_dataset_names = None
    train, val = build_datasets(multi_config, agent)
    assert set(reads) == {"city_b_train.pkl", "city_b_val.pkl"}
    assert len(train) == len(val) == 6
    assert (root / "city_b/exp/qwen35_drive_cache").is_dir()
    reads.clear()
    build_datasets(multi_config, agent)
    assert reads == []


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_missing_raw_pack_is_rebuilt_without_reloading_other_logs(multi_config, monkeypatch, damage):
    from navsim.common import indexed_scene_loader as indexed
    multi_config.scene_loading = "indexed"
    agent = QwenDriveAgent()
    train, _ = build_datasets(multi_config, agent)
    packed = train.datasets[0]._scene_loader.scene_frames_dicts.packs[0]
    if damage == "missing":
        Path(packed.path).unlink()
    else:
        Path(packed.path).write_bytes(b"incomplete")
    calls = []
    load = indexed._load_log_frames
    def record(path):
        calls.append(path.name)
        return load(path)
    monkeypatch.setattr(indexed, "_load_log_frames", record)
    rebuilt, _ = build_datasets(multi_config, agent)
    assert calls == ["city_a_train.pkl"]
    assert rebuilt[0][1]["trajectory"].shape == (50, 3)
    assert Path(packed.path).stat().st_size == packed.offsets[-1]


@pytest.mark.parametrize("limit", [None, 2])
def test_time_filter_preserves_good_windows_and_matches_eager(tmp_path, limit, caplog):
    from navsim.common.indexed_scene_loader import IndexedSceneLoader, prepare_scene_index
    frames = write_log(tmp_path, "drop", count=141)
    # A gap splits the log; only windows crossing it must be rejected.
    for frame in frames[30:]:
        frame["timestamp"] += 107902
    raw = tmp_path / "dataset/navsim_logs/train"
    with (raw / "drop.pkl").open("wb") as stream:
        pickle.dump(frames, stream)
    scene_filter = SceneFilter(16, 50, frame_interval=5, timestamp_interval_s=0.1,
                               max_scenes=limit)
    eager = filter_scenes(raw, scene_filter)
    index_path = prepare_scene_index(raw, scene_filter, tmp_path / "cache")
    lazy = IndexedSceneLoader(raw, tmp_path, scene_filter, index_path=index_path)
    expected = [f"drop-{current}" for current in range(45, 91, 5)]
    if limit:
        expected = expected[:limit]
    assert lazy.tokens == list(eager) == expected
    with index_path.open("rb") as stream:
        rejected = pickle.load(stream)["rejected_timestamps"]
    assert set(rejected) == {f"drop-{current}" for current in range(15, 45, 5)}
    assert "0.107902s" in rejected["drop-15"]["error"]
    assert "invalid timestamps" in caplog.text
    target = QwenDriveTrajectoryTargetBuilder(QwenDrivePlannerConfig())
    for token in lazy.tokens:
        timestamps = [frame["timestamp"] for frame in lazy.scene_frames_dicts[token]]
        assert target.compute_targets(timestamp_scene(timestamps))["trajectory"].shape == (50, 3)


def test_time_filter_upgrades_legacy_metadata_without_repacking(tmp_path, monkeypatch):
    from navsim.common import indexed_scene_loader as indexed
    write_log(tmp_path, "log")
    raw = tmp_path / "dataset/navsim_logs/train"
    cache = tmp_path / "cache"
    old_filter = SceneFilter(16, 50, frame_interval=5)
    old_index = indexed.prepare_scene_index(raw, old_filter, cache)
    meta_path = next(cache.rglob("*.meta.pkl"))
    with meta_path.open("rb") as stream:
        packed, rows = pickle.load(stream)
    # Simulate a cache produced before per-frame timestamps were included.
    with meta_path.open("wb") as stream:
        pickle.dump((packed, [row[:3] for row in rows]), stream)
    pack_stat = Path(packed.path).stat()

    def no_raw_read(path):
        raise AssertionError("existing frame packs must be reused")
    monkeypatch.setattr(indexed, "_load_log_frames", no_raw_read)
    new_filter = SceneFilter(16, 50, frame_interval=5, timestamp_interval_s=0.1)
    new_index = indexed.prepare_scene_index(raw, new_filter, cache)
    assert new_index != old_index
    assert Path(packed.path).stat().st_mtime_ns == pack_stat.st_mtime_ns
    assert Path(packed.path).stat().st_size == pack_stat.st_size
    with meta_path.open("rb") as stream:
        _, upgraded = pickle.load(stream)
    assert [row[3] for row in upgraded] == [1000000 + i * 100000 for i in range(76)]
    monkeypatch.setattr(indexed, "_read_pack_frame_metadata", no_raw_read)
    assert indexed.prepare_scene_index(raw, new_filter, cache) == new_index
    # Disabled filtering retains the old signature and still reads the old index.
    assert indexed.prepare_scene_index(raw, old_filter, cache) == old_index
    assert indexed.IndexedSceneLoader(raw, tmp_path, new_filter, index_path=new_index).tokens == [
        "log-15", "log-20", "log-25",
    ]


def test_timestamp_boundary_and_diagnostic_context():
    from navsim.common.scene_timestamps import timestamp_grid_error
    timestamps = np.arange(66) * 100000
    timestamps[30] += 75000
    assert timestamp_grid_error(timestamps, 15) is None
    timestamps[30] += 1
    scene = timestamp_scene(timestamps)
    scene.scene_metadata.log_name = "timing_log"
    scene.scene_metadata.initial_token = "timing_token"
    with pytest.raises(ValueError, match="log_name=timing_log.*token=timing_token"):
        QwenDriveTrajectoryTargetBuilder(QwenDrivePlannerConfig()).compute_targets(scene)


@pytest.mark.parametrize("interval, tolerance", [(0, 0.075), (float("nan"), 0.075), (0.1, -1), (0.1, float("inf"))])
def test_timestamp_filter_rejects_invalid_parameters(interval, tolerance):
    with pytest.raises(ValueError):
        SceneFilter(timestamp_interval_s=interval, max_timestamp_error_s=tolerance)


@pytest.mark.parametrize("mode", ["missing_all", "missing_one", "edge_pad"])
def test_camera_filter_uses_selected_frames_and_keeps_partial_views(multi_config, mode, caplog):
    root = Path(multi_config.mogo_manifest).parent / "city_a"
    for split in ("train", "val"):
        frames = write_log(root, f"city_a_{split}", count=101)
        views = ("CAM_L0",) if mode == "missing_one" else ("CAM_F0", "CAM_L0", "CAM_R0")
        for frame in frames[:16]:
            for view in views:
                frame["cams"][view]["data_path"] = None
        # A path outside 0/5/10/15 cannot rescue the first sample.
        for view in views:
            frames[1]["cams"][view]["data_path"] = "image.png"
            if mode == "edge_pad":
                frames[15]["cams"][view]["data_path"] = "image.png"
        with (root / f"dataset/navsim_logs/train/city_a_{split}.pkl").open("wb") as stream:
            pickle.dump(frames, stream)
    train, val = build_datasets(multi_config, QwenDriveAgent())
    keep_first = mode == "edge_pad"
    for split, dataset in (("train", train), ("val", val)):
        assert len(dataset) == (11 if keep_first else 10)
        loader = dataset.datasets[0]._scene_loader
        assert (f"city_a_{split}-15" in loader.tokens) == keep_first
        assert f"city_a_{split}-20" in loader.tokens
        if multi_config.scene_loading == "indexed":
            with loader.index_path.open("rb") as stream:
                rejected = pickle.load(stream)["rejected_cameras"]
            assert len(rejected) == (0 if keep_first else 1)
        features, targets = dataset[0]
        assert [len(paths) for paths in features["view_frame_paths"]] == [4, 4, 4]
        assert targets["trajectory"].shape == (50, 3)
    if not keep_first:
        assert "city_a_train-15" in caplog.text and "cam_l0" in caplog.text


@pytest.mark.parametrize("legacy_columns", [3, 4])
def test_camera_metadata_upgrade_reuses_packs_and_keys_indices_by_views(tmp_path, monkeypatch, legacy_columns):
    from navsim.common import indexed_scene_loader as indexed
    frames = write_log(tmp_path, "side_missing")
    for frame in frames:
        frame["cams"]["CAM_L0"]["data_path"] = None
    raw = tmp_path / "dataset/navsim_logs/train"
    with (raw / "side_missing.pkl").open("wb") as stream:
        pickle.dump(frames, stream)
    cache = tmp_path / "cache"
    plain_filter = SceneFilter(16, 50, frame_interval=5, timestamp_interval_s=0.1)
    plain_index = indexed.prepare_scene_index(raw, plain_filter, cache)
    meta_path = next(cache.rglob("*.meta.pkl"))
    with meta_path.open("rb") as stream:
        packed, rows = pickle.load(stream)
    with meta_path.open("wb") as stream:
        pickle.dump((packed, [row[:legacy_columns] for row in rows]), stream)
    before = Path(packed.path).stat()

    def fail(path):
        raise AssertionError(f"Unexpected raw read or metadata rescan: {path}")
    monkeypatch.setattr(indexed, "_load_log_frames", fail)
    selected = {view: [0, 5, 10, 15] for view in ("cam_f0", "cam_l0", "cam_r0")}
    three_views = SceneFilter(16, 50, frame_interval=5, timestamp_interval_s=0.1,
                             required_camera_frames=selected)
    filtered_index = indexed.prepare_scene_index(raw, three_views, cache)
    assert filtered_index != plain_index
    with filtered_index.open("rb") as stream:
        filtered = pickle.load(stream)
    assert not filtered["records"] and len(filtered["rejected_cameras"]) == 3
    after = Path(packed.path).stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    with meta_path.open("rb") as stream:
        _, rows = pickle.load(stream)
    assert all(len(row) == 5 for row in rows)
    monkeypatch.setattr(indexed, "_read_pack_frame_metadata", fail)
    assert indexed.prepare_scene_index(raw, three_views, cache) == filtered_index
    front_only = SceneFilter(16, 50, frame_interval=5, timestamp_interval_s=0.1,
                             required_camera_frames={"cam_f0": [0, 5, 10, 15]})
    front_index = indexed.prepare_scene_index(raw, front_only, cache)
    assert front_index not in (plain_index, filtered_index)
    front_loader = indexed.IndexedSceneLoader(raw, tmp_path, front_only, index_path=front_index)
    assert front_loader.tokens == ["side_missing-15", "side_missing-20", "side_missing-25"]
    assert indexed.prepare_scene_index(raw, plain_filter, cache) == plain_index


def test_training_camera_filter_follows_agent_view_override(multi_config):
    root = Path(multi_config.mogo_manifest).parent
    for city in ("city_a", "city_b"):
        for split in ("train", "val"):
            path = root / city / f"dataset/navsim_logs/train/{city}_{split}.pkl"
            with path.open("rb") as stream:
                frames = pickle.load(stream)
            for frame in frames:
                frame["cams"]["CAM_L0"]["data_path"] = None
            with path.open("wb") as stream:
                pickle.dump(frames, stream)
    agent = QwenDriveAgent(QwenDrivePlannerConfig(view_list=["cam_f0"]))
    train, val = build_datasets(multi_config, agent)
    assert len(train) == len(val) == 6
    assert len(train[0][0]["view_frame_paths"]) == 1
    assert train.datasets[0]._scene_loader._scene_filter.required_camera_frames == {"cam_f0": [0, 5, 10, 15]}


def test_camera_error_includes_scene_context():
    from navsim.planning.training.dataset import Dataset
    def missing(_):
        raise ValueError("所选相机整路缺失：['cam_f0']")
    builder = SimpleNamespace(compute_features=missing, get_unique_name=lambda: "camera_builder")
    scene = SimpleNamespace(scene_metadata=SimpleNamespace(log_name="broken_log", initial_token="broken_token"))
    with pytest.raises(ValueError, match="cam_f0.*log_name=broken_log.*token=broken_token"):
        Dataset._compute_builder_features(builder, scene, None)


@pytest.mark.parametrize("selection", [{"unknown": [0]}, {"cam_f0": []}, {"cam_f0": [-1]}, {"cam_f0": [16]}])
def test_camera_filter_rejects_invalid_requirements(selection):
    with pytest.raises(ValueError, match="required_camera_frames"):
        SceneFilter(16, 50, required_camera_frames=selection)
