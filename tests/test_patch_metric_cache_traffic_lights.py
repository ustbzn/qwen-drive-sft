import csv
import gzip
import lzma
import pickle
import threading
from pathlib import Path

import scripts.cache_dataset.patch_metric_cache_traffic_lights as patcher
from scripts.cache_dataset.patch_metric_cache_traffic_lights import (
    DEFAULT_DATA_SPLIT,
    DEFAULT_WORKERS,
    DatasetSpec,
    _bounded_thread_map,
    build_parser,
    process_dataset,
)


class FakeMetricCache:
    """Small pickle-compatible stand-in that avoids map/planner dependencies in tests."""

    def __init__(self, marker, traffic_light_states=None, driving_command=None):
        self.marker = marker
        self.traffic_light_states = traffic_light_states
        self.driving_command = driving_command


def _write_raw_log(path: Path, frames, compressed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    with opener(path, "wb") as file_handle:
        pickle.dump(frames, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_metric_cache(path: Path, cache: FakeMetricCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(path, "wb") as file_handle:
        pickle.dump(cache, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def _read_metric_cache(path: Path) -> FakeMetricCache:
    with lzma.open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def _write_metric_metadata(cache_root: Path, paths: list[Path]) -> None:
    metadata_path = cache_root / "metadata" / "metric_cache_metadata_node_0.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["file_name"])
        writer.writeheader()
        writer.writerows({"file_name": str(path)} for path in paths)


def _metric_path(cache_root: Path, log_name: str, token: str) -> Path:
    return cache_root / log_name / "unknown" / token / "metric_cache.pkl"


def _frame(log_name: str, token: str, states=None, command=None):
    frame = {
        "log_name": log_name,
        "token": token,
        "driving_command": [1.0, 0.0, 0.0, 0.0] if command is None else command,
    }
    if states is not None:
        frame["traffic_light_states"] = states
    return frame


def test_dry_run_is_non_mutating_and_apply_patches_all_cache_tokens(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "metric_cache"
    log_name = "log-a"
    red_light = [{"state": "red", "position": [[1.0, 2.0], [3.0, 4.0]]}]
    _write_raw_log(
        data_root / "dataset" / "navsim_logs" / "trainval" / f"{log_name}.pkl",
        [_frame(log_name, "token-a", red_light), _frame(log_name, "token-b")],
    )
    path_a = _metric_path(cache_root, log_name, "token-a")
    path_b = _metric_path(cache_root, log_name, "token-b")
    _write_metric_cache(path_a, FakeMetricCache(marker={"preserved": "a"}))
    _write_metric_cache(path_b, FakeMetricCache(marker={"preserved": "b"}))
    before_a = path_a.read_bytes()
    spec = DatasetSpec("dataset", data_root, cache_root)

    dry_run = process_dataset(spec, apply=False, workers=2, progress_every=0)

    assert dry_run.passed
    assert dry_run.needs_patch == 2
    assert dry_run.patched == 0
    assert path_a.read_bytes() == before_a

    applied = process_dataset(
        spec,
        apply=True,
        workers=2,
        progress_every=0,
        verify_after_write=True,
    )

    assert applied.passed
    assert applied.patched == 2
    cache_a = _read_metric_cache(path_a)
    cache_b = _read_metric_cache(path_b)
    assert cache_a.marker == {"preserved": "a"}
    assert cache_a.traffic_light_states == red_light
    assert cache_a.driving_command == [1.0, 0.0, 0.0, 0.0]
    assert cache_b.marker == {"preserved": "b"}
    assert cache_b.traffic_light_states == []

    mtime_after_apply = path_a.stat().st_mtime_ns
    repeated = process_dataset(spec, apply=True, workers=2, progress_every=0)
    assert repeated.passed
    assert repeated.already_current == 2
    assert repeated.patched == 0
    assert path_a.stat().st_mtime_ns == mtime_after_apply


def test_apply_refuses_missing_raw_token_before_any_cache_is_written(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "metric_cache"
    log_name = "log-a"
    _write_raw_log(
        data_root / "dataset" / "navsim_logs" / "trainval" / f"{log_name}.pkl",
        [_frame(log_name, "known-token")],
    )
    existing = _metric_path(cache_root, log_name, "known-token")
    missing = _metric_path(cache_root, log_name, "missing-token")
    _write_metric_cache(existing, FakeMetricCache(marker="known"))
    _write_metric_cache(missing, FakeMetricCache(marker="missing"))
    before_existing = existing.read_bytes()
    before_missing = missing.read_bytes()

    report = process_dataset(
        DatasetSpec("dataset", data_root, cache_root),
        apply=True,
        workers=2,
        progress_every=0,
    )

    assert not report.passed
    assert any("missing raw source frame" in error for error in report.errors)
    assert existing.read_bytes() == before_existing
    assert missing.read_bytes() == before_missing


def test_nonempty_mismatch_requires_explicit_repair_flag(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "metric_cache"
    log_name = "log-a"
    token = "token-a"
    raw_states = [{"state": "red", "position": [[1.0, 2.0], [3.0, 4.0]]}]
    _write_raw_log(
        data_root / "dataset" / "navsim_logs" / "trainval" / f"{log_name}.pkl",
        [_frame(log_name, token, raw_states)],
        compressed=True,
    )
    cache_path = _metric_path(cache_root, log_name, token)
    _write_metric_cache(
        cache_path,
        FakeMetricCache(
            marker="preserved",
            traffic_light_states=[{"state": "green"}],
            driving_command=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    spec = DatasetSpec("dataset", data_root, cache_root)

    refused = process_dataset(spec, apply=True, workers=2, progress_every=0)
    assert not refused.passed
    assert any("--repair-mismatched" in error for error in refused.errors)
    assert _read_metric_cache(cache_path).traffic_light_states == [{"state": "green"}]

    repaired = process_dataset(
        spec,
        apply=True,
        workers=2,
        progress_every=0,
        repair_mismatched=True,
        verify_after_write=True,
    )
    assert repaired.passed
    assert repaired.patched == 1
    cache = _read_metric_cache(cache_path)
    assert cache.marker == "preserved"
    assert cache.traffic_light_states == raw_states


def test_default_split_ignores_same_log_stem_in_train(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "metric_cache"
    log_name = "log-a"
    token = "token-a"
    _write_raw_log(
        data_root / "dataset" / "navsim_logs" / "train" / f"{log_name}.pkl",
        [_frame(log_name, token, [{"state": "green"}])],
    )
    red_states = [{"state": "red"}]
    _write_raw_log(
        data_root / "dataset" / "navsim_logs" / "trainval" / f"{log_name}.pkl",
        [_frame(log_name, token, red_states)],
    )
    cache_path = _metric_path(cache_root, log_name, token)
    _write_metric_cache(cache_path, FakeMetricCache(marker="preserved"))

    report = process_dataset(
        DatasetSpec("dataset", data_root, cache_root),
        apply=True,
        workers=2,
        progress_every=0,
    )

    assert report.passed
    assert _read_metric_cache(cache_path).traffic_light_states == red_states


def test_metadata_index_avoids_recursive_cache_scan(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "metric_cache"
    log_name = "log-a"
    token = "token-a"
    _write_raw_log(
        data_root / "dataset" / "navsim_logs" / "trainval" / f"{log_name}.pkl",
        [_frame(log_name, token)],
    )
    cache_path = _metric_path(cache_root, log_name, token)
    _write_metric_cache(cache_path, FakeMetricCache(marker="preserved"))
    _write_metric_metadata(cache_root, [cache_path])

    def fail_scan(*_args, **_kwargs):
        raise AssertionError("metadata index unexpectedly fell back to a directory scan")

    monkeypatch.setattr(patcher, "_build_cache_index_from_scan", fail_scan)
    report = process_dataset(
        DatasetSpec("dataset", data_root, cache_root),
        apply=False,
        workers=2,
        progress_every=0,
    )

    assert report.passed
    assert report.cache_files == 1
    assert report.needs_patch == 1



def test_metadata_index_ignores_stale_cache_paths(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "metric_cache"
    log_name = "log-a"
    token = "token-a"
    _write_raw_log(
        data_root / "dataset" / "navsim_logs" / "trainval" / f"{log_name}.pkl",
        [_frame(log_name, token)],
    )
    existing = _metric_path(cache_root, log_name, token)
    stale = _metric_path(cache_root, "deleted-log", "deleted-token")
    _write_metric_cache(existing, FakeMetricCache(marker="preserved"))
    _write_metric_metadata(cache_root, [existing, stale])

    report = process_dataset(
        DatasetSpec("dataset", data_root, cache_root),
        apply=False,
        workers=2,
        progress_every=0,
    )

    assert report.passed
    assert report.cache_files == 1
    assert report.needs_patch == 1


def test_bounded_thread_map_does_not_wait_for_input_order():
    release_slow_task = threading.Event()

    def work(item):
        if item == "slow":
            release_slow_task.wait(timeout=1)
        return item

    results = _bounded_thread_map(work, ["slow", "fast"], workers=2, max_pending_factor=1)
    try:
        assert next(results) == "fast"
    finally:
        release_slow_task.set()
        results.close()


def test_cli_defaults_to_ninety_six_workers():
    args = build_parser().parse_args(
        ["--data-root", "/tmp/data", "--metric-cache-root", "/tmp/cache"]
    )

    assert args.workers == DEFAULT_WORKERS == 96
    assert args.data_split == DEFAULT_DATA_SPLIT == "trainval"
