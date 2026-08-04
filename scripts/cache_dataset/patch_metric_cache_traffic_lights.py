#!/usr/bin/env python3
"""Patch MOGO metric caches with post-schema traffic-light fields.

This is a cache-only migration for ``2bbbcd4``.  It reads the current raw PKL
frame for every existing metric-cache ``(log_name, token)`` and writes only
``traffic_light_states`` and ``driving_command`` into the already-built cache.
It never constructs a map, planner, observation, trajectory, or Scene.

The default mode is a full dry-run.  ``--apply`` is deliberately required to
rewrite cache files, and a manifest apply additionally requires explicit
``--name`` selection or ``--all``.

Examples:

  # Validate the five datasets containing traffic-light / stop-line records.
  python "scripts/cache_dataset/patch_metric_cache_traffic_lights.py" \\
    --manifest "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml" \\
    --name beijing_0509_0626-0627 \\
    --name zhuhai_0629-0701 \\
    --name zhuhai_20260624 \\
    --name beijing_0714-0716 \\
    --name zhuhai_20260709-20260714 \\
    --workers 96

  # Apply after a successful dry-run.  Re-open each changed file for validation.
  python "scripts/cache_dataset/patch_metric_cache_traffic_lights.py" \\
    --manifest "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml" \\
    --name beijing_0509_0626-0627 \\
    --name zhuhai_0629-0701 \\
    --name zhuhai_20260624 \\
    --name beijing_0714-0716 \\
    --name zhuhai_20260709-20260714 \\
    --workers 96 --apply --verify-after-write
"""
from __future__ import annotations

import argparse
import csv
import gzip
import lzma
import os
import pickle
import stat
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRIC_CACHE_FILE_NAME = "metric_cache.pkl"
TRAFFIC_LIGHT_STATES_FIELD = "traffic_light_states"
DRIVING_COMMAND_FIELD = "driving_command"
DEFAULT_WORKERS = 96
MAX_CACHE_PROCESSES = 16
DEFAULT_DATA_SPLIT = "trainval"
MAX_REPORTED_ERRORS = 50


class MetricCachePatchError(RuntimeError):
    """Raised when a cache cannot be patched without guessing its source fields."""


@dataclass(frozen=True, order=True)
class SampleKey:
    """Stable identifier shared by raw MOGO frames and metric-cache paths."""

    log_name: str
    token: str

    def as_text(self) -> str:
        return f"{self.log_name}/{self.token}"


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset roots required for one metric-cache patch job."""

    name: str
    data_root: Path
    metric_cache_root: Path
    data_split: str = DEFAULT_DATA_SPLIT

    @property
    def log_root(self) -> Path:
        return self.data_root / "dataset" / "navsim_logs" / self.data_split


@dataclass(frozen=True)
class CacheItem:
    """One cache file and the source key that owns it."""

    key: SampleKey
    path: Path
    scenario_type: str


@dataclass(frozen=True)
class SourceFields:
    """Exact raw-frame values written by the full metric-cache builder."""

    traffic_light_states: list[Any]
    driving_command: Any
    raw_path: Path
    frame_index: int


@dataclass(frozen=True)
class RawLogTask:
    """Raw PKL file and the cache tokens that must be found in it."""

    path: Path
    log_name: str
    tokens: frozenset[str]


@dataclass
class CacheIndex:
    """Metric-cache inventory built before any cache object is unpickled."""

    items: dict[SampleKey, CacheItem] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    ignored_files: int = 0
    source: str = "scan"


@dataclass
class RawLogResult:
    """Source fields extracted from one raw PKL file."""

    task: RawLogTask
    source_fields: dict[SampleKey, SourceFields] = field(default_factory=dict)
    frame_count: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CacheInspection:
    """Read-only comparison of one cache against its current raw source frame."""

    item: CacheItem
    source_fields: SourceFields
    status: str
    error: Optional[str] = None


@dataclass(frozen=True)
class PatchResult:
    """Result of one concurrent cache write attempt."""

    path: Path
    status: str
    error: Optional[str] = None


@dataclass
class DatasetReport:
    """Summary emitted for one dataset invocation."""

    spec: DatasetSpec
    raw_logs: int = 0
    raw_frames: int = 0
    cache_files: int = 0
    already_current: int = 0
    needs_patch: int = 0
    patched: int = 0
    raced_current: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.cache_files > 0 and not self.errors


T = TypeVar("T")
R = TypeVar("R")


def _bounded_executor_map(
    executor: Executor,
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int,
    max_pending_factor: int,
) -> Iterator[R]:
    """Submit bounded work and yield whichever task completes first."""
    iterator = iter(items)
    pending: set[Future[R]] = set()

    def fill() -> None:
        while len(pending) < workers * max_pending_factor:
            try:
                item = next(iterator)
            except StopIteration:
                return
            pending.add(executor.submit(fn, item))

    try:
        fill()
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            fill()
            for future in completed:
                yield future.result()
    finally:
        for future in pending:
            future.cancel()


def _bounded_thread_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int,
    max_pending_factor: int = 4,
) -> Iterator[R]:
    """Execute bounded I/O work in threads."""
    if workers <= 0:
        raise ValueError("workers must be positive")

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        yield from _bounded_executor_map(
            executor, fn, items, workers, max_pending_factor
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _bounded_process_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int,
    max_pending_factor: int = 2,
) -> Iterator[R]:
    """Execute bounded CPU-heavy cache work in separate processes."""
    if workers <= 0:
        raise ValueError("workers must be positive")

    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        yield from _bounded_executor_map(
            executor, fn, items, workers, max_pending_factor
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _append_error(errors: list[str], message: str) -> None:
    if len(errors) < MAX_REPORTED_ERRORS:
        errors.append(message)


def _iter_pickle_logs(log_root: Path) -> Iterator[Path]:
    """Yield raw PKL files deterministically, including split subdirectories."""
    for directory, dir_names, file_names in os.walk(log_root):
        dir_names.sort()
        for file_name in sorted(file_names):
            if file_name.endswith(".pkl"):
                yield Path(directory) / file_name


def _load_raw_pickle(path: Path) -> Any:
    """Load plain or gzip-compressed raw MOGO PKLs."""
    payload = path.read_bytes()
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    return pickle.loads(payload)


def _frames_from_payload(payload: Any) -> list[Any] | tuple[Any, ...]:
    """Normalize supported raw-log payload containers to their frame sequence."""
    if isinstance(payload, (list, tuple)):
        return payload
    if isinstance(payload, Mapping):
        for key in ("frames", "scene_dict_list"):
            frames = payload.get(key)
            if isinstance(frames, (list, tuple)):
                return frames
        if "token" in payload and "ego2global_translation" in payload:
            return [payload]
    raise MetricCachePatchError(f"unsupported raw log payload type: {type(payload).__name__}")


def _add_cache_path(index: CacheIndex, metric_cache_root: Path, path: Path) -> None:
    """Validate and add one metric-cache path to an index."""
    try:
        relative_path = path.relative_to(metric_cache_root)
    except ValueError:
        _append_error(index.errors, f"metric-cache metadata path is outside root: {path}")
        return

    parts = relative_path.parts
    if len(parts) != 4 or parts[-1] != METRIC_CACHE_FILE_NAME:
        _append_error(
            index.errors,
            "unexpected metric-cache layout "
            f"(expected log/scenario_type/token/{METRIC_CACHE_FILE_NAME}): {path}",
        )
        return

    log_name, scenario_type, token, _ = parts
    key = SampleKey(log_name=log_name, token=token)
    previous = index.items.get(key)
    if previous is not None:
        _append_error(
            index.errors,
            f"duplicate metric cache for {key.as_text()}: {previous.path} and {path}",
        )
        return
    index.items[key] = CacheItem(key=key, path=path, scenario_type=scenario_type)


def _cache_metadata_csv_paths(metric_cache_root: Path) -> list[Path]:
    """Return cache-builder metadata CSVs without recursively touching cache directories."""
    metadata_root = metric_cache_root / "metadata"
    try:
        return sorted(
            path
            for path in metadata_root.iterdir()
            if path.is_file() and "_metadata_node_" in path.stem and path.suffix == ".csv"
        )
    except FileNotFoundError:
        return []


def _build_cache_index_from_metadata(metric_cache_root: Path) -> Optional[CacheIndex]:
    """Load the authoritative cache paths emitted by the metric-cache builder."""
    metadata_paths = _cache_metadata_csv_paths(metric_cache_root)
    if not metadata_paths:
        return None

    index = CacheIndex(source="metadata")
    try:
        existing_log_dirs = {
            entry.name
            for entry in os.scandir(metric_cache_root)
            if entry.name != "metadata" and entry.is_dir(follow_symlinks=False)
        }
    except OSError as exc:
        _append_error(
            index.errors,
            f"failed to list metric cache root {metric_cache_root}: "
            f"{type(exc).__name__}: {exc}",
        )
        return index

    for metadata_path in metadata_paths:
        try:
            with metadata_path.open("r", newline="") as file_handle:
                reader = csv.DictReader(file_handle)
                if reader.fieldnames is None or "file_name" not in reader.fieldnames:
                    _append_error(index.errors, f"metadata CSV has no file_name column: {metadata_path}")
                    continue
                for row_number, row in enumerate(reader, start=2):
                    file_name = (row.get("file_name") or "").strip()
                    if not file_name:
                        _append_error(
                            index.errors,
                            f"metadata CSV has an empty file_name at {metadata_path}:{row_number}",
                        )
                        continue
                    path = Path(file_name)
                    if not path.is_absolute():
                        path = metric_cache_root / path
                    try:
                        relative_path = path.relative_to(metric_cache_root)
                    except ValueError:
                        _add_cache_path(index, metric_cache_root, path)
                        continue
                    if not relative_path.parts or relative_path.parts[0] not in existing_log_dirs:
                        # Metadata can outlive a partial cache rebuild.  A stale
                        # row must not turn into a false missing-raw-frame error.
                        index.ignored_files += 1
                        continue
                    _add_cache_path(index, metric_cache_root, path)
        except (OSError, csv.Error) as exc:
            _append_error(
                index.errors,
                f"failed to read metadata CSV {metadata_path}: {type(exc).__name__}: {exc}",
            )

    if not index.items and not index.errors:
        _append_error(index.errors, f"no cache paths found in metadata under {metric_cache_root}")
    return index


def _scan_cache_log_directory(metric_cache_root: Path, log_directory: Path) -> CacheIndex:
    """Recursively scan one log directory for the metadata-free compatibility path."""
    index = CacheIndex(source="scan")
    for directory, dir_names, file_names in os.walk(log_directory):
        dir_names.sort()
        for file_name in sorted(file_names):
            if file_name == METRIC_CACHE_FILE_NAME:
                _add_cache_path(index, metric_cache_root, Path(directory) / file_name)
    return index


def _merge_cache_index(destination: CacheIndex, source: CacheIndex) -> None:
    """Merge one independently scanned log index into the dataset index."""
    for error in source.errors:
        _append_error(destination.errors, error)
    for key, item in source.items.items():
        previous = destination.items.get(key)
        if previous is not None:
            _append_error(
                destination.errors,
                f"duplicate metric cache for {key.as_text()}: {previous.path} and {item.path}",
            )
            continue
        destination.items[key] = item


def _build_cache_index_from_scan(metric_cache_root: Path, workers: int) -> CacheIndex:
    """Fall back to scanning top-level log directories concurrently."""
    index = CacheIndex(source="parallel_scan")
    try:
        log_directories = sorted(
            Path(entry.path)
            for entry in os.scandir(metric_cache_root)
            if entry.name != "metadata"
            and not entry.name.startswith(".")
            and entry.is_dir(follow_symlinks=False)
        )
    except OSError as exc:
        _append_error(index.errors, f"failed to list metric cache root {metric_cache_root}: {exc}")
        return index

    scan_fn = lambda log_directory: _scan_cache_log_directory(metric_cache_root, log_directory)
    for partial_index in _bounded_thread_map(scan_fn, log_directories, workers):
        _merge_cache_index(index, partial_index)
    if not index.items and not index.errors:
        _append_error(index.errors, f"no {METRIC_CACHE_FILE_NAME} files found under {metric_cache_root}")
    return index


def _build_cache_index(metric_cache_root: Path, workers: int) -> CacheIndex:
    """Build a fast metadata index, with a concurrent directory-scan fallback."""
    if not metric_cache_root.is_dir():
        index = CacheIndex()
        _append_error(index.errors, f"metric cache root does not exist: {metric_cache_root}")
        return index

    metadata_index = _build_cache_index_from_metadata(metric_cache_root)
    if metadata_index is not None:
        return metadata_index
    return _build_cache_index_from_scan(metric_cache_root, workers)


def _raw_paths_by_stem(log_root: Path) -> tuple[dict[str, Path], list[str]]:
    """Map raw PKL stems to paths and reject ambiguous log-name ownership."""
    paths_by_stem: dict[str, list[Path]] = defaultdict(list)
    for path in _iter_pickle_logs(log_root):
        paths_by_stem[path.stem].append(path)

    paths: dict[str, Path] = {}
    errors: list[str] = []
    for stem, candidates in sorted(paths_by_stem.items()):
        if len(candidates) != 1:
            _append_error(errors, f"duplicate raw PKL stem {stem}: {candidates}")
            continue
        paths[stem] = candidates[0]
    return paths, errors


def _source_fields_from_raw_log(task: RawLogTask) -> RawLogResult:
    """Extract only source frames whose tokens already exist in the metric cache."""
    result = RawLogResult(task=task)
    try:
        frames = _frames_from_payload(_load_raw_pickle(task.path))
        result.frame_count = len(frames)
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                continue

            token = frame.get("token")
            if token is None or str(token) not in task.tokens:
                continue

            frame_log_name = frame.get("log_name")
            if frame_log_name is None or str(frame_log_name) != task.log_name:
                _append_error(
                    result.errors,
                    f"{task.path}: frame={frame_index} token={token} has log_name={frame_log_name!r}, "
                    f"expected {task.log_name!r}",
                )
                continue
            if DRIVING_COMMAND_FIELD not in frame or frame[DRIVING_COMMAND_FIELD] is None:
                _append_error(
                    result.errors,
                    f"{task.path}: frame={frame_index} token={token} is missing {DRIVING_COMMAND_FIELD}",
                )
                continue

            try:
                traffic_light_states = list(frame.get(TRAFFIC_LIGHT_STATES_FIELD) or [])
            except Exception as exc:
                _append_error(
                    result.errors,
                    f"{task.path}: frame={frame_index} token={token} has invalid "
                    f"{TRAFFIC_LIGHT_STATES_FIELD}: {type(exc).__name__}: {exc}",
                )
                continue

            key = SampleKey(log_name=task.log_name, token=str(token))
            if key in result.source_fields:
                _append_error(result.errors, f"{task.path}: duplicate raw frame for {key.as_text()}")
                continue
            result.source_fields[key] = SourceFields(
                traffic_light_states=traffic_light_states,
                driving_command=frame[DRIVING_COMMAND_FIELD],
                raw_path=task.path,
                frame_index=frame_index,
            )
    except Exception as exc:
        _append_error(result.errors, f"{task.path}: {type(exc).__name__}: {exc}")
    return result


def _load_metric_cache(path: Path) -> Any:
    """Load the existing LZMA-compressed metric cache without importing a builder."""
    cache = pickle.loads(lzma.decompress(path.read_bytes()))
    if not hasattr(cache, "__dict__"):
        raise MetricCachePatchError(f"metric cache payload has no mutable attributes: {type(cache).__name__}")
    return cache


def _values_equal(left: Any, right: Any) -> bool:
    """Compare arbitrary pickled cache values without ambiguous NumPy truth values."""
    try:
        return pickle.dumps(left, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
            right, protocol=pickle.HIGHEST_PROTOCOL
        )
    except Exception:
        return False


def _inspect_cache_object(
    cache: Any,
    item: CacheItem,
    source_fields: SourceFields,
    repair_mismatched: bool,
) -> CacheInspection:
    """Classify a cache as current, safely patchable, or inconsistent."""
    cached_states = getattr(cache, TRAFFIC_LIGHT_STATES_FIELD, None)
    cached_command = getattr(cache, DRIVING_COMMAND_FIELD, None)
    states_match = _values_equal(cached_states, source_fields.traffic_light_states)
    command_match = _values_equal(cached_command, source_fields.driving_command)
    if states_match and command_match:
        return CacheInspection(item=item, source_fields=source_fields, status="current")

    states_missing = cached_states is None
    command_missing = cached_command is None
    if (states_missing or command_missing) and not (
        (not states_missing and not states_match) or (not command_missing and not command_match)
    ):
        return CacheInspection(item=item, source_fields=source_fields, status="needs_patch")

    if repair_mismatched:
        return CacheInspection(item=item, source_fields=source_fields, status="needs_patch")
    return CacheInspection(
        item=item,
        source_fields=source_fields,
        status="mismatch",
        error=(
            f"{item.path}: existing {TRAFFIC_LIGHT_STATES_FIELD}/{DRIVING_COMMAND_FIELD} "
            "does not match the current raw frame; rerun with --repair-mismatched "
            "only after confirming the raw dataset and cache refer to the same revision"
        ),
    )


def _inspect_cache_item(
    item: CacheItem,
    source_fields: SourceFields,
    repair_mismatched: bool,
) -> CacheInspection:
    """Load and inspect one cache, turning deserialization failures into reportable errors."""
    try:
        cache = _load_metric_cache(item.path)
        return _inspect_cache_object(cache, item, source_fields, repair_mismatched)
    except Exception as exc:
        return CacheInspection(
            item=item,
            source_fields=source_fields,
            status="error",
            error=f"{item.path}: {type(exc).__name__}: {exc}",
        )


def _inspect_cache_work_item(
    work_item: tuple[CacheItem, SourceFields, bool],
) -> CacheInspection:
    """Process-pool adapter for one cache inspection."""
    item, source_fields, repair_mismatched = work_item
    return _inspect_cache_item(item, source_fields, repair_mismatched)


def _atomic_dump_metric_cache(path: Path, cache: Any) -> None:
    """Write a cache replacement atomically while preserving the original mode bits."""
    original_mode = stat.S_IMODE(path.stat().st_mode)
    serialized = pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = lzma.compress(serialized, preset=0)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, original_mode)
        with os.fdopen(descriptor, "wb") as file_handle:
            descriptor = -1
            file_handle.write(compressed)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _patch_cache_item(
    inspection: CacheInspection,
    repair_mismatched: bool,
    verify_after_write: bool,
) -> PatchResult:
    """Recheck, patch, and atomically replace one cache to tolerate concurrent jobs."""
    try:
        cache = _load_metric_cache(inspection.item.path)
        current = _inspect_cache_object(
            cache,
            inspection.item,
            inspection.source_fields,
            repair_mismatched,
        )
        if current.status == "current":
            return PatchResult(inspection.item.path, "raced_current")
        if current.status != "needs_patch":
            return PatchResult(inspection.item.path, "error", current.error)

        setattr(cache, TRAFFIC_LIGHT_STATES_FIELD, inspection.source_fields.traffic_light_states)
        setattr(cache, DRIVING_COMMAND_FIELD, inspection.source_fields.driving_command)
        _atomic_dump_metric_cache(inspection.item.path, cache)

        if verify_after_write:
            written = _load_metric_cache(inspection.item.path)
            written_states = getattr(written, TRAFFIC_LIGHT_STATES_FIELD, None)
            written_command = getattr(written, DRIVING_COMMAND_FIELD, None)
            if not (
                _values_equal(written_states, inspection.source_fields.traffic_light_states)
                and _values_equal(written_command, inspection.source_fields.driving_command)
            ):
                raise MetricCachePatchError("post-write field verification failed")
        return PatchResult(inspection.item.path, "patched")
    except Exception as exc:
        return PatchResult(inspection.item.path, "error", f"{type(exc).__name__}: {exc}")


def _patch_cache_work_item(
    work_item: tuple[CacheInspection, bool, bool],
) -> PatchResult:
    """Process-pool adapter for one cache patch."""
    inspection, repair_mismatched, verify_after_write = work_item
    return _patch_cache_item(inspection, repair_mismatched, verify_after_write)


def _resolve_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    """Resolve either manifest datasets or one direct dataset pair."""
    if args.manifest:
        if args.data_root or args.metric_cache_root:
            raise MetricCachePatchError(
                "--manifest cannot be combined with --data-root/--metric-cache-root"
            )
        from omegaconf import OmegaConf

        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise MetricCachePatchError(f"manifest not found: {manifest_path}")
        manifest = OmegaConf.load(manifest_path)
        datasets = list(manifest.datasets)
        wanted = set(args.names or [])
        if wanted:
            datasets = [dataset for dataset in datasets if str(dataset.name) in wanted]
            missing = wanted - {str(dataset.name) for dataset in datasets}
            if missing:
                raise MetricCachePatchError(f"unknown dataset names: {sorted(missing)}")
        return [
            DatasetSpec(
                name=str(dataset.name),
                data_root=Path(str(dataset.data_root)),
                metric_cache_root=Path(str(dataset.metric_cache_path)),
                data_split=args.data_split,
            )
            for dataset in datasets
        ]

    if not args.data_root or not args.metric_cache_root:
        raise MetricCachePatchError(
            "provide --manifest, or both --data-root and --metric-cache-root"
        )
    if args.names or args.all:
        raise MetricCachePatchError("--name and --all are only supported with --manifest")
    return [
        DatasetSpec(
            name=args.dataset_name,
            data_root=Path(args.data_root),
            metric_cache_root=Path(args.metric_cache_root),
            data_split=args.data_split,
        )
    ]


def _print_progress(spec: DatasetSpec, stage: str, completed: int, total: int) -> None:
    print(f"[{spec.name}] {stage}={completed}/{total}")


def _collect_source_fields(
    spec: DatasetSpec,
    cache_index: CacheIndex,
    workers: int,
    progress_every: int,
    report: DatasetReport,
) -> dict[SampleKey, SourceFields]:
    """Match every cache key to one raw PKL frame before cache inspection begins."""
    raw_paths, path_errors = _raw_paths_by_stem(spec.log_root)
    report.errors.extend(path_errors[: max(0, MAX_REPORTED_ERRORS - len(report.errors))])
    if not spec.log_root.is_dir():
        _append_error(report.errors, f"raw log root does not exist: {spec.log_root}")
        return {}

    tokens_by_log: dict[str, set[str]] = defaultdict(set)
    for key in cache_index.items:
        tokens_by_log[key.log_name].add(key.token)

    tasks: list[RawLogTask] = []
    for log_name, tokens in sorted(tokens_by_log.items()):
        raw_path = raw_paths.get(log_name)
        if raw_path is None:
            _append_error(report.errors, f"missing raw PKL for cache log {log_name}")
            continue
        tasks.append(RawLogTask(path=raw_path, log_name=log_name, tokens=frozenset(tokens)))

    source_index: dict[SampleKey, SourceFields] = {}
    for completed, result in enumerate(
        _bounded_thread_map(_source_fields_from_raw_log, tasks, workers), start=1
    ):
        report.raw_logs += 1
        report.raw_frames += result.frame_count
        for error in result.errors:
            _append_error(report.errors, error)
        for key, source_fields in result.source_fields.items():
            if key in source_index:
                _append_error(report.errors, f"duplicate raw source frame for {key.as_text()}")
                continue
            source_index[key] = source_fields
        if progress_every > 0 and completed % progress_every == 0:
            _print_progress(spec, "raw_logs_scanned", completed, len(tasks))

    for key in sorted(cache_index.items):
        if key not in source_index:
            _append_error(report.errors, f"missing raw source frame for cache {key.as_text()}")
    return source_index


def process_dataset(
    spec: DatasetSpec,
    apply: bool,
    workers: int = DEFAULT_WORKERS,
    progress_every: int = 1000,
    repair_mismatched: bool = False,
    verify_after_write: bool = False,
) -> DatasetReport:
    """Run a fully preflighted dry-run or an atomic, resumable cache migration."""
    report = DatasetReport(spec=spec)
    print(f"\n[{spec.name}] data_root={spec.data_root}")
    print(f"[{spec.name}] metric_cache_root={spec.metric_cache_root}")

    stage_started = time.perf_counter()
    cache_index = _build_cache_index(spec.metric_cache_root, workers)
    report.cache_files = len(cache_index.items)
    report.errors.extend(cache_index.errors[:MAX_REPORTED_ERRORS])
    print(
        f"[{spec.name}] cache_index_source={cache_index.source} "
        f"cache_files={report.cache_files} ignored_metadata={cache_index.ignored_files} "
        f"elapsed={time.perf_counter() - stage_started:.2f}s"
    )
    if report.errors:
        _print_report(report, apply)
        return report

    stage_started = time.perf_counter()
    source_index = _collect_source_fields(
        spec=spec,
        cache_index=cache_index,
        workers=workers,
        progress_every=progress_every,
        report=report,
    )
    print(
        f"[{spec.name}] raw_logs_loaded={report.raw_logs} "
        f"raw_frames={report.raw_frames} elapsed={time.perf_counter() - stage_started:.2f}s"
    )
    if report.errors:
        _print_report(report, apply)
        return report

    inspections: list[CacheInspection] = []
    cache_workers = min(workers, MAX_CACHE_PROCESSES)
    work_items = [
        (item, source_index[key], repair_mismatched)
        for key, item in sorted(cache_index.items.items())
    ]
    stage_started = time.perf_counter()
    for completed, inspection in enumerate(
        _bounded_process_map(_inspect_cache_work_item, work_items, cache_workers), start=1
    ):
        if inspection.status == "current":
            report.already_current += 1
        elif inspection.status == "needs_patch":
            report.needs_patch += 1
            inspections.append(inspection)
        else:
            _append_error(
                report.errors,
                inspection.error or f"cache inspection failed: {inspection.item.path}",
            )
        if progress_every > 0 and completed % progress_every == 0:
            _print_progress(spec, "metric_caches_inspected", completed, len(work_items))
    print(
        f"[{spec.name}] metric_caches_inspected={len(work_items)} "
        f"elapsed={time.perf_counter() - stage_started:.2f}s"
    )

    if report.errors or not apply:
        _print_report(report, apply)
        return report

    patch_items = [
        (inspection, repair_mismatched, verify_after_write)
        for inspection in inspections
    ]
    stage_started = time.perf_counter()
    for completed, result in enumerate(
        _bounded_process_map(_patch_cache_work_item, patch_items, cache_workers), start=1
    ):
        if result.status == "patched":
            report.patched += 1
        elif result.status == "raced_current":
            report.raced_current += 1
        else:
            _append_error(report.errors, f"{result.path}: {result.error}")
        if progress_every > 0 and completed % progress_every == 0:
            _print_progress(spec, "metric_caches_patched", completed, len(patch_items))
    print(
        f"[{spec.name}] metric_caches_patch_attempts={len(patch_items)} "
        f"elapsed={time.perf_counter() - stage_started:.2f}s"
    )

    _print_report(report, apply)
    return report


def _print_report(report: DatasetReport, apply: bool) -> None:
    """Print a compact, machine-scannable summary without hiding failures."""
    print(
        f"[{report.spec.name}] mode={'APPLY' if apply else 'DRY-RUN'} "
        f"raw_logs={report.raw_logs} raw_frames={report.raw_frames} "
        f"cache_files={report.cache_files} current={report.already_current} "
        f"needs_patch={report.needs_patch} patched={report.patched} "
        f"raced_current={report.raced_current} errors={len(report.errors)}"
    )
    for error in report.errors:
        print(f"[{report.spec.name}] ERROR: {error}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI while keeping cache mutation opt-in and scoped."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", help="MOGO manifest; reads data_root and metric_cache_path")
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        help="manifest dataset name to process; repeatable",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="explicitly select all manifest datasets; required for manifest --apply without --name",
    )
    parser.add_argument("--data-root", help="single dataset root without --manifest")
    parser.add_argument(
        "--data-split",
        default=DEFAULT_DATA_SPLIT,
        help=f"raw navsim_logs split used to build the metric cache (default: {DEFAULT_DATA_SPLIT})",
    )
    parser.add_argument(
        "--metric-cache-root", help="single metric-cache root without --manifest"
    )
    parser.add_argument("--dataset-name", default="direct", help="label for direct mode")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "I/O worker count for raw logs, fallback cache traversal, and metric caches "
            f"(default: {DEFAULT_WORKERS})"
        ),
    )
    parser.add_argument(
        "--progress-every", type=int, default=1000, help="progress interval in files; 0 disables"
    )
    parser.add_argument(
        "--repair-mismatched",
        action="store_true",
        help="overwrite non-empty new fields that differ from the current raw PKL",
    )
    parser.add_argument(
        "--verify-after-write",
        action="store_true",
        help="re-open every changed cache and verify the two patched fields",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="atomically rewrite patchable cache files; default is dry-run only",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if Path(args.data_split).name != args.data_split or args.data_split in {".", ".."}:
        parser.error("--data-split must be one directory name")
    if args.all and args.names:
        parser.error("--all cannot be combined with --name")
    if args.apply and args.manifest and not args.names and not args.all:
        parser.error("manifest --apply requires at least one --name or explicit --all")

    try:
        specs = _resolve_specs(args)
    except MetricCachePatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not specs:
        print("ERROR: no datasets selected", file=sys.stderr)
        return 2

    print(
        f"mode={'APPLY' if args.apply else 'DRY-RUN'} datasets={len(specs)} "
        f"data_split={args.data_split} workers={args.workers} repair_mismatched={args.repair_mismatched} "
        f"verify_after_write={args.verify_after_write} cache_processes={min(args.workers, MAX_CACHE_PROCESSES)}"
    )
    successful = 0
    for spec in specs:
        report = process_dataset(
            spec=spec,
            apply=args.apply,
            workers=args.workers,
            progress_every=args.progress_every,
            repair_mismatched=args.repair_mismatched,
            verify_after_write=args.verify_after_write,
        )
        successful += int(report.passed)

    print(f"\nsummary successful={successful}/{len(specs)}")
    if not args.apply:
        print("DRY-RUN: no metric-cache file was written. Add --apply only after reviewing errors=0.")
    return 0 if successful == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
