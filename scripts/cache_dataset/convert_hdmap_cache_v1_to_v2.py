#!/usr/bin/env python3
"""Convert lane-only ReCogDrive HDMap v1 caches to the v2 tensor schema.

This fast path is intentionally limited to datasets whose raw PKL logs contain no
non-empty ``traffic_light_states`` field.  Every selected dataset is audited before
any cache file is written.  If rich traffic-control data or an unreadable/unsupported
log is found, conversion for that dataset is refused.

The source ``hdmap_vector_feature.gz`` files are never modified.  Conversion keeps
the cached center/left/right lane geometry, discards the legacy lane-linked light
state, adds the v2 NONE control fields, and writes
``hdmap_vector_feature_v2.gz`` atomically.  Existing v2 files are skipped.

Examples:

  # Audit and validate conversion only (default dry-run).
  python "scripts/cache_dataset/convert_hdmap_cache_v1_to_v2.py" \
    --manifest "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml" \
    --name beijing_0509_0626-0627 --workers 24

  # Audit first, then write missing v2 sidecars concurrently.
  python "scripts/cache_dataset/convert_hdmap_cache_v1_to_v2.py" \
    --manifest "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml" \
    --name beijing_0509_0626-0627 --workers 24 --apply
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import threading
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

import torch


SOURCE_FEATURE_NAME = "hdmap_vector_feature.gz"
TARGET_FEATURE_NAME = "hdmap_vector_feature_v2.gz"
RICH_FIELD_NAME = "traffic_light_states"

POLYGON_LANE = 0
POLYGON_LANE_CONNECTOR = 1
POLYGON_UNKNOWN = 2
ROUTE_OFF = 0
ROUTE_ON = 1
ROUTE_UNKNOWN = 2
TL_NONE = 4
CONTROL_NONE = 0
DIRECTION_NONE = 0

V1_KEYS = (
    "map_point_position",
    "map_point_vector",
    "map_point_orientation",
    "map_polygon_center",
    "map_polygon_type",
    "map_polygon_on_route",
    "map_polygon_tl_status",
    "map_polygon_has_speed_limit",
    "map_polygon_speed_limit",
    "map_point_valid_mask",
    "map_valid_mask",
)
V2_KEYS = (
    "map_point_position",
    "map_point_vector",
    "map_point_orientation",
    "map_polygon_center",
    "map_polygon_type",
    "map_polygon_on_route",
    "map_polygon_tl_status",
    "map_polygon_control_type",
    "map_polygon_tl_direction",
    "map_polygon_has_speed_limit",
    "map_polygon_speed_limit",
    "map_point_valid_mask",
    "map_valid_mask",
)


class ConversionError(RuntimeError):
    """Raised when a source cache cannot be converted without guessing."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_root: Path
    cache_root: Path

    @property
    def log_root(self) -> Path:
        return self.data_root / "dataset" / "navsim_logs"


@dataclass
class LogFileAudit:
    path: Path
    frame_count: int = 0
    field_present_frames: int = 0
    rich_frames: int = 0
    rich_example: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DatasetAudit:
    total_files: int
    scanned_files: int = 0
    frame_count: int = 0
    field_present_frames: int = 0
    rich_frames: int = 0
    rich_examples: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    log_names: Set[str] = field(default_factory=set)

    @property
    def eligible(self) -> bool:
        return (
            self.total_files > 0
            and self.scanned_files == self.total_files
            and self.rich_frames == 0
            and not self.errors
        )


@dataclass
class CacheInventory:
    source_paths: List[Path] = field(default_factory=list)
    existing_v2: int = 0
    missing_both: int = 0
    token_dirs: int = 0
    unknown_log_names: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversionResult:
    path: Path
    status: str
    error: Optional[str] = None


T = TypeVar("T")
R = TypeVar("R")


def _bounded_thread_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int,
    max_pending_factor: int = 4,
) -> Iterator[R]:
    """Run a bounded, input-ordered thread map.

    Workers still execute concurrently, but results are yielded in input order. This
    makes fail-fast PKL audits deterministic while keeping the number of queued cache
    files bounded.
    """
    if workers <= 0:
        raise ValueError("workers must be positive")

    iterator = iter(items)
    executor = ThreadPoolExecutor(max_workers=workers)
    pending: deque[Future[R]] = deque()

    def fill() -> None:
        while len(pending) < workers * max_pending_factor:
            try:
                item = next(iterator)
            except StopIteration:
                return
            pending.append(executor.submit(fn, item))

    try:
        fill()
        while pending:
            future = pending.popleft()
            yield future.result()
            fill()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _iter_pickle_logs(log_root: Path) -> Iterator[Path]:
    for directory, _, file_names in os.walk(log_root):
        for file_name in file_names:
            if file_name.endswith(".pkl"):
                yield Path(directory) / file_name


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as raw_file:
        magic = raw_file.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as file_handle:
        return pickle.load(file_handle)


def _frames_from_log(payload: Any) -> Sequence[Any]:
    if isinstance(payload, (list, tuple)):
        return payload
    if isinstance(payload, Mapping):
        for key in ("frames", "scene_dict_list"):
            frames = payload.get(key)
            if isinstance(frames, (list, tuple)):
                return frames
        if "token" in payload and "ego2global_translation" in payload:
            return [payload]
    raise ConversionError(f"unsupported raw log payload type: {type(payload).__name__}")


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes, Mapping, Sequence)):
        return len(value) > 0
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size > 0
    numel = getattr(value, "numel", None)
    if callable(numel):
        return int(numel()) > 0
    try:
        return bool(value)
    except (TypeError, ValueError, RuntimeError):
        # Ambiguous objects are treated as rich rather than silently accepted.
        return True


def audit_log_file(path: Path) -> LogFileAudit:
    result = LogFileAudit(path=path)
    try:
        frames = _frames_from_log(_load_pickle(path))
        result.frame_count = len(frames)
        for frame_index, frame in enumerate(frames):
            if isinstance(frame, Mapping):
                field_present = RICH_FIELD_NAME in frame
                value = frame.get(RICH_FIELD_NAME)
            else:
                field_present = hasattr(frame, RICH_FIELD_NAME)
                value = getattr(frame, RICH_FIELD_NAME, None)
            if not field_present:
                continue
            result.field_present_frames += 1
            if _nonempty(value):
                result.rich_frames += 1
                if result.rich_example is None:
                    result.rich_example = f"{path}:frame={frame_index}"
    except Exception as exc:
        result.error = f"{path}: {type(exc).__name__}: {exc}"
    return result


def audit_dataset_logs(log_root: Path, workers: int) -> DatasetAudit:
    paths = sorted(_iter_pickle_logs(log_root)) if log_root.is_dir() else []
    audit = DatasetAudit(total_files=len(paths), log_names={path.stem for path in paths})
    if not paths:
        audit.errors.append(f"no PKL logs found under {log_root}")
        return audit

    for result in _bounded_thread_map(audit_log_file, paths, workers):
        audit.scanned_files += 1
        audit.frame_count += result.frame_count
        audit.field_present_frames += result.field_present_frames
        audit.rich_frames += result.rich_frames
        if result.rich_example and len(audit.rich_examples) < 5:
            audit.rich_examples.append(result.rich_example)
        if result.error and len(audit.errors) < 20:
            audit.errors.append(result.error)
        if result.rich_frames or result.error:
            # One positive/error is sufficient to refuse the entire dataset.
            # Stop scheduling more large PKLs while preserving fail-closed behavior.
            break
    return audit


def inspect_cache_root(cache_root: Path, audited_log_names: Set[str]) -> CacheInventory:
    inventory = CacheInventory()
    if not cache_root.is_dir():
        inventory.unknown_log_names.append(f"cache root does not exist: {cache_root}")
        return inventory

    with os.scandir(cache_root) as log_entries:
        for log_entry in log_entries:
            if not log_entry.is_dir() or log_entry.name == "metadata":
                continue
            if log_entry.name not in audited_log_names:
                if len(inventory.unknown_log_names) < 20:
                    inventory.unknown_log_names.append(log_entry.name)
                continue
            try:
                token_entries = os.scandir(log_entry.path)
            except FileNotFoundError:
                continue
            with token_entries:
                for token_entry in token_entries:
                    if not token_entry.is_dir():
                        continue
                    inventory.token_dirs += 1
                    token_path = Path(token_entry.path)
                    source_path = token_path / SOURCE_FEATURE_NAME
                    target_path = token_path / TARGET_FEATURE_NAME
                    if target_path.is_file():
                        inventory.existing_v2 += 1
                    elif source_path.is_file():
                        inventory.source_paths.append(source_path)
                    else:
                        inventory.missing_both += 1
    return inventory


def _require_tensor(data: Mapping[str, Any], key: str) -> torch.Tensor:
    value = data.get(key)
    if not torch.is_tensor(value):
        raise ConversionError(f"{key} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ConversionError(f"{key} must be stored on CPU, got {value.device}")
    return value


def convert_v1_feature_dict(source: Any) -> Dict[str, torch.Tensor]:
    """Convert one validated, lane-only v1 payload to the exact v2 schema."""
    if not isinstance(source, Mapping):
        raise ConversionError(f"cache payload must be a mapping, got {type(source).__name__}")
    missing = [key for key in V1_KEYS if key not in source]
    if missing:
        raise ConversionError(f"missing v1 keys: {missing}")
    if "map_polygon_control_type" in source or "map_polygon_tl_direction" in source:
        raise ConversionError("source already contains v2-only fields under the v1 file name")

    tensors = {key: _require_tensor(source, key) for key in V1_KEYS}
    valid_mask = tensors["map_valid_mask"].float()
    if valid_mask.ndim != 1:
        raise ConversionError(f"map_valid_mask must have shape [M], got {tuple(valid_mask.shape)}")
    max_tokens = int(valid_mask.shape[0])
    if max_tokens <= 0:
        raise ConversionError("map_valid_mask must contain at least one token slot")

    position = tensors["map_point_position"]
    if position.ndim != 4 or position.shape[0] != max_tokens or position.shape[1] != 3 or position.shape[-1] != 2:
        raise ConversionError(
            "map_point_position must have shape [M,3,P,2], "
            f"got {tuple(position.shape)}"
        )
    sample_points = int(position.shape[2])
    expected_shapes = {
        "map_point_vector": (max_tokens, 3, sample_points, 2),
        "map_point_orientation": (max_tokens, 3, sample_points),
        "map_polygon_center": (max_tokens, 3),
        "map_polygon_type": (max_tokens,),
        "map_polygon_on_route": (max_tokens,),
        "map_polygon_tl_status": (max_tokens,),
        "map_polygon_has_speed_limit": (max_tokens,),
        "map_polygon_speed_limit": (max_tokens,),
        "map_point_valid_mask": (max_tokens, sample_points),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(tensors[key].shape) != expected_shape:
            raise ConversionError(
                f"{key} must have shape {expected_shape}, got {tuple(tensors[key].shape)}"
            )

    valid = valid_mask > 0
    polygon_type = tensors["map_polygon_type"].long().clone()
    polygon_type[~valid] = POLYGON_UNKNOWN
    valid_types = polygon_type[valid]
    if valid_types.numel() and bool(
        ((valid_types < POLYGON_LANE) | (valid_types > POLYGON_UNKNOWN)).any()
    ):
        raise ConversionError("v1 cache contains a non-lane polygon type")

    old_route = tensors["map_polygon_on_route"].long()
    valid_routes = old_route[valid]
    if valid_routes.numel() and bool(
        ((valid_routes < ROUTE_OFF) | (valid_routes > ROUTE_ON)).any()
    ):
        raise ConversionError("v1 cache contains an unsupported route state")
    route = old_route.clone()
    route[~valid] = ROUTE_UNKNOWN
    # Old v1 conflated missing/unmatched route metadata with OFF_ROUTE.  If no
    # ON_ROUTE token survived, UNKNOWN is the only non-fabricated v2 meaning.
    if not bool((valid_routes == ROUTE_ON).any()):
        route[valid] = ROUTE_UNKNOWN

    converted: Dict[str, torch.Tensor] = {
        "map_point_position": tensors["map_point_position"].float(),
        "map_point_vector": tensors["map_point_vector"].float(),
        "map_point_orientation": tensors["map_point_orientation"].float(),
        "map_polygon_center": tensors["map_polygon_center"].float(),
        "map_polygon_type": polygon_type,
        "map_polygon_on_route": route,
        # v1 lane-linked light states are not stop-line control supervision.
        "map_polygon_tl_status": torch.full((max_tokens,), TL_NONE, dtype=torch.long),
        "map_polygon_control_type": torch.full(
            (max_tokens,), CONTROL_NONE, dtype=torch.long
        ),
        "map_polygon_tl_direction": torch.full(
            (max_tokens,), DIRECTION_NONE, dtype=torch.long
        ),
        "map_polygon_has_speed_limit": tensors["map_polygon_has_speed_limit"].bool(),
        "map_polygon_speed_limit": tensors["map_polygon_speed_limit"].float(),
        "map_point_valid_mask": tensors["map_point_valid_mask"].float(),
        "map_valid_mask": valid_mask,
    }
    if tuple(converted) != V2_KEYS:
        raise AssertionError("internal v2 key ordering mismatch")
    return converted


def _atomic_dump_gzip(path: Path, data: Mapping[str, torch.Tensor]) -> None:
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with gzip.open(temporary_path, "wb", compresslevel=1) as file_handle:
            pickle.dump(data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def convert_feature_file(source_path: Path, apply: bool) -> ConversionResult:
    target_path = source_path.with_name(TARGET_FEATURE_NAME)
    if target_path.is_file():
        return ConversionResult(source_path, "skipped_existing")
    try:
        with gzip.open(source_path, "rb") as file_handle:
            source = pickle.load(file_handle)
        converted = convert_v1_feature_dict(source)
        if not apply:
            return ConversionResult(source_path, "validated")
        # Recheck immediately before the atomic write in case another job won the race.
        if target_path.is_file():
            return ConversionResult(source_path, "skipped_existing")
        _atomic_dump_gzip(target_path, converted)
        return ConversionResult(source_path, "converted")
    except Exception as exc:
        return ConversionResult(
            source_path,
            "error",
            f"{type(exc).__name__}: {exc}",
        )


def _resolve_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    if args.manifest:
        if args.data_root or args.cache_root:
            raise ConversionError("--manifest cannot be combined with --data-root/--cache-root")
        from omegaconf import OmegaConf

        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise ConversionError(f"manifest not found: {manifest_path}")
        config = OmegaConf.load(manifest_path)
        datasets = list(config.datasets)
        wanted = set(args.names or [])
        if wanted:
            datasets = [dataset for dataset in datasets if str(dataset.name) in wanted]
            missing_names = wanted - {str(dataset.name) for dataset in datasets}
            if missing_names:
                raise ConversionError(f"unknown dataset names: {sorted(missing_names)}")
        return [
            DatasetSpec(
                name=str(dataset.name),
                data_root=Path(str(dataset.data_root)),
                cache_root=Path(str(dataset.cache_path)),
            )
            for dataset in datasets
        ]

    if not args.data_root or not args.cache_root:
        raise ConversionError("provide --manifest, or both --data-root and --cache-root")
    if args.names:
        raise ConversionError("--name is only supported with --manifest")
    return [
        DatasetSpec(
            name=args.dataset_name,
            data_root=Path(args.data_root),
            cache_root=Path(args.cache_root),
        )
    ]


def _print_audit(spec: DatasetSpec, audit: DatasetAudit) -> None:
    audit_state = "complete" if audit.scanned_files == audit.total_files else "early-stop"
    print(
        f"[{spec.name}] PKL audit state={audit_state} "
        f"files_scanned={audit.scanned_files}/{audit.total_files} "
        f"frames_scanned={audit.frame_count} "
        f"field_present_scanned={audit.field_present_frames} "
        f"rich_frames_observed={audit.rich_frames}"
    )
    for example in audit.rich_examples:
        print(f"[{spec.name}] rich example: {example}")
    for error in audit.errors:
        print(f"[{spec.name}] audit error: {error}", file=sys.stderr)


def process_dataset(
    spec: DatasetSpec,
    apply: bool,
    workers: int,
    audit_workers: int,
    progress_every: int,
) -> bool:
    print(f"\n[{spec.name}] data_root={spec.data_root}")
    print(f"[{spec.name}] cache_root={spec.cache_root}")
    audit = audit_dataset_logs(spec.log_root, audit_workers)
    _print_audit(spec, audit)
    if not audit.eligible:
        print(
            f"[{spec.name}] BLOCKED: dataset is not fully confirmed as rich-field-free; "
            "no v2 cache was written.",
            file=sys.stderr,
        )
        return False

    inventory = inspect_cache_root(spec.cache_root, audit.log_names)
    if inventory.unknown_log_names:
        print(
            f"[{spec.name}] BLOCKED: cache contains unaudited log directories: "
            f"{inventory.unknown_log_names}",
            file=sys.stderr,
        )
        return False
    if inventory.token_dirs == 0:
        print(
            f"[{spec.name}] BLOCKED: no audited token directories found in cache root.",
            file=sys.stderr,
        )
        return False
    print(
        f"[{spec.name}] cache tokens={inventory.token_dirs} "
        f"to_convert={len(inventory.source_paths)} existing_v2={inventory.existing_v2} "
        f"missing_v1_and_v2={inventory.missing_both}"
    )
    if inventory.missing_both:
        print(
            f"[{spec.name}] WARNING: {inventory.missing_both} token directories contain "
            "neither v1 nor v2 HDMap cache.",
            file=sys.stderr,
        )

    counts = {"validated": 0, "converted": 0, "skipped_existing": 0, "error": 0}
    errors: List[str] = []
    worker_fn = lambda path: convert_feature_file(path, apply=apply)
    for processed, result in enumerate(
        _bounded_thread_map(worker_fn, inventory.source_paths, workers), start=1
    ):
        counts[result.status] += 1
        if result.error and len(errors) < 20:
            errors.append(f"{result.path}: {result.error}")
        if progress_every > 0 and processed % progress_every == 0:
            print(
                f"[{spec.name}] processed={processed}/{len(inventory.source_paths)} "
                f"converted={counts['converted']} validated={counts['validated']} "
                f"errors={counts['error']}"
            )

    mode_result = counts["converted"] if apply else counts["validated"]
    print(
        f"[{spec.name}] {'converted' if apply else 'validated'}={mode_result} "
        f"raced_existing={counts['skipped_existing']} errors={counts['error']}"
    )
    for error in errors:
        print(f"[{spec.name}] conversion error: {error}", file=sys.stderr)
    return counts["error"] == 0 and inventory.missing_both == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", help="MOGO manifest; reads data_root and cache_path")
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        help="manifest dataset name to convert; repeatable (default: all)",
    )
    parser.add_argument("--data-root", help="single dataset root (without --manifest)")
    parser.add_argument("--cache-root", help="single cache root (without --manifest)")
    parser.add_argument("--dataset-name", default="direct", help="label for direct mode")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, max(4, os.cpu_count() or 4)),
        help="cache conversion thread count (default: min(32, CPU count))",
    )
    parser.add_argument(
        "--audit-workers",
        type=int,
        default=4,
        help="raw PKL audit thread count; keep low to bound memory (default: 4)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=1000, help="progress interval in files"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write missing v2 files; default is audit/validation only",
    )
    args = parser.parse_args()

    if args.workers <= 0 or args.audit_workers <= 0:
        parser.error("--workers and --audit-workers must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    try:
        specs = _resolve_specs(args)
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not specs:
        print("ERROR: no datasets selected", file=sys.stderr)
        return 2

    print(
        f"mode={'APPLY' if args.apply else 'DRY-RUN'} datasets={len(specs)} "
        f"workers={args.workers} audit_workers={args.audit_workers}"
    )
    successful = 0
    for spec in specs:
        successful += int(
            process_dataset(
                spec=spec,
                apply=args.apply,
                workers=args.workers,
                audit_workers=args.audit_workers,
                progress_every=args.progress_every,
            )
        )
    print(f"\nsummary successful={successful}/{len(specs)}")
    if not args.apply:
        print("DRY-RUN: no cache files were written. Add --apply after reviewing the audit.")
    return 0 if successful == len(specs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
