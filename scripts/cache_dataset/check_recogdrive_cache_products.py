#!/usr/bin/env python3
"""Check ReCogDrive cache products against the active raw-scene selection.

The checker separates two scopes that must not be conflated:

* expected: tokens selected from raw PKLs by the configured SceneFilter. Missing
  directories or required products here make the cache incomplete for training.
* extra: token directories left in the cache root but not selected by the current
  SceneFilter. Missing products here are reported as stale-cache information and do
  not fail the check unless ``--fail-on-extra`` is used.

No cache file is modified. Use ``--report-json`` to save the complete issue lists.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from navsim.common.dataclasses import SceneFilter


DEFAULT_MANIFEST = (
    REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml"
)
DEFAULT_SCENE_FILTER_CONFIG = (
    REPO_ROOT
    / "navsim/planning/script/config/common/train_test_split/scene_filter/mogo_10hz.yaml"
)

BASE_REQUIRED_FILES = (
    "qwen3_feature.gz",
    "recogdrive_ego_feature_v1.gz",
    "hdmap_vector_feature_v2.gz",
    "trajectory_target.gz",
)
OBJECT_FILE_BY_MODE = {
    "temporal": "object_temporal_feature_v1.gz",
    "mlp": "object_vector_feature_v1.gz",
}
KNOWN_PRODUCT_FILES = (
    "qwen3_feature.gz",
    "recogdrive_ego_feature_v1.gz",
    "hdmap_vector_feature.gz",
    "hdmap_vector_feature_v2.gz",
    "object_temporal_feature_v1.gz",
    "object_vector_feature_v1.gz",
    "trajectory_target.gz",
)


@dataclass(frozen=True, order=True)
class SampleKey:
    log_name: str
    token: str

    def as_text(self) -> str:
        return f"{self.log_name}/{self.token}"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_root: Path
    cache_root: Path


@dataclass
class ExpectedIndex:
    samples: Set[SampleKey] = field(default_factory=set)
    raw_log_count: int = 0
    raw_frame_count: int = 0
    errors: List[str] = field(default_factory=list)
    duplicate_tokens: List[str] = field(default_factory=list)


@dataclass
class CacheIndex:
    paths: Dict[SampleKey, Path] = field(default_factory=dict)
    products: Dict[SampleKey, Set[str]] = field(default_factory=dict)
    empty_products: Dict[SampleKey, Set[str]] = field(default_factory=dict)
    duplicate_tokens: List[str] = field(default_factory=list)
    ignored_directories: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProductIssue:
    sample: str
    reason: str


@dataclass
class ScopeReport:
    total_samples: int
    missing_directories: List[str]
    valid_counts: Dict[str, int]
    issues: Dict[str, List[ProductIssue]]


@dataclass
class DatasetReport:
    name: str
    data_root: str
    cache_root: str
    raw_logs: int
    raw_frames: int
    expected_samples: int
    cache_directories: int
    extra_samples: int
    extra_sample_paths: List[str]
    required_files: List[str]
    expected: ScopeReport
    extra: ScopeReport
    errors: List[str]
    duplicate_expected_tokens: List[str]
    duplicate_cache_tokens: List[str]
    ignored_cache_directories: List[str]
    passed: bool


def _deduplicate(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_scene_filter(path: Path) -> SceneFilter:
    if not path.is_file():
        raise ValueError(f"scene-filter config not found: {path}")
    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"scene-filter config must be a mapping: {path}")
    values = dict(raw_config)
    values.pop("_target_", None)
    values.pop("_convert_", None)
    return SceneFilter(**values)


def _iter_log_files(log_root: Path, scene_filter: SceneFilter) -> List[Path]:
    if not log_root.is_dir():
        return []
    paths = sorted(path for path in log_root.iterdir() if path.is_file() and path.suffix == ".pkl")
    if scene_filter.log_names is not None:
        selected_names = set(scene_filter.log_names)
        paths = [path for path in paths if path.stem in selected_names]
    return paths


def build_expected_index(log_root: Path, scene_filter: SceneFilter) -> ExpectedIndex:
    """Reproduce ``filter_scenes`` token selection without constructing Scene objects."""
    index = ExpectedIndex()
    paths = _iter_log_files(log_root, scene_filter)
    if not paths:
        index.errors.append(f"no PKL logs found under {log_root}")
        return index

    allowed_tokens = set(scene_filter.tokens) if scene_filter.tokens is not None else None
    token_owner: Dict[str, SampleKey] = {}
    stop_loading = False
    for path in paths:
        try:
            with path.open("rb") as file_handle:
                frames = pickle.load(file_handle)
            if not isinstance(frames, (list, tuple)):
                raise TypeError(f"expected list/tuple, got {type(frames).__name__}")
            index.raw_log_count += 1
            index.raw_frame_count += len(frames)
            for start in range(0, len(frames), scene_filter.frame_interval):
                window = frames[
                    start : start + scene_filter.num_raw_frames : scene_filter.frame_stride
                ]
                if len(window) < scene_filter.num_frames:
                    continue
                current = window[scene_filter.num_history_frames - 1]
                if not isinstance(current, Mapping):
                    raise TypeError(
                        f"frame {start + scene_filter.num_history_frames - 1} is not a mapping"
                    )
                roadblock_ids = current.get("roadblock_ids") or []
                if scene_filter.has_route and len(roadblock_ids) == 0:
                    continue
                token = str(current["token"])
                if allowed_tokens is not None and token not in allowed_tokens:
                    continue
                first_frame = window[0]
                log_name = (
                    str(first_frame.get("log_name") or path.stem)
                    if isinstance(first_frame, Mapping)
                    else path.stem
                )
                key = SampleKey(log_name=log_name, token=token)
                previous = token_owner.get(token)
                if previous is not None and previous != key:
                    if len(index.duplicate_tokens) < 100:
                        index.duplicate_tokens.append(
                            f"{token}: {previous.log_name}, {key.log_name}"
                        )
                    continue
                token_owner[token] = key
                index.samples.add(key)
                if (
                    scene_filter.max_scenes is not None
                    and len(index.samples) >= scene_filter.max_scenes
                ):
                    stop_loading = True
                    break
        except Exception as exc:
            if len(index.errors) < 100:
                index.errors.append(f"{path}: {type(exc).__name__}: {exc}")
        if stop_loading:
            break
    return index


def build_cache_index(cache_root: Path, check_nonempty: bool = False) -> CacheIndex:
    index = CacheIndex()
    if not cache_root.is_dir():
        index.ignored_directories.append(f"cache root does not exist: {cache_root}")
        return index

    token_owner: Dict[str, SampleKey] = {}
    with os.scandir(cache_root) as log_entries:
        for log_entry in log_entries:
            if not log_entry.is_dir():
                continue
            if log_entry.name == "metadata" or log_entry.name.startswith("."):
                continue
            found_token_directory = False
            with os.scandir(log_entry.path) as token_entries:
                for token_entry in token_entries:
                    if not token_entry.is_dir():
                        continue
                    found_token_directory = True
                    key = SampleKey(log_name=log_entry.name, token=token_entry.name)
                    token_path = Path(token_entry.path)
                    index.paths[key] = token_path
                    product_names: Set[str] = set()
                    empty_names: Set[str] = set()
                    with os.scandir(token_path) as product_entries:
                        for product_entry in product_entries:
                            if not product_entry.is_file():
                                continue
                            product_names.add(product_entry.name)
                            if check_nonempty and product_entry.stat().st_size <= 0:
                                empty_names.add(product_entry.name)
                    index.products[key] = product_names
                    if empty_names:
                        index.empty_products[key] = empty_names
                    previous = token_owner.get(key.token)
                    if previous is not None and previous != key:
                        if len(index.duplicate_tokens) < 100:
                            index.duplicate_tokens.append(
                                f"{key.token}: {previous.log_name}, {key.log_name}"
                            )
                    else:
                        token_owner[key.token] = key
            if not found_token_directory and len(index.ignored_directories) < 100:
                index.ignored_directories.append(log_entry.name)
    return index


def _count_present(
    samples: Iterable[SampleKey], cache_index: CacheIndex, file_name: str
) -> int:
    return sum(
        file_name in cache_index.products.get(sample, set())
        and file_name not in cache_index.empty_products.get(sample, set())
        for sample in samples
    )


def resolve_required_files(
    args: argparse.Namespace,
    expected_samples: Set[SampleKey],
    cache_index: CacheIndex,
) -> Tuple[List[str], str]:
    required = [args.hidden_state_file, *BASE_REQUIRED_FILES[1:]]
    object_note = "none"
    if args.object_feature == "auto":
        coverage = {
            mode: _count_present(expected_samples, cache_index, file_name)
            for mode, file_name in OBJECT_FILE_BY_MODE.items()
        }
        selected_mode = max(coverage, key=coverage.get)
        if coverage[selected_mode] > 0:
            required.append(OBJECT_FILE_BY_MODE[selected_mode])
            object_note = f"{selected_mode} ({coverage[selected_mode]}/{len(expected_samples)})"
        if sum(count > 0 for count in coverage.values()) > 1:
            object_note += f"; multiple object products observed: {coverage}"
    elif args.object_feature != "none":
        required.append(OBJECT_FILE_BY_MODE[args.object_feature])
        object_note = args.object_feature
    required.extend(args.required_file or [])
    return _deduplicate(required), object_note


def inspect_scope(
    samples: Set[SampleKey],
    cache_index: CacheIndex,
    product_files: Sequence[str],
) -> ScopeReport:
    missing_directories: List[str] = []
    valid_counts = {file_name: 0 for file_name in product_files}
    issues: Dict[str, List[ProductIssue]] = {file_name: [] for file_name in product_files}
    for sample in sorted(samples):
        token_path = cache_index.paths.get(sample)
        if token_path is None:
            missing_directories.append(sample.as_text())
            for file_name in product_files:
                issues[file_name].append(ProductIssue(sample.as_text(), "missing_directory"))
            continue
        for file_name in product_files:
            if file_name not in cache_index.products.get(sample, set()):
                issues[file_name].append(ProductIssue(sample.as_text(), "missing"))
            elif file_name in cache_index.empty_products.get(sample, set()):
                issues[file_name].append(ProductIssue(sample.as_text(), "empty"))
            else:
                valid_counts[file_name] += 1
    return ScopeReport(
        total_samples=len(samples),
        missing_directories=missing_directories,
        valid_counts=valid_counts,
        issues=issues,
    )


def _scope_to_dict(scope: ScopeReport) -> Dict[str, Any]:
    return {
        "total_samples": scope.total_samples,
        "missing_directories": scope.missing_directories,
        "valid_counts": scope.valid_counts,
        "issues": {
            file_name: [
                {"sample": issue.sample, "reason": issue.reason} for issue in file_issues
            ]
            for file_name, file_issues in scope.issues.items()
        },
    }


def _dataset_report_to_dict(report: DatasetReport) -> Dict[str, Any]:
    return {
        "name": report.name,
        "data_root": report.data_root,
        "cache_root": report.cache_root,
        "raw_logs": report.raw_logs,
        "raw_frames": report.raw_frames,
        "expected_samples": report.expected_samples,
        "cache_directories": report.cache_directories,
        "extra_samples": report.extra_samples,
        "extra_sample_paths": report.extra_sample_paths,
        "required_files": report.required_files,
        "expected": _scope_to_dict(report.expected),
        "extra": _scope_to_dict(report.extra),
        "errors": report.errors,
        "duplicate_expected_tokens": report.duplicate_expected_tokens,
        "duplicate_cache_tokens": report.duplicate_cache_tokens,
        "ignored_cache_directories": report.ignored_cache_directories,
        "passed": report.passed,
    }


def inspect_dataset(
    spec: DatasetSpec,
    split: str,
    scene_filter: SceneFilter,
    args: argparse.Namespace,
) -> Tuple[DatasetReport, str]:
    log_root = spec.data_root / "dataset" / "navsim_logs" / split
    expected_index = build_expected_index(log_root, scene_filter)
    cache_index = build_cache_index(spec.cache_root, check_nonempty=args.check_nonempty)
    required_files, object_note = resolve_required_files(
        args, expected_index.samples, cache_index
    )
    product_files = _deduplicate([*KNOWN_PRODUCT_FILES, *required_files])
    extra_samples = set(cache_index.paths) - expected_index.samples
    expected_scope = inspect_scope(expected_index.samples, cache_index, product_files)
    extra_scope = inspect_scope(extra_samples, cache_index, product_files)

    errors = list(expected_index.errors)
    if not spec.cache_root.is_dir():
        errors.append(f"cache root does not exist: {spec.cache_root}")
    required_missing = sum(len(expected_scope.issues[name]) for name in required_files)
    passed = (
        not errors
        and not expected_index.duplicate_tokens
        and not expected_scope.missing_directories
        and required_missing == 0
    )
    if args.fail_on_extra and extra_samples:
        passed = False
    return (
        DatasetReport(
            name=spec.name,
            data_root=str(spec.data_root),
            cache_root=str(spec.cache_root),
            raw_logs=expected_index.raw_log_count,
            raw_frames=expected_index.raw_frame_count,
            expected_samples=len(expected_index.samples),
            cache_directories=len(cache_index.paths),
            extra_samples=len(extra_samples),
            extra_sample_paths=[sample.as_text() for sample in sorted(extra_samples)],
            required_files=required_files,
            expected=expected_scope,
            extra=extra_scope,
            errors=errors,
            duplicate_expected_tokens=expected_index.duplicate_tokens,
            duplicate_cache_tokens=cache_index.duplicate_tokens,
            ignored_cache_directories=cache_index.ignored_directories,
            passed=passed,
        ),
        object_note,
    )


def _print_examples(title: str, values: Sequence[str], limit: int) -> None:
    if not values:
        return
    print(f"  {title} examples ({min(len(values), limit)}/{len(values)}):")
    for value in values[:limit]:
        print(f"    {value}")


def print_dataset_report(
    report: DatasetReport, object_note: str, show_limit: int
) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"\n[{report.name}] status={status}")
    print(f"  data_root={report.data_root}")
    print(f"  cache_root={report.cache_root}")
    print(
        f"  raw_logs={report.raw_logs} raw_frames={report.raw_frames} "
        f"expected={report.expected_samples} cache_dirs={report.cache_directories} "
        f"extra_stale={report.extra_samples}"
    )
    print(f"  object_feature={object_note}")
    print("  product coverage (expected | extra/stale):")
    for file_name in report.expected.valid_counts:
        required_mark = "required" if file_name in report.required_files else "observed"
        expected_valid = report.expected.valid_counts[file_name]
        extra_valid = report.extra.valid_counts[file_name]
        print(
            f"    {file_name}: {expected_valid}/{report.expected.total_samples} | "
            f"{extra_valid}/{report.extra.total_samples} ({required_mark})"
        )

    _print_examples(
        "missing expected directories", report.expected.missing_directories, show_limit
    )
    for file_name in report.required_files:
        file_issues = report.expected.issues[file_name]
        _print_examples(
            f"expected issues for {file_name}",
            [f"{issue.sample} ({issue.reason})" for issue in file_issues],
            show_limit,
        )
    _print_examples(
        "extra/stale directories", report.extra_sample_paths, show_limit
    )
    for error in report.errors:
        print(f"  ERROR: {error}", file=sys.stderr)
    _print_examples("duplicate expected tokens", report.duplicate_expected_tokens, show_limit)
    _print_examples("duplicate cache tokens", report.duplicate_cache_tokens, show_limit)
    _print_examples("ignored cache directories", report.ignored_cache_directories, show_limit)


def resolve_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    if args.manifest:
        if args.data_root or args.cache_root:
            raise ValueError("--manifest cannot be combined with --data-root/--cache-root")
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            raise ValueError(f"manifest not found: {manifest_path}")
        config = OmegaConf.load(manifest_path)
        datasets = list(config.datasets)
        selected_names = set(args.names or [])
        if selected_names:
            datasets = [dataset for dataset in datasets if str(dataset.name) in selected_names]
            missing_names = selected_names - {str(dataset.name) for dataset in datasets}
            if missing_names:
                raise ValueError(f"unknown dataset names: {sorted(missing_names)}")
        return [
            DatasetSpec(
                name=str(dataset.name),
                data_root=Path(str(dataset.data_root)),
                cache_root=Path(str(dataset.cache_path)),
            )
            for dataset in datasets
        ]

    if not args.data_root or not args.cache_root:
        raise ValueError("provide --manifest, or both --data-root and --cache-root")
    if args.names:
        raise ValueError("--name is only supported with --manifest")
    return [
        DatasetSpec(
            name=args.dataset_name,
            data_root=Path(args.data_root),
            cache_root=Path(args.cache_root),
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--name", action="append", dest="names")
    parser.add_argument("--data-root")
    parser.add_argument("--cache-root")
    parser.add_argument("--dataset-name", default="direct")
    parser.add_argument(
        "--split",
        default="trainval",
        help=(
            "raw navsim_logs split; must match train_test_split.data_split used "
            "while caching (default: trainval)"
        ),
    )
    parser.add_argument(
        "--scene-filter-config", default=str(DEFAULT_SCENE_FILTER_CONFIG)
    )
    parser.add_argument("--hidden-state-file", default="qwen3_feature.gz")
    parser.add_argument(
        "--object-feature",
        choices=("auto", "none", "temporal", "mlp"),
        default="auto",
    )
    parser.add_argument(
        "--required-file",
        action="append",
        default=[],
        help="additional required product file; repeatable",
    )
    parser.add_argument("--show-limit", type=int, default=20)
    parser.add_argument("--report-json")
    parser.add_argument("--fail-on-extra", action="store_true")
    parser.add_argument(
        "--check-nonempty",
        action="store_true",
        help="also stat product sizes and treat zero-byte files as invalid",
    )
    args = parser.parse_args()
    if args.manifest is None and not args.data_root and not args.cache_root:
        args.manifest = str(DEFAULT_MANIFEST)

    if args.show_limit < 0:
        parser.error("--show-limit must be non-negative")
    try:
        specs = resolve_dataset_specs(args)
        scene_filter_path = Path(args.scene_filter_config)
        scene_filter = load_scene_filter(scene_filter_path)
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not specs:
        print("ERROR: no datasets selected", file=sys.stderr)
        return 2

    selection_metadata = {
        "split": args.split,
        "scene_filter_config": str(scene_filter_path),
        "scene_filter": asdict(scene_filter),
    }
    print(
        f"selection split={args.split} scene_filter={scene_filter_path} "
        f"history={scene_filter.num_history_frames} "
        f"future={scene_filter.num_future_frames} "
        f"stride={scene_filter.frame_stride} "
        f"interval={scene_filter.frame_interval} "
        f"has_route={scene_filter.has_route}"
    )

    reports: List[DatasetReport] = []
    for spec in specs:
        report, object_note = inspect_dataset(spec, args.split, scene_filter, args)
        reports.append(report)
        print_dataset_report(report, object_note, args.show_limit)

    if args.report_json:
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "selection": selection_metadata,
                    "datasets": [_dataset_report_to_dict(report) for report in reports],
                },
                file_handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nreport_json={output_path}")

    passed = sum(report.passed for report in reports)
    print(f"\nsummary passed={passed}/{len(reports)}")
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
