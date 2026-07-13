#!/usr/bin/env python3
"""并行检查 MOGO NAVSIM 日志中的 obj 丢帧。

检测两类异常：
1. ``empty_obj_frame``：一个短暂的空 obj 区间被前后非空帧夹住；
2. ``track_gap``：同一个 ``track_token`` 短暂消失后又重新出现。

默认扫描 ``mogo_multi.yaml`` 中所有数据集的 train/val/test，并将异常帧写入 CSV：

  python scripts/cache_dataset/check_mogo_obj_frame_drop.py \
      --output mogo_obj_frame_drops.csv

只检查他车（vehicle），并允许发现连续两帧的短暂丢失：

  python scripts/cache_dataset/check_mogo_obj_frame_drop.py \
      --classes vehicle --max-gap 2 --workers 8 \
      --output mogo_vehicle_frame_drops.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "navsim/planning/script/config/common/mogo_dataset/mogo_multi.yaml"
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
CSV_FIELDS = (
    "dataset",
    "city",
    "split",
    "log_file",
    "log_name",
    "frame_index",
    "frame_token",
    "timestamp",
    "issue_types",
    "obj_count",
    "prev_obj_count",
    "next_obj_count",
    "missing_track_count",
    "missing_track_tokens",
)


@dataclass(frozen=True)
class LogTask:
    """一个可独立扫描的日志文件。"""

    dataset: str
    city: str
    split: str
    path: Path


@dataclass
class ScanResult:
    """单个日志文件的扫描结果。"""

    task: LogTask
    frame_count: int
    records: list[dict[str, Any]]
    error: str | None = None


def _safe_len(value: Any) -> int | None:
    """字段不可取长度时返回 None，而不是中断整个数据集扫描。"""
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def _frame_objects(
    frame: Any,
    selected_classes: frozenset[str] | None,
) -> tuple[int, dict[str, str], bool]:
    """返回 (obj 数量, track->类别, 注解是否合法)。"""
    if not isinstance(frame, Mapping):
        return 0, {}, False

    anns = frame.get("anns")
    if not isinstance(anns, Mapping):
        return 0, {}, False

    annotation_keys = (
        "gt_boxes",
        "gt_names",
        "gt_velocity_3d",
        "instance_tokens",
        "track_tokens",
    )
    lengths = {key: _safe_len(anns.get(key)) for key in annotation_keys}
    valid = all(length is not None for length in lengths.values()) and len(set(lengths.values())) == 1

    boxes = anns.get("gt_boxes")
    names = anns.get("gt_names")
    track_tokens = anns.get("track_tokens")
    box_count = _safe_len(boxes) or 0
    name_count = _safe_len(names) or 0
    track_count = _safe_len(track_tokens) or 0
    paired_object_count = min(box_count, name_count)

    if selected_classes is None:
        obj_count = box_count
    else:
        obj_count = sum(str(names[index]) in selected_classes for index in range(paired_object_count))

    tracks: dict[str, str] = {}
    for index in range(min(paired_object_count, track_count)):
        name = str(names[index])
        if selected_classes is not None and name not in selected_classes:
            continue
        token = str(track_tokens[index])
        if token:
            tracks[token] = name
    return obj_count, tracks, valid


def find_drop_records(
    frames: Sequence[Any],
    task: LogTask,
    selected_classes: frozenset[str] | None,
    max_gap: int,
) -> list[dict[str, Any]]:
    """分析一个日志的连续帧，返回一帧一条的异常记录。"""
    obj_counts: list[int] = []
    tracks_per_frame: list[dict[str, str]] = []
    issue_types: dict[int, set[str]] = defaultdict(set)
    missing_tracks: dict[int, set[str]] = defaultdict(set)

    for frame_index, frame in enumerate(frames):
        obj_count, tracks, valid = _frame_objects(frame, selected_classes)
        obj_counts.append(obj_count)
        tracks_per_frame.append(tracks)
        if not valid:
            issue_types[frame_index].add("invalid_annotations")

    # 空 obj 只在短空洞前后均有 obj 时判异常，避免把日志边界和正常无目标路段
    # 误报为丢帧。
    frame_index = 0
    while frame_index < len(frames):
        if obj_counts[frame_index] != 0:
            frame_index += 1
            continue
        gap_start = frame_index
        while frame_index < len(frames) and obj_counts[frame_index] == 0:
            frame_index += 1
        gap_end = frame_index
        gap_size = gap_end - gap_start
        if (
            gap_size <= max_gap
            and gap_start > 0
            and gap_end < len(frames)
            and obj_counts[gap_start - 1] > 0
            and obj_counts[gap_end] > 0
        ):
            for missing_index in range(gap_start, gap_end):
                issue_types[missing_index].add("empty_obj_frame")

    observations: dict[str, list[int]] = defaultdict(list)
    for current_index, tracks in enumerate(tracks_per_frame):
        for track_token in tracks:
            observations[track_token].append(current_index)

    for track_token, indices in observations.items():
        for previous_index, next_index in zip(indices, indices[1:]):
            gap_size = next_index - previous_index - 1
            if not 0 < gap_size <= max_gap:
                continue
            for missing_index in range(previous_index + 1, next_index):
                issue_types[missing_index].add("track_gap")
                missing_tracks[missing_index].add(track_token)

    records: list[dict[str, Any]] = []
    for current_index in sorted(issue_types):
        frame = frames[current_index] if isinstance(frames[current_index], Mapping) else {}
        tokens = sorted(missing_tracks[current_index])
        records.append(
            {
                "dataset": task.dataset,
                "city": task.city,
                "split": task.split,
                "log_file": str(task.path),
                "log_name": frame.get("log_name", task.path.stem),
                "frame_index": current_index,
                "frame_token": frame.get("token", ""),
                "timestamp": frame.get("timestamp", ""),
                "issue_types": "|".join(sorted(issue_types[current_index])),
                "obj_count": obj_counts[current_index],
                "prev_obj_count": obj_counts[current_index - 1] if current_index > 0 else "",
                "next_obj_count": obj_counts[current_index + 1] if current_index + 1 < len(frames) else "",
                "missing_track_count": len(tokens),
                "missing_track_tokens": json.dumps(tokens, ensure_ascii=True),
            }
        )
    return records


def scan_log(
    task: LogTask,
    selected_classes: frozenset[str] | None,
    max_gap: int,
) -> ScanResult:
    """读取并扫描单个 pkl；异常被收敛到结果中，不影响其它线程。"""
    try:
        with task.path.open("rb") as file:
            frames = pickle.load(file)
        if not isinstance(frames, (list, tuple)):
            raise TypeError(f"expected list/tuple, got {type(frames).__name__}")
        records = find_drop_records(frames, task, selected_classes, max_gap)
        return ScanResult(task=task, frame_count=len(frames), records=records)
    except Exception as exc:  # 数据审计应继续扫描其它日志，并在最终状态中明确失败。
        return ScanResult(
            task=task,
            frame_count=0,
            records=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def load_manifest_datasets(manifest: Path, names: Sequence[str] | None) -> list[dict[str, str]]:
    """解析 manifest，并按可选数据集名称过滤。"""
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest}")

    config = OmegaConf.load(manifest)
    datasets = [
        {
            "name": str(dataset.name),
            "city": str(dataset.get("city", "")),
            "data_root": str(dataset.data_root),
        }
        for dataset in config.datasets
    ]
    if not names:
        return datasets

    wanted = set(names)
    selected = [dataset for dataset in datasets if dataset["name"] in wanted]
    missing = wanted - {dataset["name"] for dataset in selected}
    if missing:
        raise ValueError(f"manifest 中不存在数据集: {sorted(missing)}")
    return selected


def discover_tasks(
    datasets: Sequence[Mapping[str, str]],
    splits: Sequence[str],
) -> tuple[list[LogTask], list[Path]]:
    """发现待扫描日志；返回任务及不存在的 split 目录。"""
    tasks: list[LogTask] = []
    missing_dirs: list[Path] = []
    for dataset in datasets:
        logs_root = Path(dataset["data_root"]) / "dataset" / "navsim_logs"
        for split in splits:
            split_dir = logs_root / split
            if not split_dir.is_dir():
                missing_dirs.append(split_dir)
                continue
            tasks.extend(
                LogTask(
                    dataset=dataset["name"],
                    city=dataset["city"],
                    split=split,
                    path=path,
                )
                for path in sorted(split_dir.glob("*.pkl"))
            )
    return tasks, missing_dirs


def iter_scan_results(
    tasks: Iterable[LogTask],
    selected_classes: frozenset[str] | None,
    max_gap: int,
    workers: int,
) -> Iterable[ScanResult]:
    """有界提交线程任务，避免大数据集为所有文件同时创建 Future。"""
    task_iterator = iter(tasks)
    max_pending = workers * 2
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="obj-drop") as executor:
        pending: dict[Future[ScanResult], LogTask] = {}

        def submit_next() -> bool:
            try:
                task = next(task_iterator)
            except StopIteration:
                return False
            pending[executor.submit(scan_log, task, selected_classes, max_gap)] = task
            return True

        for _ in range(max_pending):
            if not submit_next():
                break

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                yield future.result()
                submit_next()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="MOGO 多数据集 manifest")
    parser.add_argument("--name", action="append", dest="names", help="只扫描指定数据集，可重复传入")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="要扫描的 navsim_logs 子目录")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="只检查指定 obj 类别，如 vehicle；默认检查全部类别",
    )
    parser.add_argument("--max-gap", type=int, default=1, help="判定为短暂丢失的最大连续帧数")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行读取 pkl 的线程数")
    parser.add_argument("--output", type=Path, default=Path("mogo_obj_frame_drops.csv"), help="异常帧 CSV")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_gap < 1:
        print("ERROR: --max-gap 必须 >= 1", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("ERROR: --workers 必须 >= 1", file=sys.stderr)
        return 2

    try:
        datasets = load_manifest_datasets(args.manifest, args.names)
    except (FileNotFoundError, ValueError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    tasks, missing_dirs = discover_tasks(datasets, args.splits)
    for missing_dir in missing_dirs:
        print(f"[WARN] split 目录不存在，跳过: {missing_dir}", file=sys.stderr)
    if not tasks:
        print("ERROR: 未发现可扫描的 pkl 文件", file=sys.stderr)
        return 2

    selected_classes = frozenset(args.classes) if args.classes else None
    class_text = ",".join(sorted(selected_classes)) if selected_classes else "ALL"
    print(
        f"扫描日志={len(tasks)} 数据集={len(datasets)} splits={','.join(args.splits)} "
        f"classes={class_text} max_gap={args.max_gap} workers={args.workers}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    scanned_files = 0
    scanned_frames = 0
    anomaly_frames = 0
    errors: list[tuple[str, str]] = []

    try:
        with temporary_output.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for result in iter_scan_results(
                tasks,
                selected_classes=selected_classes,
                max_gap=args.max_gap,
                workers=args.workers,
            ):
                scanned_files += 1
                scanned_frames += result.frame_count
                if result.error:
                    errors.append((str(result.task.path), result.error))
                else:
                    writer.writerows(result.records)
                    anomaly_frames += len(result.records)
                if scanned_files % 100 == 0 or scanned_files == len(tasks):
                    print(
                        f"[PROGRESS] files={scanned_files}/{len(tasks)} "
                        f"frames={scanned_frames} anomaly_frames={anomaly_frames} errors={len(errors)}"
                    )
        temporary_output.replace(args.output)
    except KeyboardInterrupt:
        print(f"\n扫描被中断，未完成的临时结果保留在: {temporary_output}", file=sys.stderr)
        return 130

    print("\n===== 扫描汇总 =====")
    print(f"日志文件: {scanned_files}")
    print(f"数据帧  : {scanned_frames}")
    print(f"异常帧  : {anomaly_frames}")
    print(f"读取失败: {len(errors)}")
    print(f"结果文件: {args.output.resolve()}")
    if errors:
        print("\n读取失败日志（最多显示 20 个）:", file=sys.stderr)
        for path, error in errors[:20]:
            print(f"  {path}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
