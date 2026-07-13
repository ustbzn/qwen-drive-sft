#!/usr/bin/env python3
"""保守插值修复 MOGO NAVSIM pkl 中整帧为空的障碍物注解。

只修复满足以下条件的内部空洞：
- 缺失区间整帧 ``gt_boxes`` 为空，且前后均有有效注解；
- 连续缺失帧数不超过 ``--max-gap``；
- 同一 ``track_token`` 在空洞两端都存在且类别一致；
- 端点时间跨度和推算速度不超过安全阈值。

box 中心、heading 和 velocity 会先从各帧 ego-local 坐标变换到全局坐标，按真实时间戳
插值，再变换回缺失帧的 ego-local 坐标。只在单侧出现的 track 不会被补入。

最简用法：只传数据集根目录，自动发现 train/val/test 下的 pkl。默认 dry-run，
报告自动写入 ``<data_root>/exp/obj_interpolation_reports/``：

  python scripts/cache_dataset/interpolate_mogo_empty_obj_frames.py \
      /path/to/dataset_root

确认 dry-run 报告后只加 ``--apply`` 写回；写回前自动创建逐文件备份：

  python scripts/cache_dataset/interpolate_mogo_empty_obj_frames.py \
      /path/to/dataset_root \
      --apply

高级用法仍可用 ``--pkl`` 或 ``--logs-dir`` 精确指定输入。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MARKER_KEY = "_obj_empty_frames_interpolated_v1"
ANNOTATION_KEYS = (
    "gt_boxes",
    "gt_names",
    "gt_velocity_3d",
    "instance_tokens",
    "track_tokens",
)
REPORT_FIELDS = (
    "pkl_path",
    "log_name",
    "frame_index",
    "frame_token",
    "timestamp",
    "track_token",
    "object_name",
    "previous_frame_index",
    "next_frame_index",
    "gap_size",
    "alpha",
    "endpoint_span_seconds",
    "inferred_speed_mps",
)


@dataclass(frozen=True)
class Observation:
    """一个原始帧中的单个障碍物观测。"""

    frame_index: int
    name: str
    track_token: str
    instance_token: str
    box: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class Addition:
    """计划追加到缺失帧的一条插值注解。"""

    frame_index: int
    name: str
    track_token: str
    instance_token: str
    box: np.ndarray
    velocity: np.ndarray


@dataclass
class RepairPlan:
    """单个 pkl 的修复计划；生成计划阶段不修改 frames。"""

    pkl_path: Path
    additions: dict[int, list[Addition]] = field(default_factory=lambda: defaultdict(list))
    report_rows: list[dict[str, Any]] = field(default_factory=list)
    empty_frames_before: int = 0
    eligible_runs: int = 0
    skipped_runs: Counter[str] = field(default_factory=Counter)
    skipped_tracks: Counter[str] = field(default_factory=Counter)

    @property
    def changed_frames(self) -> int:
        return len(self.additions)

    @property
    def added_objects(self) -> int:
        return sum(len(items) for items in self.additions.values())


@dataclass
class FileResult:
    """单文件处理结果。"""

    path: Path
    plan: RepairPlan | None = None
    applied: bool = False
    skipped_marker: bool = False
    error: str | None = None


def wrap_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def quat_to_yaw(quaternion: Sequence[float]) -> float:
    """四元数顺序为 [w, x, y, z]。"""
    if len(quaternion) != 4:
        raise ValueError(f"ego2global_rotation 长度应为 4，实际为 {len(quaternion)}")
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _ego_pose(frame: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"ego2global_translation shape 应为 (3,)，实际为 {translation.shape}")
    return translation, quat_to_yaw(frame["ego2global_rotation"])


def _box_to_global(box: np.ndarray, frame: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    translation, yaw = _ego_pose(frame)
    xy = _rotation(yaw) @ np.asarray(box[:2], dtype=np.float64) + translation[:2]
    xyz = np.array([xy[0], xy[1], float(box[2]) + translation[2]], dtype=np.float64)
    heading = wrap_angle(float(box[6]) + yaw)
    return xyz, heading


def _global_box_to_local(
    global_xyz: np.ndarray,
    global_heading: float,
    frame: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    translation, yaw = _ego_pose(frame)
    local_xy = _rotation(-yaw) @ (np.asarray(global_xyz[:2]) - translation[:2])
    local_xyz = np.array(
        [local_xy[0], local_xy[1], float(global_xyz[2]) - translation[2]],
        dtype=np.float64,
    )
    return local_xyz, wrap_angle(global_heading - yaw)


def _velocity_to_global(velocity: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray:
    _, yaw = _ego_pose(frame)
    output = np.asarray(velocity, dtype=np.float64).copy()
    output[:2] = _rotation(yaw) @ output[:2]
    return output


def _global_velocity_to_local(velocity: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray:
    _, yaw = _ego_pose(frame)
    output = np.asarray(velocity, dtype=np.float64).copy()
    output[:2] = _rotation(-yaw) @ output[:2]
    return output


def _validate_and_collect_frame(frame: Any, frame_index: int) -> dict[str, Observation]:
    if not isinstance(frame, Mapping):
        raise TypeError(f"frame[{frame_index}] 不是 mapping")
    anns = frame.get("anns")
    if not isinstance(anns, Mapping):
        raise TypeError(f"frame[{frame_index}].anns 不是 mapping")

    lengths: dict[str, int] = {}
    for key in ANNOTATION_KEYS:
        value = anns.get(key)
        if value is None:
            raise KeyError(f"frame[{frame_index}].anns 缺少 {key}")
        lengths[key] = len(value)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"frame[{frame_index}] 注解长度不一致: {lengths}")

    boxes = np.asarray(anns["gt_boxes"])
    velocities = np.asarray(anns["gt_velocity_3d"])
    count = lengths["gt_boxes"]
    if boxes.ndim != 2 or boxes.shape[0] != count or boxes.shape[1] < 7:
        raise ValueError(f"frame[{frame_index}].gt_boxes shape 非法: {boxes.shape}")
    if velocities.ndim != 2 or velocities.shape[0] != count or velocities.shape[1] < 3:
        raise ValueError(f"frame[{frame_index}].gt_velocity_3d shape 非法: {velocities.shape}")

    output: dict[str, Observation] = {}
    for object_index in range(count):
        track_token = str(anns["track_tokens"][object_index])
        if not track_token:
            continue
        if track_token in output:
            raise ValueError(f"frame[{frame_index}] 存在重复 track_token={track_token}")
        output[track_token] = Observation(
            frame_index=frame_index,
            name=str(anns["gt_names"][object_index]),
            track_token=track_token,
            instance_token=str(anns["instance_tokens"][object_index]),
            box=np.asarray(boxes[object_index], dtype=np.float64).copy(),
            velocity=np.asarray(velocities[object_index], dtype=np.float64).copy(),
        )
    return output


def _find_empty_runs(counts: Sequence[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    frame_index = 0
    while frame_index < len(counts):
        if counts[frame_index] != 0:
            frame_index += 1
            continue
        start = frame_index
        while frame_index < len(counts) and counts[frame_index] == 0:
            frame_index += 1
        runs.append((start, frame_index - 1))
    return runs


def _interpolated_instance_token(frame: Mapping[str, Any], track_token: str) -> str:
    log_name = str(frame.get("log_name", "unknown_log"))
    timestamp = str(frame.get("timestamp", frame.get("token", "unknown_frame")))
    return f"{log_name}_{timestamp}_{track_token}"


def _interpolate_observation(
    previous: Observation,
    following: Observation,
    previous_frame: Mapping[str, Any],
    following_frame: Mapping[str, Any],
    target_frame: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    previous_time = int(previous_frame["timestamp"])
    following_time = int(following_frame["timestamp"])
    target_time = int(target_frame["timestamp"])
    if not previous_time < target_time < following_time:
        raise ValueError(
            f"timestamp 非严格递增: previous={previous_time}, target={target_time}, following={following_time}"
        )

    span_seconds = (following_time - previous_time) / 1_000_000.0
    alpha = (target_time - previous_time) / (following_time - previous_time)
    previous_xyz, previous_heading = _box_to_global(previous.box, previous_frame)
    following_xyz, following_heading = _box_to_global(following.box, following_frame)
    global_xyz = previous_xyz + alpha * (following_xyz - previous_xyz)
    heading_delta = wrap_angle(following_heading - previous_heading)
    global_heading = wrap_angle(previous_heading + alpha * heading_delta)

    box = previous.box + alpha * (following.box - previous.box)
    local_xyz, local_heading = _global_box_to_local(global_xyz, global_heading, target_frame)
    box[:3] = local_xyz
    box[6] = local_heading

    previous_velocity = _velocity_to_global(previous.velocity, previous_frame)
    following_velocity = _velocity_to_global(following.velocity, following_frame)
    global_velocity = previous_velocity + alpha * (following_velocity - previous_velocity)
    velocity = _global_velocity_to_local(global_velocity, target_frame)
    inferred_speed = float(np.linalg.norm(following_xyz[:2] - previous_xyz[:2]) / span_seconds)
    return box, velocity, float(alpha), float(span_seconds), inferred_speed


def build_repair_plan(
    frames: Sequence[Any],
    pkl_path: Path,
    max_gap: int,
    max_span_seconds: float,
    max_inferred_speed_mps: float,
) -> RepairPlan:
    """构造修复计划，不修改输入 frames。"""
    plan = RepairPlan(pkl_path=pkl_path)
    per_frame_tracks: list[dict[str, Observation]] = []
    counts: list[int] = []
    for frame_index, frame in enumerate(frames):
        tracks = _validate_and_collect_frame(frame, frame_index)
        per_frame_tracks.append(tracks)
        counts.append(len(frame["anns"]["gt_boxes"]))

    empty_runs = _find_empty_runs(counts)
    plan.empty_frames_before = sum(end - start + 1 for start, end in empty_runs)
    for start, end in empty_runs:
        gap_size = end - start + 1
        if start == 0 or end + 1 >= len(frames):
            plan.skipped_runs["boundary"] += 1
            continue
        if gap_size > max_gap:
            plan.skipped_runs["too_long"] += 1
            continue

        previous_index = start - 1
        following_index = end + 1
        previous_frame = frames[previous_index]
        following_frame = frames[following_index]
        previous_tracks = per_frame_tracks[previous_index]
        following_tracks = per_frame_tracks[following_index]
        common_tokens = sorted(set(previous_tracks) & set(following_tracks))
        if not common_tokens:
            plan.skipped_runs["no_common_tracks"] += 1
            continue
        plan.eligible_runs += 1

        for track_token in common_tokens:
            previous = previous_tracks[track_token]
            following = following_tracks[track_token]
            if previous.name != following.name:
                plan.skipped_tracks["class_mismatch"] += 1
                continue
            if previous.box.shape != following.box.shape or previous.velocity.shape != following.velocity.shape:
                plan.skipped_tracks["shape_mismatch"] += 1
                continue

            span_seconds = (
                int(following_frame["timestamp"]) - int(previous_frame["timestamp"])
            ) / 1_000_000.0
            if not 0.0 < span_seconds <= max_span_seconds:
                plan.skipped_tracks["time_span"] += 1
                continue

            pending: list[tuple[int, Addition, dict[str, Any]]] = []
            track_is_safe = True
            for missing_index in range(start, end + 1):
                target_frame = frames[missing_index]
                try:
                    box, velocity, alpha, actual_span, inferred_speed = _interpolate_observation(
                        previous,
                        following,
                        previous_frame,
                        following_frame,
                        target_frame,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    plan.skipped_tracks[f"interpolation_error:{type(exc).__name__}"] += 1
                    track_is_safe = False
                    break
                if not math.isfinite(inferred_speed) or inferred_speed > max_inferred_speed_mps:
                    plan.skipped_tracks["inferred_speed"] += 1
                    track_is_safe = False
                    break

                addition = Addition(
                    frame_index=missing_index,
                    name=previous.name,
                    track_token=track_token,
                    instance_token=_interpolated_instance_token(target_frame, track_token),
                    box=box,
                    velocity=velocity,
                )
                row = {
                    "pkl_path": str(pkl_path),
                    "log_name": target_frame.get("log_name", pkl_path.stem),
                    "frame_index": missing_index,
                    "frame_token": target_frame.get("token", ""),
                    "timestamp": target_frame.get("timestamp", ""),
                    "track_token": track_token,
                    "object_name": previous.name,
                    "previous_frame_index": previous_index,
                    "next_frame_index": following_index,
                    "gap_size": gap_size,
                    "alpha": f"{alpha:.8f}",
                    "endpoint_span_seconds": f"{actual_span:.6f}",
                    "inferred_speed_mps": f"{inferred_speed:.6f}",
                }
                pending.append((missing_index, addition, row))

            if track_is_safe:
                for missing_index, addition, row in pending:
                    plan.additions[missing_index].append(addition)
                    plan.report_rows.append(row)

    return plan


def apply_repair_plan(frames: list[Any], plan: RepairPlan) -> None:
    """将计划中的注解追加到 frames，保持原数组 dtype。"""
    for frame_index, additions in sorted(plan.additions.items()):
        frame = frames[frame_index]
        anns = frame["anns"]
        additions = sorted(additions, key=lambda item: item.track_token)
        boxes = np.stack([item.box for item in additions])
        velocities = np.stack([item.velocity for item in additions])
        original_boxes = np.asarray(anns["gt_boxes"])
        original_velocities = np.asarray(anns["gt_velocity_3d"])
        anns["gt_boxes"] = np.concatenate(
            [original_boxes, boxes.astype(original_boxes.dtype, copy=False)], axis=0
        )
        anns["gt_velocity_3d"] = np.concatenate(
            [original_velocities, velocities.astype(original_velocities.dtype, copy=False)], axis=0
        )
        anns["gt_names"].extend(item.name for item in additions)
        anns["instance_tokens"].extend(item.instance_token for item in additions)
        anns["track_tokens"].extend(item.track_token for item in additions)


def _write_pickle_atomically(path: Path, frames: list[Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as output_file:
            pickle.dump(frames, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def repair_pickle(
    path: Path,
    apply: bool,
    backup_suffix: str,
    max_gap: int,
    max_span_seconds: float,
    max_inferred_speed_mps: float,
) -> FileResult:
    """扫描或修复单个 pkl。"""
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as input_file:
            frames = pickle.load(input_file)
        if not isinstance(frames, list) or not frames:
            raise TypeError(f"pkl 顶层应为非空 list，实际为 {type(frames).__name__}")
        if not isinstance(frames[0], dict):
            raise TypeError("pkl 第一帧不是 dict")
        if frames[0].get(MARKER_KEY):
            return FileResult(path=path, skipped_marker=True)

        plan = build_repair_plan(
            frames=frames,
            pkl_path=path,
            max_gap=max_gap,
            max_span_seconds=max_span_seconds,
            max_inferred_speed_mps=max_inferred_speed_mps,
        )
        if apply and plan.added_objects:
            backup_path = path.with_name(path.name + backup_suffix)
            if backup_path.exists() or backup_path.is_symlink():
                raise FileExistsError(f"备份已存在，拒绝覆盖: {backup_path}")
            shutil.copy2(path, backup_path)
            apply_repair_plan(frames, plan)
            frames[0][MARKER_KEY] = {
                "version": 1,
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "max_gap": max_gap,
                "max_span_seconds": max_span_seconds,
                "max_inferred_speed_mps": max_inferred_speed_mps,
                "changed_frames": plan.changed_frames,
                "added_objects": plan.added_objects,
                "backup": str(backup_path),
            }
            _write_pickle_atomically(path, frames)
            return FileResult(path=path, plan=plan, applied=True)
        return FileResult(path=path, plan=plan)
    except Exception as exc:
        return FileResult(path=path, error=f"{type(exc).__name__}: {exc}")


def _discover_data_root(data_root: Path) -> list[Path]:
    """发现 canonical split 中的 pkl，避免重复扫描 trainval 和备份。"""
    logs_root = data_root / "dataset" / "navsim_logs"
    if not logs_root.is_dir():
        raise FileNotFoundError(f"数据集目录缺少 dataset/navsim_logs: {data_root}")

    paths: list[Path] = []
    for split in ("train", "val", "test"):
        split_dir = logs_root / split
        if split_dir.is_dir():
            paths.extend(sorted(split_dir.glob("*.pkl")))
    # 仅当 canonical split 全为空时回退 trainval，避免重复处理软链。
    if not paths:
        trainval_dir = logs_root / "trainval"
        if trainval_dir.is_dir():
            paths.extend(sorted(trainval_dir.glob("*.pkl")))
    return paths


def _resolve_paths(
    data_root: Path | None,
    pkl_paths: Sequence[Path],
    logs_dirs: Sequence[Path],
) -> list[Path]:
    if data_root is not None and (pkl_paths or logs_dirs):
        raise ValueError("数据集根目录不能与 --pkl/--logs-dir 同时使用")

    paths = _discover_data_root(data_root) if data_root is not None else list(pkl_paths)
    for logs_dir in logs_dirs:
        if not logs_dir.is_dir():
            raise FileNotFoundError(f"logs 目录不存在: {logs_dir}")
        paths.extend(sorted(logs_dir.glob("*.pkl")))
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.resolve()
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(path)
    return unique_paths


def _write_report(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        help="数据集根目录；自动扫描 dataset/navsim_logs/{train,val,test}",
    )
    parser.add_argument("--pkl", action="append", type=Path, default=[], help="待处理 pkl，可重复传入")
    parser.add_argument("--logs-dir", action="append", type=Path, default=[], help="批量处理目录下所有 pkl")
    parser.add_argument("--max-gap", type=int, default=4, help="允许插值的最大连续空帧数")
    parser.add_argument("--max-span-seconds", type=float, default=0.6, help="空洞两端最大时间跨度")
    parser.add_argument(
        "--max-inferred-speed-mps",
        type=float,
        default=80.0,
        help="端点位置推算速度上限，超过则跳过该 track",
    )
    parser.add_argument("--report", type=Path, default=None, help="CSV 报告；使用数据集根目录时可省略")
    parser.add_argument("--apply", action="store_true", help="实际写回；默认仅 dry-run")
    parser.add_argument(
        "--backup-suffix",
        default=".before_obj_interpolation",
        help="apply 前逐文件备份后缀；备份已存在时拒绝覆盖",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_gap < 1:
        print("ERROR: --max-gap 必须 >= 1", file=sys.stderr)
        return 2
    if args.max_span_seconds <= 0.0:
        print("ERROR: --max-span-seconds 必须 > 0", file=sys.stderr)
        return 2
    if args.max_inferred_speed_mps <= 0.0:
        print("ERROR: --max-inferred-speed-mps 必须 > 0", file=sys.stderr)
        return 2
    if not args.backup_suffix:
        print("ERROR: --backup-suffix 不允许为空", file=sys.stderr)
        return 2
    try:
        paths = _resolve_paths(args.data_root, args.pkl, args.logs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print("ERROR: 未发现 pkl；请提供数据集根目录、--pkl 或 --logs-dir", file=sys.stderr)
        return 2

    if args.report is None:
        if args.data_root is not None:
            report_name = (
                "mogo_obj_interpolation_applied.csv"
                if args.apply
                else "mogo_obj_interpolation_dry_run.csv"
            )
            args.report = args.data_root / "exp" / "obj_interpolation_reports" / report_name
        else:
            args.report = Path("mogo_obj_interpolation_report.csv")

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"mode={mode} files={len(paths)} max_gap={args.max_gap} "
        f"max_span={args.max_span_seconds:.3f}s max_speed={args.max_inferred_speed_mps:.1f}m/s"
    )
    results: list[FileResult] = []
    report_rows: list[dict[str, Any]] = []
    for path in paths:
        result = repair_pickle(
            path=path,
            apply=args.apply,
            backup_suffix=args.backup_suffix,
            max_gap=args.max_gap,
            max_span_seconds=args.max_span_seconds,
            max_inferred_speed_mps=args.max_inferred_speed_mps,
        )
        results.append(result)
        if result.error:
            print(f"[ERROR] {path}: {result.error}", file=sys.stderr)
            continue
        if result.skipped_marker:
            print(f"[SKIP] 已带修复标记: {path}")
            continue
        assert result.plan is not None
        report_rows.extend(result.plan.report_rows)
        action = "applied" if result.applied else "planned"
        print(
            f"[{action}] {path}: empty_frames={result.plan.empty_frames_before} "
            f"changed_frames={result.plan.changed_frames} added_objects={result.plan.added_objects} "
            f"skipped_runs={dict(result.plan.skipped_runs)} "
            f"skipped_tracks={dict(result.plan.skipped_tracks)}"
        )

    _write_report(args.report, report_rows)
    error_count = sum(result.error is not None for result in results)
    applied_count = sum(result.applied for result in results)
    changed_frames = sum(result.plan.changed_frames for result in results if result.plan)
    added_objects = sum(result.plan.added_objects for result in results if result.plan)
    print("\n===== 汇总 =====")
    print(f"处理文件: {len(results)} 失败: {error_count} 已写回: {applied_count}")
    print(f"计划修复帧: {changed_frames} 计划新增 obj: {added_objects}")
    print(f"报告: {args.report.resolve()}")
    if not args.apply:
        print("DRY-RUN：未修改任何 pkl；确认报告后加 --apply。")
    elif applied_count:
        print(
            "原始注解已变化：使用这些数据进行模型特征缓存或 PDM 评分前，"
            "请重建相关 cache。"
        )
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
