#!/usr/bin/env python3
"""列出 Lightning 实验中各 checkpoint 的验证损失及 RL epoch reward。

示例：
  python scripts/list_checkpoint_metrics.py \
    all_exp/training_recogdrive_ego_SDE_obj_hdmap_novlm_IL/2026.07.15.03.36.07

输入可以是实验目录、lightning_logs/version_*、checkpoints 目录、event 文件，
或其中任意一个 checkpoint 文件。脚本始终列出同一 version 下的全部 checkpoint。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


EVENT_PATTERN = "events.out.tfevents.*"
DEFAULT_VAL_TAG = "val/loss_epoch"
DEFAULT_REWARD_TAG = "train/reward_epoch"


class MetricsError(RuntimeError):
    """可直接展示给命令行用户的错误。"""


@dataclass(frozen=True)
class ScalarPoint:
    epoch: Optional[int]
    step: int
    value: float


@dataclass(frozen=True)
class CheckpointMetric:
    path: Path
    epoch: Optional[int]
    checkpoint_step: Optional[int]
    metric_step: Optional[int]
    val_loss: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据模型实验路径列出 checkpoint 的 val/loss，并为 RL 模型列出每个 epoch 的 reward。"
    )
    parser.add_argument("model_path", type=Path, help="实验目录、version 目录、checkpoint 或 event 路径")
    parser.add_argument(
        "--sort",
        choices=("loss", "epoch"),
        default="loss",
        help="checkpoint 排序方式（默认按 val/loss 从小到大）",
    )
    parser.add_argument("--val-tag", default=DEFAULT_VAL_TAG, help="验证损失的 TensorBoard tag")
    parser.add_argument("--reward-tag", default=DEFAULT_REWARD_TAG, help="RL reward 的 TensorBoard tag")
    return parser.parse_args()


def find_version_dirs(model_path: Path) -> List[Path]:
    path = model_path.expanduser()
    if not path.exists():
        raise MetricsError(f"路径不存在: {path}")

    anchor = path.parent if path.is_file() else path
    for candidate in (anchor, *anchor.parents):
        if candidate.name.startswith("version_") and any(candidate.glob(EVENT_PATTERN)):
            return [candidate.resolve()]

    event_parents = {event.parent.resolve() for event in anchor.rglob(EVENT_PATTERN)}
    if not event_parents:
        raise MetricsError(f"未在路径下找到 TensorBoard event: {path}")
    return sorted(event_parents, key=str)


def find_checkpoints(version_dir: Path) -> List[Path]:
    checkpoint_dir = version_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise MetricsError(f"未找到 checkpoint 目录: {checkpoint_dir}")

    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.name)
    if not checkpoints:
        raise MetricsError(f"checkpoint 目录为空: {checkpoint_dir}")
    return checkpoints


def parse_checkpoint_name(path: Path) -> Tuple[Optional[int], Optional[int]]:
    epoch_match = re.search(r"epoch[=_](\d+)", path.stem)
    step_match = re.search(r"step[=_](\d+)", path.stem)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    step = int(step_match.group(1)) if step_match else None
    return epoch, step


def load_event_accumulator(version_dir: Path) -> Any:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise MetricsError("缺少 tensorboard，请先安装 requirements.txt 中的依赖") from exc

    accumulator = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def read_scalar_points(accumulator: Any, tag: str) -> List[ScalarPoint]:
    available_tags = set(accumulator.Tags().get("scalars", []))
    if tag not in available_tags:
        return []

    epoch_by_step: Dict[int, int] = {}
    if "epoch" in available_tags:
        for event in accumulator.Scalars("epoch"):
            epoch_by_step[event.step] = int(round(event.value))

    points_by_key: Dict[Tuple[Optional[int], int], ScalarPoint] = {}
    for event in accumulator.Scalars(tag):
        point = ScalarPoint(
            epoch=epoch_by_step.get(event.step),
            step=event.step,
            value=float(event.value),
        )
        points_by_key[(point.epoch, point.step)] = point
    return sorted(points_by_key.values(), key=lambda point: point.step)


def match_checkpoint(checkpoint: Path, val_points: Sequence[ScalarPoint]) -> CheckpointMetric:
    epoch, checkpoint_step = parse_checkpoint_name(checkpoint)
    point_by_epoch = {point.epoch: point for point in val_points if point.epoch is not None}
    matched = point_by_epoch.get(epoch) if epoch is not None else None

    if matched is None and checkpoint_step is not None:
        point_by_step = {point.step: point for point in val_points}
        matched = point_by_step.get(checkpoint_step) or point_by_step.get(checkpoint_step - 1)

    return CheckpointMetric(
        path=checkpoint,
        epoch=epoch,
        checkpoint_step=checkpoint_step,
        metric_step=matched.step if matched else None,
        val_loss=matched.value if matched else None,
    )


def sort_checkpoint_metrics(
    metrics: Sequence[CheckpointMetric], sort_by: str
) -> List[CheckpointMetric]:
    if sort_by == "epoch":
        return sorted(
            metrics,
            key=lambda item: (
                item.epoch is None,
                item.epoch if item.epoch is not None else math.inf,
                item.path.name,
            ),
        )
    return sorted(
        metrics,
        key=lambda item: (
            item.val_loss is None or not math.isfinite(item.val_loss),
            item.val_loss if item.val_loss is not None else math.inf,
            item.epoch if item.epoch is not None else math.inf,
        ),
    )


def format_optional_int(value: Optional[int]) -> str:
    return str(value) if value is not None else "N/A"


def format_optional_float(value: Optional[float]) -> str:
    return f"{value:.6f}" if value is not None else "N/A"


def print_checkpoint_table(
    metrics: Sequence[CheckpointMetric], val_tag: str, sort_by: str
) -> None:
    sort_label = val_tag if sort_by == "loss" else "epoch"
    print(f"\nCheckpoints（按 {sort_label} 排序）")
    print(f"{'rank':>4}  {'epoch':>5}  {'ckpt_step':>9}  {'event_step':>10}  {val_tag:>16}  checkpoint")
    print(f"{'-' * 4}  {'-' * 5}  {'-' * 9}  {'-' * 10}  {'-' * 16}  {'-' * 10}")
    for rank, metric in enumerate(metrics, start=1):
        print(
            f"{rank:>4}  "
            f"{format_optional_int(metric.epoch):>5}  "
            f"{format_optional_int(metric.checkpoint_step):>9}  "
            f"{format_optional_int(metric.metric_step):>10}  "
            f"{format_optional_float(metric.val_loss):>16}  "
            f"{metric.path.name}"
        )

    selectable = [
        metric
        for metric in metrics
        if metric.val_loss is not None and math.isfinite(metric.val_loss)
    ]
    if selectable:
        best = min(selectable, key=lambda metric: metric.val_loss)  # type: ignore[arg-type]
        print(f"\n推荐 checkpoint（最低 {val_tag}）: {best.path}")
    else:
        print(f"\n警告: 没有 checkpoint 能与 {val_tag} 对齐。", file=sys.stderr)


def print_reward_table(points: Sequence[ScalarPoint], reward_tag: str) -> None:
    print(f"\nRL rewards（{reward_tag}）")
    print(f"{'epoch':>5}  {'event_step':>10}  {'reward':>12}")
    print(f"{'-' * 5}  {'-' * 10}  {'-' * 12}")
    for point in sorted(points, key=lambda item: (item.epoch is None, item.epoch, item.step)):
        print(
            f"{format_optional_int(point.epoch):>5}  "
            f"{point.step:>10}  "
            f"{point.value:>12.6f}"
        )


def inspect_version(version_dir: Path, args: argparse.Namespace) -> None:
    accumulator = load_event_accumulator(version_dir)
    available_tags = set(accumulator.Tags().get("scalars", []))
    if args.val_tag not in available_tags:
        available = ", ".join(sorted(available_tags)) or "无"
        raise MetricsError(
            f"{version_dir} 中没有 {args.val_tag!r}；可用 scalar tags: {available}"
        )

    checkpoints = find_checkpoints(version_dir)
    val_points = read_scalar_points(accumulator, args.val_tag)
    metrics = [match_checkpoint(checkpoint, val_points) for checkpoint in checkpoints]
    sorted_metrics = sort_checkpoint_metrics(metrics, args.sort)

    print(f"\n日志目录: {version_dir}")
    print(f"checkpoint 数量: {len(checkpoints)}，验证指标记录数: {len(val_points)}")
    print_checkpoint_table(sorted_metrics, args.val_tag, args.sort)

    reward_points = read_scalar_points(accumulator, args.reward_tag)
    if reward_points:
        print_reward_table(reward_points, args.reward_tag)
    elif re.search(r"(?:^|[/_-])rl(?:$|[/_-])", str(version_dir), flags=re.IGNORECASE):
        print(
            f"\n警告: 路径疑似 RL 实验，但 event 中没有 {args.reward_tag!r}。",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    try:
        version_dirs = find_version_dirs(args.model_path)
        print(f"模型路径: {args.model_path.expanduser().resolve()}")
        for version_dir in version_dirs:
            inspect_version(version_dir, args)
    except MetricsError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
