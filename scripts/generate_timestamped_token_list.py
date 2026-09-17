#!/usr/bin/env python3
"""Generate an inference-ready token/timestamp list from a MOGO scene PKL.

MOGO logs are dense 10Hz frame lists, while the ``mogo_10hz`` scene filter
samples a 0.5s model cadence.  The default options mirror that filter exactly:
4 history frames, 10 future frames, a raw-frame stride of 5, and a window
interval of 5.  Consequently every emitted token can be consumed directly by
the existing evaluation pipeline.

Example:

    python scripts/generate_timestamped_token_list.py \
      /pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/badcase_TSP/dataset/navsim_logs/trainval/JLBJA20073D_badcase_TSP__20260813154927.pkl \
      --output navsim/local_config/data_selection/tsp_case_timestamped.txt
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import pickle
import sys
from typing import Any


# Allow running this file directly from a checkout without installing navsim.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from navsim.common.token_timestamp import TokenTimestamp, write_token_timestamp_list


DEFAULT_INPUT_PKL = Path(
    "/pcpt/pcpt/data/VLA/MOGO/recogdrive_dataset/badcase_TSP/dataset/"
    "navsim_logs/trainval/JLBJA20073D_badcase_TSP__20260813154927.pkl"
)
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "navsim" / "local_config" / "data_selection"


def _load_frames(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"PKL file does not exist: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty list of frame dictionaries: {path}")

    frames: list[Mapping[str, Any]] = []
    for index, frame in enumerate(payload):
        if not isinstance(frame, Mapping):
            raise ValueError(f"Frame {index} is not a mapping: {type(frame).__name__}")
        missing = {"token", "timestamp"} - set(frame)
        if missing:
            raise ValueError(f"Frame {index} is missing required fields: {sorted(missing)}")
        token = str(frame["token"]).strip()
        if not token:
            raise ValueError(f"Frame {index} contains an empty token")
        try:
            timestamp_us = int(frame["timestamp"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Frame {index} has an invalid timestamp: {frame['timestamp']!r}") from exc
        if timestamp_us < 0:
            raise ValueError(f"Frame {index} has a negative timestamp: {timestamp_us}")
        frames.append(frame)

    tokens = [str(frame["token"]).strip() for frame in frames]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"PKL contains duplicate frame tokens: {path}")
    return frames


def _select_anchor_records(
    frames: list[Mapping[str, Any]],
    *,
    num_history_frames: int,
    num_future_frames: int,
    frame_stride: int,
    frame_interval: int,
    require_route: bool,
) -> list[TokenTimestamp]:
    """Mirror ``navsim.common.dataloader.filter_scenes`` for one PKL."""

    if num_history_frames < 1:
        raise ValueError("--num-history-frames must be at least 1")
    if num_future_frames < 0:
        raise ValueError("--num-future-frames must be non-negative")
    if frame_stride < 1 or frame_interval < 1:
        raise ValueError("--frame-stride and --frame-interval must be at least 1")

    num_frames = num_history_frames + num_future_frames
    num_raw_frames = (num_frames - 1) * frame_stride + 1
    records: list[TokenTimestamp] = []
    for start in range(0, len(frames), frame_interval):
        sampled = frames[start : start + num_raw_frames : frame_stride]
        if len(sampled) < num_frames:
            continue

        anchor = sampled[num_history_frames - 1]
        if require_route and not anchor.get("roadblock_ids"):
            continue
        records.append(
            TokenTimestamp(
                token=str(anchor["token"]).strip(),
                timestamp_us=int(anchor["timestamp"]),
            )
        )

    return records


def _select_raw_records(frames: list[Mapping[str, Any]]) -> list[TokenTimestamp]:
    return [
        TokenTimestamp(token=str(frame["token"]).strip(), timestamp_us=int(frame["timestamp"]))
        for frame in frames
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 MOGO PKL 生成可用于推理的 token<TAB>timestamp_us 列表"
    )
    parser.add_argument(
        "input_pkl",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_PKL,
        help=f"输入 PKL；默认：{DEFAULT_INPUT_PKL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 TXT；默认写入 data_selection/<PKL stem>_timestamped.txt",
    )
    parser.add_argument(
        "--raw-frames",
        action="store_true",
        help="输出 PKL 中的每一帧，而不是按 mogo_10hz 规则输出 anchor token",
    )
    parser.add_argument("--num-history-frames", type=int, default=4)
    parser.add_argument("--num-future-frames", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-interval", type=int, default=5)
    parser.add_argument(
        "--allow-no-route",
        action="store_true",
        help="允许 anchor 帧缺少 roadblock_ids（默认与 SceneFilter.has_route=true 一致）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有输出文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = _load_frames(args.input_pkl)
    if args.raw_frames:
        records = _select_raw_records(frames)
    else:
        records = _select_anchor_records(
            frames,
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_stride=args.frame_stride,
            frame_interval=args.frame_interval,
            require_route=not args.allow_no_route,
        )

    output = args.output
    if output is None:
        output = DEFAULT_OUTPUT_DIR / f"{args.input_pkl.stem}_timestamped.txt"
    write_token_timestamp_list(records, output, overwrite=args.force)

    print(f"读取 {len(frames)} 帧: {args.input_pkl}")
    mode = "raw frames" if args.raw_frames else "mogo_10hz anchor frames"
    print(f"按 {mode} 写入 {len(records)} 条: {output}")
    if records:
        print(f"首条: {records[0].token}\t{records[0].timestamp_us}")
        print(f"末条: {records[-1].token}\t{records[-1].timestamp_us}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
