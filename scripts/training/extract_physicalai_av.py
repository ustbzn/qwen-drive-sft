#!/usr/bin/env python3
"""Extract the PhysicalAI modalities used by Qwen-Drive, excluding bad clips.

The output layout is ``camera/<camera>/<clip>.*`` and
``labels/egomotion/<clip>.egomotion.parquet``. No source file is changed.
Only clips listed in ``.physicalai_extract/valid_clips.jsonl`` are complete.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
import pyarrow.parquet as pq


SOURCE_DEFAULT = Path("/pcpt/pcpt/data/VLA/PhysicalAI-AV")
DEST_DEFAULT = Path("/pcpt/pcpt/data/VLA/PhysicalAI-AV_unzip")
CAMERAS = {
    "cam_f0": "camera_front_wide_120fov",
    "cam_l0": "camera_cross_left_120fov",
    "cam_r0": "camera_cross_right_120fov",
    "cam_b0": "camera_rear_tele_30fov",
    "cam_l1": "camera_rear_left_70fov",
    "cam_r1": "camera_rear_right_70fov",
}
DEFAULT_VIEWS = ("cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1", "cam_b0")
OFFSETS_US = np.asarray([-1_500_000, -1_000_000, -500_000, 0], dtype=np.int64)
EGO_COLUMNS = ("timestamp", "x", "y", "vx", "vy", "ax", "ay", "qx", "qy", "qz", "qw")
LOG = logging.getLogger("physicalai_extract")


def _signature(path: Path) -> list[int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return [stat.st_size, stat.st_mtime_ns]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def _atomic_text(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".tmp-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            for line in lines:
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def _validate_ego(path: Path, anchor_us: int) -> None:
    table = pq.read_table(path, columns=list(EGO_COLUMNS))
    data = {name: np.asarray(table[name].to_numpy()) for name in EGO_COLUMNS}
    stamps = data["timestamp"].astype(np.int64)
    if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError("egomotion timestamps are empty or not increasing")
    if stamps[0] > anchor_us - 1_500_000 or stamps[-1] < anchor_us + 5_000_000:
        raise ValueError("egomotion does not cover the training window")
    for name in EGO_COLUMNS[1:]:
        if not np.all(np.isfinite(data[name])):
            raise ValueError(f"egomotion {name} has non-finite values")
    quaternion = np.stack([data[name] for name in ("qx", "qy", "qz", "qw")], axis=1)
    if np.any(np.linalg.norm(quaternion, axis=1) < 1e-8):
        raise ValueError("egomotion has invalid quaternions")


def _frame_indices(path: Path, anchor_us: int) -> tuple[int, ...]:
    table = pq.read_table(path, columns=["timestamp", "frame_index"])
    stamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.int64)
    indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    if not len(stamps) or np.any(np.diff(stamps) <= 0):
        raise ValueError("camera timestamps are empty or not increasing")
    desired = anchor_us + OFFSETS_US
    positions = np.clip(np.searchsorted(stamps, desired), 0, len(stamps) - 1)
    previous = np.maximum(positions - 1, 0)
    positions = np.where(
        np.abs(stamps[previous] - desired) < np.abs(stamps[positions] - desired),
        previous, positions,
    )
    if np.any(np.abs(stamps[positions] - desired) > 50_000):
        raise ValueError("camera alignment exceeds 50 ms")
    selected = indices[positions]
    if np.any(selected < 0):
        raise ValueError("negative camera frame index")
    return tuple(int(index) for index in selected)


def _validate_video(path: Path, indices: tuple[int, ...]) -> None:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1, fault_tol=0)
    if max(indices) >= len(reader):
        raise ValueError(f"frame index {max(indices)} exceeds video length {len(reader)}")
    frames = reader.get_batch(list(indices)).asnumpy()
    if len(frames) != len(indices) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("video returned invalid RGB frames")
    del frames, reader


def _extract_member(archive: zipfile.ZipFile, member: str, path: Path) -> int:
    info = archive.getinfo(member)
    with archive.open(info) as source, path.open("wb") as dest:
        shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
    if path.stat().st_size != info.file_size:
        raise ValueError(f"size mismatch for {member}")
    # Reading to EOF makes zipfile check each member's CRC, including ZIP_STORED.
    return info.file_size


def _archive_specs(source: Path, chunk: int, cameras: tuple[str, ...]):
    label = source / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip"
    result = [("egomotion", label)]
    result.extend(
        (camera, source / "camera" / camera / f"{camera}.chunk_{chunk:04d}.zip")
        for camera in cameras
    )
    return result


def _output_paths(clip_id: str, cameras: tuple[str, ...]):
    yield Path("labels/egomotion") / f"{clip_id}.egomotion.parquet"
    for camera in cameras:
        yield Path("camera") / camera / f"{clip_id}.{camera}.timestamps.parquet"
        yield Path("camera") / camera / f"{clip_id}.{camera}.mp4"


def _process_chunk(chunk: int, clips: list[tuple[str, str]], args, cameras: tuple[str, ...], plan: dict):
    source, dest = args.source, args.dest
    state_path = dest / ".physicalai_extract" / "chunks" / f"chunk_{chunk:04d}.json"
    specs = _archive_specs(source, chunk, cameras)
    signatures = {name: _signature(path) for name, path in specs}
    expected = [clip_id for clip_id, _ in clips]
    if state_path.is_file():
        try:
            saved = json.loads(state_path.read_text())
        except (OSError, ValueError):
            saved = {}
        if (saved.get("plan") == plan and saved.get("signatures") == signatures
                and saved.get("expected") == expected
                and (not args.retry_failed or not saved.get("bad"))
                and all((dest / relative).is_file() and (dest / relative).stat().st_size == size
                        for relative, size in saved.get("outputs", {}).items())
                and "valid" in saved and "bad" in saved):
            for clip_id in saved["bad"]:
                for relative in _output_paths(clip_id, cameras):
                    (dest / relative).unlink(missing_ok=True)
            return saved, True

    valid, bad, outputs = [], {}, {}
    absent = [name for name, signature in signatures.items() if signature is None]
    if absent:
        bad = {clip_id: f"missing archive(s): {', '.join(absent)}" for clip_id, _ in clips}
    else:
        archives = {}
        try:
            for name, path in specs:
                archives[name] = zipfile.ZipFile(path)
            for clip_id, _ in clips:
                try:
                    with tempfile.TemporaryDirectory(prefix="clip-", dir=args.work_dir) as temp_name:
                        temp = Path(temp_name)
                        staged = []
                        ego_name = f"{clip_id}.egomotion.parquet"
                        ego_path = temp / ego_name
                        size = _extract_member(archives["egomotion"], ego_name, ego_path)
                        _validate_ego(ego_path, args.anchor_us)
                        staged.append((ego_path, Path("labels/egomotion") / ego_name, size))
                        for camera in cameras:
                            stamp_name = f"{clip_id}.{camera}.timestamps.parquet"
                            video_name = f"{clip_id}.{camera}.mp4"
                            stamp_path, video_path = temp / stamp_name, temp / video_name
                            stamp_size = _extract_member(archives[camera], stamp_name, stamp_path)
                            indices = _frame_indices(stamp_path, args.anchor_us)
                            video_size = _extract_member(archives[camera], video_name, video_path)
                            if args.video_check == "frames":
                                _validate_video(video_path, indices)
                            staged.extend((
                                (stamp_path, Path("camera") / camera / stamp_name, stamp_size),
                                (video_path, Path("camera") / camera / video_name, video_size),
                            ))
                        for staged_path, relative, file_size in staged:
                            target = dest / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(staged_path, target)
                            outputs[str(relative)] = file_size
                    valid.append(clip_id)
                except Exception as error:
                    bad[clip_id] = f"{type(error).__name__}: {error}"
        except Exception as error:
            # A broken central directory invalidates the entire affected chunk.
            for clip_id, _ in clips:
                if clip_id not in valid:
                    bad[clip_id] = f"archive error: {type(error).__name__}: {error}"
        finally:
            for archive in archives.values():
                archive.close()

    # A previous interrupted run may have published only part of a clip.
    for clip_id in bad:
        for relative in _output_paths(clip_id, cameras):
            (dest / relative).unlink(missing_ok=True)
            outputs.pop(str(relative), None)

    state = {
        "chunk": chunk, "expected": expected, "plan": plan, "signatures": signatures,
        "valid": valid, "bad": bad, "outputs": outputs,
    }
    _atomic_json(state_path, state)
    return state, False


def _load_clips(source: Path, splits: tuple[str, ...], cameras: tuple[str, ...]):
    index = pq.read_table(source / "clip_index.parquet", columns=["clip_id", "chunk", "split", "clip_is_valid"]).to_pydict()
    presence = pq.read_table(
        source / "metadata/feature_presence.parquet", columns=["clip_id", "egomotion", *cameras],
    ).to_pydict()
    available = {
        str(clip_id) for clip_id, *flags in zip(
            presence["clip_id"], presence["egomotion"], *(presence[camera] for camera in cameras)
        ) if all(flags)
    }
    by_chunk = defaultdict(list)
    excluded = 0
    for clip_id, chunk, split, is_valid in zip(
        index["clip_id"], index["chunk"], index["split"], index["clip_is_valid"]
    ):
        if split not in splits or not is_valid:
            continue
        if str(clip_id) not in available:
            excluded += 1
            continue
        if not re.fullmatch(r"[A-Za-z0-9-]+", str(clip_id)):
            raise ValueError(f"unsafe clip ID in metadata: {clip_id!r}")
        by_chunk[int(chunk)].append((str(clip_id), str(split)))
    return by_chunk, excluded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--dest", type=Path, default=DEST_DEFAULT)
    parser.add_argument("--views", nargs="+", choices=tuple(CAMERAS), default=DEFAULT_VIEWS)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val"))
    parser.add_argument("--workers", type=int, default=48, help="Concurrent ZIP chunk readers")
    parser.add_argument("--anchor-time-s", type=float, default=10.0)
    parser.add_argument("--video-check", choices=("frames", "metadata"), default="frames")
    parser.add_argument("--max-chunks", type=int, help="Process only the first N chunks for a smoke run")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.max_chunks is not None and args.max_chunks < 1):
        parser.error("--workers and --max-chunks must be positive")
    if not np.isfinite(args.anchor_time_s):
        parser.error("--anchor-time-s must be finite")
    args.source, args.dest = args.source.resolve(), args.dest.resolve()
    if args.source == args.dest or args.source in args.dest.parents or args.dest in args.source.parents:
        parser.error("source and destination must be separate directories")
    args.anchor_us = round(args.anchor_time_s * 1_000_000)
    cameras = tuple(dict.fromkeys(CAMERAS[view] for view in args.views))
    splits = tuple(dict.fromkeys(args.splits))
    plan = {"source": str(args.source), "cameras": list(cameras), "splits": list(splits), "anchor_us": args.anchor_us,
            "video_check": args.video_check,
            "metadata": [_signature(args.source / "clip_index.parquet"),
                         _signature(args.source / "metadata/feature_presence.parquet")]}
    if None in plan["metadata"]:
        parser.error("source clip_index.parquet or feature_presence.parquet is missing")
    by_chunk, excluded = _load_clips(args.source, splits, cameras)
    selected_chunks = sorted(by_chunk)
    if args.max_chunks:
        selected_chunks = selected_chunks[:args.max_chunks]
    args.dest.mkdir(parents=True, exist_ok=True)
    state_dir = args.dest / ".physicalai_extract"
    state_dir.mkdir(exist_ok=True)
    args.work_dir = state_dir / "work"
    args.work_dir.mkdir(exist_ok=True)
    for relative in ("clip_index.parquet", "metadata/feature_presence.parquet"):
        target = args.dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / relative, target)
    LOG.info("Selected %d chunks, %d clips; metadata excluded %d", len(selected_chunks),
             sum(len(by_chunk[chunk]) for chunk in selected_chunks), excluded)

    states = {}
    reused = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(_process_chunk, chunk, by_chunk[chunk], args, cameras, plan): chunk
                for chunk in selected_chunks}
        for number, future in enumerate(as_completed(jobs), 1):
            chunk = jobs[future]
            state, was_reused = future.result()
            states[chunk] = state
            reused += was_reused
            if number % 25 == 0 or number == len(jobs):
                LOG.info("Chunks %d/%d; reused %d; latest chunk %04d: valid %d, bad %d",
                         number, len(jobs), reused, chunk, len(state["valid"]), len(state["bad"]))

    split_by_clip = {clip_id: split for chunk in selected_chunks for clip_id, split in by_chunk[chunk]}
    valid_rows = ({"clip_id": clip_id, "chunk": chunk, "split": split_by_clip[clip_id]}
                  for chunk in selected_chunks for clip_id in states[chunk]["valid"])
    bad_rows = ({"clip_id": clip_id, "chunk": chunk, "split": split_by_clip[clip_id], "reason": reason}
                for chunk in selected_chunks for clip_id, reason in states[chunk]["bad"].items())
    _atomic_text(state_dir / "valid_clips.jsonl", (json.dumps(row) + "\n" for row in valid_rows))
    _atomic_text(state_dir / "bad_clips.jsonl", (json.dumps(row) + "\n" for row in bad_rows))
    summary = {"source": str(args.source), "destination": str(args.dest), "plan": plan,
               "selected_chunks": len(selected_chunks), "total_chunks": len(by_chunk),
               "metadata_excluded": excluded,
               "valid_clips": sum(len(state["valid"]) for state in states.values()),
               "bad_clips": sum(len(state["bad"]) for state in states.values()),
               "complete": len(selected_chunks) == len(by_chunk)}
    _atomic_json(state_dir / "summary.json", summary)
    LOG.info("Finished: %s", summary)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
