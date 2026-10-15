"""Lazy Qwen-Drive training adapter for NVIDIA PhysicalAI-AV clips."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import json
import os
import tempfile
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from navsim.common.dataclasses import EgoStatus
from navsim.agents.qwen_drive_planner.qwen_drive_features import (
    build_qwen_drive_ego_features,
)


# Match NVIDIA camera names to the semantic view slots consumed by the Qwen processor.
PHYSICALAI_CAMERA_BY_VIEW = {
    "cam_f0": "camera_front_wide_120fov",
    "cam_l0": "camera_cross_left_120fov",
    "cam_r0": "camera_cross_right_120fov",
    "cam_l1": "camera_rear_left_70fov",
    "cam_r1": "camera_rear_right_70fov",
    "cam_b0": "camera_rear_tele_30fov",
}

_NUM_HISTORY = 16
_NUM_FUTURE = 50
_SAMPLE_INTERVAL_US = 100_000
_CAMERA_OFFSETS_US = np.asarray([-1_500_000, -1_000_000, -500_000, 0], dtype=np.int64)
_MAX_OPEN_ARCHIVES = 8
_VIDEO_READ_CHUNK = 8 * 1024 * 1024
logger = logging.getLogger(__name__)


def _relative_poses(poses: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Transform [x, y, yaw] poses from clip coordinates into the ego frame."""
    delta = poses[:, :2] - origin[:2]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    relative = np.empty_like(poses, dtype=np.float32)
    relative[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    relative[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    relative[:, 2] = (poses[:, 2] - origin[2] + np.pi) % (2.0 * np.pi) - np.pi
    return relative


def _write_member_to_fd(archive: zipfile.ZipFile, member: str, fd: int) -> None:
    with archive.open(member, "r") as source:
        while True:
            chunk = source.read(_VIDEO_READ_CHUNK)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(fd, view)
                view = view[written:]


@dataclass
class _VideoFrameGroup:
    """Decode four requested frames once, when the Qwen processor asks for them."""

    archive_path: str
    member: str
    frame_indices: Tuple[int, ...]
    file_path: Optional[str] = None
    _decoded: Optional[np.ndarray] = None

    def frames(self) -> np.ndarray:
        if self._decoded is not None:
            return self._decoded

        try:
            from decord import VideoReader, cpu
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PhysicalAI-AV video loading requires decord; install the repository requirements"
            ) from error

        fd: Optional[int] = None
        temporary_path: Optional[str] = None
        reader = None
        try:
            video_path = self.file_path
            if video_path is None:
                with zipfile.ZipFile(self.archive_path, "r") as archive:
                    if hasattr(os, "memfd_create"):
                        fd = os.memfd_create("physicalai-video", 0)
                        _write_member_to_fd(archive, self.member, fd)
                        os.lseek(fd, 0, os.SEEK_SET)
                        video_path = f"/proc/self/fd/{fd}"
                    else:  # pragma: no cover - Linux training hosts use memfd
                        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                        temporary_path = handle.name
                        with handle:
                            _write_member_to_fd(archive, self.member, handle.fileno())
                        video_path = temporary_path

            reader = VideoReader(
                video_path,
                ctx=cpu(0),
                num_threads=1,
                fault_tol=0,
            )
            if max(self.frame_indices) >= len(reader):
                raise ValueError(
                    f"Camera frame index {max(self.frame_indices)} exceeds video length {len(reader)}: "
                    f"{self.member}"
                )
            self._decoded = reader.get_batch(list(self.frame_indices)).asnumpy()
            return self._decoded
        except Exception as error:
            if isinstance(error, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(
                f"Unable to decode {self.file_path or self.member} from {self.archive_path}: {error}"
            ) from error
        finally:
            reader = None
            if fd is not None:
                os.close(fd)
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def release(self) -> None:
        self._decoded = None


@dataclass(frozen=True)
class _VideoFrame:
    """Pickle-friendly callable consumed by the existing CameraFrame loader."""

    group: _VideoFrameGroup
    index: int

    def __call__(self) -> Image.Image:
        frame = Image.fromarray(self.group.frames()[self.index].copy()).convert("RGB")
        if self.index == len(self.group.frame_indices) - 1:
            self.group.release()
        return frame


class PhysicalAIDataset(torch.utils.data.Dataset):
    """One 10 Hz Qwen planning sample per public dataset clip.

    ZIP mode reads members on demand. Files mode uses the completed extraction
    manifest to exclude damaged clips and decodes MP4 files directly.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        view_list: Sequence[str],
        anchor_time_s: float = 10.0,
        max_samples: Optional[int] = None,
        shard_check_workers: int = 48,
        storage: str = "auto",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported PhysicalAI split {split!r}; expected train, val, or test")
        if not np.isfinite(float(anchor_time_s)):
            raise ValueError("physicalai_anchor_time_s must be finite")
        self.anchor_time_us = int(round(float(anchor_time_s) * 1_000_000))
        self.view_list = tuple(str(view) for view in view_list)
        if not self.view_list:
            raise ValueError("PhysicalAI training requires at least one configured camera view")
        unknown_views = [view for view in self.view_list if view not in PHYSICALAI_CAMERA_BY_VIEW]
        if unknown_views:
            raise ValueError(
                "No PhysicalAI camera mapping for configured Qwen views "
                f"{unknown_views}; supported views are {sorted(PHYSICALAI_CAMERA_BY_VIEW)}"
            )
        if max_samples is not None and int(max_samples) < 1:
            raise ValueError("physicalai_max_samples_per_split must be positive or null")
        if int(shard_check_workers) < 1:
            raise ValueError("physicalai_scan_workers must be positive")
        self.shard_check_workers = int(shard_check_workers)
        if storage not in ("auto", "zip", "files"):
            raise ValueError(f"physicalai_storage must be auto, zip, or files; got {storage!r}")
        self.storage = ("files" if (self.root / ".physicalai_extract/summary.json").is_file()
                        else "zip") if storage == "auto" else storage

        for relative in (
            "clip_index.parquet",
            "metadata/feature_presence.parquet",
            "labels/egomotion",
            "camera",
        ):
            if not (self.root / relative).exists():
                raise FileNotFoundError(f"PhysicalAI dataset path does not exist: {self.root / relative}")

        if self.storage == "files":
            self.samples = self._load_extracted_samples(max_samples)
            self.excluded_missing_features = 0
            self.excluded_missing_shards = 0
            self._zip_cache = OrderedDict()
            return

        clip_table = pq.read_table(
            self.root / "clip_index.parquet",
            columns=["clip_is_valid", "chunk", "split", "clip_id"],
        ).to_pydict()
        camera_columns = [PHYSICALAI_CAMERA_BY_VIEW[view] for view in self.view_list]
        presence_columns = list(dict.fromkeys(["egomotion", *camera_columns, "clip_id"]))
        presence_table = pq.read_table(
            self.root / "metadata/feature_presence.parquet", columns=presence_columns,
        ).to_pydict()
        presence = {
            str(clip_id): row_index
            for row_index, clip_id in enumerate(presence_table["clip_id"])
        }

        self.samples: List[Tuple[str, int]] = []
        missing_metadata = 0
        missing_features = 0
        for clip_id, chunk, clip_split, is_valid in zip(
            clip_table["clip_id"], clip_table["chunk"], clip_table["split"], clip_table["clip_is_valid"],
        ):
            if str(clip_split) != self.split or not bool(is_valid):
                continue
            row_index = presence.get(str(clip_id))
            if row_index is None:
                missing_metadata += 1
                continue
            if not bool(presence_table["egomotion"][row_index]) or any(
                not bool(presence_table[camera][row_index]) for camera in camera_columns
            ):
                missing_features += 1
                continue
            self.samples.append((str(clip_id), int(chunk)))

        if missing_metadata:
            raise ValueError(f"{missing_metadata} {self.split} clips are missing feature-presence metadata")
        if not self.samples:
            raise ValueError(
                f"No usable PhysicalAI clips in split={self.split!r}; "
                f"excluded for missing requested modalities={missing_features}"
            )
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]

        self.excluded_missing_features = missing_features
        self.excluded_missing_shards = 0
        self._zip_cache: OrderedDict[str, zipfile.ZipFile] = OrderedDict()
        self._validate_shards()

    def _load_extracted_samples(self, max_samples: Optional[int]) -> List[Tuple[str, int]]:
        state_dir = self.root / ".physicalai_extract"
        try:
            with (state_dir / "summary.json").open() as stream:
                summary = json.load(stream)
        except (OSError, ValueError) as error:
            raise ValueError(f"PhysicalAI extracted data requires a readable summary.json in {state_dir}") from error
        plan = summary.get("plan", {})
        requested_cameras = {PHYSICALAI_CAMERA_BY_VIEW[view] for view in self.view_list}
        if summary.get("destination") != str(self.root):
            raise ValueError(f"PhysicalAI extraction summary belongs to another directory: {state_dir}")
        if not summary.get("complete") or summary.get("selected_chunks") != summary.get("total_chunks"):
            raise ValueError(f"PhysicalAI extraction is incomplete: {state_dir / 'summary.json'}")
        if (not requested_cameras.issubset(set(plan.get("cameras", [])))
                or self.split not in plan.get("splits", [])
                or self.anchor_time_us != plan.get("anchor_us")
                or plan.get("video_check") != "frames"):
            raise ValueError(
                "PhysicalAI extraction does not cover the requested split, views, anchor, "
                "and decoded video frames"
            )
        if len(plan.get("metadata", [])) != 2:
            raise ValueError(f"PhysicalAI extraction summary has no metadata signatures: {state_dir}")
        for relative, expected in zip(
            ("clip_index.parquet", "metadata/feature_presence.parquet"),
            plan.get("metadata", []),
        ):
            path = self.root / relative
            if expected != [path.stat().st_size, path.stat().st_mtime_ns]:
                raise ValueError(f"PhysicalAI extraction metadata changed since validation: {path}")
        samples = []
        seen = set()
        manifest = state_dir / "valid_clips.jsonl"
        try:
            with manifest.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    clip_id, split, chunk = row["clip_id"], row["split"], row["chunk"]
                    if clip_id in seen:
                        raise ValueError(f"Duplicate PhysicalAI clip {clip_id} in {manifest}")
                    seen.add(clip_id)
                    if split == self.split:
                        samples.append((str(clip_id), int(chunk)))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid PhysicalAI extraction manifest {manifest}: {error}") from error
        if len(seen) != summary.get("valid_clips"):
            raise ValueError(f"PhysicalAI extraction manifest count differs from summary: {manifest}")
        if max_samples is not None:
            samples = samples[:int(max_samples)]
        if not samples:
            raise ValueError(f"No validated PhysicalAI clips in split={self.split!r}: {manifest}")
        logger.info("PhysicalAI split=%s using %d validated extracted clips from %s",
                    self.split, len(samples), manifest)
        return samples

    def _archive_path(self, kind: str, name: str, chunk: int) -> Path:
        if kind == "camera":
            return self.root / "camera" / name / f"{name}.chunk_{chunk:04d}.zip"
        return self.root / "labels" / name / f"{name}.chunk_{chunk:04d}.zip"

    def _validate_shards(self) -> None:
        chunks = sorted({chunk for _, chunk in self.samples})
        archive_specs = [("labels", "egomotion")]
        archive_specs.extend(("camera", PHYSICALAI_CAMERA_BY_VIEW[view]) for view in self.view_list)
        chunk_paths = {
            chunk: [self._archive_path(kind, name, chunk) for kind, name in archive_specs]
            for chunk in chunks
        }
        paths = [path for chunk in chunks for path in chunk_paths[chunk]]
        with ThreadPoolExecutor(max_workers=self.shard_check_workers) as executor:
            existence = dict(zip(paths, executor.map(Path.is_file, paths)))
        missing_by_chunk = {
            chunk: [str(path) for path in paths if not existence[path]]
            for chunk, paths in chunk_paths.items()
            if any(not existence[path] for path in paths)
        }
        if missing_by_chunk:
            previous_count = len(self.samples)
            missing_chunks = set(missing_by_chunk)
            self.samples = [(clip_id, chunk) for clip_id, chunk in self.samples if chunk not in missing_chunks]
            self.excluded_missing_shards = previous_count - len(self.samples)
            missing_paths = [path for shard_paths in missing_by_chunk.values() for path in shard_paths]
            logger.warning(
                "PhysicalAI split=%s excluded %d clip(s) across %d chunk(s) because %d required ZIP shard(s) are missing; first missing paths: %s",
                self.split,
                self.excluded_missing_shards,
                len(missing_by_chunk),
                len(missing_paths),
                missing_paths[:10],
            )
        if not self.samples:
            raise ValueError(
                f"No usable PhysicalAI clips in split={self.split!r} after checking required ZIP shards"
            )

    def _zip(self, path: Path) -> zipfile.ZipFile:
        key = str(path)
        archive = self._zip_cache.pop(key, None)
        if archive is None:
            archive = zipfile.ZipFile(path, "r")
        self._zip_cache[key] = archive
        while len(self._zip_cache) > _MAX_OPEN_ARCHIVES:
            _, old = self._zip_cache.popitem(last=False)
            old.close()
        return archive

    def _read_member(self, path: Path, member: str) -> bytes:
        try:
            return self._zip(path).read(member)
        except KeyError as error:
            raise FileNotFoundError(f"Missing {member} in PhysicalAI shard {path}") from error

    @staticmethod
    def _parquet_member(data: bytes) -> Dict[str, List[Any]]:
        try:
            return pq.read_table(pa.BufferReader(data)).to_pydict()
        except Exception as error:
            raise ValueError(f"Unable to parse a PhysicalAI Parquet member: {error}") from error

    def _egomotion(self, clip_id: str, chunk: int) -> Dict[str, List[Any]]:
        member = f"{clip_id}.egomotion.parquet"
        if self.storage == "files":
            return pq.read_table(self.root / "labels" / "egomotion" / member).to_pydict()
        archive_path = self._archive_path("labels", "egomotion", chunk)
        return self._parquet_member(self._read_member(archive_path, member))

    def _camera_frames(
        self, clip_id: str, chunk: int, camera_name: str,
    ) -> Tuple[_VideoFrame, ...]:
        timestamp_member = f"{clip_id}.{camera_name}.timestamps.parquet"
        if self.storage == "files":
            timestamps = pq.read_table(self.root / "camera" / camera_name / timestamp_member).to_pydict()
        else:
            archive_path = self._archive_path("camera", camera_name, chunk)
            timestamps = self._parquet_member(self._read_member(archive_path, timestamp_member))
        stamp_values = np.asarray(timestamps["timestamp"], dtype=np.int64)
        frame_values = np.asarray(timestamps["frame_index"], dtype=np.int64)
        if not len(stamp_values) or np.any(np.diff(stamp_values) <= 0):
            raise ValueError(f"Invalid or empty camera timestamps for clip {clip_id}, camera {camera_name}")
        desired = self.anchor_time_us + _CAMERA_OFFSETS_US
        positions = np.searchsorted(stamp_values, desired)
        positions = np.clip(positions, 0, len(stamp_values) - 1)
        previous = np.maximum(positions - 1, 0)
        use_previous = np.abs(stamp_values[previous] - desired) < np.abs(stamp_values[positions] - desired)
        positions = np.where(use_previous, previous, positions)
        errors_s = np.abs(stamp_values[positions] - desired) / 1_000_000.0
        if np.any(errors_s > 0.05):
            raise ValueError(
                f"Camera frame alignment exceeds 50 ms for clip {clip_id}, {camera_name}: "
                f"{errors_s.tolist()}"
            )

        video_member = f"{clip_id}.{camera_name}.mp4"
        video_path = self.root / "camera" / camera_name / video_member
        group = _VideoFrameGroup(
            archive_path=str(video_path if self.storage == "files" else archive_path),
            member=video_member,
            frame_indices=tuple(int(index) for index in frame_values[positions]),
            file_path=str(video_path) if self.storage == "files" else None,
        )
        return tuple(_VideoFrame(group, index) for index in range(len(positions)))

    def _sample_ego(self, data: Dict[str, List[Any]]) -> Tuple[List[EgoStatus], np.ndarray]:
        timestamps = np.asarray(data["timestamp"], dtype=np.int64)
        if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("PhysicalAI online egomotion timestamps must be strictly increasing")

        relative_offsets = np.arange(-(_NUM_HISTORY - 1), _NUM_FUTURE + 1, dtype=np.int64)
        grid = self.anchor_time_us + relative_offsets * _SAMPLE_INTERVAL_US
        if grid[0] < timestamps[0] or grid[-1] > timestamps[-1]:
            raise ValueError(
                "Clip does not cover the required 16 history + 50 future points at "
                f"anchor={self.anchor_time_us / 1e6:.3f}s; data spans "
                f"[{timestamps[0] / 1e6:.3f}, {timestamps[-1] / 1e6:.3f}]s"
            )

        source_time = timestamps.astype(np.float64)

        def interpolate(name: str) -> np.ndarray:
            values = np.asarray(data[name], dtype=np.float64)
            result = np.interp(grid.astype(np.float64), source_time, values)
            if not np.all(np.isfinite(result)):
                raise ValueError(f"Non-finite egomotion field {name}")
            return result

        x, y = interpolate("x"), interpolate("y")
        vx, vy = interpolate("vx"), interpolate("vy")
        ax, ay = interpolate("ax"), interpolate("ay")
        quaternion = np.stack(
            [np.asarray(data[name], dtype=np.float64) for name in ("qx", "qy", "qz", "qw")],
            axis=1,
        )
        quaternion_norm = np.linalg.norm(quaternion, axis=1)
        if not np.all(np.isfinite(quaternion_norm)) or np.any(quaternion_norm < 1e-8):
            raise ValueError("PhysicalAI egomotion contains invalid orientation quaternions")
        quaternion /= quaternion_norm[:, None]
        qx, qy, qz, qw = quaternion.T
        source_yaw = np.unwrap(
            np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        )
        yaw = np.interp(grid.astype(np.float64), source_time, source_yaw)
        poses = np.stack([x, y, yaw], axis=1).astype(np.float32)

        # NVIDIA online egomotion velocities and accelerations use the clip frame.
        # Qwen/MOGO features consume vectors in each instantaneous ego frame.
        c, s = np.cos(yaw), np.sin(yaw)
        body_velocity = np.stack([c * vx + s * vy, -s * vx + c * vy], axis=1)
        body_acceleration = np.stack([c * ax + s * ay, -s * ax + c * ay], axis=1)
        if not np.all(np.isfinite(body_velocity)) or not np.all(np.isfinite(body_acceleration)):
            raise ValueError("Non-finite transformed PhysicalAI ego dynamics")

        history_poses = _relative_poses(poses[:_NUM_HISTORY], poses[_NUM_HISTORY - 1])
        history_statuses = [
            EgoStatus(
                ego_pose=history_poses[index],
                ego_velocity=body_velocity[index].astype(np.float32),
                ego_acceleration=body_acceleration[index].astype(np.float32),
                driving_command=np.zeros(4, dtype=np.int64),
            )
            for index in range(_NUM_HISTORY)
        ]
        future_poses = _relative_poses(poses[_NUM_HISTORY:], poses[_NUM_HISTORY - 1])
        return history_statuses, future_poses

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        clip_id, chunk = self.samples[index]
        statuses, trajectory = self._sample_ego(self._egomotion(clip_id, chunk))
        features = build_qwen_drive_ego_features(
            type("AgentInputView", (), {"ego_statuses": statuses})(), _NUM_HISTORY,
        )
        view_frame_paths = []
        for view in self.view_list:
            camera_name = PHYSICALAI_CAMERA_BY_VIEW[view]
            view_frame_paths.append(list(self._camera_frames(clip_id, chunk, camera_name)))
        features["view_frame_paths"] = view_frame_paths
        features["view_list"] = self.view_list

        return features, {"trajectory": torch.from_numpy(trajectory)}, clip_id

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        for archive in self._zip_cache.values():
            archive.close()
        self._zip_cache.clear()

    def __del__(self) -> None:
        # DataLoader worker teardown should release ZIP file descriptors promptly.
        try:
            self.close()
        except Exception:
            pass


class MixedSourceDataset(torch.utils.data.Dataset):
    """Repeat the smaller source to achieve a requested integer sample ratio."""

    def __init__(self, mogo_dataset, physicalai_dataset, ratio: Tuple[int, int]) -> None:
        self.mogo_dataset = mogo_dataset
        self.physicalai_dataset = physicalai_dataset
        self.mogo_weight, self.physicalai_weight = ratio
        if self.mogo_weight < 1 or self.physicalai_weight < 1:
            raise ValueError("Mixed dataset ratio weights must both be positive integers")
        if not len(mogo_dataset) or not len(physicalai_dataset):
            raise ValueError("Mixed training requires nonempty MOGO and PhysicalAI datasets")
        repeat_units = max(
            ceil(len(mogo_dataset) / self.mogo_weight),
            ceil(len(physicalai_dataset) / self.physicalai_weight),
        )
        self._cycle_size = self.mogo_weight + self.physicalai_weight
        self._length = repeat_units * self._cycle_size

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        cycle_offset = index % self._cycle_size
        cycle_number = index // self._cycle_size
        if cycle_offset < self.mogo_weight:
            sample_index = (cycle_number * self.mogo_weight + cycle_offset) % len(self.mogo_dataset)
            return self.mogo_dataset[sample_index]
        public_offset = cycle_offset - self.mogo_weight
        sample_index = (cycle_number * self.physicalai_weight + public_offset) % len(self.physicalai_dataset)
        return self.physicalai_dataset[sample_index]


def parse_mixed_ratio(value: str) -> Optional[Tuple[int, int]]:
    """Return ``None`` for length-proportional sampling or parse ``mogo:public``."""
    value = str(value).strip().lower()
    if value in ("proportional", "sample", "samples"):
        return None
    try:
        mogo, physicalai = (int(piece) for piece in value.split(":"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "physicalai_mixture_ratio must be 'proportional' or two positive integers such as '1:1'"
        ) from error
    if mogo < 1 or physicalai < 1:
        raise ValueError("physicalai_mixture_ratio weights must be positive integers")
    return mogo, physicalai
