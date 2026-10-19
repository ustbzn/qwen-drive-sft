"""The image cache must preserve selected frames and the processor's RGB resize."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import pytest
import torch

from navsim.agents.qwen_drive_planner.qwen_drive_agent import QwenDriveAgent
from navsim.agents.qwen_drive_planner.qwen_drive_config import QwenDrivePlannerConfig
from navsim.agents.qwen_drive_planner.modeling.scene import CameraFrame, QwenDriveProcessor
from navsim.planning.training.physicalai_dataset import PhysicalAIDataset, _CachedFrame, _VideoFrame
from scripts.training.cache_physicalai_images import DEFAULT_VIEWS, main


CAMERA = "camera_front_wide_120fov"
LEFT_CAMERA = "camera_cross_left_120fov"


def test_default_cache_views_cover_all_physicalai_cameras():
    assert DEFAULT_VIEWS == ("cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1", "cam_b0")


def _extracted_fixture(root: Path) -> None:
    clips = ("clip-a", "clip-b")
    root.mkdir()
    source = root.parent / "archives"
    for camera in (CAMERA, LEFT_CAMERA):
        archive = source / "camera" / camera / f"{camera}.chunk_0000.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"archive")
    pq.write_table(pa.table({
        "clip_id": clips, "chunk": [0, 0], "split": ["train", "train"],
        "clip_is_valid": [True, True],
    }), root / "clip_index.parquet")
    (root / "metadata").mkdir()
    pq.write_table(pa.table({
        "clip_id": clips, "egomotion": [True, True], CAMERA: [True, True],
        LEFT_CAMERA: [True, True],
    }), root / "metadata/feature_presence.parquet")
    (root / "labels/egomotion").mkdir(parents=True)
    stamps = np.arange(0, 16_000_001, 100_000, dtype=np.int64)
    for clip in clips:
        ego = {"timestamp": stamps, "x": stamps / 1_000_000.0}
        for name in ("y", "vx", "vy", "ax", "ay", "qx", "qy", "qz"):
            ego[name] = np.zeros(len(stamps))
        ego["qw"] = np.ones(len(stamps))
        pq.write_table(pa.table(ego), root / "labels/egomotion" / f"{clip}.egomotion.parquet")
        for camera in (CAMERA, LEFT_CAMERA):
            camera_dir = root / "camera" / camera
            camera_dir.mkdir(parents=True, exist_ok=True)
            prefix = camera_dir / f"{clip}.{camera}"
            pq.write_table(pa.table({
                "timestamp": stamps, "frame_index": np.arange(len(stamps), dtype=np.int64),
            }), Path(f"{prefix}.timestamps.parquet"))
            Path(f"{prefix}.mp4").write_bytes(b"video")
    state = root / ".physicalai_extract"
    state.mkdir()
    (state / "valid_clips.jsonl").write_text("".join(
        json.dumps({"clip_id": clip, "chunk": 0, "split": "train"}) + "\n"
        for clip in clips
    ))
    metadata_paths = [root / "clip_index.parquet", root / "metadata/feature_presence.parquet"]
    plan = {
        "source": str(source),
        "cameras": [CAMERA, LEFT_CAMERA], "splits": ["train"], "anchor_us": 10_000_000,
        "video_check": "frames",
        "metadata": [[path.stat().st_size, path.stat().st_mtime_ns] for path in metadata_paths],
    }
    (state / "summary.json").write_text(json.dumps({
        "destination": str(root), "complete": True, "selected_chunks": 1,
        "total_chunks": 1, "valid_clips": len(clips), "plan": plan,
    }))


def test_cache_preserves_frame_selection_pixels_and_resume(tmp_path, monkeypatch):
    root, dest = tmp_path / "extracted", tmp_path / "cache"
    _extracted_fixture(root)
    decoded_paths = []

    def make_frames(indices):
        assert indices == [85, 90, 95, 100]
        frames = np.empty((4, 480, 640, 3), dtype=np.uint8)
        horizontal = np.arange(640, dtype=np.uint8)[None, :]
        for position, index in enumerate(indices):
            frames[position, :, :, 0] = horizontal
            frames[position, :, :, 1] = index
            frames[position, :, :, 2] = position * 40
        return frames

    class FakeReader:
        def __init__(self, path, **kwargs):
            decoded_paths.append(path)
            self.path = path
            assert kwargs["num_threads"] == 1

        def __len__(self):
            return 161

        def get_batch(self, indices):
            frames = make_frames(indices)
            payload = Path(self.path).read_bytes()
            if payload != b"video":
                frames[:, :, :, 0] = 17 if payload == b"other-content" else 19
            return SimpleNamespace(asnumpy=lambda: frames)

    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(VideoReader=FakeReader, cpu=lambda _: None))
    args = ["--root", str(root), "--dest", str(dest), "--views", "cam_f0",
            "--splits", "train", "--clips-per-shard", "1", "--workers", "2"]
    assert main(args) == 0
    assert len(decoded_paths) == 2
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["complete"] and manifest["clips"] == 2
    assert len(manifest["shards"]) == 2

    index = json.loads((dest / manifest["shards"][0]["index"]).read_text())
    entries = index["records"][0]["frames"]["cam_f0"]
    assert [entry["frame_index"] for entry in entries] == [85, 90, 95, 100]
    assert entries[0]["width"] < entries[3]["width"]

    dataset = PhysicalAIDataset(root, "train", ("cam_l0", "cam_f0"),
                                storage="files", image_cache_dir=dest)
    selected, _, _ = dataset[0]
    assert isinstance(selected["view_frame_paths"][0][0], _VideoFrame)
    assert isinstance(selected["view_frame_paths"][1][0], _CachedFrame)
    cached_frames = pickle.loads(pickle.dumps(selected["view_frame_paths"][1]))
    assert cached_frames[0]().mode == "RGB"
    agent = QwenDriveAgent(QwenDrivePlannerConfig(view_list=["cam_f0"]))
    scene = agent._build_scene(
        [cached_frames], selected["history"], selected["history_velocity"],
        selected["history_acceleration"], selected["ego_status"],
        selected["nav_command"].item(), view_list=["cam_f0"],
    )
    assert scene.frames_in_order()[0].pre_resized == cached_frames[0].pre_resized

    processor = object.__new__(QwenDriveProcessor)
    processor.patch_size = 16
    processor.merge_size = 2
    processor.temporal_patch_size = 2
    processor.factor = 32
    processor.min_pixels = 4 * 32**2
    processor.grid_pixel_limit = 12800 * 32**2
    processor.config = SimpleNamespace(history_image_pixels=174080, current_image_pixels=921600)
    for position, entry in enumerate(entries):
        original = make_frames([85, 90, 95, 100])[position]
        budget = 921600 if position == 3 else 174080
        expected, expected_grid = processor._patchify(CameraFrame(Image.fromarray(original)), budget)
        actual, actual_grid = processor._patchify(scene.frames_in_order()[position], budget)
        assert actual_grid == expected_grid
        assert torch.equal(actual, expected)
    processor.config.current_image_pixels += 1
    with pytest.raises(ValueError, match="do not match model settings"):
        processor._patchify(scene.frames_in_order()[0], 174080)

    assert main(args) == 0
    assert len(decoded_paths) == 2
    video = root / "camera" / CAMERA / f"clip-a.{CAMERA}.mp4"
    video.write_bytes(b"changed-video")
    assert isinstance(dataset[0][0]["view_frame_paths"][1][0], _VideoFrame)
    assert main(args) == 0
    assert len(decoded_paths) == 3

    first_index = dest / manifest["shards"][0]["index"]
    hidden_index = first_index.with_suffix(".hidden")
    first_index.rename(hidden_index)
    try:
        incomplete = PhysicalAIDataset(root, "train", ("cam_f0",),
                                       storage="files", image_cache_dir=dest)
        assert isinstance(incomplete[1][0]["view_frame_paths"][0][0], _CachedFrame)
        assert isinstance(incomplete[0][0]["view_frame_paths"][0][0], _VideoFrame)
    finally:
        hidden_index.rename(first_index)
    assert isinstance(incomplete[0][0]["view_frame_paths"][0][0], _CachedFrame)

    summary_path = root / ".physicalai_extract/summary.json"
    summary = json.loads(summary_path.read_text())
    summary["plan"]["cameras"].extend([
        "camera_cross_right_120fov", "camera_rear_left_70fov",
        "camera_rear_right_70fov", "camera_rear_tele_30fov",
    ])
    summary_path.write_text(json.dumps(summary))
    old_stat = video.stat()
    os.utime(video, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000))
    camera_dir = root / "camera" / CAMERA
    timestamp_path = camera_dir / f"clip-a.{CAMERA}.timestamps.parquet"
    archive = root.parent / "archives" / "camera" / CAMERA / f"{CAMERA}.chunk_0000.zip"
    state = {
        "plan": summary["plan"], "valid": ["clip-a", "clip-b"],
        "signatures": {CAMERA: [archive.stat().st_size, archive.stat().st_mtime_ns]},
        "outputs": {
            str(path.relative_to(root)): path.stat().st_size for path in (timestamp_path, video)
        },
    }
    chunk_dir = root / ".physicalai_extract/chunks"
    chunk_dir.mkdir()
    (chunk_dir / "chunk_0000.json").write_text(json.dumps(state))
    expanded = PhysicalAIDataset(root, "train", ("cam_f0",),
                                 storage="files", image_cache_dir=dest)
    before_verify = len(decoded_paths)
    assert isinstance(expanded[0][0]["view_frame_paths"][0][0], _CachedFrame)
    assert len(decoded_paths) == before_verify
    assert isinstance(expanded[0][0]["view_frame_paths"][0][0], _CachedFrame)
    assert len(decoded_paths) == before_verify

    video.write_bytes(b"other-content")  # Same length, different decoded pixels.
    os.utime(video, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert isinstance(expanded[0][0]["view_frame_paths"][0][0], _VideoFrame)
    assert len(decoded_paths) == before_verify + 1
