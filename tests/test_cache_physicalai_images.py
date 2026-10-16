"""The image cache must preserve selected frames and the processor's RGB resize."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch

from navsim.agents.qwen_drive_planner.modeling.scene import CameraFrame, QwenDriveProcessor
from scripts.training.cache_physicalai_images import DEFAULT_VIEWS, main


CAMERA = "camera_front_wide_120fov"


def test_default_cache_views_cover_all_physicalai_cameras():
    assert DEFAULT_VIEWS == ("cam_f0", "cam_l0", "cam_r0", "cam_l1", "cam_r1", "cam_b0")


def _extracted_fixture(root: Path) -> None:
    clips = ("clip-a", "clip-b")
    root.mkdir()
    pq.write_table(pa.table({
        "clip_id": clips, "chunk": [0, 0], "split": ["train", "train"],
        "clip_is_valid": [True, True],
    }), root / "clip_index.parquet")
    (root / "metadata").mkdir()
    pq.write_table(pa.table({
        "clip_id": clips, "egomotion": [True, True], CAMERA: [True, True],
    }), root / "metadata/feature_presence.parquet")
    (root / "labels/egomotion").mkdir(parents=True)
    camera_dir = root / "camera" / CAMERA
    camera_dir.mkdir(parents=True)
    stamps = np.arange(0, 16_000_001, 100_000, dtype=np.int64)
    for clip in clips:
        prefix = camera_dir / f"{clip}.{CAMERA}"
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
        "cameras": [CAMERA], "splits": ["train"], "anchor_us": 10_000_000,
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
            assert kwargs["num_threads"] == 1

        def __len__(self):
            return 161

        def get_batch(self, indices):
            return SimpleNamespace(asnumpy=lambda: make_frames(indices))

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

    processor = object.__new__(QwenDriveProcessor)
    processor.patch_size = 16
    processor.merge_size = 2
    processor.temporal_patch_size = 2
    processor.factor = 32
    processor.min_pixels = 4 * 32**2
    processor.grid_pixel_limit = 12800 * 32**2
    tar_path = dest / manifest["shards"][0]["tar"]
    for position, entry in enumerate(entries):
        with tar_path.open("rb") as stream:
            stream.seek(entry["offset"])
            cached = Image.open(BytesIO(stream.read(entry["bytes"]))).convert("RGB")
        original = make_frames([85, 90, 95, 100])[position]
        budget = 921600 if position == 3 else 174080
        expected, expected_grid = processor._patchify(CameraFrame(Image.fromarray(original)), budget)
        actual, actual_grid = processor._patchify(CameraFrame(cached), budget)
        assert actual_grid == expected_grid
        assert torch.equal(actual, expected)

    assert main(args) == 0
    assert len(decoded_paths) == 2
    video = root / "camera" / CAMERA / f"clip-a.{CAMERA}.mp4"
    video.write_bytes(b"changed-video")
    assert main(args) == 0
    assert len(decoded_paths) == 3
