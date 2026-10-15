"""Extraction tests use tiny ZIP_STORED clips, including one with a bad CRC."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.training.extract_physicalai_av import main


def _parquet_bytes(data):
    stream = BytesIO()
    pq.write_table(pa.table(data), stream)
    return stream.getvalue()


def _zip(path: Path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_extract_filters_bad_crc_and_resumes(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    camera = "camera_front_wide_120fov"
    clips = ("clip-good", "clip-bad")
    pq.write_table(pa.table({
        "clip_id": clips, "chunk": [0, 0], "split": ["train", "train"],
        "clip_is_valid": [True, True],
    }), source / "clip_index.parquet")
    (source / "metadata").mkdir()
    pq.write_table(pa.table({
        "clip_id": clips, "egomotion": [True, True], camera: [True, True],
    }), source / "metadata/feature_presence.parquet")
    stamps = np.arange(0, 16_000_001, 100_000, dtype=np.int64)
    ego = {"timestamp": stamps}
    for column in ("x", "y", "vx", "vy", "ax", "ay", "qx", "qy", "qz"):
        ego[column] = np.zeros(len(stamps))
    ego["qw"] = np.ones(len(stamps))
    ego_payload = _parquet_bytes(ego)
    _zip(source / "labels/egomotion/egomotion.chunk_0000.zip",
         {f"{clip}.egomotion.parquet": ego_payload for clip in clips})
    camera_archive = source / "camera" / camera / f"{camera}.chunk_0000.zip"
    members = {}
    for clip in clips:
        members[f"{clip}.{camera}.timestamps.parquet"] = _parquet_bytes({
            "timestamp": stamps, "frame_index": np.arange(len(stamps)),
        })
        members[f"{clip}.{camera}.mp4"] = b"synthetic-video"
    _zip(camera_archive, members)
    # Flip one stored member byte while keeping the central-directory CRC unchanged.
    archive_bytes = camera_archive.read_bytes()
    marker = b"synthetic-video"
    second = archive_bytes.find(marker, archive_bytes.find(marker) + len(marker))
    assert second >= 0
    corrupted = bytearray(archive_bytes)
    corrupted[second] ^= 1
    camera_archive.write_bytes(corrupted)

    argv = ["--source", str(source), "--dest", str(dest), "--views", "cam_f0",
            "--workers", "2", "--video-check", "metadata"]
    stale = dest / "camera" / camera / f"clip-bad.{camera}.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale-partial-output")
    assert main(argv) == 0
    state = dest / ".physicalai_extract"
    valid = [json.loads(line) for line in (state / "valid_clips.jsonl").read_text().splitlines()]
    bad = [json.loads(line) for line in (state / "bad_clips.jsonl").read_text().splitlines()]
    assert [row["clip_id"] for row in valid] == ["clip-good"]
    assert [row["clip_id"] for row in bad] == ["clip-bad"]
    assert "BadZipFile" in bad[0]["reason"]
    assert (dest / "camera" / camera / f"clip-good.{camera}.mp4").is_file()
    assert not stale.exists()

    checkpoint = state / "chunks/chunk_0000.json"
    before = checkpoint.stat().st_mtime_ns
    assert main(argv) == 0
    assert checkpoint.stat().st_mtime_ns == before

    _zip(camera_archive, members)
    assert main(argv + ["--retry-failed"]) == 0
    assert json.loads((state / "summary.json").read_text())["valid_clips"] == 2
