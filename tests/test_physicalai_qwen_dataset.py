"""Focused coverage for the lazy PhysicalAI-AV Qwen training adapter."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch

from navsim.planning.script.run_training_qwen_drive_il import (
    _build_physicalai_datasets,
    build_datasets,
    custom_collate_fn,
)
from navsim.agents.qwen_drive_planner.qwen_drive_agent import QwenDriveAgent
from navsim.agents.qwen_drive_planner.qwen_drive_config import QwenDrivePlannerConfig
from navsim.planning.training.physicalai_dataset import (
    MixedSourceDataset,
    PHYSICALAI_CAMERA_BY_VIEW,
    PhysicalAIDataset,
    parse_mixed_ratio,
)


VIEWS = ("cam_f0", "cam_l1", "cam_r1")
CAMERAS = tuple(PHYSICALAI_CAMERA_BY_VIEW[view] for view in VIEWS)
DEFAULT_PUBLIC_VIEWS = ("cam_f0", "cam_l0", "cam_r0")
FIXTURE_CAMERAS = tuple(dict.fromkeys(
    PHYSICALAI_CAMERA_BY_VIEW[view] for view in (*VIEWS, *DEFAULT_PUBLIC_VIEWS)
))


def _parquet_bytes(columns):
    stream = BytesIO()
    pq.write_table(pa.table(columns), stream)
    return stream.getvalue()


def _write_zip(path: Path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _make_physicalai_fixture(root: Path):
    clips = [
        {"clip_id": f"clip-{split}", "chunk": 0, "split": split, "clip_is_valid": True}
        for split in ("train", "val", "test")
    ]
    # Missing requested camera features must be filtered before shard reads.
    clips.append({"clip_id": "clip-missing-camera", "chunk": 1, "split": "train", "clip_is_valid": True})
    # Feature metadata can exist while a referenced shard is absent on disk.
    clips.append({"clip_id": "clip-missing-shard", "chunk": 1, "split": "train", "clip_is_valid": True})
    pq.write_table(pa.Table.from_pylist(clips), root / "clip_index.parquet")

    presence_rows = []
    for clip in clips:
        row = {
            "clip_id": clip["clip_id"],
            "egomotion": True,
            CAMERAS[0]: clip["clip_id"] != "clip-missing-camera",
        }
        row.update({camera: True for camera in FIXTURE_CAMERAS[1:]})
        presence_rows.append(row)
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(presence_rows), metadata / "feature_presence.parquet")

    timestamps = np.arange(-2_000_000, 18_000_001, 100_000, dtype=np.int64)
    ego_columns = {
        "timestamp": timestamps,
        "qx": np.zeros(len(timestamps)), "qy": np.zeros(len(timestamps)),
        "qz": np.zeros(len(timestamps)), "qw": np.ones(len(timestamps)),
        "x": timestamps / 1_000_000.0, "y": np.zeros(len(timestamps)),
        "vx": np.ones(len(timestamps)), "vy": np.zeros(len(timestamps)),
        "ax": np.zeros(len(timestamps)), "ay": np.zeros(len(timestamps)),
    }
    ego_payload = _parquet_bytes(ego_columns)
    _write_zip(root / "labels/egomotion/egomotion.chunk_0000.zip", {
        f"{clip['clip_id']}.egomotion.parquet": ego_payload for clip in clips[:3]
    })

    camera_stamps = np.arange(-2_000_000, 18_000_001, 100_000, dtype=np.int64)
    for camera in FIXTURE_CAMERAS:
        members = {}
        for clip in clips[:3]:
            clip_id = clip["clip_id"]
            members[f"{clip_id}.{camera}.timestamps.parquet"] = _parquet_bytes({
                "timestamp": camera_stamps,
                "frame_index": np.arange(len(camera_stamps), dtype=np.int64),
            })
            members[f"{clip_id}.{camera}.mp4"] = b"synthetic-video-payload"
        _write_zip(root / "camera" / camera / f"{camera}.chunk_0000.zip", members)


def test_physicalai_splits_window_camera_mapping_and_lazy_collate(tmp_path, monkeypatch):
    root = tmp_path / "PhysicalAI-AV"
    root.mkdir()
    _make_physicalai_fixture(root)
    dataset = PhysicalAIDataset(root, "train", VIEWS, anchor_time_s=10.0, shard_check_workers=4)

    assert len(dataset) == 1
    assert dataset.excluded_missing_shards == 1
    sample = dataset[0]
    features, targets, token = sample
    assert token == "clip-train"
    assert features["history"].shape == (16, 3)
    assert features["history_velocity"].shape == (16, 2)
    assert features["nav_command"].item() == -1
    assert features["navigation_instruction"] is None
    assert targets["trajectory"].shape == (50, 3)
    torch.testing.assert_close(targets["trajectory"][:, 0], torch.arange(1, 51).float() * 0.1)
    torch.testing.assert_close(targets["trajectory"][:, 1:], torch.zeros(50, 2))
    assert len(features["view_frame_paths"]) == len(VIEWS)
    assert features["view_list"] == VIEWS
    assert all(len(view) == 4 and all(callable(frame) for frame in view)
               for view in features["view_frame_paths"])

    class FakeReader:
        def __init__(self, path, **kwargs):
            assert path.startswith("/proc/self/fd/")
            with open(path, "rb") as stream:
                assert stream.read() == b"synthetic-video-payload"
            assert "width" not in kwargs and "height" not in kwargs

        def __len__(self):
            return 300

        def get_batch(self, indices):
            self.indices = indices
            return SimpleNamespace(asnumpy=lambda: np.full((4, 8, 8, 3), 23, dtype=np.uint8))

    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(VideoReader=FakeReader, cpu=lambda _: None))
    restored_features = pickle.loads(pickle.dumps(features))
    image = restored_features["view_frame_paths"][0][0]()
    assert isinstance(image, Image.Image) and image.getpixel((0, 0)) == (23, 23, 23)

    collated_features, collated_targets, tokens = custom_collate_fn([sample, sample])
    assert collated_features["history"].shape == (2, 16, 3)
    assert collated_targets["trajectory"].shape == (2, 50, 3)
    assert tokens == ("clip-train", "clip-train")
    assert collated_features["view_lists"] == [VIEWS, VIEWS]
    assert dataset.close() is None

    assert len(PhysicalAIDataset(root, "val", VIEWS, anchor_time_s=10.0)) == 1
    assert len(PhysicalAIDataset(root, "test", VIEWS, anchor_time_s=10.0)) == 1


def test_physicalai_default_views_are_independent_of_mogo_views(tmp_path):
    root = tmp_path / "PhysicalAI-AV"
    root.mkdir()
    _make_physicalai_fixture(root)
    mogo_agent = QwenDriveAgent(QwenDrivePlannerConfig(view_list=["cam_f0", "cam_l1", "cam_r1"]))
    train, val = _build_physicalai_datasets({"physicalai_root": str(root)}, mogo_agent)

    assert train.view_list == val.view_list == ("cam_f0", "cam_l0", "cam_r0")
    features, _, _ = train[0]
    assert features["view_list"] == train.view_list
    assert features["view_frame_paths"][1][0].group.member.endswith("camera_cross_left_120fov.mp4")
    assert features["view_frame_paths"][2][0].group.member.endswith("camera_cross_right_120fov.mp4")
    assert mogo_agent.config.view_list == ["cam_f0", "cam_l1", "cam_r1"]


class _TaggedDataset(torch.utils.data.Dataset):
    def __init__(self, tag, size):
        self.tag = tag
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.tag, index


def test_mixed_ratio_repeats_only_to_meet_requested_training_ratio():
    mixed = MixedSourceDataset(_TaggedDataset("mogo", 5), _TaggedDataset("public", 2), (1, 1))
    assert len(mixed) == 10
    assert [mixed[index][0] for index in range(len(mixed))].count("mogo") == 5
    assert [mixed[index][0] for index in range(len(mixed))].count("public") == 5
    assert parse_mixed_ratio("proportional") is None
    assert parse_mixed_ratio("2:1") == (2, 1)


def test_mixed_mode_applies_ratio_to_train_only(monkeypatch):
    import navsim.planning.script.run_training_qwen_drive_il as training

    mogo_train, mogo_val = _TaggedDataset("mogo-train", 5), _TaggedDataset("mogo-val", 2)
    public_train, public_val = _TaggedDataset("public-train", 2), _TaggedDataset("public-val", 3)
    monkeypatch.setattr(training, "_build_mogo_datasets", lambda *args, **kwargs: (mogo_train, mogo_val))
    monkeypatch.setattr(training, "_build_physicalai_datasets", lambda *args, **kwargs: (public_train, public_val))

    train, val = build_datasets({
        "dataset_mode": "mixed", "physicalai_mixture_ratio": "1:1",
    }, agent=object())
    assert isinstance(train, MixedSourceDataset)
    assert len(train) == 10
    assert len(val) == 5
    assert [val[index][0] for index in range(len(val))].count("mogo-val") == 2
    assert [val[index][0] for index in range(len(val))].count("public-val") == 3


def test_mix_mode_defaults_to_original_sample_proportions(monkeypatch):
    import navsim.planning.script.run_training_qwen_drive_il as training

    mogo_train, mogo_val = _TaggedDataset("mogo-train", 5), _TaggedDataset("mogo-val", 2)
    public_train, public_val = _TaggedDataset("public-train", 2), _TaggedDataset("public-val", 3)
    monkeypatch.setattr(training, "_build_mogo_datasets", lambda *args, **kwargs: (mogo_train, mogo_val))
    monkeypatch.setattr(training, "_build_physicalai_datasets", lambda *args, **kwargs: (public_train, public_val))

    train, val = build_datasets({"dataset_mode": "mix"}, agent=object())
    assert isinstance(train, torch.utils.data.ConcatDataset)
    assert len(train) == 7
    assert [train[index][0] for index in range(len(train))].count("mogo-train") == 5
    assert [train[index][0] for index in range(len(train))].count("public-train") == 2
    assert isinstance(val, torch.utils.data.ConcatDataset)
    assert len(val) == 5
