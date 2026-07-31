import argparse
import pickle
from pathlib import Path

from navsim.common.dataclasses import SceneFilter
from scripts.cache_dataset.check_recogdrive_cache_products import (
    BASE_REQUIRED_FILES,
    OBJECT_FILE_BY_MODE,
    DatasetSpec,
    build_expected_index,
    inspect_dataset,
)


def _args(**overrides):
    values = {
        "hidden_state_file": "qwen3_feature.gz",
        "object_feature": "auto",
        "required_file": [],
        "fail_on_extra": False,
        "check_nonempty": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_raw_log(data_root: Path, frame_count: int = 5) -> None:
    log_path = data_root / "dataset" / "navsim_logs" / "trainval" / "log-a.pkl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        {
            "token": f"token-{index}",
            "log_name": "log-a",
            "roadblock_ids": ["route-a"],
        }
        for index in range(frame_count)
    ]
    with log_path.open("wb") as file_handle:
        pickle.dump(frames, file_handle)


def _scene_filter() -> SceneFilter:
    return SceneFilter(
        num_history_frames=1,
        num_future_frames=1,
        frame_stride=1,
        frame_interval=1,
        has_route=True,
    )


def _write_product(token_path: Path, file_name: str) -> None:
    token_path.mkdir(parents=True, exist_ok=True)
    (token_path / file_name).write_bytes(b"cache")


def _write_complete_expected_products(cache_root: Path, tokens) -> None:
    required = [*BASE_REQUIRED_FILES, OBJECT_FILE_BY_MODE["temporal"]]
    for token in tokens:
        token_path = cache_root / "log-a" / token
        for file_name in required:
            _write_product(token_path, file_name)
        _write_product(token_path, "hdmap_vector_feature.gz")


def test_checker_separates_complete_expected_tokens_from_stale_extra(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    _write_raw_log(data_root)
    expected_tokens = [f"token-{index}" for index in range(4)]
    _write_complete_expected_products(cache_root, expected_tokens)

    stale_path = cache_root / "log-a" / "token-4"
    for file_name in (
        "qwen3_feature.gz",
        "recogdrive_ego_feature_v1.gz",
        "hdmap_vector_feature.gz",
        "object_temporal_feature_v1.gz",
        "trajectory_target.gz",
    ):
        _write_product(stale_path, file_name)

    report, object_note = inspect_dataset(
        DatasetSpec("dataset", data_root, cache_root),
        split="trainval",
        scene_filter=_scene_filter(),
        args=_args(),
    )

    assert report.passed
    assert report.expected_samples == 4
    assert report.cache_directories == 5
    assert report.extra_samples == 1
    assert report.extra_sample_paths == ["log-a/token-4"]
    assert report.expected.valid_counts["hdmap_vector_feature_v2.gz"] == 4
    assert report.extra.valid_counts["hdmap_vector_feature_v2.gz"] == 0
    assert "object_temporal_feature_v1.gz" in report.required_files
    assert object_note.startswith("temporal (4/4)")


def test_checker_fails_when_current_expected_token_lacks_required_product(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    _write_raw_log(data_root)
    expected_tokens = [f"token-{index}" for index in range(4)]
    _write_complete_expected_products(cache_root, expected_tokens)
    (cache_root / "log-a" / "token-2" / "hdmap_vector_feature_v2.gz").unlink()

    report, _ = inspect_dataset(
        DatasetSpec("dataset", data_root, cache_root),
        split="trainval",
        scene_filter=_scene_filter(),
        args=_args(),
    )

    assert not report.passed
    issues = report.expected.issues["hdmap_vector_feature_v2.gz"]
    assert [(issue.sample, issue.reason) for issue in issues] == [
        ("log-a/token-2", "missing")
    ]


def test_expected_index_matches_scene_filter_windowing(tmp_path):
    data_root = tmp_path / "data"
    _write_raw_log(data_root, frame_count=10)
    scene_filter = SceneFilter(
        num_history_frames=2,
        num_future_frames=2,
        frame_stride=2,
        frame_interval=2,
        has_route=True,
    )

    index = build_expected_index(
        data_root / "dataset" / "navsim_logs" / "trainval", scene_filter
    )

    # raw window=7, starts 0 and 2; current frames are start+2.
    assert sorted(sample.token for sample in index.samples) == ["token-2", "token-4"]
    assert not index.errors
