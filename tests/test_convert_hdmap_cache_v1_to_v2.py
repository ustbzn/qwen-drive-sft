import gzip
import pickle
from pathlib import Path

import torch

from scripts.cache_dataset.convert_hdmap_cache_v1_to_v2 import (
    CONTROL_NONE,
    DIRECTION_NONE,
    ROUTE_UNKNOWN,
    SOURCE_FEATURE_NAME,
    TARGET_FEATURE_NAME,
    TL_NONE,
    V2_KEYS,
    DatasetSpec,
    audit_dataset_logs,
    convert_feature_file,
    convert_v1_feature_dict,
    process_dataset,
)


def _v1_payload(route_states=None):
    max_tokens = 3
    sample_points = 4
    route_states = route_states or [1, 0, 0]
    return {
        "map_point_position": torch.arange(
            max_tokens * 3 * sample_points * 2, dtype=torch.float32
        ).reshape(max_tokens, 3, sample_points, 2),
        "map_point_vector": torch.ones(
            max_tokens, 3, sample_points, 2, dtype=torch.float32
        ),
        "map_point_orientation": torch.zeros(
            max_tokens, 3, sample_points, dtype=torch.float32
        ),
        "map_polygon_center": torch.zeros(max_tokens, 3, dtype=torch.float32),
        "map_polygon_type": torch.tensor([0, 1, 2], dtype=torch.long),
        "map_polygon_on_route": torch.tensor(route_states, dtype=torch.long),
        "map_polygon_tl_status": torch.tensor([0, 3, 3], dtype=torch.long),
        "map_polygon_has_speed_limit": torch.tensor([True, False, False]),
        "map_polygon_speed_limit": torch.tensor([10.0, 0.0, 0.0]),
        "map_point_valid_mask": torch.tensor(
            [[1.0] * sample_points, [1.0] * sample_points, [0.0] * sample_points]
        ),
        "map_valid_mask": torch.tensor([1.0, 1.0, 0.0]),
    }


def _write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)


def _write_gzip_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=1) as file_handle:
        pickle.dump(payload, file_handle)


def test_convert_v1_payload_preserves_geometry_and_adds_v2_none_semantics():
    source = _v1_payload()

    converted = convert_v1_feature_dict(source)

    assert tuple(converted) == V2_KEYS
    assert torch.equal(converted["map_point_position"], source["map_point_position"])
    assert converted["map_polygon_on_route"].tolist() == [1, 0, ROUTE_UNKNOWN]
    assert converted["map_polygon_tl_status"].tolist() == [TL_NONE] * 3
    assert converted["map_polygon_control_type"].tolist() == [CONTROL_NONE] * 3
    assert converted["map_polygon_tl_direction"].tolist() == [DIRECTION_NONE] * 3
    assert source["map_polygon_tl_status"].tolist() == [0, 3, 3]


def test_convert_v1_payload_downgrades_ambiguous_all_off_route_to_unknown():
    converted = convert_v1_feature_dict(_v1_payload(route_states=[0, 0, 0]))

    assert converted["map_polygon_on_route"].tolist() == [
        ROUTE_UNKNOWN,
        ROUTE_UNKNOWN,
        ROUTE_UNKNOWN,
    ]


def test_audit_requires_every_log_to_have_no_nonempty_rich_field(tmp_path):
    log_root = tmp_path / "dataset" / "navsim_logs" / "train"
    _write_pickle(log_root / "old.pkl", [{"token": "a"}, {"traffic_light_states": []}])

    clean_audit = audit_dataset_logs(tmp_path / "dataset" / "navsim_logs", workers=2)

    assert clean_audit.eligible
    assert clean_audit.scanned_files == 1
    assert clean_audit.field_present_frames == 1
    assert clean_audit.rich_frames == 0

    _write_pickle(
        log_root / "rich.pkl",
        [{"traffic_light_states": [{"stop_line_id": 17, "state": "red"}]}],
    )

    rich_audit = audit_dataset_logs(tmp_path / "dataset" / "navsim_logs", workers=2)

    assert not rich_audit.eligible
    assert rich_audit.scanned_files == 2
    assert rich_audit.rich_frames == 1
    assert rich_audit.rich_examples


def test_process_dataset_converts_concurrently_and_is_incremental(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    _write_pickle(
        data_root / "dataset" / "navsim_logs" / "train" / "log-a.pkl",
        [{"token": "a"}, {"traffic_light_states": None}],
    )
    for token in ("token-a", "token-b"):
        _write_gzip_pickle(
            cache_root / "log-a" / token / SOURCE_FEATURE_NAME,
            _v1_payload(),
        )
    spec = DatasetSpec("clean", data_root, cache_root)

    assert process_dataset(spec, apply=True, workers=2, audit_workers=2, progress_every=0)

    for token in ("token-a", "token-b"):
        target = cache_root / "log-a" / token / TARGET_FEATURE_NAME
        assert target.is_file()
        with gzip.open(target, "rb") as file_handle:
            converted = pickle.load(file_handle)
        assert tuple(converted) == V2_KEYS
        assert torch.all(converted["map_polygon_tl_status"] == TL_NONE)

    first_mtime = (cache_root / "log-a" / "token-a" / TARGET_FEATURE_NAME).stat().st_mtime_ns
    assert process_dataset(spec, apply=True, workers=2, audit_workers=2, progress_every=0)
    assert (
        cache_root / "log-a" / "token-a" / TARGET_FEATURE_NAME
    ).stat().st_mtime_ns == first_mtime


def test_process_dataset_refuses_rich_logs_before_writing(tmp_path):
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    _write_pickle(
        data_root / "dataset" / "navsim_logs" / "train" / "log-a.pkl",
        [{"traffic_light_states": [{"stop_line_id": 17}]}],
    )
    source = cache_root / "log-a" / "token-a" / SOURCE_FEATURE_NAME
    _write_gzip_pickle(source, _v1_payload())
    spec = DatasetSpec("rich", data_root, cache_root)

    assert not process_dataset(spec, apply=True, workers=2, audit_workers=2, progress_every=0)
    assert not source.with_name(TARGET_FEATURE_NAME).exists()


def test_invalid_v1_cache_returns_error_without_partial_target(tmp_path):
    source = tmp_path / SOURCE_FEATURE_NAME
    _write_gzip_pickle(source, {"map_valid_mask": torch.ones(3)})

    result = convert_feature_file(source, apply=True)

    assert result.status == "error"
    assert "missing v1 keys" in result.error
    assert not source.with_name(TARGET_FEATURE_NAME).exists()
