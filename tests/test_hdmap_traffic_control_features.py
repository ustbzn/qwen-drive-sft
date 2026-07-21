from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from navsim.agents.recogdrive import recogdrive_agent as recogdrive_agent_module
from navsim.agents.recogdrive.hdmap_vector_features import (
    HDMAP_VECTOR_FEATURE_NAME,
    HDMapControlType,
    HDMapPolygonType,
    HDMapRouteRelation,
    HDMapTrafficLightDirection,
    HDMapTrafficLightStatus,
    HDMapVectorFeatureBuilder,
)
from navsim.common.dataclasses import Scene, SensorConfig
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


class _Path:
    def __init__(self, states):
        self.discrete_path = states


class _MapObject:
    def __init__(self, object_id, roadblock_id, states):
        self.id = object_id
        self._roadblock_id = roadblock_id
        self.baseline_path = _Path(states)
        self.left_boundary = None
        self.right_boundary = None
        self.incoming_edges = []
        self.outgoing_edges = []
        self.speed_limit_mps = 10.0

    def get_roadblock_id(self):
        return self._roadblock_id


class _MapAPI:
    def __init__(self, lanes):
        self._lanes = lanes

    def load(self, point):
        del point

    def get_proximal_map_objects(self, point, radius, layers):
        del point, radius, layers
        return {
            SemanticMapLayer.LANE: self._lanes,
            SemanticMapLayer.LANE_CONNECTOR: [],
        }


def _scene(traffic_light_states, map_api=None, roadblock_ids=None):
    frame = SimpleNamespace(
        ego_status=SimpleNamespace(ego_pose=np.array([0.0, 0.0, 0.0])),
        roadblock_ids=list(roadblock_ids or []),
        traffic_lights=[],
        traffic_light_states=traffic_light_states,
    )
    return SimpleNamespace(
        scene_metadata=SimpleNamespace(num_history_frames=1),
        frames=[frame],
        map_api=map_api,
    )


def test_control_token_is_prioritized_and_keeps_rich_attributes():
    lane = _MapObject(
        object_id="lane-1",
        roadblock_id="route-1",
        states=[StateSE2(0.0, 0.0, 0.0), StateSE2(50.0, 0.0, 0.0)],
    )
    traffic_light_states = [
        {
            "stop_line_id": (1 << 64) - 1,
            "direction": "left",
            "state": "green",
            "position": [[5.0, -2.0, 0.0], [5.0, 2.0, 0.0]],
        },
        {
            "stop_line_id": 101,
            "direction": "straight",
            "state": "flash",
            "position": [[20.0, -2.0, 0.0], [20.0, 2.0, 0.0]],
        },
    ]
    builder = HDMapVectorFeatureBuilder(
        max_map_tokens=2,
        map_sample_points=4,
        max_control_tokens=1,
    )

    features = builder.compute_features_from_scene(
        _scene(
            traffic_light_states,
            map_api=_MapAPI([lane]),
            roadblock_ids=["route-1"],
        )
    )

    assert HDMAP_VECTOR_FEATURE_NAME == "hdmap_vector_feature_v2"
    assert features["map_valid_mask"].tolist() == [1.0, 1.0]
    assert features["map_polygon_type"].tolist() == [
        HDMapPolygonType.STOP_LINE,
        HDMapPolygonType.LANE,
    ]
    assert features["map_polygon_control_type"].tolist() == [
        HDMapControlType.TRAFFIC_LIGHT,
        HDMapControlType.NONE,
    ]
    assert features["map_polygon_tl_status"].tolist() == [
        HDMapTrafficLightStatus.RED,
        HDMapTrafficLightStatus.NONE,
    ]
    assert features["map_polygon_tl_direction"].tolist() == [
        HDMapTrafficLightDirection.STRAIGHT,
        HDMapTrafficLightDirection.NONE,
    ]
    assert features["map_polygon_on_route"].tolist() == [
        HDMapRouteRelation.ON_ROUTE,
        HDMapRouteRelation.ON_ROUTE,
    ]
    assert torch.isfinite(features["map_point_position"]).all()


def test_missing_control_metadata_creates_no_control_or_negative_token():
    builder = HDMapVectorFeatureBuilder(max_map_tokens=3, map_sample_points=4)

    features = builder.compute_features_from_scene(_scene([], map_api=None))

    assert torch.count_nonzero(features["map_valid_mask"]) == 0
    assert torch.all(features["map_polygon_control_type"] == HDMapControlType.NONE)
    assert torch.all(features["map_polygon_tl_status"] == HDMapTrafficLightStatus.NONE)
    assert torch.all(
        features["map_polygon_tl_direction"] == HDMapTrafficLightDirection.NONE
    )
    assert torch.all(features["map_polygon_on_route"] == HDMapRouteRelation.UNKNOWN)


def test_invalid_entries_do_not_hide_later_valid_control_geometry():
    builder = HDMapVectorFeatureBuilder(max_map_tokens=4, map_sample_points=4)
    states = [
        {
            "stop_line_id": 8,
            "direction": "left",
            "state": "red",
            "position": [],
        },
        {
            "stop_line_id": 8,
            "direction": "left",
            "state": "green_flash",
            "position": [[10.0, -1.0], [10.0, 1.0]],
        },
        {
            "stop_line_id": 8,
            "direction": "left",
            "state": "red",
            "position": [[10.0, -1.0], [10.0, 1.0]],
        },
        {
            "stop_line_id": 0,
            "direction": "right",
            "state": "red",
            "position": [[12.0, -1.0], [12.0, 1.0]],
        },
    ]

    candidates = builder._build_traffic_control_candidates(
        traffic_light_states=states,
        origin=StateSE2(0.0, 0.0, 0.0),
        route_ids=set(),
        route_map_objects=[],
    )

    assert len(candidates) == 1
    assert candidates[0]["map_polygon_tl_status"].item() == HDMapTrafficLightStatus.GREEN
    assert candidates[0]["map_polygon_tl_direction"].item() == HDMapTrafficLightDirection.LEFT
    assert candidates[0]["map_polygon_on_route"].item() == HDMapRouteRelation.UNKNOWN


@pytest.mark.parametrize(
    ("raw_state", "expected"),
    [
        ("green", HDMapTrafficLightStatus.GREEN),
        ("green_flash", HDMapTrafficLightStatus.GREEN),
        ("red", HDMapTrafficLightStatus.RED),
        ("flash", HDMapTrafficLightStatus.RED),
        ("yellow", HDMapTrafficLightStatus.YELLOW),
        ("off", HDMapTrafficLightStatus.UNKNOWN),
        (None, HDMapTrafficLightStatus.UNKNOWN),
    ],
)
def test_traffic_light_state_mapping(raw_state, expected):
    assert HDMapVectorFeatureBuilder._traffic_light_status(raw_state) == expected


@pytest.mark.parametrize(
    ("raw_direction", "expected"),
    [
        ("straight", HDMapTrafficLightDirection.STRAIGHT),
        ("left", HDMapTrafficLightDirection.LEFT),
        ("right", HDMapTrafficLightDirection.RIGHT),
        ("u_turn", HDMapTrafficLightDirection.U_TURN),
        ("unknown", HDMapTrafficLightDirection.UNKNOWN),
    ],
)
def test_traffic_light_direction_mapping(raw_direction, expected):
    assert HDMapVectorFeatureBuilder._traffic_light_direction(raw_direction) == expected


def test_legacy_navsim_boolean_is_red_semantics_are_not_inverted():
    status = HDMapVectorFeatureBuilder._traffic_light_status_by_lane(
        [("red-lane", True), ("green-lane", False)]
    )

    assert status == {
        "red-lane": HDMapTrafficLightStatus.RED,
        "green-lane": HDMapTrafficLightStatus.GREEN,
    }


def _raw_scene_frame(traffic_light_states_marker):
    camera_names = (
        "cam_f0",
        "cam_l0",
        "cam_l1",
        "cam_l2",
        "cam_r0",
        "cam_r1",
        "cam_r2",
        "cam_b0",
    )
    frame = {
        "token": "token",
        "log_name": "log",
        "scene_token": "scene",
        "timestamp": 0,
        "map_location": "las_vegas",
        "ego2global_translation": [0.0, 0.0, 0.0],
        "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        "ego_dynamic_state": [0.0, 0.0, 0.0, 0.0],
        "driving_command": [1.0, 0.0, 0.0, 0.0],
        "can_bus": [],
        "cams": {name: {} for name in camera_names},
        "lidar_path": "",
        "anns": {
            "gt_boxes": np.empty((0, 7), dtype=np.float32),
            "gt_names": np.empty((0,), dtype=object),
            "gt_velocity_3d": np.empty((0, 3), dtype=np.float32),
            "instance_tokens": [],
            "track_tokens": [],
        },
        "roadblock_ids": [],
        "traffic_lights": [],
    }
    if traffic_light_states_marker is not None:
        frame["traffic_light_states"] = traffic_light_states_marker
    return frame


def test_scene_loader_preserves_rich_field_and_defaults_old_logs_to_empty():
    rich_states = [{"stop_line_id": 3, "state": "red", "position": [[1, 0], [1, 1]]}]
    rich_scene = Scene.from_scene_dict_list(
        [_raw_scene_frame(rich_states)],
        sensor_blobs_path=Path("/tmp"),
        num_history_frames=1,
        num_future_frames=0,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    old_scene = Scene.from_scene_dict_list(
        [_raw_scene_frame(None)],
        sensor_blobs_path=Path("/tmp"),
        num_history_frames=1,
        num_future_frames=0,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    assert rich_scene.frames[0].traffic_light_states == rich_states
    assert rich_scene.frames[0].traffic_light_states is not rich_states
    assert old_scene.frames[0].traffic_light_states == []


class _FakePlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def cuda(self, device=None):
        del device
        return self


def test_hdmap_only_agent_declares_scene_requirement(monkeypatch):
    monkeypatch.setattr(
        recogdrive_agent_module,
        "make_recogdrive_config",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        recogdrive_agent_module,
        "ReCogDriveDiffusionPlanner",
        _FakePlanner,
    )

    agent = recogdrive_agent_module.ReCogDriveAgent(
        trajectory_sampling=object(),
        use_vlm=False,
        cache_hidden_state=True,
        use_hdmap_vector=True,
        use_object_vector=False,
    )

    assert agent.requires_scene is True

