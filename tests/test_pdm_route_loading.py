import logging
from types import SimpleNamespace

import pytest

from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.planning.simulation.planner.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner


class _MapStub:
    def __init__(self, blocks):
        self._blocks = blocks

    def get_map_object(self, object_id, layer):
        if layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
            return self._blocks.get(object_id)
        return None


class _PlannerStub(AbstractPDMPlanner):
    def initialize(self, initialization):
        pass

    def compute_planner_trajectory(self, current_input):
        return None

    def name(self):
        return "planner_stub"

    def observation_type(self):
        return None


def _planner(map_api):
    planner = object.__new__(_PlannerStub)
    planner._map_api = map_api
    planner._route_roadblock_dict = None
    planner._route_lane_dict = None
    return planner


def test_load_route_dicts_skips_missing_map_objects(caplog):
    valid_block = SimpleNamespace(id="roadblock_valid", interior_edges=[])
    planner = _planner(_MapStub({"roadblock_valid": valid_block}))

    with caplog.at_level(logging.WARNING):
        planner._load_route_dicts(["roadblock_missing", "roadblock_valid"])

    assert list(planner._route_roadblock_dict) == ["roadblock_valid"]
    assert "roadblock_missing" in caplog.text


def test_load_route_dicts_rejects_route_with_no_resolvable_objects():
    planner = _planner(_MapStub({}))

    with pytest.raises(ValueError, match="No route roadblocks could be resolved"):
        planner._load_route_dicts(["roadblock_missing"])
