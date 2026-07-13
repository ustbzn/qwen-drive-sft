import importlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)


pdm_score_module = importlib.import_module("navsim.evaluate.pdm_score")
recogdrive_score_module = importlib.import_module(
    "navsim.planning.script.run_pdm_score_recogdrive"
)


class _CapturingSimulator:
    def __init__(self):
        self.trajectory_states = None

    def simulate_proposals(self, trajectory_states, initial_ego_state):
        self.trajectory_states = trajectory_states
        return trajectory_states, None


class _FakeScorer:
    def score_proposals(self, *args, **kwargs):
        proposal_progress = args[0][:, -1, 0]
        normalized_progress = proposal_progress / np.max(proposal_progress)
        self._multi_metrics = np.ones((len(MultiMetricIndex), 2), dtype=np.float64)
        self._weighted_metrics = np.ones((len(WeightedMetricIndex), 2), dtype=np.float64)
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress
        self._weighted_metrics[WeightedMetricIndex.CENTERLINE_CONVERGENCE, 1] = 0.25
        self.aggregate_call_count = 0
        return np.array([0.8, 0.7], dtype=np.float64)

    def _aggregate_scores(self):
        self.aggregate_call_count += 1
        return np.array(
            [
                0.8,
                self._weighted_metrics[WeightedMetricIndex.CENTERLINE_CONVERGENCE, 1],
            ],
            dtype=np.float64,
        )


@pytest.mark.parametrize("use_explicit_reference", [False, True])
def test_pdm_score_selects_progress_reference(monkeypatch, use_explicit_reference):
    metric_reference = object()
    gt_reference = object()
    model_trajectory = object()
    transformed_gt = object()
    transformed_model = object()

    reference_value = 3.0 if use_explicit_reference else 1.0
    arrays = {
        metric_reference: np.full((2, 3), 1.0, dtype=np.float64),
        transformed_gt: np.full((2, 3), 3.0, dtype=np.float64),
        transformed_model: np.full((2, 3), 2.0, dtype=np.float64),
    }

    def fake_transform(trajectory, initial_ego_state):
        if trajectory is gt_reference:
            return transformed_gt
        if trajectory is model_trajectory:
            return transformed_model
        raise AssertionError("unexpected trajectory")

    monkeypatch.setattr(pdm_score_module, "transform_trajectory", fake_transform)
    monkeypatch.setattr(
        pdm_score_module,
        "get_trajectory_as_array",
        lambda trajectory, sampling, time_point: arrays[trajectory],
    )

    metric_cache = SimpleNamespace(
        ego_state=SimpleNamespace(time_point=object()),
        trajectory=metric_reference,
        observation=None,
        centerline=None,
        route_lane_ids=None,
        drivable_area_map=None,
        history_states=None,
    )
    simulator = _CapturingSimulator()

    result = pdm_score_module.pdm_score(
        metric_cache=metric_cache,
        model_trajectory=model_trajectory,
        future_sampling=object(),
        simulator=simulator,
        scorer=_FakeScorer(),
        progress_reference_trajectory=gt_reference if use_explicit_reference else None,
    )

    assert np.all(simulator.trajectory_states[0] == reference_value)
    assert np.all(simulator.trajectory_states[1] == 2.0)
    expected_model_progress = 2.0 / 3.0 if use_explicit_reference else 1.0
    assert result.ego_progress == pytest.approx(expected_model_progress)


def test_l2_loss_uses_explicit_gt_reference(monkeypatch):
    gt_reference = object()
    model_trajectory = object()
    transformed_gt = object()
    transformed_model = object()
    arrays = {
        transformed_gt: np.array([[0.0, 0.0, 1.0], [3.0, 4.0, 1.0]]),
        transformed_model: np.zeros((2, 3), dtype=np.float64),
    }

    def fake_transform(trajectory, initial_ego_state):
        if trajectory is gt_reference:
            return transformed_gt
        if trajectory is model_trajectory:
            return transformed_model
        raise AssertionError("unexpected trajectory")

    monkeypatch.setattr(recogdrive_score_module, "transform_trajectory", fake_transform)
    monkeypatch.setattr(
        recogdrive_score_module,
        "get_trajectory_as_array",
        lambda trajectory, sampling, time_point: arrays[trajectory],
    )

    metric_cache = SimpleNamespace(
        ego_state=SimpleNamespace(time_point=object()),
        trajectory=object(),
    )
    loss = recogdrive_score_module.compute_trajectory_l2_loss(
        metric_cache=metric_cache,
        model_trajectory=model_trajectory,
        future_sampling=object(),
        reference_trajectory=gt_reference,
    )

    assert loss == pytest.approx(2.5)


def test_deserialize_gathered_score_rows_uses_each_rank_size():
    rows_by_rank = [
        [{"token": "rank0"}],
        [
            {"token": "rank1", "details": "x" * 1024},
            {"token": "rank1-extra"},
        ],
    ]
    payloads = [pickle.dumps(rows) for rows in rows_by_rank]
    max_size = max(len(payload) for payload in payloads)

    gathered_results = []
    gathered_sizes = []
    for payload in payloads:
        payload_tensor = torch.tensor(list(payload), dtype=torch.uint8)
        padding = torch.zeros(max_size - len(payload), dtype=torch.uint8)
        gathered_results.append(torch.cat([payload_tensor, padding]))
        gathered_sizes.append(torch.tensor(len(payload)))

    result = recogdrive_score_module.deserialize_gathered_score_rows(
        gathered_results, gathered_sizes
    )

    assert result == rows_by_rank[0] + rows_by_rank[1]


def test_pdm_score_reaggregates_centerline_override(monkeypatch):
    metric_reference = object()
    model_trajectory = object()
    transformed_model = object()
    arrays = {
        metric_reference: np.ones((2, 3), dtype=np.float64),
        transformed_model: np.full((2, 3), 2.0, dtype=np.float64),
    }

    monkeypatch.setattr(
        pdm_score_module,
        "transform_trajectory",
        lambda trajectory, initial_ego_state: transformed_model,
    )
    monkeypatch.setattr(
        pdm_score_module,
        "get_trajectory_as_array",
        lambda trajectory, sampling, time_point: arrays[trajectory],
    )

    metric_cache = SimpleNamespace(
        ego_state=SimpleNamespace(time_point=object()),
        trajectory=metric_reference,
        observation=None,
        centerline=None,
        route_lane_ids=None,
        drivable_area_map=None,
        history_states=None,
    )
    scorer = _FakeScorer()
    result = pdm_score_module.pdm_score(
        metric_cache=metric_cache,
        model_trajectory=model_trajectory,
        future_sampling=object(),
        simulator=_CapturingSimulator(),
        scorer=scorer,
        centerline_convergence_override=1.0,
    )

    assert result.centerline_convergence == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)
    assert scorer.aggregate_call_count == 1


def test_load_scene_merge_map_normalizes_lane_change(tmp_path):
    csv_path = tmp_path / "scenario_types.csv"
    csv_path.write_text(
        "token,scenario_type,scene_merge\n"
        "token-a,global_lane_change,Lane_Change\n"
        "token-b,lane_keep,lane_keep\n",
        encoding="utf-8",
    )

    result = recogdrive_score_module.load_scene_merge_map(csv_path)

    assert result == {
        "token-a": "lane_change",
        "token-b": "lane_keep",
    }
