from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import os
import traceback
import uuid

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.common.dataloader import MetricCacheLoader, SceneLoader, SceneFilter
from navsim.common.dataclasses import NAVSIM_INTERVAL_LENGTH, SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gt"


def load_token_list(token_path: Path) -> List[str]:
    """
    Load one token per line while preserving file order and removing duplicates.
    :param token_path: token whitelist path.
    :return: ordered unique tokens.
    """
    if not token_path.is_file():
        raise FileNotFoundError(f"Token file does not exist: {token_path}")

    tokens: List[str] = []
    seen_tokens = set()
    for raw_line in token_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        token = line.split()[0]
        if token not in seen_tokens:
            tokens.append(token)
            seen_tokens.add(token)

    if not tokens:
        raise ValueError(f"Token file is empty: {token_path}")
    return tokens


def resolve_gt_num_future_frames(cfg: DictConfig, proposal_sampling: Any) -> int:
    """
    Resolve GT trajectory length in the native NAVSIM 2Hz frame count.
    :param cfg: hydra config.
    :param proposal_sampling: PDM proposal sampling.
    :return: number of future 2Hz frames used as GT trajectory.
    """
    configured_frames = getattr(cfg, "gt_num_future_frames", None)
    if configured_frames is not None:
        configured_frames = int(configured_frames)
        if configured_frames <= 0:
            raise ValueError(f"gt_num_future_frames must be positive, got {configured_frames}")
        return configured_frames

    time_horizon = float(
        getattr(
            proposal_sampling,
            "time_horizon",
            proposal_sampling.num_poses * proposal_sampling.interval_length,
        )
    )
    gt_num_future_frames = int(round(time_horizon / NAVSIM_INTERVAL_LENGTH))
    if gt_num_future_frames <= 0:
        raise ValueError(f"Could not infer GT frame count from time_horizon={time_horizon}")
    return gt_num_future_frames


def build_scene_loader(cfg: DictConfig, tokens: Optional[List[str]], log_names: Union[List[str], None] = None) -> SceneLoader:
    """
    Build a no-sensor scene loader for the selected tokens.
    :param cfg: hydra config.
    :param tokens: selected scene tokens.
    :param log_names: optional log-name filter for worker sharding.
    :return: scene loader.
    """
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.tokens = tokens
    if log_names is not None:
        scene_filter.log_names = log_names

    return SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Worker helper for GT PDM-Score evaluation.
    :param args: worker input chunks.
    :return: score rows.
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info("Starting GT scoring worker thread_id=%s, node_id=%s", thread_id, node_id)

    log_names = [a["log_file"] for a in args]
    tokens = [token for a in args for token in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, (
        "Simulator and scorer proposal sampling has to be identical"
    )
    gt_num_future_frames = resolve_gt_num_future_frames(cfg, simulator.proposal_sampling)

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_loader = build_scene_loader(cfg, tokens=tokens, log_names=log_names)

    available_tokens = set(scene_loader.tokens) & set(metric_cache_loader.tokens)
    tokens_to_evaluate = [token for token in tokens if token in available_tokens]

    score_rows: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(
            "Processing GT scenario %s / %s in thread_id=%s, node_id=%s",
            idx + 1,
            len(tokens_to_evaluate),
            thread_id,
            node_id,
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            gt_trajectory = scene.get_future_trajectory(num_trajectory_frames=gt_num_future_frames)

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=gt_trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            score_row["gt_num_future_frames"] = gt_num_future_frames
        except Exception as exc:
            logger.warning("----------- GT scoring failed for token %s:", token)
            traceback.print_exc()
            score_row["valid"] = False
            score_row["error"] = repr(exc)

        score_rows.append(score_row)

    return score_rows


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for scoring selected cases with ground-truth trajectories.
    :param cfg: hydra config.
    """
    build_logger(cfg)
    worker = build_worker(cfg)

    gt_token_path = getattr(cfg, "gt_token_path", None)
    use_all_split_tokens = gt_token_path is None or str(gt_token_path).strip().lower() in {"", "none", "null"}
    selected_tokens = None if use_all_split_tokens else load_token_list(Path(gt_token_path))
    scene_loader = build_scene_loader(cfg, tokens=selected_tokens)
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_tokens = set(scene_loader.tokens)
    metric_cache_tokens = set(metric_cache_loader.tokens)
    if selected_tokens is None:
        ordered_source_tokens = scene_loader.tokens
        total_source_tokens = len(scene_loader.tokens)
        tokens_to_evaluate = [token for token in ordered_source_tokens if token in metric_cache_tokens]
        missing_cache_tokens = [token for token in ordered_source_tokens if token not in metric_cache_tokens]
        if missing_cache_tokens:
            logger.warning("Split tokens missing metric cache: %s", len(missing_cache_tokens))
    else:
        ordered_source_tokens = selected_tokens
        total_source_tokens = len(selected_tokens)
        tokens_to_evaluate = [
            token for token in ordered_source_tokens if token in scene_tokens and token in metric_cache_tokens
        ]
        missing_scene_tokens = [token for token in ordered_source_tokens if token not in scene_tokens]
        missing_cache_tokens = [
            token for token in ordered_source_tokens if token in scene_tokens and token not in metric_cache_tokens
        ]
        if missing_scene_tokens:
            logger.warning("Selected tokens missing from scene split: %s", len(missing_scene_tokens))
        if missing_cache_tokens:
            logger.warning("Selected tokens missing metric cache: %s", len(missing_cache_tokens))
    if not tokens_to_evaluate:
        raise ValueError(
            "No token can be evaluated. Check gt_token_path, train_test_split.data_split, "
            "and metric_cache_path."
        )

    logger.info(
        "Starting GT PDM scoring for %s / %s source tokens.",
        len(tokens_to_evaluate),
        total_source_tokens,
    )

    tokens_per_log: Dict[str, List[str]] = {}
    for token in tokens_to_evaluate:
        log_file = scene_loader.token_to_log_file[token]
        tokens_per_log.setdefault(log_file, []).append(token)

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens,
        }
        for log_file, tokens in tokens_per_log.items()
    ]
    score_rows = worker_map(worker, run_pdm_score, data_points)

    pdm_score_df = pd.DataFrame(score_rows)
    num_successful_scenarios = int(pdm_score_df["valid"].sum())
    num_failed_scenarios = len(pdm_score_df) - num_successful_scenarios

    average_drop_columns = [
        column
        for column in ["token", "valid", "error", "gt_num_future_frames"]
        if column in pdm_score_df.columns
    ]
    average_row = pdm_score_df.drop(columns=average_drop_columns).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    if "error" in pdm_score_df.columns:
        average_row["error"] = ""
    if "gt_num_future_frames" in pdm_score_df.columns:
        average_row["gt_num_future_frames"] = ""
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    output_csv_path = save_path / f"{timestamp}.csv"
    pdm_score_df.to_csv(output_csv_path)
    average_score = average_row["score"] if "score" in average_row else float("nan")

    logger.info(
        """
        Finished running GT evaluation.
            Number of successful scenarios: %s.
            Number of failed scenarios: %s.
            Final average score of valid results: %s.
            Results are stored in: %s.
        """,
        num_successful_scenarios,
        num_failed_scenarios,
        average_score,
        output_csv_path,
    )


if __name__ == "__main__":
    main()
