from typing import Any, Dict, List, Union
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import sys
import uuid
# 确保 recogdrive 仓库根在 path 中，以便 Hydra 能解析 navsim.agents.recogdrive
_devkit_root = os.environ.get("NAVSIM_DEVKIT_ROOT")
if _devkit_root and _devkit_root not in sys.path:
    sys.path.insert(0, _devkit_root)
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import pickle
import io
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import numpy.typing as npt

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import absolute_to_relative_poses

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.evaluate.pdm_score import (
    pdm_score,
    transform_trajectory,
    get_trajectory_as_array,
)
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.scenario_builder.mogo_map_api_builder import build_mogo_map_api_builder_from_cfg

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def save_predicted_trajectory(output_dir: Path, token: str, trajectory: Trajectory) -> Path:
    trajectory_dir = output_dir / "predicted_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f"{token}.npz"
    np.savez_compressed(
        trajectory_path,
        poses=np.asarray(trajectory.poses, dtype=np.float32),
        interval_length=np.asarray(float(trajectory.trajectory_sampling.interval_length), dtype=np.float32),
    )
    return trajectory_path.resolve()


def simulate_lqr_trajectory(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    simulator: PDMSimulator,
) -> npt.NDArray[np.float64]:
    """复现 pdm_score 内部的 LQR + 运动学自行车模型 rollout，返回模型轨迹对应的仿真状态。

    与 pdm_score 保持完全一致的构造流程（transform_trajectory -> simulate_proposals），
    只取模型轨迹这一条 proposal 的 rollout 结果。返回全局系 state array，形状
    (num_poses+1, StateIndex.size())，列含义见 StateIndex（x/y/heading/vx/vy/ax/ay/
    转向角/转向率/角速度/角加速度），采样为 proposal_sampling（默认 41 点 @ 10Hz / 4s）。
    """
    initial_ego_state = metric_cache.ego_state
    future_sampling = simulator.proposal_sampling
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    pred_states = get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point)

    # simulate_proposals 支持 batch；这里只放模型轨迹一条 proposal。
    simulated_states, _ = simulator.simulate_proposals(pred_states[None, ...], initial_ego_state)
    return simulated_states[0]


def global_states_to_ego_poses(
    simulated_states: npt.NDArray[np.float64],
    initial_ego_state,
) -> npt.NDArray[np.float32]:
    """把 rollout 的全局系 state array 转回自车局部系 (x, y, heading)。

    以初始自车后轴位姿为原点，使得输出与模型 8 点输出、predicted_trajectories 同坐标系，
    可直接叠加对比。
    """
    origin = initial_ego_state.rear_axle
    absolute_poses = [origin] + [
        StateSE2(
            float(state[StateIndex.X]),
            float(state[StateIndex.Y]),
            float(state[StateIndex.HEADING]),
        )
        for state in simulated_states
    ]
    relative_poses = absolute_to_relative_poses(absolute_poses)
    # 丢掉首个原点（(0,0,0)），只保留 rollout 各步，对齐 simulated_states 的时间轴。
    ego_poses = np.array(
        [[pose.x, pose.y, pose.heading] for pose in relative_poses[1:]],
        dtype=np.float32,
    )
    return ego_poses


def save_lqr_trajectory(
    output_dir: Path,
    token: str,
    simulated_states: npt.NDArray[np.float64],
    ego_poses: npt.NDArray[np.float32],
    interval_length: float,
) -> Path:
    """保存 LQR + 自行车模型 rollout 轨迹：全局系完整 state array + 自车局部系 poses。"""
    trajectory_dir = output_dir / "predicted_trajectories_LQR"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f"{token}.npz"
    np.savez_compressed(
        trajectory_path,
        # 全局系完整状态（含速度/加速度/转向角等动力学量），列含义见 StateIndex。
        global_states=np.asarray(simulated_states, dtype=np.float32),
        # 自车局部系 (x, y, heading)，与模型 8 点输出同坐标系，便于直接对比。
        ego_poses=np.asarray(ego_poses, dtype=np.float32),
        interval_length=np.asarray(float(interval_length), dtype=np.float32),
    )
    return trajectory_path.resolve()


def save_model_inputs(
    output_dir: Path,
    token: str,
    features: Dict[str, "torch.Tensor"],
) -> Path:
    """保存模型输入 features（未加 batch 维的单样本张量），供部署端复现推理并对齐输出。

    features 覆盖 ego status / history / command、可选 obj、可选 hdmap，以及启用 VLM 时的
    first/last_hidden_state。全部转为 numpy 后压缩存为单个 npz。
    """
    inputs_dir = output_dir / "model_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = inputs_dir / f"{token}.npz"
    arrays = {key: value.detach().cpu().numpy() for key, value in features.items()}
    np.savez_compressed(inputs_path, **arrays)
    return inputs_path.resolve()


def compute_trajectory_l2_loss(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling,
    reference_trajectory: Trajectory | None = None,
) -> float:
    """
    Compute average L2 distance to a reference trajectory in XY plane.

    When no explicit reference is provided, preserve the historical behavior of
    comparing against the trajectory stored in the metric cache.
    """
    initial_ego_state = metric_cache.ego_state
    reference_trajectory_global = (
        transform_trajectory(reference_trajectory, initial_ego_state)
        if reference_trajectory is not None
        else metric_cache.trajectory
    )
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)

    reference_states = get_trajectory_as_array(
        reference_trajectory_global, future_sampling, initial_ego_state.time_point
    )
    pred_states = get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point)

    # Use XY only: heading has different unit/scale and should not be directly mixed with meters.
    reference_xy = reference_states[:, :2]
    pred_xy = pred_states[:, :2]
    per_step_l2 = np.linalg.norm(pred_xy - reference_xy, axis=1)
    return float(np.mean(per_step_l2))


def load_scene_merge_map(csv_path: Path) -> Dict[str, str]:
    """Load token -> scene_merge metadata used only for scoring decisions."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Scenario type CSV does not exist: {csv_path}")

    scenario_frame = pd.read_csv(csv_path, dtype="string", keep_default_na=False)
    required_columns = {"token", "scene_merge"}
    missing_columns = required_columns - set(scenario_frame.columns)
    if missing_columns:
        raise ValueError(
            f"Scenario type CSV is missing columns: {sorted(missing_columns)}"
        )

    scenario_frame = scenario_frame.loc[:, ["token", "scene_merge"]].copy()
    for column in scenario_frame.columns:
        scenario_frame[column] = scenario_frame[column].str.strip()

    if scenario_frame["token"].eq("").any():
        raise ValueError(f"Scenario type CSV contains empty tokens: {csv_path}")
    duplicated_tokens = scenario_frame.loc[
        scenario_frame["token"].duplicated(keep=False), "token"
    ].unique()
    if len(duplicated_tokens) > 0:
        raise ValueError(
            f"Scenario type CSV contains duplicate tokens: {duplicated_tokens[:5].tolist()}"
        )

    scenario_frame["scene_merge"] = scenario_frame["scene_merge"].str.lower()
    return dict(zip(scenario_frame["token"], scenario_frame["scene_merge"]))


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    legacy_gt_reference = cfg.get("use_gt_trajectory", None)
    use_gt_progress_reference = bool(
        cfg.get(
            "use_gt_progress_reference",
            False if legacy_gt_reference is None else legacy_gt_reference,
        )
    )
    if legacy_gt_reference is not None and "use_gt_progress_reference" not in cfg:
        logger.warning(
            "`use_gt_trajectory` is deprecated; interpreting it as "
            "`use_gt_progress_reference` so the model trajectory is still evaluated."
        )

    # 是否额外保存 LQR rollout 轨迹与模型输入（默认开启；见 shell 脚本 SAVE_MODEL_IO）。
    save_model_io = bool(cfg.get("save_model_io", True))

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    sensor_config = agent.get_sensor_config()
    can_return_features = hasattr(agent, "compute_trajectory_and_features")
    if save_model_io and not can_return_features:
        logger.warning(
            "save_model_io=True but agent %s has no compute_trajectory_and_features; "
            "model inputs will not be saved.",
            type(agent).__name__,
        )

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    map_api_builder = build_mogo_map_api_builder_from_cfg(cfg) if getattr(cfg, "mogo_map", None) else None
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
        map_api_builder=map_api_builder,
    )

    requires_scene = bool(getattr(agent, "requires_scene", False))
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate)

    if use_gt_progress_reference:
        logger.info("GT progress reference enabled. Model predictions will still be evaluated.")

    # 变道中心线满分模式
    lane_change_centerline_full_score = cfg.get("lane_change_centerline_full_score", False)
    scene_merge_map: Dict[str, str] = {}
    if lane_change_centerline_full_score:
        scenario_type_csv_path = cfg.get("scenario_type_csv_path", None)
        if not scenario_type_csv_path:
            raise ValueError(
                "scenario_type_csv_path is required when "
                "lane_change_centerline_full_score=true"
            )
        scene_merge_map = load_scene_merge_map(Path(scenario_type_csv_path))
        missing_tokens = sorted(set(tokens_to_evaluate) - set(scene_merge_map))
        if missing_tokens:
            raise ValueError(
                f"Scenario type CSV is missing {len(missing_tokens)} evaluation tokens; "
                f"examples: {missing_tokens[:5]}"
            )
        lane_change_count = sum(
            scene_merge_map[token] == "lane_change"
            for token in tokens_to_evaluate
        )
        logger.info(
            "Lane change centerline full score enabled for %d/%d local tokens.",
            lane_change_count,
            len(tokens_to_evaluate),
        )

    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        if dist.get_rank() == 0:
            logger.info(f"Rank {dist.get_rank()} processing scenario {idx+1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}")

        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            scene = None
            if requires_scene or use_gt_progress_reference:
                scene = scene_loader.get_scene_from_token(token)

            agent_input = scene_loader.get_agent_input_from_token(token)
            scene_arg = scene if requires_scene else None
            input_features: Dict[str, Any] = {}
            if save_model_io and can_return_features:
                trajectory, input_features = agent.compute_trajectory_and_features(
                    agent_input, scene_arg
                )
            elif requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)

            gt_progress_reference = (
                scene.get_future_trajectory() if use_gt_progress_reference else None
            )
            score_row["trajectory_source"] = "model"
            score_row["progress_reference_source"] = (
                "GT" if gt_progress_reference is not None else "PDM"
            )
            is_lane_change = (
                lane_change_centerline_full_score
                and scene_merge_map.get(token) == "lane_change"
            )

            trajectory_path = save_predicted_trajectory(Path(cfg.output_dir), token, trajectory)

            if save_model_io:
                # LQR + 运动学自行车模型 rollout：复现打分内部仿真，落盘全局系状态与自车局部系 poses。
                simulated_states = simulate_lqr_trajectory(
                    metric_cache=metric_cache,
                    model_trajectory=trajectory,
                    simulator=simulator,
                )
                lqr_ego_poses = global_states_to_ego_poses(
                    simulated_states, metric_cache.ego_state
                )
                lqr_path = save_lqr_trajectory(
                    Path(cfg.output_dir),
                    token,
                    simulated_states=simulated_states,
                    ego_poses=lqr_ego_poses,
                    interval_length=simulator.proposal_sampling.interval_length,
                )
                score_row["lqr_trajectory_path"] = str(lqr_path)
                if input_features:
                    inputs_path = save_model_inputs(Path(cfg.output_dir), token, input_features)
                    score_row["model_inputs_path"] = str(inputs_path)

            l2_loss = compute_trajectory_l2_loss(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                reference_trajectory=gt_progress_reference,
            )

            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                progress_reference_trajectory=gt_progress_reference,
                centerline_convergence_override=1.0 if is_lane_change else None,
            )

            score_row.update(asdict(pdm_result))
            score_row["trajectory_path"] = str(trajectory_path)
            score_row["l2_loss"] = l2_loss
            score_row['rank'] = dist.get_rank()
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False
            score_row["trajectory_path"] = ""
            score_row["l2_loss"] = np.nan

        pdm_results.append(score_row)
    serialized_score_rows = pickle.dumps(pdm_results)
    return serialized_score_rows


def deserialize_gathered_score_rows(
    gathered_results: List[torch.Tensor],
    gathered_sizes: List[torch.Tensor],
) -> List[Dict[str, Any]]:
    if len(gathered_results) != len(gathered_sizes):
        raise ValueError("Gathered result and size counts do not match")

    score_rows: List[Dict[str, Any]] = []
    for rank, (gathered_tensor, gathered_size) in enumerate(
        zip(gathered_results, gathered_sizes)
    ):
        payload_size = int(gathered_size.item())
        if payload_size < 0 or payload_size > gathered_tensor.numel():
            raise ValueError(
                f"Invalid gathered payload size for rank {rank}: "
                f"{payload_size} > {gathered_tensor.numel()}"
            )
        serialized_data = gathered_tensor[:payload_size].cpu().numpy().tobytes()
        score_rows.extend(pickle.loads(serialized_data))

    return score_rows

def broadcast_object(obj: Any, device: torch.device, src: int = 0) -> Any:
    """
    Helper function to broadcast an object from the source rank to all other processes.
    :param obj: Object to broadcast.
    :param device: Device to use for tensor operations.
    :param src: Source rank.
    :return: Broadcasted object.
    """
    if dist.get_rank() == src:
        buffer = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(buffer)).to(device)
        size_tensor = torch.tensor(len(tensor)).to(device)
        dist.broadcast(size_tensor, src=src)
        dist.broadcast(tensor, src=src)
    else:
        size_tensor = torch.tensor(0).to(device)
        dist.broadcast(size_tensor, src=src)
        tensor = torch.ByteTensor(size_tensor.item()).to(device)
        dist.broadcast(tensor, src=src)
        buffer = tensor.cpu().numpy().tobytes()
        obj = pickle.loads(buffer)
    return obj


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    build_logger(cfg)
    worker = build_worker(cfg)


    scene_loader = SceneLoader(
            sensor_blobs_path=None,
            data_path=Path(cfg.navsim_log_path),
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_config=SensorConfig.build_no_sensors(),
        )
    if rank == 0:
        metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
        tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))

        # --- badcase 模式：仅评测指定 token 列表中的样本 ---
        badcase_mode = cfg.get("badcase_mode", False)
        if badcase_mode:
            badcase_token_list_path = cfg.get("badcase_token_list", None)
            if badcase_token_list_path and Path(badcase_token_list_path).exists():
                logger.info(f"Badcase mode enabled. Loading token list from {badcase_token_list_path}")
                with open(badcase_token_list_path, "r") as f:
                    badcase_tokens = set(line.strip() for line in f if line.strip())
                tokens_to_evaluate = [t for t in tokens_to_evaluate if t in badcase_tokens]
                logger.info(f"Badcase mode: {len(tokens_to_evaluate)} tokens to evaluate after filtering")
            else:
                logger.warning(f"Badcase mode enabled but token list file not found: {badcase_token_list_path}")

        tokens_to_evaluate = sorted(tokens_to_evaluate)
        num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
        num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
        if num_missing_metric_cache_tokens > 0:
            logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
        if num_unused_metric_cache_tokens > 0:
            logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    else:
        tokens_to_evaluate = []

    tokens_to_evaluate = broadcast_object(tokens_to_evaluate, device=device, src=0)


    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))

    sampler = InferenceSampler(len(tokens_to_evaluate))

    data_points = []
    for idx in sampler:
        token = tokens_to_evaluate[idx]
        log_file = scene_loader.token_to_log_file[token] 
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],
        })

    serialized_score_rows = run_pdm_score(data_points)


    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    serialized_tensor = torch.ByteTensor(list(serialized_score_rows)).to(device)

    local_size = len(serialized_tensor)
    size_list = [torch.tensor(local_size).to(device) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, torch.tensor(local_size).to(device))

    max_size = max(size_list).item() 

    if local_size < max_size:
        padded_tensor = torch.cat([serialized_tensor, torch.zeros(max_size - local_size, dtype=torch.uint8).to(device)])
    else:
        padded_tensor = serialized_tensor

    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    if dist.get_rank() == 0:
        final_results = deserialize_gathered_score_rows(gathered_results, size_list)
        pdm_score_df = pd.DataFrame(final_results)

        num_sucessful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
        average_drop_columns = [
            column
            for column in [
                "token",
                "valid",
                "rank",
                "trajectory_path",
                "lqr_trajectory_path",
                "model_inputs_path",
                "trajectory_source",
                "progress_reference_source",
            ]
            if column in pdm_score_df.columns
        ]
        average_row = pdm_score_df.drop(columns=average_drop_columns).mean(skipna=True)
        average_row["token"] = "average"
        average_row["valid"] = pdm_score_df["valid"].all()
        average_row["rank"] = 0
        for path_column in ("trajectory_path", "lqr_trajectory_path", "model_inputs_path"):
            if path_column in pdm_score_df.columns:
                average_row[path_column] = ""
        if "trajectory_source" in pdm_score_df.columns:
            average_row["trajectory_source"] = ""
        if "progress_reference_source" in pdm_score_df.columns:
            average_row["progress_reference_source"] = ""
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        save_path = Path(cfg.output_dir)
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        pdm_score_df.to_csv(save_path / f"{timestamp}.csv", index=False)

        logger.info(
            f"""
            Finished running evaluation.
                Number of successful scenarios: {num_sucessful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
                Final average score of valid results: {pdm_score_df['score'].mean()}.
                Results are stored in: {save_path / f"{timestamp}.csv"}.
            """
        )


if __name__ == "__main__":
    main()
