from typing import Dict, Optional
import torch
import numpy as np
import gzip
import pickle
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from navsim.common.command_utils import (
    command_code_to_text,
    encode_nav_command_vector,
    NAV_COMMAND_DIM,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_view_utils import (
    collect_view_frame_paths_with_status,
    flatten_view_frames,
    view_labels_for,
    build_history_str,
    build_driving_question,
)
from .utils.internvl_preprocess import load_image

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


# 运动学量特征名（命令段之后的固定 9 维），命令段视开关而定。
_KINEMATIC_FEATURE_NAMES = (
    "vx",
    "vy",
    "ax",
    "ay",
    "yaw_rate",
    "curvature",
    "curvature_rate",
    "steering_wheel_angle",
    "steering_wheel_rate",
)

# 默认（use_navigation_command=False）：4 维 one-hot 命令 + 9 维运动学 = 13 维，字节级不变。
EGO_STATUS_FEATURE_NAMES = ("cmd_0", "cmd_1", "cmd_2", "cmd_3") + _KINEMATIC_FEATURE_NAMES


def ego_status_feature_names(use_navigation_command: bool = False) -> tuple:
    """按开关返回 status_feature 的字段名。

    - False（默认）：4 维 one-hot 命令 + 9 维运动学（历史布局，13 维）。
    - True：NAV_COMMAND_DIM 维 navigation_command 编码 + 9 维运动学。
    命令段在前、运动学量在后的布局与 StateAttentionEncoder(SDE) 的
    [command(可见), kinematic(可丢)] 掩码顺序一致。
    """
    if use_navigation_command:
        nav_names = tuple(f"nav_{i}" for i in range(NAV_COMMAND_DIM))
        return nav_names + _KINEMATIC_FEATURE_NAMES
    return EGO_STATUS_FEATURE_NAMES


def ego_status_dim(use_navigation_command: bool = False) -> int:
    """status_feature 维度：默认 13；启用 navigation_command 为 NAV_COMMAND_DIM + 9。"""
    return len(ego_status_feature_names(use_navigation_command))


RECOGDRIVE_EGO_FEATURE_CACHE_NAME = "recogdrive_ego_feature_v1"
# navigation_command 命令段改变了 status_feature 布局/维度，版本化缓存名避免与旧 v1 脏读混用。
RECOGDRIVE_EGO_FEATURE_CACHE_NAME_NAV = "recogdrive_ego_feature_v2_nav"
RECOGDRIVE_EGO_FEATURE_KEYS = (
    "history_trajectory",
    "high_command_one_hot",
    "status_feature",
)


def select_hidden_state(hidden_states, layer_index: int) -> torch.Tensor:
    """Select a hidden-state layer by index while preserving Python negative-index semantics."""
    return hidden_states[int(layer_index)]


def build_recogdrive_ego_features(
    agent_input: AgentInput,
    use_navigation_command: bool = False,
) -> Dict[str, torch.Tensor]:
    """构造与 VLM 配置无关的自车历史、导航命令和当前运动学特征。

    :param use_navigation_command: False（默认）用旧 4 维 one-hot 命令段（13 维，
        字节级不变）；True 用 navigation_command 编码段（NAV_COMMAND_DIM + 9 维），
        替换 one-hot，含 park 进出站的 dist/侧向信息，旧数据自动退化（见
        command_utils.encode_nav_command_vector）。
    """
    ego_statuses = agent_input.ego_statuses
    history_trajectory = torch.tensor(
        [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
        dtype=torch.float32,
    )
    current_ego_status = ego_statuses[-1]
    # high_command_one_hot 始终保留旧 4 维 one-hot（供 VLM prompt 的 command_code_to_text），
    # 与 status_feature 的命令段编码方式解耦。
    high_command_one_hot = torch.tensor(current_ego_status.driving_command, dtype=torch.float32)

    if use_navigation_command:
        # navigation_command 编码段（含 park dist/侧向），旧数据从 driving_command 退化。
        command_segment = torch.from_numpy(
            encode_nav_command_vector(
                getattr(current_ego_status, "navigation_command", None),
                current_ego_status.driving_command,
            )
        ).float()
    else:
        command_segment = high_command_one_hot.clone()

    status_feature = torch.cat(
        [
            command_segment,
            torch.tensor(current_ego_status.ego_velocity, dtype=torch.float32),
            torch.tensor(current_ego_status.ego_acceleration, dtype=torch.float32),
            torch.tensor(
                [
                    current_ego_status.ego_yaw_rate,
                    current_ego_status.ego_curvature,
                    current_ego_status.ego_curvature_rate,
                    current_ego_status.ego_steering_wheel_angle,
                    current_ego_status.ego_steering_wheel_rate,
                ],
                dtype=torch.float32,
            ),
        ],
        dim=-1,
    )
    assert status_feature.shape[-1] == ego_status_dim(use_navigation_command)

    return {
        "history_trajectory": history_trajectory.cpu(),
        "high_command_one_hot": high_command_one_hot.cpu(),
        "status_feature": status_feature.cpu(),
    }


class ReCogDriveEgoFeatureBuilder(AbstractFeatureBuilder):
    """将 no-VLM 仍需使用的小型自车特征存为独立 sidecar。"""

    def __init__(self, use_navigation_command: bool = False):
        super().__init__()
        self.use_navigation_command = bool(use_navigation_command)

    def get_unique_name(self) -> str:
        # navigation_command 命令段改变 status_feature 布局，版本化缓存名避免脏读旧 v1。
        if self.use_navigation_command:
            return RECOGDRIVE_EGO_FEATURE_CACHE_NAME_NAV
        return RECOGDRIVE_EGO_FEATURE_CACHE_NAME

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        return build_recogdrive_ego_features(agent_input, self.use_navigation_command)


class ReCogDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False,
                 first_hidden_state_index: int = 0,
                 last_hidden_state_index: int = -1,
                 view_list: Optional[list] = None,
                 num_frames: int = 1,
                 use_video_input: bool = False,
                 video_fps: float = 2.0,
                 use_navigation_command: bool = False, ):
        """
        Initializes the feature builder.

        Args:
            cache_hidden_state (bool): If True, operates in online mode, initializes the backbone,
                                       and computes the hidden state. If False, operates in offline
                                       mode, does not initialize the backbone, and returns
                                       pre-computable tensors, including a tensorized representation
                                       of the image file path.
            model_type (str, optional): The type of model to load ('internvl' or 'qwen'). Required if cache_hidden_state is True.
            checkpoint_path (str, optional): Path to the model checkpoint. Required if cache_hidden_state is True.
            device (str): The device to load the model onto.
            view_list (list, optional): Camera field names for multi-view input; defaults to ['cam_f0'].
            num_frames (int): Frames per view (oldest->newest from the tail of cameras); default 1.
            use_video_input (bool): Qwen3 only; True with num_frames>1 uses the videos= path.
            video_fps (float): fps used in VideoMetadata for the Qwen3 video path.
            use_navigation_command (bool): True 时 status_feature 命令段改用 navigation_command
                编码（含 park 进出站信息），替换旧 4 维 one-hot；旧数据自动退化。VLM prompt
                仍用 4 维 one-hot（high_command_one_hot），故 qwen3 缓存名不受影响，但缓存内的
                status_feature 维度会变——开启后须重跑该 CACHE_PATH 的缓存并与训练/eval 一致。
        """
        # cache_hidden_state：True：特征里已经带 hidden state（要么 builder 里算，要么从缓存读），agent 只负责用。
        # False：特征里不包含 hidden state，只给路径；hidden state 由 agent 里的 backbone 每次 forward 时现算。
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.model_type = (model_type or "internvl").lower()
        self.backbone = None
        self.cache_mode = cache_mode
        self.first_hidden_state_index = int(first_hidden_state_index)
        self.last_hidden_state_index = int(last_hidden_state_index)
        # 多视角 / 多帧配置（默认单视角单帧，零影响）
        self.view_list = list(view_list) if view_list else ["cam_f0"]
        self.num_frames = int(num_frames)
        self.use_video_input = bool(use_video_input)
        self.video_fps = float(video_fps)
        self.use_navigation_command = bool(use_navigation_command)

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError("In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided.")
            self.backbone = RecogDriveBackbone(
                model_type=self.model_type,
                checkpoint_path=checkpoint_path,
                device=device
            )

    def get_unique_name(self) -> str:
        # navigation_command 开启后，本 builder 写入的 status_feature 维度改变（命令段 4→NAV_COMMAND_DIM）。
        # status_feature 与 VLM hidden state 同存于该缓存文件，若沿用旧名，重跑缓存时
        # _cache_scene_with_token 的 exists() 会跳过旧文件 → 训练读到旧维度 status_feature 而与
        # planner 首层冲突（P5）。故开 nav 时缓存名加 _nav 后缀强制隔离（代价：VLM hidden state 也
        # 随之重算一次，正确性优先）。旧数据/默认关时保持历史名，零影响。
        nav_suffix = "_nav" if self.use_navigation_command else ""
        if self.model_type == "qwen3":
            n_v, n_f = len(self.view_list), int(self.num_frames)
            # 单视角单帧保持历史缓存名，旧 qwen3_feature.gz 继续可用（零影响）。
            if n_v == 1 and n_f == 1:
                return f"qwen3_feature{nav_suffix}"
            # 多视角 / 多帧的 hidden_state 序列长度不同，落到独立目录避免脏读 / 误用旧缓存训练。
            tag = f"v{n_v}_f{n_f}" + ("_vid" if (self.use_video_input and n_f > 1) else "")
            return f"qwen3_feature_{tag}{nav_suffix}"
        return f"internvl_feature{nav_suffix}"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        ego_features = build_recogdrive_ego_features(agent_input, self.use_navigation_command)
        cameras = agent_input.cameras
        history_trajectory = ego_features["history_trajectory"]
        high_command_one_hot = ego_features["high_command_one_hot"]

        if not self.cache_hidden_state:
            # 离线 / 在线 backbone 模式：只缓存单条前视路径。多视角 / 多帧的在线 eval
            # 不走这里（见 recogdrive_agent._encode_views 方案 B，从 agent_input 直接取图）。
            # NOTE: 若将来要做 offline batched 多视角训练，此处单路径编码不够，需扩展为
            #       视角×帧矩阵的多路径编码，并相应调整 collate。
            view_frame_paths, vlm_image_missing = collect_view_frame_paths_with_status(cameras, ["cam_f0"], 1)
            image_path = view_frame_paths[0][0]

            path_as_ordinals = [ord(char) for char in image_path]

            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)

            return {
                **ego_features,
                "image_path_tensor": path_tensor.cpu(),
                "vlm_image_missing": torch.tensor(vlm_image_missing, dtype=torch.bool).cpu(),
            }
        else:       # 用“当前图像 + 历史轨迹 + 导航指令”的 prompt 跑一遍 VLM，把得到的 last_hidden_state 作为视觉语言特征返回，供后面 Diffusion Planner 做轨迹预测。
            if self.backbone is None:
                raise RuntimeError("FeatureBuilder is in online mode, but the backbone was not initialized.")

            command_str = command_code_to_text(high_command_one_hot)
            history_str = build_history_str(history_trajectory)

            # 多视角 / 多帧取图（含 edge-pad），与在线 eval 共用 view_utils 保证一致。
            view_frame_paths, vlm_image_missing = collect_view_frame_paths_with_status(cameras, self.view_list, self.num_frames)
            flat = flatten_view_frames(view_frame_paths)
            question = build_driving_question(
                history_str, command_str, self.view_list, self.num_frames,
                model_type=self.model_type, flat_image_count=len(flat),
            )
            questions = [question]

            outputs = self.backbone.forward_from_paths(
                [view_frame_paths],            # batch=1
                questions,
                view_labels=view_labels_for(self.view_list),
                num_frames=self.num_frames,
                use_video_input=self.use_video_input,
                video_fps=self.video_fps,
            )
            first_hidden_state = select_hidden_state(outputs.hidden_states, self.first_hidden_state_index)
            last_hidden_state = select_hidden_state(outputs.hidden_states, self.last_hidden_state_index)

            return {
                **ego_features,
                "first_hidden_state": first_hidden_state.squeeze(0).float().cpu(),
                "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                "vlm_image_missing": torch.tensor(vlm_image_missing, dtype=torch.bool).cpu(),
            }


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
