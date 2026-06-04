from typing import Dict, Optional
import torch
import numpy as np
import gzip
import pickle
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from navsim.common.command_utils import command_code_to_text
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .recogdrive_backbone import RecogDriveBackbone
from .utils.internvl_preprocess import load_image

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


EGO_STATUS_FEATURE_NAMES = (
    "cmd_0",
    "cmd_1",
    "cmd_2",
    "cmd_3",
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


def select_hidden_state(hidden_states, layer_index: int) -> torch.Tensor:
    """Select a hidden-state layer by index while preserving Python negative-index semantics."""
    return hidden_states[int(layer_index)]


class ReCogDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False,
                 first_hidden_state_index: int = 0,
                 last_hidden_state_index: int = -1, ):
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

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError("In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided.")
            self.backbone = RecogDriveBackbone(
                model_type=self.model_type,
                checkpoint_path=checkpoint_path,
                device=device
            )

    def get_unique_name(self) -> str:
        if self.model_type == "qwen3":
            return "qwen3_feature"
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        current_ego_status = ego_statuses[-1]
        high_command_one_hot = torch.tensor(current_ego_status.driving_command, dtype=torch.float32)  # ego_statuses[-1]表示当前时刻
        status_feature = torch.cat([
            high_command_one_hot.clone(),
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
        ], dim=-1)
        assert status_feature.shape[-1] == len(EGO_STATUS_FEATURE_NAMES)


        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)
            
            path_as_ordinals = [ord(char) for char in image_path]
            
            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)
            
            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "status_feature": status_feature.cpu(),
                "image_path_tensor": path_tensor.cpu(),
            }
        else:       # 用“当前图像 + 历史轨迹 + 导航指令”的 prompt 跑一遍 VLM，把得到的 last_hidden_state 作为视觉语言特征返回，供后面 Diffusion Planner 做轨迹预测。
            if self.backbone is None:
                raise RuntimeError("FeatureBuilder is in online mode, but the backbone was not initialized.")
            
            navigation_commands = ['turn left', 'go straight', 'turn right']
            command_str = next((navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1), "unknown")
            history_str = " ".join([f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})' for i in range(4)])
            
            prompt = f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
            output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
            questions = [f"{prompt}{output_requirements}"]

            image_path = str(cameras[-1].cam_f0.image)
            outputs = self.backbone.forward_from_paths([image_path], questions)
            first_hidden_state = select_hidden_state(outputs.hidden_states, self.first_hidden_state_index)
            last_hidden_state = select_hidden_state(outputs.hidden_states, self.last_hidden_state_index)

            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "first_hidden_state": first_hidden_state.squeeze(0).float().cpu(),
                "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                "status_feature": status_feature.cpu(),
            }


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
