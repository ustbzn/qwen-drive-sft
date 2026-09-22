"""NAVSIM/Mogo -> Qwen-Drive-1.0 官方 Planning Expert 的数据适配层。

本文件是「映射关系」的唯一实现处，三条映射都以官方权重/代码为准（不是猜测）：

1. nav_command：Mogo 4 维 driving_command code -> 官方 3 类整数
   官方 `NAV_COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")`，进
   `_one_hot(index, 3)`。官方 `_one_hot` 对越界索引返回**全零行**（已实测），
   所以 u-turn / unknown 映射到 -1 是官方显式支持的 "no command"，不必硬塞成直行。
   这里复用 `command_code_to_text`，让 Mogo 的 4 维 code 语义（含 u-turn）不被绕过。

2. ego_status：-> 官方 8 维
   官方 `DrivingScene.ego_status` = concat[ego_velocity(2), ego_acceleration(2),
   driving_command(4)]，与权重 `ego_mlp.0.weight (1024, 8)` 吻合。
   Mogo `ego_dynamic_state` 实测 shape (4,)：[:2] 是 velocity、[2:] 是 acceleration。
   注意这**不是**把 RecogDrive 的 30 维 status_feature 降维——curvature / steering 等
   9 维运动学在官方 PE 里没有入口，直接不用。driving_command 原样传 4 维（官方就是
   4 维），Mogo 的 u-turn 编码 [1,1,0,0] 因此仍能进入 PE。

3. history：直接读取 16 点 @10Hz，三条独立分支
   - pose：16 点，官方 `normalize_history` 会 re-ref 到最老点并丢弃首行 ->
     15*3=45，拼 nav one-hot(3) = 48，对应 `history_encoder.0.weight (1024, 48)`
   - velocity：16*2 = 32，对应 `(1024, 32)`，**raw 不归一化**
   - acceleration：16*2 = 32，对应 `(1024, 32)`，**raw 不归一化**
   navsim 的 `local_ego_poses` 相对最后一帧，官方 history 也是 "last pose is (0,0,0)"，
   同约定，无需翻转。

K/V 不在这里缓存：实测每样本 K/V ≈ 140MB（8 层 x 2 x ~4480 tok x 4 head x 256 dim
x 2B），53k 样本 ≈ 7.1TB。因此 feature builder 只产出**图像路径 + 小张量**，VLM 前向
在训练/推理时在线跑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from navsim.common.command_utils import command_code_to_text
from navsim.common.dataclasses import AgentInput, Scene
from navsim.common.scene_timestamps import timestamp_grid_error
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from .qwen_drive_config import MOGO_VIEW_TO_OFFICIAL_LABEL, QwenDrivePlannerConfig

# 官方 NAV_COMMANDS 的索引。越界（-1）在官方 _one_hot 里得到全零行。
NAV_COMMAND_GO_STRAIGHT = 0
NAV_COMMAND_TURN_LEFT = 1
NAV_COMMAND_TURN_RIGHT = 2
NAV_COMMAND_NONE = -1

# command_code_to_text 的输出 -> 官方 3 类索引。
# u-turn / unknown 落到 -1，官方 _one_hot 返回全零行（已实测）。
_TEXT_TO_NAV_INDEX = {
    "go straight": NAV_COMMAND_GO_STRAIGHT,
    "turn left": NAV_COMMAND_TURN_LEFT,
    "turn right": NAV_COMMAND_TURN_RIGHT,
    "make a u-turn": NAV_COMMAND_NONE,
    "unknown": NAV_COMMAND_NONE,
}

QWEN_DRIVE_FEATURE_KEYS = (
    "history",
    "history_velocity",
    "history_acceleration",
    "ego_status",
    "nav_command",
)
# 图像路径是 List[str]，不是 tensor；collate 需按 list 处理。
QWEN_DRIVE_IMAGE_KEY = "view_frame_paths"


def nav_command_from_driving_command(driving_command: Any) -> int:
    """Mogo/NAVSIM 4 维 command code -> 官方 3 类 nav_command 整数。

    :return: 0=GO STRAIGHT, 1=TURN LEFT, 2=TURN RIGHT, -1=无命令（官方 _one_hot 全零行）
    """
    text = command_code_to_text(driving_command)
    return _TEXT_TO_NAV_INDEX.get(text, NAV_COMMAND_NONE)


def _edge_pad_front(array: np.ndarray, count: int) -> np.ndarray:
    """把序列首部按最早一行复制补齐到 count 行，并只保留最后 count 行。

    与官方 `benchmarks._history` 的补齐方式一致：历史不足时重复最老的一帧。
    """
    if array.shape[0] < count:
        padding = np.repeat(array[:1], count - array.shape[0], axis=0)
        array = np.concatenate([padding, array], axis=0)
    return array[-count:]


def build_qwen_drive_ego_features(
    agent_input: AgentInput,
    num_history_points: int = 16,
) -> Dict[str, torch.Tensor]:
    """构造官方 PE 需要的自车张量（history 三分支 + ego_status + nav_command）。

    所有量都在当前帧 ego 坐标系下，oldest -> newest，最后一行是当前帧。
    """
    ego_statuses = agent_input.ego_statuses
    if len(ego_statuses) != num_history_points:
        raise ValueError(
            f"需要 {num_history_points} 个连续 10Hz 历史帧，实际 {len(ego_statuses)}；"
            "请使用 mogo_qwen_drive scene filter，不能复制 2Hz 历史冒充 10Hz"
        )

    poses = np.asarray(
        [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [np.asarray(e.ego_velocity, dtype=np.float32)[:2] for e in ego_statuses],
        dtype=np.float32,
    )
    accelerations = np.asarray(
        [np.asarray(e.ego_acceleration, dtype=np.float32)[:2] for e in ego_statuses],
        dtype=np.float32,
    )

    history = _edge_pad_front(poses, num_history_points)
    history_velocity = _edge_pad_front(velocities, num_history_points)
    history_acceleration = _edge_pad_front(accelerations, num_history_points)

    current = ego_statuses[-1]
    driving_command = np.asarray(current.driving_command, dtype=np.float32).reshape(-1)
    # 官方 driving_command 是 4 维；Mogo 原始命令编码保持不变。
    if driving_command.shape[0] < 4:
        driving_command = np.pad(driving_command, (0, 4 - driving_command.shape[0]))
    driving_command = driving_command[:4]

    # 官方 8 维：velocity(2) + acceleration(2) + driving_command(4)
    ego_status = np.concatenate(
        [
            np.asarray(current.ego_velocity, dtype=np.float32)[:2],
            np.asarray(current.ego_acceleration, dtype=np.float32)[:2],
            driving_command,
        ]
    )
    assert ego_status.shape[0] == 8, f"ego_status 应为 8 维，实际 {ego_status.shape[0]}"

    nav_command = nav_command_from_driving_command(current.driving_command)

    return {
        "history": torch.from_numpy(history).float(),
        "history_velocity": torch.from_numpy(history_velocity).float(),
        "history_acceleration": torch.from_numpy(history_acceleration).float(),
        "ego_status": torch.from_numpy(ego_status).float(),
        "nav_command": torch.tensor(nav_command, dtype=torch.long),
    }


def _camera_image_path(camera_obj: Any) -> Optional[Any]:
    """接受训练用路径及 NAVSIM 评测默认加载的 RGB 数组。"""
    if camera_obj is None:
        return None
    image = getattr(camera_obj, "image", None)
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"相机数组必须是 HWC RGB，实际 {image.shape}")
        return image
    if isinstance(image, (str, Path)):
        return str(image) or None
    raise TypeError(f"不支持的相机图像类型：{type(image).__name__}")


def _fill_missing(paths: List[Optional[Any]]) -> List[Optional[Any]]:
    """帧级 edge-pad：先用前一个有效帧补，再用后一个有效帧补。"""
    filled = list(paths)
    last: Optional[str] = None
    for i, value in enumerate(filled):
        if value is not None:
            last = value
        elif last is not None:
            filled[i] = last
    nxt: Optional[str] = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is not None:
            nxt = filled[i]
        elif nxt is not None:
            filled[i] = nxt
    return filled


def collect_qwen_drive_view_frames(
    agent_input: AgentInput,
    view_list: Sequence[str],
    num_camera_frames: int,
) -> List[List[Any]]:
    """取 [view][frame] 的图像路径矩阵，oldest -> newest，每个视角末位为当前帧。

    官方 prompt 按 "view 分组、组内按时间" 排列，这里的返回结构与之一致。
    单个视角的短暂缺帧在同视角内 edge-pad；整路缺失报错，避免把前视图标成侧视图。
    """
    cameras = agent_input.cameras or []
    if not cameras:
        raise ValueError("agent_input.cameras 为空；无法构造 Qwen-Drive 图像输入")

    frames = max(1, int(num_camera_frames))
    available = len(cameras)
    take = min(frames, available)
    # 官方 4 帧覆盖 t-1.5s..t-0s。history 是 10Hz 16 点，相机帧要按 0.5s 间隔取，
    # 即在 16 点里取 index 0/5/10/15，而不是取最后 4 帧（那只有 0.3s 跨度）。
    if available >= frames and frames > 1:
        step = (available - 1) / (frames - 1)
        indices = [int(round(i * step)) for i in range(frames)]
    else:
        indices = list(range(available - take, available))
    selected = [cameras[i] for i in indices]

    raw: List[List[Optional[Any]]] = []
    for view in view_list:
        per_view = [_camera_image_path(getattr(cams, view, None)) for cams in selected]
        raw.append(_fill_missing(per_view))

    missing = [
        view for view, per_view in zip(view_list, raw)
        if not per_view or any(image is None for image in per_view)
    ]
    if missing:
        raise ValueError(f"所选相机整路缺失：{missing}；请修复数据或调整 view_list")
    return raw


class QwenDriveFeatureBuilder(AbstractFeatureBuilder):
    """产出官方 PE 所需的小张量 + 图像路径。

    刻意**不**缓存 K/V：实测每样本 140MB，53k 样本 7.1TB，离线缓存不可行。
    VLM 前向在 agent 里在线跑。
    """

    def __init__(self, config: QwenDrivePlannerConfig):
        super().__init__()
        self._config = config

    def get_unique_name(self) -> str:
        views = "-".join(self._config.view_list)
        return (
            f"qwen_drive_feature_v2_h{self._config.num_history_points}"
            f"_f{self._config.num_camera_frames}_{views}"
        )

    def compute_features(self, agent_input: AgentInput) -> Dict[str, Any]:
        features: Dict[str, Any] = build_qwen_drive_ego_features(
            agent_input, self._config.num_history_points
        )
        if len(agent_input.cameras) != self._config.num_history_points:
            raise ValueError("相机历史长度必须与 16 个 10Hz 自车历史帧一致")
        features[QWEN_DRIVE_IMAGE_KEY] = collect_qwen_drive_view_frames(
            agent_input, self._config.view_list, self._config.num_camera_frames
        )
        return features


class QwenDriveTrajectoryTargetBuilder(AbstractTargetBuilder):
    """未来 50 点 @10Hz 的监督信号。

    scene filter 必须是 mogo_qwen_drive（num_future_frames=50, frame_stride=1）；
    否则 `scene.get_future_trajectory` 拿不到 50 点，这里显式报错而不是静默截断。
    原始点按名义 10Hz 使用，不插值；允许相对当前帧网格不超过 75ms 的抖动，
    拒绝非递增时间戳、2Hz 数据和持续累积的时间漂移。
    """

    def __init__(self, config: QwenDrivePlannerConfig):
        super().__init__()
        self._config = config

    def get_unique_name(self) -> str:
        return f"qwen_drive_trajectory_target_v3_{self._config.num_future_points}_10hz"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        num_points = self._config.num_future_points
        if scene.scene_metadata.num_history_frames != self._config.num_history_points:
            raise ValueError("scene 必须包含 16 个历史帧")
        error = timestamp_grid_error(
            [frame.timestamp for frame in scene.frames], self._config.num_history_points - 1,
        )
        if error:
            metadata = scene.scene_metadata
            raise ValueError(
                f"{error}；log_name={getattr(metadata, 'log_name', 'unknown')}，"
                f"token={getattr(metadata, 'initial_token', 'unknown')}；"
                "请启用 mogo_qwen_drive 的时间戳过滤或修复源数据"
            )
        available = scene.scene_metadata.num_future_frames
        if available < num_points:
            raise ValueError(
                f"scene 只提供 {available} 个未来帧，官方 Planning Expert 需要 {num_points} 个。"
                " 请把 train_test_split/scene_filter 换成 mogo_qwen_drive"
                " (num_future_frames=50, frame_stride=1)。"
            )
        future = scene.get_future_trajectory(num_trajectory_frames=num_points)
        poses = np.asarray(future.poses, dtype=np.float32)
        assert poses.shape == (num_points, 3), f"未来轨迹形状 {poses.shape}"
        return {"trajectory": torch.from_numpy(poses)}
