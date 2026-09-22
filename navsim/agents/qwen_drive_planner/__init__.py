"""NAVSIM 接入官方 Qwen-Drive-1.0 Planning Expert 的适配层。

网络定义内置于 `modeling/`，权重使用官方发布文件；本包提供
数据适配（`qwen_drive_features`）、训练侧 flow matching（`qwen_drive_flow`）、
官方缺口补丁——batch attention mask、可配置视角（`qwen_drive_official`），以及把
三者接成 NAVSIM `AbstractAgent` 的 `qwen_drive_agent.QwenDriveAgent`。
"""

from .qwen_drive_config import (
    DEFAULT_PLANNER_PATH,
    DEFAULT_VLM_PATH,
    FULL_VIEW_LIST,
    MOGO_VIEW_TO_OFFICIAL_LABEL,
    OFFICIAL_VIEW_LIST,
    QwenDrivePlannerConfig,
)
from .qwen_drive_features import (
    QWEN_DRIVE_FEATURE_KEYS,
    QWEN_DRIVE_IMAGE_KEY,
    QwenDriveFeatureBuilder,
    QwenDriveTrajectoryTargetBuilder,
    build_qwen_drive_ego_features,
    collect_qwen_drive_view_frames,
    nav_command_from_driving_command,
)
from .qwen_drive_flow import (
    endpoint_loss,
    flow_matching_step,
    interpolate_noisy_trajectory,
    sample_flow_time,
    temporal_smoothness_loss,
)
from .qwen_drive_official import (
    MultiViewScene,
    make_multi_view_processor,
    masked_attend,
    masked_layer_forward,
    predict_endpoint_masked,
    sample_masked,
)
from .qwen_drive_agent import QwenDriveAgent

__all__ = [
    # config
    "QwenDrivePlannerConfig",
    "DEFAULT_VLM_PATH",
    "DEFAULT_PLANNER_PATH",
    "MOGO_VIEW_TO_OFFICIAL_LABEL",
    "OFFICIAL_VIEW_LIST",
    "FULL_VIEW_LIST",
    # features
    "QWEN_DRIVE_FEATURE_KEYS",
    "QWEN_DRIVE_IMAGE_KEY",
    "QwenDriveFeatureBuilder",
    "QwenDriveTrajectoryTargetBuilder",
    "build_qwen_drive_ego_features",
    "collect_qwen_drive_view_frames",
    "nav_command_from_driving_command",
    # flow matching
    "sample_flow_time",
    "interpolate_noisy_trajectory",
    "endpoint_loss",
    "temporal_smoothness_loss",
    "flow_matching_step",
    # official-gap patches
    "masked_attend",
    "masked_layer_forward",
    "predict_endpoint_masked",
    "sample_masked",
    "MultiViewScene",
    "make_multi_view_processor",
    # agent
    "QwenDriveAgent",
]
