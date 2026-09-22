"""Qwen-Drive-1.0 Planning Expert 接入 NAVSIM/RecogDrive 的适配层配置。

刻意用**普通** dataclass，不继承 `PretrainedConfig`：两者混用时 dataclass 生成的
`__post_init__` 会调用不存在的 `super().__post_init__()`，实例化即抛 AttributeError。

本类只描述"适配层"的选择（用哪些相机、权重在哪、训练超参）。
Planning Expert 自身的几何（hidden/head_dim/layers/nav_classes/ego_dim ...）一律由官方
`planner-*/config.json` 决定，**不在这里重复声明**，避免与官方权重错位。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# 官方发布权重：VLM 在根目录，Planning Expert 在 planner-sft / planner-rl 子目录。
DEFAULT_VLM_PATH = "/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B"
DEFAULT_PLANNER_PATH = "/D2/NXT/vlm/miaobinqi/models/Qwen-Drive-1.0-4B/planner-sft"

# Mogo 相机字段 -> 官方 prompt 视角标签。
# 官方 CAMERA_VIEWS 就是前三项且顺序固定，默认 view_list 与之逐字对齐。
# Mogo 当前数据的 cam_l2 / cam_r2 为空，接口保留为左右侧视角。
MOGO_VIEW_TO_OFFICIAL_LABEL = {
    "cam_f0": "<FRONT VIEW>",
    "cam_l0": "<FRONT LEFT VIEW>",
    "cam_r0": "<FRONT RIGHT VIEW>",
    "cam_l2": "<LEFT VIEW>",
    "cam_r2": "<RIGHT VIEW>",
    "cam_l1": "<BACK LEFT VIEW>",
    "cam_r1": "<BACK RIGHT VIEW>",
    "cam_b0": "<BACK VIEW>",
}

# 官方 3 视角顺序，作为默认值（复现官方数值的前提）。
OFFICIAL_VIEW_LIST: Tuple[str, ...] = ("cam_f0", "cam_l0", "cam_r0")
# 八路接口；使用时要求所选相机实际有图像。
FULL_VIEW_LIST: Tuple[str, ...] = (
    "cam_f0",
    "cam_l0",
    "cam_r0",
    "cam_l1",
    "cam_r1",
    "cam_b0",
    "cam_l2",
    "cam_r2",
)


@dataclass
class QwenDrivePlannerConfig:
    """适配层配置。"""

    # --- 权重路径 ---
    vlm_path: str = DEFAULT_VLM_PATH
    planner_path: Optional[str] = DEFAULT_PLANNER_PATH
    checkpoint_path: Optional[str] = None

    # --- 视角 / 时序 ---
    # 默认严格对齐官方 3 视角；FULL_VIEW_LIST 保留八路接口。
    view_list: List[str] = field(default_factory=lambda: list(OFFICIAL_VIEW_LIST))
    # 每个视角的相机帧数。官方训练用 4 帧（t-1.5 / t-1.0 / t-0.5 / t-0s）。
    num_camera_frames: int = 4

    # --- 轨迹几何（必须与官方 config.json 一致，此处仅作断言用）---
    num_history_points: int = 16
    num_future_points: int = 50
    trajectory_point_dim: int = 3

    # --- 训练（优化器超参是本仓库默认值，论文未公开）---
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    # 官方 Fourier 频表和 waypoint mRoPE 的舍入依赖 bf16。
    train_dtype: str = "bfloat16"
    # VLM 始终 frozen + no_grad；这里只是显式记录意图。
    freeze_vlm: bool = True
    freeze_planner: bool = False
    gradient_checkpointing: bool = True

    # --- Flow matching（论文第 5 页公式 7–8）---
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    # t_max 与官方推理端 min_one_minus_t=0.1 呼应，保证 1-t >= 0.1。
    noise_t_max: float = 0.9

    # --- PDF 公式（8），分析文档中的系数大了十倍 ---
    temporal_reg: bool = True
    lambda_delta1: float = 2.0e-4
    lambda_delta2: float = 2.0e-5

    # --- 推理 ---
    num_inference_steps: int = 10
    num_samples: int = 1
    noise_seed: int = 42
    # 官方 PDM/NAVSIM 评测栅格：50@10Hz -> 8@2Hz。
    eval_num_poses: int = 8
    eval_horizon_s: float = 4.0

    def __post_init__(self) -> None:
        if not self.view_list:
            self.view_list = list(OFFICIAL_VIEW_LIST)
        unknown = [v for v in self.view_list if v not in MOGO_VIEW_TO_OFFICIAL_LABEL]
        if unknown:
            raise ValueError(
                f"view_list 含未知相机字段 {unknown}；"
                f"可用: {sorted(MOGO_VIEW_TO_OFFICIAL_LABEL)}"
            )
        if len(set(self.view_list)) != len(self.view_list):
            raise ValueError("view_list 不能包含重复相机")
        if not self.freeze_vlm:
            raise ValueError("在线 IL 路径仅支持 freeze_vlm=True，官方 prefill 不构建梯度")
        if not 0.0 < self.noise_t_max < 1.0:
            raise ValueError("noise_t_max 必须在 (0, 1) 内")
        if self.noise_beta_alpha <= 0 or self.noise_beta_beta <= 0:
            raise ValueError("Beta 分布参数必须大于 0")
        if self.num_history_points != 16 or self.num_future_points != 50 or self.trajectory_point_dim != 3:
            raise ValueError("官方权重要求 16 个历史点、50 个未来点和 3 维轨迹")
        if self.num_camera_frames != 4:
            raise ValueError("当前官方 prompt 适配要求每路 4 帧，间隔 0.5s")
        if self.num_inference_steps < 1 or self.num_samples < 1:
            raise ValueError("num_inference_steps 和 num_samples 必须大于 0")
        if self.eval_num_poses != 8 or self.eval_horizon_s != 4.0:
            raise ValueError("当前 NAVSIM 评测接口要求 4s 内 8 个 pose")
        if self.num_camera_frames < 1:
            raise ValueError("num_camera_frames 必须 >= 1")
        if self.num_camera_frames > self.num_history_points:
            raise ValueError(
                f"num_camera_frames({self.num_camera_frames}) 不能超过 "
                f"num_history_points({self.num_history_points})"
            )

    @property
    def view_labels(self) -> List[str]:
        """prompt 中的视角标签，顺序与 view_list 一致。"""
        return [MOGO_VIEW_TO_OFFICIAL_LABEL[v] for v in self.view_list]

    @property
    def is_official_view_layout(self) -> bool:
        """是否与官方 3 视角布局逐字一致（决定能否期待复现官方数值）。"""
        return tuple(self.view_list) == OFFICIAL_VIEW_LIST and self.num_camera_frames == 4
