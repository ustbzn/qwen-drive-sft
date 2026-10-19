"""NAVSIM `AbstractAgent` 包装官方 Qwen-Drive-1.0 Planning Expert。

职责边界（v1 严格对齐官方）：

- **网络与权重**：网络源码内置于 `modeling/`，权重来自官方。VLM = `Qwen-Drive-1.0-4B`，PE = `planner-sft`，
  经 `QwenDriveForPlanning.from_pretrained(vlm_path, planner=planner_path)` 加载，
  官方 `load_planner` 是 `strict=True`，因此几何参数一处不符就会立刻报错，
  不会静默跳层。本文件不复制任何官方网络定义。
- **数据适配**：`qwen_drive_features.py`（nav 3 类、ego 8 维、history 三分支）。
- **训练侧 flow matching**：`qwen_drive_flow.py`（官方只发布了 `sample`）。
- **官方缺口补丁**：`qwen_drive_official.py`（batch attention mask、可配置视角）。

架构要点：

1. **VLM 在线前向，不做离线 K/V 缓存**。实测每样本 K/V ≈ 140MB
   （8 层 x K/V x ~4480 token x 4 kv_head x 256 head_dim x 2B），53k 样本 ≈ 7.1TB，
   离线缓存不可行。因此 feature builder 只产出图像路径 + 小张量，VLM 在
   `no_grad` 下在线跑，梯度只进 PE。

2. **逐样本 VLM 前向**。不同样本的 prompt 长度不同（history 打印到小数点后 4 位，
   位数不等），若 pad 成一个 batch 再前向，pad token 会进入 mRoPE 位置计算并污染
   `anchor`。所以 VLM 逐样本跑、K/V 出来后再 pad + mask 给 PE。

3. **train/eval 的 prompt 完全同源**。两条路径都走
   `_build_scene` -> `_processor(scene)`，没有第二份 prompt 实现。

坐标与单位：全部沿用官方约定 —— 当前帧 ego 系，x 前 y 左，heading 左转为正，
history/future 均 10Hz，history 末行为当前帧 `(0,0,0)`。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

from .modeling import CameraFrame, QwenDriveForPlanning
from .modeling.trajectory import (
    denormalize_trajectory,
    normalize_history,
    normalize_trajectory,
    resample_10hz_to_2hz,
)
from .qwen_drive_config import MOGO_VIEW_TO_OFFICIAL_LABEL, QwenDrivePlannerConfig
from .qwen_drive_features import (
    QWEN_DRIVE_IMAGE_KEY,
    QwenDriveFeatureBuilder,
    QwenDriveTrajectoryTargetBuilder,
)
from .qwen_drive_flow import flow_matching_step
from .qwen_drive_official import (
    MultiViewScene,
    make_multi_view_processor,
    sample_masked,
)

logger = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


class QwenDriveAgent(AbstractAgent):
    """官方 Planning Expert + NAVSIM/Mogo 数据适配。

    v1 目标是复现官方数值，因此默认 `freeze_vlm=True`：VLM 只提供 K/V，梯度只进 PE。
    """

    def __init__(self, config: Optional[QwenDrivePlannerConfig] = None, **kwargs: Any):
        super().__init__(requires_scene=False)
        # hydra 用 `_target_` + 扁平 kwargs 实例化；也支持直接传 config 对象。
        self._config = config if config is not None else QwenDrivePlannerConfig(**kwargs)
        self._model: Optional[torch.nn.Module] = None
        self._processor: Optional[Any] = None
        self._initialized = False

    # ------------------------------------------------------------------ 基本信息

    def name(self) -> str:
        return self.__class__.__name__

    @property
    def config(self) -> QwenDrivePlannerConfig:
        return self._config

    def get_sensor_config(self) -> SensorConfig:
        """只加载 `view_list` 里的相机，且只加载需要的历史帧。

        `num_camera_frames` 帧在 `num_history_points` 个历史帧里按等间隔选取
        （与 `collect_qwen_drive_view_frames` 的下标算法保持一致），这里把同一组
        下标声明给 SensorConfig，避免多解码无用图像。
        """
        indices = self._camera_frame_indices()
        fields = {
            "cam_f0": False,
            "cam_l0": False,
            "cam_l1": False,
            "cam_l2": False,
            "cam_r0": False,
            "cam_r1": False,
            "cam_r2": False,
            "cam_b0": False,
            "lidar_pc": False,
        }
        for view in self._config.view_list:
            fields[view] = list(indices)
        return SensorConfig(**fields)

    def _camera_frame_indices(self) -> List[int]:
        """与 `collect_qwen_drive_view_frames` 相同的帧下标（0-based，oldest 起）。"""
        frames = max(1, int(self._config.num_camera_frames))
        available = int(self._config.num_history_points)
        if frames == 1:
            return [available - 1]
        step = (available - 1) / (frames - 1)
        return [int(round(i * step)) for i in range(frames)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [QwenDriveFeatureBuilder(self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [QwenDriveTrajectoryTargetBuilder(self._config)]

    # ------------------------------------------------------------------ 初始化

    def initialize(self) -> None:
        """加载官方 VLM + planner 权重。

        官方 `load_planner` 用 `strict=True`，几何参数不匹配会直接抛错，
        因此这里不需要额外校验 —— 加载成功即证明与官方权重完全一致。
        """
        if self._initialized:
            return

        dtype = _DTYPES.get(self._config.train_dtype.lower())
        if dtype is None:
            raise ValueError(
                f"未知 train_dtype={self._config.train_dtype!r}，可选 {sorted(_DTYPES)}"
            )

        logger.info("加载官方 Qwen-Drive VLM: %s", self._config.vlm_path)
        model = QwenDriveForPlanning.from_pretrained(
            self._config.vlm_path,
            planner=self._config.planner_path,
            dtype=dtype,
            local_files_only=True,
        )
        if self._config.planner_path:
            logger.info("已加载官方 planner 权重（strict=True）: %s", self._config.planner_path)
        else:
            logger.warning("planner_path 为空：Planning Expert 为随机初始化，不是官方权重")

        # VLM 只出 K/V，永远不需要梯度；freeze_planner 用于纯推理/调试。
        model.vlm.requires_grad_(not self._config.freeze_vlm)
        model.planning_expert.requires_grad_(not self._config.freeze_planner)
        for field in ("num_history_points", "num_future_points", "trajectory_point_dim"):
            if getattr(model.config, field) != getattr(self._config, field):
                raise ValueError(f"适配配置 {field} 与官方模型配置不一致")
        if self._config.checkpoint_path:
            checkpoint = torch.load(self._config.checkpoint_path, map_location="cpu", weights_only=False)
            prefix = "agent._model.planning_expert."
            weights = {
                key[len(prefix):]: value
                for key, value in checkpoint["state_dict"].items()
                if key.startswith(prefix)
            }
            model.planning_expert.load_state_dict(weights, strict=True)

        self._model = model
        self._processor = make_multi_view_processor(model.processor)
        # 官方 `processor` 是 property，setter 会把改造后的实例写回，
        # 保证 model 内部（如 generate_trajectory）也用同一个 processor。
        model.processor = self._processor
        self._initialized = True
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._model is not None:
            self._model.vlm.eval()
        return self

    def _require_model(self) -> torch.nn.Module:
        if self._model is None:
            self.initialize()
        assert self._model is not None
        return self._model

    # ------------------------------------------------------------------ prompt 构造

    def _build_scene(
        self,
        view_frame_paths: Sequence[Sequence[str]],
        history: torch.Tensor,
        history_velocity: torch.Tensor,
        history_acceleration: torch.Tensor,
        ego_status: torch.Tensor,
        nav_command: int,
        navigation_instruction: Optional[str] = None,
        view_list: Optional[Sequence[str]] = None,
    ) -> MultiViewScene:
        """把一个样本的特征拼成官方 processor 可消费的 scene（train/eval 同源）。

        `view_frame_paths` 是 `[view][frame]` 的路径矩阵，顺序与样本的
        `view_list` 一致；无样本视角时使用 agent 配置。
        官方 prompt 按"视角分组、组内按时间"排列，`frames_in_order` 会照此展开。
        """
        view_list = list(view_list if view_list is not None else self._config.view_list)
        if len(view_frame_paths) != len(view_list):
            raise ValueError(
                f"view_frame_paths 有 {len(view_frame_paths)} 个视角，"
                f"view_list 有 {len(view_list)} 个"
            )

        labels = [MOGO_VIEW_TO_OFFICIAL_LABEL[view] for view in view_list]
        views = {
            label: [
                CameraFrame(
                    image=Image.fromarray(path) if isinstance(path, np.ndarray) else path,
                    pre_resized=getattr(path, "pre_resized", None),
                )
                for path in per_view
            ]
            for label, per_view in zip(labels, view_frame_paths)
        }

        ego = ego_status.detach().float().cpu().numpy()
        return MultiViewScene(
            views=views,
            history=history.detach().float().cpu().numpy(),
            history_velocity=history_velocity.detach().float().cpu().numpy(),
            history_acceleration=history_acceleration.detach().float().cpu().numpy(),
            # 官方 ego_status = concat[velocity(2), acceleration(2), driving_command(4)]，
            # 这里按同一布局切回三段，scene.ego_status 会重新拼成同样的 8 维。
            ego_velocity=ego[0:2],
            ego_acceleration=ego[2:4],
            driving_command=ego[4:8],
            nav_command=int(nav_command),
            view_order=tuple(labels),
            navigation_instruction=navigation_instruction,
        )

    # ------------------------------------------------------------------ VLM 前向

    @torch.no_grad()
    def _encode_scenes(
        self, features: Dict[str, Any]
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """逐样本跑 VLM，返回 pad 后的 scene K/V、anchor、有效位 mask。

        **为什么逐样本**：不同样本 prompt 长度不同，pad 后再前向会让 pad token 参与
        `get_rope_index`，污染 `anchor`（waypoint 的 mRoPE 起点）。逐样本前向后再
        在序列维 pad、并给 PE 传 mask，可以复用官方不接收 padding mask 的 prefill。

        :return: (scene_cache, anchor, scene_valid)
            scene_cache: 每层 `(key, value)`，形状 `[B, L_max, kv_heads, head_dim]`
            anchor: `[3, B]`（官方 mRoPE 的 3 个 section）
            scene_valid: `[B, L_max]` bool，True=真实 token
        """
        model = self._require_model()
        device = next(model.parameters()).device

        view_frame_paths = features[QWEN_DRIVE_IMAGE_KEY]
        batch = len(view_frame_paths)
        nav_command = features["nav_command"].reshape(-1)
        navigation_instructions = features.get("navigation_instruction", [None] * batch)
        view_lists = features.get("view_lists", [None] * batch)

        per_sample_cache: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        anchors: List[torch.Tensor] = []
        lengths: List[int] = []

        for index in range(batch):
            scene = self._build_scene(
                view_frame_paths[index],
                features["history"][index],
                features["history_velocity"][index],
                features["history_acceleration"][index],
                features["ego_status"][index],
                int(nav_command[index].item()),
                navigation_instructions[index],
                view_lists[index],
            )
            inputs = self._processor(scene, with_reasoning=False, device=device)
            # 官方 `_prefill` 内部就是 VLM 前向 + `_scene_cache` + anchor，直接复用。
            cache, anchor = model._prefill(inputs)
            per_sample_cache.append(cache)
            anchors.append(anchor)
            lengths.append(cache[0][0].shape[1])

        max_len = max(lengths)
        num_layers = len(per_sample_cache[0])

        scene_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(num_layers):
            keys = [sample[layer][0] for sample in per_sample_cache]
            values = [sample[layer][1] for sample in per_sample_cache]
            scene_cache.append(
                (
                    _pad_and_cat_sequence(keys, max_len),
                    _pad_and_cat_sequence(values, max_len),
                )
            )

        scene_valid = torch.zeros(batch, max_len, dtype=torch.bool, device=device)
        for index, length in enumerate(lengths):
            scene_valid[index, :length] = True

        # 官方 anchor 形状 [3, 1]（batch=1），沿 batch 维拼成 [3, B]。
        anchor = torch.cat(anchors, dim=1)
        return scene_cache, anchor, scene_valid

    # ------------------------------------------------------------------ 训练

    def forward(
        self,
        features: Dict[str, Any],
        targets: Optional[Dict[str, torch.Tensor]] = None,
        tokens_list: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """训练前向：VLM(no_grad) -> PE flow matching。

        签名与本仓库 `AgentLightningModule._step` 的
        `agent.forward(features, targets, tokens_list)` 一致。loss 在这里就算完并
        放进返回 dict，`compute_loss` 只做取用，避免重复前向。
        """
        model = self._require_model()
        expert = model.planning_expert
        device = next(model.parameters()).device

        if targets is None or "trajectory" not in targets:
            raise ValueError("forward 需要 targets['trajectory'] 作为 flow matching 监督")

        features = _move_tensors(features, device)
        scene_cache, anchor, scene_valid = self._encode_scenes(features)

        scale = model.trajectory_scale(device)

        # history 必须先 normalize（re-ref 到最老点、丢首行 16->15），
        # 否则 history_encoder 的 48 维入口对不上（15*3 + nav_onehot 3）。
        history = normalize_history(features["history"].float(), scale)
        history_queries = expert.encode_history(
            history,
            features["nav_command"],
            features["history_velocity"].float(),
            features["history_acceleration"].float(),
        )

        clean = normalize_trajectory(
            targets["trajectory"].to(device).float(), scale
        )

        generator: Optional[torch.Generator] = None
        outputs = flow_matching_step(
            expert,
            clean,
            history_queries,
            scene_cache,
            anchor,
            features["nav_command"],
            features["ego_status"].float(),
            alpha=self._config.noise_beta_alpha,
            beta=self._config.noise_beta_beta,
            t_max=self._config.noise_t_max,
            lambda_delta1=self._config.lambda_delta1 if self._config.temporal_reg else 0.0,
            lambda_delta2=self._config.lambda_delta2 if self._config.temporal_reg else 0.0,
            generator=generator,
            scene_valid=scene_valid,
            gradient_checkpointing=self._config.gradient_checkpointing,
        )
        return {
            "loss": outputs["loss"],
            "diffusion_loss": outputs["diffusion_loss"],
            "temporal_loss": outputs["temporal_loss"],
            "trajectory": denormalize_trajectory(
                outputs["predicted"].detach(), scale
            ),
        }

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """返回 `forward` 已算好的 loss（不重复前向）。"""
        if "loss" not in predictions:
            raise ValueError("predictions 缺少 'loss'；compute_loss 必须在 forward 之后调用")
        return predictions["loss"]

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """只优化 requires_grad 的参数（默认即 PE）。"""
        model = self._require_model()
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                "没有可训练参数：freeze_planner=True 且 freeze_vlm=True 时无法训练"
            )
        return torch.optim.AdamW(
            params,
            lr=self._config.lr,
            weight_decay=self._config.weight_decay,
            betas=(0.9, 0.95),
        )

    # ------------------------------------------------------------------ 推理

    @torch.no_grad()
    def predict_trajectory_10hz(self, features: Dict[str, Any]) -> torch.Tensor:
        """采样 50 点 @10Hz 轨迹（米/弧度），返回 `[B, 50, 3]`。

        Euler 积分逻辑、`min_one_minus_t`、noise 尺度全部取官方配置，
        与官方 `_plan_from_cache` 一致。
        """
        model = self._require_model()
        expert = model.planning_expert
        device = next(model.parameters()).device

        features = _move_tensors(features, device)
        scene_cache, anchor, scene_valid = self._encode_scenes(features)
        scale = model.trajectory_scale(device)

        batch = features["history"].shape[0]
        num_samples = max(1, int(self._config.num_samples))
        num_points = model.config.num_future_points

        history = normalize_history(features["history"].float(), scale)

        def tile(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.repeat_interleave(num_samples, dim=0)

        # 官方按 sample 索引给独立种子（sample k 恒用 seed+k），保证可复现。
        noise = model.config.noise_init_std * model._initial_noise(
            num_samples, num_points, self._config.noise_seed, device
        )
        if batch > 1:
            noise = noise.repeat(batch, 1, 1)

        normalized = sample_masked(
            expert,
            scene_cache=[(tile(k), tile(v)) for k, v in scene_cache],
            # anchor 是 [3, B]；每个样本复制 num_samples 份，保持与 tile 后的顺序对齐。
            position_anchor=anchor.repeat_interleave(num_samples, dim=1),
            history=tile(history),
            history_velocity=tile(features["history_velocity"].float()),
            history_acceleration=tile(features["history_acceleration"].float()),
            nav_command=tile(features["nav_command"]),
            ego_status=tile(features["ego_status"].float()),
            noise=noise,
            num_steps=self._config.num_inference_steps,
            min_one_minus_t=model.config.min_one_minus_t,
            scene_valid=tile(scene_valid),
        )
        trajectory = denormalize_trajectory(normalized, scale)
        if num_samples > 1:
            # 多候选时取第一条（seed 与官方单样本一致），其余留给后续打分策略。
            trajectory = trajectory.reshape(batch, num_samples, num_points, -1)[:, 0]
        return trajectory

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """NAVSIM 评测入口：50 点 @10Hz -> 8 点 @2Hz。

        NAVSIM `Trajectory` 默认 `TrajectorySampling(time_horizon=4, interval_length=0.5)`
        并在 `__post_init__` 断言 8 个 pose，所以必须重采样。用官方
        `resample_10hz_to_2hz`（取下标 [4,9,...,39]，正是 NAVSIM 插值网格的逆），
        而不是自己截断/插值。
        """
        self.eval()
        features: Dict[str, Any] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # 加 batch 维：tensor 用 unsqueeze；图像路径与导航文本按样本作为 list 传递。
        batched: Dict[str, Any] = {}
        for key, value in features.items():
            batched[key] = (
                [value]
                if key in (QWEN_DRIVE_IMAGE_KEY, "navigation_instruction")
                else value.unsqueeze(0)
            )

        trajectory = self.predict_trajectory_10hz(batched)[0].float().cpu().numpy()
        poses = resample_10hz_to_2hz(
            trajectory, horizon_s=self._config.eval_horizon_s
        )
        poses = np.asarray(poses, dtype=np.float32)
        if poses.shape[0] != self._config.eval_num_poses:
            raise ValueError(
                f"重采样得到 {poses.shape[0]} 个 pose，NAVSIM Trajectory 需要 "
                f"{self._config.eval_num_poses} 个（eval_horizon_s="
                f"{self._config.eval_horizon_s}）"
            )
        return Trajectory(poses)


def _pad_and_cat_sequence(tensors: Sequence[torch.Tensor], max_len: int) -> torch.Tensor:
    """把 `[1, L, kv_heads, head_dim]` 的张量在序列维 pad 到 max_len 后拼 batch。

    pad 位置的值不重要（PE 侧由 attention mask 屏蔽），这里填 0。
    """
    padded = []
    for tensor in tensors:
        gap = max_len - tensor.shape[1]
        if gap > 0:
            # F.pad 的 pad 元组从最后一维往前数；序列维是 dim=1（共 4 维），
            # 因此需要跳过最后两维（head_dim、kv_heads）各一对 (left, right)。
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, gap))
        padded.append(tensor)
    return torch.cat(padded, dim=0)


def _move_tensors(features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """把 tensor 搬到 device，非 tensor（图像路径矩阵）原样保留。"""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in features.items()
    }
