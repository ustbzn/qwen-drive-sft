# Copyright 2025 The Xiaomi Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
import lzma
import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_ENCODER_ARCH = "resnet101"
DEFAULT_IMAGE_ENCODER_PRETRAINED_PATH = (
    "/pcpt/pcpt/project/jiangjunyuan/code/VLA/resnet101-cd907fc2.pth"
)

import numpy as np
import torch
from timm.models.layers import Mlp
from torch import nn
from torch.distributions import Beta, Normal, kl_divergence
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from navsim.common.dataclasses import Trajectory
from navsim.planning.scenario_builder.mogo_vehicle_parameters import get_mogo_vehicle_parameters
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from .blocks.encoder import (
    ActionEncoder,
    SinusoidalPositionalEncoding,
    StateAttentionEncoder,
    SwiGLUFFN,
)
from .recogdrive_dit import LightningDiT
from .hdmap_vector_features import HDMAP_VECTOR_BATCH_KEYS
from .hdmap_vector_encoder import MapEncoder
from .safety_loss import (
    collision_clearance_loss,
    drivable_sdf_loss,
    route_alignment_loss,
)
from .safety_targets import SAFETY_TARGET_KEYS


class ObjectEncoder(nn.Module):
    """把结构化障碍物特征编成 context token，与 VLM context token 拼接进 DiT cross-attention。

    continuous_obj_features -> LayerNorm -> MLP -> D
    obj_type -> Embedding -> D
    两者相加 -> LayerNorm -> 按 valid_mask 置零（padding / dropout 的 token 严格为 0，
    与 VLM hidden state 的 zero-padding 行为一致，cross-attention 无需额外 mask）。

    token_dropout：训练期随机丢弃有效障碍物，让模型在对象缺失 / 误检时能退化到 VLM+ego。
    注意：GRPO 时必须把本模块切到 eval（见 forward_grpo），否则 rollout 与 logprob 抽到
    不同的 dropout mask，会破坏 logprob 一致性。
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        num_types: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.type_embedding = nn.Embedding(num_types, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.token_dropout = float(dropout)

    def forward(
        self,
        obj_features: torch.Tensor,
        obj_type: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """:return: [B, N_obj, embed_dim]，padding / dropped 行为全 0。"""
        tokens = self.feature_mlp(self.feature_norm(obj_features)) + self.type_embedding(obj_type.long())
        tokens = self.output_norm(tokens)

        mask = obj_valid_mask.unsqueeze(-1).to(tokens.dtype)
        if self.training and self.token_dropout > 0.0:
            keep = (torch.rand_like(obj_valid_mask.float()) >= self.token_dropout)
            mask = mask * keep.unsqueeze(-1).to(tokens.dtype)
        return tokens * mask


class StationEncoder(nn.Module):
    """把站点（进出站目标）特征编成 context token，与 VLM/obj/hdmap token 并列进 DiT cross-attention。

    结构对标 ObjectEncoder（LayerNorm + MLP），但通常只有 1 个 station token。站点特征是
    「当前活动站点投影到 ego 系」的空间目标点 + dist/侧向（见 station_features.py），为规划器
    提供显式 goal，让输出轨迹在站点处呈现进/出站形状（goal-conditioned planning）。

    padding / 无 park 的 slot 由 station_valid_mask=False 置零（与 ObjectEncoder 一致，
    cross-attention 无需额外 mask 即忽略）。第一版不加 token_dropout：站点 goal 是稀疏的强信号，
    随机丢弃会削弱进出站监督；若后续需正则可再加。
    """

    def __init__(self, feature_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        station_features: torch.Tensor,
        station_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """:return: [B, N_station, embed_dim]，padding 行为全 0。"""
        tokens = self.feature_mlp(self.feature_norm(station_features))
        tokens = self.output_norm(tokens)
        mask = station_valid_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens * mask


class ImageEncoder(nn.Module):
    """单/多视角图像共享的 ResNet 编码器（实验用，use_image_encoder）。

    把当前帧相机图像编成空间 context token，与 VLM / obj / hdmap token 并列进 DiT
    cross-attention。用于在 vlm_ablation=zero（屏蔽 VLM 语义 token）时仍向规划器注入
    图像视觉信息。参考 DiffusionDriveV2 的 ResNet->flatten->pos_embed pattern。

    uint8 (B,V,3,H,W) -> flatten B*V -> /255 -> ImageNet 标准化
      -> timm ResNet features_only 末层特征图
      -> 1x1 conv 投到 embed_dim -> AdaptiveAvgPool2d(grid) 固定 token 数
      -> flatten HxW -> + 空间/视角可学习位置嵌入 -> LayerNorm -> (B,V*N,D)。

    V=1 时也接受历史 (B,3,H,W) / (B,) 输入，且不注册 view_embed 参数，保持旧模型
    state_dict 和旧 front_image_feature_v1_*.gz 的兼容性。

    token 数 = grid_h*grid_w，与输入分辨率解耦（adaptive pool），故位置嵌入尺寸固定。
    有效位（front_image_valid=False，缺图占位）对应样本的全部 token 被 mask 置零并在
    cross-attention 里忽略，与 ObjectEncoder 的 valid_mask 行为一致。

    dtype：训练走 16-mixed autocast（模块参数 fp32，BN 由 autocast 保持 fp32），无需手动
    半精度；输出在 build_context_tokens 里 .to(vl_context_tokens.dtype) 后再拼接。
    GRPO 时必须切 eval（见 forward_grpo）关掉 token_dropout，保证 rollout/logprob 一致。
    """

    # ImageNet 归一化常量（timm 预训练权重按此统计量训练）
    _PIXEL_MEAN = (0.485, 0.456, 0.406)
    _PIXEL_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        embed_dim: int,
        arch: str = DEFAULT_IMAGE_ENCODER_ARCH,
        pretrained: bool = False,
        pretrained_path: Optional[str] = DEFAULT_IMAGE_ENCODER_PRETRAINED_PATH,
        grid_h: int = 8,
        grid_w: int = 16,
        dropout: float = 0.0,
        num_views: int = 1,
    ):
        super().__init__()
        import timm

        self.num_views = int(num_views)
        if self.num_views <= 0:
            raise ValueError("ImageEncoder num_views must be positive")

        if pretrained_path:
            # 本地权重优先：创建无预训练主干，避免 timm 触发网络下载。
            self.backbone = timm.create_model(arch, pretrained=False, features_only=True)
            self._load_local_pretrained_weights(Path(pretrained_path).expanduser())
        else:
            # 未配置本地文件时保留历史 timm 行为。
            try:
                self.backbone = timm.create_model(
                    arch, pretrained=pretrained, features_only=True
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "ImageEncoder: timm.create_model(%s, pretrained=%s) failed (%s); "
                    "falling back to pretrained=False (random init).",
                    arch, pretrained, e,
                )
                self.backbone = timm.create_model(arch, pretrained=False, features_only=True)

        feat_ch = int(self.backbone.feature_info.channels()[-1])
        self.proj = nn.Conv2d(feat_ch, embed_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((int(grid_h), int(grid_w)))
        self.num_tokens = int(grid_h) * int(grid_w)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.num_views > 1:
            self.view_embed = nn.Parameter(torch.zeros(1, self.num_views, 1, embed_dim))
            nn.init.normal_(self.view_embed, std=0.02)
        else:
            # None 不进入 state_dict，单视角 checkpoint 的参数集合保持不变。
            self.view_embed = None
        self.output_norm = nn.LayerNorm(embed_dim)
        self.token_dropout = float(dropout)

        self.register_buffer(
            "pixel_mean", torch.tensor(self._PIXEL_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(self._PIXEL_STD).view(1, 3, 1, 1), persistent=False
        )

    def _load_local_pretrained_weights(self, pretrained_path: Path) -> None:
        """严格加载 torchvision/timm ResNet 主干权重，仅允许多出分类头。"""
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"ImageEncoder pretrained weights not found: {pretrained_path}"
            )

        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        state_dict = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else None
        )
        if not isinstance(state_dict, dict):
            raise TypeError(
                "ImageEncoder pretrained checkpoint must be a state_dict or contain "
                f"state_dict: {pretrained_path}"
            )

        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        allowed_unexpected = {
            "fc.weight",
            "fc.bias",
            "classifier.weight",
            "classifier.bias",
        }
        unexpected = [
            key for key in incompatible.unexpected_keys if key not in allowed_unexpected
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "ImageEncoder local pretrained weights do not match "
                f"arch={self.backbone.__class__.__name__}: "
                f"missing={incompatible.missing_keys}, unexpected={unexpected}, "
                f"path={pretrained_path}"
            )

        loaded_keys = len(state_dict) - len(incompatible.unexpected_keys)
        logger.info(
            "ImageEncoder loaded %d backbone tensors from %s",
            loaded_keys,
            pretrained_path,
        )

    def forward(
        self,
        image: torch.Tensor,
        image_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:param image: [B,3,H,W] 或 [B,V,3,H,W] uint8/float。
        :param image_valid: [B] 或 [B,V] bool。
        :return: (tokens [B,V*N,D], keep_mask [B,V*N])；无效 / dropped token 全 0。
        """
        if image.ndim == 4:
            B, V = image.shape[0], 1
            image = image.unsqueeze(1)
        elif image.ndim == 5:
            B, V = image.shape[:2]
        else:
            raise ValueError(
                "ImageEncoder image must have shape [B,3,H,W] or [B,V,3,H,W], "
                f"got {tuple(image.shape)}"
            )
        if V != self.num_views:
            raise ValueError(
                f"ImageEncoder configured for {self.num_views} views but received {V}"
            )
        if image.shape[2] != 3:
            raise ValueError(f"ImageEncoder expects RGB input, got shape {tuple(image.shape)}")

        if image_valid.ndim == 1 and V == 1:
            image_valid = image_valid.unsqueeze(1)
        if tuple(image_valid.shape) != (B, V):
            raise ValueError(
                f"image_valid must have shape {(B, V)}, got {tuple(image_valid.shape)}"
            )

        # uint8/[0,255] -> [0,1]；已是 [0,1] float 则不重复缩放
        x = image.reshape(B * V, *image.shape[2:]).float()
        if x.dtype == torch.float32 and x.max() > 1.5:
            x = x / 255.0
        elif image.dtype == torch.uint8:
            x = x / 255.0
        x = (x - self.pixel_mean) / self.pixel_std

        feat = self.backbone(x)[-1]          # [B, C, H', W']
        feat = self.proj(feat)               # [B, embed_dim, H', W']
        feat = self.pool(feat)               # [B, embed_dim, grid_h, grid_w]
        tokens = feat.flatten(2).transpose(1, 2)  # [B*V, N, embed_dim]
        tokens = tokens + self.pos_embed.to(tokens.dtype)
        if self.view_embed is None:
            # 保持历史单视角计算路径与参数结构。
            tokens = self.output_norm(tokens)
            tokens = tokens.reshape(B, self.num_tokens, -1)
        else:
            tokens = tokens.reshape(B, V, self.num_tokens, -1)
            tokens = self.output_norm(tokens + self.view_embed.to(tokens.dtype))
            tokens = tokens.flatten(1, 2)

        keep = image_valid.bool().unsqueeze(-1).expand(B, V, self.num_tokens)
        keep = keep.reshape(B, V * self.num_tokens)
        if self.training and self.token_dropout > 0.0:
            drop = torch.rand(B, V * self.num_tokens, device=tokens.device) >= self.token_dropout
            keep = keep & drop
        tokens = tokens * keep.unsqueeze(-1).to(tokens.dtype)
        return tokens, keep


class TemporalPyramidEncoder(nn.Module):
    """短序列时序金字塔（纯 PyTorch Conv1d 版时序 FPN，不依赖 natten）。

    输入差分序列 [M, C_in, T]（M=slot 数，T=T_delta，mogo=3），多尺度聚合 + top-down
    FPN 融合，输出取当前帧侧最后时间步 [M, dim]。pluto NATSequenceEncoder 为 21 帧 +
    natten 设计，对 3-4 帧短序列改用自适应 Conv1d 金字塔，保留「多尺度时序聚合」语义。
    用 GroupNorm(不是 BatchNorm)，对可变/小批次稳定。
    """

    def __init__(self, in_chans: int, dim: int):
        super().__init__()
        self.tokenizer = nn.Conv1d(in_chans, dim, kernel_size=3, padding=1)
        # 细尺度分支（全分辨率）
        self.l0 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, dim),
            nn.SiLU(),
        )
        # 中尺度分支（下采样 2x），T<2 时在 forward 里跳过
        self.l1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, dim),
            nn.SiLU(),
        )
        self.fpn_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: [M, C_in, T]; :return: [M, dim]（当前帧侧最后时间步）。"""
        T = x.shape[-1]
        x = self.tokenizer(x)                    # [M, dim, T]
        l0 = self.l0(x)                          # [M, dim, T]
        g = l0.mean(dim=-1, keepdim=True)        # 全局尺度 [M, dim, 1]
        fused = l0 + g
        if T >= 2:
            l1 = self.l1(x)                      # [M, dim, ceil(T/2)]
            l1_up = F.interpolate(l1, size=T, mode="linear", align_corners=False)
            fused = fused + l1_up
        fused = self.fpn_conv(fused)             # [M, dim, T]
        return fused[:, :, -1]                   # [M, dim]


class TemporalObjectEncoder(nn.Module):
    """时序障碍物编码器：abs 几何分支 + 动态 motion 分支，加法融合。

    - abs 分支（所有 slot）：与 ObjectEncoder 同构（LayerNorm+MLP+type_embedding），
      编当前帧几何/位置约束。
    - motion 分支：差分序列 -> TemporalPyramidEncoder -> motion_proj，仅对 (valid & is_dynamic)
      的 slot 通过 gate 保留；静态/padding slot 的 motion 置零。
    - final = output_norm(abs_token + motion)，按 valid_mask 置零、训练期 token_dropout。

    motion_proj 零初始化 -> 加载旧 ckpt / 训练初期 motion≈0，final≈abs（等价旧 MLP），
    平滑收敛，且 padding slot 天然输出 0。对所有 slot 都跑 temporal（而非 gather），避免
    DDP 下某 batch 无动态物体时 temporal 参数收不到梯度而报 unused-parameter。
    GRPO 需切 eval（同 ObjectEncoder），关掉 token_dropout 保证 rollout/logprob 一致。
    """

    def __init__(
        self,
        feature_dim: int,
        diff_dim: int,
        embed_dim: int,
        num_types: int,
        hidden_dim: int,
        temporal_dim: int,
        dropout: float = 0.0,
        sanitize_diff: bool = True,
        diff_clamp: float = 20.0,
    ):
        super().__init__()
        # 时序差分输入净化开关（默认开，正常数据零影响；兜底极少数 ego_pose 异常样本的 1e5~1e6 坏值）。
        self.sanitize_diff = bool(sanitize_diff)
        self.diff_clamp = float(diff_clamp)
        # abs 分支：与 ObjectEncoder 结构一致
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.type_embedding = nn.Embedding(num_types, embed_dim)
        # motion 分支
        self.temporal = TemporalPyramidEncoder(in_chans=diff_dim, dim=temporal_dim)
        self.motion_proj = nn.Linear(temporal_dim, embed_dim)
        nn.init.zeros_(self.motion_proj.weight)
        nn.init.zeros_(self.motion_proj.bias)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.token_dropout = float(dropout)

    def forward(
        self,
        obj_features: torch.Tensor,
        obj_diff_features: torch.Tensor,
        obj_type: torch.Tensor,
        obj_valid_mask: torch.Tensor,
        obj_is_dynamic: torch.Tensor,
    ) -> torch.Tensor:
        """:return: [B, N, embed_dim]，padding / dropped 行为全 0（与 ObjectEncoder 一致）。"""
        B, N = obj_features.shape[0], obj_features.shape[1]
        abs_token = self.feature_mlp(self.feature_norm(obj_features)) + self.type_embedding(obj_type.long())

        # 对所有 slot 跑 temporal（DDP 友好），再用 gate 只保留有效动态 slot 的 motion。
        # 净化：极少数样本因历史帧 ego_pose 异常，diff 出现 1e5~1e6 坏值，fp16 下经 Conv1d
        # tokenizer 溢出成 inf/nan 并污染整个 batch。入口 nan_to_num + 对称 clamp 到 ±diff_clamp
        # （正常 diff 已归一化到 O(1)，实测全量 max≈6；clamp 到 20 不触及真实数据，仅截断坏值）。
        # 在 permute/reshape 前做，最省算子且不改变形状语义。
        if self.sanitize_diff:
            obj_diff_features = torch.nan_to_num(
                obj_diff_features, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(-self.diff_clamp, self.diff_clamp)
        diff = obj_diff_features.permute(0, 1, 3, 2).reshape(B * N, obj_diff_features.shape[-1], obj_diff_features.shape[-2])
        motion = self.motion_proj(self.temporal(diff)).view(B, N, -1)
        gate = (obj_valid_mask.bool() & obj_is_dynamic.bool()).unsqueeze(-1).to(motion.dtype)
        motion = motion * gate

        tokens = self.output_norm(abs_token + motion)
        mask = obj_valid_mask.unsqueeze(-1).to(tokens.dtype)
        if self.training and self.token_dropout > 0.0:
            keep = (torch.rand_like(obj_valid_mask.float()) >= self.token_dropout)
            mask = mask * keep.unsqueeze(-1).to(tokens.dtype)
        return tokens * mask


@dataclass
class FlowConfig:
    """Configuration specific to Flow Matching."""
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    mean_variance_net: bool = False
    sde_noise_scale: float = 0.1
    # Plan B: single std floor shared by flow sampling AND logprob re-evaluation.
    # Replaces the old DDIM-style min_sampling/min_logprob split so that the
    # Gaussian that produced x_{t+1} is the exact same Gaussian get_logprobs scores
    # it under.
    #
    # flow_std = max(sde_noise_scale*sqrt(dt), sde_min_std). With 5 steps dt=0.2,
    # sde_noise_scale*sqrt(dt) = 0.1*0.447 = 0.0447 < 0.05, so the sde_min_std floor
    # dominates -> effective flow_std = 0.05. In normalized space the denorm scales
    # are x:33.37 y:21.0 (see norm_odo), so 0.05 == +-1.67m / +-1.05m of injected
    # noise per Euler step, on par with the DDIM rollout std (0.04). The earlier
    # value 0.1 meant +-3.3m / +-2.1m per step: flow rollouts were far noisier than
    # the deterministic eval path, which corrupted the policy learned by GRPO.
    # IMPORTANT: lowering ONLY sde_min_std does not lower flow_std unless
    # sde_noise_scale*sqrt(dt) is also below the new floor (with scale=0.2 it was
    # 0.089 and would have pinned flow_std at 0.089). Always watch the flow_std
    # diagnostic (Plan D) to confirm the value actually in effect.
    sde_min_std: float = 0.05

@dataclass
class DDPMConfig:
    """Configuration specific to DDPM."""
    num_train_timesteps: int = 100

@dataclass
class DDIMConfig:
    """Configuration specific to DDIM."""
    num_train_timesteps: int = 100
    ddim_eta: float = 0.0

@dataclass
class GRPOConfig:
    """Configuration specific to GRPO training."""
    denoised_clip_value: float = 1.0
    eval_randn_clip_value: float = 1.0
    randn_clip_value: float = 5.0
    final_action_clip_value: float = 1.0
    eps_clip_value: Optional[float] = None
    eval_min_sampling_denoising_std: float = 0.0001
    min_sampling_denoising_std: float = 0.04
    min_logprob_denoising_std: float = 0.1
    clip_advantage_lower_quantile: float = 0.0
    clip_advantage_upper_quantile: float = 1.0
    gamma_denoising: float = 0.6
    
    metric_cache_path: str = "/path/to/metric_cache_train"
    reference_policy_checkpoint: str = "/path/to/IL_Model.ckpt"
    scorer_config: PDMScorerConfig = field(default_factory=lambda: PDMScorerConfig(
        progress_weight=5.0, ttc_weight=5.0, comfortable_weight=2.0
    ))
    vehicle_parameters: Any = field(default_factory=get_mogo_vehicle_parameters)
    # --- Step 3 (optional BC-tightening switch; defaults reproduce Step 1+2) ---
    # bc_coeff weights the behavior-cloning anchor toward the reference policy.
    # Default 0.1 == Step 1+2. Raise it (e.g. 0.3) via agent.bc_coeff=0.3 to stop
    # RL from pushing stop / low-speed scenes forward.
    # bc_teacher_deterministic chooses how old_policy rolls out the BC teacher
    # chains: False (default) = stochastic SDE rollout (Step 1+2); True = clean ODE
    # rollout (std=0) so BC anchors to the deterministic reference trajectory
    # instead of a +-noise rollout. Both are overridable from recogdrive_agent.yaml.
    bc_coeff: float = 0.1
    bc_teacher_deterministic: bool = False


@dataclass
class ReCogDriveDiffusionPlannerConfig(PretrainedConfig):
    """A refined configuration for the ReCogDriveDiffusionPlanner."""
    # --- Core Architecture ---
    diffusion_model_cfg: dict = field(default_factory=dict)
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    action_dim: int = 3
    action_horizon: int = 8
    add_pos_embed: bool = True
    max_seq_len: int = 8
    ego_status_encoder_type: Literal['mlp', 'attention'] = 'mlp'
    ego_status_dim: int = 13
    # SDE(状态丢弃编码器)参数，仅 ego_status_encoder_type='attention' 时生效，与 vlm_ablation 正交。
    # status_feature 前 ego_num_command_states 维是导航命令(cmd_0..cmd_3)，永久可见；其余
    # 运动学量(vx/vy/ax/ay/yaw_rate/curvature/curvature_rate/steer_angle/steer_rate)训练时
    # 按 ego_state_dropout 概率随机遮蔽，抑制"外推当前自车运动学"的捷径。
    ego_state_dropout: float = 0.75
    ego_num_command_states: int = 4

    sampling_method: Literal['flow', 'ddpm', 'ddim'] = 'ddim'
    num_inference_steps: int = 5
    model_dtype: str = "float16"
    grpo: bool = False
    vlm_size: str = 'large'
    vl_hidden_dim: int = 1536
    normalize_vl_hidden_states: bool = False
    vl_hidden_norm_eps: float = 1e-5

    # --- hdmap vector token（默认关闭，零影响）---
    use_hdmap_vector: bool = False
    max_map_tokens: int = 128
    map_sample_points: int = 20
    # 开启后 MapEncoder 额外编码左右车道边界几何（per-point 6->13 维），token 数/缓存均不变。
    use_lane_boundaries: bool = False
    # fp16 数值加固开关（默认关，字节级零影响）：对进 MapEncoder 的局部米制坐标做
    # nan_to_num→clamp→scale。仅影响 planner 内 MapEncoder.forward，不改缓存；从头训练效果
    # 一致，但旧 ckpt 不可热启（首层输入尺度变了）。多城市/长训 fp16 数值稳健性用。
    hdmap_input_norm: bool = False
    num_map_polygon_types: int = 4
    num_map_route_states: int = 3
    num_map_tl_states: int = 5
    num_map_control_types: int = 3
    num_map_tl_directions: int = 6

    # --- VLM 消融开关（默认 none，零影响）---
    # "none": 正常使用 VLM hidden state。
    # "zero": 将 VLM 的 summary / context token 在投影后置零，只消融 VLM 分支；
    #         obj token / ego status / history & action 等非 VLM 输入完整保留。
    #         置零发生在 encode_vl_context_tokens 内部（拼 obj 之前），故 obj token 不受影响。
    #         IL / eval / GRPO-rollout / GRPO-logprob 四处入口共用 encode_vl_* 函数，自动一致。
    vlm_ablation: Literal["none", "zero"] = "none"

    # --- VLM 总开关（默认 True=加载并使用 VLM）---
    # True: 与历史行为完全一致（构造 vl_*_projector/gate/fusion_projector，DiT 用 VLM
    #   summary/context token）。
    # False: 完全不使用 VLM —— planner 不构造任何 vl_*_projector/gate/fusion_projector（模型
    #   更小），DiT 的 cross-attention context 只由 obj/hdmap/img token 组成（若三者都关，则
    #   context 为空、DiT 退化为纯 ego(adaLN)+action self-attention）；agent 侧不加载 Qwen3、
    #   eval 不在线前向。与 vlm_ablation 的区别：zero 仍加载 VLM 并在投影后置零（不省算力/参数），
    #   use_vlm=False 直接不构造、不前向。ckpt 不可与 use_vlm=True 互移植（投影层集合不同，会被
    #   initialize() 的 strict=False 静默丢弃），须重训 IL 并重派生 RL reference。
    use_vlm: bool = True

    # --- 障碍物 obj token（默认关闭，零影响）---
    use_object_vector: bool = False
    obj_feature_dim: int = 15
    obj_num_types: int = 5
    object_token_dropout: float = 0.1
    # 编码器类型: "mlp"=ObjectEncoder(单帧几何); "temporal"=TemporalObjectEncoder(差分+时序FPN+静态MLP)
    object_encoder_type: Literal["mlp", "temporal"] = "mlp"
    obj_diff_dim: int = 9                 # 差分特征维（temporal 分支），须与 object_temporal_features.OBJECT_DIFF_DIM 一致
    obj_temporal_dim: int = 128           # 时序金字塔内部维
    static_speed_threshold: float = 0.5   # 动/静划分阈值(m/s)，数据侧划分用；此处保留以便配置透传
    # 时序差分输入净化（默认开，对正常数据字节级零影响）。极少数样本（实测上海 predata 批次
    # 6/72万）因**原始 pkl 某帧 gt_boxes 漏做 global->ego 变换**，该帧 object 坐标是 UTM 大数
    # (3.4e6)，导致差分 obj_diff_features 出现 1e5~1e6 量级坏值（object 在该中间帧瞬移出去再
    # 回来，产生等量反向 delta）。注意根因在 object 坐标、不在 ego_pose：ego 变换其实成功抵消了
    # UTM 偏移，抵消不掉的是 object 自身 baked-in 的大数（已回溯原始 pkl 核实，见 memory）。
    # 这些坏值经 fp16 的 Conv1d tokenizer 溢出成 inf/nan，污染整个 batch 的 loss。入口处
    # nan_to_num + 对称 clamp 到 ±obj_diff_clamp（正常 diff 已归一化到 O(1)，实测全量 max≈6，
    # clamp 到 20 不触及真实数据、仅兜底坏值），彻底阻断 nan。与数据侧修复(_build_diff_features
    # 异常帧置零)互补：数据侧根治新缓存、此处兜底已固化在旧缓存里的坏样本，缓存无需重跑即可安全
    # 训练。真正的上游根治需在 DrivePi_mogo parseRosBag 修 gt_boxes 变换。
    sanitize_obj_diff: bool = True
    obj_diff_clamp: float = 20.0

    # --- 站点 token（进出站导航，默认关闭，零影响）---
    # True: 把当前活动站点（从 navigation_command 的 park + route_centerline 派生）投影到 ego 系，
    #   编成 station token 拼到 context 末尾进 DiT cross-attention，为规划器提供显式进出站目标点。
    #   无 park / 旧数据 → station 全 padding（valid_mask=False），编码器置零、DiT 忽略，等价无约束。
    # 需与 use_navigation_command 搭配使用（命令段告诉模型「要进站」，station token 给出「站在哪」）。
    use_station_token: bool = False
    station_feature_dim: int = 6          # 须与 station_features.STATION_FEATURE_DIM 一致
    max_station_tokens: int = 1           # 站点 token 数（第一版固定 1）

    # --- 单/多视角图像 ResNet encoder token（默认关闭，零影响；实验用）---
    # 开启后把当前帧图像经共享 ResNet 编成空间 token，拼到 vl_context_tokens 后进 DiT
    # cross-attention。与 vlm_ablation 正交：vlm_ablation=zero 仅置零 VLM token，图像 token
    # 保留，从而"屏蔽 VLM 语义、保留图像视觉信息"。需补跑特征缓存生成 front_image .gz；
    # 已有 hidden state 不重算。
    use_image_encoder: bool = False
    image_encoder_arch: str = DEFAULT_IMAGE_ENCODER_ARCH
    image_encoder_pretrained: bool = False
    image_encoder_pretrained_path: Optional[str] = DEFAULT_IMAGE_ENCODER_PRETRAINED_PATH
    image_grid_h: int = 8                  # ResNet 特征图自适应池化到的 token 网格高
    image_grid_w: int = 16                 # 网格宽；token 数 = grid_h*grid_w（默认 128）
    image_token_dropout: float = 0.0       # 训练期图像 token 随机丢弃概率（GRPO 时强制 eval 关闭）
    num_image_views: int = 1               # 视角数；顺序由 agent.image_view_list 定义

    # --- 显式安全损失（默认关闭，保持旧训练行为）---
    use_explicit_safety_loss: bool = False
    lambda_collision: float = 0.1
    lambda_offroad: float = 0.05
    lambda_route: float = 0.02
    collision_margin: float = 0.5
    offroad_margin: float = 0.5
    route_lateral_margin: float = 1.5
    route_heading_weight: float = 0.5
    safety_smooth_beta: float = 10.0
    safety_snr_weighting: bool = True
    route_loss_batch_chunk_size: int = 128
    # Mogo 轨迹 pose 位于后轴中心；碰撞和越界 box 必须换算到几何中心。
    ego_front_length: float = 4.868
    ego_rear_length: float = 1.127
    ego_width: float = 2.100

    tune_projector: bool = True
    tune_diffusion_model: bool = True
    
    flow_cfg: FlowConfig = field(default_factory=FlowConfig)
    ddpm_cfg: DDPMConfig = field(default_factory=DDPMConfig)
    ddim_cfg: DDIMConfig = field(default_factory=DDIMConfig)
    grpo_cfg: GRPOConfig = field(default_factory=GRPOConfig)


class ReCogDriveDiffusionPlanner(nn.Module):
    config_class = ReCogDriveDiffusionPlannerConfig

    def __init__(self, config: ReCogDriveDiffusionPlannerConfig):
        super().__init__()
        self.config = config
        if (
            min(config.lambda_collision, config.lambda_offroad, config.lambda_route)
            < 0.0
        ):
            raise ValueError("Safety loss weights must be non-negative")
        if config.safety_smooth_beta <= 0.0:
            raise ValueError("safety_smooth_beta must be positive")
        if config.route_loss_batch_chunk_size <= 0:
            raise ValueError("route_loss_batch_chunk_size must be positive")

        self.model = LightningDiT(**config.diffusion_model_cfg)

        if config.ego_status_encoder_type == 'attention':
            self.ego_status_encoder = StateAttentionEncoder(
                state_dim=config.ego_status_dim,
                embed_dim=config.input_embedding_dim,
                num_command_states=config.ego_num_command_states,
                state_dropout=config.ego_state_dropout,
            )
        else: 
            self.ego_status_encoder = Mlp(
                in_features=config.ego_status_dim,
                hidden_features=config.hidden_size,
                out_features=config.input_embedding_dim,
                norm_layer=nn.LayerNorm
            )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.input_embedding_dim,
        )
        vl_hidden_dim = int(config.vl_hidden_dim)

        if config.use_vlm:
            if config.normalize_vl_hidden_states:
                self.vl_summary_norm = nn.LayerNorm(vl_hidden_dim, eps=config.vl_hidden_norm_eps)
                self.vl_context_first_norm = nn.LayerNorm(vl_hidden_dim, eps=config.vl_hidden_norm_eps)
                self.vl_context_last_norm = nn.LayerNorm(vl_hidden_dim, eps=config.vl_hidden_norm_eps)
            else:
                self.vl_summary_norm = nn.Identity()
                self.vl_context_first_norm = nn.Identity()
                self.vl_context_last_norm = nn.Identity()

            self.vl_summary_projector = nn.Linear(vl_hidden_dim, config.input_embedding_dim)
            self.vl_context_first_projector = nn.Linear(vl_hidden_dim, config.input_embedding_dim)
            self.vl_context_last_projector = nn.Linear(vl_hidden_dim, config.input_embedding_dim)
            self.vl_context_gate = nn.Linear(config.input_embedding_dim * 2, config.input_embedding_dim)
            nn.init.zeros_(self.vl_context_gate.weight)
            nn.init.constant_(self.vl_context_gate.bias, 1.0)

            self.fusion_projector = nn.Linear(config.input_embedding_dim * 2, config.input_embedding_dim)
        else:
            # use_vlm=False：不构造任何 VLM 投影 / 门控 / 融合层（模型更小）。
            # build_context_tokens 走 _build_context_tokens_no_vlm，build_action_latents
            # 直接返回 action_features（不再拼 VLM summary），故这些层永不被引用。
            self.vl_summary_norm = None
            self.vl_context_first_norm = None
            self.vl_context_last_norm = None
            self.vl_summary_projector = None
            self.vl_context_first_projector = None
            self.vl_context_last_projector = None
            self.vl_context_gate = None
            self.fusion_projector = None

        self.map_encoder = MapEncoder(
            hidden_size=config.input_embedding_dim,
            num_polygon_types=config.num_map_polygon_types,
            num_route_states=config.num_map_route_states,
            num_tl_states=config.num_map_tl_states,
            num_control_types=config.num_map_control_types,
            num_tl_directions=config.num_map_tl_directions,
            use_lane_boundaries=config.use_lane_boundaries,
            input_norm=config.hdmap_input_norm,
        ) if config.use_hdmap_vector else None

        # 障碍物编码器：把 obj 特征编成 context token，拼到 vl_context_tokens 后面。
        # 必须在下方 _init_grpo() 的 deepcopy(self) 之前创建，old_policy 才会带上它。
        if config.use_object_vector:
            if config.object_encoder_type == "temporal":
                self.object_encoder = TemporalObjectEncoder(
                    feature_dim=config.obj_feature_dim,
                    diff_dim=config.obj_diff_dim,
                    embed_dim=config.input_embedding_dim,
                    num_types=config.obj_num_types,
                    hidden_dim=config.hidden_size,
                    temporal_dim=config.obj_temporal_dim,
                    dropout=config.object_token_dropout,
                    sanitize_diff=config.sanitize_obj_diff,
                    diff_clamp=config.obj_diff_clamp,
                )
            else:
                self.object_encoder = ObjectEncoder(
                    feature_dim=config.obj_feature_dim,
                    embed_dim=config.input_embedding_dim,
                    num_types=config.obj_num_types,
                    hidden_dim=config.hidden_size,
                    dropout=config.object_token_dropout,
                )
        else:
            self.object_encoder = None

        # 站点编码器（进出站导航）。同 object_encoder，必须在下方 _init_grpo() 的
        # deepcopy(self) 之前创建，old_policy 才会带上它，GRPO rollout/logprob 才条件一致。
        if config.use_station_token:
            self.station_encoder = StationEncoder(
                feature_dim=config.station_feature_dim,
                embed_dim=config.input_embedding_dim,
                hidden_dim=config.hidden_size,
            )
        else:
            self.station_encoder = None

        # 单/多视角 ResNet 编码器（实验用）。同 object_encoder，必须在下方 _init_grpo() 的
        # deepcopy(self) 之前创建，old_policy 才会带上它，GRPO rollout/logprob 才条件一致。
        if config.use_image_encoder:
            self.image_encoder = ImageEncoder(
                embed_dim=config.input_embedding_dim,
                arch=config.image_encoder_arch,
                pretrained=config.image_encoder_pretrained,
                pretrained_path=config.image_encoder_pretrained_path,
                grid_h=config.image_grid_h,
                grid_w=config.image_grid_w,
                dropout=config.image_token_dropout,
                num_views=config.num_image_views,
            )
        else:
            self.image_encoder = None

        output_dim = 2 * config.action_dim if (
            config.sampling_method == 'flow' and config.flow_cfg.mean_variance_net
        ) else config.action_dim
        
        self.action_decoder = Mlp(
            in_features=self.model.output_dim,
            hidden_features=config.hidden_size,
            out_features=output_dim,
            norm_layer=nn.LayerNorm
        )
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.config.sampling_method == 'flow':
            self._init_flow_sampler(config.flow_cfg)
        elif self.config.sampling_method == 'ddpm':
            self._init_ddpm_sampler(config.ddpm_cfg)
        elif self.config.sampling_method == 'ddim':
            self._init_ddim_sampler(config.ddim_cfg)

        if config.grpo:
            self._init_grpo(config.grpo_cfg)

    def encode_vl_summary_tokens(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        tokens = self.vl_summary_projector(self.vl_summary_norm(last_hidden_state))
        if self.config.vlm_ablation == "zero":
            return torch.zeros_like(tokens)
        return tokens

    def encode_vl_context_tokens(
        self,
        first_hidden_state: torch.Tensor,
        last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        first_context_tokens = self.vl_context_first_projector(self.vl_context_first_norm(first_hidden_state))
        last_context_tokens = self.vl_context_last_projector(self.vl_context_last_norm(last_hidden_state))
        context_gate = torch.sigmoid(
            self.vl_context_gate(torch.cat([first_context_tokens, last_context_tokens], dim=-1))
        )
        context_tokens = context_gate * last_context_tokens + (1.0 - context_gate) * first_context_tokens
        # VLM 消融：仅置零 VLM context（在 _build_context_tokens 拼 obj 之前），obj token 保留。
        if self.config.vlm_ablation == "zero":
            return torch.zeros_like(context_tokens)
        return context_tokens

    def encode_map_context_tokens(self, action_input: Optional[BatchFeature]) -> Optional[torch.Tensor]:
        """编码 hdmap vector context tokens。"""
        if not self.config.use_hdmap_vector or self.map_encoder is None or action_input is None:
            return None
        if any(key not in action_input for key in HDMAP_VECTOR_BATCH_KEYS):
            logger.warning("use_hdmap_vector=True but HDMap keys missing from action_input; map encoder skipped.")
            return None
        return self.map_encoder(
            point_position=action_input.map_point_position,
            point_vector=action_input.map_point_vector,
            point_orientation=action_input.map_point_orientation,
            polygon_type=action_input.map_polygon_type,
            polygon_on_route=action_input.map_polygon_on_route,
            polygon_tl_status=action_input.map_polygon_tl_status,
            polygon_control_type=action_input.map_polygon_control_type,
            polygon_tl_direction=action_input.map_polygon_tl_direction,
            polygon_has_speed_limit=action_input.map_polygon_has_speed_limit,
            polygon_speed_limit=action_input.map_polygon_speed_limit,
            point_valid_mask=action_input.map_point_valid_mask,
            map_valid_mask=action_input.map_valid_mask,
        )

    def _extract_obj_inputs(self, action_input: BatchFeature) -> Optional[Dict[str, torch.Tensor]]:
        """从 action_input 取 obj_* 字段；未启用 obj token 或字段缺失时返回 None。"""
        if self.object_encoder is None:
            return None
        if "obj_features" not in action_input.data:
            return None
        obj_inputs = {
            "obj_features": action_input.obj_features,
            "obj_type": action_input.obj_type,
            "obj_valid_mask": action_input.obj_valid_mask,
        }
        # temporal 编码器额外需要差分序列与动静标记
        if isinstance(self.object_encoder, TemporalObjectEncoder):
            if "obj_diff_features" not in action_input.data:
                return None
            obj_inputs["obj_diff_features"] = action_input.obj_diff_features
            obj_inputs["obj_is_dynamic"] = action_input.obj_is_dynamic
        return obj_inputs

    def _extract_img_inputs(self, action_input: Optional[BatchFeature]) -> Optional[Dict[str, torch.Tensor]]:
        """从 action_input 取图像字段；未启用图像 encoder 或字段缺失时返回 None。"""
        if self.image_encoder is None or action_input is None:
            return None
        if "front_image" not in action_input.data:
            return None
        return {
            "front_image": action_input.front_image,
            "front_image_valid": action_input.front_image_valid,
        }

    def encode_image_context_tokens(
        self, action_input: Optional[BatchFeature]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码图像 context tokens，返回 [B,V*N,D] token 与 [B,V*N] mask。"""
        img_inputs = self._extract_img_inputs(action_input)
        if img_inputs is None:
            return None, None
        tokens, keep = self.image_encoder(
            img_inputs["front_image"],
            img_inputs["front_image_valid"],
        )
        return tokens, keep

    def _extract_station_inputs(
        self, action_input: Optional[BatchFeature]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """从 action_input 取 station_* 字段；未启用 station token 或字段缺失时返回 None。"""
        if self.station_encoder is None or action_input is None:
            return None
        if "station_features" not in action_input.data:
            return None
        return {
            "station_features": action_input.station_features,
            "station_valid_mask": action_input.station_valid_mask,
        }

    def encode_station_context_tokens(
        self, action_input: Optional[BatchFeature]
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码站点 context tokens，返回 [B,N_station,D] token 与 [B,N_station] mask。"""
        station_inputs = self._extract_station_inputs(action_input)
        if station_inputs is None:
            return None, None
        tokens = self.station_encoder(
            station_inputs["station_features"],
            station_inputs["station_valid_mask"],
        )
        return tokens, station_inputs["station_valid_mask"].bool()

    def build_context_tokens(
        self,
        first_hidden_state: torch.Tensor,
        last_hidden_state: torch.Tensor,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (vl_summary_tokens, vl_context_tokens, context_key_padding_mask).

        统一构建 DiT cross-attention 的 context tokens，同时支持 hdmap 和 obj：
        context = concat([vl_context_tokens, map_context_tokens, obj_context_tokens], dim=1)

        context_key_padding_mask is a bool tensor [B, N_ctx] where True = valid token.
        VLM tokens are always valid; map/obj padding slots are marked False.
        Returns None for the mask when both HDMap and Object are disabled (backward-compat path).
        """
        if not self.config.use_vlm:
            return self._build_context_tokens_no_vlm(action_input)

        vl_summary_tokens = self.encode_vl_summary_tokens(last_hidden_state)
        vl_context_tokens = self.encode_vl_context_tokens(first_hidden_state, last_hidden_state)

        # 收集 hdmap context tokens
        map_context_tokens = self.encode_map_context_tokens(action_input)

        # 收集 obj context tokens
        obj_inputs = self._extract_obj_inputs(action_input) if action_input is not None else None
        obj_context_tokens = None
        if self.object_encoder is not None and obj_inputs is not None:
            if isinstance(self.object_encoder, TemporalObjectEncoder):
                obj_context_tokens = self.object_encoder(
                    obj_inputs["obj_features"].to(vl_context_tokens.dtype),
                    obj_inputs["obj_diff_features"].to(vl_context_tokens.dtype),
                    obj_inputs["obj_type"],
                    obj_inputs["obj_valid_mask"],
                    obj_inputs["obj_is_dynamic"],
                )
            else:
                obj_context_tokens = self.object_encoder(
                    obj_inputs["obj_features"].to(vl_context_tokens.dtype),
                    obj_inputs["obj_type"],
                    obj_inputs["obj_valid_mask"],
                )

        # 收集单/多视角图像 context tokens（实验用）
        img_context_tokens, img_keep_mask = self.encode_image_context_tokens(action_input)

        # 收集站点 context tokens（进出站导航）
        station_context_tokens, station_keep_mask = self.encode_station_context_tokens(action_input)

        # 拼接所有 context tokens 和构建 mask
        B, N_vl = vl_context_tokens.shape[0], vl_context_tokens.shape[1]
        context_tokens_list = [vl_context_tokens]
        mask_list = [torch.ones(B, N_vl, dtype=torch.bool, device=vl_context_tokens.device)]

        if map_context_tokens is not None:
            context_tokens_list.append(map_context_tokens)
            map_mask = action_input.map_valid_mask.bool().to(device=vl_context_tokens.device)
            mask_list.append(map_mask)

        if obj_context_tokens is not None:
            context_tokens_list.append(obj_context_tokens)
            obj_mask = obj_inputs["obj_valid_mask"].bool().to(device=vl_context_tokens.device)
            mask_list.append(obj_mask)

        if img_context_tokens is not None:
            context_tokens_list.append(img_context_tokens.to(vl_context_tokens.dtype))
            mask_list.append(img_keep_mask.bool().to(device=vl_context_tokens.device))

        if station_context_tokens is not None:
            context_tokens_list.append(station_context_tokens.to(vl_context_tokens.dtype))
            mask_list.append(station_keep_mask.to(device=vl_context_tokens.device))

        if len(context_tokens_list) > 1:
            vl_context_tokens = torch.cat(context_tokens_list, dim=1)
            context_key_padding_mask = torch.cat(mask_list, dim=1)
        else:
            context_key_padding_mask = None

        return vl_summary_tokens, vl_context_tokens, context_key_padding_mask

    def _build_context_tokens_no_vlm(
        self,
        action_input: Optional[BatchFeature] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """use_vlm=False 分支：不使用任何 VLM hidden state / 投影层。

        context 仅由 obj / hdmap / img token 组成（三者都关则为 [B, 0, D] 空张量，DiT 会对
        空 context 退化为纯 self-attention，不产生 NaN）。返回的 vl_summary_tokens 是一个
        [B, 1, D] 零占位张量，仅用于向下游 get_action / sample_chain / get_logprobs 传递
        batch/device/dtype；build_action_latents 在 use_vlm=False 时会忽略它。
        """
        assert action_input is not None, (
            "use_vlm=False requires action_input (for status_feature shape/device)."
        )
        ref = action_input.status_feature
        B = ref.shape[0]
        device = ref.device
        dtype = next(self.model.parameters()).dtype
        D = self.config.input_embedding_dim

        context_tokens_list = []
        mask_list = []

        # hdmap context tokens
        map_context_tokens = self.encode_map_context_tokens(action_input)
        if map_context_tokens is not None:
            context_tokens_list.append(map_context_tokens.to(dtype))
            mask_list.append(action_input.map_valid_mask.bool().to(device=device))

        # obj context tokens（与 build_context_tokens 的 obj 分支等价，复制以保持 use_vlm=True
        # 路径字节级不变）
        obj_inputs = self._extract_obj_inputs(action_input)
        if self.object_encoder is not None and obj_inputs is not None:
            if isinstance(self.object_encoder, TemporalObjectEncoder):
                obj_context_tokens = self.object_encoder(
                    obj_inputs["obj_features"].to(dtype),
                    obj_inputs["obj_diff_features"].to(dtype),
                    obj_inputs["obj_type"],
                    obj_inputs["obj_valid_mask"],
                    obj_inputs["obj_is_dynamic"],
                )
            else:
                obj_context_tokens = self.object_encoder(
                    obj_inputs["obj_features"].to(dtype),
                    obj_inputs["obj_type"],
                    obj_inputs["obj_valid_mask"],
                )
            context_tokens_list.append(obj_context_tokens.to(dtype))
            mask_list.append(obj_inputs["obj_valid_mask"].bool().to(device=device))

        # 单/多视角图像 context tokens
        img_context_tokens, img_keep_mask = self.encode_image_context_tokens(action_input)
        if img_context_tokens is not None:
            context_tokens_list.append(img_context_tokens.to(dtype))
            mask_list.append(img_keep_mask.bool().to(device=device))

        # 站点 context tokens（进出站导航）
        station_context_tokens, station_keep_mask = self.encode_station_context_tokens(action_input)
        if station_context_tokens is not None:
            context_tokens_list.append(station_context_tokens.to(dtype))
            mask_list.append(station_keep_mask.to(device=device))

        # 占位 summary token（[B, 1, D] 零向量），仅承载 batch/device/dtype。
        vl_summary_tokens = torch.zeros(B, 1, D, dtype=dtype, device=device)

        if len(context_tokens_list) == 0:
            # 无任何结构化 token：空 context，DiT 会退化为纯 self-attention。
            context_tokens = torch.zeros(B, 0, D, dtype=dtype, device=device)
            return vl_summary_tokens, context_tokens, None

        context_tokens = (
            torch.cat(context_tokens_list, dim=1)
            if len(context_tokens_list) > 1 else context_tokens_list[0]
        )
        context_key_padding_mask = (
            torch.cat(mask_list, dim=1) if len(mask_list) > 1 else mask_list[0]
        )
        return vl_summary_tokens, context_tokens, context_key_padding_mask

    def _repeat_action_input(self, action_input: BatchFeature, repeats: int) -> BatchFeature:
        repeated = {}
        for key, value in action_input.items():
            repeated[key] = value.repeat_interleave(repeats, 0) if isinstance(value, torch.Tensor) else value
        return BatchFeature(data=repeated)

    def _init_flow_sampler(self, cfg: FlowConfig):
        """Initializes components required for Flow Matching."""
        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)
        self.num_timestep_buckets = cfg.num_timestep_buckets

    def _init_ddpm_sampler(self, cfg: DDPMConfig):
        """Initializes buffers required for DDPM, using original naming."""
        ddpm_betas = self.cosine_beta_schedule(cfg.num_train_timesteps)
        self.register_buffer('ddpm_betas', ddpm_betas)

        ddpm_alphas = 1.0 - ddpm_betas
        self.register_buffer('ddpm_alphas', ddpm_alphas)

        ddpm_alphas_cumprod = torch.cumprod(ddpm_alphas, dim=0)
        self.register_buffer('ddpm_alphas_cumprod', ddpm_alphas_cumprod)

        ddpm_alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), ddpm_alphas_cumprod[:-1]])
        self.register_buffer('ddpm_alphas_cumprod_prev', ddpm_alphas_cumprod_prev)

        self.register_buffer('ddpm_sqrt_alphas_cumprod', torch.sqrt(ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod))
        self.register_buffer('ddpm_sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / ddpm_alphas_cumprod - 1.0))

        ddpm_var = ddpm_betas * (1.0 - ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod)
        self.register_buffer('ddpm_var', ddpm_var)
        self.register_buffer('ddpm_logvar_clipped', torch.log(ddpm_var.clamp(min=1e-20)))
        
        self.register_buffer('ddpm_mu_coef1', ddpm_betas * torch.sqrt(ddpm_alphas_cumprod_prev) / (1.0 - ddpm_alphas_cumprod))
        self.register_buffer('ddpm_mu_coef2', (1.0 - ddpm_alphas_cumprod_prev) * torch.sqrt(ddpm_alphas) / (1.0 - ddpm_alphas_cumprod))

    def _init_ddim_sampler(self, cfg: DDIMConfig):
        """Initializes buffers required for DDIM sampling, using original naming."""
        self._init_ddpm_sampler(DDPMConfig(num_train_timesteps=cfg.num_train_timesteps))

        self.eta = EtaFixed(base_eta=1.0).to(self.device)
        for param in self.eta.parameters():
            param.requires_grad = False
        ddim_steps = self.config.num_inference_steps
        self.ddim_steps = ddim_steps
        self.ft_denoising_steps = ddim_steps
        ddim_eta = cfg.ddim_eta 

        self.ddpm_num_train_timesteps = cfg.num_train_timesteps
        step_ratio = self.ddpm_num_train_timesteps // ddim_steps
        ddim_t = torch.arange(0, ddim_steps) * step_ratio
        self.register_buffer('ddim_t_schedule', ddim_t.long()) # Die originalen Zeitpunkte

        ddim_alphas = self.ddpm_alphas_cumprod[self.ddim_t_schedule].clone().to(torch.float32)
        ddim_alphas_prev = torch.cat([
            torch.tensor([1.0], dtype=torch.float32),
            self.ddpm_alphas_cumprod[self.ddim_t_schedule[:-1]]
        ])
        ddim_sqrt_one_minus_alphas = (1.0 - ddim_alphas) ** 0.5

        ddim_sigmas = ddim_eta * (
            (1 - ddim_alphas_prev) / (1 - ddim_alphas) *
            (1 - ddim_alphas / ddim_alphas_prev)
        )**0.5

        # Flip all for sampling order (T -> 0)
        def flip_buffer(name, tensor):
            self.register_buffer(name, torch.flip(tensor, [0]))

        flip_buffer('ddim_t', self.ddim_t_schedule)
        flip_buffer('ddim_alphas', ddim_alphas)
        flip_buffer('ddim_alphas_sqrt', torch.sqrt(ddim_alphas))
        flip_buffer('ddim_alphas_prev', ddim_alphas_prev)
        flip_buffer('ddim_sqrt_one_minus_alphas', ddim_sqrt_one_minus_alphas)
        flip_buffer('ddim_sigmas', ddim_sigmas)

    def _init_grpo(self, cfg: GRPOConfig):
        """Initializes components and hyperparameters for GRPO training."""
        self.denoised_clip_value = cfg.denoised_clip_value
        self.eval_randn_clip_value = cfg.eval_randn_clip_value
        self.randn_clip_value = cfg.randn_clip_value
        self.final_action_clip_value = cfg.final_action_clip_value
        self.eps_clip_value = cfg.eps_clip_value
        self.eval_min_sampling_denoising_std = cfg.eval_min_sampling_denoising_std
        self.min_sampling_denoising_std = cfg.min_sampling_denoising_std
        self.min_logprob_denoising_std = cfg.min_logprob_denoising_std
        self.clip_advantage_lower_quantile = cfg.clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = cfg.clip_advantage_upper_quantile
        self.gamma_denoising = cfg.gamma_denoising
        
        self.metric_cache_loader = MetricCacheLoader(cfg.metric_cache_path)
        proposal_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        self.simulator = PDMSimulator(proposal_sampling)
        self.train_scorer = PDMScorer(
            proposal_sampling,
            cfg.scorer_config,
            vehicle_parameters=cfg.vehicle_parameters,
        )
        
        try:
            state_dict = torch.load(
                cfg.reference_policy_checkpoint,
                map_location="cpu",
                weights_only=False,
            )["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in state_dict.items():
                if k.startswith("agent.action_head."):
                    k2 = k[len("agent.action_head."):]
                else:
                    k2 = k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
                else:
                    print(f"Skip loading '{k}' → '{k2}' (checkpoint shape {tuple(v.shape)} vs model shape {tuple(model_dict.get(k2, v).shape)})")
            self.load_state_dict(filtered_ckpt, strict=True)
        except FileNotFoundError:
            print(f"Warning: GRPO checkpoint not found at {cfg.reference_policy_checkpoint}. Skipping loading.")
        
        self.old_policy = copy.deepcopy(self)
        self.old_policy.eval()
        for param in self.old_policy.parameters():
            param.requires_grad = False

    @staticmethod
    def cosine_beta_schedule(timesteps: int, s: float = 0.008, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Calculates a cosine noise schedule as proposed in the iDDPM paper.
        
        This method is static as it does not depend on the instance's state.
        """
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
        return torch.tensor(betas_clipped, dtype=dtype)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extracts values from tensor `a` at indices `t` and reshapes them
        to be broadcastable with a tensor of shape `x_shape`.
        
        This method is static as it does not depend on the instance's state.
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    @staticmethod
    def make_timesteps(batch_size: int, i: int, device: torch.device) -> torch.Tensor:
        """
        Creates a tensor of a constant value `i` for a given batch size and device.
        
        This method is static as it does not depend on the instance's state.
        """
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        return t


    def set_frozen_modules_to_eval_mode(self):
        """
        Sets frozen parts of the model to evaluation mode during training.
        This is necessary to disable behaviors like dropout in the frozen layers.
        """
        if self.training:

            if not self.config.tune_projector:
                self.ego_status_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.use_vlm:
                    self.vl_summary_projector.eval()
                    self.vl_context_first_projector.eval()
                    self.vl_context_last_projector.eval()
                    self.vl_context_gate.eval()
                    self.fusion_projector.eval()
                if self.object_encoder is not None:
                    self.object_encoder.eval()
                if self.image_encoder is not None:
                    self.image_encoder.eval()
                if self.station_encoder is not None:
                    self.station_encoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            
            if not self.config.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        """Samples time for training based on the sampling method."""
        if self.config.sampling_method == 'flow':
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            return (self.config.flow_cfg.noise_s - sample) / self.config.flow_cfg.noise_s
        elif self.config.sampling_method in ['ddpm', 'ddim']:
            return torch.randint(0, self.ddpm_num_train_timesteps, (batch_size,), device=device).long()
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

    def build_action_latents(
        self,
        vl_summary_tokens: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        """Build DiT action tokens without directly injecting historical trajectory."""
        if not self.config.use_vlm:
            # use_vlm=False：不拼 VLM summary（fusion_projector 未构造），action latent 即
            # action_features 本身（已是 input_embedding_dim 宽）。
            return action_features
        vl_summary = vl_summary_tokens.mean(1).unsqueeze(1).expand(
            -1, action_features.shape[1], -1
        )
        return self.fusion_projector(torch.cat((vl_summary, action_features), dim=2))


    def _predict_flow(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        vl_summary_tokens: torch.Tensor,
        vl_context_tokens: torch.Tensor,
        ego_status_features: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicts the flow velocity for a normalized action state."""
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if hasattr(self, "position_embedding"):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + self.position_embedding(pos_ids)

        action_latents = self.build_action_latents(vl_summary_tokens, action_features)
        model_output = self.model(
            action_latents=action_latents,
            context_tokens=vl_context_tokens,
            condition_features=ego_status_features,
            timesteps=t,
            context_key_padding_mask=context_key_padding_mask,
        )
        pred = self.action_decoder(model_output)
        return pred.chunk(2, dim=-1)[0] if self.config.flow_cfg.mean_variance_net else pred

    def _flow_step_index(self, step: int, num_steps: int) -> int:
        return int(step / max(num_steps, 1) * self.config.flow_cfg.num_timestep_buckets)

    def _flow_transition_std(
        self,
        x: torch.Tensor,
        dt: float,
        *,
        deterministic: bool,
        for_logprob: bool = False,
    ) -> torch.Tensor:
        # Deterministic *sampling* (eval / get_action) collapses to zero std. The
        # logprob path (for_logprob=True) must keep a non-degenerate std even under
        # deterministic=False, otherwise Normal(mean, 0) is undefined.
        if deterministic and not for_logprob:
            return torch.zeros_like(x)

        # Plan B: sampling and logprob share ONE floor (sde_min_std), so the
        # transition that drew x_{t+1} and the Normal that scores it are identical.
        # Do NOT route the DDIM-only min_sampling/min_logprob floors through here;
        # those still govern the DDIM/DDPM branches.
        sde_min_std = self.config.flow_cfg.sde_min_std
        std_value = max(self.config.flow_cfg.sde_noise_scale * math.sqrt(dt), sde_min_std)
        return torch.full_like(x, std_value)

    def _flow_transition_mean_std(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        vl_summary_tokens: torch.Tensor,
        vl_context_tokens: torch.Tensor,
        ego_status_features: torch.Tensor,
        *,
        deterministic: bool,
        for_logprob: bool = False,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_flow = self._predict_flow(
            x,
            t,
            vl_summary_tokens,
            vl_context_tokens,
            ego_status_features,
            context_key_padding_mask=context_key_padding_mask,
        )
        mean = x + dt * pred_flow.to(dtype=x.dtype)
        std = self._flow_transition_std(
            x,
            dt,
            deterministic=deterministic,
            for_logprob=for_logprob,
        )
        return mean, std


    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        vl_summary_tokens: torch.Tensor,
        vl_context_tokens: torch.Tensor,
        ego_status_features: torch.Tensor,
        deterministic: bool = True,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the mean and log variance of the reverse process p(x_{t-1} | x_t).
        Also returns the predicted x0.
        """
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(model_dtype)
        action_features = self.action_encoder(x, t)
        if hasattr(self, 'position_embedding'):
            pos_ids = torch.arange(action_features.shape[1], device=x.device)
            action_features = action_features + self.position_embedding(pos_ids)

        action_latents = self.build_action_latents(vl_summary_tokens, action_features)

        model_output = self.model(
            action_latents=action_latents,
            context_tokens=vl_context_tokens,
            condition_features=ego_status_features,
            timesteps=t,
            context_key_padding_mask=context_key_padding_mask,
        )
        pred_noise = self.action_decoder(model_output)

        if self.config.sampling_method == 'ddpm':
            x_recon = self.extract(self.ddpm_sqrt_recip_alphas_cumprod, t, x.shape) * x - \
                      self.extract(self.ddpm_sqrt_recipm1_alphas_cumprod, t, x.shape) * pred_noise
        elif self.config.sampling_method == 'ddim':
            alpha_t = self.extract(self.ddim_alphas, index, x.shape)
            sqrt_one_minus_alpha_t = self.extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
            x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t**0.5)
        else:
             raise ValueError(f"p_mean_variance not supported for method: {self.config.sampling_method}")

        denoised_clip_value = getattr(self, 'denoised_clip_value', 1.0)
        x_recon.clamp_(-denoised_clip_value, denoised_clip_value)

        if self.config.sampling_method == 'ddpm':
            model_mean = self.extract(self.ddpm_mu_coef1, t, x.shape) * x_recon + \
                         self.extract(self.ddpm_mu_coef2, t, x.shape) * x
            model_log_variance = self.extract(self.ddpm_logvar_clipped, t, x.shape)
        elif self.config.sampling_method == 'ddim':
            alpha_prev = self.extract(self.ddim_alphas_prev, index, x.shape)
            
            pred_noise = (x - (alpha_t**0.5) * x_recon) / sqrt_one_minus_alpha_t

            eps_clip_value = getattr(self, 'eps_clip_value', None)
            if eps_clip_value is not None:
                pred_noise.clamp_(-eps_clip_value, eps_clip_value)

            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(x).unsqueeze(1)

            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)

            pred_dir_xt = (1.0 - alpha_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
            model_mean = (alpha_prev**0.5) * x_recon + pred_dir_xt
            model_log_variance = torch.log(sigma**2 + 1e-20)

        return model_mean, model_log_variance, x_recon
    # 只做扩散训练，输入是 VLM hidden states + conditioning；安全监督单独传入。
    def forward(
        self,
        first_hidden_state: Optional[torch.Tensor],
        last_hidden_state: Optional[torch.Tensor],
        action_input: BatchFeature,
        safety_targets: Optional[BatchFeature] = None,
    ) -> BatchFeature:
        """Compute diffusion and optional explicit safety losses."""
        vl_summary_tokens, vl_context_tokens, ctx_mask = self.build_context_tokens(
            first_hidden_state, last_hidden_state, action_input
        )
        ego_status_features = self.ego_status_encoder(action_input.status_feature)
        gt_actions = self.norm_odo(action_input.action)

        if self.config.sampling_method == "flow":
            noise = torch.randn_like(gt_actions)
            t_cont = self.sample_time(
                gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype
            )
            t_cont_reshaped = t_cont[:, None, None]
            noisy_actions = (1 - t_cont_reshaped) * noise + t_cont_reshaped * gt_actions
            velocity_target = gt_actions - noise
            t_discrete = (t_cont * self.num_timestep_buckets).long()
            pred_velocity = self._predict_flow(
                noisy_actions,
                t_discrete,
                vl_summary_tokens,
                vl_context_tokens,
                ego_status_features,
                context_key_padding_mask=ctx_mask,
            )
            diffusion_loss = F.mse_loss(pred_velocity, velocity_target, reduction="mean")
            x0_hat = noisy_actions + (1.0 - t_cont_reshaped) * pred_velocity
            safety_sample_weight = t_cont.square()
        else:
            noise = torch.randn_like(gt_actions)
            t_discrete = self.sample_time(
                gt_actions.shape[0], device=gt_actions.device, dtype=gt_actions.dtype
            )
            sqrt_alpha = self.extract(
                self.ddpm_sqrt_alphas_cumprod, t_discrete, gt_actions.shape
            )
            sqrt_one_minus_alpha = self.extract(
                self.ddpm_sqrt_one_minus_alphas_cumprod, t_discrete, gt_actions.shape
            )
            noisy_actions = sqrt_alpha * gt_actions + sqrt_one_minus_alpha * noise

            action_features = self.action_encoder(noisy_actions, t_discrete)
            if hasattr(self, "position_embedding"):
                pos_ids = torch.arange(action_features.shape[1], device=gt_actions.device)
                action_features += self.position_embedding(pos_ids)
            action_latents = self.build_action_latents(vl_summary_tokens, action_features)
            model_output = self.model(
                action_latents=action_latents,
                context_tokens=vl_context_tokens,
                condition_features=ego_status_features,
                timesteps=t_discrete,
                context_key_padding_mask=ctx_mask,
            )
            pred_noise = self.action_decoder(model_output)
            diffusion_loss = F.mse_loss(pred_noise, noise, reduction="mean")
            # epsilon parameterization: reconstruct normalized clean trajectory.
            x0_hat = (noisy_actions - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
            safety_sample_weight = sqrt_alpha.square().reshape(gt_actions.shape[0])

        if not self.config.safety_snr_weighting:
            safety_sample_weight = torch.ones_like(safety_sample_weight)

        zero = diffusion_loss.new_zeros(())
        auxiliary_losses = {
            "collision_loss": zero,
            "offroad_loss": zero,
            "route_loss": zero,
        }
        if self.config.use_explicit_safety_loss:
            if safety_targets is None:
                raise ValueError(
                    "Explicit safety loss requires a separate safety_targets batch"
                )
            missing = [key for key in SAFETY_TARGET_KEYS if key not in safety_targets]
            if missing:
                raise KeyError(f"Missing explicit safety target keys: {missing}")
            x0_phys = self.denorm_odo(x0_hat)
            auxiliary_losses = self._compute_auxiliary_losses(
                x0_phys, action_input, safety_targets, safety_sample_weight
            )

        total_loss = (
            diffusion_loss
            + self.config.lambda_collision * auxiliary_losses["collision_loss"]
            + self.config.lambda_offroad * auxiliary_losses["offroad_loss"]
            + self.config.lambda_route * auxiliary_losses["route_loss"]
        )
        return BatchFeature(
            data={
                "loss": total_loss,
                "diffusion_loss": diffusion_loss,
                **auxiliary_losses,
            }
        )

    def get_action(
        self,
        first_hidden_state: torch.Tensor,
        last_hidden_state: torch.Tensor,
        action_input: BatchFeature,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_init_noise: bool = False,
        return_step_noises: bool = False,
        step_noises: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """
        Generates action trajectories via the configured sampling method.

        This method strictly preserves the original logic for each sampler,
        including specific clipping and noise handling for DDPM and DDIM.

        Args:
            first_hidden_state (torch.Tensor): Early vision-language hidden states from the backbone.
            last_hidden_state (torch.Tensor): Final vision-language hidden states from the backbone.
            action_input (BatchFeature): Input containing ego status conditioning.
            init_actions (Optional[torch.Tensor]): An initial trajectory to start
                the denoising from. If None, starts from pure noise.
            deterministic (bool): If True, DDIM sampling will be deterministic (eta=0).
            return_init_noise (bool): If True, also return the initial sampling noise
                (the pure-noise starting point, or the provided init_actions) under
                the "init_noise" key. Zero impact on the sampled trajectory: the noise
                is only observed, not altered. Deployment can reuse it as init_actions
                to reproduce the sampling start point.
            return_step_noises (bool): If True (DDIM only), also return every per-step
                sampling noise (post-clip, exactly as consumed) stacked as
                [num_steps, B, action_horizon, action_dim] under the "step_noises"
                key. Pure observation: cloned after the same clamp the update uses,
                before it multiplies std. Zero impact on pred_traj. Under
                deterministic=False these per-step noises plus init_noise fully
                determine the trajectory, so replaying them reproduces it exactly.
            step_noises (Optional[torch.Tensor]): If provided (DDIM only), inject these
                per-step noises instead of sampling torch.randn each step. Shape must be
                [num_steps, B, action_horizon, action_dim] (or broadcastable). Used to
                reproduce a previously-saved sampling run: pass the saved init_actions
                and step_noises to replay the exact trajectory. Default None keeps the
                historical randn behavior unchanged (zero impact).

        Returns:
            BatchFeature: A batch containing the final predicted trajectory.
        """
        vl_summary_tokens, vl_context_tokens, ctx_mask = self.build_context_tokens(
            first_hidden_state, last_hidden_state, action_input
        )

        ego_embeds = self.ego_status_encoder(
            action_input.status_feature
        )

        B, D = vl_summary_tokens.shape[0], self.config.action_dim
        device, dtype = vl_summary_tokens.device, vl_summary_tokens.dtype

        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )
        # 采样起点快照：在去噪循环覆盖 current_actions 之前保存，供部署端复现推理。
        init_noise_snapshot = current_actions.clone() if return_init_noise else None
        # 每步噪声观测/注入（仅 DDIM）。观测：累积每步实际消费的 noise_sample（post-clip）。
        # 注入：外部传入 step_noises 时逐步替代 torch.randn，用于逐点复现已保存的采样过程。
        step_noise_snapshots = [] if return_step_noises else None

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = self._flow_step_index(step, self.config.num_inference_steps)
                t = torch.full((B,), idx, device=device, dtype=torch.long)
                mean, std = self._flow_transition_mean_std(
                    current_actions,
                    t,
                    dt,
                    vl_summary_tokens,
                    vl_context_tokens,
                    ego_embeds,
                    deterministic=True,
                    context_key_padding_mask=ctx_mask,
                )
                noise_sample = torch.randn_like(current_actions)
                if hasattr(self, 'eval_randn_clip_value') and self.eval_randn_clip_value is not None:
                    noise_sample.clamp_(-self.eval_randn_clip_value, self.eval_randn_clip_value)
                current_actions = mean + std * noise_sample

        elif self.config.sampling_method == 'ddpm':
            step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
            timesteps_to_iterate = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            
            for i, t_int in enumerate(timesteps_to_iterate):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions,
                    t_batch,
                    index_batch,
                    vl_summary_tokens,
                    vl_context_tokens,
                    ego_embeds,
                    deterministic,
                    context_key_padding_mask=ctx_mask,
                )

                noise_sample = torch.randn_like(current_actions)
                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                if t_int == 0:
                    std.zero_()
                else:
                    std = torch.clamp(std, min=1e-3)

                if hasattr(self, 'eval_randn_clip_value') and self.eval_randn_clip_value is not None:
                    noise_sample.clamp_(-self.eval_randn_clip_value, self.eval_randn_clip_value)

                current_actions = mean + std * noise_sample

                if hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None and i == len(timesteps_to_iterate) - 1:
                    current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)

        elif self.config.sampling_method == 'ddim':
            eval_min_sampling_denoising_std = getattr(self, 'eval_min_sampling_denoising_std', 0.0001)
            eval_randn_clip_value = getattr(self, 'eval_randn_clip_value', 1.0)
            for i in range(self.ddim_steps):
                t_batch = self.make_timesteps(B, self.ddim_t[i], device)
                index_batch = self.make_timesteps(B, i, device)

                mean, logvar, _ = self.p_mean_variance(
                    current_actions,
                    t_batch,
                    index_batch,
                    vl_summary_tokens,
                    vl_context_tokens,
                    ego_embeds,
                    deterministic,
                    context_key_padding_mask=ctx_mask,
                )

                std = torch.exp(0.5 * logvar)

                std = std.to(dtype)

                # 注入端：若提供 step_noises，则第 i 步用外部噪声替代 randn，逐点复现已存采样。
                # 否则保持历史行为，每步现采 randn（默认路径，零影响）。
                if step_noises is not None:
                    noise_sample = step_noises[i].to(device=current_actions.device, dtype=dtype)
                else:
                    noise_sample = torch.randn_like(current_actions)

                if deterministic:
                    std.zero_()
                else:
                    std = std.clamp(min=eval_min_sampling_denoising_std)

                noise_sample.clamp_(-eval_randn_clip_value, eval_randn_clip_value)

                # 观测端：clamp 之后、更新之前 clone —— 此刻 noise_sample 正是本步实际消费的值。
                # 只读不改，对 current_actions 更新零影响；deterministic=True 时 std=0，噪声不参与
                # 输出但仍如实记录本步随机源。
                if step_noise_snapshots is not None:
                    step_noise_snapshots.append(noise_sample.clone())

                current_actions = mean + std * noise_sample
        else:
            raise ValueError(f"Unsupported sampling method: {self.config.sampling_method}")

        final_action_clip_value = getattr(self, 'final_action_clip_value', 1.0)
        if final_action_clip_value is not None:
            current_actions.clamp_(-final_action_clip_value, final_action_clip_value)

        final_actions = self.denorm_odo(current_actions)

        data = {"pred_traj": final_actions}
        if return_init_noise and init_noise_snapshot is not None:
            data["init_noise"] = init_noise_snapshot
        # 每步噪声堆叠为 [num_steps, B, action_horizon, action_dim]（仅 DDIM 有值；其它采样器
        # 未累积则为空列表，不写出该键）。
        if step_noise_snapshots:
            data["step_noises"] = torch.stack(step_noise_snapshots, dim=0)
        return BatchFeature(data=data)

    def sample_chain(
        self,
        first_hidden_state: torch.Tensor,
        last_hidden_state: torch.Tensor,
        ego_status_features: torch.Tensor,
        init_actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
    ):
        vl_summary_tokens, vl_context_tokens, ctx_mask = self.build_context_tokens(
            first_hidden_state, last_hidden_state, action_input
        )
        # 从 vl_summary_tokens 取 batch/device/dtype：use_vlm=True 时它是投影后的 summary，
        # use_vlm=False 时是 [B,1,D] 零占位张量（见 _build_context_tokens_no_vlm），故无需
        # 触碰 hidden state（use_vlm=False 下为 None）。数值上与原先取 last_hidden_state 一致。
        B, D = vl_summary_tokens.shape[0], self.config.action_dim
        device, dtype = vl_summary_tokens.device, vl_summary_tokens.dtype

        ego_status_features = self.ego_status_encoder(
            ego_status_features
        )

        current_actions = init_actions if init_actions is not None else torch.randn(
            (B, self.config.action_horizon, D), device=device, dtype=dtype
        )
        denoising_chain = [current_actions.clone()]

        if self.config.sampling_method == 'flow':
            dt = 1.0 / self.config.num_inference_steps
            for step in range(self.config.num_inference_steps):
                idx = self._flow_step_index(step, self.config.num_inference_steps)
                t_batch = torch.full((B,), idx, device=device, dtype=torch.long)
                mean, std = self._flow_transition_mean_std(
                    current_actions,
                    t_batch,
                    dt,
                    vl_summary_tokens,
                    vl_context_tokens,
                    ego_status_features,
                    deterministic=deterministic,
                    context_key_padding_mask=ctx_mask,
                )
                noise_sample = torch.randn_like(current_actions)
                if hasattr(self, 'randn_clip_value') and self.randn_clip_value is not None:
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)
                current_actions = mean + std * noise_sample
                # Align the rollout terminal with eval get_action(): clip ONLY the
                # last Euler step into [-clip, clip] before recording it, mirroring
                # the DDIM/DDPM branch below (and the final clamp at the end of
                # get_action). Clipping before append keeps chains[-1] -- scored by
                # get_logprobs -- identical to final_actions scored by reward_fn, and
                # stops accumulated Euler noise from drifting the terminal trajectory
                # outside the normalized [-1, 1] range that eval always clamps to.
                if (
                    step == self.config.num_inference_steps - 1
                    and hasattr(self, 'final_action_clip_value')
                    and self.final_action_clip_value is not None
                ):
                    current_actions = current_actions.clamp_(
                        -self.final_action_clip_value, self.final_action_clip_value
                    )
                denoising_chain.append(current_actions.clone())

        elif self.config.sampling_method in ['ddpm', 'ddim']:
            if self.config.sampling_method == 'ddpm':
                step_size = self.config.ddpm_cfg.num_train_timesteps // self.config.num_inference_steps
                timesteps = list(reversed(range(0, self.config.ddpm_cfg.num_train_timesteps, step_size)))
            else:
                timesteps = self.ddim_t

            for i, t_int in enumerate(timesteps):
                t_batch = self.make_timesteps(B, t_int, device)
                index_batch = self.make_timesteps(B, i, device) if self.config.sampling_method == 'ddim' else t_batch
                mean, logvar, _ = self.p_mean_variance(
                    current_actions,
                    t_batch,
                    index_batch,
                    vl_summary_tokens,
                    vl_context_tokens,
                    ego_status_features,
                    deterministic,
                    context_key_padding_mask=ctx_mask,
                )

                std = torch.exp(0.5 * logvar).to(dtype)
                noise_sample = torch.randn_like(current_actions)

                if self.config.sampling_method == 'ddim':
                    if deterministic:
                        std.zero_()
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)
                else:
                    if deterministic and t_int == 0:
                        std = torch.zeros_like(std)
                    elif deterministic:
                        std = std.clamp(min=1e-3)
                    else:
                        std = std.clamp(min=self.min_sampling_denoising_std)

                if hasattr(self, 'randn_clip_value') and self.randn_clip_value is not None:
                    noise_sample = noise_sample.clamp_(-self.randn_clip_value, self.randn_clip_value)

                current_actions = mean + std * noise_sample

                if i == len(timesteps) - 1 and hasattr(self, 'final_action_clip_value') and self.final_action_clip_value is not None:
                    current_actions = current_actions.clamp_(-self.final_action_clip_value, self.final_action_clip_value)

                denoising_chain.append(current_actions.clone())
        else:
            raise ValueError(f'Unsupported sampling method: {self.config.sampling_method}')

        final_actions = self.denorm_odo(current_actions)
        chain_tensor = torch.stack(denoising_chain, dim=1)

        return chain_tensor.detach(), final_actions.detach()

    def get_logprobs(
        self,
        first_hidden_state: torch.Tensor,
        last_hidden_state: torch.Tensor,
        ego_status_features: torch.Tensor,
        chains: torch.Tensor,
        deterministic: bool = False,
        action_input: Optional[BatchFeature] = None,
    ) -> torch.Tensor:
        """Calculates the log probability of a full denoising chain."""
        B, K1, H, D = chains.shape
        num_denoising_steps = K1 - 1

        vl_summary_tokens, vl_context_tokens, ctx_mask = self.build_context_tokens(
            first_hidden_state, last_hidden_state, action_input
        )

        ego_status_features = self.ego_status_encoder(
            ego_status_features
        )

        conditioning_embeds = {
            'vl_summary_tokens': vl_summary_tokens,
            'vl_context_tokens': vl_context_tokens,
            'ego_status_features': ego_status_features,
        }

        batched_conditioning = {}
        for key, value in conditioning_embeds.items():
            batched_conditioning[key] = value.unsqueeze(1).repeat(
                1, num_denoising_steps, *(1,) * (value.ndim - 1)
            ).flatten(0, 1)

        # Repeat mask to match (B * num_denoising_steps) batch dimension.
        if ctx_mask is not None:
            batched_ctx_mask = ctx_mask.unsqueeze(1).repeat(1, num_denoising_steps, 1).flatten(0, 1)
        else:
            batched_ctx_mask = None

        if self.config.sampling_method == 'flow':
            dt = 1.0 / num_denoising_steps
            t_single = torch.tensor(
                [self._flow_step_index(step, num_denoising_steps) for step in range(num_denoising_steps)],
                device=chains.device,
                dtype=torch.long,
            )
            t_batch = t_single.repeat(B)

            x_t = chains[:, :-1].reshape(-1, H, D)
            x_t_next = chains[:, 1:].reshape(-1, H, D)
            mean, std = self._flow_transition_mean_std(
                x_t,
                t_batch,
                dt,
                batched_conditioning['vl_summary_tokens'],
                batched_conditioning['vl_context_tokens'],
                batched_conditioning['ego_status_features'],
                deterministic=deterministic,
                for_logprob=True,
                context_key_padding_mask=batched_ctx_mask,
            )
            dist = Normal(mean, std)
            return dist.log_prob(x_t_next)

        if self.config.sampling_method == 'ddim':
            t_single = self.ddim_t[-num_denoising_steps:]
            indices_single = torch.arange(
                start=self.ddim_steps - num_denoising_steps,
                end=self.ddim_steps,
                device=chains.device
            )
            indices_batch = indices_single.repeat(B)
        else:
            t_single = torch.arange(start=K1 - 2, end=-1, step=-1, device=chains.device)
            indices_batch = None

        t_batch = t_single.repeat(B)

        x_t = chains[:, :-1].reshape(-1, H, D)
        x_t_minus_1 = chains[:, 1:].reshape(-1, H, D)

        mean, logvar, _ = self.p_mean_variance(
            x_t, t_batch, indices_batch,
            batched_conditioning['vl_summary_tokens'],
            batched_conditioning['vl_context_tokens'],
            batched_conditioning['ego_status_features'],
            deterministic=deterministic,
            context_key_padding_mask=batched_ctx_mask,
        )

        std = torch.exp(0.5 * logvar).clamp(min=self.min_logprob_denoising_std)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x_t_minus_1)
        
        return log_prob

    def forward_grpo(
        self,
        first_hidden_state: torch.Tensor,
        last_hidden_state: torch.Tensor,
        action_input: BatchFeature,
        tokens_list,
        sample_time: int = 8,
        deterministic=False,
        use_bc_loss: bool = True
    ) -> BatchFeature:
        """Computes the Diffusion-GRPO loss."""
        self.set_frozen_modules_to_eval_mode()
        # Plan A: force the ego-status encoder to eval during GRPO. The attention
        # variant applies state_dropout=0.75 in train mode, which would draw
        # different masks in sample_chain vs get_logprobs and break the mean
        # consistency the logprob math relies on. No-op for the default MLP encoder
        # (LayerNorm only), so this is always safe to call here.
        self.ego_status_encoder.eval()
        # 与 ego_status_encoder 同理：obj encoder 的 token_dropout 在 train 模式下会在
        # sample_chain / get_logprobs 抽到不同 mask，破坏 logprob 一致性。GRPO 全程切 eval，
        # 让 rollout 与打分用同一份确定的 obj context。对 IL 无影响（IL 不走这里）。
        if self.object_encoder is not None:
            self.object_encoder.eval()
        # 图像 encoder 的 token_dropout 同理会破坏 rollout/logprob 一致性；GRPO 全程切 eval。
        # ResNet 的 BatchNorm 在 eval 下用 running stats（而非 batch 统计），rollout 与
        # logprob 用同一份确定的图像 context，避免两次前向 BN 统计不同导致 mean 漂移。
        if self.image_encoder is not None:
            self.image_encoder.eval()
        # 站点编码器无 dropout（仅 LayerNorm+MLP），切 eval 对数值无影响，保持与其他 context
        # 编码器一致的 GRPO 惯例（rollout/logprob 两次前向条件一致）。
        if self.station_encoder is not None:
            self.station_encoder.eval()
        B = last_hidden_state.shape[0] if self.config.use_vlm else action_input.status_feature.shape[0]
        G = sample_time

        if self.config.use_vlm:
            first_hidden_state_rep = first_hidden_state.repeat_interleave(G, 0)
            last_hidden_state_rep = last_hidden_state.repeat_interleave(G, 0)
        else:
            # use_vlm=False：无 hidden state，占位 None，下游 build_context_tokens 会忽略。
            first_hidden_state_rep = None
            last_hidden_state_rep = None
        status_feature_rep = action_input.status_feature.repeat_interleave(G, 0)
        action_input_rep = self._repeat_action_input(action_input, G)

        # obj 条件必须和 first/last/status 一样按 G 复制，rollout 与 logprob 才条件一致。
        obj_inputs = self._extract_obj_inputs(action_input)
        if obj_inputs is not None:
            obj_inputs_rep = {k: v.repeat_interleave(G, 0) for k, v in obj_inputs.items()}
        else:
            obj_inputs_rep = None

        # Plan C: the rollout only supplies fixed targets (chains) and rewards
        # (trajs); gradients flow solely through get_logprobs further down, which
        # re-runs the forward with grad. Building the graph here would just waste
        # memory, so do the rollout under no_grad. The teacher rollout below is
        # already wrapped the same way.
        with torch.no_grad():
            chains, trajs = self.sample_chain(
                first_hidden_state_rep,
                last_hidden_state_rep,
                status_feature_rep,
                deterministic=False,
                action_input=action_input_rep,
            )

        tokens_rep = [tok for tok in tokens_list for _ in range(G)]
        unique_tokens = set(tokens_list)
        metric_cache = {}
        for token in unique_tokens:
            path = self.metric_cache_loader.metric_cache_paths[token]
            with lzma.open(path, 'rb') as f:
                metric_cache[token] = pickle.load(f)
        rewards = self.reward_fn(trajs, tokens_rep, metric_cache)

        rewards_matrix = rewards.view(B, G)
        mean_r = rewards_matrix.mean(dim=1, keepdim=True)
        std_r = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_matrix - mean_r) / std_r).view(-1).detach()
        
        adv_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        adv_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=adv_min, max=adv_max)

        # Plan D: within-group reward spread is what drives the advantage; if it
        # collapses to ~0 the group is degenerate and GRPO gets no signal.
        # advantage_std is taken post-clamp so it reflects what actually weights
        # the policy loss.
        reward_std = rewards_matrix.std(dim=1).mean().detach()
        advantage_std = advantages.std().detach()

        num_denoising_steps = chains.shape[1] - 1
        denoising_indices = torch.arange(num_denoising_steps, device=advantages.device)
        discount = (self.gamma_denoising ** (num_denoising_steps - denoising_indices - 1))
    
        
        adv_steps = advantages.view(B, G, 1).expand(-1, -1, num_denoising_steps)  
        discount = discount.view(1, 1, num_denoising_steps).expand(B, G, num_denoising_steps)  
        adv_weighted_flat = (adv_steps * discount).reshape(-1)             

        raw_log_probs = self.get_logprobs(
            first_hidden_state_rep,
            last_hidden_state_rep,
            status_feature_rep,
            chains,
            deterministic=False,
            action_input=action_input_rep,
        )
        # Plan D: diagnose the raw (pre-clamp) logprobs. A high clamp hit rate
        # means the [-5, 2] guard is silently zeroing gradients for those steps.
        logprob_clamp_lo, logprob_clamp_hi = -5.0, 2.0
        logprob_clamp_hit_rate = (
            (raw_log_probs <= logprob_clamp_lo) | (raw_log_probs >= logprob_clamp_hi)
        ).float().mean().detach()
        logprob_mean = raw_log_probs.mean().detach()
        log_probs = raw_log_probs.clamp(min=logprob_clamp_lo, max=logprob_clamp_hi).mean(dim=[1, 2])

        policy_loss = -torch.mean(log_probs * adv_weighted_flat)
        total_loss = policy_loss

        bc_loss = 0.0
        if use_bc_loss:
            with torch.no_grad():
                # Step 3 switch: bc_teacher_deterministic=True makes the reference
                # policy roll out a clean ODE trajectory (std=0), so BC anchors to the
                # deterministic reference instead of a +-noise SDE rollout. Default
                # False keeps the Step 1+2 stochastic teacher.
                teacher_chains, _ = self.old_policy.sample_chain(
                    first_hidden_state,
                    last_hidden_state,
                    action_input.status_feature,
                    deterministic=self.config.grpo_cfg.bc_teacher_deterministic,
                    action_input=action_input,
                )
            bc_logp = self.get_logprobs(
                first_hidden_state,
                last_hidden_state,
                action_input.status_feature,
                teacher_chains,
                deterministic=False,
                action_input=action_input,
            )
            bc_logp = bc_logp.clamp(min=-5, max=2)
            K_steps = chains.shape[1] - 1
            bc_logp = bc_logp.view(-1, K_steps, chains.shape[2], chains.shape[3]).mean(dim=[1,2,3])
            bc_loss = -bc_logp.mean()
            total_loss = total_loss + self.config.grpo_cfg.bc_coeff * bc_loss

        # Plan D: surface the std that actually governs the flow transition, so a
        # sweep over sde_noise_scale / sde_min_std is observable. With the defaults
        # the floor dominates (flow_std == sde_min_std) and sde_noise_scale has no
        # effect until it is raised above the floor.
        if self.config.sampling_method == 'flow':
            flow_dt = 1.0 / max(num_denoising_steps, 1)
            flow_std = torch.tensor(
                max(self.config.flow_cfg.sde_noise_scale * math.sqrt(flow_dt), self.config.flow_cfg.sde_min_std),
                device=total_loss.device,
                dtype=torch.float32,
            )
        else:
            flow_std = torch.zeros((), device=total_loss.device, dtype=torch.float32)

        return BatchFeature(data={
            "loss": total_loss,
            "reward": rewards.mean(),
            "policy_loss": policy_loss,
            "bc_loss": bc_loss,
            "reward_std": reward_std,
            "advantage_std": advantage_std,
            "logprob_mean": logprob_mean,
            "logprob_clamp_hit_rate": logprob_clamp_hit_rate,
            "flow_std": flow_std,
        })

    # ------------------------------------------------------------------ #
    # Explicit safety losses: future OBB, drivable SDF, and route alignment.
    # ------------------------------------------------------------------ #

    def _compute_auxiliary_losses(
        self,
        pred_traj_phys: torch.Tensor,
        action_input: BatchFeature,
        safety_targets: BatchFeature,
        sample_weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zero = pred_traj_phys.sum() * 0.0
        losses = {
            "collision_loss": zero,
            "offroad_loss": zero,
            "route_loss": zero,
        }

        if self.config.lambda_collision > 0.0:
            losses["collision_loss"] = collision_clearance_loss(
                pred_traj_phys,
                safety_targets.future_obj_boxes,
                safety_targets.future_obj_valid_mask,
                front_length=self.config.ego_front_length,
                rear_length=self.config.ego_rear_length,
                ego_width=self.config.ego_width,
                safety_margin=self.config.collision_margin,
                smooth_beta=self.config.safety_smooth_beta,
                sample_weight=sample_weight,
            )

        if self.config.lambda_offroad > 0.0:
            losses["offroad_loss"] = drivable_sdf_loss(
                pred_traj_phys,
                safety_targets.drivable_sdf,
                safety_targets.drivable_sdf_origin,
                safety_targets.drivable_sdf_resolution,
                safety_targets.drivable_sdf_valid,
                front_length=self.config.ego_front_length,
                rear_length=self.config.ego_rear_length,
                ego_width=self.config.ego_width,
                safety_margin=self.config.offroad_margin,
                smooth_beta=self.config.safety_smooth_beta,
                sample_weight=sample_weight,
            )

        if self.config.lambda_route > 0.0:
            required_map_keys = (
                "map_point_position",
                "map_point_vector",
                "map_polygon_type",
                "map_polygon_on_route",
                "map_point_valid_mask",
                "map_valid_mask",
            )
            missing = [key for key in required_map_keys if key not in action_input]
            if missing:
                raise KeyError(f"Route safety loss requires HDMap keys: {missing}")
            losses["route_loss"] = route_alignment_loss(
                pred_traj_phys,
                action_input.map_point_position,
                action_input.map_point_vector,
                action_input.map_polygon_type,
                action_input.map_polygon_on_route,
                action_input.map_point_valid_mask,
                action_input.map_valid_mask,
                lateral_margin=self.config.route_lateral_margin,
                heading_weight=self.config.route_heading_weight,
                smooth_beta=self.config.safety_smooth_beta,
                sample_weight=sample_weight,
                batch_chunk_size=self.config.route_loss_batch_chunk_size,
            )

        return losses

    def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)
    
    def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
        """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
        x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
        y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
        heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
        return torch.cat([x, y, heading], dim=-1)

    def reward_fn(
        self,
        pred_traj: torch.Tensor,
        tokens_list,
        cache_dict,
    ) -> torch.Tensor:
        """Calculates PDM scores for a batch of predicted trajectories."""
        pred_np = pred_traj.detach().cpu().numpy()
        rewards = []
        for i, token in enumerate(tokens_list):
            trajectory = Trajectory(pred_np[i])
            metric_cache = cache_dict[token]
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.train_scorer,
            )
            rewards.append(asdict(pdm_result)["score"])
        return torch.tensor(rewards, device=pred_traj.device, dtype=pred_traj.dtype).detach()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

#https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/eta.py#L12
class EtaFixed(nn.Module):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        self.eta_logit = nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta

        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, x):
        """Match input batch size, but do not depend on input"""
        B = len(x)
        device = x.device
        eta_normalized = torch.tanh(self.eta_logit)

        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)
