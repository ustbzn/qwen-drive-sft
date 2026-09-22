"""Flow-matching 训练侧：官方只发布了 `PlanningExpert.sample`，loss 与 t 采样在这里。

参数化必须与官方推理端严格对应，否则加载 planner-sft 权重后行为会漂移。

官方 `sample` 的更新式（planning_expert.py:402-418）：

    waypoints = noise                       # t = 0
    for index in range(num_steps):
        t        = index / num_steps
        endpoint = predict_endpoint(waypoints, t, ...)     # 网络直接预测 x1
        remaining = max(1 - t, min_one_minus_t)
        waypoints += (endpoint - waypoints) / remaining * step

即网络输出是**干净终点 x1**（clean-endpoint / x-parameterization），不是速度场。
配套的前向过程是直线插值

    x_t = (1 - t) * noise + t * x1

其诱导速度 (x1 - x_t) / (1 - t) == x1 - noise，与官方 sampler 的除法一致。

论文第 5-6 页明确规定对诱导速度做 MSE，等价于终点误差除以 (1-t)^2。
时间从 Beta(1.5, 1) 采样后截断到 0.9，不是将整个分布缩放到 [0, 0.9]。
公式（8）的一阶、二阶 Huber 正则系数分别是 2e-4 和 2e-5。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def sample_flow_time(
    batch_size: int,
    device: torch.device,
    alpha: float = 1.5,
    beta: float = 1.0,
    t_max: float = 0.9,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """从 Beta(alpha, beta) 采样并截断到 t_max，返回 float32 的 [B]。"""
    if alpha <= 0 or beta <= 0:
        raise ValueError(f"Beta 参数必须 > 0，得到 alpha={alpha}, beta={beta}")
    if not 0.0 < t_max < 1.0:
        raise ValueError("t_max 必须在 (0, 1) 内")
    # torch.distributions.Beta 不接受 generator；用两个 Gamma 的比值以支持可复现采样。
    gamma_a = torch._standard_gamma(
        torch.full((batch_size,), float(alpha), device=device), generator=generator
    )
    gamma_b = torch._standard_gamma(
        torch.full((batch_size,), float(beta), device=device), generator=generator
    )
    s = gamma_a / (gamma_a + gamma_b).clamp_min(1e-12)
    return s.clamp(0.0, float(t_max))


def interpolate_noisy_trajectory(
    clean: torch.Tensor,
    noise: torch.Tensor,
    flow_time: torch.Tensor,
) -> torch.Tensor:
    """直线插值 ``x_t = (1 - t) * noise + t * clean``。

    :param clean: 归一化后的真值轨迹 ``[B, T, 3]``
    :param noise: 同形状标准正态
    :param flow_time: ``[B]``，取值 [0, 1)
    """
    t = flow_time.view(-1, 1, 1).to(clean.dtype)
    return (1.0 - t) * noise + t * clean


def endpoint_loss(
    predicted: torch.Tensor,
    clean: torch.Tensor,
    flow_time: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """论文诱导速度 MSE 的等价式：mean(((predicted-clean)/(1-t))**2)。

    :param predicted: ``predict_endpoint`` 的输出 ``[B, T, 3]``（float32）
    :param clean: 归一化真值 ``[B, T, 3]``
    :param valid_mask: 可选 ``[B, T]``，1 表示该 waypoint 有监督
    """
    remaining = 1.0 - flow_time.float().reshape(-1, 1, 1)
    error = ((predicted.float() - clean.float()) / remaining).pow(2)
    return _masked_mean(error, valid_mask)


def _masked_mean(error: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if valid_mask is None:
        return error.mean()
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1)
    total = (error * mask).sum()
    count = mask.sum() * error.shape[-1]
    return total / count.clamp_min(1.0)


def temporal_smoothness_loss(
    predicted: torch.Tensor,
    clean: torch.Tensor,
    lambda_delta1: float = 2.0e-4,
    lambda_delta2: float = 2.0e-5,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """归一化轨迹的一阶/二阶 Huber 正则；只监督全部相邻点有效的差分。"""
    predicted, clean = predicted.float(), clean.float()
    loss = predicted.new_zeros(())
    if lambda_delta1 > 0.0 and predicted.shape[1] >= 2:
        d1_pred = predicted[:, 1:] - predicted[:, :-1]
        d1_true = clean[:, 1:] - clean[:, :-1]
        mask = None if valid_mask is None else valid_mask[:, 1:] & valid_mask[:, :-1]
        loss = loss + lambda_delta1 * _masked_mean(
            F.huber_loss(d1_pred, d1_true, reduction="none"), mask
        )
    if lambda_delta2 > 0.0 and predicted.shape[1] >= 3:
        d2_pred = predicted[:, 2:] - 2 * predicted[:, 1:-1] + predicted[:, :-2]
        d2_true = clean[:, 2:] - 2 * clean[:, 1:-1] + clean[:, :-2]
        mask = None if valid_mask is None else (
            valid_mask[:, 2:] & valid_mask[:, 1:-1] & valid_mask[:, :-2]
        )
        loss = loss + lambda_delta2 * _masked_mean(
            F.huber_loss(d2_pred, d2_true, reduction="none"), mask
        )
    return loss


def flow_matching_step(
    expert,
    clean_normalized: torch.Tensor,
    history_queries: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scene_cache,
    position_anchor: torch.Tensor,
    nav_command: torch.Tensor,
    ego_status: torch.Tensor,
    *,
    alpha: float = 1.5,
    beta: float = 1.0,
    t_max: float = 0.9,
    valid_mask: Optional[torch.Tensor] = None,
    lambda_delta1: float = 2.0e-4,
    lambda_delta2: float = 2.0e-5,
    generator: Optional[torch.Generator] = None,
    scene_valid: Optional[torch.Tensor] = None,
    gradient_checkpointing: bool = False,
) -> Dict[str, torch.Tensor]:
    """跑一次前向并算 loss。

    直接复用官方 `predict_endpoint`（已验证对 out_proj 与输入均可微），不改动官方代码。

    :param clean_normalized: 归一化真值 ``[B, T, 3]``
    :param scene_valid: 可选 ``[B, L_scene]`` bool，True=真实 scene token、False=batch
        pad 出来的占位。训练 batch>1 时不同样本 prompt 长度不同，`scene_cache` 已在
        `qwen_drive_agent._encode_scenes` 里 pad 到 batch 内最大长度；这里必须屏蔽 pad
        位置，否则 waypoint token 会把无意义的 pad K/V 也当作有效上下文加权平均。
        None 时（batch==1 或调用方已保证等长）退化为官方原始 `expert.predict_endpoint`，
        数值与官方完全一致。
    :return: dict 含 ``loss`` / ``diffusion_loss`` / ``temporal_loss`` / ``predicted``
    """
    batch = clean_normalized.shape[0]
    device = clean_normalized.device
    flow_time = sample_flow_time(
        batch, device, alpha=alpha, beta=beta, t_max=t_max, generator=generator
    )
    noise = torch.randn(
        clean_normalized.shape, device=device, dtype=torch.float32, generator=generator
    )
    noisy = interpolate_noisy_trajectory(clean_normalized.float(), noise, flow_time)

    if scene_valid is None:
        predicted = expert.predict_endpoint(
            noisy,
            flow_time,
            history_queries,
            scene_cache,
            position_anchor,
            nav_command,
            ego_status,
        )
    else:
        from .qwen_drive_official import predict_endpoint_masked

        predicted = predict_endpoint_masked(
            expert,
            noisy,
            flow_time,
            history_queries,
            scene_cache,
            position_anchor,
            nav_command,
            ego_status,
            scene_valid=scene_valid,
            gradient_checkpointing=gradient_checkpointing,
        )

    diffusion = endpoint_loss(predicted, clean_normalized, flow_time, valid_mask)
    temporal = temporal_smoothness_loss(
        predicted, clean_normalized.float(), lambda_delta1, lambda_delta2, valid_mask
    )
    return {
        "loss": diffusion + temporal,
        "diffusion_loss": diffusion.detach(),
        "temporal_loss": temporal.detach(),
        "predicted": predicted,
        "flow_time": flow_time.detach(),
    }
