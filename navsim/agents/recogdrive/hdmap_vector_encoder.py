from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class PointsEncoder(nn.Module):
    """Encode sampled polyline points into one token per map object."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, point_features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        point_tokens = self.point_mlp(point_features)
        mask = point_mask.unsqueeze(-1).to(dtype=point_tokens.dtype)
        point_tokens = point_tokens.masked_fill(mask <= 0, -1e4)
        pooled = point_tokens.max(dim=1).values
        has_points = (point_mask.sum(dim=1, keepdim=True) > 0).to(dtype=pooled.dtype)
        return pooled * has_points


class MapEncoder(nn.Module):
    """Encode HDMap vector tensors to DiT context tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_polygon_types: int = 3,
        num_tl_states: int = 4,
        use_lane_boundaries: bool = False,
        input_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_lane_boundaries = bool(use_lane_boundaries)
        # fp16 数值加固（默认关，字节级零影响）：对进 encoder 的局部米制坐标做
        # nan_to_num→宽 clamp→scale 到 O(1)。局部坐标实测 <652m（远低于 fp16 上限 65504），
        # clamp 阈值取 ±1000 覆盖全部真实数据不截断，仅兜底极端/脏坐标；scale 让首层 Linear
        # 输入落到 O(1-10)，消除训练中激活/梯度被大尺度输入放大导致的溢出。首层是
        # Linear→LayerNorm，LayerNorm 吸收全局 scale，故从头训练表达力/效果一致；旧 ckpt 不可
        # 热启（首层输入尺度变了会错位）。
        self.input_norm = bool(input_norm)
        self._pos_clamp, self._pos_scale = 1000.0, 100.0
        self._vec_clamp, self._vec_scale = 100.0, 10.0
        # 中心线 per-point 6 维 [pos(2), vec(2), sinθ, cosθ]；开启车道边界再拼 7 维边界几何 -> 13 维。
        point_input_dim = 13 if self.use_lane_boundaries else 6
        self.centerline_encoder = PointsEncoder(input_dim=point_input_dim, hidden_size=hidden_size)
        self.type_embedding = nn.Embedding(num_polygon_types, hidden_size)
        self.route_embedding = nn.Embedding(2, hidden_size)
        self.tl_embedding = nn.Embedding(num_tl_states, hidden_size)
        self.speed_mlp = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        point_position: torch.Tensor,
        point_vector: torch.Tensor,
        point_orientation: torch.Tensor,
        polygon_type: torch.Tensor,
        polygon_on_route: torch.Tensor,
        polygon_tl_status: torch.Tensor,
        polygon_has_speed_limit: torch.Tensor,
        polygon_speed_limit: torch.Tensor,
        point_valid_mask: torch.Tensor,
        map_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_map_tokens, _, sample_points, _ = point_position.shape
        dtype = self.output_norm.weight.dtype

        # fp16 数值加固：nan_to_num→宽 clamp→scale。必须在下面 center/boundary 计算之前做，
        # 使中心线特征与从 point_position 现算的边界几何共享同一 scaled 坐标（13 维尺度自洽）。
        # padding polygon 坐标为 0，经此变换仍为 0，末端再乘 map_valid_mask 置零，语义不变。
        # orientation 走 sin/cos 已有界、speed_limit 量级小，均无需处理。
        if self.input_norm:
            point_position = torch.nan_to_num(
                point_position, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(-self._pos_clamp, self._pos_clamp) / self._pos_scale
            point_vector = torch.nan_to_num(
                point_vector, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(-self._vec_clamp, self._vec_clamp) / self._vec_scale

        center_position = point_position[:, :, 0].to(dtype=dtype)
        center_vector = point_vector[:, :, 0].to(dtype=dtype)
        center_orientation = point_orientation[:, :, 0].to(dtype=dtype)
        point_features = torch.cat(
            [
                center_position,
                center_vector,
                torch.sin(center_orientation).unsqueeze(-1),
                torch.cos(center_orientation).unsqueeze(-1),
            ],
            dim=-1,
        )

        # 开启车道边界：追加左右边界显式横向几何（7 维），拼到中心线 per-point 特征后 -> 13 维。
        if self.use_lane_boundaries:
            boundary_features = self._boundary_point_features(
                point_position, center_position, center_orientation
            )
            point_features = torch.cat([point_features, boundary_features], dim=-1)

        flattened_points = point_features.reshape(batch_size * max_map_tokens, sample_points, -1)
        flattened_mask = point_valid_mask.reshape(batch_size * max_map_tokens, sample_points)
        polyline_tokens = self.centerline_encoder(flattened_points, flattened_mask)
        polyline_tokens = polyline_tokens.reshape(batch_size, max_map_tokens, -1)

        polygon_type = polygon_type.clamp(min=0, max=self.type_embedding.num_embeddings - 1).long()
        polygon_on_route = polygon_on_route.clamp(min=0, max=1).long()
        polygon_tl_status = polygon_tl_status.clamp(min=0, max=self.tl_embedding.num_embeddings - 1).long()

        speed_limit = polygon_speed_limit.to(dtype=dtype).unsqueeze(-1)
        has_speed_limit = polygon_has_speed_limit.to(dtype=dtype).unsqueeze(-1)
        speed_features = torch.cat([speed_limit, has_speed_limit], dim=-1)

        tokens = (
            polyline_tokens
            + self.type_embedding(polygon_type)
            + self.route_embedding(polygon_on_route)
            + self.tl_embedding(polygon_tl_status)
            + self.speed_mlp(speed_features)
        )
        tokens = self.output_norm(tokens)
        return tokens * map_valid_mask.to(dtype=tokens.dtype).unsqueeze(-1)

    def _boundary_point_features(
        self,
        point_position: torch.Tensor,
        center_position: torch.Tensor,
        center_orientation: torch.Tensor,
    ) -> torch.Tensor:
        """从左右边界采样点算显式横向几何，返回 (B, M, P, 7)。

        left/right 与 center 同 index（各自按弧长重采样到 P），逐点作差即得横向走廊几何：
        左/右边界相对中心的偏移(4) + 车道宽(1) + 到左/右边界的带符号横向距离(2)。
        padding polygon 三线全 0 -> 输出全 0，末端再乘 map_valid_mask 置零；
        fallback(±1.75m 合成) 宽恒 3.5m、对称，作为中性先验可接受。
        """
        dtype = center_position.dtype
        left_position = point_position[:, :, 1].to(dtype=dtype)
        right_position = point_position[:, :, 2].to(dtype=dtype)
        left_offset = left_position - center_position                              # (B,M,P,2)
        right_offset = right_position - center_position                            # (B,M,P,2)
        lane_width = (left_position - right_position).norm(dim=-1, keepdim=True)   # (B,M,P,1)
        # 中心线单位法向 n = (-sinθc, cosθc)，把偏移投影成带符号横向距离（左正右负与朝向无关）。
        normal = torch.stack(
            [-torch.sin(center_orientation), torch.cos(center_orientation)], dim=-1
        )                                                                          # (B,M,P,2)
        signed_left = (left_offset * normal).sum(dim=-1, keepdim=True)             # (B,M,P,1)
        signed_right = (right_offset * normal).sum(dim=-1, keepdim=True)           # (B,M,P,1)
        return torch.cat(
            [left_offset, right_offset, lane_width, signed_left, signed_right], dim=-1
        )                                                                          # (B,M,P,7)
