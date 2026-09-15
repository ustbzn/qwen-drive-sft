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
        num_polygon_types: int = 4,
        num_route_states: int = 3,
        num_tl_states: int = 5,
        num_control_types: int = 3,
        num_tl_directions: int = 6,
        num_boundary_types: int = 3,
        use_lane_boundaries: bool = False,
        encode_boundary_constraint: bool = False,
        input_norm: bool = False,
        _expand_from_13dim: bool = True,  # 新增：自动扩展旧checkpoint
    ) -> None:
        super().__init__()
        self.use_lane_boundaries = bool(use_lane_boundaries)
        self.encode_boundary_constraint = bool(encode_boundary_constraint)
        self._expand_from_13dim = bool(_expand_from_13dim)
        # fp16 数值加固（默认关，字节级零影响）：对进 encoder 的局部米制坐标做
        # nan_to_num→宽 clamp→scale 到 O(1)。局部坐标实测 <652m（远低于 fp16 上限 65504），
        # clamp 阈值取 ±1000 覆盖全部真实数据不截断，仅兜底极端/脏坐标；scale 让首层 Linear
        # 输入落到 O(1-10)，消除训练中激活/梯度被大尺度输入放大导致的溢出。首层是
        # Linear→LayerNorm，LayerNorm 吸收全局 scale，故从头训练表达力/效果一致；旧 ckpt 不可
        # 热启（首层输入尺度变了会错位）。
        self.input_norm = bool(input_norm)
        self._pos_clamp, self._pos_scale = 1000.0, 100.0
        self._vec_clamp, self._vec_scale = 100.0, 10.0
        # 中心线 per-point 6 维 [pos(2), vec(2), sinθ, cosθ]；
        # + use_lane_boundaries: 追加 7 维边界几何 -> 13 维
        # + encode_boundary_constraint: 再追加 4 维边界约束 (left_type, left_weight, right_type, right_weight) -> 17 维
        point_input_dim = 6
        if self.use_lane_boundaries:
            point_input_dim += 7
        if self.encode_boundary_constraint:
            point_input_dim += 4
        self.centerline_encoder = PointsEncoder(input_dim=point_input_dim, hidden_size=hidden_size)
        self.type_embedding = nn.Embedding(num_polygon_types, hidden_size)
        self.route_embedding = nn.Embedding(num_route_states, hidden_size)
        self.tl_embedding = nn.Embedding(num_tl_states, hidden_size)
        self.control_embedding = nn.Embedding(num_control_types, hidden_size)
        self.direction_embedding = nn.Embedding(num_tl_directions, hidden_size)
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
        polygon_control_type: torch.Tensor,
        polygon_tl_direction: torch.Tensor,
        polygon_has_speed_limit: torch.Tensor,
        polygon_speed_limit: torch.Tensor,
        point_valid_mask: torch.Tensor,
        map_valid_mask: torch.Tensor,
        boundary_constraint: Optional[torch.Tensor] = None,
        boundary_type: Optional[torch.Tensor] = None,
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

        # 开启边界约束编码：追加 4 维 [left_type_normalized, left_constraint, right_type_normalized, right_constraint]
        if self.encode_boundary_constraint and boundary_constraint is not None and boundary_type is not None:
            boundary_constraint_features = self._boundary_constraint_features(
                boundary_constraint, boundary_type
            )
            point_features = torch.cat([point_features, boundary_constraint_features], dim=-1)

        flattened_points = point_features.reshape(batch_size * max_map_tokens, sample_points, -1)
        flattened_mask = point_valid_mask.reshape(batch_size * max_map_tokens, sample_points)
        polyline_tokens = self.centerline_encoder(flattened_points, flattened_mask)
        polyline_tokens = polyline_tokens.reshape(batch_size, max_map_tokens, -1)

        polygon_type = polygon_type.clamp(min=0, max=self.type_embedding.num_embeddings - 1).long()
        polygon_on_route = polygon_on_route.clamp(
            min=0, max=self.route_embedding.num_embeddings - 1
        ).long()
        polygon_tl_status = polygon_tl_status.clamp(
            min=0, max=self.tl_embedding.num_embeddings - 1
        ).long()
        polygon_control_type = polygon_control_type.clamp(
            min=0, max=self.control_embedding.num_embeddings - 1
        ).long()
        polygon_tl_direction = polygon_tl_direction.clamp(
            min=0, max=self.direction_embedding.num_embeddings - 1
        ).long()

        speed_limit = polygon_speed_limit.to(dtype=dtype).unsqueeze(-1)
        has_speed_limit = polygon_has_speed_limit.to(dtype=dtype).unsqueeze(-1)
        speed_features = torch.cat([speed_limit, has_speed_limit], dim=-1)

        tokens = (
            polyline_tokens
            + self.type_embedding(polygon_type)
            + self.route_embedding(polygon_on_route)
            + self.tl_embedding(polygon_tl_status)
            + self.control_embedding(polygon_control_type)
            + self.direction_embedding(polygon_tl_direction)
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

    def _boundary_constraint_features(
        self,
        boundary_constraint: torch.Tensor,
        boundary_type: torch.Tensor,
    ) -> torch.Tensor:
        """从边界类型和约束权重构建 per-point 边界约束特征 (B, M, 2, P) -> (B, M, P, 4)。

        Args:
            boundary_constraint: (B, M, 2, P) 边界约束权重，1.0=NON_CROSSABLE, 0.0=其他
            boundary_type: (B, M, 2, P) 边界类型整数编码 (0=DEFAULT, 1=CROSSABLE, 2=NON_CROSSABLE)

        Returns:
            (B, M, P, 4) 特征: [left_type_normalized, left_constraint, right_type_normalized, right_constraint]
                - type_normalized: 类型归一化到 [0, 1] 范围
                - constraint: 约束权重，直接使用 boundary_constraint
        """
        dtype = self.output_norm.weight.dtype
        # boundary_constraint: (B, M, 2, P)
        # boundary_type: (B, M, 2, P)
        # 转换为 (B, M, P, 2) 以便拼接
        constraint_left = boundary_constraint[:, :, 0, :].to(dtype=dtype)  # (B, M, P)
        constraint_right = boundary_constraint[:, :, 1, :].to(dtype=dtype)  # (B, M, P)
        type_left = boundary_type[:, :, 0, :].to(dtype=dtype) / 2.0        # (B, M, P) 归一化到 [0, 1]
        type_right = boundary_type[:, :, 1, :].to(dtype=dtype) / 2.0       # (B, M, P)

        return torch.stack(
            [type_left, constraint_left, type_right, constraint_right], dim=-1
        )  # (B, M, P, 4)

    def load_state_dict(self, state_dict, strict=True):
        """加载checkpoint时自动扩展13→17维权重（兼容旧checkpoint）。"""
        if not self._expand_from_13dim or not self.encode_boundary_constraint:
            return super().load_state_dict(state_dict, strict=strict)

        # 检查是否需要扩展：centerline_encoder.point_mlp.0.weight 从 (hidden, 13) → (hidden, 17)
        key = "centerline_encoder.point_mlp.0.weight"
        if key in state_dict:
            old_weight = state_dict[key]
            expected_in = 13 if self.use_lane_boundaries else 6
            if old_weight.shape[1] == expected_in:  # 旧checkpoint是13维（或6维）
                # 扩展到17维（或10维）：新增4维boundary constraint，用零初始化
                import torch
                new_in = expected_in + 4
                new_weight = torch.zeros(old_weight.shape[0], new_in, dtype=old_weight.dtype)
                new_weight[:, :expected_in] = old_weight  # 复制旧权重
                # 新增的4维用小随机值初始化，让网络可学习
                torch.nn.init.normal_(new_weight[:, expected_in:], std=0.02)
                state_dict[key] = new_weight

                # 同样扩展bias（如果存在）
                bias_key = "centerline_encoder.point_mlp.0.bias"
                if bias_key in state_dict:
                    # bias维度不变，无需扩展
                    pass

        return super().load_state_dict(state_dict, strict=strict)
