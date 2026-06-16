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
        pooled = point_tokens.max(dim=2).values
        has_points = (point_mask.sum(dim=1, keepdim=True) > 0).to(dtype=pooled.dtype)
        return pooled * has_points


class MapEncoder(nn.Module):
    """Encode HDMap vector tensors to DiT context tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_polygon_types: int = 3,
        num_tl_states: int = 4,
    ) -> None:
        super().__init__()
        self.centerline_encoder = PointsEncoder(input_dim=6, hidden_size=hidden_size)
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
