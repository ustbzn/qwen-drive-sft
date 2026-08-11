from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def _smooth_hinge(value: torch.Tensor, beta: float) -> torch.Tensor:
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    return F.softplus(value * beta) / beta


def _weighted_valid_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reduce one scalar per sample while excluding samples without supervision."""
    valid_weights = valid.to(values.dtype)
    weights = valid_weights
    if sample_weight is not None:
        weights = weights * sample_weight.to(device=values.device, dtype=values.dtype)
    # Normalize by supervised samples, not by SNR weights, so high-noise x0 estimates
    # are attenuated even when every item in a batch has a low alpha_bar.
    denominator = valid_weights.sum()
    numerator = (values * weights).sum()
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(values.dtype).eps),
        numerator * 0.0,
    )


def ego_box_center(
    trajectory: torch.Tensor,
    front_length: float,
    rear_length: float,
) -> torch.Tensor:
    """Convert rear-axle trajectory poses to geometric box centers."""
    heading = trajectory[..., 2]
    center_offset = 0.5 * (float(front_length) - float(rear_length))
    offset = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    return trajectory[..., :2] + center_offset * offset


def ego_box_corners(
    trajectory: torch.Tensor,
    front_length: float,
    rear_length: float,
    width: float,
) -> torch.Tensor:
    """Return four ego box corners as [B, T, 4, 2]."""
    center = ego_box_center(trajectory, front_length, rear_length)
    heading = trajectory[..., 2]
    forward = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    lateral = torch.stack([-torch.sin(heading), torch.cos(heading)], dim=-1)
    half_length = 0.5 * (float(front_length) + float(rear_length))
    half_width = 0.5 * float(width)
    longitudinal_sign = trajectory.new_tensor([1.0, 1.0, -1.0, -1.0])
    lateral_sign = trajectory.new_tensor([1.0, -1.0, -1.0, 1.0])
    return (
        center.unsqueeze(-2)
        + longitudinal_sign.view(1, 1, 4, 1)
        * half_length
        * forward.unsqueeze(-2)
        + lateral_sign.view(1, 1, 4, 1)
        * half_width
        * lateral.unsqueeze(-2)
    )


def oriented_box_signed_clearance(
    ego_trajectory: torch.Tensor,
    object_boxes: torch.Tensor,
    front_length: float,
    rear_length: float,
    ego_width: float,
    smooth_max_beta: float = 20.0,
) -> torch.Tensor:
    """Differentiable SAT clearance for ego/object oriented rectangles.

    Args:
        ego_trajectory: [B, T, 3] rear-axle poses.
        object_boxes: [B, T, N, 5] center x/y, heading, length, width.

    Returns:
        [B, T, N], positive for separation and negative for overlap.
    """
    if ego_trajectory.ndim != 3 or ego_trajectory.shape[-1] != 3:
        raise ValueError("ego_trajectory must have shape [B, T, 3]")
    if object_boxes.ndim != 4 or object_boxes.shape[-1] != 5:
        raise ValueError("object_boxes must have shape [B, T, N, 5]")
    if ego_trajectory.shape[:2] != object_boxes.shape[:2]:
        raise ValueError("ego and object batch/time dimensions must match")
    if smooth_max_beta <= 0.0:
        raise ValueError("smooth_max_beta must be positive")

    trajectory = ego_trajectory.float()
    boxes = object_boxes.float()
    ego_heading = trajectory[..., 2]
    obj_heading = boxes[..., 2]

    ego_forward = torch.stack(
        [torch.cos(ego_heading), torch.sin(ego_heading)], dim=-1
    )
    ego_lateral = torch.stack(
        [-torch.sin(ego_heading), torch.cos(ego_heading)], dim=-1
    )
    obj_forward = torch.stack(
        [torch.cos(obj_heading), torch.sin(obj_heading)], dim=-1
    )
    obj_lateral = torch.stack(
        [-torch.sin(obj_heading), torch.cos(obj_heading)], dim=-1
    )

    num_objects = boxes.shape[2]
    ego_forward_expanded = ego_forward.unsqueeze(2).expand(-1, -1, num_objects, -1)
    ego_lateral_expanded = ego_lateral.unsqueeze(2).expand_as(ego_forward_expanded)
    axes = torch.stack(
        [ego_forward_expanded, ego_lateral_expanded, obj_forward, obj_lateral],
        dim=-2,
    )

    ego_center = ego_box_center(
        trajectory, front_length=front_length, rear_length=rear_length
    )
    center_delta = boxes[..., :2] - ego_center.unsqueeze(2)
    center_projection = torch.abs(
        (center_delta.unsqueeze(-2) * axes).sum(dim=-1)
    )

    ego_half_length = 0.5 * (float(front_length) + float(rear_length))
    ego_half_width = 0.5 * float(ego_width)
    ego_radius = (
        torch.abs((ego_forward_expanded.unsqueeze(-2) * axes).sum(dim=-1))
        * ego_half_length
        + torch.abs((ego_lateral_expanded.unsqueeze(-2) * axes).sum(dim=-1))
        * ego_half_width
    )

    obj_half_length = boxes[..., 3].clamp_min(0.0).unsqueeze(-1) * 0.5
    obj_half_width = boxes[..., 4].clamp_min(0.0).unsqueeze(-1) * 0.5
    obj_radius = (
        torch.abs((obj_forward.unsqueeze(-2) * axes).sum(dim=-1))
        * obj_half_length
        + torch.abs((obj_lateral.unsqueeze(-2) * axes).sum(dim=-1))
        * obj_half_width
    )

    axis_separation = center_projection - ego_radius - obj_radius
    # log-mean-exp is a sign-preserving smooth approximation of max for overlap.
    return (
        torch.logsumexp(smooth_max_beta * axis_separation, dim=-1)
        - math.log(axis_separation.shape[-1])
    ) / smooth_max_beta


def collision_clearance_loss(
    trajectory: torch.Tensor,
    object_boxes: torch.Tensor,
    object_valid_mask: torch.Tensor,
    front_length: float,
    rear_length: float,
    ego_width: float,
    safety_margin: float,
    smooth_beta: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    clearance = oriented_box_signed_clearance(
        trajectory,
        object_boxes,
        front_length=front_length,
        rear_length=rear_length,
        ego_width=ego_width,
        smooth_max_beta=smooth_beta,
    )
    valid = object_valid_mask.bool()
    if valid.shape != clearance.shape:
        raise ValueError("object_valid_mask must have shape [B, T, N]")
    time_valid = valid.any(dim=-1)
    closest = clearance.masked_fill(~valid, torch.inf).min(dim=-1).values
    closest = torch.where(time_valid, closest, torch.zeros_like(closest))
    per_time = _smooth_hinge(float(safety_margin) - closest, smooth_beta)
    per_time = torch.where(time_valid, per_time, torch.zeros_like(per_time))
    per_sample = per_time.sum(dim=-1) / time_valid.sum(dim=-1).clamp_min(1)
    return _weighted_valid_mean(per_sample, time_valid.any(dim=-1), sample_weight)


def drivable_sdf_loss(
    trajectory: torch.Tensor,
    drivable_sdf: torch.Tensor,
    sdf_origin: torch.Tensor,
    sdf_resolution: torch.Tensor,
    sdf_valid: torch.Tensor,
    front_length: float,
    rear_length: float,
    ego_width: float,
    safety_margin: float,
    smooth_beta: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bilinearly sample positive-inside SDF at all four ego corners."""
    if drivable_sdf.ndim != 4 or drivable_sdf.shape[1] != 1:
        raise ValueError("drivable_sdf must have shape [B, 1, H, W]")
    batch_size, _, height, width = drivable_sdf.shape
    if height < 2 or width < 2:
        raise ValueError("drivable_sdf height/width must both be at least 2")
    if trajectory.shape[0] != batch_size:
        raise ValueError("trajectory and drivable_sdf batch sizes must match")

    trajectory_float = trajectory.float()
    sdf = drivable_sdf.to(device=trajectory.device, dtype=torch.float32)
    origin = sdf_origin.to(device=trajectory.device, dtype=torch.float32)
    resolution = sdf_resolution.to(
        device=trajectory.device, dtype=torch.float32
    ).reshape(-1)
    if resolution.numel() == 1:
        resolution = resolution.expand(batch_size)
    if origin.shape != (batch_size, 2) or resolution.shape != (batch_size,):
        raise ValueError("invalid SDF origin/resolution batch shape")

    corners = ego_box_corners(
        trajectory_float,
        front_length=front_length,
        rear_length=rear_length,
        width=ego_width,
    )
    x_min = origin[:, 0, None, None]
    y_min = origin[:, 1, None, None]
    x_max = x_min + resolution[:, None, None] * (width - 1)
    y_max = y_min + resolution[:, None, None] * (height - 1)

    grid_x = 2.0 * (corners[..., 0] - x_min) / (x_max - x_min) - 1.0
    grid_y = 2.0 * (corners[..., 1] - y_min) / (y_max - y_min) - 1.0
    sample_grid = torch.stack([grid_x, grid_y], dim=-1)
    sampled_sdf = F.grid_sample(
        sdf,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[:, 0]

    outside_dx = F.relu(x_min - corners[..., 0]) + F.relu(corners[..., 0] - x_max)
    outside_dy = F.relu(y_min - corners[..., 1]) + F.relu(corners[..., 1] - y_max)
    outside_distance = torch.linalg.vector_norm(
        torch.stack([outside_dx, outside_dy], dim=-1), dim=-1
    )
    effective_sdf = sampled_sdf - outside_distance
    corner_penalty = _smooth_hinge(
        float(safety_margin) - effective_sdf, smooth_beta
    )
    per_sample = corner_penalty.mean(dim=(-1, -2))
    return _weighted_valid_mean(per_sample, sdf_valid.bool(), sample_weight)


def route_alignment_loss(
    trajectory: torch.Tensor,
    map_point_position: torch.Tensor,
    map_point_vector: torch.Tensor,
    map_polygon_type: torch.Tensor,
    map_polygon_on_route: torch.Tensor,
    map_point_valid_mask: torch.Tensor,
    map_valid_mask: torch.Tensor,
    lateral_margin: float,
    heading_weight: float,
    smooth_beta: float,
    sample_weight: Optional[torch.Tensor] = None,
    lane_polygon_types: Sequence[int] = (0, 1),
    batch_chunk_size: int = 128,
) -> torch.Tensor:
    """Align trajectory position and heading to the nearest local route tangent."""
    if batch_chunk_size <= 0:
        raise ValueError("batch_chunk_size must be positive")
    positions = map_point_position[:, :, 0].float()
    vectors = map_point_vector[:, :, 0].float()
    lane_type_mask = torch.zeros_like(map_valid_mask, dtype=torch.bool)
    for polygon_type in lane_polygon_types:
        lane_type_mask |= map_polygon_type == int(polygon_type)
    route_token_mask = (
        lane_type_mask & (map_polygon_on_route == 1) & map_valid_mask.bool()
    )
    point_mask = route_token_mask.unsqueeze(-1) & map_point_valid_mask.bool()
    point_mask &= torch.linalg.vector_norm(vectors, dim=-1) > 1e-6

    flat_positions = positions.flatten(1, 2)
    flat_vectors = vectors.flatten(1, 2)
    flat_mask = point_mask.flatten(1, 2)
    trajectory_float = trajectory.float()

    per_sample_parts = []
    valid_parts = []
    for start in range(0, trajectory.shape[0], batch_chunk_size):
        end = min(start + batch_chunk_size, trajectory.shape[0])
        chunk_positions = flat_positions[start:end]
        chunk_vectors = flat_vectors[start:end]
        chunk_mask = flat_mask[start:end]
        chunk_trajectory = trajectory_float[start:end]
        has_route = chunk_mask.any(dim=-1)

        distances = torch.cdist(chunk_trajectory[..., :2], chunk_positions)
        distances = distances.masked_fill(~chunk_mask.unsqueeze(1), torch.inf)
        min_distance, nearest_index = distances.min(dim=-1)
        min_distance = torch.where(
            has_route.unsqueeze(-1), min_distance, torch.zeros_like(min_distance)
        )
        gather_index = nearest_index.unsqueeze(-1).expand(-1, -1, 2)
        nearest_vector = torch.gather(chunk_vectors, 1, gather_index)
        tangent_heading = torch.atan2(nearest_vector[..., 1], nearest_vector[..., 0])

        lateral_penalty = _smooth_hinge(
            min_distance - float(lateral_margin), smooth_beta
        )
        heading_penalty = 1.0 - torch.cos(
            chunk_trajectory[..., 2] - tangent_heading
        )
        per_sample = (lateral_penalty + float(heading_weight) * heading_penalty).mean(
            dim=-1
        )
        per_sample_parts.append(
            torch.where(has_route, per_sample, torch.zeros_like(per_sample))
        )
        valid_parts.append(has_route)

    per_sample_loss = torch.cat(per_sample_parts, dim=0)
    valid = torch.cat(valid_parts, dim=0)
    return _weighted_valid_mean(per_sample_loss, valid, sample_weight)
