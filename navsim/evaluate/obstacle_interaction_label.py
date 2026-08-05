"""Resolve per-token obstacle-interaction labels for PDM evaluation CSV output.

A token has interaction if, on any GT frame from the current timestamp through
the next ``horizon_s`` seconds, an oriented obstacle box of class
vehicle / pedestrian / bicycle intersects the ego-centered axis-aligned query
rectangle in that frame's ego coordinates.
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon, box as shapely_box

logger = logging.getLogger(__name__)

INTERACTION_CLASSES = frozenset({"vehicle", "pedestrian", "bicycle"})
DEFAULT_RANGE_X: Tuple[float, float] = (-10.0, 20.0)
DEFAULT_RANGE_Y: Tuple[float, float] = (-5.0, 5.0)
DEFAULT_HORIZON_S: float = 4.0


def _oriented_box_polygon(
    x: float,
    y: float,
    length: float,
    width: float,
    heading: float,
) -> Polygon:
    """Build a 2D oriented rectangle polygon in ego coordinates."""
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    cos_h = math.cos(float(heading))
    sin_h = math.sin(float(heading))
    corners_local = (
        (-half_l, -half_w),
        (half_l, -half_w),
        (half_l, half_w),
        (-half_l, half_w),
    )
    corners = [
        (
            float(x) + cos_h * lx - sin_h * ly,
            float(y) + sin_h * lx + cos_h * ly,
        )
        for lx, ly in corners_local
    ]
    return Polygon(corners)


def obstacle_intersects_query_rect(
    obstacle_box: Sequence[float],
    range_x: Tuple[float, float] = DEFAULT_RANGE_X,
    range_y: Tuple[float, float] = DEFAULT_RANGE_Y,
) -> bool:
    """True if an oriented GT box intersects the ego-frame query rectangle."""
    if len(obstacle_box) < 7:
        return False
    length = float(obstacle_box[3])
    width = float(obstacle_box[4])
    if length <= 0.0 or width <= 0.0:
        return False

    query = shapely_box(range_x[0], range_y[0], range_x[1], range_y[1])
    obstacle = _oriented_box_polygon(
        x=float(obstacle_box[0]),
        y=float(obstacle_box[1]),
        length=length,
        width=width,
        heading=float(obstacle_box[6]),
    )
    if obstacle.is_empty or not obstacle.is_valid:
        return False
    return bool(query.intersects(obstacle))


def frame_has_obstacle_interaction(
    names: Iterable[Any],
    boxes: Any,
    range_x: Tuple[float, float] = DEFAULT_RANGE_X,
    range_y: Tuple[float, float] = DEFAULT_RANGE_Y,
    allowed_classes: Iterable[str] = INTERACTION_CLASSES,
) -> bool:
    """Check one GT frame for query-rectangle intersection with allowed classes."""
    allowed = {str(name).strip().lower() for name in allowed_classes}
    box_arr = np.asarray(boxes)
    name_list = list(names)
    if box_arr.ndim != 2 or box_arr.shape[0] == 0:
        return False

    for idx in range(box_arr.shape[0]):
        name = str(name_list[idx]).strip().lower() if idx < len(name_list) else ""
        if name not in allowed:
            continue
        if obstacle_intersects_query_rect(box_arr[idx], range_x=range_x, range_y=range_y):
            return True
    return False


def resolve_obstacle_interaction(
    frames: List[Dict[str, Any]],
    current_idx: int,
    horizon_s: float = DEFAULT_HORIZON_S,
    range_x: Tuple[float, float] = DEFAULT_RANGE_X,
    range_y: Tuple[float, float] = DEFAULT_RANGE_Y,
) -> bool:
    """True if any GT frame in [t0, t0+horizon_s] has an interacting obstacle."""
    if current_idx < 0 or current_idx >= len(frames):
        raise IndexError(
            f"current_idx={current_idx} out of range for {len(frames)} frames"
        )

    t0 = int(frames[current_idx]["timestamp"])
    horizon_us = float(horizon_s) * 1e6
    for idx in range(current_idx, len(frames)):
        if int(frames[idx]["timestamp"]) - t0 > horizon_us:
            break
        anns = frames[idx].get("anns") or {}
        if frame_has_obstacle_interaction(
            names=anns.get("gt_names", []),
            boxes=anns.get("gt_boxes", []),
            range_x=range_x,
            range_y=range_y,
        ):
            return True
    return False


def build_token_obstacle_interaction_map(
    navsim_log_path: Path,
    token_to_log_name: Dict[str, str],
    tokens: List[str],
    horizon_s: float = DEFAULT_HORIZON_S,
    range_x: Tuple[float, float] = DEFAULT_RANGE_X,
    range_y: Tuple[float, float] = DEFAULT_RANGE_Y,
) -> Dict[str, bool]:
    """Build token -> has_obstacle_interaction using full 10Hz GT PKL frames."""
    tokens_by_log: Dict[str, List[str]] = {}
    for token in tokens:
        log_name = token_to_log_name.get(token)
        if log_name is None:
            continue
        tokens_by_log.setdefault(log_name, []).append(token)

    resolved: Dict[str, bool] = {}
    for log_name, log_tokens in tokens_by_log.items():
        pkl_path = Path(navsim_log_path) / f"{log_name}.pkl"
        if not pkl_path.is_file():
            logger.warning(
                "Missing log PKL for obstacle-interaction labeling: %s; "
                "defaulting %d tokens to False",
                pkl_path,
                len(log_tokens),
            )
            for token in log_tokens:
                resolved[token] = False
            continue

        with open(pkl_path, "rb") as f:
            frames: List[Dict[str, Any]] = pickle.load(f)

        token_to_idx = {frame["token"]: idx for idx, frame in enumerate(frames)}
        for token in log_tokens:
            current_idx = token_to_idx.get(token)
            if current_idx is None:
                logger.warning(
                    "Token %s not found in %s; defaulting has_obstacle_interaction to False",
                    token,
                    pkl_path,
                )
                resolved[token] = False
                continue
            resolved[token] = resolve_obstacle_interaction(
                frames,
                current_idx,
                horizon_s=horizon_s,
                range_x=range_x,
                range_y=range_y,
            )
    return resolved
