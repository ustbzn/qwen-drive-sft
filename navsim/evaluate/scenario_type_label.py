"""Resolve per-token scenario_type labels for PDM evaluation CSV output."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_TYPE = "lane_keep"


def normalize_scenario_type(value: Any) -> str:
    """Normalize a raw PKL scenario_type value; missing/blank -> lane_keep."""
    if value is None:
        return DEFAULT_SCENARIO_TYPE
    label = str(value).strip().lower()
    return label if label else DEFAULT_SCENARIO_TYPE


def resolve_scenario_type_label(
    scenario_types: List[Any],
    timestamps_us: List[int],
    current_idx: int,
    horizon_s: float = 2.0,
) -> str:
    """Resolve CSV scenario_type from the current frame and a short future window.

    Rules:
    - current is non-lane_keep -> use current
    - current is lane_keep and a non-lane_keep appears within horizon_s -> first such label
    - current and the horizon window are all lane_keep -> lane_keep
    """
    if current_idx < 0 or current_idx >= len(scenario_types):
        raise IndexError(
            f"current_idx={current_idx} out of range for {len(scenario_types)} frames"
        )
    if len(scenario_types) != len(timestamps_us):
        raise ValueError("scenario_types and timestamps_us must have the same length")

    current_label = normalize_scenario_type(scenario_types[current_idx])
    if current_label != DEFAULT_SCENARIO_TYPE:
        return current_label

    t0 = int(timestamps_us[current_idx])
    horizon_us = float(horizon_s) * 1e6
    for idx in range(current_idx + 1, len(scenario_types)):
        if int(timestamps_us[idx]) - t0 > horizon_us:
            break
        future_label = normalize_scenario_type(scenario_types[idx])
        if future_label != DEFAULT_SCENARIO_TYPE:
            return future_label
    return DEFAULT_SCENARIO_TYPE


def build_token_scenario_type_map(
    navsim_log_path: Path,
    token_to_log_name: Dict[str, str],
    tokens: List[str],
    horizon_s: float = 2.0,
) -> Dict[str, str]:
    """Build token -> resolved scenario_type using full 10Hz PKL frames."""
    tokens_by_log: Dict[str, List[str]] = {}
    for token in tokens:
        log_name = token_to_log_name.get(token)
        if log_name is None:
            continue
        tokens_by_log.setdefault(log_name, []).append(token)

    resolved: Dict[str, str] = {}
    for log_name, log_tokens in tokens_by_log.items():
        pkl_path = Path(navsim_log_path) / f"{log_name}.pkl"
        if not pkl_path.is_file():
            logger.warning(
                "Missing log PKL for scenario_type labeling: %s; "
                "defaulting %d tokens to %s",
                pkl_path,
                len(log_tokens),
                DEFAULT_SCENARIO_TYPE,
            )
            for token in log_tokens:
                resolved[token] = DEFAULT_SCENARIO_TYPE
            continue

        with open(pkl_path, "rb") as f:
            frames: List[Dict[str, Any]] = pickle.load(f)

        token_to_idx = {frame["token"]: idx for idx, frame in enumerate(frames)}
        scenario_types = [frame.get("scenario_type") for frame in frames]
        timestamps_us = [int(frame["timestamp"]) for frame in frames]

        for token in log_tokens:
            current_idx = token_to_idx.get(token)
            if current_idx is None:
                logger.warning(
                    "Token %s not found in %s; defaulting scenario_type to %s",
                    token,
                    pkl_path,
                    DEFAULT_SCENARIO_TYPE,
                )
                resolved[token] = DEFAULT_SCENARIO_TYPE
                continue
            resolved[token] = resolve_scenario_type_label(
                scenario_types,
                timestamps_us,
                current_idx,
                horizon_s=horizon_s,
            )
    return resolved
