"""Shared timestamp checks for scene selection and trajectory supervision."""
from typing import Optional, Sequence

import numpy as np


def timestamp_grid_error(
    timestamps: Sequence[int], current_index: int,
    interval_s: float = 0.1, max_error_s: float = 0.075,
) -> Optional[str]:
    """Return a diagnostic for a non-increasing or off-grid microsecond sequence."""
    stamps = np.asarray(timestamps, dtype=np.int64)
    if stamps.ndim != 1 or not 0 <= current_index < len(stamps) or len(stamps) < 2:
        return "scene 时间戳数量不足，无法检查时间网格"
    relative_times = (stamps - stamps[current_index]).astype(np.float64) / 1.0e6
    nominal_times = (np.arange(len(stamps)) - current_index) * interval_s
    max_offset = float(np.max(np.abs(relative_times - nominal_times)))
    intervals = np.diff(stamps).astype(np.float64) / 1.0e6
    # Avoid rejecting an exactly-on-boundary microsecond value due to FP rounding.
    if np.all(intervals > 0) and max_offset <= max_error_s + 1.0e-9:
        return None
    return (
        f"scene 时间戳不满足 {1.0 / interval_s:g}Hz 网格：最大偏差 {max_offset:.6f}s，"
        f"要求时间递增且相对当前帧的偏差不超过 {max_error_s:g}s；"
        f"相邻间隔范围 [{intervals.min():.6f}, {intervals.max():.6f}]s"
    )
