"""Camera-path availability checks shared by eager and indexed scene selection."""
from typing import Any, Dict, Mapping, Sequence, Optional

# Bit positions are persisted in raw log metadata. Preserve this order.
CAMERA_NAMES = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")
CAMERA_BITS = {view: 1 << index for index, view in enumerate(CAMERA_NAMES)}


def camera_presence_mask(camera_dict: Mapping[str, Any]) -> int:
    """Match Cameras.from_camera_dict's case folding and empty data_path handling."""
    mask = 0
    for name, camera in camera_dict.items():
        if camera and camera.get("data_path"):
            mask |= CAMERA_BITS.get(name.lower(), 0)
    return mask


def camera_requirements(sensor_config: Any, num_history_frames: int) -> Dict[str, list]:
    """Use the same views and history iterations that the agent actually loads."""
    required = {}
    for index in range(num_history_frames):
        for view in sensor_config.get_sensors_at_iteration(index):
            if view in CAMERA_BITS:
                required.setdefault(view, []).append(index)
    return required


def camera_grid_error(
    history_masks: Sequence[int], required_camera_frames: Mapping[str, Sequence[int]],
) -> Optional[str]:
    """Allow same-view padding when any requested frame exists; reject empty views."""
    missing = [view for view, indices in required_camera_frames.items()
               if not any(history_masks[index] & CAMERA_BITS[view] for index in indices)]
    if not missing:
        return None
    selected = {view: list(required_camera_frames[view]) for view in missing}
    return f"所选相机整路缺失：{missing}；所选历史帧下标：{selected}"
