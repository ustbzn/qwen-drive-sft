"""Shared helpers for ReCogDrive multi-view / multi-frame VLM inputs.

This module is the single source of truth for:
  1. turning ``AgentInput.cameras`` into a ``[view][frame]`` matrix of image
     paths (with edge-padding for missing views/frames), and
  2. building the driving prompt text.

Both the cache-mode feature builder (``recogdrive_features.py``) and the online
eval path (``recogdrive_agent.py``) import from here so the image selection and
prompt wording stay byte-for-byte identical between the two paths.

Design notes (see plan deep-churning-shannon):
  - ``cameras`` is ordered oldest -> newest, ``cameras[-1]`` is the current
    frame. Multi-frame clips are taken from the tail (oldest -> newest) to match
    the ``history_trajectory`` ordering (t-3 .. t-0).
  - View count is ``len(view_list)`` (single source of truth, no ``num_views``).
  - ReCogDrive caches VLM hidden states, not images. More views/frames only make
    the hidden-state sequence longer (hidden_dim stays 2048), which the planner
    and collate already tolerate via mean-pooling / pad_sequence.
"""

from typing import List, Optional

import torch

# 视角字段名 -> prompt 语义标签。措辞沿用 navsim/agents/intervl_agent.py 的 6 视角写法。
# 注意：mogo pkl 的 cam_l2 / cam_r2 为空，多视角实际应使用真实存在的 cam_l1 / cam_r1
# （真实左后 / 右后）。这里同时给出两套后侧映射，便于不同 view_list 选择。
VIEW_LABELS = {
    "cam_f0": "FRONT VIEW",
    "cam_l0": "FRONT LEFT VIEW",
    "cam_r0": "FRONT RIGHT VIEW",
    "cam_l1": "BACK LEFT VIEW",
    "cam_r1": "BACK RIGHT VIEW",
    "cam_l2": "BACK LEFT VIEW",
    "cam_r2": "BACK RIGHT VIEW",
    "cam_b0": "BACK VIEW",
}


def _camera_image_path(camera_obj) -> Optional[str]:
    """从单个 Camera dataclass 取图像路径；空相机 / 空路径返回 None。"""
    img = getattr(camera_obj, "image", None)
    if img is None:
        return None
    s = str(img)
    if not s or s == "None":
        return None
    return s


def _edge_pad(paths: List[Optional[str]]) -> List[Optional[str]]:
    """对单视角的帧路径序列做 edge-pad：用前 / 后最近的有效帧填充 None。

    先正向填充（用前一个有效帧），再反向填充（用后一个有效帧）。只要序列里至少
    有一个有效帧，返回的序列就全部非 None；整段为 None 时原样返回（由调用方处理）。
    """
    n = len(paths)
    filled: List[Optional[str]] = list(paths)
    # 正向：用上一个有效帧填后面的 None
    last_valid: Optional[str] = None
    for i in range(n):
        if filled[i]:
            last_valid = filled[i]
        elif last_valid is not None:
            filled[i] = last_valid
    # 反向：用下一个有效帧填前面残留的 None
    next_valid: Optional[str] = None
    for i in range(n - 1, -1, -1):
        if filled[i]:
            next_valid = filled[i]
        elif next_valid is not None:
            filled[i] = next_valid
    return filled


def collect_view_frame_paths(
    cameras,
    view_list: List[str],
    num_frames: int = 1,
) -> List[List[str]]:
    """从 ``AgentInput.cameras`` 取出 ``[view][frame]`` 的图像路径矩阵。

    :param cameras: ``AgentInput.cameras``，``List[Cameras]``，oldest -> newest，
        ``cameras[-1]`` 为当前帧。
    :param view_list: 相机字段名列表（如 ``['cam_f0']`` 或
        ``['cam_f0', 'cam_b0', 'cam_l0', 'cam_r0']``），顺序即 prompt 标签顺序。
    :param num_frames: 从末尾往前取的帧数（oldest -> newest），上限 ``len(cameras)``。
    :return: ``out[i]`` 为第 i 个视角（顺序同 ``view_list``）的一段帧路径序列
        （oldest -> newest）。单帧时长度为 1。所有元素均为有效路径（已 edge-pad）。
    """
    if not view_list:
        view_list = ["cam_f0"]
    n_frames = max(1, min(int(num_frames), len(cameras)))
    frame_slice = cameras[len(cameras) - n_frames:]  # oldest -> newest

    # 1. 逐视角逐帧取原始路径，并做帧级 edge-pad
    raw: List[List[Optional[str]]] = []
    for view in view_list:
        per_view = [_camera_image_path(getattr(cams, view, None)) for cams in frame_slice]
        raw.append(_edge_pad(per_view))

    # 2. 视角级 edge-pad：整视角全空时，复制第一个有有效帧的视角
    fallback_view: Optional[List[str]] = None
    for per_view in raw:
        if all(p for p in per_view):
            fallback_view = [str(p) for p in per_view]
            break
    if fallback_view is None:
        # 所有视角都至少缺一帧（含全空）；尝试用任意有效帧拼一个兜底序列
        any_path = next((p for per_view in raw for p in per_view if p), None)
        if any_path is None:
            raise RuntimeError(
                "collect_view_frame_paths: no valid camera image found for any "
                f"view in {view_list}; cannot build VLM input."
            )
        fallback_view = [any_path] * n_frames

    out: List[List[str]] = []
    for per_view in raw:
        if all(p for p in per_view):
            out.append([str(p) for p in per_view])
        else:
            # 该视角整段不可用 -> 用兜底视角复制（决策 5：edge-pad，不报错）
            out.append(list(fallback_view))
    return out


def flatten_view_frames(view_frame_paths: List[List[str]]) -> List[str]:
    """按 (view, frame) 顺序扁平成单层路径列表（``images=`` 路径用，每帧一张图）。"""
    return [p for per_view in view_frame_paths for p in per_view]


def view_labels_for(view_list: List[str]) -> List[str]:
    """返回与 ``view_list`` 对应的 prompt 标签；缺失映射回退到字段名大写。"""
    if not view_list:
        view_list = ["cam_f0"]
    return [VIEW_LABELS.get(v, v.upper()) for v in view_list]


def build_history_str(history_trajectory: torch.Tensor) -> str:
    """构造历史轨迹文案 ``   - t-3: (x, y, heading) ...``（oldest -> newest）。

    与改造前 recogdrive_features.py / recogdrive_agent.py 两处文案逐字一致。
    ``history_trajectory`` 形状 ``(T, 3)``，T 通常为 4。
    """
    # 延迟 import 避免与 backbone / utils 形成环（format_number 在 utils.utils）。
    from .utils.utils import format_number

    n = history_trajectory.shape[0]
    return " ".join(
        f"   - t-{n - 1 - i}: ("
        f"{format_number(history_trajectory[i, 0].item())}, "
        f"{format_number(history_trajectory[i, 1].item())}, "
        f"{format_number(history_trajectory[i, 2].item())})"
        for i in range(n)
    )


def build_driving_question(
    history_str: str,
    command_str: str,
    view_list: List[str],
    num_frames: int,
    *,
    model_type: str,
    flat_image_count: int,
) -> str:
    """组装与组合（视角数 × 帧数）匹配的完整 question。

    perception 行随组合切换；单视角单帧保持改造前文案，确保零影响。
    InternVL 在 question 文本里放 ``flat_image_count`` 个 ``<image>`` 占位；
    Qwen3 不放占位（由 chat message 的 ``{type:image/video}`` 负责）。
    """
    n_views = len(view_list) if view_list else 1
    multi_frame = int(num_frames) > 1
    if n_views == 1 and not multi_frame:
        perception = "1. Visual perception from front camera view"
    elif n_views > 1 and not multi_frame:
        perception = "1. Visual perception from the surrounding camera views"
    elif n_views == 1 and multi_frame:
        perception = "1. Visual perception from continuous front camera views (temporal)"
    else:
        perception = "1. Visual perception from surrounding multi-view temporal camera clips"

    body = (
        "As an autonomous driving system, predict the vehicle's trajectory based on:\n"
        f"{perception}\n"
        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
        f"3. Active navigation command: [{command_str.upper()}]"
    )
    output_requirements = (
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )

    if model_type == "internvl":
        placeholders = "".join("<image>" for _ in range(max(1, flat_image_count))) + "\n"
        return f"{placeholders}{body}{output_requirements}"

    # Qwen3：占位由 chat message 的 {type:image/video} 负责，question 不放真实占位。
    # 但单视角单帧时保留改造前的字面 "<image>\n" 前缀（被当纯文本处理），逐字复刻
    # 原 prompt，确保 hidden_state 数值零影响。多视角 / 多帧不加该前缀。
    if n_views == 1 and not multi_frame:
        return f"<image>\n{body}{output_requirements}"
    return f"{body}{output_requirements}"
