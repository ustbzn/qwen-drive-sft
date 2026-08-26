from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np


def _command_to_binary_list(command: Any) -> List[int]:
    """Convert a command tensor/array/list into a flattened binary code."""
    if hasattr(command, "detach"):
        command = command.detach().cpu().numpy()
    elif hasattr(command, "cpu") and hasattr(command, "numpy"):
        command = command.cpu().numpy()

    command_array = np.asarray(command, dtype=np.float32).reshape(-1)
    return [int(value >= 0.5) for value in command_array]


def command_code_to_text(command: Any) -> str:
    """Return a prompt-friendly navigation command string from NAVSIM/Mogo command code.

    Legacy NAVSIM uses a 3-dim one-hot command: left / straight / right.
    Mogo keeps the 4-dim NAVSIM field but treats it as a command code, adding
    unknown and u-turn without changing the pkl schema.
    """
    code = _command_to_binary_list(command)

    if len(code) >= 4:
        code4 = tuple(code[:4])
        if code4 == (1, 0, 0, 0):
            return "turn left"
        if code4 == (0, 1, 0, 0):
            return "go straight"
        if code4 == (0, 0, 1, 0):
            return "turn right"
        if code4 == (1, 1, 0, 0):
            return "make a u-turn"
        return "unknown"

    if len(code) == 3:
        legacy_commands = ["turn left", "go straight", "turn right"]
        active_indices = [idx for idx, value in enumerate(code) if value == 1]
        return legacy_commands[active_indices[0]] if len(active_indices) == 1 else "unknown"

    return "unknown"


# --------------------------------------------------------------------------- #
# navigation_command 统一编码（进出站导航）
#
# 新版 Mogo pkl 每帧带 navigation_command：恒为两个四元组 [type, dist, a, b]，
#   type ∈ {None, straight, turn left, turn right, u turn, park}，
#   dist  = 到该动作/站点入口的沿线剩余距离（m），park 的 dist 从远到近递减，
#   (a,b) = 车道范围；对 park，(1,1)=站点在左侧、(-1,-1)=站点在右侧；
#           对转向/直行，(a,b) 是从左/右数的车道序号范围（如 [2,3]），不是简单侧向。
#   语义详见 docs/recogdrive_navigation_command_generation.md。
#
# encode_nav_command_vector 把两个四元组**都**编码进定长向量，替换旧 ego state 的 4 维
# one-hot 命令段。布局（NAV_COMMAND_DIM 维）：
#   [ 主命令 type_onehot(6) | dist_norm | dist_log | lane_a_norm | lane_b_norm     (10)
#   | 次命令 type_onehot(6) | dist_norm                                            (7)
#   | has_park | park_dist_norm | park_dist_log | park_side ]                      (4)
# - 主命令 = 第一条非 park 动作（无则取 park）；次命令 = 第二条非 park 动作（覆盖「下一个
#   路口动作」/提前变道信息）；park 单独抽出（进出站关键信号，与路口动作并存）。
# - lane_a/b：主命令的车道范围两端都编码（park 时同为侧向标记 ±1）。
# - dist 双尺度：dist_norm=dist/200 clamp[0,1]（近距离高分辨率）+ dist_log=log1p(dist)/
#   log1p(1000)（远距离不饱和，保留 200~1000m 的区分度）。
# - park_side 用 park (a,b) 符号（park 的左右侧语义明确正确）；转向的车道范围不当 side。
# - 旧数据无 navigation_command 时，从 4 维 driving_command 退化：仅填主命令 type_onehot，
#   其余段置 0，保证开关开启后旧数据仍可训练。
# --------------------------------------------------------------------------- #

# type 规范顺序（索引即 one-hot 位）。'none' 兼容空槽 / 未识别。
NAV_COMMAND_TYPES: Tuple[str, ...] = (
    "none",
    "straight",
    "turn left",
    "turn right",
    "u turn",
    "park",
)
_NAV_TYPE_TO_ID = {name: idx for idx, name in enumerate(NAV_COMMAND_TYPES)}

# 命令向量维度：主命令(6+1+1+1+1=10) + 次命令(6+1=7) + park(1+1+1+1=4) = 21
_N_TYPES = len(NAV_COMMAND_TYPES)
NAV_COMMAND_DIM: int = (_N_TYPES + 4) + (_N_TYPES + 1) + 4

# dist 归一化尺度（m）。生产扫描上限 1000m，绝大多数动作/站点在 200m 内；dist_norm 用 200
# 归一化 clamp[0,1] 给近距离高分辨率，dist_log 用 log1p 保留远距离区分度（缓解 P4 截断）。
NAV_COMMAND_DIST_NORM: float = 200.0
NAV_COMMAND_DIST_LOG_NORM: float = math.log1p(1000.0)
# 车道序号归一化尺度（一般机动车道数 <= ~10）。
NAV_COMMAND_LANE_NORM: float = 10.0


def _normalize_nav_type(raw_type: Any) -> str:
    """把 navigation_command 的 type 字段规范到 NAV_COMMAND_TYPES 之一。"""
    if raw_type is None:
        return "none"
    name = str(raw_type).strip().lower()
    if name in _NAV_TYPE_TO_ID:
        return name
    # 常见别名兜底
    if name in ("", "nan", "null"):
        return "none"
    if name in ("uturn", "u-turn"):
        return "u turn"
    if name in ("left",):
        return "turn left"
    if name in ("right",):
        return "turn right"
    return "none"


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _norm_dist(dist: Any) -> float:
    """dist(m) -> [0,1]，None/nan 安全置 0。"""
    if dist is None or _is_nan(dist):
        return 0.0
    return float(min(max(float(dist) / NAV_COMMAND_DIST_NORM, 0.0), 1.0))


def _norm_dist_log(dist: Any) -> float:
    """dist(m) -> log 尺度 [0,~1]，缓解 200m 以上距离在 _norm_dist 下的饱和。None/nan→0。"""
    if dist is None or _is_nan(dist):
        return 0.0
    d = max(float(dist), 0.0)
    return float(min(math.log1p(d) / NAV_COMMAND_DIST_LOG_NORM, 1.5))


def _norm_lane(value: Any) -> float:
    """车道序号 a/b -> 归一化浮点（保留符号与大小），None/nan→0。"""
    if value is None or _is_nan(value):
        return 0.0
    return float(np.clip(float(value) / NAV_COMMAND_LANE_NORM, -1.5, 1.5))


def _side_from_ab(a: Any, b: Any) -> float:
    """从四元组的 (a,b) 车道范围取侧向标记：>0=左(+1)，<0=右(-1)，其余=0。

    park 的 (a,b)=(1,1) 左侧、(-1,-1) 右侧；路口动作的 (a,b) 是车道序号，符号同样
    区分左右侧编码（左转/掉头触及最左用正数，右转/靠边停车触及最右用负数），故用符号
    作为粗粒度侧向信号即可（第一版不解析具体车道号）。
    """
    if a is None or _is_nan(a):
        return 0.0
    a_val = float(a)
    if a_val > 0:
        return 1.0
    if a_val < 0:
        return -1.0
    return 0.0


def _parse_quad(quad: Any) -> Optional[Tuple[str, Any, Any, Any]]:
    """解析单条四元组 [type, dist, a, b]；非法/空槽返回 None。"""
    if quad is None:
        return None
    try:
        raw_type = quad[0]
        dist = quad[1] if len(quad) > 1 else None
        a = quad[2] if len(quad) > 2 else None
        b = quad[3] if len(quad) > 3 else None
    except (TypeError, IndexError, KeyError):
        return None
    nav_type = _normalize_nav_type(raw_type)
    if nav_type == "none":
        return None
    return nav_type, dist, a, b


def encode_nav_command_vector(
    navigation_command: Any,
    driving_command: Any = None,
) -> np.ndarray:
    """把 navigation_command 的两个四元组**都**编码成定长 NAV_COMMAND_DIM 向量。

    :param navigation_command: 新版两个四元组 [[type,dist,a,b],[...]]；None/空则退化。
    :param driving_command: 旧版 4 维 one-hot；navigation_command 缺失时用于退化填主命令 type。
    :return: float32 向量，布局
        [主命令 type_onehot(6)|dist_norm|dist_log|lane_a|lane_b
        | 次命令 type_onehot(6)|dist_norm
        | has_park|park_dist_norm|park_dist_log|park_side]

    - 主命令 = 第一条非 park 动作（无则 park）；次命令 = 第二条非 park 动作（下一路口动作/
      提前变道，无则留 0）；park 单独抽出。
    - 旧数据（navigation_command 空）：仅按 driving_command 填主命令 type_onehot，其余段 0。
    """
    vec = np.zeros(NAV_COMMAND_DIM, dtype=np.float32)
    # 段偏移：主命令[0:10]，次命令 type[10:16]/dist[16]，park[17:21]
    off_second = _N_TYPES + 4
    off_park = off_second + _N_TYPES + 1

    quads: List[Tuple[str, Any, Any, Any]] = []
    if navigation_command is not None:
        try:
            for quad in navigation_command:
                parsed = _parse_quad(quad)
                if parsed is not None:
                    quads.append(parsed)
        except TypeError:
            quads = []

    if not quads:
        # --- 旧数据退化：从 4 维 driving_command 映射主命令 type，其余段留 0 ---
        text = command_code_to_text(driving_command) if driving_command is not None else "unknown"
        text_to_type = {
            "go straight": "straight",
            "turn left": "turn left",
            "turn right": "turn right",
            "make a u-turn": "u turn",
        }
        nav_type = text_to_type.get(text, "none")
        vec[_NAV_TYPE_TO_ID[nav_type]] = 1.0
        return vec

    # --- 新数据：拆出 park 与非 park 命令序列 ---
    park_quad = next((q for q in quads if q[0] == "park"), None)
    non_park = [q for q in quads if q[0] != "park"]
    main_quad = non_park[0] if non_park else park_quad
    second_quad = non_park[1] if len(non_park) > 1 else None

    # 主命令：type + 双尺度 dist + 车道范围两端
    main_type, main_dist, main_a, main_b = main_quad
    vec[_NAV_TYPE_TO_ID[main_type]] = 1.0
    vec[_N_TYPES] = _norm_dist(main_dist)
    vec[_N_TYPES + 1] = _norm_dist_log(main_dist)
    vec[_N_TYPES + 2] = _norm_lane(main_a)
    vec[_N_TYPES + 3] = _norm_lane(main_b)

    # 次命令：type + dist_norm（下一个路口动作 / 提前变道信息）
    if second_quad is not None:
        second_type, second_dist, _, _ = second_quad
        vec[off_second + _NAV_TYPE_TO_ID[second_type]] = 1.0
        vec[off_second + _N_TYPES] = _norm_dist(second_dist)

    # park：进出站关键信号，与路口动作并存
    if park_quad is not None:
        _, park_dist, park_a, park_b = park_quad
        vec[off_park] = 1.0                              # has_park
        vec[off_park + 1] = _norm_dist(park_dist)        # park_dist_norm
        vec[off_park + 2] = _norm_dist_log(park_dist)    # park_dist_log
        vec[off_park + 3] = _side_from_ab(park_a, park_b)  # park_side（±1 语义明确）

    return vec


def parse_park_command(navigation_command: Any) -> Optional[Tuple[float, float]]:
    """抽取当前活动 park（进出站）命令的 (dist_m, side)，无 park 返回 None。

    :return: (dist_m, side)；dist_m = 到站沿线剩余距离(米，None/nan→0)，
        side = +1 左侧站 / -1 右侧站 / 0 未知。供 station_features 派生站点 token 用。
    """
    if navigation_command is None:
        return None
    try:
        for quad in navigation_command:
            parsed = _parse_quad(quad)
            if parsed is not None and parsed[0] == "park":
                _, dist, a, b = parsed
                dist_val = 0.0 if (dist is None or _is_nan(dist)) else float(dist)
                return dist_val, _side_from_ab(a, b)
    except TypeError:
        return None
    return None
