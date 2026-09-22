"""基于仓库内 Qwen-Drive 模型补充训练 attention mask 和可配置相机视角。

原版网络与预处理代码保存在 modeling/，本模块仅负责训练和多视角适配。
默认三视角 prompt、无 padding 的前向及采样均由回归测试与原版实现对齐。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modeling.planning_expert import _apply_rotary, _one_hot
from .modeling.scene import CAMERA_VIEWS, NAV_COMMANDS, REASONING_REQUEST, DrivingScene, QwenDriveProcessor

NAV_COMMAND_TEXT_MARKER = "2. Active navigation command: ["
NAV_COMMAND_NONE_TEXT = "NONE"


# --------------------------------------------------------------------------- #
# a. 带 attention mask 的 layer forward
# --------------------------------------------------------------------------- #


def masked_attend(
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """官方 `_attend` 的带 mask 版本。

    :param query: ``[B, L_q, heads, head_dim]``
    :param key/value: ``[B, L_kv, kv_heads, head_dim]``
    :param attn_mask: ``[B, 1, 1, L_kv]`` 的 bool（True=可见）；None 时全可见。
    :return: ``[B, L_q, heads, head_dim]``

    mask 非 None 时强制走 SDPA：flash_attn_func 不接受任意 mask，用 varlen 接口才行，
    而 SDPA 已经够用（K/V 是 no_grad 产出的，attention 本身不是瓶颈）。
    """
    if attn_mask is None and getattr(layer, "attn_implementation", "sdpa") == "flash_attention_2":
        from flash_attn import flash_attn_func

        return flash_attn_func(query, key, value, causal=False)

    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask,
        enable_gqa=layer.num_heads != layer.num_kv_heads,
    )
    return out.transpose(1, 2)


def masked_layer_forward(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    scene_key: torch.Tensor,
    scene_value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    condition: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """官方 `PlanningExpertLayer.forward` 的带 mask 版本，权重全部复用 ``layer`` 自身。

    逐行对应官方实现（`planning_expert.py:221-253`）；唯一差别是 attend 传入 mask。
    mask=None 时输出与官方 forward 数值一致，由测试断言。
    """
    batch, length, _ = hidden_states.shape
    modulation = layer.adaln_modulation(condition).chunk(6, dim=-1)
    shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
        m.unsqueeze(1) for m in modulation
    )

    residual = hidden_states
    x = layer.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
    query, gate, key, value = layer._split_qkv(layer.qkv_proj(x))
    query = _apply_rotary(layer.q_norm(query), cos, sin)
    key = _apply_rotary(layer.k_norm(key), cos, sin)

    attn = masked_attend(
        layer,
        query,
        torch.cat([scene_key, key], dim=1),
        torch.cat([scene_value, value], dim=1),
        attn_mask,
    ).reshape(batch, length, -1)
    attn = attn * torch.sigmoid(gate.reshape(batch, length, -1))
    hidden_states = residual + (1 + gate_attn) * layer.o_proj(attn)

    residual = hidden_states
    x = layer.post_attention_layernorm(hidden_states) * (1 + scale_ffn) + shift_ffn
    swiglu_gate, swiglu_up = layer.gate_up_proj(x).chunk(2, dim=-1)
    return residual + (1 + gate_ffn) * layer.down_proj(F.silu(swiglu_gate) * swiglu_up)


def predict_endpoint_masked(
    expert: torch.nn.Module,
    waypoints: torch.Tensor,
    flow_time: torch.Tensor,
    history_queries: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scene_cache: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    position_anchor: torch.Tensor,
    nav_command: torch.Tensor,
    ego_status: torch.Tensor,
    scene_valid: Optional[torch.Tensor] = None,
    gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """官方 `PlanningExpert.predict_endpoint` 的带 mask 版本（训练/batch 推理用）。

    :param scene_valid: ``[B, L_scene]`` bool，True=真实 token，False=padding。
        None 时行为与官方一致。
    :return: ``[B, num_points, 3]`` float32 的 clean endpoint 预测。

    与官方唯一的语义差异是 padding 位置被屏蔽；其余（dtype、Fourier 频表、mRoPE 相位、
    condition 组成）全部走官方模块，不重算。
    """
    dtype = expert.dtype
    waypoints = waypoints.to(dtype)
    batch, length, _ = waypoints.shape
    pose_query, velocity_query, acceleration_query = history_queries

    time_condition = expert.time_mlp(expert.time_embed(flow_time).to(dtype))
    waypoint_index = torch.arange(length, device=waypoints.device)
    broadcast = [
        expert.trajectory_proj(waypoints),
        expert.fourier_encoder(waypoints),
        time_condition.unsqueeze(1).expand(-1, length, -1),
        pose_query.unsqueeze(1).expand(-1, length, -1),
        expert.waypoint_embed(waypoint_index).unsqueeze(0).expand(batch, -1, -1).to(dtype),
        velocity_query.unsqueeze(1).expand(-1, length, -1),
        acceleration_query.unsqueeze(1).expand(-1, length, -1),
    ]
    hidden_states = expert.query_fusion(torch.cat(broadcast, dim=-1))

    nav_onehot = _one_hot(nav_command, expert.config.nav_command_classes, dtype)
    condition = time_condition + expert.nav_mlp(nav_onehot) + expert.ego_mlp(ego_status.to(dtype))

    positions = expert._waypoint_positions(position_anchor, length)
    cos, sin = expert.rotary_emb(positions, dtype)

    attn_mask: Optional[torch.Tensor] = None
    if scene_valid is not None:
        # waypoint token 之间恒可见，拼在 scene 之后（官方 attend 的 K/V 顺序）。
        waypoint_valid = torch.ones(
            batch, length, dtype=torch.bool, device=scene_valid.device
        )
        attn_mask = torch.cat([scene_valid, waypoint_valid], dim=1)[:, None, None, :]

    layers_per_kv = expert.config.layers_per_kv
    for index, layer in enumerate(expert.layers):
        scene_key, scene_value = scene_cache[index // layers_per_kv]
        arguments = (
            layer, hidden_states, scene_key.to(dtype), scene_value.to(dtype),
            cos, sin, condition, attn_mask,
        )
        if gradient_checkpointing and expert.training and torch.is_grad_enabled():
            hidden_states = checkpoint(masked_layer_forward, *arguments, use_reentrant=False)
        else:
            hidden_states = masked_layer_forward(*arguments)
    return expert.out_proj(expert.final_layernorm(hidden_states)).float()


@torch.no_grad()
def sample_masked(
    expert: torch.nn.Module,
    scene_cache: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    position_anchor: torch.Tensor,
    history: torch.Tensor,
    history_velocity: torch.Tensor,
    history_acceleration: torch.Tensor,
    nav_command: torch.Tensor,
    ego_status: torch.Tensor,
    noise: torch.Tensor,
    num_steps: int,
    min_one_minus_t: float,
    scene_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """官方 `PlanningExpert.sample` 的带 mask 版本，Euler 积分逻辑逐行照抄官方。"""
    history_queries = expert.encode_history(
        history, nav_command, history_velocity, history_acceleration
    )
    waypoints = noise.float()
    step = 1.0 / num_steps
    for index in range(num_steps):
        flow_time = torch.full(
            (waypoints.shape[0],), index * step, device=waypoints.device, dtype=torch.float32
        )
        endpoint = predict_endpoint_masked(
            expert,
            waypoints,
            flow_time,
            history_queries,
            scene_cache,
            position_anchor,
            nav_command,
            ego_status,
            scene_valid=scene_valid,
        )
        remaining = max(1.0 - index * step, min_one_minus_t)
        waypoints = waypoints + (endpoint - waypoints) / remaining * step
    return waypoints


# --------------------------------------------------------------------------- #
# b. 可配置视角顺序的 scene / processor（保留 8 路相机接口）
# --------------------------------------------------------------------------- #


@dataclass
class MultiViewScene:
    """官方 `DrivingScene` 的视角可配置版本。

    官方 `DrivingScene` 把视角顺序硬编码在模块级 `CAMERA_VIEWS`（3 项）上：
    `__post_init__` 校验这 3 项存在，`frames_in_order` 也只遍历它们。要支持 8 路相机
    实验就必须让顺序可配置。默认 `view_order` 传官方 3 项时，本类与官方行为一致。

    字段语义与官方 `DrivingScene` 相同：所有轨迹量在当前帧 ego 系，x 前 y 左，
    heading 左转为正；history oldest -> newest，末行是当前帧 (0,0,0)。
    """

    views: Mapping[str, Sequence[Any]]
    history: Any
    history_velocity: Any
    history_acceleration: Any
    ego_velocity: Sequence[float]
    ego_acceleration: Sequence[float]
    driving_command: Sequence[float]
    nav_command: int
    view_order: Tuple[str, ...]
    instruction_text: Optional[str] = None
    token: str = ""

    def __post_init__(self) -> None:
        import numpy as np

        self.history = np.asarray(self.history)
        self.history_velocity = np.asarray(self.history_velocity)
        self.history_acceleration = np.asarray(self.history_acceleration)
        missing = [v for v in self.view_order if v not in self.views]
        if missing:
            raise ValueError(f"scene {self.token!r} 缺少视角 {missing}")

    @property
    def ego_status(self):
        """velocity, acceleration, one-hot driving command —— 顺序与官方一致。"""
        import numpy as np

        return np.concatenate(
            [
                np.asarray(self.ego_velocity, dtype=np.float32),
                np.asarray(self.ego_acceleration, dtype=np.float32),
                np.asarray(self.driving_command, dtype=np.float32),
            ]
        )

    @property
    def num_camera_frames(self) -> int:
        return len(self.views[self.view_order[0]])

    def frames_in_order(self) -> List[Any]:
        """所有帧，先按视角分组、组内按时间 —— 与官方排列一致。"""
        return [frame for view in self.view_order for frame in self.views[view]]

    def instruction(self) -> str:
        """委托官方 `DrivingScene.instruction` 生成文案，只修正越界 nav_command。

        官方实现只用到 history / nav_command / num_camera_frames，与视角集合无关，
        所以这里借官方类算文案（视角只填必需的 3 项占位）。

        **越界 nav_command 的处理**：官方末行是
        ``NAV_COMMANDS[int(self.nav_command)]``。Python 负索引会把 -1 静默解释成
        ``NAV_COMMANDS[-1] == "TURN RIGHT"``，而张量侧 `_one_hot(-1, 3)` 给的是全零行
        （"无命令"）。两个通道会自相矛盾：文字说右转、one-hot 说没命令。Mogo 的
        u-turn / unknown 正好映射到 -1，所以这条路径一定会被走到。
        这里把文字侧改成 ``[NONE]``，与全零 one-hot 语义一致。
        """
        if self.instruction_text is not None:
            return self.instruction_text
        placeholder = {view: self.views[self.view_order[0]] for view in CAMERA_VIEWS}
        index = int(self.nav_command)
        in_range = 0 <= index < len(NAV_COMMANDS)
        proxy = DrivingScene(
            views=placeholder,
            history=self.history,
            history_velocity=self.history_velocity,
            history_acceleration=self.history_acceleration,
            ego_velocity=self.ego_velocity,
            ego_acceleration=self.ego_acceleration,
            driving_command=self.driving_command,
            # 越界时先用合法索引生成，再替换命令文字；否则官方会走负索引。
            nav_command=index if in_range else 0,
        )
        text = proxy.instruction()
        if in_range:
            return text
        head, marker, _ = text.rpartition(NAV_COMMAND_TEXT_MARKER)
        if not marker:
            raise RuntimeError(
                "官方 instruction 文案格式已变（未找到 "
                f"{NAV_COMMAND_TEXT_MARKER!r}），无法安全改写越界 nav_command"
            )
        return f"{head}{marker}{NAV_COMMAND_NONE_TEXT}]"


def make_multi_view_processor(base_processor: Any) -> Any:
    """给官方 processor 打一层视角可配置的壳。

    官方 `build_input_ids` 遍历模块级 `CAMERA_VIEWS`；这里改成遍历 scene 自带的
    `view_order`。默认 3 视角时与官方逐 token 一致（测试断言）。
    """
    class MultiViewProcessor(QwenDriveProcessor):
        def build_input_ids(self, scene, token_counts, with_reasoning):
            view_order = getattr(scene, "view_order", CAMERA_VIEWS)
            if tuple(view_order) == tuple(CAMERA_VIEWS):
                return super().build_input_ids(scene, token_counts, with_reasoning)
            body: List[int] = []
            per_view = scene.num_camera_frames
            for view_index, view in enumerate(view_order):
                body += self._encode(view)
                for frame_index in range(per_view):
                    body += self._encode(f"frame: {frame_index}")
                    count = token_counts[view_index * per_view + frame_index]
                    body += (
                        [self.vision_start_id]
                        + [self.image_token_id] * count
                        + [self.vision_end_id]
                    )
            instruction = scene.instruction()
            if with_reasoning:
                instruction += REASONING_REQUEST
            body += self._encode(instruction)

            assistant_header = [self.im_start_id] + self._encode("assistant") + self.newline_ids
            prompt = (
                [self.im_start_id]
                + self._encode("user")
                + self.newline_ids
                + body
                + [self.im_end_id]
                + self.newline_ids
                + assistant_header
            )
            if not with_reasoning:
                prompt += [self.im_end_id] + self.newline_ids
            return prompt

    return MultiViewProcessor(base_processor.tokenizer, base_processor.config)
