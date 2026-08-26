import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)



class SinusoidalPositionalEncoding(nn.Module):
    """
    Generates a sinusoidal encoding of shape (B, T, D) given timesteps of
    shape (B, T).
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps (torch.Tensor): A tensor of shape (B, T).

        Returns:
            torch.Tensor: The positional encoding. Shape: (B, T, D).
        """
        timesteps = timesteps.float()
        device = timesteps.device
        half_dim = self.embedding_dim // 2

        exponent = -torch.log(torch.tensor(10000.0)) / half_dim
        exponent = torch.arange(half_dim, dtype=torch.float32, device=device) * exponent

        freqs = timesteps.unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        encoding = torch.cat([sin, cos], dim=-1)
        return encoding


class ActionEncoder(nn.Module):
    """
    Encodes a sequence of actions and their corresponding timesteps into a
    fixed-size embedding sequence.
    """
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

        self.fc1 = nn.Linear(action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions (torch.Tensor): Action sequence. Shape: (B, T, action_dim).
            timesteps (torch.Tensor): Timestep for each action. Shape: (B, T).

        Returns:
            torch.Tensor: The encoded action sequence. Shape: (B, T, hidden_size).
        """
        B, T, _ = actions.shape

        action_embedding = self.fc1(actions)

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        time_embedding = self.pos_encoding(timesteps).to(dtype=action_embedding.dtype)

        x = torch.cat([action_embedding, time_embedding], dim=-1)
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return x


class StateAttentionEncoder(nn.Module):
    """
    State Dropout Encoder (SDE)，参考 pluto 的自车状态编码：把 ``state_dim`` 维 status_feature
    的每一维单独投影成 token，再用一个可学习 query 做多头注意力池化成单个 embedding。

    ``state_dim`` / ``num_command_states`` 均为构造参数，随 agent 配置变化，不写死：
    - 默认（use_navigation_command=False）：state_dim=13，前 num_command_states=4 维是旧
      4 维 one-hot 导航命令(cmd_0..cmd_3)。
    - 启用 navigation_command：state_dim=30，前 num_command_states=NAV_COMMAND_DIM=21 维是
      navigation_command 编码段（主命令 type/dist/车道范围 + 次命令 + park 的 dist/侧向）。
    agent 通过 self._ego_status_dim / self._ego_num_command_states 计算后经 planner config
    传入（见 recogdrive_agent.py），二者对应 recogdrive_features.ego_status_feature_names(...)
    的布局。

    关键在于训练时对状态做 dropout 以抑制"捷径学习"：status_feature 的前
    ``num_command_states`` 维是导航命令/意图——不是捷径，永久可见；其余维是自车运动学量
    (vx/vy/ax/ay/yaw_rate/curvature/curvature_rate/steering_wheel_angle/steering_wheel_rate)
    ——有它们模型能直接外推 4s 轨迹，故按 ``state_dropout`` 概率随机遮蔽，逼模型转向
    地图/障碍物/站点/VLM 上下文。

    注意与 pluto 的差异：pluto 前 3 维是 ego 系恒为 0 的位置/朝向(永久可见)，此处对应的
    永久可见维是导航命令段。掩码拼接顺序 [command(可见), kinematic(可丢)] 必须与
    recogdrive_features.ego_status_feature_names(use_navigation_command) 的
    [命令段在前, 运动学量在后] 布局一致。

    本编码器与 agent.vlm_ablation 正交：dropout 只作用于 ego status token，
    vlm_ablation=none / zero 下均生效。
    """
    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        num_command_states: int,
        state_dropout: float = 0.75,
        num_heads: int = 4
    ):
        super().__init__()
        assert state_dim > num_command_states, \
            "state_dim must be greater than num_command_states (need droppable kinematic states)"

        self.state_dim = state_dim
        self.num_command_states = num_command_states
        self.state_dropout = state_dropout

        self.linears = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(state_dim)
        ])
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.pos_embed = nn.Parameter(torch.Tensor(1, state_dim, embed_dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes learnable embeddings."""
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The flat state vector. Shape: (B, state_dim).

        Returns:
            torch.Tensor: The encoded state embedding. Shape: (B, embed_dim).
        """
        x_embed_list = [
            linear(x[:, i, None]) for i, linear in enumerate(self.linears)
        ]
        x_embed = torch.stack(x_embed_list, dim=1)
        x_embed = x_embed + self.pos_embed

        key_padding_mask = None
        if self.training and self.state_dropout > 0:
            # 命令维度(前 num_command_states 维)永久可见：False = 不遮蔽。
            command_mask = torch.zeros(
                (x_embed.shape[0], self.num_command_states),
                device=x.device,
                dtype=torch.bool,
            )
            # 运动学量(其余维)按 state_dropout 随机遮蔽：True = 遮蔽(query 注意不到)。
            kinematic_mask = torch.rand(
                (x_embed.shape[0], self.state_dim - self.num_command_states),
                device=x.device,
            ) < self.state_dropout
            # 命令在前、运动学量在后，与 EGO_STATUS_FEATURE_NAMES 布局一致；命令永久可见
            # 保证至少 num_command_states 个有效 key，MHA 不会出现整行全遮蔽的 NaN。
            key_padding_mask = torch.cat([command_mask, kinematic_mask], dim=1)

        query = self.query.expand(x_embed.shape[0], -1, -1)
        x_state, _ = self.attn(
            query=query,
            key=x_embed,
            value=x_embed,
            key_padding_mask=key_padding_mask,
        )

        return x_state.squeeze(1)