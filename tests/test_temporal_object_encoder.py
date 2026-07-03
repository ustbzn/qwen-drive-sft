"""时序障碍物编码器（TemporalObjectEncoder / ObjectTemporalFeatureBuilder）单测。

无 GPU / 无数据依赖，覆盖两处易回归的正确性核心：
1. 坐标变换：历史帧对象统一到当前 ego 系后，世界静止物的帧间位移应≈0
   （证明 ego 自运动被正确扣除，是差分编码物理正确性的基石）。
2. 模型侧 encoder：输出形状、零影响启动（motion_proj 零初始化）、padding 置零、
   短序列（T_delta=1）鲁棒、motion 分支确实接入。

运行：
    python -m pytest tests/test_temporal_object_encoder.py -q
"""

import math

import numpy as np
import pytest
import torch

from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.agents.recogdrive.object_temporal_features import (
    _local_to_global_se2,
    _rotate_velocity,
    OBJECT_DIFF_DIM,
)
from navsim.agents.recogdrive.recogdrive_diffusion_planner import (
    TemporalObjectEncoder,
    TemporalPyramidEncoder,
)


def _to_current(local_xyt, ego_f, ego_c):
    """把帧 f ego-local 的单个 SE2 状态变换到当前帧 c ego-local。"""
    g = _local_to_global_se2(np.array([local_xyt], dtype=np.float64), np.array(ego_f, dtype=np.float64))
    return convert_absolute_to_relative_se2_array(StateSE2(*[float(v) for v in ego_c]), g)[0]


# --------------------------- 坐标变换 --------------------------- #

def test_static_object_zero_displacement_after_unify():
    """世界静止物：不同历史帧的观测统一到当前 ego 系后应是同一点（帧间位移≈0）。"""
    ego_f = [0.0, 0.0, 0.0]
    ego_c = [4.4, 0.0, 0.0]  # ego 沿 +x 前进 4.4m
    # 世界静止物 global [20,0]：帧 f(ego@0) local=[20,0]，帧 c(ego@4.4) local=[15.6,0]
    tf = _to_current([20.0, 0.0, 0.0], ego_f, ego_c)
    tc = _to_current([15.6, 0.0, 0.0], ego_c, ego_c)
    assert np.allclose(tf[:2], [15.6, 0.0], atol=1e-3)
    assert np.allclose(tc[:2], [15.6, 0.0], atol=1e-3)
    # 关键：统一系后位移≈0；若不做变换，dx 会是 15.6-20=-4.4（被 ego 运动污染）
    assert abs(tc[0] - tf[0]) < 1e-3 and abs(tc[1] - tf[1]) < 1e-3


def test_rotation_transform():
    """ego 朝 +y(π/2) 时，全局 [10,0] 应落到 ego 右侧 [0,-10]。"""
    r = convert_absolute_to_relative_se2_array(StateSE2(0.0, 0.0, math.pi / 2), np.array([[10.0, 0.0, 0.0]]))[0]
    assert np.allclose(r[:2], [0.0, -10.0], atol=1e-3)


def test_velocity_rotation():
    """速度矢量 [5,0] 从朝 0 的帧旋到朝 π/2 的帧，应变为 [0,-5]。"""
    vc = _rotate_velocity(np.array([[5.0, 0.0]]), yaw_from=0.0, yaw_to=math.pi / 2)[0]
    assert np.allclose(vc, [0.0, -5.0], atol=1e-3)


# --------------------------- 模型侧 encoder --------------------------- #

def _make_encoder(embed_dim=32, temporal_dim=16, dropout=0.0):
    return TemporalObjectEncoder(
        feature_dim=15, diff_dim=OBJECT_DIFF_DIM, embed_dim=embed_dim,
        num_types=5, hidden_dim=64, temporal_dim=temporal_dim, dropout=dropout,
    )


def _inputs(B=2, N=64, Td=3):
    return dict(
        obj_features=torch.randn(B, N, 15),
        obj_diff_features=torch.randn(B, N, Td, OBJECT_DIFF_DIM),
        obj_type=torch.randint(0, 5, (B, N)),
        obj_valid_mask=torch.ones(B, N, dtype=torch.bool),
        obj_is_dynamic=torch.ones(B, N, dtype=torch.bool),
    )


def test_encoder_output_shape():
    enc = _make_encoder(embed_dim=32).eval()
    x = _inputs()
    out = enc(x["obj_features"], x["obj_diff_features"], x["obj_type"], x["obj_valid_mask"], x["obj_is_dynamic"])
    assert out.shape == (2, 64, 32)


def test_motion_proj_zero_init_no_impact():
    """motion_proj 零初始化 -> 动/静 gate 对输出无影响（motion≡0，等价纯 abs），零影响启动。"""
    enc = _make_encoder().eval()
    assert enc.motion_proj.weight.abs().sum().item() == 0
    assert enc.motion_proj.bias.abs().sum().item() == 0
    x = _inputs()
    o_dyn = enc(x["obj_features"], x["obj_diff_features"], x["obj_type"], x["obj_valid_mask"],
                torch.ones(2, 64, dtype=torch.bool))
    o_sta = enc(x["obj_features"], x["obj_diff_features"], x["obj_type"], x["obj_valid_mask"],
                torch.zeros(2, 64, dtype=torch.bool))
    assert torch.allclose(o_dyn, o_sta, atol=1e-6)


def test_padding_slots_are_zero():
    """valid=False 的 padding slot 输出严格为 0（与 ObjectEncoder 一致，cross-attn 无需额外 mask）。"""
    enc = _make_encoder().eval()
    x = _inputs()
    valid = torch.zeros(2, 64, dtype=torch.bool)
    valid[:, :10] = True
    out = enc(x["obj_features"], x["obj_diff_features"], x["obj_type"], valid, x["obj_is_dynamic"])
    assert out[~valid].abs().sum().item() == 0


def test_short_sequence_robustness():
    """T_delta=1（时序金字塔跳过下采样分支）不应报错。"""
    enc = _make_encoder().eval()
    x = _inputs(Td=1)
    out = enc(x["obj_features"], x["obj_diff_features"], x["obj_type"], x["obj_valid_mask"], x["obj_is_dynamic"])
    assert out.shape == (2, 64, 32)


def test_motion_branch_active_when_trained():
    """非零 motion_proj 时，动态 slot 输出应异于静态（motion 分支确实接入并影响输出）。"""
    enc = _make_encoder().train()
    enc.token_dropout = 0.0
    torch.nn.init.normal_(enc.motion_proj.weight, std=0.5)
    torch.nn.init.normal_(enc.motion_proj.bias, std=0.5)
    x = _inputs()
    o_dyn = enc(x["obj_features"], x["obj_diff_features"], x["obj_type"], x["obj_valid_mask"],
                torch.ones(2, 64, dtype=torch.bool))
    o_sta = enc(x["obj_features"], x["obj_diff_features"], x["obj_type"], x["obj_valid_mask"],
                torch.zeros(2, 64, dtype=torch.bool))
    assert not torch.allclose(o_dyn, o_sta)


def test_temporal_pyramid_gradients():
    """时序金字塔可反向：backward 后 tokenizer/fpn 权重有梯度。"""
    pyr = TemporalPyramidEncoder(in_chans=OBJECT_DIFF_DIM, dim=16)
    x = torch.randn(8, OBJECT_DIFF_DIM, 3, requires_grad=True)
    pyr(x).sum().backward()
    assert pyr.tokenizer.weight.grad is not None and pyr.tokenizer.weight.grad.abs().sum() > 0
    assert pyr.fpn_conv.weight.grad is not None and pyr.fpn_conv.weight.grad.abs().sum() > 0
