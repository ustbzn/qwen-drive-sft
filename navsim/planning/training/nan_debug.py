"""零侵入的 NaN 定位钩子（默认关闭，由环境变量开启）。

用途：坐实训练时 `train/loss_step: nan` 的**真正源头模块**，而不是靠猜测。给 agent 的每个
子模块注册 forward hook；因为 forward hook 按"模块 forward 完成顺序"触发（子模块先于父模块），
第一个"输出含 NaN"的命中模块，就是最内层最早产生 NaN 的地方。命中时打印其限定名、类型、
输入/输出的 nan/inf/max，并给出判据（输入干净 => NaN 由本模块产生；输入已坏 => 源头在上游）。

只对**输出 NaN** 触发（inf 仅在报告里显示辅助判断），从而不会被注意力 mask 里合法的 -inf 误触发。

开关（环境变量 RECOGDRIVE_NAN_DEBUG）：
  未设 / 0 / false / off  -> 不安装，零影响（默认）。
  1 / true / raise        -> 安装并在首个命中处抛 RuntimeError 停止（便于拿栈定位）。
  warn / report           -> 安装但只打印不中断（命中一次后不再重复刷屏）。

典型用法：
  RECOGDRIVE_NAN_DEBUG=1 bash scripts/training/run_recogdrive_qwen3_train_multi_node_2b.sh
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterator, Tuple

import torch

logger = logging.getLogger(__name__)

_STATE = {"installed": False, "hit": False}


def _iter_tensors(obj: Any, _depth: int = 0) -> Iterator[torch.Tensor]:
    """递归提取容器/ModelOutput 里的张量（限制深度，避免遍历超大 past_key_values 过深）。"""
    if _depth > 4:
        return
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item, _depth + 1)
    elif isinstance(obj, dict):  # transformers ModelOutput 继承自 OrderedDict
        for item in obj.values():
            yield from _iter_tensors(item, _depth + 1)


def _summarize(tensors: Iterator[torch.Tensor]) -> Tuple[int, int, float]:
    """返回 (nan 计数, inf 计数, max|.|)；只统计浮点张量。"""
    n_nan = n_inf = 0
    max_abs = 0.0
    for t in tensors:
        if not torch.is_tensor(t) or not t.is_floating_point() or t.numel() == 0:
            continue
        tf = t.detach().float()
        n_nan += int(torch.isnan(tf).sum())
        n_inf += int(torch.isinf(tf).sum())
        max_abs = max(max_abs, float(torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0).abs().max()))
    return n_nan, n_inf, max_abs


def _make_hook(name: str, module: torch.nn.Module, raise_on_hit: bool):
    module_type = module.__class__.__name__

    def hook(_mod: torch.nn.Module, inputs: Any, output: Any) -> None:
        if _STATE["hit"]:
            return
        # 只在输出含 NaN 时触发（inf 不触发，避开注意力 mask 合法 -inf）。
        out_has_nan = any(
            torch.is_tensor(t) and t.is_floating_point() and bool(torch.isnan(t.detach()).any())
            for t in _iter_tensors(output)
        )
        if not out_has_nan:
            return
        _STATE["hit"] = True
        in_nan, in_inf, in_max = _summarize(_iter_tensors(inputs))
        out_nan, out_inf, out_max = _summarize(_iter_tensors(output))
        clean_inputs = (in_nan == 0 and in_inf == 0)
        verdict = (
            "*** NaN 在本模块内部产生（输入干净）— 这就是源头 ***"
            if clean_inputs
            else "输入已含 nan/inf -> 源头在上游或更内层子模块（看更早的命中）"
        )
        msg = (
            "\n" + "=" * 72 + "\n"
            "[NAN-DEBUG] 首个输出含 NaN 的模块:\n"
            f"  module : {name}  ({module_type})\n"
            f"  output : nan={out_nan} inf={out_inf} max|.|={out_max:.4g}\n"
            f"  inputs : nan={in_nan} inf={in_inf} max|.|={in_max:.4g}\n"
            f"  verdict: {verdict}\n"
            + "=" * 72
        )
        logger.error(msg)
        print(msg, flush=True)
        if raise_on_hit:
            raise RuntimeError(
                f"[NAN-DEBUG] NaN detected in output of module '{name}' ({module_type}). "
                f"inputs_clean={clean_inputs}. Set RECOGDRIVE_NAN_DEBUG=0 to disable this hook."
            )

    return hook


def maybe_install_nan_debug(root_module: torch.nn.Module) -> None:
    """按 RECOGDRIVE_NAN_DEBUG 环境变量给 root_module 的所有子模块装 NaN forward hook。

    默认（未设/0/false/off）为无操作，对训练零影响。仅安装一次。
    """
    mode = os.environ.get("RECOGDRIVE_NAN_DEBUG", "").strip().lower()
    if mode in ("", "0", "false", "off", "no"):
        return
    if _STATE["installed"]:
        return
    if not isinstance(root_module, torch.nn.Module):
        logger.warning("[NAN-DEBUG] root is not an nn.Module (%s); skip.", type(root_module))
        return

    raise_on_hit = mode not in ("warn", "report", "warning")
    count = 0
    for name, submodule in root_module.named_modules():
        if name == "":  # 跳过根模块自身，聚焦有名字的子模块
            continue
        submodule.register_forward_hook(_make_hook(name, submodule, raise_on_hit))
        count += 1
    _STATE["installed"] = True
    banner = (
        f"[NAN-DEBUG] 已在 {count} 个子模块上安装 NaN forward hook "
        f"(mode={'raise' if raise_on_hit else 'warn'})。命中即定位第一个产生 NaN 的模块。"
        f" 关闭：RECOGDRIVE_NAN_DEBUG=0。"
    )
    logger.warning(banner)
    print(banner, flush=True)
