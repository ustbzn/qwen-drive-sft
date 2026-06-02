from __future__ import annotations

from typing import Any, List

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
