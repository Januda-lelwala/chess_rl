"""Action as two 8x8 planes (from-square, to-square) in player view.

Legal moves are *not* extra board channels. The policy is hard-masked; the
discriminator only sees the move that was actually played.
"""

from __future__ import annotations

import torch

from chess_env.actions import ACTION_FROM_SQUARE, ACTION_TO_SQUARE, NUM_ACTIONS


def action_planes(actions: torch.Tensor) -> torch.Tensor:
    """Encode action indices as ``(N, 2, 8, 8)`` from/to planes."""
    actions = actions.long().view(-1)
    if actions.numel() == 0:
        return torch.zeros(0, 2, 8, 8, device=actions.device, dtype=torch.float32)
    if torch.any((actions < 0) | (actions >= NUM_ACTIONS)):
        raise ValueError("action index out of range")

    from_table = torch.as_tensor(ACTION_FROM_SQUARE, device=actions.device, dtype=torch.long)
    to_table = torch.as_tensor(ACTION_TO_SQUARE, device=actions.device, dtype=torch.long)
    from_sq = from_table[actions]
    to_sq = to_table[actions]

    n = actions.shape[0]
    planes = torch.zeros(n, 2, 8, 8, device=actions.device, dtype=torch.float32)
    idx = torch.arange(n, device=actions.device)
    planes[idx, 0, torch.div(from_sq, 8, rounding_mode="floor"), from_sq % 8] = 1.0
    valid = to_sq >= 0
    if bool(valid.any()):
        dest = to_sq[valid]
        planes[idx[valid], 1, torch.div(dest, 8, rounding_mode="floor"), dest % 8] = 1.0
    return planes


def pack_discriminator_input(boards: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Concatenate board planes with the chosen-move planes: ``(N, 22, 8, 8)``."""
    return torch.cat([boards, action_planes(actions)], dim=1)
