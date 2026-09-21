"""Residual policy / critic networks following R3GAN backbone rules.

No BatchNorm (incompatible with R1/R2 and with per-sample scoring). Leaky ReLU,
identity residual blocks, Fixup-style zero-init of the last conv in each block.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from chess_env.actions import NUM_ACTIONS
from chess_env.encode import N_PLANES

ILLEGAL_LOGIT = -1e9
LEAK = 0.2


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU(LEAK, inplace=True)
        nn.init.kaiming_normal_(self.conv1.weight, a=LEAK)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class BoardEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(LEAK, inplace=True)
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        nn.init.kaiming_normal_(self.stem.weight, a=LEAK)
        nn.init.zeros_(self.stem.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.act(self.stem(x)))


class Generator(nn.Module):
    """Policy π(a|s) + value V(s). Illegal logits are hard-masked to -1e9."""

    def __init__(
        self,
        in_channels: int = N_PLANES,
        width: int = 64,
        blocks: int = 4,
        n_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.encoder = BoardEncoder(in_channels, width, blocks)
        self.activation = nn.LeakyReLU(LEAK, inplace=True)
        self.policy_conv = nn.Conv2d(width, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 8 * 8, n_actions)
        self.value_conv = nn.Conv2d(width, 8, kernel_size=1)
        self.value_fc = nn.Linear(8 * 8 * 8, 1)

    def forward(self, board: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(board)
        logits = self.policy_fc(self.activation(self.policy_conv(hidden)).flatten(1))
        logits = logits.masked_fill(mask <= 0, ILLEGAL_LOGIT)
        value = torch.tanh(self.value_fc(self.activation(self.value_conv(hidden)).flatten(1)))
        return logits, value.squeeze(-1)

    def act(
        self,
        board: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(board, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()


class Discriminator(nn.Module):
    """Relativistic critic D(s, a) → ℝ. Higher = more expert-like."""

    def __init__(self, in_channels: int = N_PLANES + 2, width: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.encoder = BoardEncoder(in_channels, width, blocks)
        self.head = nn.Linear(width, 1)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(packed)
        pooled = hidden.mean(dim=(2, 3))
        return self.head(pooled).squeeze(-1)
