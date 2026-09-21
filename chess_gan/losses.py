"""R3GAN objective: relativistic pairing GAN + zero-centered R1 and R2.

Huang, Gokaslan, Kuleshov, Tompkin, NeurIPS 2024.
``f(t) = -softplus(-t)`` in the paper; with a critic that is *higher for real*
this is ``softplus(fake - real)`` for the discriminator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from chess_gan.features import pack_discriminator_input


def zero_centered_penalty(score: torch.Tensor, inputs: torch.Tensor, gamma: float) -> torch.Tensor:
    """(γ/2) E[ ||∇_x D(x)||^2 ] — R1 on real, R2 on fake."""
    grad = torch.autograd.grad(
        outputs=score.sum(),
        inputs=inputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (gamma / 2.0) * grad.flatten(1).square().sum(dim=1).mean()


def discriminator_loss(
    discriminator: nn.Module,
    real_boards: torch.Tensor,
    real_actions: torch.Tensor,
    fake_boards: torch.Tensor,
    fake_actions: torch.Tensor,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    real_x = pack_discriminator_input(real_boards, real_actions).detach().requires_grad_(True)
    fake_x = pack_discriminator_input(fake_boards, fake_actions).detach().requires_grad_(True)
    real_score = discriminator(real_x)
    fake_score = discriminator(fake_x)

    adversarial = F.softplus(fake_score - real_score).mean()
    r1 = zero_centered_penalty(real_score, real_x, gamma)
    r2 = zero_centered_penalty(fake_score, fake_x, gamma)
    loss = adversarial + r1 + r2
    metrics = {
        "d_loss": float(loss.detach()),
        "d_adv": float(adversarial.detach()),
        "d_r1": float(r1.detach()),
        "d_r2": float(r2.detach()),
        "d_real": float(real_score.detach().mean()),
        "d_fake": float(fake_score.detach().mean()),
    }
    return loss, metrics


@torch.no_grad()
def critic_scores(discriminator: nn.Module, boards: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    packed = pack_discriminator_input(boards, actions)
    return discriminator(packed)
