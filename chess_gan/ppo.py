"""PPO update of the generator. Discrete actions are not differentiable, so
the R3GAN generator step is a policy gradient with the critic as extra reward.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from chess_gan.nets import Generator
from chess_gan.rollout import Rollout


def ppo_update(
    generator: Generator,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    returns: torch.Tensor,
    device: torch.device,
    *,
    clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 3,
    minibatch: int = 256,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    generator.train()
    boards = torch.from_numpy(rollout.boards).to(device)
    masks = torch.from_numpy(rollout.masks.astype("float32")).to(device)
    actions = torch.from_numpy(rollout.actions.astype("int64")).to(device)
    old_logp = torch.from_numpy(rollout.logps).to(device)
    returns = returns.to(device)

    advantages = returns - torch.from_numpy(rollout.values).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    n = boards.shape[0]
    if n == 0:
        return {"ppo_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    last: dict[str, float] = {}
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, minibatch):
            idx = perm[start : start + minibatch]
            logits, value = generator(boards[idx], masks[idx])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[idx])
            ratio = (logp - old_logp[idx]).exp()
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (value - returns[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
            optimizer.step()
            last = {
                "ppo_loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
            }
    return last
