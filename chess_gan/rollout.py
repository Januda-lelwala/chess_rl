"""Self-play collection for the generator (policy) and PPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from chess_env.actions import NUM_ACTIONS
from chess_env.encode import N_PLANES
from chess_env.env import ChessEnv
from chess_env.episode import play_episode
from chess_gan.nets import Generator


@dataclass
class Rollout:
    boards: NDArray[np.float32]
    masks: NDArray[np.int8]
    actions: NDArray[np.int32]
    logps: NDArray[np.float32]
    values: NDArray[np.float32]
    returns: NDArray[np.float32]
    n_games: int
    wins: int
    draws: int
    losses: int
    avg_plies: float

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def _policy_fn(generator: Generator, device: torch.device, log: list):
    def _policy(obs, info) -> int:
        board = torch.from_numpy(np.ascontiguousarray(obs["board"])).unsqueeze(0).to(device)
        mask = torch.from_numpy(np.ascontiguousarray(obs["action_mask"]).astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            action, logp, value, _ = generator.act(board, mask, deterministic=False)
        log.append(
            {
                "board": obs["board"],
                "mask": obs["action_mask"],
                "action": int(action.item()),
                "logp": float(logp.item()),
                "value": float(value.item()),
            }
        )
        return int(action.item())

    return _policy


def collect_selfplay(
    env: ChessEnv,
    generator: Generator,
    n_games: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Rollout:
    generator.eval()
    records: list[dict] = []
    returns: list[float] = []
    wins = draws = losses = 0
    plies = 0
    for _ in range(n_games):
        game_log: list[dict] = []
        policy = _policy_fn(generator, device, game_log)
        seed = int(rng.integers(0, 2**31 - 1))
        episode = play_episode(policy, policy, seed=seed, env=env)
        plies += len(episode.steps)
        if episode.winner == "white":
            wins += 1
        elif episode.winner == "black":
            losses += 1
        else:
            draws += 1
        for step, rec in zip(episode.steps, game_log):
            if step.info.get("illegal"):
                continue
            records.append(rec)
            returns.append(episode.return_for_side(step.side))

    n = len(records)
    boards = np.zeros((n, N_PLANES, 8, 8), dtype=np.float32)
    masks = np.zeros((n, NUM_ACTIONS), dtype=np.int8)
    actions = np.zeros(n, dtype=np.int32)
    logps = np.zeros(n, dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    for i, rec in enumerate(records):
        boards[i] = rec["board"]
        masks[i] = rec["mask"]
        actions[i] = rec["action"]
        logps[i] = rec["logp"]
        values[i] = rec["value"]
    return Rollout(
        boards=boards,
        masks=masks,
        actions=actions,
        logps=logps,
        values=values,
        returns=np.asarray(returns, dtype=np.float32),
        n_games=n_games,
        wins=wins,
        draws=draws,
        losses=losses,
        avg_plies=plies / max(n_games, 1),
    )
