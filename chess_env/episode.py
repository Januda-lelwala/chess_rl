"""Self-play helper that attaches the game result to every ply.

The raw env only rewards the terminal mover. GAN / RL training should use
``Episode.return_for_side`` so both colors get the game outcome.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import chess
import numpy as np
from numpy.typing import NDArray

from chess_env.env import ChessEnv, Observation

Policy = Callable[[Observation, dict[str, Any]], int]


@dataclass
class Step:
    observation: Observation
    action: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    fen_before: str
    side: str


@dataclass
class Episode:
    steps: list[Step] = field(default_factory=list)
    winner: str | None = None
    result: str = "*"
    termination: str | None = None

    def pgn_moves(self) -> str:
        board = chess.Board()
        sans: list[str] = []
        for i, step in enumerate(self.steps):
            if step.info.get("illegal"):
                sans.append(f"{{illegal {step.info.get('move')}}}")
                break
            move = chess.Move.from_uci(step.info["move"])
            san = board.san(move)
            if i % 2 == 0:
                sans.append(f"{board.fullmove_number}. {san}")
            else:
                sans.append(san)
            board.push(move)
        return " ".join(sans)

    def return_for_side(self, side: str) -> float:
        """+1 if ``side`` won, -1 if it lost, 0 on draw / truncation."""
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == side else -1.0

    def policy_targets(self) -> tuple[list[NDArray[np.float32]], list[float]]:
        """Boards and the game return from the mover's side — GAN/RL labels."""
        boards: list[NDArray[np.float32]] = []
        returns: list[float] = []
        for step in self.steps:
            if step.info.get("illegal"):
                continue
            boards.append(step.observation["board"])
            returns.append(self.return_for_side(step.side))
        return boards, returns


def play_episode(
    white: Policy,
    black: Policy,
    *,
    seed: int | None = None,
    fen: str | None = None,
    max_plies: int = 512,
    env: ChessEnv | None = None,
) -> Episode:
    """Play one game. ``white`` / ``black`` map ``(obs, info) -> action``."""
    close_env = False
    if env is None:
        env = ChessEnv(max_plies=max_plies)
        close_env = True
    options = {"fen": fen} if fen else None
    obs, info = env.reset(seed=seed, options=options)
    episode = Episode()
    try:
        while True:
            side = info["turn"]
            fen_before = info["fen"]
            policy = white if side == "white" else black
            action = policy(obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            episode.steps.append(
                Step(
                    observation=obs,
                    action=action,
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=next_info,
                    fen_before=fen_before,
                    side=side,
                )
            )
            obs, info = next_obs, next_info
            if terminated or truncated:
                episode.winner = next_info["winner"]
                episode.result = next_info["result"] or "*"
                episode.termination = next_info["termination"]
                break
        return episode
    finally:
        if close_env:
            env.close()


def random_policy(env: ChessEnv) -> Policy:
    def _policy(obs: Observation, info: dict[str, Any]) -> int:
        legal = np.flatnonzero(obs["action_mask"])
        return int(env.np_random.choice(legal))

    return _policy
