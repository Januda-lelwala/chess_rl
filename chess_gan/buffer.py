"""Replay of (board, action, mask) pairs for the discriminator's real set."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chess_env.actions import NUM_ACTIONS
from chess_env.encode import N_PLANES
from chess_env.episode import Episode


class TransitionBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.boards = np.zeros((capacity, N_PLANES, 8, 8), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.masks = np.zeros((capacity, NUM_ACTIONS), dtype=np.int8)
        self.size = 0
        self.cursor = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        board: NDArray[np.float32],
        action: int,
        mask: NDArray[np.int8],
    ) -> None:
        i = self.cursor
        self.boards[i] = board
        self.actions[i] = action
        self.masks[i] = mask
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(self, episode: Episode, winners_only: bool = False) -> None:
        for step in episode.steps:
            if step.info.get("illegal"):
                continue
            if winners_only and episode.return_for_side(step.side) <= 0:
                continue
            self.add(step.observation["board"], int(step.action), step.observation["action_mask"])

    def sample(self, n: int, rng: np.random.Generator) -> tuple[NDArray, NDArray, NDArray]:
        if self.size == 0:
            raise RuntimeError("TransitionBuffer is empty.")
        idx = rng.integers(0, self.size, size=n)
        return self.boards[idx], self.actions[idx], self.masks[idx]
