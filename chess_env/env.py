"""Gymnasium chess environment built on ``python-chess``."""

from __future__ import annotations

from typing import Any

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from chess_env.actions import NUM_ACTIONS, action_mask, decode_action, encode_move
from chess_env.encode import N_PLANES, encode_board

Observation = dict[str, NDArray[Any]]


class ChessEnv(gym.Env):
    """Single-agent chess env. The agent always plays the side to move.

    Observations are player-relative, so one policy can play both colors.
    That is the interface a later GAN generator / RL policy should use.

    ``step`` reward is sparse and from the mover's point of view: ``+1`` if
    that move delivered a win, ``-1`` on a self-loss (illegal move), ``0``
    on a draw or on non-terminal plies. Assign game outcome to *every* move
    of a side in the trainer — see :func:`chess_env.episode.play_episode`.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: str | None = None,
        max_plies: int = 512,
        claim_draw: bool = True,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.max_plies = max_plies
        self.claim_draw = claim_draw
        self.board = chess.Board()
        self._last_move: chess.Move | None = None
        self._illegal = False

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(N_PLANES, 8, 8),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(NUM_ACTIONS,),
                    dtype=np.int8,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        fen = options.get("fen")
        self.board = chess.Board(fen) if fen else chess.Board()
        self._last_move = None
        self._illegal = False
        observation = self._observation()
        if self.render_mode == "human":
            self.render()
        return observation, self._info()

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self.board.is_game_over(claim_draw=self.claim_draw):
            raise RuntimeError("Cannot step a finished game; call reset().")

        move = decode_action(self.board, int(action))
        if move not in self.board.legal_moves:
            self._illegal = True
            self._last_move = move
            observation = self._observation()
            info = self._info()
            info["illegal"] = True
            if self.render_mode == "human":
                self.render()
            return observation, -1.0, True, False, info

        self.board.push(move)
        self._last_move = move
        self._illegal = False

        outcome = self.board.outcome(claim_draw=self.claim_draw)
        terminated = outcome is not None
        truncated = (not terminated) and self.board.ply() >= self.max_plies
        reward = 0.0
        if outcome is not None and outcome.winner is not None:
            mover_is_white = self.board.turn == chess.BLACK
            mover_won = outcome.winner == chess.WHITE if mover_is_white else outcome.winner == chess.BLACK
            reward = 1.0 if mover_won else -1.0

        observation = self._observation()
        info = self._info(outcome)
        if truncated:
            info["termination"] = "MAX_PLIES"
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def sample_legal_action(self) -> int:
        """Sample uniformly from the current legal-action mask."""
        mask = np.flatnonzero(action_mask(self.board))
        if mask.size == 0:
            raise RuntimeError("No legal actions.")
        return int(self.np_random.choice(mask))

    def legal_action_indices(self) -> list[int]:
        return [encode_move(self.board, move) for move in self.board.legal_moves]

    def render(self) -> str | None:
        text = self._ansi()
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def _ansi(self) -> str:
        board = self.board.unicode(borders=False, empty_square="·")
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        last = self._last_move.uci() if self._last_move else "-"
        return f"{board}\n{turn} to move. Last: {last}. FEN: {self.board.fen()}"

    def _observation(self) -> Observation:
        mask = np.asarray(action_mask(self.board), dtype=np.int8)
        return {"board": encode_board(self.board), "action_mask": mask}

    def _info(self, outcome: chess.Outcome | None = None) -> dict[str, Any]:
        if outcome is None and not self._illegal:
            outcome = self.board.outcome(claim_draw=self.claim_draw)
        winner: str | None = None
        result: str | None = None
        termination: str | None = None
        if self._illegal:
            result = "illegal"
            termination = "ILLEGAL_MOVE"
        elif outcome is not None:
            result = outcome.result()
            termination = outcome.termination.name
            if outcome.winner is True:
                winner = "white"
            elif outcome.winner is False:
                winner = "black"
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "ply": self.board.ply(),
            "move": None if self._last_move is None else self._last_move.uci(),
            "illegal": self._illegal,
            "winner": winner,
            "result": result,
            "termination": termination,
        }
