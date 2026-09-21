"""1-ply material + piece-square expert used as the initial 'real' distribution.

A GAN needs a real data source. Before self-play winners exist we bootstrap
from this greedy player (ε-greedy so the buffer is not a single line).
"""

from __future__ import annotations

from typing import Any

import chess
import numpy as np

from chess_env.actions import encode_move
from chess_env.env import ChessEnv, Observation

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# python-chess square order, White's view (rank 1 = indices 0-7).
_PAWN = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
)
_KNIGHT = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)
_KING = (
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
)
PST = {
    chess.PAWN: _PAWN,
    chess.KNIGHT: _KNIGHT,
    chess.BISHOP: _KNIGHT,
    chess.ROOK: (0,) * 64,
    chess.QUEEN: (0,) * 64,
    chess.KING: _KING,
}


def evaluate_white(board: chess.Board) -> int:
    if board.is_checkmate():
        return -10_000 if board.turn == chess.WHITE else 10_000
    if board.is_game_over(claim_draw=True):
        return 0
    score = 0
    for piece_type, value in PIECE_VALUE.items():
        table = PST[piece_type]
        for square in board.pieces(piece_type, chess.WHITE):
            score += value + table[square]
        for square in board.pieces(piece_type, chess.BLACK):
            score -= value + table[chess.square_mirror(square)]
    return score


def greedy_move(board: chess.Board) -> chess.Move:
    best_move: chess.Move | None = None
    best_score = -10**18
    for move in board.legal_moves:
        board.push(move)
        white_score = evaluate_white(board)
        board.pop()
        score = white_score if board.turn == chess.WHITE else -white_score
        if score > best_score:
            best_score = score
            best_move = move
    if best_move is None:
        raise RuntimeError("No legal moves.")
    return best_move


def expert_policy(env: ChessEnv, rng: np.random.Generator, epsilon: float = 0.15):
    def _policy(obs: Observation, info: dict[str, Any]) -> int:
        if rng.random() < epsilon:
            legal = np.flatnonzero(obs["action_mask"])
            return int(rng.choice(legal))
        move = greedy_move(env.board)
        return encode_move(env.board, move)

    return _policy
