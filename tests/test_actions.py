from __future__ import annotations

import chess
import pytest

from chess_env.actions import (
    NUM_ACTIONS,
    action_mask,
    decode_action,
    encode_move,
)

FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4K3/8/8/8/8/8/p7/4k3 b - - 0 1",
    "8/4P3/8/8/8/8/8/k6K w - - 0 1",
    "k6K/8/8/8/8/8/4p3/8 b - - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
]


@pytest.mark.parametrize("fen", FENS)
def test_encode_decode_roundtrip(fen: str) -> None:
    board = chess.Board(fen)
    seen: set[int] = set()
    for move in board.legal_moves:
        action = encode_move(board, move)
        assert 0 <= action < NUM_ACTIONS
        assert action not in seen, f"collision on {move.uci()} in {fen}"
        seen.add(action)
        assert decode_action(board, action) == move
    assert sum(action_mask(board)) == len(seen) == board.legal_moves.count()


def test_castling_is_king_two_step() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle = chess.Move.from_uci("e1g1")
    action = encode_move(board, castle)
    assert decode_action(board, action) == castle
    assert castle in board.legal_moves


def test_en_passant_roundtrip() -> None:
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    move = chess.Move.from_uci("e5d6")
    assert move in board.legal_moves
    assert decode_action(board, encode_move(board, move)) == move


def test_underpromotion_roundtrip() -> None:
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    knight = chess.Move.from_uci("a7a8n")
    queen = chess.Move.from_uci("a7a8q")
    assert knight in board.legal_moves
    assert queen in board.legal_moves
    assert decode_action(board, encode_move(board, knight)) == knight
    assert decode_action(board, encode_move(board, queen)) == queen
    assert encode_move(board, knight) != encode_move(board, queen)


def test_black_and_white_castling_share_player_view_action() -> None:
    white = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    black = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    white_oo = encode_move(white, chess.Move.from_uci("e1g1"))
    black_oo = encode_move(black, chess.Move.from_uci("e8g8"))
    assert white_oo == black_oo
