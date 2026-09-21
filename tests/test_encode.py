from __future__ import annotations

import chess
import numpy as np

from chess_env.encode import N_PLANES, encode_board


def test_startpos_planes() -> None:
    planes = encode_board(chess.Board())
    assert planes.shape == (N_PLANES, 8, 8)
    assert planes.dtype == np.float32
    # Our pawns on rank 2, their pawns on rank 7.
    assert planes[0, 1, :].sum() == 8
    assert planes[6, 6, :].sum() == 8
    # Our king on e1, their king on e8.
    assert planes[5, 0, 4] == 1
    assert planes[11, 7, 4] == 1
    assert planes[12].all() and planes[13].all()  # our castling
    assert planes[14].all() and planes[15].all()  # their castling
    assert planes[16].sum() == 0  # no ep
    assert planes[17].sum() == 0  # we are white
    assert planes[18].sum() == 0  # halfmove 0


def test_black_to_move_is_mirrored() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    planes = encode_board(board)
    # Current player is Black: 8 pawns on our 2nd rank.
    assert board.turn == chess.BLACK
    assert planes[0, 1, :].sum() == 8
    # White's e-pawn appears as an opposing pawn on e5 (rank index 4, file 4).
    assert planes[6, 4, 4] == 1
    assert planes[17].all()  # colour plane: we are actually Black
