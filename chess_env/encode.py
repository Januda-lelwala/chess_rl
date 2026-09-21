"""Board -> tensor encoding, always from the side-to-move's point of view."""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

# 0-5   our P,N,B,R,Q,K
# 6-11  their P,N,B,R,Q,K
# 12    our kingside castling (filled plane)
# 13    our queenside castling
# 14    their kingside castling
# 15    their queenside castling
# 16    en passant target square
# 17    we are actually Black (filled plane)
# 18    fifty-move clock / 100, clipped to 1
# 19    position has repeated at least once (filled plane)
N_PLANES = 20
PIECE_ORDER: tuple[int, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def encode_board(board: chess.Board) -> NDArray[np.float32]:
    """Return a ``(20, 8, 8)`` float32 tensor in player-relative coordinates.

    Axis 1 is chess rank (0 = player's back rank after mirroring), axis 2 is
    file (0 = a-file). If it is Black to move the board is vertically
    mirrored and colors are swapped, so the current player is always "white"
    sitting on rank 1.
    """
    view = board if board.turn == chess.WHITE else board.mirror()
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)

    for i, piece_type in enumerate(PIECE_ORDER):
        for square in view.pieces(piece_type, chess.WHITE):
            planes[i, chess.square_rank(square), chess.square_file(square)] = 1.0
        for square in view.pieces(piece_type, chess.BLACK):
            planes[6 + i, chess.square_rank(square), chess.square_file(square)] = 1.0

    if view.has_kingside_castling_rights(chess.WHITE):
        planes[12] = 1.0
    if view.has_queenside_castling_rights(chess.WHITE):
        planes[13] = 1.0
    if view.has_kingside_castling_rights(chess.BLACK):
        planes[14] = 1.0
    if view.has_queenside_castling_rights(chess.BLACK):
        planes[15] = 1.0

    if view.ep_square is not None:
        planes[
            16,
            chess.square_rank(view.ep_square),
            chess.square_file(view.ep_square),
        ] = 1.0

    if board.turn == chess.BLACK:
        planes[17] = 1.0

    planes[18] = min(board.halfmove_clock / 100.0, 1.0)

    if board.is_repetition(2):
        planes[19] = 1.0

    return planes
