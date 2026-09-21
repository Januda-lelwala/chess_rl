"""AlphaZero-style discrete chess actions: 8 x 8 x 73 = 4672.

The 73 planes from a from-square are:

* 0-55: queen-like slides, 8 directions x 7 distances
* 56-63: knight jumps
* 64-72: underpromotions (3 directions x N/B/R)

Queen promotions are encoded as ordinary one-step queen-like moves to the
last rank. Actions are always encoded in the current player's view: if it
is Black to move, squares are vertically mirrored so the player sits on
rank 1, matching :func:`chess.Board.mirror`.
"""

from __future__ import annotations

import chess

QUEEN_PLANES = 56
KNIGHT_PLANES = 8
UNDERPROMO_PLANES = 9
PLANES = QUEEN_PLANES + KNIGHT_PLANES + UNDERPROMO_PLANES  # 73
NUM_ACTIONS = 64 * PLANES  # 4672

# N, NE, E, SE, S, SW, W, NW — file, rank
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)

KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

UNDERPROMO_PIECES: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

_KNIGHT_INDEX = {delta: i for i, delta in enumerate(KNIGHT_DELTAS)}
_QUEEN_DIR_INDEX = {delta: i for i, delta in enumerate(QUEEN_DIRECTIONS)}


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def mirror_move(move: chess.Move) -> chess.Move:
    """Vertically flip a move (a1 <-> a8), keeping the promotion piece."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def to_player_view(board: chess.Board, move: chess.Move) -> chess.Move:
    if board.turn == chess.WHITE:
        return move
    return mirror_move(move)


def from_player_view(board: chess.Board, move: chess.Move) -> chess.Move:
    if board.turn == chess.WHITE:
        return move
    return mirror_move(move)


def _plane_from_player_move(move: chess.Move) -> int:
    dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        if dy != 1 or dx not in (-1, 0, 1):
            raise ValueError(f"Unencodable underpromotion: {move.uci()}")
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        return QUEEN_PLANES + KNIGHT_PLANES + (dx + 1) * 3 + piece_idx

    knight_idx = _KNIGHT_INDEX.get((dx, dy))
    if knight_idx is not None:
        return QUEEN_PLANES + knight_idx

    if dx == 0 and dy == 0:
        raise ValueError(f"Unencodable null move: {move.uci()}")
    if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
        raise ValueError(f"Unencodable move: {move.uci()}")

    dist = max(abs(dx), abs(dy))
    dir_idx = _QUEEN_DIR_INDEX[(_sign(dx), _sign(dy))]
    return (dist - 1) * 8 + dir_idx


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Map a legal ``python-chess`` move to an action index in ``[0, 4672)``."""
    view_move = to_player_view(board, move)
    plane = _plane_from_player_move(view_move)
    return view_move.from_square * PLANES + plane


def action_squares(action: int) -> tuple[int, int | None]:
    """Player-view ``(from_square, to_square)`` for an action index.

    ``to_square`` is ``None`` when the plane slides off the board. The
    discriminator uses this geometry as two 8x8 planes; it does not need
    a legal-move mask in the board tensor.
    """
    if action < 0 or action >= NUM_ACTIONS:
        return 0, None
    from_square = action // PLANES
    plane = action % PLANES
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    if plane < QUEEN_PLANES:
        dist = plane // 8 + 1
        dx, dy = QUEEN_DIRECTIONS[plane % 8]
        to_file = from_file + dx * dist
        to_rank = from_rank + dy * dist
    elif plane < QUEEN_PLANES + KNIGHT_PLANES:
        dx, dy = KNIGHT_DELTAS[plane - QUEEN_PLANES]
        to_file = from_file + dx
        to_rank = from_rank + dy
    else:
        sub = plane - QUEEN_PLANES - KNIGHT_PLANES
        dx = sub // 3 - 1
        dy = 1
        to_file = from_file + dx
        to_rank = from_rank + dy

    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        return from_square, None
    return from_square, chess.square(to_file, to_rank)


def _underpromotion_piece(action: int) -> int | None:
    plane = action % PLANES
    if plane < QUEEN_PLANES + KNIGHT_PLANES:
        return None
    return UNDERPROMO_PIECES[(plane - QUEEN_PLANES - KNIGHT_PLANES) % 3]


def decode_action(board: chess.Board, action: int) -> chess.Move:
    """Map an action index back to a ``python-chess`` move on ``board``.

    Geometry that leaves the board becomes ``Move.null()``, which is never
    legal. Pawn moves onto the last rank that were encoded as queen-like
    slides are decoded as queen promotions.
    """
    from_square, to_square = action_squares(action)
    if to_square is None:
        return chess.Move.null()

    promotion = _underpromotion_piece(action)
    view_move = chess.Move(from_square, to_square, promotion=promotion)
    real_move = from_player_view(board, view_move)

    if real_move.promotion is None:
        piece = board.piece_at(real_move.from_square)
        dest_rank = chess.square_rank(real_move.to_square)
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and dest_rank in (0, 7)
        ):
            real_move = chess.Move(
                real_move.from_square,
                real_move.to_square,
                promotion=chess.QUEEN,
            )

    return real_move


def action_mask(board: chess.Board) -> list[int]:
    """Return a length-4672 0/1 mask of legal actions on ``board``."""
    mask = [0] * NUM_ACTIONS
    for move in board.legal_moves:
        mask[encode_move(board, move)] = 1
    return mask


def _square_lookup_tables() -> tuple[tuple[int, ...], tuple[int, ...]]:
    from_sq: list[int] = []
    to_sq: list[int] = []
    for action in range(NUM_ACTIONS):
        origin, dest = action_squares(action)
        from_sq.append(origin)
        to_sq.append(-1 if dest is None else dest)
    return tuple(from_sq), tuple(to_sq)


ACTION_FROM_SQUARE, ACTION_TO_SQUARE = _square_lookup_tables()
