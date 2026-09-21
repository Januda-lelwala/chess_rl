"""Play a game in the terminal: human and/or random legal moves."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import chess

from chess_env.env import ChessEnv, Observation
from chess_env.episode import play_episode, random_policy


def _parse_human_move(board: chess.Board, text: str) -> chess.Move:
    text = text.strip()
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse move {text!r} as SAN or UCI.") from exc
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {text}")
    return move


def human_policy(env: ChessEnv) -> Any:
    from chess_env.actions import encode_move

    def _policy(obs: Observation, info: dict[str, Any]) -> int:
        print()
        print(env.render())
        legal = [env.board.san(m) for m in env.board.legal_moves]
        print("Legal (SAN):", ", ".join(legal[:24]) + (" ..." if len(legal) > 24 else ""))
        while True:
            raw = input(f"{info['turn'].capitalize()} to move (SAN/UCI, or quit): ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                sys.exit(0)
            try:
                move = _parse_human_move(env.board, raw)
                return encode_move(env.board, move)
            except ValueError as exc:
                print(exc)

    return _policy


def _side_policy(kind: str, env: ChessEnv, args: argparse.Namespace):
    if kind == "human":
        return human_policy(env)
    if kind == "random":
        return random_policy(env)
    if kind == "expert":
        import numpy as np

        from chess_gan.expert import expert_policy

        rng = np.random.default_rng(args.seed if args.seed is not None else 0)
        return expert_policy(env, rng, epsilon=0.0)
    if kind == "gan":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when a side is 'gan'")
        from chess_gan.agent import GanAgent

        return GanAgent(args.checkpoint, device=args.device, deterministic=True)
    raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play chess in the RL environment.")
    parser.add_argument("--white", choices=("human", "random", "expert", "gan"), default="human")
    parser.add_argument("--black", choices=("human", "random", "expert", "gan"), default="random")
    parser.add_argument("--fen", default=None, help="Start from this FEN instead of the initial position.")
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default="models/r3gan.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    env = ChessEnv(max_plies=args.max_plies)
    white = _side_policy(args.white, env, args)
    black = _side_policy(args.black, env, args)

    episode = play_episode(
        white,
        black,
        seed=args.seed,
        fen=args.fen,
        max_plies=args.max_plies,
        env=env,
    )

    print()
    print(env.render())
    print("Result:", episode.result, episode.termination or "")
    print(episode.pgn_moves())
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
