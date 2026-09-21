"""Evaluate a checkpoint against random and the greedy expert."""

from __future__ import annotations

import argparse

import numpy as np

from chess_env.env import ChessEnv
from chess_env.episode import play_episode, random_policy
from chess_gan.agent import GanAgent
from chess_gan.expert import expert_policy


def _match(env, agent, opponent, n_games: int, rng: np.random.Generator) -> dict[str, float]:
    wins = draws = losses = 0
    for i in range(n_games):
        seed = int(rng.integers(0, 2**31 - 1))
        if i % 2 == 0:
            episode = play_episode(agent, opponent, seed=seed, env=env)
            winner_is_agent = episode.winner == "white"
            loser_is_agent = episode.winner == "black"
        else:
            episode = play_episode(opponent, agent, seed=seed, env=env)
            winner_is_agent = episode.winner == "black"
            loser_is_agent = episode.winner == "white"
        if episode.winner is None:
            draws += 1
        elif winner_is_agent:
            wins += 1
        elif loser_is_agent:
            losses += 1
    games = max(n_games, 1)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / games,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a chess GAN checkpoint.")
    parser.add_argument("--checkpoint", default="models/r3gan.pt")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=256)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    env = ChessEnv(max_plies=args.max_plies)
    agent = GanAgent(args.checkpoint, device=args.device, deterministic=True)
    random = random_policy(env)
    expert = expert_policy(env, rng, epsilon=0.0)

    vs_random = _match(env, agent, random, args.games, rng)
    vs_expert = _match(env, agent, expert, args.games, rng)
    env.close()
    print(f"vs random: {vs_random}")
    print(f"vs greedy expert: {vs_expert}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
