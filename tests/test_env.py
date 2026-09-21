from __future__ import annotations

import chess
import numpy as np

from chess_env import ChessEnv, NUM_ACTIONS, encode_move, play_episode
from chess_env.episode import random_policy


def test_reset_observation_shapes() -> None:
    env = ChessEnv()
    obs, info = env.reset(seed=0)
    assert obs["board"].shape == (20, 8, 8)
    assert obs["action_mask"].shape == (NUM_ACTIONS,)
    assert obs["action_mask"].sum() == 20  # starting position
    assert info["turn"] == "white"
    assert info["fen"] == chess.STARTING_FEN
    env.close()


def test_scholars_mate() -> None:
    env = ChessEnv()
    obs, info = env.reset()
    ucis = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
    reward = 0.0
    terminated = False
    for uci in ucis:
        action = encode_move(env.board, chess.Move.from_uci(uci))
        assert obs["action_mask"][action] == 1
        obs, reward, terminated, truncated, info = env.step(action)
        assert not truncated
    assert terminated
    assert reward == 1.0
    assert info["winner"] == "white"
    assert info["termination"] == "CHECKMATE"
    env.close()


def test_stalemate_is_draw() -> None:
    env = ChessEnv()
    env.reset(options={"fen": "k7/8/8/8/8/8/1Q6/7K w - - 0 1"})
    move = chess.Move.from_uci("b2b6")
    obs, reward, terminated, truncated, info = env.step(encode_move(env.board, move))
    assert terminated and not truncated
    assert reward == 0.0
    assert info["winner"] is None
    assert info["result"] == "1/2-1/2"
    assert info["termination"] == "STALEMATE"
    assert obs["action_mask"].sum() == 0
    env.close()


def test_illegal_action_loses() -> None:
    env = ChessEnv()
    obs, _ = env.reset()
    illegal = int(np.flatnonzero(obs["action_mask"] == 0)[0])
    obs, reward, terminated, truncated, info = env.step(illegal)
    assert terminated and not truncated
    assert reward == -1.0
    assert info["illegal"] is True
    env.close()


def test_custom_fen() -> None:
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    env = ChessEnv()
    _, info = env.reset(options={"fen": fen})
    assert env.board.fen() == fen
    assert info["fen"] == fen
    env.close()


def test_random_games_finish_legally() -> None:
    env = ChessEnv()
    policy = random_policy(env)
    for seed in range(8):
        episode = play_episode(policy, policy, seed=seed, env=env)
        assert not any(step.info["illegal"] for step in episode.steps)
        assert episode.steps[-1].terminated or episode.steps[-1].truncated
        assert episode.termination is not None
        boards, returns = episode.policy_targets()
        assert len(boards) == len(returns) == len(episode.steps)
        if episode.winner == "white":
            assert returns[0] == 1.0
            if len(returns) > 1:
                assert returns[1] == -1.0
        elif episode.winner == "black":
            assert returns[0] == -1.0
            assert returns[1] == 1.0
        else:
            assert all(r == 0.0 for r in returns)
    env.close()
