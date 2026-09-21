from __future__ import annotations

import chess
import numpy as np
import torch

from chess_env.actions import encode_move
from chess_env.encode import encode_board
from chess_env.env import ChessEnv
from chess_gan.expert import greedy_move
from chess_gan.features import action_planes
from chess_gan.losses import critic_scores, discriminator_loss
from chess_gan.nets import Discriminator, Generator
from chess_gan.ppo import ppo_update
from chess_gan.rollout import Rollout


def _start_batch(n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    env = ChessEnv()
    obs, _ = env.reset()
    env.close()
    board = torch.from_numpy(obs["board"]).unsqueeze(0).repeat(n, 1, 1, 1)
    mask = torch.from_numpy(obs["action_mask"].astype(np.float32)).unsqueeze(0).repeat(n, 1)
    return board, mask


def test_policy_samples_only_legal_moves() -> None:
    gen = Generator(width=16, blocks=1)
    gen.eval()
    board, mask = _start_batch(32)
    legal = set(torch.nonzero(mask[0] > 0, as_tuple=False).view(-1).tolist())
    with torch.no_grad():
        actions, _, _, _ = gen.act(board, mask, deterministic=False)
    for action in actions.tolist():
        assert action in legal


def test_deterministic_act_is_argmax() -> None:
    gen = Generator(width=16, blocks=1)
    board, mask = _start_batch(4)
    with torch.no_grad():
        actions, _, _, _ = gen.act(board, mask, deterministic=True)
        logits, _ = gen(board, mask)
    assert torch.equal(actions, logits.argmax(dim=-1))


def test_action_planes_e2e4() -> None:
    board = chess.Board()
    action = encode_move(board, chess.Move.from_uci("e2e4"))
    planes = action_planes(torch.tensor([action]))
    assert planes.shape == (1, 2, 8, 8)
    # e2 = file 4, rank 1; e4 = file 4, rank 3 in player view (white).
    assert float(planes[0, 0, 1, 4]) == 1.0
    assert float(planes[0, 1, 3, 4]) == 1.0
    assert float(planes[0, 0].sum()) == 1.0
    assert float(planes[0, 1].sum()) == 1.0


def test_r3gan_loss_backward() -> None:
    disc = Discriminator(width=16, blocks=1)
    board, _ = _start_batch(4)
    real_actions = torch.tensor([encode_move(chess.Board(), chess.Move.from_uci("e2e4"))] * 4)
    fake_actions = torch.tensor([encode_move(chess.Board(), chess.Move.from_uci("e2e3"))] * 4)
    loss, metrics = discriminator_loss(disc, board, real_actions, board, fake_actions, gamma=1.0)
    loss.backward()
    grads = [p.grad.abs().sum() for p in disc.parameters() if p.grad is not None]
    assert loss.ndim == 0
    assert any(g > 0 for g in grads)
    assert "d_r1" in metrics and "d_r2" in metrics


def test_ppo_step_runs() -> None:
    gen = Generator(width=16, blocks=1)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-3, betas=(0.0, 0.99))
    board, mask = _start_batch(16)
    with torch.no_grad():
        actions, logps, values, _ = gen.act(board, mask)
    rollout = Rollout(
        boards=board.numpy(),
        masks=mask.numpy().astype(np.int8),
        actions=actions.numpy().astype(np.int32),
        logps=logps.numpy().astype(np.float32),
        values=values.numpy().astype(np.float32),
        returns=np.zeros(16, dtype=np.float32),
        n_games=1,
        wins=0,
        draws=1,
        losses=0,
        avg_plies=16.0,
    )
    metrics = ppo_update(gen, opt, rollout, torch.zeros(16), torch.device("cpu"), epochs=1, minibatch=8)
    assert "policy_loss" in metrics


def test_greedy_expert_is_legal() -> None:
    board = chess.Board()
    move = greedy_move(board)
    assert move in board.legal_moves


def test_train_from_kwargs_defaults_include_resume() -> None:
    from chess_gan.train import build_parser

    args = build_parser().parse_args([])
    assert args.resume is False
    assert args.checkpoint == "models/r3gan.pt"


def test_critic_scores_shape() -> None:
    disc = Discriminator(width=16, blocks=1)
    boards = torch.from_numpy(encode_board(chess.Board())).unsqueeze(0)
    actions = torch.tensor([encode_move(chess.Board(), chess.Move.from_uci("g1f3"))])
    scores = critic_scores(disc, boards, actions)
    assert scores.shape == (1,)
