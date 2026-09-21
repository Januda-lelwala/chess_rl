"""Train the chess policy GAN (R3GAN critic + masked PPO generator)."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from chess_env.env import ChessEnv
from chess_env.episode import play_episode, random_policy
from chess_gan.agent import auto_device
from chess_gan.buffer import TransitionBuffer
from chess_gan.expert import expert_policy
from chess_gan.losses import critic_scores, discriminator_loss
from chess_gan.nets import Discriminator, Generator
from chess_gan.ppo import ppo_update
from chess_gan.rollout import collect_selfplay


def _ema_update(ema: Generator, model: Generator, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1.0 - decay)
        for ema_b, b in zip(ema.buffers(), model.buffers()):
            ema_b.copy_(b)


def _to_torch_batch(boards, actions, device):
    return (
        torch.from_numpy(np.ascontiguousarray(boards)).to(device),
        torch.from_numpy(np.ascontiguousarray(actions).astype(np.int64)).to(device),
    )


def evaluate_vs_random(env: ChessEnv, agent, n_games: int, rng: np.random.Generator) -> dict[str, float]:
    random = random_policy(env)
    wins = draws = losses = 0
    for i in range(n_games):
        seed = int(rng.integers(0, 2**31 - 1))
        if i % 2 == 0:
            episode = play_episode(agent, random, seed=seed, env=env)
            if episode.winner == "white":
                wins += 1
            elif episode.winner == "black":
                losses += 1
            else:
                draws += 1
        else:
            episode = play_episode(random, agent, seed=seed, env=env)
            if episode.winner == "black":
                wins += 1
            elif episode.winner == "white":
                losses += 1
            else:
                draws += 1
    games = max(n_games, 1)
    return {
        "eval_win": wins / games,
        "eval_draw": draws / games,
        "eval_loss": losses / games,
        "eval_score": (wins + 0.5 * draws) / games,
    }


def seed_expert_buffer(env: ChessEnv, buffer: TransitionBuffer, n_games: int, rng: np.random.Generator) -> None:
    policy = expert_policy(env, rng, epsilon=0.15)
    for _ in range(n_games):
        seed = int(rng.integers(0, 2**31 - 1))
        episode = play_episode(policy, policy, seed=seed, env=env)
        buffer.add_episode(episode)


def train(args: argparse.Namespace) -> None:
    device = auto_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    env = ChessEnv(max_plies=args.max_plies)
    generator = Generator(width=args.width, blocks=args.blocks).to(device)
    discriminator = Discriminator(width=args.width, blocks=args.blocks).to(device)
    ema = deepcopy(generator).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)

    opt_g = Adam(generator.parameters(), lr=args.lr_g, betas=(0.0, 0.99))
    opt_d = Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.99))

    buffer = TransitionBuffer(capacity=args.buffer)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_path.with_suffix(".jsonl")
    on_checkpoint = getattr(args, "on_checkpoint", None)

    config = {
        "width": args.width,
        "blocks": args.blocks,
        "gan_coef": args.gan_coef,
        "gamma_r1": args.gamma_r1,
    }

    start_iter = 1
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(payload["generator"])
        discriminator.load_state_dict(payload["discriminator"])
        ema.load_state_dict(payload["ema"])
        if "opt_g" in payload:
            opt_g.load_state_dict(payload["opt_g"])
        if "opt_d" in payload:
            opt_d.load_state_dict(payload["opt_d"])
        start_iter = int(payload.get("iter", 0)) + 1
        print(f"Next iteration: {start_iter}")

    print(f"Seeding expert buffer with {args.seed_games} greedy games on {device}...")
    seed_expert_buffer(env, buffer, args.seed_games, rng)
    print(f"Expert transitions: {len(buffer)}")

    if start_iter > args.iterations:
        print(f"Checkpoint already at iter {start_iter - 1}; nothing to do.")
        env.close()
        return

    for iteration in range(start_iter, args.iterations + 1):
        t0 = time.time()
        rollout = collect_selfplay(env, generator, args.games, device, rng)
        expert = expert_policy(env, rng, epsilon=0.15)
        for _ in range(args.expert_games):
            seed = int(rng.integers(0, 2**31 - 1))
            buffer.add_episode(play_episode(expert, expert, seed=seed, env=env))
        # Self-imitation: winning plies join the real set.
        for i in range(len(rollout)):
            if rollout.returns[i] > 0:
                buffer.add(rollout.boards[i], int(rollout.actions[i]), rollout.masks[i])

        fake_boards_np, fake_actions_np = rollout.boards, rollout.actions
        d_metrics: dict[str, float] = {}
        n_fake = len(rollout)
        if n_fake == 0:
            print(f"iter {iteration}: empty rollout, skipping")
            continue
        for _ in range(args.d_steps):
            real_b, real_a, _ = buffer.sample(min(args.batch_size, n_fake), rng)
            idx = rng.integers(0, n_fake, size=min(args.batch_size, n_fake))
            real_boards, real_actions = _to_torch_batch(real_b, real_a, device)
            fake_boards, fake_actions = _to_torch_batch(fake_boards_np[idx], fake_actions_np[idx], device)
            loss_d, d_metrics = discriminator_loss(
                discriminator,
                real_boards,
                real_actions,
                fake_boards,
                fake_actions,
                gamma=args.gamma_r1,
            )
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_d.step()

        boards_t, actions_t = _to_torch_batch(rollout.boards, rollout.actions, device)
        d_score = critic_scores(discriminator, boards_t, actions_t)
        returns = torch.from_numpy(rollout.returns).to(device) + args.gan_coef * torch.tanh(d_score)
        ppo_metrics = ppo_update(
            generator,
            opt_g,
            rollout,
            returns,
            device,
            epochs=args.ppo_epochs,
            minibatch=args.batch_size,
            ent_coef=args.ent_coef,
        )
        _ema_update(ema, generator, args.ema)

        metrics = {
            "iter": iteration,
            "time_s": round(time.time() - t0, 2),
            "buffer": len(buffer),
            "plies": len(rollout),
            "avg_plies": round(rollout.avg_plies, 1),
            "selfplay_wdl": f"{rollout.wins}/{rollout.draws}/{rollout.losses}",
            **{k: round(v, 4) for k, v in d_metrics.items()},
            **{k: round(v, 4) for k, v in ppo_metrics.items()},
        }

        if iteration % args.eval_every == 0 or iteration == 1:
            def ema_policy(obs, info, _agent_gen=ema):
                board = torch.from_numpy(np.ascontiguousarray(obs["board"])).unsqueeze(0).to(device)
                mask = torch.from_numpy(np.ascontiguousarray(obs["action_mask"]).astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, *_ = _agent_gen.act(board, mask, deterministic=True)
                return int(action.item())

            eval_metrics = evaluate_vs_random(env, ema_policy, args.eval_games, rng)
            metrics.update({k: round(v, 4) for k, v in eval_metrics.items()})
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "ema": ema.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "config": config,
                    "iter": iteration,
                    "metrics": metrics,
                },
                ckpt_path,
            )
            if on_checkpoint is not None:
                on_checkpoint()

        print(json.dumps(metrics))
        with log_path.open("a") as fh:
            fh.write(json.dumps(metrics) + "\n")

    env.close()
    print(f"Wrote {ckpt_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train R3GAN chess policy.")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--games", type=int, default=4, help="Self-play games per iteration.")
    p.add_argument("--expert-games", type=int, default=2)
    p.add_argument("--seed-games", type=int, default=8)
    p.add_argument("--eval-games", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-plies", type=int, default=160)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--d-steps", type=int, default=2)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--gan-coef", type=float, default=0.5)
    p.add_argument("--gamma-r1", type=float, default=1.0)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", default="models/r3gan.pt")
    p.add_argument("--resume", action="store_true", help="Continue from --checkpoint if it exists.")
    return p


def train_from_kwargs(**kwargs) -> None:
    """Build an argparse namespace from defaults plus overrides, then train."""
    args = build_parser().parse_args([])
    for key, value in kwargs.items():
        setattr(args, key, value)
    train(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
