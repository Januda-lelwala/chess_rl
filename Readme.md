# chess_rl

RL chess bot. First piece: a legal chess **game environment** that a policy (later a GAN generator, or any RL agent) can play in.

The rules engine is [`python-chess`](https://python-chess.readthedocs.io/). This repo wraps it as a Gymnasium environment with a fixed action space, a neural-net board tensor, and a legal-move mask.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Play a game

```bash
# You (White) vs a random legal-move bot
python -m chess_env.play

# Two random bots (prints PGN at the end)
python -m chess_env.play --white random --black random --seed 0
```

Enter moves in SAN (`e4`, `Nf3`, `O-O`) or UCI (`e2e4`).

## Environment

```python
from chess_env import ChessEnv

env = ChessEnv()
obs, info = env.reset()          # or reset(options={"fen": "..."})
action = env.sample_legal_action()
obs, reward, terminated, truncated, info = env.step(action)
```

`obs` is a dict:

| Key | Shape | Meaning |
| --- | --- | --- |
| `board` | `(20, 8, 8)` float32 | Player-relative planes (see below) |
| `action_mask` | `(4672,)` int8 | `1` = legal action this turn |

The agent **always plays the side to move**. When it is Black's turn the board is vertically mirrored so the current player sits on rank 1. One network can therefore play both colors.

### Action space

`Discrete(4672)` = AlphaZero's `8×8×73` map:

- 56 queen-like slides (8 directions × 7 distances)
- 8 knight jumps
- 9 underpromotions (left / straight / right × N, B, R)

Queen promotions are ordinary one-step slides to the last rank. Castling is the king moving two files. Always mask with `obs["action_mask"]` — an unmasked illegal action ends the episode with reward `-1`.

### Observation planes

From the current player's view:

| Plane | Content |
| --- | --- |
| 0–5 | Our P, N, B, R, Q, K |
| 6–11 | Their P, N, B, R, Q, K |
| 12–13 | Our O-O / O-O-O (filled) |
| 14–15 | Their O-O / O-O-O (filled) |
| 16 | En passant target square |
| 17 | We are actually Black (filled) |
| 18 | Fifty-move clock / 100 |
| 19 | Position has repeated (filled) |

Axis 1 is rank (`0` = our back rank), axis 2 is file (`0` = a-file).

### Rewards

`step` is sparse: `0` until the game ends, then `+1` if the mover just won, `0` on a draw. That only labels the last ply. For training, use `play_episode` so every move of a side gets the game outcome:

```python
from chess_env import ChessEnv, play_episode
from chess_env.episode import random_policy

env = ChessEnv()
bot = random_policy(env)
episode = play_episode(bot, bot, seed=0, env=env)
boards, returns = episode.policy_targets()  # +1 / 0 / -1 per ply
```

Draws (stalemate, fifty-move, threefold, insufficient material) are claimed automatically.

## Tests

```bash
pytest
```

## Legal moves and the GAN

Available moves are **not** extra board planes. They are a constraint on the action, not a picture of the position.

- The board tensor stays `(20, 8, 8)` — pieces, castling, ep, colour, clocks.
- `obs["action_mask"]` is applied as **−∞ on illegal logits** in the policy head (AlphaZero / Leela style). The generator physically cannot sample an illegal move.
- The discriminator sees the **chosen** move as two 8×8 planes (from-square, to-square) concatenated with the board. Illegal `(s, a)` pairs never enter training.

Stuffing the 4672-d mask into the state would waste capacity and force the net to *learn* legality instead of being hard-constrained.

## GAN (R3GAN)

The trainer is **R3GAN** (Huang et al., NeurIPS 2024) adapted to discrete chess:

| Piece | Role |
| --- | --- |
| Generator | Residual CNN policy `π(a\|s)` + value head. Illegal actions masked. |
| Discriminator | Residual critic `D(s, a)`. Higher score = more like strong play. |
| Loss | Relativistic pairing GAN + zero-centered **R1** and **R2** gradient penalties. No BatchNorm, Adam `β₁=0`. |
| Generator step | Actions are discrete, so G is updated with **PPO**, reward = game outcome + `gan_coef * tanh(D(s,a))`. |
| Real data | ε-greedy material+PST expert, plus winning self-play plies (self-imitation). |

```bash
# Train locally (writes models/r3gan.pt)
python -m chess_gan.train --iterations 50 --games 4 --device auto

# Evaluate vs random and the greedy expert
python -m chess_gan.eval --checkpoint models/r3gan.pt --games 20

# Play against the trained bot
python -m chess_env.play --white human --black gan --checkpoint models/r3gan.pt
```

### Train on Modal

Runs the same trainer on a cloud GPU. Checkpoints are stored on a Modal Volume and copied back to `models/r3gan.pt`.

```bash
pip install "modal>=1.0"
modal setup                         # once: browser login

# from the repo root
modal run -m chess_gan.train_modal
modal run -m chess_gan.train_modal --gpu A100 --iterations 500 --games 32

# resume an interrupted run (default) or pull weights only
modal run -m chess_gan.train_modal --resume
modal run -m chess_gan.train_modal --download-only
```

GPU options: `T4`, `L4`, `A10` (default), `A100`, `A100-80GB`, `L40S`, `H100`. Timeout default is 8 hours (`--timeout-hours 24` for the platform max). If a container dies, the next run loads `/vol/models/r3gan.pt` from the `chess-rl-models` Volume.

```bash
modal volume get chess-rl-models models/r3gan.pt models/r3gan.pt
```

This is adversarial *imitation of play*, not an image GAN. The relativistic critic compares real `(s, a)` to generated `(s, a)` pairwise, which is what stops mode collapse in modern GANs. Game outcome stays in the reward so the policy cannot win the GAN game by playing nonsense that merely fools D.
