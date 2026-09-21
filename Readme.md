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

## Next

A GAN for chess is not an image GAN. The usual fit is **generative adversarial imitation**: the generator is a policy over this action space, the discriminator scores “does this look like strong play?”, and they train on games produced here. This environment is the substrate for that.
