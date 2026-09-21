"""Train the chess R3GAN policy on Modal GPUs.

One-time setup::

    pip install modal
    modal setup

From the repo root::

    modal run -m chess_gan.train_modal
    modal run -m chess_gan.train_modal --gpu A100 --iterations 500 --games 32

Checkpoints live on the ``chess-rl-models`` Volume at ``/vol/models/r3gan.pt``.
The local entrypoint also copies the final weights to ``models/r3gan.pt``.

Pull a checkpoint later without training::

    modal volume get chess-rl-models models/r3gan.pt models/r3gan.pt
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "chess-rl-gan"
VOLUME_NAME = "chess-rl-models"
VOLUME_MOUNT = "/vol"
CHECKPOINT_REL = "models/r3gan.pt"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.4",
        "numpy>=2.0",
        "chess>=1.11",
        "gymnasium>=1.0",
    )
    .add_local_python_source("chess_env", "chess_gan")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

MINUTES = 60
DEFAULT_TIMEOUT = 8 * 60 * MINUTES  # 8 hours; Modal max is 24h


@app.function(
    gpu="A10",
    timeout=DEFAULT_TIMEOUT,
    volumes={VOLUME_MOUNT: volume},
    retries=modal.Retries(max_retries=3, initial_delay=1.0),
)
def train_remote(
    iterations: int = 200,
    games: int = 16,
    expert_games: int = 4,
    seed_games: int = 16,
    eval_games: int = 16,
    eval_every: int = 10,
    max_plies: int = 160,
    width: int = 64,
    blocks: int = 4,
    batch_size: int = 256,
    d_steps: int = 2,
    ppo_epochs: int = 3,
    gan_coef: float = 0.5,
    gamma_r1: float = 1.0,
    ent_coef: float = 0.01,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    ema: float = 0.999,
    buffer: int = 50_000,
    seed: int = 0,
    resume: bool = True,
) -> str:
    """Run R3GAN training on the container GPU. Returns the volume-relative ckpt path."""
    import torch

    from chess_gan.train import train_from_kwargs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Modal container.")

    ckpt = f"{VOLUME_MOUNT}/{CHECKPOINT_REL}"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {ckpt}")

    def persist() -> None:
        volume.commit()

    train_from_kwargs(
        iterations=iterations,
        games=games,
        expert_games=expert_games,
        seed_games=seed_games,
        eval_games=eval_games,
        eval_every=eval_every,
        max_plies=max_plies,
        width=width,
        blocks=blocks,
        batch_size=batch_size,
        d_steps=d_steps,
        ppo_epochs=ppo_epochs,
        gan_coef=gan_coef,
        gamma_r1=gamma_r1,
        ent_coef=ent_coef,
        lr_g=lr_g,
        lr_d=lr_d,
        ema=ema,
        buffer=buffer,
        seed=seed,
        device="cuda",
        checkpoint=ckpt,
        resume=resume,
        on_checkpoint=persist,
    )
    persist()
    return CHECKPOINT_REL


def _download_checkpoint(remote_rel: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    volume.reload()
    size = 0
    with local_path.open("wb") as fh:
        for chunk in volume.read_file(remote_rel):
            fh.write(chunk)
            size += len(chunk)
    print(f"Downloaded {remote_rel} -> {local_path} ({size} bytes)")


@app.local_entrypoint()
def main(
    gpu: str = "A10",
    timeout_hours: float = 8.0,
    iterations: int = 200,
    games: int = 16,
    expert_games: int = 4,
    seed_games: int = 16,
    eval_games: int = 16,
    eval_every: int = 10,
    max_plies: int = 160,
    width: int = 64,
    blocks: int = 4,
    batch_size: int = 256,
    d_steps: int = 2,
    ppo_epochs: int = 3,
    gan_coef: float = 0.5,
    gamma_r1: float = 1.0,
    ent_coef: float = 0.01,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    ema: float = 0.999,
    buffer: int = 50_000,
    seed: int = 0,
    resume: bool = True,
    download: bool = True,
    local_checkpoint: str = "models/r3gan.pt",
    download_only: bool = False,
) -> None:
    """Launch remote training (or just download the latest Volume checkpoint)."""
    if download_only:
        _download_checkpoint(CHECKPOINT_REL, Path(local_checkpoint))
        return

    timeout = int(timeout_hours * 60 * MINUTES)
    print(f"Starting Modal training on {gpu} for up to {timeout_hours:g}h ({iterations} iters)")
    remote_rel = train_remote.with_options(
        gpu=gpu,
        timeout=timeout,
        volumes={VOLUME_MOUNT: volume},
    ).remote(
        iterations=iterations,
        games=games,
        expert_games=expert_games,
        seed_games=seed_games,
        eval_games=eval_games,
        eval_every=eval_every,
        max_plies=max_plies,
        width=width,
        blocks=blocks,
        batch_size=batch_size,
        d_steps=d_steps,
        ppo_epochs=ppo_epochs,
        gan_coef=gan_coef,
        gamma_r1=gamma_r1,
        ent_coef=ent_coef,
        lr_g=lr_g,
        lr_d=lr_d,
        ema=ema,
        buffer=buffer,
        seed=seed,
        resume=resume,
    )
    print(f"Remote checkpoint: volume {VOLUME_NAME}:{remote_rel}")
    if download:
        try:
            _download_checkpoint(remote_rel, Path(local_checkpoint))
        except Exception as exc:
            print(f"Local download failed ({exc}). Pull it with:")
            print(f"  modal volume get {VOLUME_NAME} {CHECKPOINT_REL} {local_checkpoint}")
