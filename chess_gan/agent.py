"""Load a trained generator and use it as a play_episode policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_env.env import Observation
from chess_gan.nets import Generator


def auto_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GanAgent:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "auto",
        deterministic: bool = True,
    ) -> None:
        self.device = auto_device(device) if isinstance(device, str) else device
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = payload.get("config", {})
        self.generator = Generator(
            width=cfg.get("width", 64),
            blocks=cfg.get("blocks", 4),
        ).to(self.device)
        state = payload.get("ema", payload["generator"])
        self.generator.load_state_dict(state)
        self.generator.eval()
        self.deterministic = deterministic

    def __call__(self, obs: Observation, info: dict[str, Any]) -> int:
        board = torch.from_numpy(np.ascontiguousarray(obs["board"])).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(np.ascontiguousarray(obs["action_mask"]).astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, *_ = self.generator.act(board, mask, deterministic=self.deterministic)
        return int(action.item())
