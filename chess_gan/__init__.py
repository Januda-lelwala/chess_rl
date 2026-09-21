"""R3GAN chess policy: relativistic pairing GAN + legal-move masking."""

from chess_gan.agent import GanAgent
from chess_gan.nets import Discriminator, Generator

__all__ = ["Discriminator", "GanAgent", "Generator"]
