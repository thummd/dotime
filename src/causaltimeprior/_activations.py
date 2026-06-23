"""Activation functions used by the SCM mechanisms.

Reimplemented from the ``Do-PFN-prior`` default config (``dopfnprior.configs``)
so the package carries no submodule dependency. These are the non-standard
compositions the mechanism prior draws from; plain activations (``tanh``,
``relu``, ``sin``, ``cos`` …) live directly in the mechanism modules.

Attribution: the ``tanh(x^2)`` / ``tanh(relu(x))`` mechanism family originates
with Do-PFN (Oossen et al.).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Tanh", "TanhX2", "TanhReLU"]


class Tanh(nn.Module):
    """``tanh(x)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class TanhX2(nn.Module):
    """``tanh(x^2)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.pow(x, 2))


class TanhReLU(nn.Module):
    """``tanh(relu(x))``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.relu(x))
