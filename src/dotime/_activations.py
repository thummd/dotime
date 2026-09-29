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

__all__ = ["Tanh", "TanhReLU", "TanhSquare", "TanhX2"]


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


class TanhSquare(nn.Module):
    """``tanh(x)^2``, a bounded stand-in for the prior's ``x^2`` activation.

    It matches ``x^2`` near the origin and stays in ``[0, 1]``. The hardening
    option ``bounded_square`` swaps it in, because ``x^2`` is the one activation
    of the prior that is neither bounded nor 1-Lipschitz, so no weight rescaling
    can keep a lagged loop through it from blowing up.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) ** 2
