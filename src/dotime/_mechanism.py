"""Base structural mechanism: a linear combination of parents + activation + noise.

Reimplemented from ``Do-PFN-prior`` (``dopfnprior.scm.simple_mechanism``) so the
package is self-contained. :class:`~dotime.temporal_mechanism.TemporalMechanism`
extends this to support lagged parents.

Attribution: Do-PFN (Oossen et al.). RNG-determining operations (the uniform
``[-1, 1]`` weight/bias initialisation) are kept faithful for reproducibility.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["SimpleMechanism"]


class SimpleMechanism(nn.Module):
    """A single linear layer over named parents followed by an activation.

    Parameters
    ----------
    node_names:
        Names of all candidate parent nodes (one learnable weight each).
    activation:
        Activation applied to the weighted sum before adding noise.
    device:
        Device for the parameters.
    generator:
        Optional RNG for reproducible weight/bias initialisation.
    """

    def __init__(
        self,
        node_names: list[str],
        activation: nn.Module,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.activation = activation
        self.device = device
        weights_map = {
            v: nn.Parameter(2 * torch.rand(1, device=device, generator=generator) - 1)
            for v in node_names
        }
        self.weights = nn.ParameterDict(weights_map)
        self.bias = nn.Parameter(2 * torch.rand(1, device=device, generator=generator) - 1)

    def __str__(self) -> str:
        weights = {v: value.item() for v, value in self.weights.items()}
        return f"Activation: {self.activation}\nBias: {self.bias.item()}\nWeights: {weights}"

    def forward(self, parent_values: dict[Any, Tensor], eps: Tensor) -> Tensor:
        if len(parent_values) == 0:
            return eps
        weighted_inputs = [
            parent_values[v] * weight for v, weight in self.weights.items() if v in parent_values
        ]
        combined = torch.sum(torch.stack(weighted_inputs), dim=0)
        return self.activation(combined + self.bias) + eps
