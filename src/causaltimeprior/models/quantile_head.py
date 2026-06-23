"""Pure quantile prediction head for Do-Over-Time-PFN.

Replaces bar distribution with direct quantile predictions trained via
pinball (quantile) loss. No bucket calibration needed.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union


class QuantileHead(nn.Module):
    """Predicts quantile values directly via pinball loss."""

    def __init__(
        self,
        embed_size: int = 512,
        tau_levels: Optional[List[float]] = None,
    ):
        super().__init__()
        if tau_levels is None:
            tau_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        self.n_quantiles = len(tau_levels)
        self.register_buffer(
            'tau_levels', torch.tensor(tau_levels, dtype=torch.float32)
        )

        self.projection = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, self.n_quantiles),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Produce quantile predictions.

        Parameters
        ----------
        h : (B, E) embedding from cross-variable mixer

        Returns
        -------
        quantiles : (B, Q) predicted quantile values
        """
        return self.projection(h)

    def loss(self, preds: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute pinball (quantile) loss.

        Parameters
        ----------
        preds : (B, Q) predicted quantile values
        y_true : (B,) normalized target values

        Returns
        -------
        loss : scalar
        """
        error = y_true.unsqueeze(-1) - preds  # (B, Q)
        tau = self.tau_levels.unsqueeze(0)     # (1, Q)
        loss = torch.where(error >= 0, tau * error, (tau - 1.0) * error)
        return loss.mean()

    def predict_median(self, preds: torch.Tensor) -> torch.Tensor:
        """Return the median (tau=0.5) prediction.

        Parameters
        ----------
        preds : (B, Q)

        Returns
        -------
        median : (B,)
        """
        median_idx = (self.tau_levels - 0.5).abs().argmin()
        return preds[:, median_idx]

    def predict_mean(self, preds: torch.Tensor) -> torch.Tensor:
        """Approximate mean as average of quantile predictions.

        Parameters
        ----------
        preds : (B, Q)

        Returns
        -------
        mean : (B,)
        """
        return preds.mean(dim=-1)
