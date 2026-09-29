"""Pure quantile prediction head for Do-Over-Time-PFN.

Replaces bar distribution with direct quantile predictions trained via
pinball (quantile) loss. No bucket calibration needed.
"""

import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    """Predicts quantile values directly via pinball loss."""

    def __init__(
        self,
        embed_size: int = 512,
        tau_levels: list[float] | None = None,
        n_horizon_heads: int = 0,
    ):
        """Quantile head with an optional bank of per-horizon projections.

        Args:
            embed_size: Width of the mixer output.
            tau_levels: Quantile levels to predict.
            n_horizon_heads: ``0`` keeps one shared projection (legacy state
                dict). ``K > 0`` allocates K projections and routes each query
                to the one indexed by its horizon (offset clamped to ``K-1``),
                so the gradients of one horizon cannot move the readout of
                another. A shared readout trained on mixed horizons collapses
                to the pre-onset mean even from a warm start.
        """
        super().__init__()
        self.n_horizon_heads = int(n_horizon_heads)
        if tau_levels is None:
            tau_levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        self.n_quantiles = len(tau_levels)
        self.register_buffer("tau_levels", torch.tensor(tau_levels, dtype=torch.float32))

        def _proj() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.GELU(),
                nn.Linear(embed_size, self.n_quantiles),
            )

        if self.n_horizon_heads > 0:
            self.horizon_projections = nn.ModuleList(_proj() for _ in range(self.n_horizon_heads))
            self.projection = None
        else:
            self.projection = _proj()
            self.horizon_projections = None

    def forward(self, h: torch.Tensor, horizon: torch.Tensor | None = None) -> torch.Tensor:
        """Produce quantile predictions.

        Args:
            h: ``(B, E)`` embedding from the cross-variable mixer.
            horizon: ``(B,)`` integer query horizons; required when the head
                was built with ``n_horizon_heads > 0`` and ignored otherwise.

        Returns:
            ``(B, Q)`` predicted quantile values.

        Raises:
            ValueError: If per-horizon heads are configured and ``horizon`` is
                missing.
        """
        if self.n_horizon_heads == 0:
            return self.projection(h)
        if horizon is None:
            raise ValueError("QuantileHead with n_horizon_heads > 0 needs the query horizon")
        idx = horizon.to(h.device).long().clamp(0, self.n_horizon_heads - 1)
        # All projections see the whole batch, then a gather picks one per
        # query: the head is tiny next to the encoder, so K-fold head compute
        # is cheaper than masked loops and keeps a single kernel path.
        stacked = torch.stack([proj(h) for proj in self.horizon_projections], dim=1)  # (B, K, Q)
        return stacked.gather(1, idx.view(-1, 1, 1).expand(-1, 1, self.n_quantiles)).squeeze(1)

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
        tau = self.tau_levels.unsqueeze(0)  # (1, Q)
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
