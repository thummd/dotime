"""Bar distribution output head for Do-Over-Time-PFN.

Outputs logits over n_buckets using pfns.model.bar_distribution,
following Do-PFN's classification-as-regression approach.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union

from pfns.model.bar_distribution import FullSupportBarDistribution, get_bucket_borders

from causaltimeprior.models.pinball_loss import extract_quantiles, pinball_loss


class BarDistributionHead(nn.Module):
    """Predicts a bar distribution over the target value."""

    def __init__(self, embed_size: int = 512, n_buckets: int = 1000):
        super().__init__()
        self.n_buckets = n_buckets

        self.projection = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, n_buckets),
        )

        # Will be set after calibration
        self.bar_dist: Optional[FullSupportBarDistribution] = None
        # borders has n_buckets+1 entries
        self.register_buffer('borders', torch.zeros(n_buckets + 1))

    def set_bar_distribution(self, bar_dist: FullSupportBarDistribution, borders: torch.Tensor):
        """Set calibrated bar distribution and bucket borders."""
        self.bar_dist = bar_dist
        self.borders = borders

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Produce logits over buckets.

        Parameters
        ----------
        h : (B, E) embedding from cross-variable mixer

        Returns
        -------
        logits : (B, n_buckets) raw logits
        """
        return self.projection(h)

    def loss(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute bar distribution loss (mean over batch).

        Parameters
        ----------
        logits : (B, n_buckets)
        y_true : (B,) normalized target values

        Returns
        -------
        loss : scalar
        """
        if self.bar_dist is None:
            raise RuntimeError("Bar distribution not calibrated. Call set_bar_distribution() first.")
        # FullSupportBarDistribution returns per-sample losses
        per_sample_loss = self.bar_dist(logits, y_true.unsqueeze(-1))  # (B,)
        return per_sample_loss.mean()

    def predict_mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute mean prediction from logits.

        Parameters
        ----------
        logits : (B, n_buckets)

        Returns
        -------
        mean : (B,) predicted mean values
        """
        probs = torch.softmax(logits, dim=-1)  # (B, n_buckets)

        # Compute bucket centers from borders (n_buckets+1 entries)
        centers = (self.borders[:-1] + self.borders[1:]) / 2  # (n_buckets,)
        return (probs * centers.unsqueeze(0)).sum(dim=-1)

    def predict_quantiles(
        self,
        logits: torch.Tensor,
        tau_levels: Union[torch.Tensor, List[float]],
        temperature: float = 200.0,
    ) -> torch.Tensor:
        """Extract quantile predictions from logits.

        Parameters
        ----------
        logits : (B, n_buckets)
        tau_levels : quantile levels, e.g. [0.1, 0.5, 0.9]
        temperature : sharpness of soft bucket selection

        Returns
        -------
        quantiles : (B, len(tau_levels))
        """
        if not isinstance(tau_levels, torch.Tensor):
            tau_levels = torch.tensor(tau_levels, device=logits.device, dtype=logits.dtype)
        return extract_quantiles(logits, self.borders, tau_levels, temperature)

    def compute_pinball_loss(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        tau_levels: Union[torch.Tensor, List[float]],
        temperature: float = 200.0,
    ) -> torch.Tensor:
        """Compute pinball loss from logits and targets.

        Parameters
        ----------
        logits : (B, n_buckets)
        y_true : (B,) normalized target values
        tau_levels : quantile levels
        temperature : sharpness of soft bucket selection

        Returns
        -------
        loss : scalar
        """
        if not isinstance(tau_levels, torch.Tensor):
            tau_levels = torch.tensor(tau_levels, device=logits.device, dtype=logits.dtype)
        quantile_preds = extract_quantiles(logits, self.borders, tau_levels, temperature)
        return pinball_loss(quantile_preds, y_true, tau_levels)


def calibrate_bar_distribution(
    dataloader,
    n_buckets: int = 1000,
) -> tuple:
    """Calibrate bucket borders from training data.

    Collects normalized Y values, computes their range, and creates
    uniform bucket borders covering that range.

    Parameters
    ----------
    dataloader : iterable yielding batches with 'Y_true_norm'
    n_buckets : number of buckets

    Returns
    -------
    bar_dist : FullSupportBarDistribution
    borders : (n_buckets + 1,) bucket borders
    """
    all_ys = []
    for batch in dataloader:
        all_ys.append(batch['Y_true_norm'])

    ys = torch.cat(all_ys)
    # Filter out NaN/Inf
    ys = ys[torch.isfinite(ys)]

    # Use full_range to avoid the duplicate-borders bug in get_bucket_borders
    y_min = ys.min().item()
    y_max = ys.max().item()
    # Widen range slightly to ensure all values fall within
    margin = (y_max - y_min) * 0.01 + 1e-6
    full_range = (y_min - margin, y_max + margin)

    borders = get_bucket_borders(n_buckets, full_range=full_range)
    bar_dist = FullSupportBarDistribution(borders)
    return bar_dist, borders
