"""Differentiable quantile extraction and pinball loss for bar distributions.

Enables quantile robustness testing by training with pinball (quantile) loss
as an auxiliary objective alongside the bar distribution loss.
"""

import torch


def extract_quantiles(
    logits: torch.Tensor,
    borders: torch.Tensor,
    tau_levels: torch.Tensor,
    temperature: float = 200.0,
) -> torch.Tensor:
    """Differentiably extract quantile predictions from bar distribution logits.

    Uses soft bucket selection so gradients flow back through the logits.

    Parameters
    ----------
    logits : (B, K) raw logits over K buckets
    borders : (K+1,) bucket boundaries
    tau_levels : (Q,) quantile levels in (0, 1)
    temperature : sharpness of soft bucket selection

    Returns
    -------
    quantiles : (B, Q) predicted quantile values
    """
    probs = torch.softmax(logits, dim=-1)  # (B, K)
    cdf = torch.cumsum(probs, dim=-1)  # (B, K) CDF at right edge of each bucket

    # CDF at left edge of each bucket
    cdf_left = torch.cat(
        [torch.zeros(probs.shape[0], 1, device=probs.device, dtype=probs.dtype), cdf[:, :-1]],
        dim=-1,
    )  # (B, K)

    bucket_left = borders[:-1]  # (K,)
    bucket_width = borders[1:] - borders[:-1]  # (K,)

    results = []
    for tau in tau_levels:
        # For each bucket, compute the quantile value it would give
        # if the tau-th quantile falls in that bucket
        frac = ((tau - cdf_left) / (probs + 1e-12)).clamp(0.0, 1.0)  # (B, K)
        candidate = bucket_left + frac * bucket_width  # (B, K)

        # Soft-select the crossing bucket: peaked where cdf_left is closest to tau
        log_weights = -temperature * (cdf_left - tau).abs()  # (B, K)
        weights = torch.softmax(log_weights, dim=-1)  # (B, K)

        q = (weights * candidate).sum(dim=-1)  # (B,)
        results.append(q)

    return torch.stack(results, dim=-1)  # (B, Q)


def pinball_loss(
    quantile_preds: torch.Tensor,
    y_true: torch.Tensor,
    tau_levels: torch.Tensor,
) -> torch.Tensor:
    """Compute pinball (quantile) loss.

    L_tau(y, q) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)

    Parameters
    ----------
    quantile_preds : (B, Q) predicted quantile values
    y_true : (B,) true target values
    tau_levels : (Q,) quantile levels

    Returns
    -------
    loss : scalar, mean pinball loss over samples and quantiles
    """
    error = y_true.unsqueeze(-1) - quantile_preds  # (B, Q)
    tau = tau_levels.unsqueeze(0)  # (1, Q)
    loss = torch.where(error >= 0, tau * error, (tau - 1.0) * error)  # (B, Q)
    return loss.mean()
