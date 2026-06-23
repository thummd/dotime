"""Per-variable normalization for heterogeneous-scale time series."""

import torch
from typing import Dict, Optional, Tuple


def per_variable_normalize(
    X_obs: torch.Tensor,
    variable_mask: torch.Tensor,
    eps: float = 1e-2,
    int_onset_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each variable independently using its observational statistics.

    Statistics are computed over pre-intervention timesteps. When
    ``int_onset_idx`` is provided, the pre-intervention window is taken
    explicitly as ``t < int_onset_idx[b]``. When it is None, all timesteps
    are treated as valid (purely observational data).

    Earlier versions inferred the pre-intervention window from
    ``(X_obs != 0)``, which silently dropped any *legitimate* zero value
    (e.g. relu activations producing exact zeros, or pre-dose PK samples).
    Empirically this dropped ~19% of pre-intervention positions under our
    hardened TSCM prior — biasing per-variable means upward and
    underestimating spreads. Always prefer the explicit time mask.

    Parameters
    ----------
    X_obs : (B, T, N_max) observational time series
    variable_mask : (B, N_max) binary mask for real variables
    eps : numerical stability
    int_onset_idx : (B,) per-sample first post-intervention index. If
        provided, only positions with t < int_onset_idx contribute to the
        per-variable statistics.

    Returns
    -------
    X_norm : (B, T, N_max) normalized series
    means : (B, N_max) per-variable means
    stds : (B, N_max) per-variable stds
    """
    B, T, _ = X_obs.shape
    if int_onset_idx is not None:
        t_idx = torch.arange(T, device=X_obs.device).view(1, T, 1)
        time_mask = (t_idx < int_onset_idx.view(B, 1, 1)).to(X_obs.dtype)  # (B, T, 1)
    else:
        time_mask = torch.ones(B, T, 1, device=X_obs.device, dtype=X_obs.dtype)

    # Broadcast time mask over variables and exclude padded variables from
    # the per-variable count (padded vars are all zero anyway).
    var_mask = variable_mask.unsqueeze(1)                              # (B, 1, N)
    use_mask = time_mask * var_mask                                    # (B, T, N)

    n_valid = use_mask.sum(dim=1).clamp(min=1)                         # (B, N)
    means = (X_obs * use_mask).sum(dim=1) / n_valid                    # (B, N)
    sq_diff = ((X_obs - means.unsqueeze(1)) * use_mask) ** 2
    # Bessel's correction: divide by (n-1) to match torch.std default
    stds = (sq_diff.sum(dim=1) / (n_valid - 1).clamp(min=1)).sqrt() + eps

    # Zero out stats for padded variables
    mask = variable_mask                 # (B, N_max)
    means = means * mask
    stds = stds * mask + (1 - mask)      # padded vars get std=1 to avoid div-by-zero

    # Normalize
    X_norm = (X_obs - means.unsqueeze(1)) / stds.unsqueeze(1)

    # Zero out padded variables
    X_norm = X_norm * mask.unsqueeze(1)

    return X_norm, means, stds


def normalize_target(
    Y_true: torch.Tensor,
    query_target: torch.Tensor,
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    """Normalize target values using the same per-variable statistics.

    Parameters
    ----------
    Y_true : (B,) target values
    query_target : (B,) variable indices
    means : (B, N_max)
    stds : (B, N_max)

    Returns
    -------
    Y_norm : (B,) normalized targets
    """
    batch_idx = torch.arange(Y_true.shape[0], device=Y_true.device)
    target_mean = means[batch_idx, query_target]
    target_std = stds[batch_idx, query_target]
    y_norm = (Y_true - target_mean) / target_std
    return torch.clamp(y_norm, -10.0, 10.0)


def normalize_batch(
    batch: Dict[str, torch.Tensor],
    target_key: str = "Y_true",
) -> Dict[str, torch.Tensor]:
    """Normalize a full batch in-place, adding normalized fields.

    Handles both flat batches (B trajectories, B queries) and multi-query
    batches where _traj_idx maps B_total queries to B unique trajectories.

    Adds: X_obs_norm, Y_true_norm, _norm_means, _norm_stds
    """
    X_norm, means, stds = per_variable_normalize(
        batch['X_obs'], batch['variable_mask'],
        int_onset_idx=batch.get('int_onset_idx'),
    )
    batch['X_obs_norm'] = X_norm
    batch['_norm_means'] = means
    batch['_norm_stds'] = stds

    # For multi-query batches, expand means/stds to query dimension via _traj_idx
    if '_traj_idx' in batch:
        traj_idx = batch['_traj_idx']
        q_means = means[traj_idx]   # (B_total, N_max)
        q_stds = stds[traj_idx]     # (B_total, N_max)
    else:
        q_means = means
        q_stds = stds

    query_target = batch['query_target']
    q_idx = torch.arange(query_target.shape[0], device=query_target.device)

    if target_key == "Y_causal_effect" and "Y_causal_effect" in batch:
        target_std = q_stds[q_idx, query_target]
        Y_norm = torch.clamp(batch['Y_causal_effect'] / target_std, -10.0, 10.0)
    else:
        target_mean = q_means[q_idx, query_target]
        target_std = q_stds[q_idx, query_target]
        Y_norm = torch.clamp((batch[target_key] - target_mean) / target_std, -10.0, 10.0)

    batch['Y_true_norm'] = Y_norm
    return batch
