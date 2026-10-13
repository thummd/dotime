"""Joint-token encoder branch (recovery program, wave W3 variant a).

The per-variable encoder tokenises each variable's own history and mean-pools
it, so no representation of how the variables move together ever exists
before the mixer. This branch tokenises each pre-onset time step as the whole
cross-section ``x[t], x[t-1], ..., x[t-L]`` over all variables, runs a small
transformer over those steps, reads out the last steps before the onset and
returns one vector per trajectory. :class:`DoOverTimePFN` appends it to the
per-variable tokens, where the mixer's attention can read it. Unlike the
statistic tokens of ``stat_tokens.py`` it hands the model no estimator, only
the chance to learn one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class JointTokenEncoder(nn.Module):
    """Transformer over cross-sectional time-step tokens of the pre-onset history.

    Args:
        n_max: Padded number of variables (token width per lag).
        embed_size: Model width.
        n_layers: Transformer layers.
        n_heads: Attention heads.
        token_lags: Lagged cross-sections packed into every step token.
        context_window: Pre-onset steps kept, counted back from the onset.
        readout_last_k: Steps before the onset averaged into the output.
        pos_init_std: Scale of the learned relative position embedding at init.
    """

    def __init__(
        self,
        n_max: int,
        embed_size: int,
        n_layers: int = 2,
        n_heads: int = 4,
        token_lags: int = 4,
        context_window: int = 200,
        readout_last_k: int = 4,
        pos_init_std: float = 1.0,
    ) -> None:
        super().__init__()
        if token_lags < 0:
            raise ValueError("token_lags must be >= 0")
        self.n_max = n_max
        self.token_lags = int(token_lags)
        self.context_window = int(context_window)
        self.readout_last_k = int(readout_last_k)
        # Token: the cross-section at t and its lags, plus the variable mask so
        # padded columns are distinguishable from real zeros.
        self.embed = nn.Linear((self.token_lags + 1) * n_max + n_max, embed_size)
        self.pos = nn.Parameter(torch.randn(self.context_window, embed_size) * pos_init_std)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=n_heads,
            dim_feedforward=4 * embed_size,
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        # norm_first disables the nested-tensor fast path anyway; saying so
        # avoids the warning, which this package's tests treat as an error.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(embed_size), nn.Linear(embed_size, embed_size))

    def forward(
        self,
        X_obs_norm: torch.Tensor,
        variable_mask: torch.Tensor,
        int_onset_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        """One vector per trajectory from its pre-onset cross-sections.

        Args:
            X_obs_norm: ``(B, T, N_max)`` normalised history.
            variable_mask: ``(B, N_max)``.
            int_onset_idx: ``(B,)`` onset rows; ``None`` means the whole series is history.

        Returns:
            ``(B, E)``.
        """
        B, T, N = X_obs_norm.shape
        L, cw = self.token_lags, self.context_window
        device = X_obs_norm.device
        onset = (
            int_onset_idx
            if int_onset_idx is not None
            else torch.full((B,), T, device=device, dtype=torch.long)
        )
        tokens = X_obs_norm.new_zeros(B, cw, (L + 1) * N + N)
        valid = torch.zeros(B, cw, dtype=torch.bool, device=device)
        for b in range(B):
            o = int(onset[b])
            start = max(0, o - cw)
            n_rows = o - start
            if n_rows <= 0:
                continue
            t = torch.arange(start, o, device=device)
            lagged = [X_obs_norm[b, t]]
            for lag in range(1, L + 1):
                lagged.append(
                    X_obs_norm[b, (t - lag).clamp(min=0)]
                    * (t - lag >= 0).unsqueeze(1).to(X_obs_norm.dtype)
                )
            cross = torch.cat([*lagged, variable_mask[b].unsqueeze(0).expand(n_rows, -1)], dim=1)
            # Right-align so the last token is always the step before the onset.
            tokens[b, cw - n_rows :] = cross
            valid[b, cw - n_rows :] = True
        h = self.embed(tokens) + self.pos.unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=~valid)
        # Readout: the last k valid steps, which sit at the right end by construction.
        k = self.readout_last_k
        last = h[:, cw - k :, :]
        last_valid = valid[:, cw - k :].unsqueeze(2).to(h.dtype)
        pooled = (last * last_valid).sum(dim=1) / last_valid.sum(dim=1).clamp(min=1.0)
        return self.out(pooled)
