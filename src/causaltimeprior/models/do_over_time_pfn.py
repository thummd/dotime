"""Do-Over-Time-PFN: Main model for temporal causal effect estimation.

Three-stage architecture:
1. Per-variable temporal encoding (trajectory-specific, query-agnostic)
2. Cross-variable causal reasoning with intervention/query context
3. Output head: quantile predictions or bar distribution

The encoder (Stage 1) is the expensive part (~90% of compute) and depends
only on X_obs — NOT on the intervention or query. Use encode() + query()
to compute the encoder ONCE per trajectory and reuse for many queries.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from causaltimeprior.models.encoder import TemporalEncoder
from causaltimeprior.models.cross_variable_mixer import CrossVariableMixer
from causaltimeprior.models.bar_head import BarDistributionHead
from causaltimeprior.models.quantile_head import QuantileHead
from causaltimeprior.models.value_bypass import make_value_bypass


class DoOverTimePFN(nn.Module):
    """In-context causal effect estimation for temporal data.

    Predicts P(X_j^{do}(t_query) | X_obs, intervention_spec) via either
    a bar distribution over buckets or direct quantile predictions.
    """

    def __init__(
        self,
        n_max: int = 41,
        embed_size: int = 512,
        n_heads: int = 4,
        n_encoder_layers: int = 10,
        n_cross_attn_heads: int = 4,
        n_buckets: int = 1000,
        encoder_backend: str = "transformer",
        encoder_config: dict = None,
        head_type: str = "bar",
        tau_levels: Optional[List[float]] = None,
        n_mixer_layers: int = 1,
        context_window: int = 200,
        value_bypass: str = "none",
    ):
        super().__init__()
        self.head_type = head_type
        self.value_bypass_mode = value_bypass

        self.temporal_encoder = TemporalEncoder(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            backend=encoder_backend,
            encoder_config=encoder_config,
            context_window=context_window,
        )

        self.cross_variable_mixer = CrossVariableMixer(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_cross_attn_heads,
            n_mixer_layers=n_mixer_layers,
        )

        if head_type == "quantile":
            self.quantile_head = QuantileHead(
                embed_size=embed_size,
                tau_levels=tau_levels,
            )
            self.bar_head = None
        else:
            self.bar_head = BarDistributionHead(
                embed_size=embed_size,
                n_buckets=n_buckets,
            )
            self.quantile_head = None

        self.value_bypass = make_value_bypass(value_bypass, embed_size)

    @property
    def head(self):
        """Return the active output head."""
        return self.quantile_head if self.head_type == "quantile" else self.bar_head

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 1: Encode trajectory (expensive, compute once per trajectory).

        Parameters
        ----------
        batch : dict with X_obs_norm: (B, T, N_max), variable_mask: (B, N_max)

        Returns
        -------
        h_vars : (B, N_max, E) per-variable temporal representations
        """
        return self.temporal_encoder(
            batch['X_obs_norm'],
            batch['variable_mask'],
            int_onset_idx=batch.get('int_onset_idx'),
        )

    def query(self, h_vars: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 2+3: Query with intervention/outcome spec (cheap, run many times).

        Parameters
        ----------
        h_vars : (B, N_max, E) from encode() — can be expanded via repeat_interleave
        batch : dict with intervention_*, query_*, variable_mask (all shape (B,))

        Returns
        -------
        output : (B, Q) quantile predictions or (B, n_buckets) logits
        """
        h_causal = self.cross_variable_mixer(
            h_vars=h_vars,
            intervention_target=batch['intervention_target'],
            intervention_type=batch['intervention_type'],
            intervention_value=batch['intervention_value'],
            intervention_time_start=batch['intervention_time_start'],
            intervention_time_end=batch['intervention_time_end'],
            query_target=batch['query_target'],
            query_time=batch['query_time'],
            variable_mask=batch['variable_mask'],
        )
        if self.value_bypass is not None:
            h_causal = self.value_bypass(h_causal, batch['intervention_value'])
        return self.head(h_causal)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Full forward pass (encode + query). Use encode()+query() for caching.

        Parameters
        ----------
        batch : dict with X_obs_norm, variable_mask, intervention_*, query_*

        Returns
        -------
        output : (B, Q) or (B, n_buckets)
        """
        h_vars = self.encode(batch)
        return self.query(h_vars, batch)

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        output = self.forward(batch)
        return self.head.loss(output, batch['Y_true_norm'])

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict mean values for a batch."""
        output = self.forward(batch)
        return self.head.predict_mean(output)
