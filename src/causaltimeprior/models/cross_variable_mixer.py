"""Cross-variable causal reasoning module.

Stage 2 of the two-stage architecture: combines per-variable temporal
representations with intervention and query specifications via cross-attention,
enabling the model to learn which variables are causally relevant.
"""

import torch
import torch.nn as nn


class CrossVariableMixer(nn.Module):
    """Mixes per-variable representations with intervention/query context.

    Uses cross-attention where:
    - Query: intervention + query encoding (what causal question are we asking?)
    - Keys/Values: per-variable temporal representations (what does the data say?)
    """

    def __init__(
        self,
        n_max: int = 41,
        embed_size: int = 512,
        n_heads: int = 4,
        n_mixer_layers: int = 1,
    ):
        super().__init__()
        self.n_max = n_max
        self.embed_size = embed_size
        self.n_mixer_layers = n_mixer_layers

        # Intervention encoder: one-hot target (N_max) + type (3) + value (1) + t_start (1) + t_end (1)
        intervention_input_dim = n_max + 6
        self.intervention_encoder = nn.Sequential(
            nn.Linear(intervention_input_dim, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )

        # Query encoder: one-hot target (N_max) + query_time (1)
        query_input_dim = n_max + 1
        self.query_encoder = nn.Sequential(
            nn.Linear(query_input_dim, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )

        # Cross-attention layers (stacked for iterative causal reasoning)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_size, num_heads=n_heads, batch_first=True,
            )
            for _ in range(n_mixer_layers)
        ])
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(embed_size) for _ in range(n_mixer_layers)
        ])
        # Fix 2b: per-layer gated residual. Learns how much attention output
        # (which depends on intervention spec) replaces vs. adds to context.
        # Init bias=0 so sigmoid(0)=0.5 — equal mixing at init.
        self.gate_projs = nn.ModuleList([
            nn.Linear(embed_size, embed_size) for _ in range(n_mixer_layers)
        ])
        for g in self.gate_projs:
            nn.init.zeros_(g.bias)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
        )

    def forward(
        self,
        h_vars: torch.Tensor,
        intervention_target: torch.Tensor,
        intervention_type: torch.Tensor,
        intervention_value: torch.Tensor,
        intervention_time_start: torch.Tensor,
        intervention_time_end: torch.Tensor,
        query_target: torch.Tensor,
        query_time: torch.Tensor,
        variable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine variable representations with intervention/query context.

        Parameters
        ----------
        h_vars : (B, N_max, E) per-variable temporal representations
        intervention_target : (B,) int indices
        intervention_type : (B,) int 0/1/2
        intervention_value : (B,) float
        intervention_time_start : (B,) float in [0, 1]
        intervention_time_end : (B,) float in [0, 1]
        query_target : (B,) int indices
        query_time : (B,) float in [0, 1]
        variable_mask : (B, N_max) binary mask

        Returns
        -------
        h_causal : (B, E) causal reasoning embedding
        """
        B = h_vars.shape[0]
        device = h_vars.device

        # Encode intervention specification
        int_target_onehot = torch.zeros(B, self.n_max, device=device)
        int_target_onehot.scatter_(1, intervention_target.unsqueeze(1), 1.0)

        int_type_onehot = torch.zeros(B, 3, device=device)
        int_type_onehot.scatter_(1, intervention_type.unsqueeze(1), 1.0)

        int_features = torch.cat([
            int_target_onehot,
            int_type_onehot,
            intervention_value.unsqueeze(1),
            intervention_time_start.unsqueeze(1),
            intervention_time_end.unsqueeze(1),
        ], dim=1)  # (B, N_max + 6)

        h_int = self.intervention_encoder(int_features)  # (B, E)

        # Encode query specification
        query_target_onehot = torch.zeros(B, self.n_max, device=device)
        query_target_onehot.scatter_(1, query_target.unsqueeze(1), 1.0)

        query_features = torch.cat([
            query_target_onehot,
            query_time.unsqueeze(1),
        ], dim=1)  # (B, N_max + 1)

        h_query = self.query_encoder(query_features)  # (B, E)

        # Combine intervention + query as attention query
        context = (h_int + h_query).unsqueeze(1)  # (B, 1, E)

        # Cross-attention over variable representations (stacked layers)
        key_padding_mask = (variable_mask == 0)  # (B, N_max)

        for attn, norm, gate_proj in zip(
            self.cross_attn_layers, self.attn_norms, self.gate_projs
        ):
            h_out, _ = attn(
                query=context,
                key=h_vars,
                value=h_vars,
                key_padding_mask=key_padding_mask,
            )  # (B, 1, E)
            # Gated residual: learnable mix between attention output and prior context
            gate = torch.sigmoid(gate_proj(context))  # (B, 1, E)
            context = norm(gate * h_out + (1 - gate) * context)

        h_causal = context.squeeze(1)  # (B, E)
        h_causal = self.output_proj(h_causal)  # (B, E)

        return h_causal
