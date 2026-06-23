"""Per-variable temporal encoder for Do-Over-Time-PFN.

Stage 1 of the two-stage architecture: encodes each variable's time series
independently, producing per-variable temporal representations.

Supports two backends:
- "gdp": GatedDeltaProductEncoder from TempoPFN (requires GPU + fla)
- "transformer": Standard nn.TransformerEncoder (CPU-compatible fallback)

Both backends follow the same interface: (B, T, N_max) -> (B, N, E)
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Per-variable temporal encoder.

    Encodes each variable's time series independently via either
    GatedDeltaProduct (TempoPFN) or standard Transformer layers,
    then pools over time to produce per-variable representations.
    """

    def __init__(
        self,
        n_max: int = 41,
        embed_size: int = 512,
        n_heads: int = 4,
        n_layers: int = 10,
        backend: str = "transformer",
        encoder_config: dict | None = None,
        context_window: int = 200,
    ):
        super().__init__()
        self.n_max = n_max
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.backend = backend
        self.context_window = context_window

        # Per-scalar value embedding (from TempoPFN pattern)
        self.expand_values = nn.Linear(1, embed_size, bias=True)

        # Learnable absolute positional encoding (backward compat fallback).
        # Size = max T supported when int_onset_idx is NOT provided. The
        # relative encoding path (used when int_onset_idx is given — the
        # default for training and eval) is bounded by context_window and
        # does NOT depend on this cap. So T=5000 trajectories work fine
        # as long as int_onset_idx is in the batch.
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 1024, embed_size) * 0.02  # abs-path max T=1024
        )

        # Learnable relative positional encoding (distance to intervention)
        # Indexed by rel_pos + context_window to shift to non-negative.
        # Covers positions from -context_window to +context_window.
        self.rel_pos_encoding = nn.Parameter(
            torch.randn(1, 2 * context_window + 1, embed_size) * 0.02
        )

        if backend == "gdp":
            self._init_gdp_layers(embed_size, n_heads, n_layers, encoder_config or {})
        elif backend == "transformer":
            self._init_transformer_layers(embed_size, n_heads, n_layers)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _init_gdp_layers(self, embed_size, n_heads, n_layers, encoder_config):
        """Initialize GatedDeltaProduct encoder layers (TempoPFN).

        The ``gdp`` backend is optional and GPU-only: it needs the ``[gdp]``
        extra (``tempopfn`` + ``fla``). The default ``transformer`` backend has
        no such dependency.
        """
        try:
            from tempopfn.models.blocks import GatedDeltaProductEncoder
        except ImportError:
            try:
                # Fall back to the upstream TempoPFN source layout.
                from src.models.blocks import GatedDeltaProductEncoder
            except ImportError as exc:
                raise ImportError(
                    "the 'gdp' encoder backend requires the optional [gdp] extra "
                    "(TempoPFN + flash-linear-attention) and a CUDA GPU:\n"
                    "    pip install 'causaltimeprior[gdp]'\n"
                    "Use the default backend='transformer' to run on CPU without it."
                ) from exc

        config = {
            "expand_v": 1.0,
            "use_short_conv": True,
            "conv_size": 32,
            "use_gate": True,
            "use_forget_gate": True,
            "num_householder": 4,
        }
        config.update(encoder_config)

        self.encoder_layers = nn.ModuleList(
            [
                GatedDeltaProductEncoder(
                    layer_idx=i,
                    token_embed_dim=embed_size,
                    num_heads=n_heads,
                    **config,
                )
                for i in range(n_layers)
            ]
        )

        # Learnable initial hidden states (weaving, from TempoPFN)
        head_k_dim = embed_size // n_heads
        expand_v = config.get("expand_v", 1.0)
        head_v_dim = int(head_k_dim * expand_v)
        self.initial_hidden_state = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, n_heads, head_k_dim, head_v_dim) / head_k_dim,
                    requires_grad=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.weaving = config.get("weaving", True)

    def _init_transformer_layers(self, embed_size, n_heads, n_layers):
        """Initialize standard Transformer encoder layers."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=n_heads,
            dim_feedforward=embed_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        X_obs: torch.Tensor,
        variable_mask: torch.Tensor,
        int_onset_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode observational time series per-variable.

        Parameters
        ----------
        X_obs : (B, T, N_max) normalized observational series
        variable_mask : (B, N_max) binary mask for real variables
        int_onset_idx : (B,) integer index of intervention onset per sample.
            If provided, uses relative positional encoding (distance to
            intervention) and truncates to context_window. If None, falls
            back to absolute positional encoding.

        Returns
        -------
        h_vars : (B, N_max, E) per-variable temporal representations
        """
        _B, _T, _N = X_obs.shape

        if int_onset_idx is not None:
            return self._forward_relative(X_obs, variable_mask, int_onset_idx)
        else:
            return self._forward_absolute(X_obs, variable_mask)

    def _forward_absolute(self, X_obs, variable_mask):
        """Original forward with absolute positional encoding (backward compat)."""
        B, T, N = X_obs.shape

        # Guardrail: the absolute path is bounded by pos_encoding buffer size.
        # For T>1024 callers should provide int_onset_idx to use the relative
        # path (context_window truncation).
        if self.pos_encoding.shape[1] < T:
            raise ValueError(
                f"Absolute encoding path supports T<={self.pos_encoding.shape[1]}; "
                f"got T={T}. Provide int_onset_idx to use the relative path "
                f"(truncates to context_window={self.context_window} regardless of T)."
            )

        # Time mask: which timesteps have data (pre-intervention, non-zero)
        time_mask = X_obs.abs().sum(dim=-1) > 0  # (B, T)

        # Embed per-variable: (B, T, N) -> (B, T, N, 1) -> (B, T, N, E)
        x = self.expand_values(X_obs.unsqueeze(-1))  # (B, T, N, E)

        # Add positional encoding (shared across variables)
        x = x + self.pos_encoding[:, :T, :].unsqueeze(2)  # broadcast over N

        # Vectorize: (B, T, N, E) -> (B*N, T, E)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, T, E)
        x = x.view(B * N, T, self.embed_size)

        # Encode
        x = self._forward_gdp(x, B, N) if self.backend == "gdp" else self._forward_transformer(x)

        # Mask-aware pool: average only over non-zero (pre-intervention) timesteps
        tm = time_mask.unsqueeze(1).expand(B, N, T)  # (B, N, T)
        tm = tm.reshape(B * N, T).unsqueeze(-1)  # (B*N, T, 1)
        h = (x * tm).sum(dim=1) / tm.sum(dim=1).clamp(min=1)  # (B*N, E)

        # Reshape: (B*N, E) -> (B, N, E)
        h = h.view(B, N, self.embed_size)
        h = h * variable_mask.unsqueeze(-1)
        return h

    def _forward_relative(self, X_obs, variable_mask, int_onset_idx):
        """Forward with relative positional encoding and context window truncation."""
        B, _T, N = X_obs.shape
        cw = self.context_window

        # Truncate to context window: [int_onset - cw, int_onset) per sample
        # Build truncated tensor of shape (B, cw, N)
        X_trunc = torch.zeros(B, cw, N, device=X_obs.device, dtype=X_obs.dtype)
        rel_positions = torch.zeros(B, cw, device=X_obs.device, dtype=torch.long)
        time_mask = torch.zeros(B, cw, device=X_obs.device, dtype=torch.bool)

        for b in range(B):
            onset = int(int_onset_idx[b].item())
            start = max(0, onset - cw)
            length = onset - start  # actual number of pre-intervention steps
            if length > 0:
                X_trunc[b, cw - length : cw, :] = X_obs[b, start:onset, :]
                # Relative positions: distance to intervention (negative = before)
                # e.g., if onset=45, start=0, length=45: positions are -45, -44, ..., -1
                rel_pos = torch.arange(start - onset, 0, device=X_obs.device)  # (length,)
                rel_positions[b, cw - length : cw] = rel_pos + cw  # shift to non-negative index
                time_mask[b, cw - length : cw] = True

        # Embed values: (B, cw, N, 1) -> (B, cw, N, E)
        x = self.expand_values(X_trunc.unsqueeze(-1))

        # Add relative positional encoding (shared across variables)
        # rel_positions: (B, cw) -> index into rel_pos_encoding (1, 2*cw+1, E)
        rel_pos_embed = self.rel_pos_encoding[0, rel_positions, :]  # (B, cw, E)
        x = x + rel_pos_embed.unsqueeze(2)  # broadcast over N

        # Vectorize: (B, cw, N, E) -> (B*N, cw, E)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, cw, E)
        x = x.view(B * N, cw, self.embed_size)

        # Encode
        x = self._forward_gdp(x, B, N) if self.backend == "gdp" else self._forward_transformer(x)

        # Mask-aware pool
        tm = time_mask.unsqueeze(1).expand(B, N, cw)  # (B, N, cw)
        tm = tm.reshape(B * N, cw).unsqueeze(-1)  # (B*N, cw, 1)
        h = (x * tm).sum(dim=1) / tm.sum(dim=1).clamp(min=1)  # (B*N, E)

        h = h.view(B, N, self.embed_size)
        h = h * variable_mask.unsqueeze(-1)
        return h

    def _forward_gdp(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Forward through GatedDeltaProduct layers with state weaving."""
        if self.weaving:
            hidden_state = torch.zeros_like(self.initial_hidden_state[0].repeat(B * N, 1, 1, 1))
            for i, layer in enumerate(self.encoder_layers):
                x, hidden_state = layer(
                    x,
                    hidden_state + self.initial_hidden_state[i].repeat(B * N, 1, 1, 1),
                )
        else:
            for i, layer in enumerate(self.encoder_layers):
                init_state = self.initial_hidden_state[i].repeat(B * N, 1, 1, 1)
                x, _ = layer(x, init_state)
        return x

    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through standard Transformer encoder."""
        return self.transformer(x)
