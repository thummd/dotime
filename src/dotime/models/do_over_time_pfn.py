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

from dotime.models.bar_head import BarDistributionHead
from dotime.models.cross_variable_mixer import CrossVariableMixer
from dotime.models.encoder import TemporalEncoder
from dotime.models.joint_encoder import JointTokenEncoder
from dotime.models.quantile_head import QuantileHead
from dotime.models.stat_tokens import InContextStatistics
from dotime.models.value_bypass import make_value_bypass


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
        encoder_config: dict | None = None,
        head_type: str = "bar",
        tau_levels: list[float] | None = None,
        n_mixer_layers: int = 1,
        context_window: int = 200,
        value_bypass: str = "none",
        readout: str = "mean",
        readout_last_k: int = 4,
        pos_init_std: float = 0.02,
        token_lags: int = 0,
        horizon_embed: str = "none",
        horizon_heads: int = 0,
        horizon_mixers: int = 0,
        query_a_ref: bool = False,
        stat_tokens: str = "none",
        stat_lags: int = 3,
        joint_branch: bool = False,
        joint_layers: int = 2,
        role_embed: bool = False,
    ):
        super().__init__()
        self.head_type = head_type
        self.query_a_ref = bool(query_a_ref)
        # do-over-time-pfn recovery program, wave W3: representations the
        # slope can be read from. "ols_a" adds pre-onset regression
        # coefficients on the treatment to every variable token (variant b),
        # "ols_path" adds the queried variable's regression on every variable
        # given the treatment and the role flags (variant c), joint_branch
        # appends one token over the pre-onset cross-sections (variant a).
        if stat_tokens not in ("none", "ols_a", "ols_path"):
            raise ValueError(f"Unknown stat_tokens: {stat_tokens!r} (none | ols_a | ols_path)")
        self.stat_tokens = stat_tokens
        self.joint_branch = bool(joint_branch)
        # The encoder treats every column with the same weights and positions,
        # so a variable token carries no column identity and the mixer's
        # one-hot indices cannot address it. Two learned vectors mark the
        # treated and the queried token at query time; zero at init.
        self.role_embed = nn.Parameter(torch.zeros(2, embed_size)) if role_embed else None
        # A bank of K mixers implies a bank of K heads: nothing trainable after
        # the encoder is shared across horizons, so one horizon's gradient
        # cannot move another horizon's path (a shared mixer drifts a working
        # one-step predictor back to the pre-onset mean under mixed horizons).
        self.horizon_mixers_n = int(horizon_mixers)
        if self.horizon_mixers_n > 0:
            horizon_heads = self.horizon_mixers_n
        self.horizon_heads = horizon_heads
        self.value_bypass_mode = value_bypass
        self.readout = readout
        self.horizon_embed = horizon_embed

        self.temporal_encoder = TemporalEncoder(
            n_max=n_max,
            embed_size=embed_size,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            backend=encoder_backend,
            encoder_config=encoder_config,
            context_window=context_window,
            readout=readout,
            readout_last_k=readout_last_k,
            pos_init_std=pos_init_std,
            token_lags=token_lags,
        )

        def _mixer() -> CrossVariableMixer:
            return CrossVariableMixer(
                n_max=n_max,
                embed_size=embed_size,
                n_heads=n_cross_attn_heads,
                n_mixer_layers=n_mixer_layers,
                horizon_embed=horizon_embed,
                query_a_ref=query_a_ref,
            )

        if self.horizon_mixers_n > 0:
            self.horizon_mixers = nn.ModuleList(_mixer() for _ in range(self.horizon_mixers_n))
            self.cross_variable_mixer = None
        else:
            self.cross_variable_mixer = _mixer()
            self.horizon_mixers = None

        if head_type == "quantile":
            self.quantile_head = QuantileHead(
                embed_size=embed_size,
                tau_levels=tau_levels,
                n_horizon_heads=horizon_heads,
            )
            self.bar_head = None
        else:
            self.bar_head = BarDistributionHead(
                embed_size=embed_size,
                n_buckets=n_buckets,
            )
            self.quantile_head = None

        self.value_bypass = make_value_bypass(value_bypass, embed_size)
        self.in_context_stats = (
            InContextStatistics(
                embed_size, n_lags=stat_lags, window=context_window, mode=stat_tokens
            )
            if stat_tokens != "none"
            else None
        )
        self.joint_encoder = (
            JointTokenEncoder(
                n_max=n_max,
                embed_size=embed_size,
                n_layers=joint_layers,
                n_heads=n_heads,
                token_lags=max(token_lags, 1),
                context_window=context_window,
                readout_last_k=readout_last_k,
                pos_init_std=pos_init_std,
            )
            if self.joint_branch
            else None
        )

    @property
    def head(self):
        """Return the active output head."""
        return self.quantile_head if self.head_type == "quantile" else self.bar_head

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 1: Encode trajectory (expensive, compute once per trajectory).

        Parameters
        ----------
        batch : dict with X_obs_norm: (B, T, N_max), variable_mask: (B, N_max)

        Returns
        -------
        h_vars : (B, N_max, E) per-variable temporal representations
        """
        h_vars = self.temporal_encoder(
            batch["X_obs_norm"],
            batch["variable_mask"],
            int_onset_idx=batch.get("int_onset_idx"),
        )
        if self.in_context_stats is not None:
            h_vars = self.in_context_stats(h_vars, batch)
        if self.joint_encoder is not None:
            # One extra token per trajectory; query() extends the key mask for it.
            h_joint = self.joint_encoder(
                batch["X_obs_norm"], batch["variable_mask"], batch.get("int_onset_idx")
            )
            h_vars = torch.cat([h_vars, h_joint.unsqueeze(1)], dim=1)
        return h_vars

    def _mark_roles(
        self, h_vars: torch.Tensor, intervention_target: torch.Tensor, query_target: torch.Tensor
    ) -> torch.Tensor:
        """Add the treated and queried role vectors to the two tokens they name.

        Args:
            h_vars: ``(B, N, E)`` tokens, one row per query.
            intervention_target: ``(B,)`` treated column per query (zero when withheld).
            query_target: ``(B,)`` queried column per query.

        Returns:
            ``(B, N, E)`` tokens with the role vectors added.
        """
        n = h_vars.shape[1]
        is_a = torch.nn.functional.one_hot(intervention_target.long(), n).to(h_vars.dtype)
        is_y = torch.nn.functional.one_hot(query_target.long(), n).to(h_vars.dtype)
        return (
            h_vars + is_a.unsqueeze(2) * self.role_embed[0] + is_y.unsqueeze(2) * self.role_embed[1]
        )

    def query(self, h_vars: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stage 2+3: Query with intervention/outcome spec (cheap, run many times).

        Parameters
        ----------
        h_vars : (B, N_max, E) from encode() — can be expanded via repeat_interleave
        batch : dict with intervention_*, query_*, variable_mask (all shape (B,))

        Returns
        -------
        output : (B, Q) quantile predictions or (B, n_buckets) logits
        """
        query_offset = self._query_offset(batch)
        variable_mask = batch["variable_mask"]
        if self.joint_branch:
            variable_mask = torch.cat(
                [variable_mask, variable_mask.new_ones(variable_mask.shape[0], 1)], dim=1
            )
        if self.role_embed is not None:
            h_vars = self._mark_roles(h_vars, batch["intervention_target"], batch["query_target"])
        mixer_kwargs = dict(
            intervention_target=batch["intervention_target"],
            intervention_type=batch["intervention_type"],
            intervention_value=batch["intervention_value"],
            intervention_time_start=batch["intervention_time_start"],
            intervention_time_end=batch["intervention_time_end"],
            query_target=batch["query_target"],
            query_time=batch["query_time"],
            variable_mask=variable_mask,
            query_offset=query_offset,
            a_ref=batch.get("a_ref_norm") if self.query_a_ref else None,
        )
        if self.horizon_mixers_n == 0:
            h_causal = self.cross_variable_mixer(h_vars=h_vars, **mixer_kwargs)
        else:
            if query_offset is None:
                raise ValueError("horizon_mixers > 0 needs a query offset in the batch")
            idx = query_offset.to(h_vars.device).long().clamp(0, self.horizon_mixers_n - 1)
            h_causal = h_vars.new_zeros(h_vars.shape[0], h_vars.shape[-1])
            # One forward per horizon present in the batch (at most K), each on
            # its own subset: cheaper than K full passes through the mixer.
            for k in idx.unique().tolist():
                sel = (idx == k).nonzero(as_tuple=True)[0]
                sub = {
                    name: (t[sel] if isinstance(t, torch.Tensor) else t)
                    for name, t in mixer_kwargs.items()
                }
                h_causal[sel] = self.horizon_mixers[k](h_vars=h_vars[sel], **sub)
        if self.value_bypass is not None:
            h_causal = self.value_bypass(h_causal, batch["intervention_value"])
        if self.head_type == "quantile" and self.horizon_heads > 0:
            return self.head(h_causal, horizon=query_offset)
        return self.head(h_causal)

    @staticmethod
    def _query_offset(batch):
        """Horizon of each query in steps after the end of the visible history.

        Uses ``batch['query_offset']`` when the dataloader attached it; otherwise
        derives it from ``query_time`` (fraction of T), ``int_onset_idx`` and the
        trajectory length, mapping queries to trajectories via ``_traj_idx``.
        Returns None when the batch carries no way to derive it.
        """
        if "query_offset" in batch:
            return batch["query_offset"]
        if "query_time" not in batch or "int_onset_idx" not in batch:
            return None
        x = batch.get("X_obs_norm", batch.get("X_obs"))
        if x is None:
            return None
        T = x.shape[1]
        onset = batch["int_onset_idx"]
        if "_traj_idx" in batch:
            onset = onset[batch["_traj_idx"]]
        qti = torch.round(batch["query_time"].float() * T).long()
        return (qti - onset.to(qti.device)).clamp(min=0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch."""
        output = self.forward(batch)
        return self.head.loss(output, batch["Y_true_norm"])

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict mean values for a batch."""
        output = self.forward(batch)
        return self.head.predict_mean(output)
