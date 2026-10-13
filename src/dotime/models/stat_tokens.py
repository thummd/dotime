"""In-context statistic tokens (recovery program, wave W3 variants b and c).

The effect sign in the training prior is ``sign(slope) * sign(v - a_ref)``,
and the slope has a random sign per sampled system, so it has to be read off
the pre-onset history. The per-variable encoder mean-pools each variable on
its own and never forms a cross-variable statistic (diagnostic D3 of October
2026: the trained model's response does not depend on the true slope's sign).
This module hands the model the statistic directly. For every variable ``j``
it fits, on the pre-onset rows of the normalised history, the ridge regression

    x_j[t] ~ x_A[t], x_A[t-1], ..., x_A[t-L], x_j[t-1], 1          (mode "ols_a")

and projects the coefficients and the fit's R^2 into the embedding space,
adding the result to that variable's encoder representation. The treatment
itself gets a zero vector (its regression on itself carries nothing).

Mode ``"ols_path"`` (variant c) adds the second leg of a path through ``j``
to the queried variable ``Y``,

    x_Y[t] ~ x_j[t], ..., x_j[t-L], x_A[t], ..., x_A[t-L], x_Y[t-1], 1,

so that the token of a mediator carries both ``A -> j`` and ``j -> Y`` given
``A``, and the per-token projection can form their product. The token of the
queried variable then carries its own regression on the treatment, and two
role flags (treated, queried) tell the projection which token it is. Pilot
rows 1 to 6 showed why this matters: the tokens have no column identity, so
with "ols_a" the mixer averages the coefficient of every non-treated
variable on the treatment, and on back_door the confounder's reverse
association dilutes the outcome's (0.70 measured against the 0.92 of the
outcome's own regression).

The computation is deterministic and has no trainable parameters besides the
projection, so a model with it is an estimator-informed PFN: a win with it
supports that narrower claim, and the pilot protocol says which variant ran.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MODES = ("ols_a", "ols_path")


class InContextStatistics(nn.Module):
    """Additive per-variable embedding of pre-onset regression coefficients.

    Args:
        embed_size: Width of the encoder representations the features are added to.
        n_lags: Lags of the regressors in the design, 0 to ``n_lags``.
        ridge: Ridge penalty, not applied to the intercept.
        window: At most this many pre-onset rows are used (the encoder's context window).
        clamp: Coefficients are clamped to ``[-clamp, clamp]`` before projection.
        min_rows: Fewer usable rows than this gives zero features for the sample.
        mode: ``"ols_a"`` (regression of every variable on the treatment) or
            ``"ols_path"`` (that plus the queried variable's regression on every
            variable given the treatment, and the role flags).

    Raises:
        ValueError: If ``n_lags`` is negative or ``mode`` is unknown.
    """

    def __init__(
        self,
        embed_size: int,
        n_lags: int = 3,
        ridge: float = 1e-2,
        window: int = 200,
        clamp: float = 10.0,
        min_rows: int = 12,
        mode: str = "ols_a",
    ) -> None:
        super().__init__()
        if n_lags < 0:
            raise ValueError("n_lags must be >= 0")
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode!r} ({' | '.join(MODES)})")
        self.n_lags = int(n_lags)
        self.ridge = float(ridge)
        self.window = int(window)
        self.clamp = float(clamp)
        self.min_rows = int(min_rows)
        self.mode = mode
        # Leg 1: n_lags + 1 treatment lags, the own lag, the intercept; plus R^2.
        self.n_leg1 = self.n_lags + 4
        # Leg 2 (ols_path): n_lags + 1 lags of the variable, n_lags + 1 lags of the
        # treatment given the variable, R^2; then the two role flags.
        self.n_leg2 = 2 * (self.n_lags + 1) + 1 if mode == "ols_path" else 0
        self.n_roles = 2 if mode == "ols_path" else 0
        self.n_features = self.n_leg1 + self.n_leg2 + self.n_roles
        self.proj = nn.Sequential(
            nn.Linear(self.n_features, embed_size), nn.GELU(), nn.Linear(embed_size, embed_size)
        )
        # A zero last layer at init keeps a freshly built model identical to one
        # without the module until training moves it.
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def _solve(
        self, design: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ridge coefficients and R^2 of one design per variable.

        Args:
            design: ``(N, R, d)`` regressors, the intercept in the last column.
            target: ``(N, R)`` regressands.

        Returns:
            Coefficients ``(N, d)`` and R^2 ``(N,)``.
        """
        d = design.shape[2]
        gram = design.transpose(1, 2) @ design  # (N, d, d)
        reg = torch.eye(d, device=design.device, dtype=design.dtype) * self.ridge
        reg[-1, -1] = 0.0  # the intercept is not penalised
        rhs = design.transpose(1, 2) @ target.unsqueeze(2)  # (N, d, 1)
        coef = torch.linalg.solve(gram + reg, rhs).squeeze(2)  # (N, d)
        fitted = (design @ coef.unsqueeze(2)).squeeze(2)  # (N, R)
        resid = target - fitted
        tot = ((target - target.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        r2 = 1.0 - (resid**2).sum(dim=1) / tot.clamp(min=1e-8)
        return coef, r2

    @torch.no_grad()
    def features(
        self,
        X_obs_norm: torch.Tensor,
        variable_mask: torch.Tensor,
        int_onset_idx: torch.Tensor,
        intervention_target: torch.Tensor,
        query_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The raw statistic per variable.

        Args:
            X_obs_norm: ``(B, T, N_max)`` normalised history (rows from the onset on are zero).
            variable_mask: ``(B, N_max)`` ones for real variables.
            int_onset_idx: ``(B,)`` onset row per sample.
            intervention_target: ``(B,)`` treatment column per sample.
            query_target: ``(B,)`` queried column per sample; needed by ``"ols_path"``.

        Returns:
            ``(B, N_max, n_features)`` features, zero for padded variables, the
            treatment, and samples with too few pre-onset rows.

        Raises:
            ValueError: If ``mode`` is ``"ols_path"`` and ``query_target`` is missing.
        """
        if self.mode == "ols_path" and query_target is None:
            raise ValueError("ols_path statistic tokens need the queried column")
        B, _, N = X_obs_norm.shape
        L = self.n_lags
        out = X_obs_norm.new_zeros(B, N, self.n_features)
        for b in range(B):
            onset = int(int_onset_idx[b])
            start = max(L + 1, onset - self.window)
            n_rows = onset - start
            if n_rows < self.min_rows:
                continue
            a = int(intervention_target[b])
            x = X_obs_norm[b]  # (T, N)
            t = torch.arange(start, onset, device=x.device)
            a_lags = torch.stack([x[t - lag, a] for lag in range(L + 1)], dim=1)  # (R, L+1)
            own_lag = x[t - 1, :]  # (R, N)
            target = x[t, :]  # (R, N)
            ones = torch.ones(n_rows, 1, device=x.device, dtype=x.dtype)
            # One design per variable j: [treatment lags | x_j[t-1] | 1], solved for all j at once.
            design = torch.cat(
                [
                    a_lags.unsqueeze(0).expand(N, -1, -1),
                    own_lag.t().unsqueeze(2),
                    ones.unsqueeze(0).expand(N, -1, -1),
                ],
                dim=2,
            )  # (N, R, L+3)
            coef, r2 = self._solve(design, target.t())
            parts = [coef.clamp(-self.clamp, self.clamp), r2.clamp(0.0, 1.0).unsqueeze(1)]
            if self.mode == "ols_path":
                y = int(query_target[b])
                # Second leg, one design per variable j: [x_j lags | treatment lags | x_Y[t-1] | 1]
                # with x_Y[t] as the regressand, the same for every j.
                j_lags = torch.stack([x[t - lag, :] for lag in range(L + 1)], dim=2)  # (R, N, L+1)
                design2 = torch.cat(
                    [
                        j_lags.permute(1, 0, 2),
                        a_lags.unsqueeze(0).expand(N, -1, -1),
                        x[t - 1, y].reshape(1, n_rows, 1).expand(N, -1, -1),
                        ones.unsqueeze(0).expand(N, -1, -1),
                    ],
                    dim=2,
                )  # (N, R, 2L+4)
                coef2, r2_2 = self._solve(design2, x[t, y].unsqueeze(0).expand(N, -1))
                leg2 = torch.cat(
                    [
                        coef2[:, : 2 * (L + 1)].clamp(-self.clamp, self.clamp),
                        r2_2.clamp(0.0, 1.0).unsqueeze(1),
                    ],
                    dim=1,
                )
                # The queried variable regressed on itself at lag 0 is a tautology.
                leg2[y] = 0.0
                roles = x.new_zeros(N, 2)
                roles[a, 0] = 1.0
                roles[y, 1] = 1.0
                parts += [leg2, roles]
            feat = torch.cat(parts, dim=1)
            feat = feat * variable_mask[b].unsqueeze(1)
            feat[a] = 0.0
            if self.mode == "ols_path":
                # The treatment's token keeps its role flag only.
                feat[a, self.n_features - 2] = 1.0
            out[b] = feat
        return out

    def forward(self, h_vars: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Add the projected statistics to the per-variable representations.

        Args:
            h_vars: ``(B, N_max, E)`` encoder output.
            batch: Needs ``X_obs_norm``, ``variable_mask``, ``int_onset_idx`` and
                ``intervention_target``; ``"ols_path"`` also needs ``query_target``.
                In a multi-query batch the treatment is read per trajectory
                through ``_traj_idx``.

        Returns:
            ``(B, N_max, E)``.

        Raises:
            KeyError: If the batch lacks a required field.
            NotImplementedError: For ``"ols_path"`` on a multi-query batch, whose
                queried column differs between the queries of one trajectory.
        """
        target = batch["intervention_target"]
        multi = "_traj_idx" in batch and target.shape[0] != h_vars.shape[0]
        if multi:
            # Query-level field in a multi-query batch: the treatment is a property of
            # the trajectory, so any query of a trajectory gives the same value.
            per_traj = target.new_zeros(h_vars.shape[0])
            per_traj[batch["_traj_idx"]] = target
            target = per_traj
        query_target = None
        if self.mode == "ols_path":
            # The queried column is query-level: with several queries per trajectory
            # there is no single second leg per token.
            if "_traj_idx" in batch:
                raise NotImplementedError("ols_path statistic tokens need one query per trajectory")
            query_target = batch["query_target"]
        feats = self.features(
            batch["X_obs_norm"],
            batch["variable_mask"],
            batch["int_onset_idx"],
            target,
            query_target,
        )
        return h_vars + self.proj(feats.to(h_vars.dtype))
