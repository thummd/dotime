"""Small-MLP drift mechanism for continuous-time causal models.

The linear Ornstein-Uhlenbeck mechanism in :mod:`ou_mechanism` assumes
that the drift on variable ``v`` is a linear function of its parents
and a linear relaxation on ``X_v`` itself::

    dX_v = ( -theta_v * X_v + sum_u w_{v,u} X_u ) dt + sigma_v dW_v.

Phase 10 extends the prior to nonlinear drifts::

    dX_v = ( -theta_v * X_v + s_v * tanh(W2 tanh(W1 inp + b1) + b2) ) dt + sigma_v dW_v,

where ``inp = [X_v, X_{u_1}, ..., X_{u_k}]`` concatenates the variable's
own state and its parents, and ``(W1, b1, W2, b2)`` are a two-layer
tanh-MLP whose weights are *freshly sampled* per mechanism (not
trained).  Like the OU mechanism, drawing a new SCM per batch means
the PFN amortises causal inference across a family of distinct drift
functions rather than learning a single one.

Design choices
--------------
1. **Explicit mean-reversion term.**  We retain ``-theta_v * X_v`` in
   front of the MLP output.  The MLP output is bounded to
   ``[-s_v, s_v]`` by the outer ``tanh``, so ``theta_v > 0`` guarantees
   that trajectories remain bounded; without this, a random MLP could
   introduce positive feedback and explode under Euler-Maruyama.
2. **Hand-written forward.**  We store ``W1, b1, W2, b2`` as
   :class:`torch.Tensor` attributes and run the MLP manually in
   :meth:`drift`.  This avoids registering an :class:`nn.Module`
   (which would drag in parameter bookkeeping and complicate
   multi-processing / pickling in the on-the-fly dataloader).
3. **AR(1) reduction at zero MLP weights.**  If ``W1 = W2 = 0``, the
   mechanism is exactly a mean-reverting OU drift with no parental
   influence -- consistent with the discrete AR(1) special case that
   the OU mechanism also recovers.
4. **Drop-in interface.**  The class exposes ``drift``, ``step``,
   ``sigma``, and ``parents`` with the same signatures as
   :class:`OUMechanism`, so :class:`ContinuousSCM._step` can dispatch
   over both without a type check.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class NeuralDriftMechanism:
    """Parameters of a small-MLP drift mechanism for a single variable.

    Attributes
    ----------
    theta : float
        Mean-reversion rate (must be positive), applied to ``X_v``
        *outside* the MLP to guarantee bounded trajectories.
    sigma : float
        Diffusion coefficient (must be positive).
    out_scale : float
        Post-MLP gain (must be non-negative).  The MLP output lies in
        ``[-1, 1]`` after the outer tanh, so this scales the effective
        drift contribution from parents + nonlinear self-interaction.
    W1 : torch.Tensor
        First-layer weight matrix, shape ``(hidden, 1 + len(parents))``.
    b1 : torch.Tensor
        First-layer bias, shape ``(hidden,)``.
    W2 : torch.Tensor
        Second-layer weight matrix, shape ``(1, hidden)``.
    b2 : torch.Tensor
        Second-layer bias, shape ``(1,)``.
    parents : tuple of int
        Indices (into the variable list) of this mechanism's parents.
        Empty tuple means a root node; the MLP still operates on
        ``X_v`` alone in that case.
    """

    theta: float
    sigma: float
    out_scale: float
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor
    parents: Sequence[int] = ()

    def __post_init__(self) -> None:
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.out_scale < 0:
            raise ValueError(f"out_scale must be non-negative, got {self.out_scale}")

        in_dim = 1 + len(self.parents)
        if self.W1.dim() != 2 or self.W1.shape[1] != in_dim:
            raise ValueError(f"W1 must have shape (hidden, {in_dim}), got {tuple(self.W1.shape)}")
        hidden = self.W1.shape[0]
        if self.b1.shape != (hidden,):
            raise ValueError(f"b1 must have shape ({hidden},), got {tuple(self.b1.shape)}")
        if self.W2.shape != (1, hidden):
            raise ValueError(f"W2 must have shape (1, {hidden}), got {tuple(self.W2.shape)}")
        if self.b2.shape != (1,):
            raise ValueError(f"b2 must have shape (1,), got {tuple(self.b2.shape)}")

    # ------------------------------------------------------------------ forward

    def _build_input(
        self,
        x_self: torch.Tensor,
        x_parents: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate ``[x_self, x_parents]`` into a 1-D tensor."""
        x_self_flat = x_self.reshape(1).to(self.W1.dtype)
        if len(self.parents) == 0:
            return x_self_flat
        return torch.cat([x_self_flat, x_parents.to(self.W1.dtype)])

    def drift(
        self,
        x_self: torch.Tensor,
        x_parents: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic drift ``-theta * X_v + out_scale * MLP([X_v, Pa])``.

        Parameters
        ----------
        x_self : torch.Tensor
            Scalar tensor with the variable's current value.
        x_parents : torch.Tensor
            1-D tensor of parent values (empty for roots).

        Returns
        -------
        torch.Tensor
            Scalar drift ``dX / dt`` at the current state.
        """
        inp = self._build_input(x_self, x_parents)
        hidden = torch.tanh(self.W1.to(inp) @ inp + self.b1.to(inp))
        nn_out = torch.tanh(self.W2.to(inp) @ hidden + self.b2.to(inp)).squeeze()
        return -self.theta * x_self + self.out_scale * nn_out

    def step(
        self,
        x_self: torch.Tensor,
        x_parents: torch.Tensor,
        dt: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the variable by one Euler-Maruyama step.

        ``X(t + dt) = X(t) + drift(X(t)) * dt + sigma * sqrt(dt) * noise``
        """
        drift = self.drift(x_self, x_parents)
        return x_self + drift * dt + self.sigma * torch.sqrt(dt) * noise


# ---------------------------------------------------------------------- sampling


def sample_neural_drift_mechanism(
    parents: Sequence[int],
    theta_range: tuple = (0.5, 2.0),
    sigma_range: tuple = (0.2, 0.8),
    out_scale_range: tuple = (0.5, 2.0),
    hidden_dim: int = 8,
    weight_scale: float | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> NeuralDriftMechanism:
    """Draw a random :class:`NeuralDriftMechanism` from priors on its parameters.

    - ``theta ~ Uniform(theta_range)`` (mean-reversion rate, kept outside
      the MLP for stability).
    - ``sigma ~ Uniform(sigma_range)``.
    - ``out_scale ~ Uniform(out_scale_range)`` (post-tanh gain; effective
      drift contribution stays in ``[-out_scale, out_scale]``).
    - MLP weights use Xavier/Glorot-style initialisation:
      ``W ~ N(0, std=weight_scale / sqrt(fan_in))`` with ``weight_scale``
      defaulting to ``1.0``.  Biases are drawn ``N(0, 0.1^2)`` so the
      nonlinearity is active from the start rather than stuck at zero.
    """
    if theta_range[0] <= 0 or theta_range[1] <= theta_range[0]:
        raise ValueError(f"invalid theta_range: {theta_range}")
    if sigma_range[0] <= 0 or sigma_range[1] <= sigma_range[0]:
        raise ValueError(f"invalid sigma_range: {sigma_range}")
    if out_scale_range[0] < 0 or out_scale_range[1] < out_scale_range[0]:
        raise ValueError(f"invalid out_scale_range: {out_scale_range}")
    if hidden_dim < 1:
        raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

    ws = 1.0 if weight_scale is None else float(weight_scale)
    if ws <= 0:
        raise ValueError(f"weight_scale must be positive, got {ws}")

    u = torch.empty(3, device=device, dtype=dtype)
    u.uniform_(0.0, 1.0, generator=generator)
    theta = float(theta_range[0] + u[0] * (theta_range[1] - theta_range[0]))
    sigma = float(sigma_range[0] + u[1] * (sigma_range[1] - sigma_range[0]))
    out_scale = float(out_scale_range[0] + u[2] * (out_scale_range[1] - out_scale_range[0]))

    in_dim = 1 + len(parents)
    std1 = ws / (in_dim**0.5)
    std2 = ws / (hidden_dim**0.5)

    W1 = torch.empty(hidden_dim, in_dim, device=device, dtype=dtype)
    W1.normal_(mean=0.0, std=std1, generator=generator)
    b1 = torch.empty(hidden_dim, device=device, dtype=dtype)
    b1.normal_(mean=0.0, std=0.1, generator=generator)
    W2 = torch.empty(1, hidden_dim, device=device, dtype=dtype)
    W2.normal_(mean=0.0, std=std2, generator=generator)
    b2 = torch.empty(1, device=device, dtype=dtype)
    b2.normal_(mean=0.0, std=0.1, generator=generator)

    return NeuralDriftMechanism(
        theta=theta,
        sigma=sigma,
        out_scale=out_scale,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        parents=tuple(parents),
    )
