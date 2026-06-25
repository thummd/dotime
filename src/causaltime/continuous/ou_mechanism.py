"""Ornstein-Uhlenbeck mechanism for continuous-time causal models.

An OU mechanism describes the dynamics of a single variable ``X_v`` with
lagged causal influence from its parents ``Pa(v)`` as:

    dX_v = ( -theta_v * X_v + sum_{u in Pa(v)} w_{v,u} * X_u ) dt + sigma_v dW_v

where ``theta_v > 0`` is the mean-reversion rate, ``sigma_v > 0`` is the
diffusion coefficient, and ``w_{v,u}`` are real-valued parental weights.
The long-run mean is implicitly zero (shift in :class:`ContinuousSCM` if
needed).

Equivalence with AR(1)
----------------------
Discretising the univariate OU SDE (no parents) via Euler-Maruyama with
step ``dt`` gives::

    X(t + dt) = X(t) - theta * X(t) * dt + sigma * sqrt(dt) * Z
              = (1 - theta * dt) * X(t) + sigma * sqrt(dt) * Z,   Z ~ N(0, 1)

which is exactly the AR(1) form with coefficient ``c2 = 1 - theta*dt``
and noise scale ``c3 = sigma * sqrt(dt)``. At ``dt = 1`` and ``theta =
1 - w`` this recovers the discrete-time mechanism in
``dotime/prior/batched_tscm.py`` (without parental drift).

This module only provides the mechanism spec and per-step drift /
diffusion computation; the integration loop lives in
:class:`ContinuousSCM`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass
class OUMechanism:
    """Parameters of a linear-drift OU mechanism for a single variable.

    Attributes
    ----------
    theta : float
        Mean-reversion rate (must be positive). Smaller ``theta`` means
        slower relaxation toward zero.
    sigma : float
        Diffusion coefficient (must be positive). Controls stochastic
        variability per unit time.
    parent_weights : torch.Tensor
        1-D tensor of shape ``(len(parents),)`` with the weight
        ``w_{v,u}`` for each parent ``u`` (in the order recorded in
        ``parents``).
    parents : tuple of int
        Indices (into the variable list) of this mechanism's parents.
        Empty tuple means a root node.
    """

    theta: float
    sigma: float
    parent_weights: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    parents: Sequence[int] = ()

    def __post_init__(self) -> None:
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if len(self.parents) != self.parent_weights.numel():
            raise ValueError(
                f"parent_weights length {self.parent_weights.numel()} does not match "
                f"number of parents {len(self.parents)}"
            )

    def drift(
        self,
        x_self: torch.Tensor,
        x_parents: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic drift at the current state.

        Parameters
        ----------
        x_self : torch.Tensor
            Scalar tensor with the variable's current value.
        x_parents : torch.Tensor
            1-D tensor of shape ``(len(parents),)`` with parents' current
            values (in the same order as ``self.parents``). Pass an empty
            tensor if the variable has no parents.

        Returns
        -------
        torch.Tensor
            Scalar drift ``dX / dt`` at the current state.
        """
        drift = -self.theta * x_self
        if self.parent_weights.numel() > 0:
            drift = drift + (self.parent_weights.to(x_parents) * x_parents).sum()
        return drift

    def step(
        self,
        x_self: torch.Tensor,
        x_parents: torch.Tensor,
        dt: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the variable by one Euler-Maruyama step.

        ``X(t + dt) = X(t) + drift(X(t)) * dt + sigma * sqrt(dt) * noise``

        Parameters
        ----------
        x_self : torch.Tensor
            Scalar tensor with the variable's value at time ``t``.
        x_parents : torch.Tensor
            Parents' values at time ``t`` (see :meth:`drift`).
        dt : torch.Tensor
            Scalar positive time increment.
        noise : torch.Tensor
            Scalar standard-normal sample. The simulator pre-samples all
            noise up front so that counterfactual pairs can share noise.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the variable's value at time ``t + dt``.
        """
        drift = self.drift(x_self, x_parents)
        return x_self + drift * dt + self.sigma * torch.sqrt(dt) * noise


def sample_ou_mechanism(
    parents: Sequence[int],
    theta_range: tuple = (0.5, 2.0),
    sigma_range: tuple = (0.2, 0.8),
    weight_scale: float = 0.8,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> OUMechanism:
    """Draw a random :class:`OUMechanism` from priors on its parameters.

    - ``theta ~ Uniform(theta_range)``, controlling the mean-reversion
      timescale ``1 / theta``.
    - ``sigma ~ Uniform(sigma_range)``, controlling noise intensity.
    - ``parent_weights ~ N(0, weight_scale^2)``.

    Parameters
    ----------
    parents : sequence of int
        Indices of the parent variables. May be empty.
    theta_range, sigma_range : tuple of float
        Half-open uniform priors on ``theta`` and ``sigma``. Both bounds
        must be positive.
    weight_scale : float
        Standard deviation of the Gaussian prior on parental weights.
    generator : torch.Generator, optional
        RNG for reproducibility.
    device, dtype : torch RNG specifications
        Applied to the sampled parent-weight tensor.

    Returns
    -------
    OUMechanism
    """
    if theta_range[0] <= 0 or theta_range[1] <= theta_range[0]:
        raise ValueError(f"invalid theta_range: {theta_range}")
    if sigma_range[0] <= 0 or sigma_range[1] <= sigma_range[0]:
        raise ValueError(f"invalid sigma_range: {sigma_range}")

    u = torch.empty(2, device=device, dtype=dtype)
    u.uniform_(0.0, 1.0, generator=generator)
    theta = float(theta_range[0] + u[0] * (theta_range[1] - theta_range[0]))
    sigma = float(sigma_range[0] + u[1] * (sigma_range[1] - sigma_range[0]))

    if len(parents) == 0:
        w = torch.empty(0, device=device, dtype=dtype)
    else:
        w = torch.empty(len(parents), device=device, dtype=dtype)
        w.normal_(mean=0.0, std=weight_scale, generator=generator)

    return OUMechanism(
        theta=theta,
        sigma=sigma,
        parent_weights=w,
        parents=tuple(parents),
    )
