"""Random-graph continuous-time prior.

The named-TSCM samplers in :mod:`tscm_sampler` fix both the topology
and the (intervention, outcome) choice.  For the workshop paper we
also want a training distribution that covers *arbitrary* DAGs with
arbitrary (A, Y) roles -- analogous to CausalTimePrior's random-graph
path in the discrete-time DoT-PFN pipeline.

This module defines:

- :class:`RandomContinuousSCMSampler`: samples a fresh ContinuousSCM
  on each call with random N in ``[n_min, n_max_prior]``, random
  topology via :meth:`ContinuousSCM.sample_random`, and random
  (A, Y) roles.  Each non-(A, Y) node is independently marked hidden
  with probability ``hidden_prob``.  Hidden nodes still participate
  in the simulated dynamics but are masked out of ``X_obs``,
  ``X_int``, and ``variable_mask`` by the extended prior -- the same
  semantics as the hidden ``U`` in named structures like
  ``front_door`` and ``instrumental_variable``.
- :class:`RandomContinuousExtendedPrior`: subclasses
  :class:`ContinuousExtendedPrior` and overrides ``_sample_scm_context``
  to plug the random sampler into the existing generate_sample /
  generate_batch pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .continuous_scm import ContinuousSCM
from .extended_prior import (
    ContinuousExtendedPrior,
    _SampledSCMContext,
)
from .regime_switching import ContinuousRegimeSwitchingSCM


def _pick_intervention_and_outcome(
    n_vars: int,
    rng: np.random.RandomState,
    require_distinct_roots: bool = False,
) -> tuple:
    """Choose ``(intervention_target, outcome_var)`` in topological order.

    By construction :meth:`ContinuousSCM.sample_random` places variable
    ``i`` after all its parents (parents are uniformly drawn from
    ``[0, i)``), so picking ``outcome_var > intervention_target`` makes
    the outcome at least reachable from later variables in topological
    order.  That's a mild guarantee that queries are non-trivial for
    most sampled graphs, without needing to trace descendants.

    Parameters
    ----------
    n_vars : int
        Number of variables in the sampled SCM.
    rng : np.random.RandomState
        RNG for reproducibility.
    require_distinct_roots : bool
        Unused in the current heuristic; kept for API parity with a
        possible future implementation that forces A to be a root and
        Y to be a leaf.
    """
    if n_vars < 2:
        raise ValueError(f"need n_vars >= 2 to pick distinct (A, Y), got {n_vars}")

    a_idx = int(rng.randint(0, n_vars - 1))  # exclude the last node
    y_idx = int(rng.randint(a_idx + 1, n_vars))
    return a_idx, y_idx


class RandomContinuousSCMSampler:
    """Random-graph continuous-time SCM sampler.

    Parameters
    ----------
    n_min, n_max_prior : int
        Bounds on the number of variables per sample (inclusive).  The
        default range matches CausalTimePrior's discrete-time prior.
    edge_prob : float
        Erdos-Renyi edge probability on the topological order (each
        pair ``(i, j)`` with ``i < j`` gets an edge with this
        probability).  Defaults to ``0.3`` to match CTP.
    theta_range, sigma_range, weight_scale :
        Forwarded to :func:`sample_ou_mechanism`.
    seed : int
        Base RNG seed.  The sampler uses an internal
        :class:`torch.Generator` and :class:`np.random.RandomState`
        that advance on every call; only the starting point is seeded
        here.
    """

    def __init__(
        self,
        n_min: int = 3,
        n_max_prior: int = 10,
        edge_prob: float = 0.3,
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        weight_scale: float = 0.5,
        hidden_prob: float = 0.0,
        regime_prob: float = 0.0,
        regime_count_range: tuple = (2, 3),
        sticky_alpha: float = 9.0,
        other_alpha: float = 0.5,
        mechanism_kind: str = "linear",
        p_neural: float = 0.0,
        neural_hidden_dim: int = 8,
        neural_out_scale_range: tuple = (0.5, 2.0),
        seed: int = 0,
    ) -> None:
        if not 2 <= n_min <= n_max_prior:
            raise ValueError(
                f"require 2 <= n_min <= n_max_prior, got n_min={n_min}, "
                f"n_max_prior={n_max_prior}"
            )
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError(f"edge_prob must be in [0, 1], got {edge_prob}")
        if not 0.0 <= hidden_prob <= 1.0:
            raise ValueError(f"hidden_prob must be in [0, 1], got {hidden_prob}")
        if not 0.0 <= regime_prob <= 1.0:
            raise ValueError(f"regime_prob must be in [0, 1], got {regime_prob}")
        if not (1 <= regime_count_range[0] <= regime_count_range[1]):
            raise ValueError(
                f"regime_count_range must be (lo, hi) with 1 <= lo <= hi, "
                f"got {regime_count_range}"
            )
        if mechanism_kind not in ("linear", "neural", "mixed"):
            raise ValueError(
                f"mechanism_kind must be 'linear', 'neural', or 'mixed'; "
                f"got {mechanism_kind!r}"
            )
        if not 0.0 <= p_neural <= 1.0:
            raise ValueError(f"p_neural must be in [0, 1], got {p_neural}")

        self.n_min = int(n_min)
        self.n_max_prior = int(n_max_prior)
        self.edge_prob = float(edge_prob)
        self.theta_range = tuple(theta_range)
        self.sigma_range = tuple(sigma_range)
        self.weight_scale = float(weight_scale)
        self.hidden_prob = float(hidden_prob)
        self.regime_prob = float(regime_prob)
        self.regime_count_range = tuple(regime_count_range)
        self.sticky_alpha = float(sticky_alpha)
        self.other_alpha = float(other_alpha)
        self.mechanism_kind = str(mechanism_kind)
        self.p_neural = float(p_neural)
        self.neural_hidden_dim = int(neural_hidden_dim)
        self.neural_out_scale_range = tuple(neural_out_scale_range)

        self._torch_gen = torch.Generator().manual_seed(int(seed))
        self._np_rng = np.random.RandomState(int(seed))

    @property
    def n_vars(self) -> int:
        """Upper bound on variables per sample (for init-time sanity checks)."""
        return self.n_max_prior

    def sample(
        self,
        generator: Optional[torch.Generator] = None,
    ) -> tuple:
        """Return ``(scm, n_vars, A_topo, Y_topo, hidden_topo)``.

        ``hidden_topo`` is a list of topological-order indices that
        should be masked out of ``X_obs`` / ``X_int`` / ``variable_mask``
        by the extended prior.  ``A`` and ``Y`` are guaranteed to be
        *excluded* from the hidden list.

        With probability ``regime_prob`` the returned ``scm`` is a
        :class:`ContinuousRegimeSwitchingSCM` with ``R`` regimes drawn
        uniformly from ``regime_count_range``; otherwise it's a plain
        :class:`ContinuousSCM`.  Both have a compatible ``simulate``
        signature so the extended prior can dispatch uniformly.

        The ``generator`` arg is kept for interface parity with
        :class:`ContinuousTSCMSampler` but the sampler's own
        ``_torch_gen`` / ``_np_rng`` are used throughout for
        reproducibility.
        """
        n_vars = int(self._np_rng.randint(self.n_min, self.n_max_prior + 1))

        is_regime_switching = (
            self.regime_prob > 0.0 and self._np_rng.rand() < self.regime_prob
        )
        if is_regime_switching:
            n_regimes = int(self._np_rng.randint(
                self.regime_count_range[0], self.regime_count_range[1] + 1,
            ))
            scm = ContinuousRegimeSwitchingSCM.sample_random(
                n_vars=n_vars,
                n_regimes=n_regimes,
                edge_prob=self.edge_prob,
                theta_range=self.theta_range,
                sigma_range=self.sigma_range,
                weight_scale=self.weight_scale,
                sticky_alpha=self.sticky_alpha,
                other_alpha=self.other_alpha,
                generator=self._torch_gen,
            )
        else:
            scm = ContinuousSCM.sample_random(
                n_vars=n_vars,
                edge_prob=self.edge_prob,
                theta_range=self.theta_range,
                sigma_range=self.sigma_range,
                weight_scale=self.weight_scale,
                mechanism_kind=self.mechanism_kind,
                p_neural=self.p_neural,
                neural_hidden_dim=self.neural_hidden_dim,
                neural_out_scale_range=self.neural_out_scale_range,
                generator=self._torch_gen,
            )

        a_idx, y_idx = _pick_intervention_and_outcome(n_vars, self._np_rng)

        # Decide which of the non-(A, Y) nodes are hidden.  We never hide
        # A or Y because the model needs to intervene on A and predict Y.
        hidden_topo: list = []
        if self.hidden_prob > 0.0:
            for v in range(n_vars):
                if v == a_idx or v == y_idx:
                    continue
                if self._np_rng.rand() < self.hidden_prob:
                    hidden_topo.append(v)

        return scm, n_vars, a_idx, y_idx, hidden_topo


class RandomContinuousExtendedPrior(ContinuousExtendedPrior):
    """Model-ready batch generator with random-graph continuous-time SCMs.

    Extends :class:`ContinuousExtendedPrior` by overriding the
    topology-sampling hook; all other behaviour (schedule sampling,
    intervention kind + value, counterfactual / interventional pair
    modes, canonical permutation, query sampling) is inherited.

    The ``tscm_structure`` argument to the parent ``__init__`` is still
    accepted to keep the constructor signature compatible with the
    discrete-time dataloader's single-string knob, but the value is
    only used to pick a no-op placeholder when the parent initialises
    its ``ContinuousTSCMSampler`` cache.  The actual sampler used at
    runtime is :class:`RandomContinuousSCMSampler`.
    """

    def __init__(
        self,
        n_min: int = 3,
        n_max_prior: int = 10,
        edge_prob: float = 0.3,
        hidden_prob: float = 0.0,
        regime_prob: float = 0.0,
        regime_count_range: tuple = (2, 3),
        sticky_alpha: float = 9.0,
        other_alpha: float = 0.5,
        mechanism_kind: str = "linear",
        p_neural: float = 0.0,
        neural_hidden_dim: int = 8,
        neural_out_scale_range: tuple = (0.5, 2.0),
        tscm_structure_placeholder: str = "rct_no_confounding",
        **kwargs,
    ) -> None:
        # Parent __init__ expects theta_range, sigma_range, weight_scale; we
        # forward them to the parent AND to the random sampler so the OU
        # hyperprior stays consistent across the fixed- and random-graph paths.
        super().__init__(tscm_structure=tscm_structure_placeholder, **kwargs)

        self.random_sampler = RandomContinuousSCMSampler(
            n_min=n_min,
            n_max_prior=n_max_prior,
            edge_prob=edge_prob,
            theta_range=self._forwarded_kwarg(kwargs, "theta_range", (0.5, 2.0)),
            sigma_range=self._forwarded_kwarg(kwargs, "sigma_range", (0.2, 0.6)),
            weight_scale=self._forwarded_kwarg(kwargs, "weight_scale", 0.5),
            hidden_prob=hidden_prob,
            regime_prob=regime_prob,
            regime_count_range=regime_count_range,
            sticky_alpha=sticky_alpha,
            other_alpha=other_alpha,
            mechanism_kind=mechanism_kind,
            p_neural=p_neural,
            neural_hidden_dim=neural_hidden_dim,
            neural_out_scale_range=neural_out_scale_range,
            seed=self._seed,
        )

    @staticmethod
    def _forwarded_kwarg(kwargs: dict, key: str, default):
        return kwargs.get(key, default)

    @property
    def n_vars(self) -> int:
        """Upper bound on variables per trajectory."""
        return self.random_sampler.n_max_prior

    @property
    def hidden_vars(self) -> list:
        """Init-time accessor; the per-sample hidden indices live on the
        :class:`_SampledSCMContext` returned by ``_sample_scm_context``.

        Because :class:`RandomContinuousSCMSampler` samples a fresh set
        of hidden indices per trajectory, this attribute is *not* a
        stable list -- it reports the configured ``hidden_prob``
        indirectly by advertising an empty default.  Downstream code
        should read the context object rather than this property.
        """
        return []

    # ------------------------------------------------------------------ hook

    def _sample_scm_context(self) -> _SampledSCMContext:
        scm, n_vars, a_idx, y_idx, hidden_topo = self.random_sampler.sample(
            generator=self._torch_gen,
        )
        return _SampledSCMContext(
            scm=scm,
            n_vars=n_vars,
            intervention_target_topo=a_idx,
            outcome_var_topo=y_idx,
            hidden_vars_topo=list(hidden_topo),
        )
