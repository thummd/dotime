"""Continuous-time analogue of :mod:`dotime.prior.tscm_sampler`.

Takes one of the named :class:`TSCMStructure` topologies (back-door,
front-door, instrumental variable, ...) and produces a
:class:`ContinuousSCM` whose variables are parameterised as
Ornstein-Uhlenbeck mechanisms on that topology.

Why a separate class rather than one on ``TSCMSampler``
-------------------------------------------------------
``TSCMSampler`` constructs discrete-time :class:`TemporalSCM` objects
with both instantaneous (``G_0``) and lagged (``G_lags``) edges.  In
continuous time the Euler-Maruyama update advances every variable
simultaneously using only *previous-step* parent values, so the
``G_0`` / ``G_lags[0]`` distinction collapses into a single parent set
(the union of the two predecessor sets, minus autoregressive self-loops
that are already captured by the OU drift term).  This module performs
that reduction.

The mapping preserves every topology needed for identifiability case
studies, so downstream evaluation code can use the same structure names.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from causaltimeprior.tscm_sampler import TSCMSampler, TSCMStructure

from .continuous_scm import ContinuousSCM
from .ou_mechanism import OUMechanism, sample_ou_mechanism


class ContinuousTSCMSampler:
    """Sample a :class:`ContinuousSCM` with a fixed :class:`TSCMStructure`.

    Parameters
    ----------
    structure : TSCMStructure
        Named topology to sample on.  All 8 structures defined in
        :class:`TSCMStructure` are supported because the topology
        extraction is delegated to :class:`TSCMSampler`.
    theta_range : tuple of float
        Uniform prior on the OU mean-reversion rate.  Small theta means
        slow relaxation toward zero (long memory); large theta means
        fast relaxation (short memory).
    sigma_range : tuple of float
        Uniform prior on the OU diffusion coefficient.
    weight_scale : float
        Standard deviation of the Gaussian prior on parental weights.
        Matches the ``sigma_w`` interpretation in
        :mod:`dotime.prior.tscm_sampler` but with a smaller default
        (continuous-time trajectories are integrated, so large weights
        are more likely to destabilise the dynamics).
    use_lagged_edges : bool
        Kept for API parity with :class:`TSCMSampler`.  In the
        continuous setting every edge is effectively lagged (evaluated
        on pre-step parent values); toggling this only matters for the
        instantaneous-only variants that rely on it, which we don't
        use here.  Defaults to ``True``.
    """

    def __init__(
        self,
        structure: TSCMStructure,
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        weight_scale: float = 0.5,
        use_lagged_edges: bool = True,
    ) -> None:
        self.structure = structure
        self.theta_range = theta_range
        self.sigma_range = sigma_range
        self.weight_scale = weight_scale
        self.use_lagged_edges = use_lagged_edges

        # Build a single discrete sampler just to harvest the topology.
        # Mechanism / noise settings here are irrelevant — we only need
        # the DAG and topological node order.
        self._topology_helper = TSCMSampler(
            structure=structure,
            max_lag=1,
            use_lagged_edges=use_lagged_edges,
        )
        self._dag = self._topology_helper._build_dag()
        self._node_names: List[str] = list(self._dag.topo_order)
        self._parents_per_node: List[List[int]] = self._build_parent_lists()

    # ------------------------------------------------------------------ topology

    def _build_parent_lists(self) -> List[List[int]]:
        """Collapse ``G_0`` and ``G_lags[0]`` into a single parent list per node.

        The DAG is addressed by topological-order index.  We read
        instantaneous predecessors from ``G_0`` (networkx DiGraph over
        node *names*) and lagged predecessors from ``G_lags[0]`` (numpy
        matrix over topological *indices*).  Autoregressive self-loops
        in ``G_lags`` are dropped because the OU drift term ``-theta *
        X_v`` already captures self-attenuation.
        """
        topo = self._node_names
        name_to_idx = {name: i for i, name in enumerate(topo)}
        parents: List[List[int]] = [[] for _ in topo]

        # Instantaneous edges (from G_0): parents[v] += G_0.predecessors(v)
        for v_name in topo:
            for u_name in self._dag.G_0.predecessors(v_name):
                v_idx = name_to_idx[v_name]
                u_idx = name_to_idx[u_name]
                if u_idx != v_idx:
                    parents[v_idx].append(u_idx)

        # Lagged edges (from G_lags[0]): G_lags[0][u_idx, v_idx] == 1 means u(t-1) -> v(t)
        if len(self._dag.G_lags) > 0:
            G_lag = self._dag.G_lags[0]
            N = len(topo)
            for v in range(N):
                for u in range(N):
                    if u != v and G_lag[u, v] > 0:
                        if u not in parents[v]:
                            parents[v].append(u)

        return parents

    # ------------------------------------------------------------------ metadata

    @property
    def node_names(self) -> List[str]:
        """Topological-order node names (e.g. ``['X', 'A', 'Y']``)."""
        return list(self._node_names)

    @property
    def n_vars(self) -> int:
        return len(self._node_names)

    def get_hidden_vars(self) -> List[int]:
        """Indices of unobserved variables (U), or empty list if all observed."""
        return self._topology_helper.get_hidden_vars()

    def get_intervention_target(self) -> int:
        """Topological-order index of the treatment variable ``A``."""
        return self._topology_helper.get_intervention_target()

    def get_outcome_var(self) -> int:
        """Topological-order index of the outcome variable ``Y``."""
        return self._topology_helper.get_outcome_var()

    # ------------------------------------------------------------------ sampling

    def sample(
        self,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> ContinuousSCM:
        """Draw a random :class:`ContinuousSCM` with the stored topology.

        Parameters
        ----------
        generator : torch.Generator, optional
            RNG for mechanism parameter draws.
        device, dtype : torch specs
            Applied to the constructed SCM's tensors.

        Returns
        -------
        ContinuousSCM
        """
        mechanisms: List[OUMechanism] = []
        for v_idx in range(self.n_vars):
            mechanisms.append(
                sample_ou_mechanism(
                    parents=tuple(self._parents_per_node[v_idx]),
                    theta_range=self.theta_range,
                    sigma_range=self.sigma_range,
                    weight_scale=self.weight_scale,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            )
        return ContinuousSCM(mechanisms, device=device, dtype=dtype)
