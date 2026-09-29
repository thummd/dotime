"""Regime-switching temporal SCMs for DoTime.

This module implements temporal SCMs where the causal structure and/or mechanisms
can change over time according to a discrete Markov process.
"""

import warnings

import numpy as np
import torch

from dotime._sampling import DistributionSampler
from dotime.interventions import InterventionSpec, InterventionType
from dotime.temporal_graph import TemporalDAG
from dotime.temporal_mechanism import TemporalMechanism
from dotime.utils import check_divergence, clip_values


class RegimeSwitchingTemporalSCM:
    """
    Regime-switching temporal SCM.

    At each time step, a discrete regime variable d_t ∈ {0,...,R-1} determines
    which causal structure and mechanisms are active:

    X_t^(i) = f_i^(d_t)(Pa_{G^(d_t)}(X_t^(i))) + ε_t^(i)

    Regime transitions follow a Markov chain: d_t ~ Categorical(P[d_{t-1}, :])
    """

    def __init__(
        self,
        dags: list[TemporalDAG],
        mechanisms: list[dict[str, TemporalMechanism]],
        noise: dict[str, DistributionSampler],
        transition_matrix: np.ndarray,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        divergence_threshold: float | None = None,
    ):
        """
        Parameters
        ----------
        dags : List[TemporalDAG]
            List of temporal DAGs, one per regime.
        mechanisms : List[Dict[str, TemporalMechanism]]
            List of mechanism dictionaries, one per regime.
        noise : Dict[str, DistributionSampler]
            Noise distributions (shared across regimes).
        transition_matrix : np.ndarray
            Regime transition matrix P of shape (R, R).
        device : torch.device
            Device for computation.
        dtype : torch.dtype
            Data type.
        divergence_threshold : float, optional
            If set, a simulation in which any value exceeds this magnitude
            (burn-in included) counts as diverged. It then returns zeros with a
            ``RuntimeWarning``, as ``TemporalSCM`` does above 500. ``None``
            (default) keeps the v1.0.0 check, which flags only NaN, inf or
            values above 1e6 and so cannot fire on values clipped to +-1000.

        Raises
        ------
        ValueError
            If ``divergence_threshold`` is set but not positive.
        """
        if divergence_threshold is not None and not divergence_threshold > 0:
            raise ValueError(f"divergence_threshold must be positive, got {divergence_threshold}")
        self.dags = dags
        self.mechanisms = mechanisms
        self.noise = noise
        self.transition_matrix = transition_matrix
        self.device = device
        self.dtype = dtype
        self.divergence_threshold = divergence_threshold

        self.num_regimes = len(dags)
        assert len(mechanisms) == self.num_regimes
        assert transition_matrix.shape == (self.num_regimes, self.num_regimes)

        # Precompute parents for each regime
        self._regime_parents = []
        for dag in dags:
            instant_parents = {v: list(dag.G_0.predecessors(v)) for v in dag.topo_order}
            lagged_parents = self._compute_lagged_parents(dag)
            self._regime_parents.append(
                {
                    "instant": instant_parents,
                    "lagged": lagged_parents,
                    "topo": dag.topo_order,
                }
            )

    def _compute_lagged_parents(self, dag: TemporalDAG) -> dict[str, list[list[str]]]:
        """Compute lagged parents for a given DAG."""
        lagged_parents = {}
        node_to_idx = {v: i for i, v in enumerate(dag.topo_order)}

        for v in dag.topo_order:
            v_idx = node_to_idx[v]
            parents_per_lag = []

            for k in range(dag.K):
                G_k = dag.G_lags[k]
                parents_k = [
                    dag.topo_order[j] for j in range(len(dag.topo_order)) if G_k[j, v_idx] > 0
                ]
                parents_per_lag.append(parents_k)

            lagged_parents[v] = parents_per_lag

        return lagged_parents

    def _diverged(self, buffer: torch.Tensor) -> bool:
        """Decide whether a finished simulation counts as diverged.

        The check runs once, on the whole buffer, rather than stopping the
        simulation early as ``TemporalSCM`` does. ``TemporalSCM`` pre-draws all
        of its noise, so stopping early costs it nothing. Here noise and regime
        transitions are drawn step by step, so an early stop would shift every
        later draw from the episode generator and the global numpy RNG. The
        threshold would then change the randomness of all following episodes,
        not just this one.

        Parameters
        ----------
        buffer : torch.Tensor
            The full simulated buffer of shape ``(T + burn_in, N)``.

        Returns
        -------
        bool
            True for NaN, inf or values above 1e6, and, when
            ``divergence_threshold`` is set, for any value above it.
        """
        if check_divergence(buffer):
            return True
        return self.divergence_threshold is not None and bool(
            (buffer.abs() > self.divergence_threshold).any()
        )

    @torch.no_grad()
    def sample_observational(
        self,
        T: int,
        burn_in: int = 50,
        generator: torch.Generator | None = None,
        return_regimes: bool = False,
    ) -> torch.Tensor:
        """
        Sample observational data with regime switching.

        Parameters
        ----------
        return_regimes : bool
            If True, return (X, regimes) tuple. Otherwise just X.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, np.ndarray]
            Time series (T, N), or (time series, regimes) if return_regimes=True.
        """
        total_T = T + burn_in
        N = len(self.dags[0].topo_order)

        buffer = torch.zeros(total_T, N, device=self.device, dtype=self.dtype)
        regimes = np.zeros(total_T, dtype=np.int32)

        # Initialize regime
        regime = np.random.randint(0, self.num_regimes)

        for t in range(total_T):
            # Sample regime transition
            if t > 0:
                regime = np.random.choice(self.num_regimes, p=self.transition_matrix[regime])
            regimes[t] = regime

            # Get regime-specific structures
            parents = self._regime_parents[regime]
            self.dags[regime]
            mechs = self.mechanisms[regime]

            # Forward simulation
            for i, v in enumerate(parents["topo"]):
                instant_parents_v = parents["instant"][v]
                parent_values_instant = {
                    p: buffer[t, parents["topo"].index(p)] for p in instant_parents_v
                }

                parent_values_lagged = []
                for k, parents_k in enumerate(parents["lagged"][v]):
                    if t >= k + 1:
                        parent_values_k = {
                            p: buffer[t - k - 1, parents["topo"].index(p)] for p in parents_k
                        }
                    else:
                        parent_values_k = {}
                    parent_values_lagged.append(parent_values_k)

                eps = self.noise[v].sample(generator=generator)
                eps_tensor = torch.tensor([eps], device=self.device, dtype=self.dtype)

                value = mechs[v](parent_values_instant, parent_values_lagged, eps_tensor)
                value = clip_values(value)
                buffer[t, i] = value.item()

        if self._diverged(buffer):
            warnings.warn(
                "Regime-switching SCM diverged; returning zeros.", RuntimeWarning, stacklevel=2
            )
            if return_regimes:
                return torch.zeros(T, N, device=self.device, dtype=self.dtype), regimes[burn_in:]
            return torch.zeros(T, N, device=self.device, dtype=self.dtype)

        if return_regimes:
            return buffer[burn_in:], regimes[burn_in:]
        return buffer[burn_in:]

    @torch.no_grad()
    def sample_interventional(
        self,
        T: int,
        intervention: InterventionSpec,
        burn_in: int = 50,
        generator: torch.Generator | None = None,
        return_regimes: bool = False,
    ) -> torch.Tensor:
        """
        Sample interventional data with regime switching.

        Parameters
        ----------
        return_regimes : bool
            If True, return (X, regimes) tuple. Otherwise just X.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, np.ndarray]
            Time series (T, N), or (time series, regimes) if return_regimes=True.
        """
        total_T = T + burn_in
        N = len(self.dags[0].topo_order)

        buffer = torch.zeros(total_T, N, device=self.device, dtype=self.dtype)
        regimes = np.zeros(total_T, dtype=np.int32)

        regime = np.random.randint(0, self.num_regimes)

        for t in range(total_T):
            if t > 0:
                regime = np.random.choice(self.num_regimes, p=self.transition_matrix[regime])
            regimes[t] = regime

            parents = self._regime_parents[regime]
            self.dags[regime]
            mechs = self.mechanisms[regime]

            for i, v in enumerate(parents["topo"]):
                is_intervened = (i in intervention.targets) and (t - burn_in in intervention.times)

                if is_intervened:
                    if intervention.intervention_type == InterventionType.HARD:
                        buffer[t, i] = intervention.values
                    elif intervention.intervention_type == InterventionType.TIME_VARYING:
                        buffer[t, i] = intervention.values(t - burn_in)
                    else:  # SOFT
                        instant_parents_v = parents["instant"][v]
                        parent_values_instant = {
                            p: buffer[t, parents["topo"].index(p)] for p in instant_parents_v
                        }

                        parent_values_lagged = []
                        for k, parents_k in enumerate(parents["lagged"][v]):
                            if t >= k + 1:
                                parent_values_k = {
                                    p: buffer[t - k - 1, parents["topo"].index(p)]
                                    for p in parents_k
                                }
                            else:
                                parent_values_k = {}
                            parent_values_lagged.append(parent_values_k)

                        eps = self.noise[v].sample(generator=generator)
                        eps_tensor = torch.tensor([eps], device=self.device, dtype=self.dtype)

                        value = mechs[v](parent_values_instant, parent_values_lagged, eps_tensor)
                        value = value + intervention.values
                        value = clip_values(value)
                        buffer[t, i] = value.item()
                else:
                    instant_parents_v = parents["instant"][v]
                    parent_values_instant = {
                        p: buffer[t, parents["topo"].index(p)] for p in instant_parents_v
                    }

                    parent_values_lagged = []
                    for k, parents_k in enumerate(parents["lagged"][v]):
                        if t >= k + 1:
                            parent_values_k = {
                                p: buffer[t - k - 1, parents["topo"].index(p)] for p in parents_k
                            }
                        else:
                            parent_values_k = {}
                        parent_values_lagged.append(parent_values_k)

                    eps = self.noise[v].sample(generator=generator)
                    eps_tensor = torch.tensor([eps], device=self.device, dtype=self.dtype)

                    value = mechs[v](parent_values_instant, parent_values_lagged, eps_tensor)
                    value = clip_values(value)
                    buffer[t, i] = value.item()

        if self._diverged(buffer):
            warnings.warn(
                "Regime-switching SCM diverged; returning zeros.", RuntimeWarning, stacklevel=2
            )
            if return_regimes:
                return torch.zeros(T, N, device=self.device, dtype=self.dtype), regimes[burn_in:]
            return torch.zeros(T, N, device=self.device, dtype=self.dtype)

        if return_regimes:
            return buffer[burn_in:], regimes[burn_in:]
        return buffer[burn_in:]
