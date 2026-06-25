"""Temporal SCM with time-stepped forward simulation."""

import warnings

import torch

from dotime._sampling import DistributionSampler
from dotime.interventions import InterventionSpec, InterventionType
from dotime.temporal_graph import TemporalDAG
from dotime.temporal_mechanism import TemporalMechanism
from dotime.utils import check_divergence, clip_values


class TemporalSCM:
    """
    Temporal Structural Causal Model with time-stepped forward simulation.

    Extends Do-PFN's SCM to support temporal dependencies with lags.
    """

    def __init__(
        self,
        dag: TemporalDAG,
        mechanisms: dict[str, TemporalMechanism],
        noise: dict[str, DistributionSampler],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        dag : TemporalDAG
            Temporal DAG with instantaneous and lagged edges.
        mechanisms : Dict[str, TemporalMechanism]
            Mechanisms for each variable.
        noise : Dict[str, DistributionSampler]
            Noise distributions for each variable.
        device : torch.device
            Device for computation.
        dtype : torch.dtype
            Data type for computation.
        """
        self.dag = dag
        self.mechanisms = mechanisms
        self.noise = noise
        self.device = device
        self.dtype = dtype

        # Store topology
        self._topo = dag.topo_order
        self._G_0 = dag.G_0
        self._G_lags = dag.G_lags
        self._K = dag.K

        # Pre-compute name-to-index mapping (O(1) lookups instead of O(N) list scans)
        self._topo_idx = {v: i for i, v in enumerate(self._topo)}

        # Parents for each variable (by name, for mechanism compatibility)
        self._instant_parents = {v: list(self._G_0.predecessors(v)) for v in self._topo}
        self._lagged_parents = self._compute_lagged_parents()

        # Pre-resolved parent indices (avoid dict comprehensions in hot loop)
        self._instant_parent_idx = {
            i: [self._topo_idx[p] for p in self._instant_parents[v]]
            for i, v in enumerate(self._topo)
        }
        self._lagged_parent_idx = {
            i: [[self._topo_idx[p] for p in parents_k] for parents_k in self._lagged_parents[v]]
            for i, v in enumerate(self._topo)
        }
        # Pre-resolved parent names paired with indices (for mechanism weight lookup)
        self._instant_parent_pairs = {
            i: [(p, self._topo_idx[p]) for p in self._instant_parents[v]]
            for i, v in enumerate(self._topo)
        }
        self._lagged_parent_pairs = {
            i: [
                [(p, self._topo_idx[p]) for p in parents_k] for parents_k in self._lagged_parents[v]
            ]
            for i, v in enumerate(self._topo)
        }

    def _compute_lagged_parents(self) -> dict[str, list[list[str]]]:
        """Compute lagged parents for each variable."""
        lagged_parents = {}

        for v in self._topo:
            v_idx = self._topo_idx[v]
            parents_per_lag = []

            for k in range(self._K):
                G_k = self._G_lags[k]
                # Find parents at lag k+1
                parents_k = [self._topo[j] for j in range(len(self._topo)) if G_k[j, v_idx] > 0]
                parents_per_lag.append(parents_k)

            lagged_parents[v] = parents_per_lag

        return lagged_parents

    @torch.no_grad()
    def _simulate(
        self,
        total_T: int,
        burn_in: int = 50,
        intervention: InterventionSpec | None = None,
        generator: torch.Generator | None = None,
        divergence_check_interval: int = 50,
    ) -> torch.Tensor | None:
        """Unified forward simulation, optionally with intervention.

        Returns buffer[burn_in:] on success, or None on early divergence.
        """
        N = len(self._topo)
        total_T - burn_in
        buffer = torch.zeros(total_T, N, device=self.device, dtype=self.dtype)

        # Pre-sample all noise (eliminates per-step tensor creation + RNG state swaps)
        all_noise = {}
        for v in self._topo:
            all_noise[v] = (
                self.noise[v]
                .distribution.sample((total_T,))
                .to(device=self.device, dtype=self.dtype)
            )

        # Pre-compute intervention lookup set for O(1) checks
        int_targets = set()
        int_times = set()
        int_type = None
        int_values = None
        if intervention is not None:
            int_targets = set(intervention.targets)
            int_times = set(intervention.times)
            int_type = intervention.intervention_type
            int_values = intervention.values

        # Forward simulation
        for t in range(total_T):
            # Early divergence detection
            if (
                divergence_check_interval > 0
                and t > 0
                and t % divergence_check_interval == 0
                and buffer[t - 1].abs().max() > 500
            ):
                return None

            for i, v in enumerate(self._topo):
                # Check intervention
                if intervention is not None and i in int_targets and (t - burn_in) in int_times:
                    if int_type == InterventionType.HARD:
                        buffer[t, i] = int_values
                        continue
                    elif int_type == InterventionType.TIME_VARYING:
                        buffer[t, i] = int_values(t - burn_in)
                        continue
                    # SOFT: fall through to mechanism, add shift after

                # Gather instantaneous parent values using pre-resolved indices
                parent_values_instant = {
                    p: buffer[t, idx] for p, idx in self._instant_parent_pairs[i]
                }

                # Gather lagged parent values
                parent_values_lagged = []
                for k, pairs_k in enumerate(self._lagged_parent_pairs[i]):
                    if t >= k + 1:
                        parent_values_lagged.append(
                            {p: buffer[t - k - 1, idx] for p, idx in pairs_k}
                        )
                    else:
                        parent_values_lagged.append({})

                # Noise (pre-sampled, already a tensor)
                eps = all_noise[v][t].unsqueeze(0)

                # Apply mechanism
                value = self.mechanisms[v](parent_values_instant, parent_values_lagged, eps)

                # Soft intervention shift
                if (
                    intervention is not None
                    and int_type == InterventionType.SOFT
                    and i in int_targets
                    and (t - burn_in) in int_times
                ):
                    value = value + int_values

                buffer[t, i] = clip_values(value)

        # Final divergence check
        if check_divergence(buffer):
            return None

        return buffer[burn_in:]

    @torch.no_grad()
    def sample_observational(
        self,
        T: int,
        burn_in: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample observational data from the temporal SCM.

        Parameters
        ----------
        T : int
            Length of time series to generate (after burn-in).
        burn_in : int
            Number of burn-in steps to discard.
        generator : torch.Generator, optional
            RNG for reproducibility.

        Returns
        -------
        torch.Tensor
            Time series data of shape (T, N) where N is number of variables.
        """
        result = self._simulate(T + burn_in, burn_in=burn_in, generator=generator)
        if result is None:
            N = len(self._topo)
            warnings.warn(
                "SCM diverged during simulation; returning zeros.", RuntimeWarning, stacklevel=2
            )
            return torch.zeros(T, N, device=self.device, dtype=self.dtype)
        return result

    @torch.no_grad()
    def sample_interventional(
        self,
        T: int,
        intervention: InterventionSpec,
        burn_in: int = 50,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Sample interventional data from the temporal SCM.

        Parameters
        ----------
        T : int
            Length of time series to generate (after burn-in).
        intervention : InterventionSpec
            Intervention specification.
        burn_in : int
            Number of burn-in steps to discard.
        generator : torch.Generator, optional
            RNG for reproducibility.

        Returns
        -------
        torch.Tensor
            Time series data of shape (T, N) under intervention.
        """
        result = self._simulate(
            T + burn_in,
            burn_in=burn_in,
            intervention=intervention,
            generator=generator,
        )
        if result is None:
            N = len(self._topo)
            warnings.warn(
                "SCM diverged during interventional simulation; returning zeros.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.zeros(T, N, device=self.device, dtype=self.dtype)
        return result
