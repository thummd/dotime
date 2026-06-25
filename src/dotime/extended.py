"""Extended DoTime wrapper for Do-Over-Time-PFN.

Wraps DoTime.generate_pair() to produce model-ready dicts with:
- Padding to N_max=41 with variable masks
- Intervention time windows (start, end) instead of single time
- Intervention type encoding (0=hard, 1=soft, 2=time_varying)
- Query target/time sampling with downstream probability
"""

import contextlib

import numpy as np
import torch

from dotime.batched_tscm import BatchedTSCMSimulator
from dotime.interventions import InterventionSpec, InterventionType
from dotime.prior import DoTime
from dotime.tscm_sampler import TSCMSampler, TSCMStructure

# Map intervention types to integers
INTERVENTION_TYPE_MAP = {
    InterventionType.HARD: 0,
    InterventionType.SOFT: 1,
    InterventionType.TIME_VARYING: 2,
}


def pad_to_max_nodes(X: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Pad time series to have max_nodes variables."""
    T, N = X.shape
    if max_nodes > N:
        padding = torch.zeros(T, max_nodes - N, dtype=X.dtype, device=X.device)
        return torch.cat([X, padding], dim=1)
    return X[:, :max_nodes]


def _apply_hidden_mask(
    X_obs_padded: torch.Tensor,
    X_int_padded: torch.Tensor,
    variable_mask: torch.Tensor,
    hidden_canonical: list,
) -> None:
    """Mask out hidden variables in-place after canonical permutation.

    Applies three things, all of which must hold jointly for the hidden
    variable's trajectory to be invisible to the downstream model:

    1. ``variable_mask[h] = 0`` so the cross-variable mixer's key-padding
       mask treats the position as padding.
    2. ``X_obs_padded[:, h] = 0`` so the per-variable temporal encoder
       sees zero input for the hidden variable. Without this step the
       encoder still processes the hidden trajectory through its shared
       weights -- the final ``h * variable_mask`` zeros the *output* but
       lets gradients from the hidden channel flow back into the encoder
       parameters, which is a training-data leak.
    3. Same for ``X_int_padded``. This matters mainly for the Y_true /
       Y_causal_effect fields the caller extracts later; if the caller
       ever reads X_int at a hidden position it should get zero.

    ``hidden_canonical`` must be a list of *canonical-order* indices
    (i.e. already remapped through the canonical permutation if one was
    applied). All tensors are modified in place.
    """
    for h in hidden_canonical:
        X_obs_padded[:, h] = 0.0
        X_int_padded[:, h] = 0.0
        variable_mask[h] = 0.0


class TSCMPrior:
    """Drop-in replacement for DoTime that generates from a single TSCM structure.

    Has the same ``generate_pair(T)`` interface so ``ExtendedDoTime``
    can swap it in transparently.
    """

    def __init__(
        self,
        structure: TSCMStructure,
        burn_in: int = 50,
        seed: int = 42,
        use_lagged_edges: bool = True,
        intervention_scale: float = 2.0,
        sigma_w: float = 0.5,
    ):
        self.sampler = TSCMSampler(
            structure,
            max_lag=1,
            use_lagged_edges=use_lagged_edges,
            sigma_w=sigma_w,
            sigma_b=sigma_w * 0.5,
        )
        self.hidden_vars = self.sampler.get_hidden_vars()
        self.burn_in = burn_in
        self.intervention_scale = intervention_scale
        self.gen = torch.Generator().manual_seed(seed)
        self.config = {"burn_in": burn_in}

        # Canonical permutation: A at index 0, Y at index N-1, others in between.
        # The permutation maps topo-order indices -> canonical-order indices.
        # perm[canonical_idx] = topo_idx.
        self._a_idx_topo = self.sampler.get_intervention_target()
        self._y_idx_topo = self.sampler.get_outcome_var()
        dag = self.sampler._build_dag()
        N = len(dag.topo_order)
        middle = [i for i in range(N) if i != self._a_idx_topo and i != self._y_idx_topo]
        self.canonical_perm = [self._a_idx_topo, *middle, self._y_idx_topo]
        self.canonical_inv_perm = [0] * N
        for canon_idx, topo_idx in enumerate(self.canonical_perm):
            self.canonical_inv_perm[topo_idx] = canon_idx

    def get_outcome_var(self) -> int:
        """Canonical-order index of the outcome variable Y.

        After canonical reordering, Y is always at the last real column (N-1).
        Used by evaluation/prior_eval.py (collaborator interface) to pin the
        query target to the outcome.
        """
        return self.canonical_inv_perm[self._y_idx_topo]

    def get_intervention_target(self) -> int:
        """Canonical-order index of the treatment variable A (always 0)."""
        return self.canonical_inv_perm[self._a_idx_topo]

    def generate_pair(self, T: int):
        """Return (X_obs, X_int, intervention, scm) like DoTime.

        Data is returned in topo-order; the canonical reordering is applied
        at the ExtendedDoTime level after all re-simulation logic.
        """
        scm = self.sampler.sample(generator=self.gen)
        len(scm._topo)

        X_obs = scm.sample_observational(T=T, burn_in=self.burn_in, generator=self.gen)

        # Intervention target: always the treatment variable A (topo-order index).
        # This was previously `valid[0]` which picked the first non-hidden variable,
        # which for BD (topo=[X, A, Y]) incorrectly selected X.
        int_target = self._a_idx_topo

        # Intervention at a random time in [10, T-10], matching CTP's range
        t_lo = min(10, T - 1)
        t_hi = max(t_lo + 1, T - 10)
        int_time = int(torch.randint(t_lo, t_hi, (1,), generator=self.gen).item())
        int_value = float(torch.randn(1, generator=self.gen).item() * self.intervention_scale)

        intervention = InterventionSpec(
            targets=[int_target],
            times=[int_time],
            intervention_type=InterventionType.HARD,
            values=int_value,
        )

        X_int = scm.sample_interventional(
            T=T,
            intervention=intervention,
            burn_in=self.burn_in,
            generator=self.gen,
        )
        return X_obs, X_int, intervention, scm


class ExtendedDoTime:
    """CTP wrapper that produces model-ready dicts for Do-Over-Time-PFN."""

    def __init__(
        self,
        n_max: int = 41,
        n_min: int = 3,
        n_max_prior: int = 10,
        t_range: tuple = (50, 200),
        burn_in: int = 50,
        downstream_prob: float = 0.7,
        seed: int = 42,
        chain_prob: float = 0.15,
        regime_switching_prob: float = 0.15,
        intervention_source: str = "prior",
        tscm_structure: str | None = None,
        use_lagged_edges: bool = True,
        intervention_scale: float = 2.0,
        causal_mask_mode: str = "full",
        dynamics_burn_in: int = 0,
        sim_device: str = "cpu",
        query_offset_range: tuple = (0, 0),
        hardening: dict | None = None,
    ):
        self.n_max = n_max
        self.t_range = t_range
        self.downstream_prob = downstream_prob
        self.intervention_source = intervention_source
        self.causal_mask_mode = causal_mask_mode
        self.dynamics_burn_in = dynamics_burn_in
        self._sim_device = sim_device
        # query_offset_range: (lo, hi) — sample offset in [lo, hi] per query,
        # query_time_idx = int_time + offset (clamped to T-1). Fix 3 for FD.
        self.query_offset_range = tuple(query_offset_range)

        self.tscm_structure = tscm_structure
        self.intervention_scale = intervention_scale
        self._seed = seed
        self._burn_in_total = burn_in + dynamics_burn_in

        if tscm_structure is not None:
            structure_enum = TSCMStructure(tscm_structure)
            self.prior = TSCMPrior(
                structure_enum,
                burn_in=burn_in + dynamics_burn_in,
                seed=seed,
                use_lagged_edges=use_lagged_edges,
                intervention_scale=intervention_scale,
            )
            # Batched simulator for vectorized generation.
            # Hardening knobs (sigma_w, noise_std, max_lag, unit_norm_rows,
            # spectral_rho, bias_scale, add_self_memory_lags, positive_ar_diag)
            # can be supplied via `hardening`; missing keys keep legacy defaults.
            h = dict(hardening) if hardening else {}
            self.batched_sim = BatchedTSCMSimulator(
                structure_enum,
                max_lag=h.pop("max_lag", 1),
                use_lagged_edges=use_lagged_edges,
                sigma_w=h.pop("sigma_w", 0.5),
                noise_std=h.pop("noise_std", 0.3),
                unit_norm_rows=h.pop("unit_norm_rows", False),
                spectral_rho=h.pop("spectral_rho", None),
                bias_scale=h.pop("bias_scale", None),
                add_self_memory_lags=h.pop("add_self_memory_lags", False),
                positive_ar_diag=h.pop("positive_ar_diag", False),
            )
            if h:
                raise ValueError(f"Unknown hardening keys: {list(h.keys())}")
        else:
            config = {
                "N_max": n_max_prior,
                "burn_in": burn_in + dynamics_burn_in,
            }
            self.prior = DoTime(
                config=config,
                seed=seed,
                chain_prob=chain_prob,
                regime_switching_prob=regime_switching_prob,
            )
        self.rng = np.random.RandomState(seed)

    def sample_T(self) -> int:
        """Sample a time series length uniformly from t_range."""
        return self.rng.randint(self.t_range[0], self.t_range[1] + 1)

    def generate_sample(
        self,
        T: int | None = None,
        n_queries: int = 1,
        query_mode: str = "single",
    ) -> dict[str, torch.Tensor]:
        """Generate a single model-ready sample with one or more query points.

        Parameters
        ----------
        T : int, optional
            Time series length (sampled from t_range if None).
        n_queries : int
            Number of (query_target, query_time) pairs per trajectory.
            When > 1, query_target/query_time/Y_true/Y_causal_effect
            are tensors of shape (n_queries,) instead of scalars.

        Returns dict with:
            X_obs: (T, N_max) padded observational series
            X_int: (T, N_max) padded interventional series
            variable_mask: (N_max,) binary mask for real variables
            intervention_target: scalar int
            intervention_type: scalar int (0=hard, 1=soft, 2=time_varying)
            intervention_value: scalar float
            intervention_time_start: scalar float in [0, 1]
            intervention_time_end: scalar float in [0, 1]
            query_target: scalar int or (n_queries,) ints
            query_time: scalar float or (n_queries,) floats
            Y_true: scalar float or (n_queries,) floats
            Y_causal_effect: scalar float or (n_queries,) floats
            num_vars: scalar int
        """
        if T is None:
            T = self.sample_T()

        # Generate with divergence retry (up to 20 attempts for long trajectories)
        for _ in range(20):
            X_obs, X_int, intervention, scm = self.prior.generate_pair(T=T)
            if (
                not torch.isnan(X_obs).any()
                and not torch.isnan(X_int).any()
                and X_obs.abs().max() < 10
                and X_int.abs().max() < 10
            ):
                break

        N = X_obs.shape[1]

        # Causal masking: zero out X_obs at and after intervention onset.
        int_onset = min(intervention.times)
        intervention_target = intervention.targets[0] if intervention.targets else 0
        X_obs_masked = X_obs.clone()
        X_obs_masked[int_onset:] = 0.0

        if self.causal_mask_mode == "interpolation":
            # Restore the treatment variable at int_onset with its OBSERVATIONAL
            # value. The causal model additionally receives the intervention spec
            # (A_int = v) via the mixer; the obs-only model only sees A_obs here.
            X_obs_masked[int_onset, intervention_target] = X_obs[int_onset, intervention_target]

        # Pad to N_max
        X_obs_padded = pad_to_max_nodes(X_obs_masked, self.n_max)
        X_obs_full_padded = pad_to_max_nodes(X_obs, self.n_max)  # unmasked, for release
        X_int_padded = pad_to_max_nodes(X_int, self.n_max)

        # Variable mask.  Hidden-variable exclusion and hidden-position
        # zeroing happen AFTER the canonical permutation below so the
        # mask's indexing matches the permuted X_obs / X_int tensors.
        variable_mask = torch.zeros(self.n_max)
        variable_mask[:N] = 1.0

        # Intervention info (intervention_target already set above for masking)
        time_start = min(intervention.times)
        time_end = max(intervention.times)

        # Intervention value (scalar representation)
        if callable(intervention.values):
            mid_time = (time_start + time_end) // 2
            intervention_value = float(intervention.values(mid_time))
        else:
            intervention_value = float(intervention.values)

        # Re-simulate with an observed-scale intervention value if requested.
        # Sample a value from the pre-intervention history of the intervention
        # target, create a new InterventionSpec, and re-run the SCM so that
        # (intervention_value, X_int) stays consistent.
        #
        # Modes:
        #   "prior"             — keep the prior-sampled value (no re-simulation).
        #   "positivity_aware"  — clip prior value to [obs_mean - 3σ, obs_mean + 3σ]
        #                         and re-simulate. Preserves prior shape, enforces
        #                         positivity (intervention within observed support).
        #   "observed_discrete" — pick a random past value (measure-zero for
        #                         continuous variables, kept for backward compat).
        #   "observed_normal"   — sample from N(mean(pre_int), std(pre_int)).
        #   "observed_uniform"  — sample from U[min(pre_int), max(pre_int)].
        #
        # "observed" is accepted as a legacy alias for "observed_discrete".
        mode = self.intervention_source
        if mode == "observed":
            mode = "observed_discrete"

        if mode == "positivity_aware":
            pre_int = X_obs[:int_onset, intervention_target]
            if pre_int.numel() > 1 and float(pre_int.std().item()) > 1e-4:
                mu = float(pre_int.mean().item())
                sigma = float(pre_int.std().item())
                clipped = float(np.clip(intervention_value, mu - 3 * sigma, mu + 3 * sigma))
                if clipped != intervention_value:
                    new_intervention = InterventionSpec(
                        targets=intervention.targets,
                        times=intervention.times,
                        intervention_type=InterventionType.HARD,
                        values=clipped,
                    )
                    X_int_new = scm.sample_interventional(
                        T=T,
                        intervention=new_intervention,
                        burn_in=self.prior.config.get("burn_in", 50),
                    )
                    if not torch.isnan(X_int_new).any() and X_int_new.abs().max() < 10:
                        X_int = X_int_new
                        X_int_padded = pad_to_max_nodes(X_int, self.n_max)
                        intervention = new_intervention
                        intervention_value = clipped

        elif mode in ("observed_discrete", "observed_normal", "observed_uniform"):
            pre_int = X_obs[:int_onset, intervention_target]
            if pre_int.numel() > 0 and float(pre_int.std().item()) > 1e-4:
                pre_np = pre_int.detach().cpu().numpy()
                if mode == "observed_discrete":
                    obs_value = float(pre_np[self.rng.randint(len(pre_np))])
                elif mode == "observed_normal":
                    mu = float(pre_np.mean())
                    sigma = float(pre_np.std(ddof=1)) if len(pre_np) > 1 else 0.0
                    obs_value = float(self.rng.randn() * max(sigma, 1e-4) + mu)
                else:  # observed_uniform
                    lo = float(pre_np.min())
                    hi = float(pre_np.max())
                    obs_value = float(self.rng.uniform(lo, hi)) if hi > lo else lo

                new_intervention = InterventionSpec(
                    targets=intervention.targets,
                    times=intervention.times,
                    intervention_type=InterventionType.HARD,
                    values=obs_value,
                )
                X_int_new = scm.sample_interventional(
                    T=T,
                    intervention=new_intervention,
                    burn_in=self.prior.config.get("burn_in", 50),
                )
                if not torch.isnan(X_int_new).any() and X_int_new.abs().max() < 10:
                    X_int = X_int_new
                    X_int_padded = pad_to_max_nodes(X_int, self.n_max)
                    intervention = new_intervention
                    intervention_value = obs_value

        intervention_type = INTERVENTION_TYPE_MAP[intervention.intervention_type]

        # Positivity score: how OOD is intervention_value relative to observed support?
        pre_int = X_obs[:int_onset, intervention_target]
        if pre_int.numel() > 1 and float(pre_int.std().item()) > 1e-4:
            obs_mu = float(pre_int.mean().item())
            obs_sigma = float(pre_int.std().item())
            positivity_score = max(0.0, abs(intervention_value - obs_mu) / obs_sigma - 3.0)
        else:
            positivity_score = 0.0

        # Fix 2a: normalize intervention_value by observed std for scale-consistency
        # with positional/timing features in the mixer. Store raw value too.
        intervention_value_raw = intervention_value
        if pre_int.numel() > 1 and float(pre_int.std().item()) > 1e-4:
            intervention_value_norm = intervention_value / max(float(pre_int.std().item()), 1e-4)
        else:
            intervention_value_norm = intervention_value

        # Fix 2c: canonical column reordering (TSCMPrior only). Put treatment A at
        # column 0, outcome Y at column N-1, remaining (covariates/hidden) in between.
        # Remap intervention_target and the padded tensors.
        canonical_inv_perm = getattr(self.prior, "canonical_inv_perm", None)
        if canonical_inv_perm is not None:
            perm = self.prior.canonical_perm  # canonical_idx -> topo_idx
            full_perm = list(perm) + list(range(N, self.n_max))
            perm_t = torch.tensor(full_perm, dtype=torch.long)
            X_obs_padded = X_obs_padded.index_select(dim=1, index=perm_t)
            X_int_padded = X_int_padded.index_select(dim=1, index=perm_t)
            intervention_target = canonical_inv_perm[intervention_target]
            hidden_vars_topo = getattr(self.prior, "hidden_vars", [])
            hidden_canonical = [canonical_inv_perm[h] for h in hidden_vars_topo]
        else:
            hidden_canonical = list(getattr(self.prior, "hidden_vars", []))

        # Hide unobserved variables from the model (variable_mask + X_obs/X_int
        # zeroing). Must happen AFTER canonical permutation so hidden_canonical
        # refers to the same columns as the permuted trajectory tensors.
        _apply_hidden_mask(X_obs_padded, X_int_padded, variable_mask, hidden_canonical)

        # Query sampling — aligned with identifiability theory:
        # P(Y_{t+offset} | do(A_t), H_{t-1},...,H_{t-K})
        # Fix 3: sample per-query offset from query_offset_range so the mediator
        # (or any propagation chain) has time to reflect the intervention.
        other_vars = [v for v in range(N) if v != intervention_target]
        int_time = min(int(np.mean(intervention.times)), T - 1)
        offset_lo, offset_hi = self.query_offset_range

        # If the prior exposes a canonical outcome (TSCMPrior), restrict queries
        # to that variable so we evaluate p(Y | do(A)) — not confounders/mediators.
        # For full CTP prior (no fixed structure), fall back to all non-intervention.
        fixed_outcome = (
            self.prior.get_outcome_var() if hasattr(self.prior, "get_outcome_var") else None
        )

        def _sample_qtime():
            if offset_hi <= offset_lo:
                return min(int_time + offset_lo, T - 1)
            off = int(self.rng.randint(offset_lo, offset_hi + 1))
            return min(int_time + off, T - 1)

        query_targets = []
        query_time_idxs = []

        if query_mode == "all_pairs" and other_vars:
            # Query the canonical outcome (or all non-intervention vars if no outcome).
            candidates = [fixed_outcome] if fixed_outcome is not None else other_vars
            for qt in candidates:
                query_targets.append(qt)
                query_time_idxs.append(_sample_qtime())
        else:
            # Single mode: canonical outcome if available, else random.
            for _ in range(n_queries):
                if fixed_outcome is not None:
                    qt = fixed_outcome
                elif other_vars:
                    qt = int(self.rng.choice(other_vars))
                else:
                    qt = intervention_target
                query_targets.append(qt)
                query_time_idxs.append(_sample_qtime())

        # Ground truth: raw interventional value and causal effect
        y_trues = [
            float(X_int_padded[qti, qt].item())
            for qt, qti in zip(query_targets, query_time_idxs, strict=False)
        ]
        y_obs_vals = [
            float(X_obs_padded[qti, qt].item())
            for qt, qti in zip(query_targets, query_time_idxs, strict=False)
        ]
        y_effects = [yi - yo for yi, yo in zip(y_trues, y_obs_vals, strict=False)]

        # Flatten to scalars if single query (backwards compatible)
        actual_n_queries = len(query_targets)
        if actual_n_queries == 1:
            query_target_t = torch.tensor(query_targets[0], dtype=torch.long)
            query_time_t = torch.tensor(query_time_idxs[0] / T, dtype=torch.float32)
            y_true_t = torch.tensor(y_trues[0], dtype=torch.float32)
            y_obs_t = torch.tensor(y_obs_vals[0], dtype=torch.float32)
            y_effect_t = torch.tensor(y_effects[0], dtype=torch.float32)
        else:
            query_target_t = torch.tensor(query_targets, dtype=torch.long)
            query_time_t = torch.tensor([qti / T for qti in query_time_idxs], dtype=torch.float32)
            y_true_t = torch.tensor(y_trues, dtype=torch.float32)
            y_obs_t = torch.tensor(y_obs_vals, dtype=torch.float32)
            y_effect_t = torch.tensor(y_effects, dtype=torch.float32)

        return {
            "X_obs": X_obs_padded,  # (T, N_max) causally masked (model input)
            "X_obs_full": X_obs_full_padded,  # (T, N_max) unmasked (released data)
            "X_int": X_int_padded,  # (T, N_max)
            "variable_mask": variable_mask,  # (N_max,)
            "int_onset_idx": torch.tensor(int_onset, dtype=torch.long),
            "intervention_target": torch.tensor(intervention_target, dtype=torch.long),
            "intervention_type": torch.tensor(intervention_type, dtype=torch.long),
            "intervention_value": torch.tensor(intervention_value_norm, dtype=torch.float32),
            "intervention_value_raw": torch.tensor(intervention_value_raw, dtype=torch.float32),
            "intervention_time_start": torch.tensor(time_start / T, dtype=torch.float32),
            "intervention_time_end": torch.tensor(time_end / T, dtype=torch.float32),
            "positivity_score": torch.tensor(positivity_score, dtype=torch.float32),
            "query_target": query_target_t,
            "query_time": query_time_t,
            "Y_true": y_true_t,
            "Y_obs": y_obs_t,
            "Y_causal_effect": y_effect_t,
            "num_vars": torch.tensor(N, dtype=torch.long),
        }

    def generate_batch(
        self,
        batch_size: int,
        T: int | None = None,
        n_queries: int = 1,
        num_workers: int = 0,
        query_mode: str = "single",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Generate a batch of model-ready samples.

        All samples in a batch share the same T (sampled once if not provided).
        Multi-query batches include a '_traj_idx' field that maps each query
        to its source trajectory (for encoder caching).

        Parameters
        ----------
        query_mode : "single" (random queries) or "all_pairs" (all outcome vars)

        Returns dict with:
            X_obs, variable_mask: (B, ...) unique trajectories
            intervention_*, query_*, Y_*: (B_total,) per-query (B_total = sum of queries)
            _traj_idx: (B_total,) index into trajectory dimension
        """
        if T is None:
            T = self.sample_T()

        # Use batched vectorized simulation for TSCM structures (much faster)
        if self.tscm_structure is not None and hasattr(self, "batched_sim"):
            return self._generate_batch_vectorized(batch_size, T, n_queries, query_mode)

        if num_workers > 0:
            samples = self._generate_parallel(batch_size, T, n_queries, num_workers, query_mode)
        else:
            samples = [
                self.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)
                for _ in range(batch_size)
            ]

        return self._collate_batch(samples)

    def _generate_batch_vectorized(self, batch_size, T, n_queries, query_mode):
        """Generate a batch using the batched vectorized SCM simulator.

        Much faster than per-sample generation for fixed TSCM structures.
        Produces the same output format as _collate_batch().
        """
        sim = self.batched_sim
        seed = int(self.rng.randint(0, 2**31))

        # Fix 1: positivity_aware in batched path = per-sample 3σ clip
        positivity_clip = self.intervention_source == "positivity_aware"

        pairs = sim.generate_pairs(
            B=batch_size,
            T=T,
            burn_in=self._burn_in_total,
            device=self._sim_device,
            intervention_scale=self.intervention_scale,
            seed=seed,
            positivity_clip=positivity_clip,
        )

        # Move bulk simulation results to CPU once for the per-sample dict
        # construction below (Python loops + .item() calls are CPU-bound).
        X_obs_all = pairs["X_obs"].cpu()  # (B, T, N_raw)
        X_int_all = pairs["X_int"].cpu()  # (B, T, N_raw)
        valid = pairs["valid"].cpu()  # (B,)
        int_target = int(pairs["int_target"][0].item())
        int_time_idx = int(pairs["int_time"][0].item())
        int_values_per_sample = pairs["int_value"].cpu()  # (B,) per-sample
        N = pairs["N"]
        hidden_vars = pairs["hidden_vars"]

        # Build per-sample dicts and use the existing _collate_batch
        samples = []
        for b in range(batch_size):
            if not valid[b]:
                # Skip diverged — generate a fallback via sequential path
                s = self.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)
                samples.append(s)
                continue

            X_obs = X_obs_all[b]  # (T, N_raw) in topo order
            X_int = X_int_all[b]

            # Causal masking (topo order)
            X_obs_masked = X_obs.clone()
            X_obs_masked[int_time_idx:] = 0.0
            if self.causal_mask_mode == "interpolation":
                X_obs_masked[int_time_idx, int_target] = X_obs[int_time_idx, int_target]

            # Pad to n_max (topo order in first N cols)
            X_obs_padded = pad_to_max_nodes(X_obs_masked, self.n_max)
            X_int_padded = pad_to_max_nodes(X_int, self.n_max)

            # Variable mask -- hidden-variable exclusion and X_obs/X_int
            # zeroing happen AFTER canonical permutation below so the mask
            # indexing aligns with the permuted trajectory tensors.
            variable_mask = torch.zeros(self.n_max)
            variable_mask[:N] = 1.0

            # Fix 2a: positivity score + normalize intervention_value (topo indices)
            # Use this sample's actual int_value (not the batch's)
            pre_int = X_obs[:int_time_idx, int_target]
            int_value_raw = float(int_values_per_sample[b].item())
            if pre_int.numel() > 1 and float(pre_int.std().item()) > 1e-4:
                obs_mu = float(pre_int.mean().item())
                obs_sigma = float(pre_int.std().item())
                positivity_score = max(
                    0.0, abs(int_value_raw - obs_mu) / max(obs_sigma, 1e-4) - 3.0
                )
                int_value_norm = int_value_raw / max(obs_sigma, 1e-4)
            else:
                positivity_score = 0.0
                int_value_norm = int_value_raw

            # Fix 2c: canonical column reordering on padded tensors (topo -> canonical)
            canonical_inv_perm = getattr(self.prior, "canonical_inv_perm", None)
            int_target_out = int_target
            if canonical_inv_perm is not None:
                perm = self.prior.canonical_perm  # canonical_idx -> topo_idx
                full_perm = list(perm) + list(range(N, self.n_max))
                perm_t = torch.tensor(full_perm, dtype=torch.long)
                X_obs_padded = X_obs_padded.index_select(dim=1, index=perm_t)
                X_int_padded = X_int_padded.index_select(dim=1, index=perm_t)
                int_target_out = canonical_inv_perm[int_target]
                # Remap hidden_vars to canonical indices for query filtering
                hidden_canonical = [canonical_inv_perm[h] for h in hidden_vars]
            else:
                hidden_canonical = list(hidden_vars)

            # Hide unobserved variables (variable_mask + X_obs/X_int zeroing).
            # Must happen AFTER canonical permutation so hidden_canonical
            # refers to the same columns as the permuted trajectory tensors.
            _apply_hidden_mask(
                X_obs_padded,
                X_int_padded,
                variable_mask,
                hidden_canonical,
            )

            # Query targets. If the prior has a canonical outcome (TSCMPrior), pin
            # queries to Y. Otherwise fall back to all non-hidden non-intervention.
            fixed_outcome = (
                self.prior.get_outcome_var() if hasattr(self.prior, "get_outcome_var") else None
            )
            other_vars = [v for v in range(N) if v != int_target_out and v not in hidden_canonical]
            if query_mode == "all_pairs" and other_vars:
                candidates = [fixed_outcome] if fixed_outcome is not None else other_vars
                query_targets = list(candidates)
            else:
                if fixed_outcome is not None:
                    query_targets = [fixed_outcome] * max(1, n_queries)
                elif other_vars:
                    query_targets = [int(self.rng.choice(other_vars))] * max(1, n_queries)
                else:
                    query_targets = [int_target_out] * max(1, n_queries)

            # Fix 3: per-query offset from query_offset_range
            offset_lo, offset_hi = self.query_offset_range
            if offset_hi <= offset_lo:
                query_time_idxs = [min(int_time_idx + offset_lo, T - 1)] * len(query_targets)
            else:
                query_time_idxs = [
                    min(int_time_idx + int(self.rng.randint(offset_lo, offset_hi + 1)), T - 1)
                    for _ in query_targets
                ]

            # Ground truth (canonical-indexed padded tensors, per-query time)
            y_trues = [
                float(X_int_padded[qti, qt].item())
                for qt, qti in zip(query_targets, query_time_idxs, strict=False)
            ]
            y_obs_vals = [
                float(X_obs_padded[qti, qt].item())
                for qt, qti in zip(query_targets, query_time_idxs, strict=False)
            ]
            y_effects = [yi - yo for yi, yo in zip(y_trues, y_obs_vals, strict=False)]

            actual_nq = len(query_targets)
            if actual_nq == 1:
                qt_t = torch.tensor(query_targets[0], dtype=torch.long)
                qtime_t = torch.tensor(query_time_idxs[0] / T, dtype=torch.float32)
                yt_t = torch.tensor(y_trues[0], dtype=torch.float32)
                yo_t = torch.tensor(y_obs_vals[0], dtype=torch.float32)
                ye_t = torch.tensor(y_effects[0], dtype=torch.float32)
            else:
                qt_t = torch.tensor(query_targets, dtype=torch.long)
                qtime_t = torch.tensor([q / T for q in query_time_idxs], dtype=torch.float32)
                yt_t = torch.tensor(y_trues, dtype=torch.float32)
                yo_t = torch.tensor(y_obs_vals, dtype=torch.float32)
                ye_t = torch.tensor(y_effects, dtype=torch.float32)

            samples.append(
                {
                    "X_obs": X_obs_padded,
                    "X_int": X_int_padded,
                    "variable_mask": variable_mask,
                    "intervention_target": torch.tensor(int_target_out, dtype=torch.long),
                    "intervention_type": torch.tensor(0, dtype=torch.long),  # HARD
                    "intervention_value": torch.tensor(int_value_norm, dtype=torch.float32),
                    "intervention_value_raw": torch.tensor(int_value_raw, dtype=torch.float32),
                    "intervention_time_start": torch.tensor(int_time_idx / T, dtype=torch.float32),
                    "intervention_time_end": torch.tensor(int_time_idx / T, dtype=torch.float32),
                    "query_target": qt_t,
                    "query_time": qtime_t,
                    "Y_true": yt_t,
                    "Y_obs": yo_t,
                    "Y_causal_effect": ye_t,
                    "num_vars": torch.tensor(N, dtype=torch.long),
                    "int_onset_idx": torch.tensor(int_time_idx, dtype=torch.long),
                    "positivity_score": torch.tensor(positivity_score, dtype=torch.float32),
                }
            )

        return self._collate_batch(samples)

    def _get_worker_pool(self, n_procs: int):
        """Lazily create (or resize) a persistent ProcessPoolExecutor.

        Spinning up a pool per batch wastes fork+init overhead on every step;
        keeping one around across calls lets workers amortise that cost. We
        rebuild only when the requested worker count changes.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        pool = getattr(self, "_worker_pool", None)
        pool_size = getattr(self, "_worker_pool_size", 0)
        if pool is not None and pool_size == n_procs:
            return pool
        if pool is not None:
            with contextlib.suppress(Exception):
                pool.shutdown(wait=False, cancel_futures=True)
        ctx = mp.get_context("fork")
        self._worker_pool = ProcessPoolExecutor(
            max_workers=n_procs,
            mp_context=ctx,
            initializer=_worker_pool_init,
            initargs=(self,),
        )
        self._worker_pool_size = n_procs
        return self._worker_pool

    def close_worker_pool(self):
        """Shut down the persistent pool (called on GC or explicitly)."""
        pool = getattr(self, "_worker_pool", None)
        if pool is not None:
            with contextlib.suppress(Exception):
                pool.shutdown(wait=False, cancel_futures=True)
            self._worker_pool = None
            self._worker_pool_size = 0

    def __del__(self):
        self.close_worker_pool()

    def _generate_parallel(self, batch_size, T, n_queries, num_workers, query_mode):
        """Generate samples in parallel using a persistent process pool."""
        n_procs = min(num_workers, batch_size)
        pool = self._get_worker_pool(n_procs)
        base_seed = int(self.rng.randint(0, 2**31))
        args = [(T, n_queries, query_mode, base_seed + i) for i in range(batch_size)]
        # map() returns an iterator; materialise to a list before the caller
        # iterates so any worker exception surfaces here, not downstream.
        samples = list(pool.map(_generate_sample_worker_persistent, args))
        return samples

    @staticmethod
    def _collate_batch(samples):
        """Collate samples into a batch with _traj_idx for encoder caching.

        Trajectory-level fields (X_obs, variable_mask) are stacked to (B, ...).
        Query-level fields are concatenated to (B_total,) with _traj_idx mapping
        each query back to its trajectory.
        """
        query_keys = {"query_target", "query_time", "Y_true", "Y_obs", "Y_causal_effect"}
        # Check if any sample has multi-query (tensor with dim > 0 for query fields)
        is_multi = any(s["query_target"].dim() > 0 for s in samples)

        if not is_multi:
            # All scalar queries — simple stack, no _traj_idx needed
            return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}

        # Multi-query: build _traj_idx and separate trajectory vs query fields
        batch = {}
        traj_indices = []
        intervention_keys = {
            "intervention_target",
            "intervention_type",
            "intervention_value",
            "intervention_time_start",
            "intervention_time_end",
        }

        # Stack trajectory-level fields (unique per trajectory)
        traj_keys = {k for k in samples[0] if k not in query_keys and k not in intervention_keys}
        for key in traj_keys:
            batch[key] = torch.stack([s[key] for s in samples])

        # Build _traj_idx and concatenate query + intervention fields
        for i, s in enumerate(samples):
            nq = s["query_target"].numel()
            traj_indices.append(torch.full((nq,), i, dtype=torch.long))
        batch["_traj_idx"] = torch.cat(traj_indices)

        for key in query_keys:
            parts = []
            for s in samples:
                v = s[key]
                parts.append(v.unsqueeze(0) if v.dim() == 0 else v)
            batch[key] = torch.cat(parts)

        # Intervention fields: repeat per query count
        for key in intervention_keys:
            parts = []
            for s in samples:
                nq = s["query_target"].numel()
                v = s[key]
                parts.append(v.unsqueeze(0).expand(nq) if v.dim() == 0 else v)
            batch[key] = torch.cat(parts)

        return batch


def _generate_sample_worker(args):
    """Top-level function for multiprocessing (must be picklable).

    Each worker reseeds its RNG to avoid correlated samples across processes.
    """
    prior, T, n_queries, query_mode, worker_seed = args
    # Reseed per-worker to avoid sharing RNG state across forked processes
    prior.rng = np.random.RandomState(worker_seed)
    if hasattr(prior.prior, "gen"):
        prior.prior.gen = torch.Generator().manual_seed(worker_seed)
    return prior.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)


# --- Persistent-pool variant -------------------------------------------------
# The parent passes the prior once via the initializer; the worker keeps it
# as a module global. Subsequent generate_batch() calls then only pickle a
# tiny (T, n_queries, query_mode, seed) tuple per sample instead of the whole
# prior. set_sharing_strategy('file_system') avoids the "received 0 items of
# ancdata" crash that hits when many tensors cross the fd-passing socket.
_WORKER_PRIOR = None


def _worker_pool_init(prior):
    """Runs once per worker when the persistent pool starts."""
    global _WORKER_PRIOR
    _WORKER_PRIOR = prior
    try:
        import torch.multiprocessing as tmp

        tmp.set_sharing_strategy("file_system")
    except Exception:
        pass


def _generate_sample_worker_persistent(args):
    """Worker entrypoint that uses the prior cached in the worker globals."""
    T, n_queries, query_mode, worker_seed = args
    prior = _WORKER_PRIOR
    prior.rng = np.random.RandomState(worker_seed)
    if hasattr(prior.prior, "gen"):
        prior.prior.gen = torch.Generator().manual_seed(worker_seed)
    return prior.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)
