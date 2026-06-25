"""Model-ready batch generator for continuous-time temporal causal PFNs.

This is the continuous-time counterpart to
:class:`dotime.prior.extended_prior.ExtendedCausalTime`.  It wraps a
:class:`ContinuousTSCMSampler`, draws an observation schedule from one
of the variable-Delta-t families in :mod:`time_schedule`, and produces a
model-ready dict for every call.

Contract with the rest of the pipeline
--------------------------------------
The returned dict matches the discrete-time contract used by
:mod:`dotime.data.temporal_dataloader` and :mod:`dotime.model`, plus two
new fields:

- ``times`` : ``(T,)`` float tensor of absolute observation times.
- ``dts`` : ``(T - 1,)`` float tensor with ``dts[i] = times[i+1] -
  times[i]``.  Useful for encoder variants that consume log-Delta-t
  features separately.

Two existing fields change their semantics in the continuous setting:

- ``int_onset_idx`` : integer index into ``times`` at which the
  intervention starts.  Matches the discrete-time field exactly.
- ``intervention_time_start`` / ``intervention_time_end`` : stay
  normalised to ``[0, 1]`` over the observation window
  ``[times[0], times[-1]]`` so the existing mixer head is unchanged.

Additionally, we expose ``t_int_start``, ``t_int_end``, and ``t_query``
in absolute time units for encoders that want them.

Counterfactual vs interventional pairs
--------------------------------------
``pair_mode`` selects between
:meth:`ContinuousSCM.sample_counterfactual_pair` (shared noise; Pearl
rung 3) and :meth:`ContinuousSCM.sample_interventional_pair`
(independent noise; rung 2 — matches the discrete-time DoT-PFN default).
Counterfactual pairs are the natural training signal for the workshop
paper; interventional pairs are kept for regression testing against
DoT-PFN behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from causaltime.tscm_sampler import TSCMStructure


@dataclass
class _SampledSCMContext:
    """Everything :meth:`generate_sample` needs to know about the sampled SCM.

    Collecting these in one object lets :class:`ContinuousExtendedPrior`
    hand off topology-dependent decisions to a subclass hook
    (:meth:`_sample_scm_context`).  That keeps the random-graph variant
    from having to duplicate all of ``generate_sample``.
    """

    scm: object  # ContinuousSCM (forward-ref to avoid circular import)
    n_vars: int
    intervention_target_topo: int
    outcome_var_topo: int
    hidden_vars_topo: list[int] = field(default_factory=list)

    @property
    def canonical_perm(self) -> list[int]:
        """``A -> 0, Y -> N-1, others in between (topological order preserved)``."""
        middle = [
            i
            for i in range(self.n_vars)
            if i != self.intervention_target_topo and i != self.outcome_var_topo
        ]
        return [self.intervention_target_topo, *middle, self.outcome_var_topo]

    @property
    def topo_to_canon(self) -> list[int]:
        """Inverse permutation: ``canon_idx[topo_idx] = canonical index of that topo index``."""
        inv = [0] * self.n_vars
        for canon_idx, topo_idx in enumerate(self.canonical_perm):
            inv[topo_idx] = canon_idx
        return inv


# -------------------------------------------------------- time-varying profiles


@dataclass
class _StepProfile:
    """Step intervention: ``c(t) = lo`` for ``t < t_mid`` else ``hi``.

    Kept at module level so instances are picklable (multiprocessing
    dataloaders require this).
    """

    t_mid: float
    lo: float
    hi: float

    def __call__(self, t: float) -> float:
        return self.hi if t >= self.t_mid else self.lo


@dataclass
class _RampProfile:
    """Linear ramp from ``val_start`` at ``t_start`` to ``val_end`` at ``t_end``."""

    t_start: float
    t_end: float
    val_start: float
    val_end: float

    def __call__(self, t: float) -> float:
        if self.t_end == self.t_start:
            return self.val_end
        frac = (t - self.t_start) / (self.t_end - self.t_start)
        frac = max(0.0, min(1.0, frac))
        return self.val_start + frac * (self.val_end - self.val_start)


@dataclass
class _SineProfile:
    """One-period sinusoid: ``c(t) = amplitude * sin(2*pi*(t - t_start)/period)``."""

    t_start: float
    period: float
    amplitude: float

    def __call__(self, t: float) -> float:
        return self.amplitude * math.sin(2.0 * math.pi * (t - self.t_start) / self.period)


from .continuous_scm import (
    ContinuousIntervention,
    InterventionKind,
)
from .time_schedule import (
    exponential_schedule,
    jittered_schedule,
    regular_schedule,
)
from .tscm_sampler import ContinuousTSCMSampler


def _pad_to_max_nodes(X: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """Right-pad ``X`` of shape ``(T, N)`` to ``(T, max_nodes)`` with zeros."""
    T, N = X.shape
    if max_nodes <= N:
        return X[:, :max_nodes]
    padding = torch.zeros(T, max_nodes - N, dtype=X.dtype, device=X.device)
    return torch.cat([X, padding], dim=1)


def _build_schedule(
    schedule: str,
    T: int,
    dt: float,
    jitter: float,
    exp_rate: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the requested schedule family."""
    if schedule == "regular":
        return regular_schedule(T=T, dt=dt)
    if schedule == "jittered":
        return jittered_schedule(T=T, dt=dt, jitter=jitter, generator=generator)
    if schedule == "exponential":
        return exponential_schedule(T=T, rate=exp_rate, generator=generator)
    raise ValueError(f"unknown schedule: {schedule!r}")


class ContinuousExtendedPrior:
    """Continuous-time model-ready sample generator.

    Parameters
    ----------
    tscm_structure : str
        One of the :class:`TSCMStructure` values (``back_door``,
        ``front_door``, ...).  Random-topology sampling (the CTP path in
        the discrete prior) is intentionally not supported here; the
        workshop paper is scoped to the named identifiability
        structures.
    n_max : int
        Pad width along the variable axis.  Defaults to 41 to match
        DoT-PFN's CausalChamber-motivated default.
    t_range : tuple of int
        Uniform prior on ``T`` (number of observations).
    schedule : {"regular", "jittered", "exponential"}
        Observation schedule family.  ``regular`` reproduces the
        discrete-time behaviour at ``dt=1.0``.
    dt : float
        Mean inter-observation gap for ``regular`` / ``jittered``
        schedules.
    jitter : float
        Used only by ``jittered``; see :func:`jittered_schedule`.
    exp_rate : float
        Used only by ``exponential``; inter-arrival rate.
    pair_mode : {"counterfactual", "interventional"}
        Selects paired-sample semantics (see module docstring).
    intervention_value_scale : float
        Standard deviation of the Gaussian prior on hard-intervention
        values.
    intervention_window_frac : tuple of float
        Lower/upper bounds (as fractions of total trajectory duration)
        for the intervention window length.  Default ``(0.1, 0.3)``
        matches the discrete-time roughly-30% default.
    intervention_kind_probs : tuple of float
        ``(p_hard, p_soft, p_time_varying)`` probabilities for sampling
        the intervention kind per trajectory.  Must be nonnegative and
        sum to a positive number (they will be normalised).  Defaults
        to ``(0.5, 0.3, 0.2)`` matching the CausalTime defaults.
    intervention_source : {"prior", "positivity_aware"}
        Source of the intervention value.  ``"prior"`` draws from
        ``N(0, intervention_value_scale^2)``.  ``"positivity_aware"``
        additionally clips the sampled hard-intervention value to
        ``[mean - 3*sigma, mean + 3*sigma]`` of the pre-intervention
        target variable's observational values, so that interventions
        stay inside the support seen by the encoder.  Only applied to
        hard interventions (soft and time-varying already live inside
        the observed range by construction).
    time_varying_profile : {"step", "ramp", "sine", "random"}
        Which time-varying profile family to sample.  ``"random"`` picks
        uniformly among the three fixed profiles per sample.
    soft_shift_scale : float
        Standard deviation of the Gaussian prior on the additive drift
        shift ``delta`` for soft interventions.
    theta_range, sigma_range, weight_scale : forwarded to :class:`ContinuousTSCMSampler`.
    seed : int
        Seeds the initial ``torch.Generator`` and ``numpy`` RNG.
    """

    _INT_KIND_ORDER = (
        InterventionKind.HARD,
        InterventionKind.SOFT,
        InterventionKind.TIME_VARYING,
    )

    def __init__(
        self,
        tscm_structure: str = "back_door",
        n_max: int = 41,
        t_range: tuple = (50, 200),
        schedule: str = "regular",
        dt: float = 1.0,
        jitter: float = 0.3,
        exp_rate: float = 1.0,
        pair_mode: str = "counterfactual",
        intervention_value_scale: float = 2.0,
        intervention_window_frac: tuple = (0.1, 0.3),
        intervention_kind_probs: tuple = (1.0, 0.0, 0.0),
        intervention_source: str = "prior",
        time_varying_profile: str = "random",
        soft_shift_scale: float = 1.0,
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.6),
        weight_scale: float = 0.5,
        num_substeps: int = 1,
        p_no_context: float = 0.0,
        seed: int = 42,
    ) -> None:
        if pair_mode not in ("counterfactual", "interventional"):
            raise ValueError(f"invalid pair_mode: {pair_mode!r}")
        if not isinstance(num_substeps, int) or num_substeps < 1:
            raise ValueError(f"num_substeps must be a positive int, got {num_substeps}")
        if not 0.0 <= p_no_context <= 1.0:
            raise ValueError(f"p_no_context must be in [0, 1], got {p_no_context}")
        if intervention_source not in ("prior", "positivity_aware"):
            raise ValueError(f"invalid intervention_source: {intervention_source!r}")
        if time_varying_profile not in ("step", "ramp", "sine", "random"):
            raise ValueError(f"invalid time_varying_profile: {time_varying_profile!r}")

        probs = np.asarray(intervention_kind_probs, dtype=np.float64)
        if probs.shape != (3,) or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError(
                f"intervention_kind_probs must be length-3 nonnegative with positive sum, got {intervention_kind_probs}"
            )
        self.intervention_kind_probs = tuple((probs / probs.sum()).tolist())

        self.n_max = n_max
        self.t_range = tuple(t_range)
        self.schedule = schedule
        self.dt = float(dt)
        self.jitter = float(jitter)
        self.exp_rate = float(exp_rate)
        self.pair_mode = pair_mode
        self.intervention_value_scale = float(intervention_value_scale)
        self.intervention_window_frac = tuple(intervention_window_frac)
        self.intervention_source = intervention_source
        self.time_varying_profile = time_varying_profile
        self.soft_shift_scale = float(soft_shift_scale)
        self.num_substeps = int(num_substeps)
        self.p_no_context = float(p_no_context)

        self.sampler = ContinuousTSCMSampler(
            structure=TSCMStructure(tscm_structure),
            theta_range=theta_range,
            sigma_range=sigma_range,
            weight_scale=weight_scale,
        )

        self._seed = seed
        self._torch_gen = torch.Generator().manual_seed(seed)
        self._np_rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------ hook

    def _sample_scm_context(self) -> _SampledSCMContext:
        """Sample one SCM and return its topology-dependent metadata.

        Subclasses override this to plug in a different sampler (e.g. the
        random-graph :class:`RandomContinuousExtendedPrior`).  The
        default implementation returns the fixed-TSCM metadata cached on
        ``self.sampler``.
        """
        scm = self.sampler.sample(generator=self._torch_gen)
        return _SampledSCMContext(
            scm=scm,
            n_vars=self.sampler.n_vars,
            intervention_target_topo=self.sampler.get_intervention_target(),
            outcome_var_topo=self.sampler.get_outcome_var(),
            hidden_vars_topo=list(self.sampler.get_hidden_vars()),
        )

    # ------------------------------------------------------------------ helpers

    @property
    def n_vars(self) -> int:
        """Fixed-topology samplers expose a constant; random-graph samplers
        still expose a useful *upper bound* (``n_max_prior``) via the
        override in :class:`RandomContinuousExtendedPrior`."""
        return self.sampler.n_vars

    @property
    def hidden_vars(self) -> list[int]:
        """Backward-compat accessor -- callers outside ``generate_sample``
        may still want the fixed-topology hidden variable list.  For
        random-graph samplers this falls back to an empty list."""
        if hasattr(self.sampler, "get_hidden_vars"):
            return list(self.sampler.get_hidden_vars())
        return []

    def sample_T(self) -> int:
        """Sample a trajectory length uniformly from ``t_range``."""
        return int(self._np_rng.randint(self.t_range[0], self.t_range[1] + 1))

    @staticmethod
    def _permute(X: torch.Tensor, canonical_perm: list[int]) -> torch.Tensor:
        """Apply an arbitrary canonical topological-order permutation."""
        return X[:, canonical_perm]

    # ------------------------------------------------------------------ sample

    def generate_sample(
        self,
        T: int | None = None,
        n_queries: int = 1,
        query_mode: str = "single",
    ) -> dict[str, torch.Tensor]:
        """Draw one trajectory with a single or multi-query batch entry.

        Parameters
        ----------
        T : int, optional
            Number of observations; sampled from ``t_range`` if omitted.
        n_queries : int
            Number of (variable, time) query points attached to this
            trajectory.  ``query_target`` / ``query_time`` / ``Y_true`` /
            ``Y_causal_effect`` become 1-D tensors of this length when
            > 1.
        query_mode : {"single", "all_pairs"}
            "single" picks ``n_queries`` random (variable, time) points.
            "all_pairs" queries every variable at every sampled time
            (``n_queries`` becomes the number of distinct times).

        Returns
        -------
        dict
            See the module docstring for field semantics.
        """
        if T is None:
            T = self.sample_T()

        # 1. Sample observation schedule
        times, dts = _build_schedule(
            schedule=self.schedule,
            T=T,
            dt=self.dt,
            jitter=self.jitter,
            exp_rate=self.exp_rate,
            generator=self._torch_gen,
        )
        span = float((times[-1] - times[0]).item())

        # 2. Sample SCM and its topology-dependent metadata via the hook.
        ctx = self._sample_scm_context()
        scm = ctx.scm
        n_vars = ctx.n_vars
        canonical_perm = ctx.canonical_perm
        topo_to_canon = ctx.topo_to_canon

        # 3. Sample intervention window and kind.  The *value* is
        #    deferred until after we have simulated X_obs so that
        #    positivity-aware clipping can see the pre-intervention
        #    support.
        a_idx_topo = ctx.intervention_target_topo
        win_frac = float(self._np_rng.uniform(*self.intervention_window_frac))
        win_len = max(self.dt * 2, win_frac * span)

        # Phase 13b: with probability ``p_no_context`` we force the
        # intervention to start at the very first observation, so the
        # encoder runs on an empty pre-intervention window.  This
        # matches the regime that the PK adapters
        # (``build_theophylline_batch``, ``build_warfarin_batch``) use
        # at evaluation time -- training never saw such samples
        # before, so the cross-variable mixer was extrapolating on
        # those benchmarks.  Sampling 5--20% of training trajectories
        # in this regime brings PK eval inside the training
        # distribution at a small cost in the rich-context regime.
        no_context = self._np_rng.rand() < self.p_no_context
        if no_context:
            t_int_start = float(times[0].item())
            t_int_end = t_int_start + win_len
        else:
            earliest_start = times[0].item() + 0.3 * span
            latest_start = times[-1].item() - win_len
            if latest_start <= earliest_start:
                earliest_start = times[0].item() + 0.1 * span
                latest_start = times[-1].item() - self.dt
            t_int_start = float(self._np_rng.uniform(earliest_start, latest_start))
            t_int_end = t_int_start + win_len
        intervention_kind = self._sample_intervention_kind()

        # 4. Simulate X_obs (never sees the intervention).  Pre-sample
        #    the noise (and, for regime-switching SCMs, the regime
        #    trajectory) up front so that the counterfactual path can
        #    share them with the interventional simulation below.
        fine_steps = (times.numel() - 1) * self.num_substeps
        shared_noise = scm._draw_noise(fine_steps, generator=self._torch_gen)
        shared_regime_traj = None
        if hasattr(scm, "_draw_regime_trajectory"):
            shared_regime_traj = scm._draw_regime_trajectory(
                times.numel(),
                generator=self._torch_gen,
            )
        _, X_obs_raw = scm.simulate(
            times,
            dts,
            intervention=None,
            noise=shared_noise,
            regime_trajectory=shared_regime_traj,
            num_substeps=self.num_substeps,
        )

        # 5. Compute int_onset_idx and derive pre-intervention stats
        #    from X_obs_raw; these feed positivity-aware value clipping.
        onset_mask = times >= t_int_start
        int_onset_idx = int(onset_mask.float().argmax().item()) if onset_mask.any() else T - 1
        pre_int_target = X_obs_raw[:int_onset_idx, a_idx_topo]

        # 6. Sample intervention value (respects kind + positivity).
        intervention_value, intervention_values_field = self._sample_intervention_value(
            kind=intervention_kind,
            pre_int_target=pre_int_target,
            t_int_start=t_int_start,
            t_int_end=t_int_end,
        )
        intervention = ContinuousIntervention(
            target=a_idx_topo,
            t_start=t_int_start,
            t_end=t_int_end,
            kind=intervention_kind,
            value=intervention_values_field,
        )

        # 7. Simulate the interventional / counterfactual trajectory.
        #    Counterfactual pair_mode reuses shared_noise + shared regime
        #    trajectory so pre-window trajectories match bit-for-bit;
        #    interventional pair_mode draws fresh noise (and, for
        #    regime-switching SCMs, a fresh regime trajectory too).
        if self.pair_mode == "counterfactual":
            _, X_int_raw = scm.simulate(
                times,
                dts,
                intervention=intervention,
                noise=shared_noise,
                regime_trajectory=shared_regime_traj,
                num_substeps=self.num_substeps,
            )
        else:
            _, X_int_raw = scm.simulate(
                times,
                dts,
                intervention=intervention,
                generator=self._torch_gen,
                num_substeps=self.num_substeps,
            )
        X_obs = X_obs_raw
        X_int = X_int_raw

        # 8. Apply canonical permutation (A -> 0, Y -> N-1, ...)
        X_obs = self._permute(X_obs, canonical_perm)
        X_int = self._permute(X_int, canonical_perm)
        intervention_target_canon = topo_to_canon[a_idx_topo]
        hidden_canon = [topo_to_canon[h] for h in ctx.hidden_vars_topo]

        # 9. Hide unobserved variables from the model in both X_obs and X_int.
        #    The simulator has already run on the full (visible + hidden)
        #    system, so dynamics of observed variables correctly reflect the
        #    hidden confounder -- we only strip the model's *view* of the
        #    hidden nodes here.  This is the mechanism that makes structures
        #    like front_door and instrumental_variable actually test the
        #    model's identifiability reasoning.
        if hidden_canon:
            for h in hidden_canon:
                X_obs[:, h] = 0.0
                X_int[:, h] = 0.0

        # 10. Causal masking: zero out post-intervention observations
        X_obs_masked = X_obs.clone()
        X_obs_masked[int_onset_idx:] = 0.0

        # 11. Pad to n_max and build the variable mask.  Hidden positions are
        #     marked with 0 so the encoder treats them as padding.
        X_obs_padded = _pad_to_max_nodes(X_obs_masked, self.n_max)
        X_obs_full_padded = _pad_to_max_nodes(X_obs, self.n_max)  # unmasked, for release
        X_int_padded = _pad_to_max_nodes(X_int, self.n_max)
        variable_mask = torch.zeros(self.n_max)
        variable_mask[:n_vars] = 1.0
        for h in hidden_canon:
            variable_mask[h] = 0.0

        # 9. Sample queries (variable, time) pairs.  Query time defaults
        #    to the intervention window midpoint offset by a small jitter
        #    sampled from the post-intervention region.
        query_target_idx, query_time_idx = self._sample_queries(
            T=T,
            n_queries=n_queries,
            query_mode=query_mode,
            int_onset_idx=int_onset_idx,
            intervention_target_canon=intervention_target_canon,
            ctx=ctx,
        )
        query_time_abs = times[query_time_idx]
        y_true = X_int[query_time_idx, query_target_idx]
        y_obs = X_obs[query_time_idx, query_target_idx]
        y_causal_effect = y_true - y_obs

        # 10. Times normalised to [0, 1] for compatibility with the existing mixer.
        times_norm = (times - times[0]) / max(span, 1e-6)
        t_int_start_norm = (t_int_start - times[0].item()) / max(span, 1e-6)
        t_int_end_norm = (t_int_end - times[0].item()) / max(span, 1e-6)
        query_time_norm = times_norm[query_time_idx]

        sample: dict[str, torch.Tensor] = {
            # Trajectories
            "X_obs": X_obs_padded,
            "X_obs_full": X_obs_full_padded,
            "X_int": X_int_padded,
            "variable_mask": variable_mask,
            "num_vars": torch.tensor(n_vars),
            # Schedule (new for continuous-time)
            "times": times,
            "dts": dts,
            # Intervention (existing contract)
            "int_onset_idx": torch.tensor(int_onset_idx, dtype=torch.long),
            "intervention_target": torch.tensor(intervention_target_canon, dtype=torch.long),
            "intervention_type": torch.tensor(
                self._INT_KIND_ORDER.index(intervention_kind),
                dtype=torch.long,
            ),
            "intervention_value": torch.tensor(intervention_value, dtype=torch.float32),
            "intervention_time_start": torch.tensor(t_int_start_norm, dtype=torch.float32),
            "intervention_time_end": torch.tensor(t_int_end_norm, dtype=torch.float32),
            # Absolute-time intervention fields (new for continuous-time)
            "t_int_start": torch.tensor(t_int_start, dtype=torch.float32),
            "t_int_end": torch.tensor(t_int_end, dtype=torch.float32),
            # Query
            "query_target": query_target_idx.long()
            if n_queries > 1
            else torch.tensor(int(query_target_idx.item()), dtype=torch.long),
            "query_time": query_time_norm
            if n_queries > 1
            else torch.tensor(float(query_time_norm.item()), dtype=torch.float32),
            "t_query": query_time_abs
            if n_queries > 1
            else torch.tensor(float(query_time_abs.item()), dtype=torch.float32),
            "Y_true": y_true
            if n_queries > 1
            else torch.tensor(float(y_true.item()), dtype=torch.float32),
            "Y_obs": y_obs
            if n_queries > 1
            else torch.tensor(float(y_obs.item()), dtype=torch.float32),
            "Y_causal_effect": y_causal_effect
            if n_queries > 1
            else torch.tensor(float(y_causal_effect.item()), dtype=torch.float32),
        }
        return sample

    # ------------------------------------------------------------------ interventions

    def _sample_intervention_kind(self) -> InterventionKind:
        """Pick one of ``HARD`` / ``SOFT`` / ``TIME_VARYING`` per ``intervention_kind_probs``."""
        idx = self._np_rng.choice(3, p=self.intervention_kind_probs)
        return self._INT_KIND_ORDER[idx]

    def _sample_intervention_value(
        self,
        kind: InterventionKind,
        pre_int_target: torch.Tensor,
        t_int_start: float,
        t_int_end: float,
    ) -> tuple[float, object]:
        """Sample (scalar_representation, intervention_values_field).

        Returns a ``(scalar, values)`` pair where ``scalar`` is a
        single-float summary for the batch dict's
        ``intervention_value`` field (used as a scalar feature in the
        mixer), and ``values`` is the object consumed by
        :class:`ContinuousIntervention.value`.

        - ``HARD``: both entries are the sampled constant.
        - ``SOFT``: both entries are the additive drift shift.
        - ``TIME_VARYING``: the scalar is a representative value at the
          window midpoint; ``values`` is a picklable callable from
          :mod:`intervention_profiles`.
        """
        if kind is InterventionKind.HARD:
            value = float(self._np_rng.randn() * self.intervention_value_scale)
            if (
                self.intervention_source == "positivity_aware"
                and pre_int_target.numel() > 1
                and float(pre_int_target.std().item()) > 1e-4
            ):
                mu = float(pre_int_target.mean().item())
                sigma = float(pre_int_target.std().item())
                value = float(np.clip(value, mu - 3.0 * sigma, mu + 3.0 * sigma))
            return value, value

        if kind is InterventionKind.SOFT:
            delta = float(self._np_rng.randn() * self.soft_shift_scale)
            return delta, delta

        # TIME_VARYING
        profile_name = self.time_varying_profile
        if profile_name == "random":
            profile_name = self._np_rng.choice(["step", "ramp", "sine"])

        amplitude = self.intervention_value_scale
        if profile_name == "step":
            t_mid = 0.5 * (t_int_start + t_int_end)
            lo = -amplitude
            hi = amplitude
            callable_obj = _StepProfile(t_mid=float(t_mid), lo=float(lo), hi=float(hi))
            scalar = 0.0  # mean of a symmetric step over the window
        elif profile_name == "ramp":
            lo = -amplitude
            hi = amplitude
            callable_obj = _RampProfile(
                t_start=float(t_int_start),
                t_end=float(t_int_end),
                val_start=float(lo),
                val_end=float(hi),
            )
            scalar = 0.0  # mean of the linear ramp over the window
        elif profile_name == "sine":
            period = max(t_int_end - t_int_start, self.dt)
            callable_obj = _SineProfile(
                t_start=float(t_int_start),
                period=float(period),
                amplitude=float(amplitude),
            )
            scalar = 0.0  # mean of a full sine cycle is zero
        else:
            raise ValueError(f"unknown time_varying_profile: {profile_name!r}")

        return float(scalar), callable_obj

    # ------------------------------------------------------------------ queries

    def _sample_queries(
        self,
        T: int,
        n_queries: int,
        query_mode: str,
        int_onset_idx: int,
        intervention_target_canon: int,
        ctx: _SampledSCMContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(query_target_idx, query_time_idx)`` both shape ``(n_queries,)``.

        Query times are sampled from the post-intervention region so the
        downstream effect is observable.  Query targets are drawn
        uniformly from the observed variables (excluding hidden ones).
        The intervention target itself is allowed as a query target so
        the model also learns the direct effect of ``do(A := c)`` on A.
        """
        hidden_canon = {ctx.topo_to_canon[h] for h in ctx.hidden_vars_topo}
        observable_vars = [v for v in range(ctx.n_vars) if v not in hidden_canon]

        t_lo = int_onset_idx
        t_hi = max(int_onset_idx + 1, T - 1)
        if t_lo >= t_hi:
            t_lo = max(0, T - 2)
            t_hi = T - 1

        if query_mode == "all_pairs":
            times_idx = torch.tensor(
                self._np_rng.randint(t_lo, t_hi + 1, size=n_queries),
                dtype=torch.long,
            )
            targets_idx = torch.tensor(
                self._np_rng.choice(observable_vars, size=n_queries, replace=True),
                dtype=torch.long,
            )
        else:
            times_idx = torch.tensor(
                self._np_rng.randint(t_lo, t_hi + 1, size=n_queries),
                dtype=torch.long,
            )
            targets_idx = torch.tensor(
                self._np_rng.choice(observable_vars, size=n_queries, replace=True),
                dtype=torch.long,
            )
        return targets_idx, times_idx

    # ------------------------------------------------------------------ batch

    def generate_batch(
        self,
        batch_size: int,
        n_queries: int = 1,
        query_mode: str = "single",
        T: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stack ``batch_size`` samples into batched tensors.

        All samples in a batch share the same ``T`` (sampled once at the
        start of the call) so the trajectory tensors can be stacked
        cleanly.  Query tensors stack along dim 0 regardless.
        """
        if T is None:
            T = self.sample_T()
        samples = [
            self.generate_sample(T=T, n_queries=n_queries, query_mode=query_mode)
            for _ in range(batch_size)
        ]
        batch: dict[str, torch.Tensor] = {}
        for key in samples[0]:
            batch[key] = torch.stack([s[key] for s in samples], dim=0)
        return batch
