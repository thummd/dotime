"""Multivariate continuous-time structural causal model.

Ties together a topological variable order, per-variable
:class:`OUMechanism`, and an observation :mod:`time_schedule` into a
forward simulator.  The simulator advances each variable by
Euler-Maruyama steps of size ``dt_i`` (variable across the trajectory)
and records the state at each observation time.

Key responsibilities this class implements:

1. **Lagged parent dynamics.**  Drift on variable ``v`` uses parent
   values at the previous observation time, mirroring the discrete-time
   ``lag = 1`` convention of :mod:`dotime.prior.batched_tscm`.  This
   keeps mechanisms causal by construction even when ``dt`` varies, and
   reproduces AR(1) dynamics when ``dt == 1``.

2. **Interventions.**  Hard interventions clamp a target variable to a
   constant during a half-open time window; soft interventions add a
   constant drift term; time-varying interventions replace the target
   with a user-supplied callable ``c(t)``.

3. **Counterfactual pairs.**  ``sample_counterfactual_pair`` pre-draws a
   single table of standard-normal increments and reuses it for both
   the observational and interventional trajectories.  Outside the
   intervention window the two trajectories are then identical modulo
   downstream causal propagation of the intervention -- i.e. true
   counterfactual semantics, not interventional semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from .neural_drift_mechanism import (
    NeuralDriftMechanism,
    sample_neural_drift_mechanism,
)
from .ou_mechanism import OUMechanism, sample_ou_mechanism

MechanismLike = Union[OUMechanism, NeuralDriftMechanism]


class InterventionKind(Enum):
    HARD = "hard"
    SOFT = "soft"
    TIME_VARYING = "time_varying"


@dataclass
class ContinuousIntervention:
    """Specification of an intervention on a :class:`ContinuousSCM`.

    Attributes
    ----------
    target : int
        Index (into the variable list) of the intervened variable.
    t_start, t_end : float
        Half-open time window ``[t_start, t_end)`` during which the
        intervention is active.  Observations strictly outside this
        window follow the unmodified SCM dynamics.
    kind : InterventionKind
        Hard (``do(X := c)``), soft (``dX/dt <- dX/dt + delta``), or
        time-varying (``do(X := c(t))``).
    value : float or Callable[[float], float]
        Constant for hard/soft; function of absolute time for
        time-varying.  For soft interventions this is the additive
        drift shift ``delta``.
    """

    target: int
    t_start: float
    t_end: float
    kind: InterventionKind
    value: Union[float, Callable[[float], float]]

    def is_active(self, t: float) -> bool:
        return self.t_start <= t < self.t_end

    def eval_value(self, t: float) -> float:
        if callable(self.value):
            return float(self.value(t))
        return float(self.value)


class ContinuousSCM:
    """Continuous-time SCM with lagged linear-drift mechanisms.

    The graph is assumed to have no instantaneous edges (all causal
    influence is mediated through the previous observation time).  That
    keeps the Euler-Maruyama step trivially causal regardless of
    topological order and exactly recovers the discrete-time behaviour
    of :mod:`dotime.prior.batched_tscm` at ``dt == 1``.

    Instantaneous edges could be added later by evaluating mechanisms
    in topological order within each timestep; for the ICML workshop
    scope, lagged-only is sufficient.

    Parameters
    ----------
    mechanisms : sequence of OUMechanism
        One mechanism per variable, in the order ``[0, 1, ..., N - 1]``.
        ``mechanisms[v].parents`` may reference any indices in ``[0, N)``
        other than ``v`` itself.
    device, dtype
        Applied to all simulation tensors.
    """

    def __init__(
        self,
        mechanisms: Sequence[MechanismLike],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if len(mechanisms) == 0:
            raise ValueError("ContinuousSCM requires at least one mechanism")
        for v, mech in enumerate(mechanisms):
            if v in mech.parents:
                raise ValueError(f"variable {v} has itself as a parent; self-loops disallowed")
            for u in mech.parents:
                if not 0 <= u < len(mechanisms):
                    raise ValueError(f"parent index {u} out of range for variable {v}")

        self.mechanisms = list(mechanisms)
        self.n_vars = len(mechanisms)
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------ sampling

    @classmethod
    def sample_random(
        cls,
        n_vars: int,
        edge_prob: float = 0.3,
        theta_range: tuple = (0.5, 2.0),
        sigma_range: tuple = (0.2, 0.8),
        weight_scale: float = 0.8,
        mechanism_kind: str = "linear",
        p_neural: float = 0.0,
        neural_hidden_dim: int = 8,
        neural_out_scale_range: tuple = (0.5, 2.0),
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ContinuousSCM":
        """Sample a random continuous-time SCM.

        A variable ``v`` can have any earlier variable ``u < v`` as a
        parent with probability ``edge_prob`` (this is a topological
        ordering and guarantees acyclicity in the lagged graph too).

        Mechanism family per variable is controlled by
        ``mechanism_kind``:

        - ``"linear"`` (default, Phase 1 behaviour): every mechanism is
          a linear-drift :class:`OUMechanism`.
        - ``"neural"``: every mechanism is a small-MLP
          :class:`NeuralDriftMechanism` (Phase 10).
        - ``"mixed"``: each mechanism is independently neural with
          probability ``p_neural`` and linear otherwise.
        """
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError(f"edge_prob must be in [0, 1], got {edge_prob}")
        if mechanism_kind not in ("linear", "neural", "mixed"):
            raise ValueError(
                f"mechanism_kind must be 'linear', 'neural', or 'mixed'; "
                f"got {mechanism_kind!r}"
            )
        if not 0.0 <= p_neural <= 1.0:
            raise ValueError(f"p_neural must be in [0, 1], got {p_neural}")

        mechs: List[MechanismLike] = []
        for v in range(n_vars):
            parents: List[int] = []
            if v > 0:
                mask = torch.empty(v, device=device, dtype=dtype)
                mask.uniform_(0.0, 1.0, generator=generator)
                parents = [int(u) for u in range(v) if mask[u] < edge_prob]

            use_neural = mechanism_kind == "neural"
            if mechanism_kind == "mixed" and p_neural > 0.0:
                coin = torch.empty(1, device=device, dtype=dtype)
                coin.uniform_(0.0, 1.0, generator=generator)
                use_neural = bool(coin.item() < p_neural)

            if use_neural:
                mechs.append(
                    sample_neural_drift_mechanism(
                        parents=parents,
                        theta_range=theta_range,
                        sigma_range=sigma_range,
                        out_scale_range=neural_out_scale_range,
                        hidden_dim=neural_hidden_dim,
                        weight_scale=weight_scale,
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                mechs.append(
                    sample_ou_mechanism(
                        parents=parents,
                        theta_range=theta_range,
                        sigma_range=sigma_range,
                        weight_scale=weight_scale,
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    )
                )
        return cls(mechs, device=device, dtype=dtype)

    # ------------------------------------------------------------------ helpers

    def _draw_noise(
        self,
        num_steps: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Pre-sample all standard-normal increments for an entire trajectory.

        Returns a tensor of shape ``(num_steps, n_vars)`` containing
        i.i.d. ``N(0, 1)`` samples. Sampling up front (rather than once
        per step) lets :meth:`sample_counterfactual_pair` reuse the same
        noise realisation across the observational and interventional
        runs.
        """
        noise = torch.empty(num_steps, self.n_vars, device=self.device, dtype=self.dtype)
        noise.normal_(mean=0.0, std=1.0, generator=generator)
        return noise

    def _step(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        noise_row: torch.Tensor,
        intervention: Optional[ContinuousIntervention],
        t_next: float,
    ) -> torch.Tensor:
        """Advance the state vector ``x`` by one Euler-Maruyama step.

        Parent values are taken from ``x`` (the pre-step state), so the
        update is simultaneous across variables and exactly causal for
        lagged-only graphs.
        """
        x_new = torch.empty_like(x)
        for v, mech in enumerate(self.mechanisms):
            if len(mech.parents) == 0:
                x_parents = torch.empty(0, device=x.device, dtype=x.dtype)
            else:
                x_parents = torch.stack([x[u] for u in mech.parents])

            drift = mech.drift(x[v], x_parents)
            if (
                intervention is not None
                and intervention.target == v
                and intervention.is_active(t_next)
                and intervention.kind is InterventionKind.SOFT
            ):
                drift = drift + float(intervention.eval_value(t_next))

            x_new[v] = x[v] + drift * dt + mech.sigma * torch.sqrt(dt) * noise_row[v]

            if intervention is not None and intervention.target == v and intervention.is_active(t_next):
                if intervention.kind is InterventionKind.HARD:
                    x_new[v] = torch.tensor(
                        intervention.eval_value(t_next),
                        device=x.device,
                        dtype=x.dtype,
                    )
                elif intervention.kind is InterventionKind.TIME_VARYING:
                    x_new[v] = torch.tensor(
                        intervention.eval_value(t_next),
                        device=x.device,
                        dtype=x.dtype,
                    )

        return x_new

    # ------------------------------------------------------------------ simulate

    def simulate(
        self,
        times: torch.Tensor,
        dts: torch.Tensor,
        intervention: Optional[ContinuousIntervention] = None,
        x0: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        regime_trajectory: Optional[torch.Tensor] = None,
        num_substeps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the SCM forward on a given observation schedule.

        ``regime_trajectory`` is accepted but ignored by this base class;
        the argument is kept for interface parity with
        :class:`dotime.prior.continuous.regime_switching.ContinuousRegimeSwitchingSCM`
        so the extended prior can dispatch both SCM types uniformly.

        Parameters
        ----------
        times : torch.Tensor
            Shape ``(T,)``; strictly increasing observation times.
        dts : torch.Tensor
            Shape ``(T - 1,)``; ``dts[i] = times[i+1] - times[i]``.
            Passed explicitly to avoid recomputing and to let callers
            share a pre-computed grid between paired simulations.
        intervention : ContinuousIntervention, optional
            If given, applied according to its kind and time window.
        x0 : torch.Tensor, optional
            Initial state ``X(times[0])`` of shape ``(n_vars,)``.
            Defaults to zeros.
        noise : torch.Tensor, optional
            Pre-sampled noise increments.  Shape must be
            ``((T - 1) * num_substeps, n_vars)``.  Passing this in lets
            callers share a noise realisation between paired simulations
            -- this is the mechanism used by
            :meth:`sample_counterfactual_pair` to produce true
            counterfactual rather than interventional pairs.  Note that
            when ``num_substeps > 1`` the required noise is longer
            than the observation schedule to cover all fine-grid steps.
        generator : torch.Generator, optional
            Only used when ``noise`` is not provided.
        num_substeps : int, default 1
            Phase-11 fine-grid integration.  Each observation gap
            ``Delta_i = times[i+1] - times[i]`` is split into
            ``num_substeps`` Euler-Maruyama sub-steps of size
            ``Delta_i / num_substeps`` with independent noise per
            sub-step.  ``num_substeps = 1`` recovers naive
            observation-grid integration (tier B in the paper); values
            much greater than 1 approximate tier-C fine-grid
            integration so that the law of the trajectory is
            (approximately) schedule-invariant as required by
            Definition 3.1 of the paper.

        Returns
        -------
        times : torch.Tensor
            Echoed schedule ``(T,)``.
        trajectory : torch.Tensor
            State trajectory of shape ``(T, n_vars)``; only the
            observation times are recorded even when
            ``num_substeps > 1``.
        """
        if times.dim() != 1:
            raise ValueError(f"times must be 1-D, got shape {tuple(times.shape)}")
        if dts.dim() != 1 or dts.numel() != times.numel() - 1:
            raise ValueError(
                f"dts must be 1-D of length T - 1, got shape {tuple(dts.shape)} for T = {times.numel()}"
            )
        if not isinstance(num_substeps, int) or num_substeps < 1:
            raise ValueError(f"num_substeps must be a positive int, got {num_substeps}")
        T = times.numel()
        fine_steps_total = (T - 1) * num_substeps
        if noise is None:
            noise = self._draw_noise(fine_steps_total, generator=generator)
        elif noise.shape != (fine_steps_total, self.n_vars):
            raise ValueError(
                f"noise has shape {tuple(noise.shape)}, expected "
                f"({fine_steps_total}, {self.n_vars})"
            )

        if x0 is None:
            x = torch.zeros(self.n_vars, device=self.device, dtype=self.dtype)
        else:
            x = x0.to(device=self.device, dtype=self.dtype).clone()

        trajectory = torch.empty(T, self.n_vars, device=self.device, dtype=self.dtype)
        trajectory[0] = x
        for i in range(T - 1):
            fine_dt = dts[i] / num_substeps
            t_i = float(times[i].item())
            t_next = float(times[i + 1].item())
            for k in range(num_substeps):
                # Time at the END of this sub-step.  The intervention
                # policy is evaluated at this forward time, matching the
                # ``t_next`` convention used in the single-step path.
                t_fine = t_i + (k + 1) * float(fine_dt.item())
                if k == num_substeps - 1:
                    # Snap to exact observation time to avoid float drift.
                    t_fine = t_next
                noise_idx = i * num_substeps + k
                x = self._step(
                    x,
                    fine_dt,
                    noise[noise_idx],
                    intervention=intervention,
                    t_next=t_fine,
                )
            trajectory[i + 1] = x

        return times, trajectory

    def sample_counterfactual_pair(
        self,
        times: torch.Tensor,
        dts: torch.Tensor,
        intervention: ContinuousIntervention,
        x0: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        num_substeps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return matched ``(times, X_obs, X_cf)`` with shared noise.

        ``X_obs`` is the observational trajectory (no intervention);
        ``X_cf`` is the *counterfactual* trajectory under ``intervention``
        **using the same noise realisation**.  Before the intervention
        window the two trajectories agree exactly (up to numerical
        error); after it they diverge only through causal propagation of
        the intervention, not through independent noise.

        Contrast with :meth:`sample_interventional_pair`, which draws
        fresh noise for the interventional run and thus returns
        interventional rather than counterfactual semantics.

        ``num_substeps`` forwards to :meth:`simulate` and controls
        fine-grid integration; the shared noise is drawn on the fine
        grid so the counterfactual property is preserved at any
        refinement level.
        """
        fine_steps = (times.numel() - 1) * num_substeps
        noise = self._draw_noise(fine_steps, generator=generator)
        _, x_obs = self.simulate(
            times, dts, intervention=None, x0=x0, noise=noise,
            num_substeps=num_substeps,
        )
        _, x_cf = self.simulate(
            times, dts, intervention=intervention, x0=x0, noise=noise,
            num_substeps=num_substeps,
        )
        return times, x_obs, x_cf

    def sample_interventional_pair(
        self,
        times: torch.Tensor,
        dts: torch.Tensor,
        intervention: ContinuousIntervention,
        x0: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        num_substeps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return matched ``(times, X_obs, X_int)`` with independent noise.

        This reproduces the discrete-time pipeline's ``generate_pair``
        semantics: each trajectory draws its own noise, so the two
        diverge everywhere -- including before the intervention window
        -- purely due to different realisations.  Use this for
        population-level interventional training; use
        :meth:`sample_counterfactual_pair` for individual-level
        counterfactual training.
        """
        _, x_obs = self.simulate(
            times, dts, intervention=None, x0=x0, generator=generator,
            num_substeps=num_substeps,
        )
        _, x_int = self.simulate(
            times, dts, intervention=intervention, x0=x0,
            generator=generator, num_substeps=num_substeps,
        )
        return times, x_obs, x_int
