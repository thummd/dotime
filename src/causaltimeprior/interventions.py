"""Intervention specifications and sampling for CausalTimePrior."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


@dataclass
class StepIntervention:
    """Step function intervention profile (picklable replacement for lambda)."""
    step_time: int
    def __call__(self, t):
        return 2.0 if t >= self.step_time else -2.0


@dataclass
class RampIntervention:
    """Ramp intervention profile (picklable replacement for lambda)."""
    start_time: int
    intervention_length: int
    def __call__(self, t):
        return -2.0 + 4.0 * (t - self.start_time) / self.intervention_length


@dataclass
class SineIntervention:
    """Sinusoidal intervention profile (picklable replacement for lambda)."""
    start_time: int
    freq: float
    def __call__(self, t):
        return 2.0 * np.sin(self.freq * (t - self.start_time))


@dataclass
class TrajectoryIntervention:
    """Trajectory-based intervention profile (picklable replacement for lambda)."""
    trajectory_dict: dict
    def __call__(self, t):
        return self.trajectory_dict.get(t, 0.0)


class InterventionType(Enum):
    """Types of interventions."""
    HARD = "hard"           # do(X_i := c)
    SOFT = "soft"           # X_i = f_i(...) + delta
    TIME_VARYING = "time_varying"  # do(X_i := c(t))


# Picklable time-varying intervention profiles, keyed by name for (de)serialization.
_PROFILE_REGISTRY = {
    "StepIntervention": StepIntervention,
    "RampIntervention": RampIntervention,
    "SineIntervention": SineIntervention,
    "TrajectoryIntervention": TrajectoryIntervention,
}


@dataclass
class InterventionSpec:
    """Specification of an intervention on a temporal SCM.

    Attributes:
        targets: List of variable indices to intervene on
        times: List of time indices when intervention is active
        intervention_type: Type of intervention (hard, soft, time-varying)
        values: Intervention values (constant, shift, or time-varying function)
    """
    targets: list[int]
    times: list[int]
    intervention_type: InterventionType
    values: float | torch.Tensor | Callable

    def to_dict(self) -> dict:
        """JSON-serializable view of the spec (round-trips via :meth:`from_dict`).

        The ``values`` field is encoded by kind: a scalar, a dense ``tensor``
        list, or a named time-varying ``profile`` with its parameters.
        """
        from dataclasses import asdict, is_dataclass

        v = self.values
        if isinstance(v, torch.Tensor):
            values = {"kind": "tensor", "data": v.detach().cpu().tolist()}
        elif is_dataclass(v) and type(v).__name__ in _PROFILE_REGISTRY:
            values = {"kind": "profile", "name": type(v).__name__, "params": asdict(v)}
        elif callable(v):
            raise TypeError(
                f"cannot serialize intervention value of type {type(v).__name__!r}; "
                "use a registered profile dataclass (StepIntervention, ...)"
            )
        else:
            values = {"kind": "scalar", "data": float(v)}
        return {
            "targets": list(self.targets),
            "times": list(self.times),
            "intervention_type": self.intervention_type.value,
            "values": values,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InterventionSpec":
        """Reconstruct an :class:`InterventionSpec` from :meth:`to_dict` output."""
        v = d["values"]
        kind = v["kind"]
        if kind == "scalar":
            values: float | torch.Tensor | Callable = float(v["data"])
        elif kind == "tensor":
            values = torch.tensor(v["data"], dtype=torch.float32)
        elif kind == "profile":
            values = _PROFILE_REGISTRY[v["name"]](**v["params"])
        else:
            raise ValueError(f"unknown intervention value kind {kind!r}")
        return cls(
            targets=list(d["targets"]),
            times=list(d["times"]),
            intervention_type=InterventionType(d["intervention_type"]),
            values=values,
        )


class InterventionSampler:
    """Samples random intervention specifications for temporal SCMs."""

    def __init__(
        self,
        N: int,
        T: int,
        p_hard: float = 0.5,
        p_soft: float = 0.3,
        p_time_varying: float = 0.2,
        max_targets: int = 2,
        min_intervention_length: int = 10,
        generator: torch.Generator | None = None,
    ):
        """
        Parameters
        ----------
        N : int
            Number of variables in the SCM.
        T : int
            Length of time series.
        p_hard : float
            Probability of hard intervention.
        p_soft : float
            Probability of soft intervention.
        p_time_varying : float
            Probability of time-varying intervention.
        max_targets : int
            Maximum number of variables to intervene on.
        min_intervention_length : int
            Minimum length of intervention period.
        generator : torch.Generator, optional
            RNG for reproducibility.
        """
        self.N = N
        self.T = T
        self.p_hard = p_hard
        self.p_soft = p_soft
        self.p_time_varying = p_time_varying
        self.max_targets = min(max_targets, N)
        self.min_intervention_length = min_intervention_length
        self.generator = generator

        # Normalize probabilities
        total = p_hard + p_soft + p_time_varying
        self.p_hard /= total
        self.p_soft /= total
        self.p_time_varying /= total

    def sample(self) -> InterventionSpec:
        """Sample a random intervention specification.
        
        Returns
        -------
        InterventionSpec
            Sampled intervention specification.
        """
        # Sample intervention type
        r = torch.rand(1, generator=self.generator).item()
        if r < self.p_hard:
            intervention_type = InterventionType.HARD
        elif r < self.p_hard + self.p_soft:
            intervention_type = InterventionType.SOFT
        else:
            intervention_type = InterventionType.TIME_VARYING

        # Sample targets (1 to max_targets variables)
        num_targets = int(torch.randint(1, self.max_targets + 1, (1,), generator=self.generator).item())
        targets = torch.randperm(self.N, generator=self.generator)[:num_targets].tolist()

        # Sample intervention times (contiguous period)
        intervention_length = int(torch.randint(
            self.min_intervention_length,
            self.T - self.min_intervention_length + 1,
            (1,),
            generator=self.generator
        ).item())
        start_time = int(torch.randint(
            self.min_intervention_length,
            self.T - intervention_length + 1,
            (1,),
            generator=self.generator
        ).item())
        times = list(range(start_time, start_time + intervention_length))

        # Sample intervention values based on type
        if intervention_type == InterventionType.HARD:
            # Hard intervention: constant value
            value = torch.randn(1, generator=self.generator).item() * 2.0
            values = value

        elif intervention_type == InterventionType.SOFT:
            # Soft intervention: additive shift
            value = torch.randn(1, generator=self.generator).item() * 1.0
            values = value

        else:  # TIME_VARYING
            # Time-varying intervention: choose profile type
            profile_type = int(torch.randint(0, 4, (1,), generator=self.generator).item())

            if profile_type == 0:  # Step function
                step_time = start_time + intervention_length // 2
                values = StepIntervention(step_time)

            elif profile_type == 1:  # Ramp
                values = RampIntervention(start_time, intervention_length)

            elif profile_type == 2:  # Sinusoidal
                freq = 2 * np.pi / intervention_length
                values = SineIntervention(start_time, freq)

            else:  # Sampled trajectory
                trajectory = torch.randn(intervention_length, generator=self.generator) * 2.0
                trajectory_dict = {start_time + i: trajectory[i].item() for i in range(intervention_length)}
                values = TrajectoryIntervention(trajectory_dict)

        return InterventionSpec(
            targets=targets,
            times=times,
            intervention_type=intervention_type,
            values=values,
        )
