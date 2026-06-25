"""Continuous-time extensions to the DoTime.

This subpackage hosts SDE-based mechanism samplers, Ornstein-Uhlenbeck
parameterisations, variable Delta-t scheduling, and a model-ready batch
generator used by the ICML FMSD 2026 workshop paper.  Discrete-time
code in the parent :mod:`dotime.prior` package stays untouched.

Public API
----------
:class:`OUMechanism`, :func:`sample_ou_mechanism`
    Per-variable linear-drift OU mechanism specification.
:class:`ContinuousSCM`
    Multivariate SCM that integrates OU mechanisms via Euler-Maruyama
    on an arbitrary observation schedule, with support for hard, soft,
    and time-varying interventions, as well as true counterfactual
    pairs via shared noise.
:class:`ContinuousIntervention`, :class:`InterventionKind`
    Intervention specification consumed by :class:`ContinuousSCM`.
:class:`ContinuousTSCMSampler`
    Named :class:`TSCMStructure` topology with OU mechanisms
    (back-door, front-door, IV, RCT, ...).
:class:`ContinuousExtendedPrior`
    Model-ready batch generator (analogue of
    :class:`ExtendedDoTime`).
:mod:`time_schedule`
    Helpers for regular, jittered, and Poisson-irregular observation
    grids.
"""

from .continuous_scm import (
    ContinuousIntervention,
    ContinuousSCM,
    InterventionKind,
)
from .extended_prior import ContinuousExtendedPrior
from .neural_drift_mechanism import (
    NeuralDriftMechanism,
    sample_neural_drift_mechanism,
)
from .ou_mechanism import OUMechanism, sample_ou_mechanism
from .random_sampler import (
    RandomContinuousExtendedPrior,
    RandomContinuousSCMSampler,
)
from .regime_switching import (
    ContinuousRegimeSwitchingSCM,
    sample_sticky_transition_matrix,
)
from .time_schedule import (
    exponential_schedule,
    from_times,
    jittered_schedule,
    regular_schedule,
)
from .tscm_sampler import ContinuousTSCMSampler

__all__ = [
    "ContinuousExtendedPrior",
    "ContinuousIntervention",
    "ContinuousRegimeSwitchingSCM",
    "ContinuousSCM",
    "ContinuousTSCMSampler",
    "InterventionKind",
    "NeuralDriftMechanism",
    "OUMechanism",
    "RandomContinuousExtendedPrior",
    "RandomContinuousSCMSampler",
    "exponential_schedule",
    "from_times",
    "jittered_schedule",
    "regular_schedule",
    "sample_neural_drift_mechanism",
    "sample_ou_mechanism",
    "sample_sticky_transition_matrix",
]
