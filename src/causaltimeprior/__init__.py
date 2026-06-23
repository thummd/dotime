"""CausalTimePrior: synthetic temporal SCMs with interventions for causal foundation models.

The top-level namespace exposes the lightweight *core* — the prior, the temporal
SCM types, intervention specifications, and regime-switching builders — eagerly,
because these depend only on the core runtime requirements (torch, numpy,
networkx).

Heavier submodules that pull in optional dependencies (baselines, frozen-suite
loaders, plotting, evaluation) are imported *lazily* via :pep:`562`, so a bare
``import causaltimeprior`` stays fast and does not fail when an optional extra
is not installed. Access them as attributes and they are resolved on first use::

    import causaltimeprior as ctp

    prior = ctp.CausalTimePrior(seed=42)        # eager core
    suite = ctp.benchmarks.load_benchmark(...)  # lazily imports the submodule

Example
-------
>>> from causaltimeprior import CausalTimePrior
>>> prior = CausalTimePrior(seed=42)
>>> X_obs, X_int, intervention, scm = prior.generate_pair(T=100)
>>> X_obs.shape == X_int.shape
True
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# Eager core (core runtime deps only)
# --------------------------------------------------------------------------- #

from causaltimeprior.prior import CausalTimePrior
from causaltimeprior.temporal_scm import TemporalSCM
from causaltimeprior.temporal_graph import TemporalDAG, TemporalGraphBuilder
from causaltimeprior.temporal_mechanism import TemporalMechanism
from causaltimeprior.temporal_scm_builder import TemporalSCMBuilder
from causaltimeprior.interventions import (
    InterventionSampler,
    InterventionSpec,
    InterventionType,
)
from causaltimeprior.regime_switching import RegimeSwitchingTemporalSCM
from causaltimeprior.regime_switching_builder import RegimeSwitchingSCMBuilder
from causaltimeprior.utils import DEFAULT_CONFIG

# --------------------------------------------------------------------------- #
# Lazy submodules (PEP 562). These names resolve to submodules on first access
# and may require optional extras:
#   extended       -> counterfactual sampling modes
#   data           -> streaming dataloaders
#   benchmarks     -> frozen suite loaders          (needs: evaluation extra / pyarrow)
#   baselines      -> classical & Bayesian baselines(needs: baselines extra)
#   evaluation     -> metrics + evaluation harness
#   visualization  -> plotting helpers              (needs: matplotlib, in core)
# --------------------------------------------------------------------------- #

_LAZY_SUBMODULES = frozenset(
    {
        "extended",
        "data",
        "benchmarks",
        "baselines",
        "evaluation",
        "visualization",
    }
)


def __getattr__(name: str):  # noqa: D401  (PEP 562 module-level __getattr__)
    """Resolve lazy submodules on first attribute access."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache so subsequent access is direct
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


# Make the lazy submodules visible to type checkers and IDEs without importing
# them at runtime.
if TYPE_CHECKING:
    from causaltimeprior import (  # noqa: F401
        baselines,
        benchmarks,
        data,
        evaluation,
        extended,
        visualization,
    )


__all__ = [
    "__version__",
    # --- core types ---
    "CausalTimePrior",
    "TemporalSCM",
    "TemporalDAG",
    "TemporalGraphBuilder",
    "TemporalMechanism",
    "TemporalSCMBuilder",
    "InterventionSpec",
    "InterventionType",
    "InterventionSampler",
    "RegimeSwitchingTemporalSCM",
    "RegimeSwitchingSCMBuilder",
    "DEFAULT_CONFIG",
    # --- lazy submodules ---
    "extended",
    "data",
    "benchmarks",
    "baselines",
    "evaluation",
    "visualization",
]
