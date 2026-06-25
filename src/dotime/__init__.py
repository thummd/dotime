"""DoTime: synthetic temporal SCMs with interventions for causal foundation models.

The top-level namespace exposes the lightweight *core* — the prior, the temporal
SCM types, intervention specifications, and regime-switching builders — eagerly,
because these depend only on the core runtime requirements (torch, numpy,
networkx).

Heavier submodules that pull in optional dependencies (baselines, frozen-suite
loaders, plotting, evaluation) are imported *lazily* via :pep:`562`, so a bare
``import dotime`` stays fast and does not fail when an optional extra
is not installed. Access them as attributes and they are resolved on first use::

    import dotime as ctp

    prior = ctp.DoTime(seed=42)        # eager core
    suite = ctp.benchmarks.load_benchmark(...)  # lazily imports the submodule

Example
-------
>>> from dotime import DoTime
>>> prior = DoTime(seed=42)
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

from dotime.interventions import (
    InterventionSampler,
    InterventionSpec,
    InterventionType,
)
from dotime.prior import DoTime
from dotime.regime_switching import RegimeSwitchingTemporalSCM
from dotime.regime_switching_builder import RegimeSwitchingSCMBuilder
from dotime.temporal_graph import TemporalDAG, TemporalGraphBuilder
from dotime.temporal_mechanism import TemporalMechanism
from dotime.temporal_scm import TemporalSCM
from dotime.temporal_scm_builder import TemporalSCMBuilder
from dotime.utils import DEFAULT_CONFIG

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
        "continuous",
        "models",
        "data",
        "benchmarks",
        "baselines",
        "evaluation",
        "visualization",
    }
)


def __getattr__(name: str):
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
    from dotime import (
        baselines,
        benchmarks,
        continuous,
        data,
        evaluation,
        extended,
        models,
        visualization,
    )


__all__ = [
    "__version__",
    # --- core types ---
    "DoTime",
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
    "continuous",
    "models",
    "data",
    "benchmarks",
    "baselines",
    "evaluation",
    "visualization",
]
