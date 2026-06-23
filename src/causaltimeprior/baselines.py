"""Reference baselines for CausalTimePrior benchmark suites.

A small registry maps baseline *names* to constructors, so the CLI and the
evaluation harness can request a baseline by string (mirroring the
``BASELINE_STRING_TO_CLASS`` table in the original ``tscm_identifiability.py``).

Public surface
--------------
- :class:`Baseline`     — the predict interface every baseline implements.
- :func:`available`     — list registered baseline names.
- :func:`get`           — instantiate a baseline by name.
- :func:`register`      — decorator to add a baseline to the registry.

Implementers: the bodies of the model-backed baselines (PFN / Chronos / PCMCI /
Bayesian ITS) are stubbed with ``TODO(consolidate)`` where they must call the
real classes living in ``dotime`` and the existing ``baselines.py``. The
trivial baselines (Zero, Mean, VAR-OLS) are implemented so the harness runs
end-to-end immediately.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from causaltimeprior.benchmarks import Episode

__all__ = ["Baseline", "available", "get", "register"]


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


@runtime_checkable
class Baseline(Protocol):
    """Predict interventional outcomes for an episode's queries.

    Implementations return a 1-D tensor aligned with ``episode.query_target`` /
    ``episode.query_time`` — one predicted value per query.
    """

    name: str

    def predict(self, episode: Episode) -> torch.Tensor: ...


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, Callable[..., Baseline]] = {}


def register(name: str) -> Callable[[Callable[..., Baseline]], Callable[..., Baseline]]:
    """Class/factory decorator: register a baseline constructor under ``name``."""

    def _decorator(ctor: Callable[..., Baseline]) -> Callable[..., Baseline]:
        if name in _REGISTRY:
            raise ValueError(f"baseline {name!r} is already registered")
        _REGISTRY[name] = ctor
        return ctor

    return _decorator


def available() -> list[str]:
    """Return the names of all registered baselines."""
    return sorted(_REGISTRY)


def get(name: str, **kwargs: object) -> Baseline:
    """Instantiate a registered baseline by name.

    Extra keyword arguments are forwarded to the baseline constructor.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {available()}")
    return _REGISTRY[name](**kwargs)


# --------------------------------------------------------------------------- #
# Trivial baselines (fully implemented)
# --------------------------------------------------------------------------- #


@register("Zero")
class ZeroBaseline:
    """Predicts zero for every query. Sanity-check lower bound."""

    name = "Zero"

    def predict(self, episode: Episode) -> torch.Tensor:
        return torch.zeros(episode.query_target.numel())


@register("Mean")
class MeanBaseline:
    """Predicts the pre-intervention mean of the queried variable."""

    name = "Mean"

    def predict(self, episode: Episode) -> torch.Tensor:
        preds = []
        for q in range(episode.query_target.numel()):
            var = int(episode.query_target[q])
            preds.append(episode.x_obs[:, var].mean())
        return torch.stack(preds)


@register("VAR-OLS")
class VAROLSBaseline:
    """Linear vector-autoregression fit by OLS on the observational trajectory.

    A genuinely causal-naive baseline: it forecasts the queried variable from
    its own and others' lagged values, ignoring the intervention semantics.
    """

    name = "VAR-OLS"

    def __init__(self, lag: int = 3):
        self.lag = lag

    def predict(self, episode: Episode) -> torch.Tensor:
        x = episode.x_obs.detach().cpu().numpy()  # (T, N)
        coef, mean = self._fit(x)
        preds = []
        for q in range(episode.query_target.numel()):
            var = int(episode.query_target[q])
            # One-step-ahead prediction from the tail of the trajectory.
            hist = x[-self.lag :].reshape(-1)
            yhat = mean[var] + coef[var] @ (hist - np.tile(mean, self.lag))
            preds.append(float(yhat))
        return torch.tensor(preds, dtype=torch.float32)

    def _fit(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t, n = x.shape
        mean = x.mean(axis=0)
        xc = x - mean
        rows, targets = [], []
        for s in range(self.lag, t):
            rows.append(xc[s - self.lag : s].reshape(-1))
            targets.append(xc[s])
        a = np.asarray(rows)            # (T-lag, lag*N)
        b = np.asarray(targets)         # (T-lag, N)
        # Ridge-stabilised least squares: coef has shape (N, lag*N).
        gram = a.T @ a + 1e-3 * np.eye(a.shape[1])
        coef = np.linalg.solve(gram, a.T @ b).T
        return coef, mean


# --------------------------------------------------------------------------- #
# Model-backed baselines (templates — wire to real implementations)
# --------------------------------------------------------------------------- #


@register("Oracle")
class OracleBaseline:
    """Ground-truth SCM rollout. Upper bound on synthetic suites only.

    TODO(consolidate): the generating SCM is available at suite-build time;
    persist the true counterfactual target into Episode.metadata (or compute it
    from a stored SCM handle) and return it here. On suites without a stored
    oracle this should raise a clear error rather than guess.
    """

    name = "Oracle"

    def predict(self, episode: Episode) -> torch.Tensor:
        if "y_oracle" in episode.metadata:
            return torch.as_tensor(episode.metadata["y_oracle"], dtype=torch.float32)
        # If y_true carries the exact counterfactual for synthetic suites, use it.
        if episode.y_true is not None and episode.y_true.numel():
            return episode.y_true.float()
        raise RuntimeError("Oracle baseline requires a stored ground-truth target")


@register("PCMCI+")
class PCMCIBaseline:
    """PCMCI+ causal discovery (tigramite) + linear effect estimate.

    Requires the ``baselines`` extra: ``pip install 'causaltimeprior[baselines]'``.
    TODO(consolidate): run PCMCI+ to recover the lagged graph, then estimate the
    interventional effect by linear adjustment on the discovered parents.
    """

    name = "PCMCI+"

    def __init__(self, lag: int = 3, alpha: float = 0.05):
        try:
            import tigramite  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "PCMCI+ baseline needs the 'baselines' extra: "
                "pip install 'causaltimeprior[baselines]'"
            ) from exc
        self.lag = lag
        self.alpha = alpha

    def predict(self, episode: Episode) -> torch.Tensor:
        raise NotImplementedError("wire PCMCIBaseline.predict to tigramite + adjustment")


@register("BayesianITS")
class BayesianPiecewiseITSBaseline:
    """Bayesian piecewise interrupted-time-series reference (CausalPy).

    Intended for CTP-RegimeSwitch-v1, where a 2-regime episode reduces to a
    classic ABA ITS design.
    Requires the ``baselines`` extra.
    TODO(consolidate): fit a CausalPy InterruptedTimeSeries on the pre/post
    split implied by the intervention window; return the posterior-mean
    counterfactual at the query time.
    """

    name = "BayesianITS"

    def __init__(self) -> None:
        try:
            import causalpy  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "Bayesian ITS baseline needs the 'baselines' extra: "
                "pip install 'causaltimeprior[baselines]'"
            ) from exc

    def predict(self, episode: Episode) -> torch.Tensor:
        raise NotImplementedError("wire BayesianPiecewiseITSBaseline.predict to CausalPy")


@register("Chronos")
class ChronosObservationalBaseline:
    """Chronos forecaster used observationally (intervention-unaware).

    TODO(consolidate): reuse the existing `Chronos2Observational` wrapper from
    the original baselines module rather than re-implementing it.
    """

    name = "Chronos"

    def predict(self, episode: Episode) -> torch.Tensor:
        raise NotImplementedError("adapt Chronos2Observational into this interface")


@register("DoOverTimePFN")
class DoOverTimePFNBaseline:
    """The Do-Over-Time-PFN model (interventional).

    TODO(consolidate): load a trained DoOverTimePFN checkpoint and adapt the
    existing `BackDoorDoTPFNCausalEffect` call path into `predict`. Accept a
    `checkpoint` path/handle in __init__ so the harness can point at a specific
    model. This is the headline method, not a baseline — keep its wiring honest.
    """

    name = "DoOverTimePFN"

    def __init__(self, checkpoint: str | None = None, device: str = "cpu"):
        self.checkpoint = checkpoint
        self.device = device

    def predict(self, episode: Episode) -> torch.Tensor:
        raise NotImplementedError("load checkpoint and adapt BackDoorDoTPFNCausalEffect")
