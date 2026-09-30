"""Estimators for the detection-power analysis of dot-Identifiability-v1.

Every estimator predicts the interventional *level* of the queried outcome at
an arbitrary do-value, so the analysis can query it twice per episode: at the
episode's intervention value ``v`` (the official effect-sign protocol) and at
``a_ref``, the factual treatment of the intervened row (the false-effect test).
With shared-noise counterfactual targets, ``do(A = a_ref)`` reproduces the
factual trajectory, so the level an estimator predicts there is its estimate of
``y_obs`` and ``pred(v) - pred(a_ref)`` is its estimated effect.

Estimator classes, as the analysis reports them:

- ``naive``: ``Zero``, ``Mean`` (TrajMean), ``AR1`` and ``VAR-OLS`` from
  :mod:`dotime.baselines`, unchanged. None of them reads the do-value.
- ``association``: ``NaiveOLS`` from :mod:`dotime.baselines`, the unadjusted
  regression ``Y_t ~ 1 + A_t + Y_{t-1}``. It is pending until the package
  registers it; this module never reimplements it.
- ``do-SVAR``: :class:`DoSVAR`, a recursive structural VAR in the canonical
  column order with the treatment clamped at the onset row.
- ``identification-aware``: ``BackDoorOLS`` and ``IV2SLS`` from
  :mod:`dotime.baselines`, and :class:`FrontDoorOLS`.
- ``router``: :class:`GraphRouter`, which picks one of the above from the
  structure's DAG (:func:`route`).
- ``oracle``: the stored ground truth (:class:`OracleEstimator`).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import networkx as nx
import numpy as np
import torch

from dotime import baselines
from dotime.baselines import (
    BackDoorOLSBaseline,
    _back_door_columns,
    _canonical_summary_graph,
    _front_door_columns,
    _ols_fit,
)
from dotime.evaluation import query_obs_levels

if TYPE_CHECKING:
    from dotime.benchmarks import Episode

__all__ = [
    "ASSOCIATION",
    "CLASS_OF",
    "ESTIMATOR_ORDER",
    "IDENTIFICATION_AWARE",
    "NAIVE",
    "DoSVAR",
    "Estimator",
    "FrontDoorOLS",
    "GraphRouter",
    "OracleEstimator",
    "PackageEstimator",
    "PendingEstimatorError",
    "Route",
    "build_estimators",
    "do_value",
    "onset_row",
    "query_col",
    "reference_do_value",
    "route",
    "svar_forecast",
    "treatment_col",
    "with_do_value",
]

NAIVE = ("Zero", "Mean", "AR1", "VAR-OLS")
ASSOCIATION = ("NaiveOLS",)
IDENTIFICATION_AWARE = ("BackDoorOLS", "IV2SLS", "FrontDoorOLS")
ESTIMATOR_ORDER = (
    *NAIVE,
    *ASSOCIATION,
    "do-SVAR",
    *IDENTIFICATION_AWARE,
    "GraphRouter",
    "Oracle",
)
CLASS_OF = {
    **dict.fromkeys(NAIVE, "naive"),
    **dict.fromkeys(ASSOCIATION, "association"),
    "do-SVAR": "do-SVAR",
    **dict.fromkeys(IDENTIFICATION_AWARE, "identification-aware"),
    "GraphRouter": "router",
    "Oracle": "oracle",
}

# do-SVAR settings of the original comparator (see DoSVAR).
SVAR_ORDER = 3
SVAR_RIDGE = 1e-2
# Standard-deviation floor of dotime.normalization.per_variable_normalize, which
# produced the loader statistics the original do-SVAR ran on.
NORM_EPS = 1e-2


class PendingEstimatorError(RuntimeError):
    """Raised when a prediction needs an estimator the package does not register yet."""


# --------------------------------------------------------------------------- #
# Episode accessors
# --------------------------------------------------------------------------- #


def onset_row(episode: Episode) -> int:
    """Row of the (one-shot) intervention, the first post-intervention step.

    Args:
        episode: A benchmark episode.

    Returns:
        The smallest intervened time index.

    Raises:
        ValueError: If the episode records no intervention time.
    """
    times = episode.intervention.times
    if not times:
        raise ValueError(f"episode {episode.scm_id} records no intervention time")
    return int(min(times))


def treatment_col(episode: Episode) -> int:
    """Column of the intervened variable.

    Args:
        episode: A benchmark episode.

    Returns:
        The single intervention target.

    Raises:
        ValueError: If the episode intervenes on other than exactly one variable.
    """
    targets = list(episode.intervention.targets)
    if len(targets) != 1:
        raise ValueError(f"episode {episode.scm_id} intervenes on {targets}, expected one target")
    return int(targets[0])


def query_col(episode: Episode) -> int:
    """Column of the queried variable.

    Args:
        episode: A benchmark episode.

    Returns:
        The single query target.

    Raises:
        ValueError: If the episode has other than exactly one query.
    """
    targets = episode.query_target.reshape(-1)
    if targets.numel() != 1:
        raise ValueError(f"episode {episode.scm_id} has {targets.numel()} queries, expected one")
    return int(targets[0])


def do_value(episode: Episode) -> float:
    """The episode's intervention value ``v``.

    Args:
        episode: A benchmark episode.

    Returns:
        The scalar hard-intervention value.

    Raises:
        ValueError: If the value is not a scalar (a soft or time-varying
            intervention), which the level estimators cannot take.
    """
    values = episode.intervention.values
    if isinstance(values, bool) or not isinstance(values, (int, float)):
        raise ValueError(f"episode {episode.scm_id} has a non-scalar intervention value")
    return float(values)


def reference_do_value(episode: Episode) -> float:
    """Factual treatment ``a_ref`` at the intervened row.

    Setting ``do(A = a_ref)`` at the onset leaves a shared-noise trajectory
    exactly as observed, so it is the reference against which an estimated
    effect is measured. For a query at the onset row it is ``a_obs(t_q)``.

    Args:
        episode: A benchmark episode.

    Returns:
        ``x_obs[onset, A]``.
    """
    return float(episode.x_obs[onset_row(episode), treatment_col(episode)])


def with_do_value(episode: Episode, value: float) -> Episode:
    """Copy of an episode whose intervention sets the treatment to ``value``.

    The package baselines read the do-value from ``episode.intervention.values``,
    so evaluating them at another value means handing them a copy with that
    field replaced. Every tensor is shared with the input episode.

    Args:
        episode: A benchmark episode.
        value: The do-value to substitute.

    Returns:
        A new episode that differs only in the intervention value.
    """
    return dataclasses.replace(
        episode, intervention=dataclasses.replace(episode.intervention, values=float(value))
    )


def _single_prediction(pred: object, episode: Episode, name: str) -> float:
    """Unpack a one-query prediction as a float32-exact Python float.

    Args:
        pred: The tensor a :mod:`dotime.baselines` baseline returned.
        episode: The episode it predicts, for the error message.
        name: The baseline's name, for the error message.

    Returns:
        The prediction, rounded to float32 exactly as
        ``dotime.reference.reference_table.run_baseline`` stores it.

    Raises:
        ValueError: If the baseline returned other than one prediction.
    """
    values = torch.as_tensor(pred, dtype=torch.float32).reshape(-1)
    if values.numel() != 1:
        raise ValueError(
            f"{name} returned {values.numel()} predictions for episode {episode.scm_id}"
        )
    return float(values[0])


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class Estimator(Protocol):
    """A level predictor that can be evaluated at any do-value.

    Attributes:
        name: Row label in the analysis tables.
    """

    name: str

    def uses_do_value(self, structure: str | None) -> bool:
        """Whether predictions on ``structure`` depend on the do-value at all.

        Args:
            structure: The episode's structure label.

        Returns:
            ``False`` when the estimator declines the structure (or never reads
            the do-value), so its predicted effect is zero by construction.
        """
        ...

    def predict(self, episode: Episode, value: float) -> float:
        """Predict the queried level under ``do(A = value)`` at the onset row.

        Args:
            episode: The episode to predict.
            value: The do-value.

        Returns:
            The predicted interventional level at the query row.
        """
        ...


# --------------------------------------------------------------------------- #
# Package baselines
# --------------------------------------------------------------------------- #


class PackageEstimator:
    """A registered :mod:`dotime.baselines` baseline, evaluated at any do-value.

    Args:
        name: A registered baseline name among the naive, association and
            package identification-aware estimators.

    Raises:
        ValueError: If ``name`` is not one of those.
        KeyError: If the package does not register ``name``.
    """

    _NAMES = (*NAIVE, *ASSOCIATION, "BackDoorOLS", "IV2SLS")

    def __init__(self, name: str) -> None:
        if name not in self._NAMES:
            raise ValueError(f"{name!r} is not a package estimator of this analysis")
        self.name = name
        self._model = baselines.get(name)

    def uses_do_value(self, structure: str | None) -> bool:
        """Whether the package baseline reads the do-value on ``structure``.

        Args:
            structure: The episode's structure label.

        Returns:
            ``False`` for the naive baselines, and for BackDoorOLS and IV2SLS
            on structures they decline (they then predict the pre-onset outcome
            mean). ``True`` for NaiveOLS, which applies to every structure.
        """
        if self.name in NAIVE:
            return False
        if self.name == "BackDoorOLS":
            return structure in BackDoorOLSBaseline._BACK_DOOR
        if self.name == "IV2SLS":
            # IV2SLSBaseline.predict tests this literal inline; it has no
            # class attribute to read, so the two must be kept in step.
            return structure == "instrumental_variable"
        return True

    def predict(self, episode: Episode, value: float) -> float:
        """Predict the queried level under ``do(A = value)``.

        Args:
            episode: The episode to predict.
            value: The do-value.

        Returns:
            The baseline's prediction as a float32-exact float.
        """
        # The episode's own value goes through untouched, so these are exactly
        # the predictions the released reference rows were computed from.
        ep = episode if value == do_value(episode) else with_do_value(episode, value)
        return _single_prediction(self._model.predict(ep), episode, self.name)


class OracleEstimator:
    """Stored ground truth, the ceiling of every metric.

    At the episode's do-value it is the package ``Oracle`` (``y_true``). At the
    factual treatment ``a_ref`` it is ``y_obs``: under shared-noise pairing,
    ``do(A = a_ref)`` at the onset reproduces the factual trajectory. It knows
    no other level.
    """

    name = "Oracle"

    def __init__(self) -> None:
        self._model = baselines.get("Oracle")

    def uses_do_value(self, structure: str | None) -> bool:
        """The oracle's level always depends on the do-value.

        Args:
            structure: The episode's structure label (unused).

        Returns:
            ``True``.
        """
        return True

    def predict(self, episode: Episode, value: float) -> float:
        """Return the true level at ``v`` or at ``a_ref``.

        Args:
            episode: The episode to predict.
            value: The episode's do-value or its factual treatment.

        Returns:
            ``y_true`` at ``v``, ``y_obs`` at ``a_ref``.

        Raises:
            ValueError: For any other value, or for ``a_ref`` on a suite whose
                arms do not share noise (the factual level is then not the
                outcome of ``do(A = a_ref)``).
        """
        if value == do_value(episode):
            return _single_prediction(self._model.predict(episode), episode, self.name)
        if value == reference_do_value(episode):
            if episode.metadata.get("pair_mode") != "counterfactual":
                raise ValueError(
                    f"episode {episode.scm_id} does not share noise across arms, so "
                    "y_obs is not the outcome of do(A = a_ref)"
                )
            return float(query_obs_levels(episode).reshape(-1)[0])
        raise ValueError("the oracle knows the outcome only at v and at the factual treatment")


# --------------------------------------------------------------------------- #
# Front-door adjustment
# --------------------------------------------------------------------------- #


class FrontDoorOLS:
    """Linear front-door estimator through the structure's observed mediator.

    Fits ``alpha`` from ``M_s ~ 1 + A_s + M_{s-1}`` and ``gamma`` from
    ``Y_s ~ 1 + M_s + A_s + Y_{s-1}`` on the pre-onset rows, where conditioning
    on ``A_s`` blocks the back-door path ``M <- A <- U -> Y``. The effect per
    unit of treatment is ``tau = alpha * gamma`` and the prediction is
    ``mean(Y_pre) + tau * (value - mean(A_pre))``. The mediator column comes
    from the structure's DAG (``dotime.baselines._front_door_columns``); on a
    structure without exactly one observed mediator, or with fewer than four
    pre-onset rows, it predicts the pre-onset outcome mean, as BackDoorOLS and
    IV2SLS do outside their structures.
    """

    name = "FrontDoorOLS"

    def uses_do_value(self, structure: str | None) -> bool:
        """Whether the structure has the single observed mediator the estimator needs.

        Args:
            structure: The episode's structure label.

        Returns:
            ``True`` for ``mediator``, ``front_door`` and ``confounder_mediator``.
        """
        if structure is None:
            return False
        try:
            _front_door_columns(structure)
        except ValueError:
            return False
        return True

    def predict(self, episode: Episode, value: float) -> float:
        """Predict the queried level under ``do(A = value)`` by front-door adjustment.

        Args:
            episode: The episode to predict.
            value: The do-value.

        Returns:
            The predicted level.

        Raises:
            ValueError: If an applicable episode does not use its structure's
                canonical column layout.
        """
        x = episode.x_obs.detach().cpu().numpy()
        t_len, n = x.shape
        y = query_col(episode)
        structure = episode.structure
        fit_end = max(2, min(onset_row(episode), t_len))
        if structure is None or not self.uses_do_value(structure) or fit_end < 4:
            # Same fallback, and the same float32 mean, as BackDoorOLS and IV2SLS.
            return float(x[:fit_end, y].mean())
        n_vars, a_col, y_col, m_col = _front_door_columns(structure)
        if n != n_vars or treatment_col(episode) != a_col or y != y_col:
            raise ValueError(
                f"FrontDoorOLS: a {structure!r} episode needs {n_vars} canonical "
                f"columns with A in {a_col} and Y queried in {y_col}; got {n} columns, "
                f"A in {treatment_col(episode)}, Y in {y}"
            )
        a, m, yy = (x[:fit_end, c].astype(np.float64) for c in (a_col, m_col, y_col))
        alpha = _ols_fit(np.column_stack([a[1:], m[:-1]]), m[1:])[1]
        gamma = _ols_fit(np.column_stack([m[1:], a[1:], yy[:-1]]), yy[1:])[1]
        return float(yy.mean() + alpha * gamma * (value - a.mean()))


# --------------------------------------------------------------------------- #
# do-SVAR
# --------------------------------------------------------------------------- #


def _ridge_fit(features: np.ndarray, targets: np.ndarray, lam: float) -> np.ndarray:
    """Ridge weights with an unpenalised intercept in the last feature column.

    Args:
        features: ``(n, d)`` design matrix whose last column is the intercept.
        targets: ``(n,)`` targets.
        lam: Ridge penalty on every other column.

    Returns:
        ``(d,)`` weights.
    """
    # Logic of _ridge_fit in do-over-time-pfn scripts/oracle_horizon.py.
    d = features.shape[1]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0
    return np.linalg.solve(features.T @ features + reg, features.T @ targets)


def svar_forecast(
    history: np.ndarray,
    p: int,
    steps: int,
    lam: float,
    target_col: int,
    clamp: tuple[int, float] | None = None,
) -> float:
    """Recursive structural VAR(p) rolled ``steps`` ahead, with an optional onset clamp.

    Column ``j`` is regressed on the contemporaneous values of the columns
    before it plus ``p`` lags of every column, so a same-step intervention on
    the first column reaches the later ones within the onset row. At the first
    generated row the clamped column is set to its value (a one-shot hard
    intervention); afterwards the rollout is free.

    Args:
        history: ``(W, n)`` standardised pre-onset rows.
        p: Lag order.
        steps: Rows to generate after the history (query offset ``k`` -> ``k + 1``).
        lam: Ridge penalty.
        target_col: Column whose value at the last generated row is returned.
        clamp: ``(column, value)`` set at the first generated row, or ``None``.

    Returns:
        The forecast of ``target_col``, or its history mean when the window is
        shorter than three times the number of parameters per equation.
    """
    # Logic of svar_forecast in do-over-time-pfn scripts/var_baselines.py
    # (private repository, same author), ported line for line.
    w, n = history.shape
    n_params = n * p + n + 1
    if w < 3 * n_params:
        return float(history[:, target_col].mean())
    lag_rows = np.stack([history[i : i + p].reshape(-1) for i in range(w - p)])
    cur = history[p:]
    weights = []
    for j in range(n):
        feats = np.concatenate([cur[:, :j], lag_rows, np.ones((w - p, 1))], axis=1)
        weights.append(_ridge_fit(feats, cur[:, j], lam))
    buf = [history[i] for i in range(w - p, w)]
    for s in range(steps):
        lags = np.stack(buf[-p:]).reshape(-1)
        row = np.zeros(n)
        for j in range(n):
            if clamp is not None and s == 0 and j == clamp[0]:
                row[j] = clamp[1]
                continue
            row[j] = float(np.concatenate([row[:j], lags, [1.0]]) @ weights[j])
        buf.append(row)
    return float(buf[-1][target_col])


class DoSVAR:
    """Clamped recursive structural VAR, the linear do-operator on the history.

    Standardises the pre-onset columns with the loader statistics of the
    original comparator (mean, Bessel-corrected standard deviation plus 1e-2),
    drops all-zero columns (the zeroed hidden variables), fits
    :func:`svar_forecast` with ``p = 3`` lags and ridge ``1e-2`` in the
    canonical column order (treatment first, outcome last), clamps the treatment
    to the do-value at the onset row and rolls forward to the query row. The
    recursive order is causal only when no observed variable drives the
    treatment within a step: a confounder that follows ``A`` in the canonical
    order is regressed on ``A`` and moves with the clamp.

    Args:
        order: Lag order ``p``.
        ridge: Ridge penalty.
    """

    name = "do-SVAR"

    def __init__(self, order: int = SVAR_ORDER, ridge: float = SVAR_RIDGE) -> None:
        self.order = order
        self.ridge = ridge

    def uses_do_value(self, structure: str | None) -> bool:
        """The clamp always enters the forecast (short histories aside).

        Args:
            structure: The episode's structure label (unused).

        Returns:
            ``True``.
        """
        return True

    def predict(self, episode: Episode, value: float) -> float:
        """Predict the queried level under ``do(A = value)`` with the clamped SVAR.

        Args:
            episode: The episode to predict.
            value: The do-value.

        Returns:
            The predicted level on the raw scale.

        Raises:
            ValueError: If the query precedes the onset, the onset leaves no
                history, or the kept columns do not put the treatment first and
                the outcome last.
        """
        x = episode.x_obs.detach().cpu().numpy().astype(np.float64)
        onset = onset_row(episode)
        a, y = treatment_col(episode), query_col(episode)
        q = int(episode.query_time_idx.reshape(-1)[0])
        if q < onset or onset < 1:
            raise ValueError(f"episode {episode.scm_id}: query row {q}, onset {onset}")
        pre = x[:onset]
        keep = [c for c in range(x.shape[1]) if c in (a, y) or bool(np.any(pre[:, c] != 0.0))]
        if keep[0] != a or keep[-1] != y:
            raise ValueError(
                f"do-SVAR needs the treatment first and the outcome last; episode "
                f"{episode.scm_id} keeps columns {keep} with A={a}, Y={y}"
            )
        pre = pre[:, keep]
        mean = pre.mean(axis=0)
        std = np.sqrt(((pre - mean) ** 2).sum(axis=0) / max(onset - 1, 1)) + NORM_EPS
        z = svar_forecast(
            (pre - mean) / std,
            self.order,
            q - onset + 1,
            self.ridge,
            len(keep) - 1,
            clamp=(0, (value - mean[0]) / std[0]),
        )
        return float(z * std[-1] + mean[-1])


# --------------------------------------------------------------------------- #
# Graph router
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Route:
    """The estimator the structure's DAG calls for, and why.

    Attributes:
        estimator: Name of the routed estimator, or ``None`` for ``tau = 0``.
        rule: The identification argument, as reported in the tables.
        identified: Whether the effect is identified from observational data.
    """

    estimator: str | None
    rule: str
    identified: bool


def _hidden_confounders(summary: nx.DiGraph, hidden: set[str]) -> set[str]:
    """Hidden variables that reach both ``A`` and ``Y``, the latter not through ``A``.

    Args:
        summary: The structure's summary graph.
        hidden: Names of its hidden variables.

    Returns:
        The hidden common causes of treatment and outcome.
    """
    without_a = summary.subgraph(set(summary) - {"A"})
    return {h for h in hidden if nx.has_path(summary, h, "A") and nx.has_path(without_a, h, "Y")}


def _front_door_mediator(summary: nx.DiGraph, hidden: set[str]) -> str | None:
    """The observed mediator satisfying the front-door criterion, if exactly one does.

    Args:
        summary: The structure's summary graph.
        hidden: Names of its hidden variables.

    Returns:
        The mediator's name, or ``None``.
    """
    on_path = (nx.descendants(summary, "A") & nx.ancestors(summary, "Y")) - hidden
    if len(on_path) != 1:
        return None
    (m,) = on_path
    # M must intercept every directed path from A to Y ...
    if nx.has_path(summary.subgraph(set(summary) - {m}), "A", "Y"):
        return None
    # ... and no hidden variable may reach M other than through A, which leaves
    # A -> M unconfounded and every back-door path from M to Y through A.
    without_a = summary.subgraph(set(summary) - {"A"})
    if any(nx.has_path(without_a, h, m) for h in hidden):
        return None
    return m


def _instrument(summary: nx.DiGraph, hidden: set[str]) -> str | None:
    """An observed instrument for ``A``: relevant, excluded and unconfounded.

    Args:
        summary: The structure's summary graph.
        hidden: Names of its hidden variables.

    Returns:
        The first such variable by name, or ``None``.
    """
    without_a = summary.subgraph(set(summary) - {"A"})
    for z in sorted(set(summary) - hidden - {"A", "Y"}):
        relevant = nx.has_path(summary, z, "A")
        excluded = not nx.has_path(without_a, z, "Y")
        unconfounded = not any(nx.has_path(summary, h, z) for h in hidden)
        if relevant and excluded and unconfounded:
            return z
    return None


@functools.cache
def route(structure: str) -> Route:
    """Pick the estimator a structure's DAG licenses.

    The rules, in order: without a hidden confounder of ``A`` and ``Y``, the
    back-door criterion holds with the DAG's adjustment set (``BackDoorOLS``),
    or with the empty set (unadjusted, ``NaiveOLS``). With one, ``A`` not being
    an ancestor of ``Y`` gives ``tau = 0`` by do-calculus rule 3; otherwise a
    front-door mediator gives ``FrontDoorOLS`` and an instrument ``IV2SLS``.
    If none exists the effect is not identified and the router reports the
    unadjusted ``NaiveOLS`` estimate, labelled as such.

    Args:
        structure: A :class:`~dotime.tscm_sampler.TSCMStructure` value.

    Returns:
        The :class:`Route`.

    Raises:
        ValueError: If ``structure`` is not a named structure.
    """
    _names, summary, hidden = _canonical_summary_graph(structure)
    if not _hidden_confounders(summary, hidden):
        if _back_door_columns(structure)[3]:
            return Route("BackDoorOLS", "back-door adjustment", True)
        return Route("NaiveOLS", "unadjusted (no confounding)", True)
    if not nx.has_path(summary, "A", "Y"):
        return Route(None, "tau = 0 (rule 3: A is not an ancestor of Y)", True)
    if _front_door_mediator(summary, hidden) is not None:
        return Route("FrontDoorOLS", "front-door adjustment", True)
    if _instrument(summary, hidden) is not None:
        return Route("IV2SLS", "instrumental variable", True)
    return Route("NaiveOLS", "no identification (unadjusted estimate)", False)


class GraphRouter:
    """Predict with the estimator the structure's DAG calls for (:func:`route`).

    Under ``tau = 0`` it predicts the pre-onset outcome mean (the package
    ``Mean`` baseline) at every do-value, so its predicted effect is exactly
    zero. A route to an estimator the package does not register yet is pending.

    Args:
        estimators: The analysis estimators by name, ``None`` where pending.
    """

    name = "GraphRouter"

    def __init__(self, estimators: Mapping[str, Estimator | None]) -> None:
        self._estimators = estimators
        self._no_effect = PackageEstimator("Mean")

    def _routed(self, structure: str | None) -> Estimator | None:
        """The routed estimator, ``None`` under ``tau = 0``.

        Args:
            structure: The episode's structure label.

        Returns:
            The estimator instance.

        Raises:
            ValueError: If the episode has no structure label.
            PendingEstimatorError: If the route needs an unregistered estimator.
        """
        if structure is None:
            raise ValueError("GraphRouter needs the episode's structure")
        name = route(structure).estimator
        if name is None:
            return None
        est = self._estimators.get(name)
        if est is None:
            raise PendingEstimatorError(f"{structure!r} routes to {name}, which is pending")
        return est

    def is_pending(self, structure: str | None) -> bool:
        """Whether the route for ``structure`` needs an unregistered estimator.

        Args:
            structure: The episode's structure label.

        Returns:
            ``True`` if predictions on this structure are pending.
        """
        try:
            self._routed(structure)
        except PendingEstimatorError:
            return True
        return False

    def uses_do_value(self, structure: str | None) -> bool:
        """Whether the routed estimator reads the do-value.

        Args:
            structure: The episode's structure label.

        Returns:
            ``False`` under ``tau = 0``, otherwise the routed estimator's answer.
        """
        est = self._routed(structure)
        return est is not None and est.uses_do_value(structure)

    def predict(self, episode: Episode, value: float) -> float:
        """Predict with the routed estimator.

        Args:
            episode: The episode to predict.
            value: The do-value.

        Returns:
            The routed estimator's prediction, or the pre-onset outcome mean
            under ``tau = 0``.
        """
        est = self._routed(episode.structure)
        if est is None:
            return self._no_effect.predict(episode, do_value(episode))
        return est.predict(episode, value)


# --------------------------------------------------------------------------- #
# Registry of the analysis
# --------------------------------------------------------------------------- #


def build_estimators(names: Sequence[str] = ESTIMATOR_ORDER) -> dict[str, Estimator | None]:
    """Instantiate the analysis estimators, ``None`` for those still pending.

    Args:
        names: Estimators to build, a subset of :data:`ESTIMATOR_ORDER`. The
            router routes only among the other estimators built here.

    Returns:
        ``{name: estimator}`` in the order of :data:`ESTIMATOR_ORDER`, with
        ``None`` for a package estimator that is not registered yet.

    Raises:
        ValueError: If a name is not an estimator of this analysis.
    """
    unknown = [n for n in names if n not in ESTIMATOR_ORDER]
    if unknown:
        raise ValueError(f"unknown estimators {unknown}; choose from {list(ESTIMATOR_ORDER)}")
    registered = set(baselines.available())
    out: dict[str, Estimator | None] = {}
    for name in ESTIMATOR_ORDER:
        if name not in names or name == "GraphRouter":
            continue
        if name == "do-SVAR":
            out[name] = DoSVAR()
        elif name == "FrontDoorOLS":
            out[name] = FrontDoorOLS()
        elif name == "Oracle":
            out[name] = OracleEstimator()
        else:
            out[name] = PackageEstimator(name) if name in registered else None
    if "GraphRouter" in names:
        routed = dict(out)
        # The router needs every estimator it may route to, whether or not the
        # caller asked for that estimator's own row.
        for name in ("BackDoorOLS", "IV2SLS", "FrontDoorOLS", "NaiveOLS"):
            if name not in routed:
                routed[name] = (
                    FrontDoorOLS()
                    if name == "FrontDoorOLS"
                    else (PackageEstimator(name) if name in registered else None)
                )
        out["GraphRouter"] = GraphRouter(routed)
    return {name: out[name] for name in ESTIMATOR_ORDER if name in out}
