"""Reference baselines for DoTime benchmark suites.

A small registry maps baseline *names* to constructors, so the CLI and the
evaluation harness can request a baseline by string (mirroring the
``BASELINE_STRING_TO_CLASS`` table in the original ``tscm_identifiability.py``).

**Public surface**

- :class:`Baseline`     — the predict interface every baseline implements.
- :func:`available`     — list registered baseline names.
- :func:`get`           — instantiate a baseline by name.
- :func:`register`      — decorator to add a baseline to the registry.

Implemented: the trivial baselines (``Zero``, ``Mean``/TrajMean, ``AR1``,
``VAR-OLS``), the classical structural baselines (``BackDoorOLS``, ``IV2SLS``),
the unadjusted ``NaiveOLS`` (``BackDoorOLS`` with an empty adjustment set),
``Oracle`` (stored ground truth), and ``DoOverTimePFN`` (checkpoint-backed, the
``[models]`` extra). ``PCMCI+`` / ``BayesianITS`` / ``Chronos`` require the
``[baselines]`` extra and raise an actionable error until that dependency and
their wiring are present.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import networkx as nx
import numpy as np
import torch

if TYPE_CHECKING:
    from dotime.benchmarks import Episode

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


def _pre_onset_index(episode: Episode) -> int:
    """First post-intervention step (onset); falls back to the full length."""
    times = episode.intervention.times
    return min(times) if times else episode.x_obs.shape[0]


@register("Mean")
class MeanBaseline:
    """Predicts the pre-intervention mean of the queried variable (a.k.a. TrajMean)."""

    name = "Mean"

    def predict(self, episode: Episode) -> torch.Tensor:
        onset = _pre_onset_index(episode)
        preds = []
        for q in range(episode.query_target.numel()):
            var = int(episode.query_target[q])
            pre = episode.x_obs[:onset, var]
            preds.append(pre.mean() if pre.numel() else episode.x_obs[:, var].mean())
        return torch.stack(preds)


@register("AR1")
class AR1Baseline:
    """Predicts the last pre-intervention value of the queried variable."""

    name = "AR1"

    def predict(self, episode: Episode) -> torch.Tensor:
        onset = _pre_onset_index(episode)
        preds = []
        for q in range(episode.query_target.numel()):
            var = int(episode.query_target[q])
            last = max(0, min(onset, episode.x_obs.shape[0]) - 1)
            preds.append(episode.x_obs[last, var])
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
        t, _n = x.shape
        mean = x.mean(axis=0)
        xc = x - mean
        rows, targets = [], []
        for s in range(self.lag, t):
            rows.append(xc[s - self.lag : s].reshape(-1))
            targets.append(xc[s])
        a = np.asarray(rows)  # (T-lag, lag*N)
        b = np.asarray(targets)  # (T-lag, N)
        # Ridge-stabilised least squares: coef has shape (N, lag*N).
        gram = a.T @ a + 1e-3 * np.eye(a.shape[1])
        coef = np.linalg.solve(gram, a.T @ b).T
        return coef, mean


# --------------------------------------------------------------------------- #
# Classical structural baselines (correct adjustment per graph)
# --------------------------------------------------------------------------- #


def _ols_fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ridge-stabilised OLS coefficients for ``target ~ [1, design]``."""
    x = np.column_stack([np.ones(len(design)), design])
    gram = x.T @ x + 1e-6 * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ target)


def _canonical_summary_graph(structure: str) -> tuple[list[str], nx.DiGraph, set[str]]:
    """Column names, summary graph and hidden variables of a named structure.

    The graph comes from the structure's definition in
    :mod:`dotime.tscm_sampler` and the column order from
    :class:`~dotime.extended.TSCMPrior`, which lays out the released ``x_obs``
    (treatment ``A`` first, outcome ``Y`` last). The summary graph has an edge
    ``u -> v`` when ``u`` drives ``v`` instantaneously or at some lag, so the
    descendants of ``A`` in it are the variables that ``A`` affects at any lag.

    Args:
        structure: A :class:`~dotime.tscm_sampler.TSCMStructure` value, e.g.
            ``"front_door"``.

    Returns:
        ``(names, summary, hidden)``: the variable name of each canonical
        column, the summary graph over those names, and the names of the
        hidden variables.

    Raises:
        ValueError: If ``structure`` is not a named structure.
    """
    # Lazy import: only structure-aware adjustment needs the generator modules.
    # Building a TSCMPrior draws no random numbers (its generator is private),
    # so this never perturbs a caller's RNG stream.
    from dotime.extended import TSCMPrior
    from dotime.tscm_sampler import TSCMStructure

    prior = TSCMPrior(TSCMStructure(structure))
    dag = prior.sampler._build_dag()
    topo = list(dag.topo_order)
    summary = nx.DiGraph(dag.G_0)
    for lag in dag.G_lags:
        for i, j in zip(*np.nonzero(lag), strict=True):
            # Self-loops are autoregression, which cannot make another
            # variable a descendant of A.
            if i != j:
                summary.add_edge(topo[i], topo[j])
    names = [topo[t] for t in prior.canonical_perm]
    return names, summary, {topo[h] for h in prior.hidden_vars}


@functools.cache
def _back_door_columns(structure: str) -> tuple[int, int, int, tuple[int, ...]]:
    """Canonical treatment, outcome and back-door adjustment columns of a structure.

    The adjustment set is every observed variable other than ``A`` and ``Y``
    that is not a descendant of ``A`` in the summary graph (see
    :func:`_canonical_summary_graph`). For ``back_door``,
    ``observed_confounder`` and ``confounder_mediator`` that is the confounder
    ``X``. It is a valid back-door set only when no hidden variable confounds
    ``A`` and ``Y``, so callers apply it to the back-door family alone.

    Args:
        structure: A :class:`~dotime.tscm_sampler.TSCMStructure` value, e.g.
            ``"confounder_mediator"``.

    Returns:
        ``(n_vars, treatment_col, outcome_col, adjustment_cols)`` as canonical
        column indices.

    Raises:
        ValueError: If ``structure`` is not a named structure.
    """
    names, summary, hidden = _canonical_summary_graph(structure)
    # The back-door criterion excludes every descendant of A: adjusting for a
    # mediator (M on A -> M -> Y) blocks part of the effect, and adjusting for
    # a collider opens a spurious path.
    excluded = {"A", "Y"} | nx.descendants(summary, "A") | hidden
    adjust = tuple(col for col, var in enumerate(names) if var not in excluded)
    return len(names), names.index("A"), names.index("Y"), adjust


@functools.cache
def _front_door_columns(structure: str) -> tuple[int, int, int, int]:
    """Canonical treatment, outcome and front-door mediator columns of a structure.

    The mediator is the observed variable on the causal path from ``A`` to
    ``Y``: a descendant of ``A`` and an ancestor of ``Y`` in the summary graph
    (see :func:`_canonical_summary_graph`). For ``mediator`` and ``front_door``
    that is ``M``. Column position cannot identify it, because ``front_door``
    lays out its columns as A, U, M, Y with the hidden confounder ``U`` first
    among the middle columns.

    Args:
        structure: A :class:`~dotime.tscm_sampler.TSCMStructure` value, e.g.
            ``"front_door"``.

    Returns:
        ``(n_vars, treatment_col, outcome_col, mediator_col)`` as canonical
        column indices.

    Raises:
        ValueError: If ``structure`` is not a named structure, or if it does
            not have exactly one observed mediator (the front-door estimator
            conditions on a single one).
    """
    names, summary, hidden = _canonical_summary_graph(structure)
    on_path = (nx.descendants(summary, "A") & nx.ancestors(summary, "Y")) - hidden
    mediators = [col for col, var in enumerate(names) if var in on_path]
    if len(mediators) != 1:
        raise ValueError(
            f"structure {structure!r} has observed mediators "
            f"{[names[c] for c in mediators]}, the front-door estimator needs exactly one"
        )
    return len(names), names.index("A"), names.index("Y"), mediators[0]


@register("BackDoorOLS")
class BackDoorOLSBaseline:
    """Linear back-door adjustment: E[Y_t | do(A=v)] = E_X[ E[Y_t | A=v, X, Y_{t-1}] ].

    Fits an OLS outcome model ``Y_t ~ A_t + X_t + Y_{t-1}`` on the pre-intervention
    observational data, then plugs the intervention value for A and averages over
    the observed confounder distribution. ``X`` is the structure's back-door
    adjustment set, read off its DAG: the observed variables other than A and Y
    that are not descendants of A. That is the confounder X for ``back_door``,
    ``observed_confounder`` and ``confounder_mediator``. The mediator M of
    ``confounder_mediator`` lies on the causal path A -> M -> Y, so it is never
    adjusted for. Applicable to the back-door family; on other structures it
    falls back to the pre-intervention outcome mean.
    """

    name = "BackDoorOLS"
    _BACK_DOOR: ClassVar[set[str]] = {"back_door", "observed_confounder", "confounder_mediator"}

    @staticmethod
    def _adjustment_set(structure: str, n: int, a: int, y: int) -> list[int]:
        """Columns to adjust for when estimating the effect of column ``a`` on ``y``.

        Args:
            structure: The episode's structure, a member of the back-door family.
            n: Number of columns in the episode's ``x_obs``.
            a: Treatment column (the intervention target).
            y: Queried column.

        Returns:
            Column indices of the adjustment set.

        Raises:
            ValueError: If ``n`` or ``a`` disagree with the structure's canonical
                column layout, in which case no adjustment set can be trusted.
        """
        n_vars, treatment, outcome, back_door = _back_door_columns(structure)
        if n != n_vars or a != treatment:
            raise ValueError(
                f"BackDoorOLS: a {structure!r} episode needs {n_vars} canonical columns "
                f"with the treatment in column {treatment}; got {n} columns and treatment {a}"
            )
        if y == outcome:
            return list(back_door)
        # Only dot-Continuous-v1 queries a variable other than the outcome (the
        # treatment itself or the confounder). The back-door set is defined for
        # the outcome, so these queries keep the behaviour behind the published
        # Continuous rows: adjust for every other column.
        return [v for v in range(n) if v not in (a, y)]

    def predict(self, episode: Episode) -> torch.Tensor:
        """Predict each query's interventional level by back-door adjustment.

        Args:
            episode: Episode to predict. Back-door-family episodes must use the
                canonical column layout of the released suites.

        Returns:
            1-D float tensor with one prediction per query.

        Raises:
            ValueError: If a back-door-family episode does not match its
                structure's canonical column layout.
        """
        x = episode.x_obs.detach().cpu().numpy()
        t_len, n = x.shape
        a = episode.intervention.targets[0] if episode.intervention.targets else 0
        onset = min(episode.intervention.times) if episode.intervention.times else t_len
        preds = []
        for q in range(episode.query_target.numel()):
            y = int(episode.query_target[q])
            fit_end = max(2, min(onset, t_len))
            if episode.structure not in self._BACK_DOOR or fit_end < 4:
                preds.append(float(x[:fit_end, y].mean()))
                continue
            adj = self._adjustment_set(episode.structure, n, a, y)
            # Design over t in [1, fit_end): [A_t, X_t..., Y_{t-1}] -> Y_t
            a_t = x[1:fit_end, a]
            x_t = x[1:fit_end, adj] if adj else np.empty((fit_end - 1, 0))
            y_prev = x[0 : fit_end - 1, y]
            design = np.column_stack([a_t, x_t, y_prev])
            coef = _ols_fit(design, x[1:fit_end, y])
            # Predict at do(A = intervention value), averaging over observed X rows.
            a_val = (
                float(episode.intervention.values)
                if isinstance(episode.intervention.values, (int, float))
                else float(a_t.mean())
            )
            x_rows = x[1:fit_end, adj] if adj else np.empty((fit_end - 1, 0))
            yhat = (
                coef[0]
                + coef[1] * a_val
                + (x_rows @ coef[2 : 2 + len(adj)] if adj else 0.0)
                + coef[2 + len(adj)] * y_prev
            )
            preds.append(float(np.mean(yhat)))
        return torch.tensor(preds, dtype=torch.float32)


@register("IV2SLS")
class IV2SLSBaseline:
    """Two-stage least squares for the instrumental-variable structure.

    Stage 1 regresses the treatment on the instrument(s) ``Z``; stage 2 regresses
    the outcome on the fitted treatment. The intervention effect is the stage-2
    treatment coefficient: ``E[Y | do(A=v)] = beta_0 + beta_A * v``. On non-IV
    structures it falls back to the pre-intervention outcome mean.
    """

    name = "IV2SLS"

    def predict(self, episode: Episode) -> torch.Tensor:
        x = episode.x_obs.detach().cpu().numpy()
        t_len, n = x.shape
        a = episode.intervention.targets[0] if episode.intervention.targets else 0
        onset = min(episode.intervention.times) if episode.intervention.times else t_len
        fit_end = max(2, min(onset, t_len))
        preds = []
        for q in range(episode.query_target.numel()):
            y = int(episode.query_target[q])
            instruments = [v for v in range(n) if v not in (a, y)]
            if episode.structure != "instrumental_variable" or fit_end < 4 or not instruments:
                preds.append(float(x[:fit_end, y].mean()))
                continue
            z = x[:fit_end, instruments]
            a_obs = x[:fit_end, a]
            y_obs = x[:fit_end, y]
            # Stage 1: A ~ Z ; Stage 2: Y ~ Ahat -> slope is the IV effect estimate.
            s1 = _ols_fit(z, a_obs)
            a_hat = s1[0] + z @ s1[1:]
            # Weak-instrument guard: 2SLS is unreliable when Z explains little of A.
            denom = float(np.var(a_obs))
            stage1_r2 = float(np.var(a_hat)) / denom if denom > 1e-8 else 0.0
            if stage1_r2 < 0.1:
                preds.append(float(y_obs.mean()))
                continue
            s2 = _ols_fit(a_hat.reshape(-1, 1), y_obs)
            beta_a = s2[1]
            a_val = (
                float(episode.intervention.values)
                if isinstance(episode.intervention.values, (int, float))
                else float(a_obs.mean())
            )
            # Centered prediction: baseline outcome + effect of moving A from its
            # observed mean to the intervention value (robust to extrapolation).
            preds.append(float(y_obs.mean() + beta_a * (a_val - a_obs.mean())))
        return torch.tensor(preds, dtype=torch.float32)


@register("NaiveOLS")
class NaiveOLSBaseline:
    """Unadjusted regression: takes E[Y_t | do(A=v)] to be E[ E[Y_t | A=v, Y_{t-1}] ].

    :class:`BackDoorOLSBaseline` with an empty adjustment set. It fits the OLS
    outcome model ``Y_t ~ A_t + Y_{t-1}`` on the pre-intervention observational
    data, plugs in the intervention value for A and averages over the observed
    history. Reading the observational association of A and Y as causal, it
    carries the omitted-variable bias of every confounder of A and Y. It is the
    reference for that bias: on ``back_door`` it is what ``BackDoorOLS`` reports
    without adjusting for X, and on ``bow_graph`` no observed variable blocks
    A <- U -> Y, so no observed adjustment set removes it. It applies to every
    episode, the generic prior's included, with the first intervention target
    as the treatment and the queried variable as the outcome. When the
    pre-intervention window is too short to fit, it falls back to the
    pre-intervention outcome mean.
    """

    name = "NaiveOLS"

    def predict(self, episode: Episode) -> torch.Tensor:
        """Predict each query's interventional level from the unadjusted regression.

        Args:
            episode: Episode to predict. Any structure label, or none.

        Returns:
            1-D float tensor with one prediction per query.

        Raises:
            IndexError: If the intervention target or a query target is not a
                column of ``episode.x_obs``.
        """
        x = episode.x_obs.detach().cpu().numpy()
        t_len = x.shape[0]
        a = episode.intervention.targets[0] if episode.intervention.targets else 0
        onset = min(episode.intervention.times) if episode.intervention.times else t_len
        fit_end = max(2, min(onset, t_len))
        preds = []
        for q in range(episode.query_target.numel()):
            y = int(episode.query_target[q])
            if fit_end < 4:
                preds.append(float(x[:fit_end, y].mean()))
                continue
            # BackDoorOLS's design, fit window, do-value and history average with
            # the adjustment columns dropped, so the two differ only by what
            # adjusting for the back-door set removes.
            a_t = x[1:fit_end, a]
            y_prev = x[0 : fit_end - 1, y]
            coef = _ols_fit(np.column_stack([a_t, y_prev]), x[1:fit_end, y])
            a_val = (
                float(episode.intervention.values)
                if isinstance(episode.intervention.values, (int, float))
                else float(a_t.mean())
            )
            preds.append(float(np.mean(coef[0] + coef[1] * a_val + coef[2] * y_prev)))
        return torch.tensor(preds, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Model-backed baselines (templates — wire to real implementations)
# --------------------------------------------------------------------------- #


@register("Oracle")
class OracleBaseline:
    """Ground-truth interventional outcome. Upper bound on synthetic suites only.

    Reads the released ``y_true`` (the exact interventional level at the query).
    Whether that level is also a counterfactual depends on the generator: the
    continuous suite and the v1.1.0 discrete suites share one noise stream
    across arms, the v1.0.0 discrete suites do not. On suites without a stored
    target this raises rather than guesses.
    """

    name = "Oracle"

    def predict(self, episode: Episode) -> torch.Tensor:
        if "y_oracle" in episode.metadata:
            return torch.as_tensor(episode.metadata["y_oracle"], dtype=torch.float32)
        # y_true is the exact interventional outcome on every synthetic suite.
        if episode.y_true is not None and episode.y_true.numel():
            return episode.y_true.float()
        raise RuntimeError("Oracle baseline requires a stored ground-truth target")


@register("PCMCI+")
class PCMCIBaseline:
    """PCMCI+ causal discovery (tigramite) + linear effect estimate.

    Requires the ``baselines`` extra: ``pip install 'dotime[baselines]'``.
    TODO(consolidate): run PCMCI+ to recover the lagged graph, then estimate the
    interventional effect by linear adjustment on the discovered parents.
    """

    name = "PCMCI+"

    def __init__(self, lag: int = 3, alpha: float = 0.05):
        try:
            import tigramite  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "PCMCI+ baseline needs the 'baselines' extra: pip install 'dotime[baselines]'"
            ) from exc
        self.lag = lag
        self.alpha = alpha

    def predict(self, episode: Episode) -> torch.Tensor:
        raise NotImplementedError("wire PCMCIBaseline.predict to tigramite + adjustment")


@register("BayesianITS")
class BayesianPiecewiseITSBaseline:
    """Bayesian piecewise interrupted-time-series reference (CausalPy).

    Intended for dot-RegimeSwitch-v1, where a 2-regime episode reduces to a
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
                "Bayesian ITS baseline needs the 'baselines' extra: pip install 'dotime[baselines]'"
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


_INT_TYPE_CODE = {"hard": 0, "soft": 1, "time_varying": 2}


def _episode_to_batch(episode: Episode, n_max: int, device: str) -> dict:
    """Convert a released Episode into the model's normalized, padded batch.

    Mirrors ``ExtendedDoTime.generate_sample``: causal masking (zero
    ``x_obs`` from the intervention onset), per-variable normalization over the
    pre-intervention window, and the intervention/query field encoding. Returns a
    batch of size 1 with the normalization stats so predictions can be mapped back
    to the raw scale.
    """
    from dotime.normalization import normalize_batch

    x_obs = episode.x_obs
    t_len, n = x_obs.shape
    onset = min(episode.intervention.times) if episode.intervention.times else t_len
    int_target = episode.intervention.targets[0] if episode.intervention.targets else 0

    # Causal masking (idempotent if the episode is already masked).
    masked = x_obs.clone()
    masked[onset:] = 0.0

    x_padded = torch.zeros(t_len, n_max)
    x_padded[:, :n] = masked
    var_mask = torch.zeros(n_max)
    var_mask[:n] = 1.0

    raw_value = episode.intervention.values
    raw_value = float(raw_value) if isinstance(raw_value, (int, float)) else 0.0
    pre = x_obs[:onset, int_target] if onset > 0 else x_obs[:, int_target]
    int_value_norm = raw_value / max(float(pre.std().item()) if pre.numel() > 1 else 1.0, 1e-4)

    def _norm_time(v: float) -> float:
        return v if v <= 1.0 else v / t_len

    q_time = float(episode.query_time[0]) if episode.query_time.numel() else float(t_len - 1)
    batch = {
        "X_obs": x_padded.unsqueeze(0).to(device),
        "variable_mask": var_mask.unsqueeze(0).to(device),
        "int_onset_idx": torch.tensor([onset], device=device),
        "intervention_target": torch.tensor([int_target], device=device),
        "intervention_type": torch.tensor(
            [_INT_TYPE_CODE.get(episode.intervention.intervention_type.value, 0)], device=device
        ),
        "intervention_value": torch.tensor([int_value_norm], dtype=torch.float32, device=device),
        "intervention_time_start": torch.tensor(
            [
                _norm_time(
                    float(min(episode.intervention.times) if episode.intervention.times else 0)
                )
            ],
            device=device,
        ),
        "intervention_time_end": torch.tensor(
            [
                _norm_time(
                    float(max(episode.intervention.times) if episode.intervention.times else 0)
                )
            ],
            device=device,
        ),
        "query_target": torch.tensor([int(episode.query_target[0])], device=device),
        "query_time": torch.tensor([_norm_time(q_time)], dtype=torch.float32, device=device),
        "Y_true": episode.y_true[:1].to(device),
    }
    normalize_batch(batch)
    return batch


@register("DoOverTimePFN")
class DoOverTimePFNBaseline:
    """The Do-Over-Time-PFN causal foundation model (the headline method).

    Loads a trained checkpoint (``[models]`` extra) and predicts the raw
    interventional outcome at the query: it builds the model's normalized batch
    from the Episode, runs the model in normalized space, then maps the predicted
    mean back to the raw scale with the query variable's normalization stats.

    Pass a ``checkpoint`` path. NOTE: reproducing the paper's reference numbers
    requires the checkpoint trained for the corresponding suite/structure and a
    matched evaluation protocol (Phase 8 verification); this wiring is the
    inference path, validated to run and produce finite predictions.
    """

    name = "DoOverTimePFN"

    def __init__(self, checkpoint: str | None = None, device: str = "cpu"):
        if checkpoint is None:
            raise ValueError(
                "DoOverTimePFN baseline needs a trained checkpoint: "
                "baselines.get('DoOverTimePFN', checkpoint='/path/to/best.pt')"
            )
        from dotime.models.loader import load_dotpfn

        self.device = device
        self.model = load_dotpfn(checkpoint, device=device)
        self.n_max = int(getattr(self.model, "n_max", 41))

    @torch.no_grad()
    def predict(self, episode: Episode) -> torch.Tensor:
        batch = _episode_to_batch(episode, self.n_max, self.device)
        out = self.model(batch)
        head = getattr(self.model, "quantile_head", None) or getattr(self.model, "bar_head", None)
        if head is None:
            raise RuntimeError("model has neither a quantile_head nor a bar_head")
        pred_norm = head.predict_mean(out).reshape(-1)
        # Map back to the raw scale with the query variable's stats.
        q = int(episode.query_target[0])
        mean = batch["_norm_means"][0, q]
        std = batch["_norm_stds"][0, q]
        return (pred_norm * std + mean).cpu()
