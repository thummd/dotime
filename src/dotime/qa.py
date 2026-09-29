"""Per-arm target QA: log and assert target statistics before any number is trusted.

Seed protocols guard against variance. They do not guard against a
systematically corrupted target: the v1 observational training arm was all
zeros and passed every seed check. So every suite build, benchmark run and
training loader records what it is about to score or fit, per arm and per
structure, and refuses to continue when an arm is degenerate.

The three arms of a query are

* ``y_obs_level``: the observational level of the queried variable at the
  query row (:func:`dotime.evaluation.query_obs_levels` by default),
* ``y_int_level``: the interventional or counterfactual level, ``y_true``,
* ``effect``: ``y_int_level - y_obs_level``.

Both level arms must be finite, have positive variance and be nonzero on at
least half of the queries (:class:`QAThresholds`). The effect arm is checked
when the run scores or trains on it: it must be nonzero on at least 5% of the
queries that can carry an effect. A query cannot carry one when its
structure's DAG has no directed path from the treatment ``A`` to the outcome
``Y`` (``observed_confounder`` and ``unobserved_confounder``), or when the query
comes fewer steps after the onset than the shortest such path is long
(``mediator`` at offset 0). :func:`is_null_effect` reads this off the named
structure's temporal DAG rather than a hand-kept table.

:func:`target_qa` checks episodes (suites and evaluator inputs) and
:func:`batch_target_qa` checks training batches. Both return a JSON-able
:class:`QAReport` and raise :class:`TargetQAError` on failure unless told not
to. They only read tensors, so they never change a random stream.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dotime.evaluation import DEFAULT_DIR_TARGET, DIR_TARGETS

if TYPE_CHECKING:
    from dotime.benchmarks import Episode

__all__ = [
    "ARMS",
    "QAReport",
    "QAThresholds",
    "TargetQAError",
    "arm_stats",
    "batch_target_qa",
    "effect_lag",
    "is_null_effect",
    "target_qa",
]

#: Names of the three target arms, in reporting order.
ARMS = ("y_obs_level", "y_int_level", "effect")

_LEVEL_ARMS = ("y_obs_level", "y_int_level")


class TargetQAError(RuntimeError):
    """Raised when targets fail :func:`target_qa` or :func:`batch_target_qa`.

    Args:
        message: What failed, one problem per line.
        report: The full :class:`QAReport`, for callers that record it. It is
            optional so that the exception survives pickling, which rebuilds
            it from the message alone.
    """

    def __init__(self, message: str, report: QAReport | None = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class QAThresholds:
    """Floors that target arms must clear.

    Finiteness and a positive variance of both level arms are always required.

    Args:
        min_level_nonzero_frac: Minimum fraction of nonzero values in each
            level arm. A mostly zero level arm is a masked or diverged target,
            not data.
        min_effect_nonzero_frac: Minimum fraction of nonzero effects among the
            queries that can carry an effect, when the effect is checked.
        min_group_n: Groups (structures) with fewer queries are reported but
            not asserted, since their fractions are too noisy to fail a run on.
        min_pooled_n: The pooled arms are asserted from this many queries on.
            Two is the smallest count with a meaningful variance.
    """

    min_level_nonzero_frac: float = 0.5
    min_effect_nonzero_frac: float = 0.05
    min_group_n: int = 10
    min_pooled_n: int = 2


@dataclass
class QAReport:
    """Outcome of one target QA run.

    Args:
        passed: Whether every asserted scope cleared every threshold.
        problems: One message per failed check of an asserted scope.
        pooled: Statistics of all queries: ``n``, ``asserted``, one
            :func:`arm_stats` dict per arm in :data:`ARMS`,
            ``n_null_effect_exempt`` and, when the effect is checked,
            ``effect_checked`` (the effect arm without exempt queries).
        groups: The same statistics per group label, e.g. per structure.
        notes: Failed checks of scopes too small to assert.
        dir_target: ``"level"`` or ``"effect"``, the target the run scores.
        check_effect: Whether the effect arm was asserted.
        thresholds: The thresholds used.
        label: Optional name of what was checked, e.g. a structure.
    """

    passed: bool
    problems: list[str]
    pooled: dict[str, Any]
    groups: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    dir_target: str = DEFAULT_DIR_TARGET
    check_effect: bool = False
    thresholds: QAThresholds = field(default_factory=QAThresholds)
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-able view of the report, safe for ``json.dumps(allow_nan=False)``.

        Returns:
            Dict with ``passed``, ``problems``, ``notes``, ``dir_target``,
            ``check_effect``, ``thresholds``, ``label``, ``pooled`` and ``groups``.
        """
        return {
            "passed": self.passed,
            "problems": list(self.problems),
            "notes": list(self.notes),
            "dir_target": self.dir_target,
            "check_effect": self.check_effect,
            "thresholds": asdict(self.thresholds),
            "label": self.label,
            "pooled": self.pooled,
            "groups": self.groups,
        }


def arm_stats(values: Any) -> dict[str, Any]:
    """Summarise one target arm.

    Args:
        values: Target values, anything ``numpy.asarray`` accepts.

    Returns:
        Dict with ``n`` (all values), ``n_nonfinite`` (NaN or infinite
        values) and, over the finite values, ``nonzero_frac``, ``mean``,
        ``var`` (population variance) and ``abs_max``. Those four are ``None``
        when no value is finite, so the dict stays valid JSON.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    out: dict[str, Any] = {"n": int(x.size), "n_nonfinite": int(x.size - finite.size)}
    if finite.size == 0:
        return {**out, "nonzero_frac": None, "mean": None, "var": None, "abs_max": None}
    return {
        **out,
        "nonzero_frac": float(np.mean(finite != 0.0)),
        "mean": float(np.mean(finite)),
        "var": float(np.var(finite)),
        "abs_max": float(np.max(np.abs(finite))),
    }


_UNKNOWN = object()


@functools.cache
def _structure_lag(structure: str) -> Any:
    """Cached :func:`effect_lag`, with unknown names mapped to a sentinel.

    Args:
        structure: A structure name.

    Returns:
        The minimum lag, ``None`` when there is no path, or ``_UNKNOWN``.
    """
    import networkx as nx

    from dotime.tscm_sampler import TSCMSampler, TSCMStructure

    try:
        member = TSCMStructure(structure)
    except ValueError:
        return _UNKNOWN
    dag = TSCMSampler(member)._build_dag()
    graph = nx.DiGraph()
    graph.add_nodes_from(dag.topo_order)
    for u, v in dag.G_0.edges():
        graph.add_edge(u, v, lag=0)
    for k, g_k in enumerate(dag.G_lags):
        # TemporalSCM reads G_k[j, i] > 0 as "topo node j at t - (k + 1) is a
        # parent of topo node i at t"; the same convention is used here.
        for j, i in zip(*np.nonzero(g_k), strict=True):
            u, v = dag.topo_order[int(j)], dag.topo_order[int(i)]
            if u != v and (not graph.has_edge(u, v) or graph[u][v]["lag"] > k + 1):
                graph.add_edge(u, v, lag=k + 1)
    try:
        return int(nx.shortest_path_length(graph, "A", "Y", weight="lag"))
    except nx.NetworkXNoPath:
        return None


def effect_lag(structure: str) -> int | None:
    """Minimum lag of a directed path from the treatment ``A`` to the outcome ``Y``.

    The path runs through the named structure's temporal DAG
    (``TSCMSampler(TSCMStructure(structure))._build_dag()``): an instantaneous
    edge adds no lag and an edge of ``G_lags[k]`` adds ``k + 1`` steps.

    Args:
        structure: A :class:`~dotime.tscm_sampler.TSCMStructure` value, or its
            legacy alias ``"rct_no_confounding"``.

    Returns:
        The smallest number of steps after which an intervention on ``A`` can
        reach ``Y``, or ``None`` when no directed path exists at any lag.

    Raises:
        ValueError: If ``structure`` does not name a structure.
    """
    lag = _structure_lag(structure)
    if lag is _UNKNOWN:
        raise ValueError(f"{structure!r} is not a named TSCM structure")
    return lag


def is_null_effect(structure: str | None, offset: int | None) -> bool:
    """Whether a query cannot carry an effect by construction of its structure.

    Args:
        structure: The episode's structure label, or ``None``.
        offset: Query row minus intervention onset, or ``None`` if unknown.

    Returns:
        ``True`` when the structure has no directed ``A -> Y`` path, or when
        ``offset`` is below the path's minimum lag. Unknown or missing
        structure names (generic and regime episodes) are never exempt, and
        neither is a lagged structure whose offset is unknown.
    """
    if structure is None:
        return False
    lag = _structure_lag(structure)
    if lag is _UNKNOWN:
        return False
    if lag is None:
        return True
    return offset is not None and offset < lag


def _fmt(value: float | None) -> str:
    """Format a statistic for a log line.

    Args:
        value: A statistic, or ``None`` when undefined.

    Returns:
        Four decimals, or ``"n/a"``.
    """
    return "n/a" if value is None else f"{value:.4f}"


def _scope(
    y_obs: np.ndarray, y_int: np.ndarray, effect: np.ndarray, exempt: np.ndarray, check_effect: bool
) -> dict[str, Any]:
    """Statistics of one scope (the pooled queries or one group).

    Args:
        y_obs: Observational levels.
        y_int: Interventional or counterfactual levels.
        effect: Effects, aligned with the levels.
        exempt: Which queries cannot carry an effect.
        check_effect: Whether to add ``effect_checked``.

    Returns:
        The scope dict described in :class:`QAReport`, without ``asserted``.
    """
    out: dict[str, Any] = {
        "n": int(y_int.size),
        "y_obs_level": arm_stats(y_obs),
        "y_int_level": arm_stats(y_int),
        "effect": arm_stats(effect),
        "n_null_effect_exempt": int(exempt.sum()),
    }
    if check_effect:
        out["effect_checked"] = arm_stats(effect[~exempt])
    return out


def _scope_problems(
    name: str, scope: dict[str, Any], check_effect: bool, thresholds: QAThresholds, min_n: int
) -> list[str]:
    """Check one scope against the thresholds.

    Args:
        name: Scope name for the messages, e.g. ``"pooled"``.
        scope: Output of :func:`_scope`.
        check_effect: Whether the effect arm is asserted.
        thresholds: The floors.
        min_n: Queries the effect check needs among non-exempt queries.

    Returns:
        One message per failed check.
    """
    problems = []
    for arm in ARMS:
        st = scope[arm]
        if st["n_nonfinite"]:
            problems.append(f"{name} {arm}: {st['n_nonfinite']} of {st['n']} values are not finite")
    for arm in _LEVEL_ARMS:
        st = scope[arm]
        if st["var"] is None or st["var"] <= 0.0:
            problems.append(f"{name} {arm}: variance {_fmt(st['var'])} is not positive")
        if st["nonzero_frac"] is None or st["nonzero_frac"] < thresholds.min_level_nonzero_frac:
            problems.append(
                f"{name} {arm}: nonzero_frac {_fmt(st['nonzero_frac'])} < "
                f"{thresholds.min_level_nonzero_frac}"
            )
    if check_effect:
        st = scope["effect_checked"]
        if st["n"] >= min_n and (
            st["nonzero_frac"] is None or st["nonzero_frac"] < thresholds.min_effect_nonzero_frac
        ):
            problems.append(
                f"{name} effect: nonzero_frac {_fmt(st['nonzero_frac'])} < "
                f"{thresholds.min_effect_nonzero_frac} over {st['n']} queries that can "
                "carry an effect"
            )
    return problems


def _log_scope(log: Callable[[str], None], name: str, scope: dict[str, Any]) -> None:
    """Write one line per arm of a scope.

    Args:
        log: Line sink.
        name: Scope name.
        scope: Output of :func:`_scope` with ``asserted``.
    """
    tag = "" if scope["asserted"] else " (reported, not asserted)"
    for arm in ARMS:
        st = scope[arm]
        log(
            f"[target QA] {name:24s} {arm:11s} n={st['n']} nonfinite={st['n_nonfinite']} "
            f"nonzero_frac={_fmt(st['nonzero_frac'])} mean={_fmt(st['mean'])} "
            f"var={_fmt(st['var'])} abs_max={_fmt(st['abs_max'])}{tag}"
        )
    if scope["n_null_effect_exempt"]:
        log(
            f"[target QA] {name:24s} {scope['n_null_effect_exempt']} of {scope['n']} queries "
            "cannot carry an effect by structure (exempt from the effect check)"
        )
    if "effect_checked" in scope:
        st = scope["effect_checked"]
        log(
            f"[target QA] {name:24s} effect over the {st['n']} queries that can carry one: "
            f"nonzero_frac={_fmt(st['nonzero_frac'])}{tag}"
        )


def _run(
    y_obs: np.ndarray,
    y_int: np.ndarray,
    effect: np.ndarray,
    exempt: np.ndarray,
    groups: Sequence[str | None],
    *,
    dir_target: str,
    check_effect: bool,
    thresholds: QAThresholds,
    raise_on_failure: bool,
    log: Callable[[str], None] | None,
    label: str | None,
) -> QAReport:
    """Assemble, log and assert a report from flat per-query arrays.

    Args:
        y_obs: Observational levels, one per query.
        y_int: Interventional or counterfactual levels.
        effect: Effects.
        exempt: Which queries cannot carry an effect.
        groups: Group label per query, ``None`` for pooled-only queries.
        dir_target: Target the run scores, recorded in the report.
        check_effect: Whether the effect arm is asserted.
        thresholds: The floors.
        raise_on_failure: Raise :class:`TargetQAError` if a check fails.
        log: Line sink, or ``None`` for silence.
        label: Name recorded in the report.

    Returns:
        The report.

    Raises:
        TargetQAError: If a check fails and ``raise_on_failure`` is set.
    """
    # A labelled run (one training structure) names its pooled scope after it.
    pooled_name = label or "pooled"
    pooled = _scope(y_obs, y_int, effect, exempt, check_effect)
    pooled["asserted"] = pooled["n"] >= thresholds.min_pooled_n
    problems: list[str] = []
    notes: list[str] = []
    if pooled["n"] == 0:
        problems.append("no queries to check")
    found = _scope_problems(pooled_name, pooled, check_effect, thresholds, thresholds.min_pooled_n)
    (problems if pooled["asserted"] else notes).extend(found)
    labels = np.asarray([str(g) if g is not None else "" for g in groups], dtype=object)
    group_stats: dict[str, dict[str, Any]] = {}
    for name in sorted({str(g) for g in groups if g is not None}):
        m = labels == name
        scope = _scope(y_obs[m], y_int[m], effect[m], exempt[m], check_effect)
        scope["asserted"] = scope["n"] >= thresholds.min_group_n
        found = _scope_problems(name, scope, check_effect, thresholds, thresholds.min_group_n)
        (problems if scope["asserted"] else notes).extend(found)
        group_stats[name] = scope
    report = QAReport(
        passed=not problems,
        problems=problems,
        pooled=pooled,
        groups=group_stats,
        notes=notes,
        dir_target=dir_target,
        check_effect=check_effect,
        thresholds=thresholds,
        label=label,
    )
    if log is not None:
        _log_scope(log, pooled_name, pooled)
        for name, scope in group_stats.items():
            _log_scope(log, name, scope)
        for note in notes:
            log(f"[target QA] note: {note}")
        verdict = "passed" if report.passed else "FAILED: " + "; ".join(problems)
        log(
            f"[target QA] {verdict} (dir_target={dir_target}, effect checked={check_effect}, "
            f"{sum(s['asserted'] for s in group_stats.values())} of {len(group_stats)} groups "
            "asserted)"
        )
    if problems and raise_on_failure:
        raise TargetQAError("target QA failed:\n  " + "\n  ".join(problems), report)
    return report


def _group_key(ep: Episode, group_by: str | Callable[[Episode], Any] | None) -> str | None:
    """Group label of an episode.

    Args:
        ep: The episode.
        group_by: ``None``, a callable, or an ``Episode`` attribute or
            metadata key such as ``"structure"`` or ``"tier"``.

    Returns:
        The label as a string, or ``None`` when the episode has none.
    """
    if group_by is None:
        return None
    if callable(group_by):
        value = group_by(ep)
    elif hasattr(ep, group_by):
        value = getattr(ep, group_by)
    else:
        value = ep.metadata.get(group_by)
    return None if value is None else str(value)


def _query_offsets(ep: Episode) -> list[int | None]:
    """Rows between the intervention onset and each query of an episode.

    Args:
        ep: The episode.

    Returns:
        ``query row - onset`` per query, ``None`` when the intervention
        records no time.
    """
    rows = [int(r) for r in ep.query_time_idx.tolist()]
    if not ep.intervention.times:
        return [None] * len(rows)
    onset = min(int(t) for t in ep.intervention.times)
    return [r - onset for r in rows]


def target_qa(
    episodes: Iterable[Episode],
    *,
    obs_levels: Sequence[Any] | None = None,
    dir_target: str = DEFAULT_DIR_TARGET,
    group_by: str | Callable[[Episode], Any] | None = "structure",
    thresholds: QAThresholds | None = None,
    raise_on_failure: bool = True,
    log: Callable[[str], None] | None = print,
) -> QAReport:
    """Log and assert per-arm target statistics of episodes, pooled and per group.

    Args:
        episodes: The episodes about to be written, scored or trained on.
        obs_levels: Observational level(s) of each episode's queries, aligned
            with ``episodes``, e.g. from the Identifiability 1.0.0 realignment
            sidecar. By default :func:`dotime.evaluation.query_obs_levels`.
        dir_target: ``"level"``, or ``"effect"`` to also assert the effect arm
            on the queries that can carry an effect (see :func:`is_null_effect`).
        group_by: Group key, ``"structure"`` by default. Episodes without a
            label only enter the pooled statistics. ``None`` disables groups.
        thresholds: Floors, :class:`QAThresholds` defaults if ``None``.
        raise_on_failure: Raise instead of only reporting a failure.
        log: Line sink for the statistics, ``print`` by default, ``None`` for
            silence.

    Returns:
        The :class:`QAReport`.

    Raises:
        ValueError: If ``dir_target`` is unknown, or ``obs_levels`` does not
            match the episodes or their query counts.
        TargetQAError: If a check fails and ``raise_on_failure`` is set.
    """
    from dotime.evaluation import query_obs_levels

    if dir_target not in DIR_TARGETS:
        raise ValueError(f"dir_target must be one of {DIR_TARGETS}, got {dir_target!r}")
    episodes = list(episodes)
    if obs_levels is not None and len(obs_levels) != len(episodes):
        raise ValueError(f"{len(obs_levels)} obs_levels for {len(episodes)} episodes")
    y_obs_parts, y_int_parts, exempt, groups = [], [], [], []
    for i, ep in enumerate(episodes):
        y_int = torch.as_tensor(ep.y_true, dtype=torch.float64).reshape(-1).numpy()
        raw = query_obs_levels(ep) if obs_levels is None else obs_levels[i]
        y_obs = np.asarray(torch.as_tensor(raw).double().reshape(-1))
        if y_obs.size != y_int.size:
            raise ValueError(
                f"episode {ep.scm_id}: {y_obs.size} observational levels for {y_int.size} queries"
            )
        y_obs_parts.append(y_obs)
        y_int_parts.append(y_int)
        exempt += [is_null_effect(ep.structure, off) for off in _query_offsets(ep)]
        groups += [_group_key(ep, group_by)] * y_int.size
    y_obs_all = np.concatenate(y_obs_parts) if y_obs_parts else np.zeros(0)
    y_int_all = np.concatenate(y_int_parts) if y_int_parts else np.zeros(0)
    return _run(
        y_obs_all,
        y_int_all,
        y_int_all - y_obs_all,
        np.asarray(exempt, dtype=bool),
        groups,
        dir_target=dir_target,
        check_effect=dir_target == "effect",
        thresholds=thresholds or QAThresholds(),
        raise_on_failure=raise_on_failure,
        log=log,
        label=None,
    )


def _batch_arms(batch: Mapping[str, Any], structure: str | None) -> tuple[np.ndarray, ...]:
    """Flat target arms and exemptions of one generated batch.

    Args:
        batch: A batch from ``ExtendedDoTime.generate_batch``.
        structure: The structure that generated it, or ``None``.

    Returns:
        ``(y_obs, y_int, effect, exempt)`` per query. The effect is the
        batch's own ``Y_causal_effect`` when present, since that is what a
        model trained on the effect fits.

    Raises:
        KeyError: If the batch lacks ``Y_true`` or ``Y_obs``.
    """
    y_int = torch.as_tensor(batch["Y_true"]).detach().double().reshape(-1).cpu().numpy()
    y_obs = torch.as_tensor(batch["Y_obs"]).detach().double().reshape(-1).cpu().numpy()
    if "Y_causal_effect" in batch:
        effect = torch.as_tensor(batch["Y_causal_effect"]).detach().double().reshape(-1)
        effect_np = effect.cpu().numpy()
    else:
        effect_np = y_int - y_obs
    offsets: list[int | None] = [None] * y_int.size
    if structure is not None and {"query_time", "int_onset_idx", "X_obs"} <= set(batch):
        t_len = int(batch["X_obs"].shape[1])
        rows = torch.round(torch.as_tensor(batch["query_time"]).double().reshape(-1) * t_len)
        onset = torch.as_tensor(batch["int_onset_idx"]).reshape(-1)
        if "_traj_idx" in batch:
            onset = onset[torch.as_tensor(batch["_traj_idx"]).reshape(-1)]
        offsets = [int(r) - int(o) for r, o in zip(rows.tolist(), onset.tolist(), strict=True)]
    exempt = np.asarray([is_null_effect(structure, off) for off in offsets], dtype=bool)
    return y_obs, y_int, effect_np, exempt


def batch_target_qa(
    batches: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    structure: str | None = None,
    target_key: str = "Y_true",
    thresholds: QAThresholds | None = None,
    raise_on_failure: bool = True,
    log: Callable[[str], None] | None = print,
) -> QAReport:
    """Log and assert per-arm target statistics of generated training batches.

    Reads the raw targets ``Y_obs``, ``Y_true`` and ``Y_causal_effect`` (not the
    normalized ones), so it checks what the generator produced.

    Args:
        batches: One batch dict or several, all from one structure.
        structure: The ``tscm_structure`` that generated them, ``None`` for the
            generic prior. It labels the report and decides exemptions.
        target_key: The key the model is trained on. ``"Y_causal_effect"``
            also asserts the effect arm, except on queries that cannot carry
            an effect.
        thresholds: Floors, :class:`QAThresholds` defaults if ``None``.
        raise_on_failure: Raise instead of only reporting a failure.
        log: Line sink, ``print`` by default, ``None`` for silence.

    Returns:
        The :class:`QAReport`, with the statistics under ``pooled``.

    Raises:
        KeyError: If a batch lacks ``Y_true`` or ``Y_obs``.
        TargetQAError: If a check fails and ``raise_on_failure`` is set.
    """
    if isinstance(batches, Mapping):
        batches = [batches]
    parts = [_batch_arms(b, structure) for b in batches]
    y_obs, y_int, effect, exempt = (
        np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in range(4)
    )
    check_effect = target_key == "Y_causal_effect"
    return _run(
        y_obs,
        y_int,
        effect,
        exempt.astype(bool),
        [None] * y_int.size,
        dir_target="effect" if check_effect else "level",
        check_effect=check_effect,
        thresholds=thresholds or QAThresholds(),
        raise_on_failure=raise_on_failure,
        log=log,
        label=structure or "generic prior",
    )
