"""Ground-truth lagged causal graphs of generated episodes.

A released episode stores trajectories, an intervention and a query, but not the
graph that produced them, so nothing in a frozen suite says how far the
intervention sits from the queried variable. :class:`LaggedGraph` records that
graph in the episode's own column order, and :func:`path_lag` measures the
treatment-to-query path on it.

**Lag convention.** An edge ``(src, dst, lag)`` means that column ``src`` at step
``t - lag`` is an argument of the mechanism of column ``dst`` at step ``t``. Lag 0
is a same-step (instantaneous) edge. For a :class:`~dotime.TemporalSCM` the edge
``(j, i, k + 1)`` corresponds to ``G_lags[k][j, i] > 0`` in topological
coordinates. Self-edges ``(i, i, lag)`` are autoregressive terms.

**Effective edges.** A mechanism reads a parent only through a weight stored
under that parent's name, so an edge counts as effective exactly when the
child's mechanism has such a weight at that lag. ``edges`` holds the effective
edges. ``reads_parents`` says whether they are all the sampled ones. The
regime-switching SCMs of the v1.0.0 suites are the case where they are not: their
builder renamed each regime's nodes to ``X0..X{N-1}`` but kept the weights under
the old names, so no mechanism reads any parent and the effective graph is empty.
``regime_edges`` still records the graphs each regime sampled, which a build with
``regime_canonical_weights`` makes effective without drawing different random
numbers.

**Continuous time.** The continuous prior advances every variable by an
Euler-Maruyama step that reads the previous observation of its parents, so its
instantaneous and lag-1 parents collapse into one set of lag-1 edges (``time ==
"continuous"``), and the mean reversion of each variable is a lag-1 self-edge.
The lag unit is one observation step of the default single Euler step per
observation.

Every constructor only reads attributes or builds fixed structures, so recording
a graph draws no random numbers and leaves every generator untouched.
"""

from __future__ import annotations

import gzip
import heapq
import json
import operator
from collections import deque
from collections.abc import Iterable
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

__all__ = ["LaggedGraph", "PathLag", "load_graph_sidecar", "path_lag"]

Edge = tuple[int, int, int]

_TIMES = ("discrete", "continuous")


def _names(values: Iterable[Any], field_name: str) -> tuple[str, ...]:
    """Coerce node names to a tuple of plain ``str``.

    Args:
        values: The names.
        field_name: Field being validated, for the error message.

    Returns:
        The names as built-in strings, so they serialize as JSON strings.

    Raises:
        TypeError: If a name is not a string.
    """
    out = []
    for v in values:
        if not isinstance(v, str):
            raise TypeError(f"{field_name} holds a non-string node name {v!r}")
        out.append(str(v))
    return tuple(out)


def _edge_tuple(raw: Iterable[Iterable[Any]], n_nodes: int) -> tuple[Edge, ...]:
    """Validate edges and put them in canonical (sorted, unique) form.

    ``operator.index`` rather than ``int`` so that numpy integers are accepted
    and converted, while a float such as ``1.5`` is refused instead of being
    truncated.

    Args:
        raw: ``(src, dst, lag)`` triples.
        n_nodes: Number of graph nodes, released columns plus latent nodes.

    Returns:
        Sorted, de-duplicated triples of built-in ints.

    Raises:
        TypeError: If an entry is not integral.
        ValueError: If an edge is not a triple, names a node outside the graph
            or has a negative lag.
    """
    edges = set()
    for e in raw:
        triple = tuple(operator.index(x) for x in e)
        if len(triple) != 3:
            raise ValueError(f"edge {e!r} is not a (src, dst, lag) triple")
        src, dst, lag = triple
        if not (0 <= src < n_nodes and 0 <= dst < n_nodes):
            raise ValueError(f"edge {triple} names a node outside the {n_nodes}-node graph")
        if lag < 0:
            raise ValueError(f"edge {triple} has a negative lag")
        edges.add((src, dst, lag))
    return tuple(sorted(edges))


@dataclass(frozen=True)
class LaggedGraph:
    """The lagged causal graph of one episode, in released column order.

    Graph node ``i < n`` is released column ``i``. When a builder releases only
    some SCM nodes, the ones it drops follow as nodes ``n, n + 1, ...``, named in
    ``latent``, so that paths through them still count. The constructor
    normalizes its inputs to built-in types, which keeps :meth:`to_dict` output
    JSON-serializable even when numpy integers were passed in.

    Attributes:
        n: Number of released columns.
        columns: SCM node name of each released column.
        edges: Effective edges ``(src, dst, lag)`` over all graph nodes, sorted.
            For a regime-switching SCM, the union over regimes.
        k_sampled: Maximum lag the prior sampled (``K``). It bounds the lags an
            edge could have, whether or not any edge was drawn at that lag.
        k_eff: Largest lag among ``edges``, self-edges included, or 0 when no
            effective edge is lagged.
        hidden: Released columns that hold no data, for example the zeroed
            hidden confounder ``U`` of the identifiability structures.
        reads_parents: Whether every sampled parent is read by its child's
            mechanism, i.e. whether ``edges`` are all the sampled edges.
        regime_edges: For a regime-switching SCM, the edges each regime's graph
            sampled, whether or not the mechanisms read them. ``None`` otherwise.
        time: ``"discrete"``, or ``"continuous"`` for the continuous prior, whose
            lags count observation steps.
        latent: Names of SCM nodes that are not released columns.

    Raises:
        TypeError: If a field has the wrong type (non-integral index, non-string
            name, non-bool ``reads_parents``).
        ValueError: If the fields are inconsistent: ``n`` differs from the number
            of columns, a name repeats, an edge or hidden index leaves the graph,
            ``k_eff`` is not the largest effective lag, ``k_sampled`` is below it
            or ``time`` is unknown.
    """

    n: int
    columns: tuple[str, ...]
    edges: tuple[Edge, ...]
    k_sampled: int
    k_eff: int
    hidden: tuple[int, ...] = ()
    reads_parents: bool = True
    regime_edges: tuple[tuple[Edge, ...], ...] | None = None
    time: str = "discrete"
    latent: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        columns = _names(self.columns, "columns")
        latent = _names(self.latent, "latent")
        n = operator.index(self.n)
        if n != len(columns):
            raise ValueError(f"n={n} but {len(columns)} column names were given")
        names = columns + latent
        if len(set(names)) != len(names):
            raise ValueError(f"node names repeat: {names}")
        edges = _edge_tuple(self.edges, len(names))
        regime_edges = (
            None
            if self.regime_edges is None
            else tuple(_edge_tuple(r, len(names)) for r in self.regime_edges)
        )
        hidden = tuple(sorted({operator.index(h) for h in self.hidden}))
        if any(not 0 <= h < n for h in hidden):
            raise ValueError(f"hidden columns {hidden} are not all in [0, {n})")
        k_sampled = operator.index(self.k_sampled)
        k_eff = operator.index(self.k_eff)
        largest = max((lag for _, _, lag in edges), default=0)
        if k_eff != largest:
            raise ValueError(f"k_eff={k_eff} but the largest effective lag is {largest}")
        if k_sampled < k_eff:
            raise ValueError(f"k_sampled={k_sampled} is below the effective lag {k_eff}")
        # A strict check: bool("false") is True, and this flag is what the
        # sidecars use to mark the regime SCMs that read no parent.
        if not isinstance(self.reads_parents, bool):
            raise TypeError(f"reads_parents must be a bool, got {self.reads_parents!r}")
        if self.time not in _TIMES:
            raise ValueError(f"time must be one of {_TIMES}, got {self.time!r}")
        normalized = {
            "n": n,
            "columns": columns,
            "edges": edges,
            "k_sampled": k_sampled,
            "k_eff": k_eff,
            "hidden": hidden,
            "regime_edges": regime_edges,
            "latent": latent,
        }
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def n_nodes(self) -> int:
        """Number of graph nodes, released columns plus latent nodes."""
        return self.n + len(self.latent)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view with built-in values only (lists, ints, strs, bools).

        Returns:
            Dict with one key per field, which :meth:`from_dict` reads back.
        """
        return {
            "n": self.n,
            "columns": list(self.columns),
            "latent": list(self.latent),
            "edges": [list(e) for e in self.edges],
            "hidden": list(self.hidden),
            "k_sampled": self.k_sampled,
            "k_eff": self.k_eff,
            "reads_parents": self.reads_parents,
            "regime_edges": (
                None
                if self.regime_edges is None
                else [[list(e) for e in regime] for regime in self.regime_edges]
            ),
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LaggedGraph:
        """Rebuild a graph from :meth:`to_dict` output.

        Keys that are not fields, such as the ``"path"`` entry that
        ``record_graph`` stores next to the graph, are ignored, so an episode's
        ``metadata["graph"]`` can be passed as is.

        Args:
            d: The dict.

        Returns:
            The graph.

        Raises:
            KeyError: If a field without a default is missing.
            TypeError: If a value has the wrong type (see the class).
            ValueError: If the values are inconsistent (see the class).
        """
        kwargs = {}
        for f in fields(cls):
            if f.name in d:
                kwargs[f.name] = d[f.name]
            elif f.default is MISSING:
                raise KeyError(f"graph dict has no {f.name!r} entry")
        return cls(**kwargs)

    @classmethod
    def from_scm(
        cls,
        scm: Any,
        columns: Iterable[str] | None = None,
        hidden: Iterable[int] = (),
    ) -> LaggedGraph:
        """Extract the graph of a sampled discrete-time SCM.

        Reads the parent lists the simulator hands to each mechanism and checks,
        per parent and lag, whether the mechanism holds a weight under that
        parent's name. That is the rule ``TemporalMechanism.forward`` applies, so
        the result is the graph the simulation actually used.

        Args:
            scm: A :class:`~dotime.TemporalSCM` (diverse, chain or named-structure
                SCMs) or a :class:`~dotime.RegimeSwitchingTemporalSCM`.
            columns: SCM node name of each released column, in released order.
                ``None`` means the generator released every node in the SCM's
                topological order, as the generic and regime generators do. SCM
                nodes left out become latent nodes after the released ones.
            hidden: Released columns that hold no data.

        Returns:
            The graph, with ``regime_edges`` set for a regime-switching SCM.

        Raises:
            TypeError: If ``scm`` is neither SCM type.
            ValueError: If ``columns`` repeats a name or names a node the SCM
                does not have, or ``hidden`` leaves the released columns.
        """
        from dotime.regime_switching import RegimeSwitchingTemporalSCM
        from dotime.temporal_scm import TemporalSCM

        # (topo, instant parents, lagged parents, mechanisms) per simulated
        # system. Any, because the SCM modules store these untyped.
        systems: list[tuple[Any, Any, Any, Any]]
        if isinstance(scm, RegimeSwitchingTemporalSCM):
            names = list(scm.dags[0].topo_order)
            systems = [
                (parents["topo"], parents["instant"], parents["lagged"], scm.mechanisms[r])
                for r, parents in enumerate(scm._regime_parents)
            ]
            k_sampled = max(dag.K for dag in scm.dags)
        elif isinstance(scm, TemporalSCM):
            names = list(scm._topo)
            systems = [(names, scm._instant_parents, scm._lagged_parents, scm.mechanisms)]
            k_sampled = int(scm._K)
        else:
            raise TypeError(f"cannot extract a graph from {type(scm).__name__}")

        released = tuple(names) if columns is None else tuple(columns)
        if not set(released) <= set(names) or len(set(released)) != len(released):
            raise ValueError(f"columns {released} must be distinct nodes of the SCM {names}")
        latent = tuple(v for v in names if v not in set(released))
        index = {v: i for i, v in enumerate(released + latent)}

        per_system = [
            _system_edges(topo, instant, lagged, mechs, index)
            for topo, instant, lagged, mechs in systems
        ]
        effective: set[Edge] = set()
        for _, read in per_system:
            effective |= read
        edges = tuple(sorted(effective))
        return cls(
            n=len(released),
            columns=released,
            edges=edges,
            k_sampled=k_sampled,
            k_eff=max((lag for _, _, lag in edges), default=0),
            hidden=tuple(hidden),
            # Compared per regime: an edge one regime reads must not hide the
            # same edge being ignored by another regime's mechanism.
            reads_parents=all(sampled == read for sampled, read in per_system),
            regime_edges=(
                tuple(tuple(sorted(sampled)) for sampled, _ in per_system)
                if isinstance(scm, RegimeSwitchingTemporalSCM)
                else None
            ),
            time="discrete",
            latent=latent,
        )

    @classmethod
    def from_structure(cls, name: str) -> LaggedGraph:
        """Graph of a named identifiability structure in released column order.

        Built from the fixed structure (``TSCMSampler(..., max_lag=1)``), with the
        canonical column order of the release: treatment ``A`` first, outcome
        ``Y`` last, the other nodes in topological order between them. The hidden
        confounder ``U`` is listed in ``hidden``, because the release zeroes it.
        Every mechanism of a named structure holds a weight for every node, so
        all edges are effective.

        Args:
            name: A :class:`~dotime.tscm_sampler.TSCMStructure` value such as
                ``"back_door"`` (the legacy label ``"rct_no_confounding"`` is
                accepted).

        Returns:
            The graph with ``time == "discrete"``.

        Raises:
            ValueError: If ``name`` is not a structure.
        """
        from dotime.tscm_sampler import TSCMSampler, TSCMStructure

        sampler = TSCMSampler(TSCMStructure(name), max_lag=1)
        dag = sampler._build_dag()
        topo = list(dag.topo_order)
        col = _canonical_columns(topo, topo.index("A"), topo.index("Y"))
        edges = [(col[u], col[v], 0) for u, v in dag.G_0.edges()]
        for k, g_k in enumerate(dag.G_lags):
            edges += [
                (col[topo[j]], col[topo[i]], k + 1)
                for j in range(len(topo))
                for i in range(len(topo))
                if g_k[j, i] > 0
            ]
        return cls._from_canonical(col, edges, dag.K, sampler.get_hidden_vars(), topo, "discrete")

    @classmethod
    def from_continuous_structure(cls, name: str) -> LaggedGraph:
        """Graph of a named structure under the continuous-time prior.

        The continuous sampler collapses each node's instantaneous and lag-1
        parents into one parent set whose values enter the drift one Euler step
        later, so every edge has lag 1, and each variable's mean reversion is a
        lag-1 self-edge. Columns follow the continuous release: ``A`` first,
        ``Y`` last, hidden ``U`` zeroed.

        Args:
            name: A :class:`~dotime.tscm_sampler.TSCMStructure` value.

        Returns:
            The graph with ``time == "continuous"``.

        Raises:
            ValueError: If ``name`` is not a structure.
        """
        from dotime.continuous.tscm_sampler import ContinuousTSCMSampler
        from dotime.tscm_sampler import TSCMStructure

        sampler = ContinuousTSCMSampler(TSCMStructure(name))
        topo = sampler.node_names
        col = _canonical_columns(topo, sampler.get_intervention_target(), sampler.get_outcome_var())
        edges = [
            (col[topo[u]], col[topo[v]], 1)
            for v, parents in enumerate(sampler._parents_per_node)
            for u in parents
        ]
        edges += [(c, c, 1) for c in col.values()]
        return cls._from_canonical(col, edges, 1, sampler.get_hidden_vars(), topo, "continuous")

    @classmethod
    def _from_canonical(
        cls,
        col: dict[str, int],
        edges: list[Edge],
        k_sampled: int,
        hidden_topo: Iterable[int],
        topo: list[str],
        time: str,
    ) -> LaggedGraph:
        """Assemble a named-structure graph once its edges are in column order.

        Args:
            col: Node name to released column.
            edges: Edges in released column order.
            k_sampled: The structure's maximum lag.
            hidden_topo: Topological indices the generator zeroes.
            topo: The structure's topological order.
            time: ``"discrete"`` or ``"continuous"``.

        Returns:
            The graph.
        """
        edge_tuple = _edge_tuple(edges, len(col))
        return cls(
            n=len(col),
            columns=tuple(sorted(col, key=col.__getitem__)),
            edges=edge_tuple,
            k_sampled=k_sampled,
            k_eff=max((lag for _, _, lag in edge_tuple), default=0),
            hidden=tuple(col[topo[h]] for h in hidden_topo),
            reads_parents=True,
            regime_edges=None,
            time=time,
        )


def _canonical_columns(topo: list[str], a_topo: int, y_topo: int) -> dict[str, int]:
    """Released column of each node of a named structure.

    Mirrors ``TSCMPrior.canonical_perm`` and the continuous prior's
    ``canonical_perm``: treatment first, outcome last, the rest in topological
    order. Recomputed here rather than read from a prior because building a
    prior seeds a generator, and graph extraction must touch no RNG.

    Args:
        topo: Node names in topological order.
        a_topo: Topological index of the treatment.
        y_topo: Topological index of the outcome.

    Returns:
        Map from node name to released column index.
    """
    middle = [i for i in range(len(topo)) if i not in (a_topo, y_topo)]
    return {topo[t]: c for c, t in enumerate([a_topo, *middle, y_topo])}


def _system_edges(
    topo: list[str],
    instant: dict[str, list[str]],
    lagged: dict[str, list[list[str]]],
    mechs: dict[str, Any],
    index: dict[str, int],
) -> tuple[set[Edge], set[Edge]]:
    """Sampled and effective edges of one graph-and-mechanisms system.

    Args:
        topo: Node names in simulation order.
        instant: Child name to its same-step parent names.
        lagged: Child name to its parent names per lag, index ``k`` for lag
            ``k + 1``.
        mechs: Child name to its ``TemporalMechanism``.
        index: Node name to graph node index.

    Returns:
        ``(sampled, effective)`` edge sets in graph node indices.
    """
    sampled: set[Edge] = set()
    effective: set[Edge] = set()
    for v in topo:
        mech = mechs[v]
        for p in instant[v]:
            edge = (index[p], index[v], 0)
            sampled.add(edge)
            if p in mech.weights_instant:
                effective.add(edge)
        for k, parents_k in enumerate(lagged[v]):
            # forward() zips the lagged weights with the lagged parents
            # without strict=True, so a lag with no weight dict is never read.
            weights_k = mech.weights_lagged[k] if k < len(mech.weights_lagged) else {}
            for p in parents_k:
                edge = (index[p], index[v], k + 1)
                sampled.add(edge)
                if p in weights_k:
                    effective.add(edge)
    return sampled, effective


@dataclass(frozen=True)
class PathLag:
    """How an intervention's sources reach one target on a :class:`LaggedGraph`.

    Only effective edges between distinct nodes count. A self-edge delays a
    node's own memory but never brings the effect closer to another node.

    Attributes:
        reachable: Whether a directed path leads from a source to the target. A
            target that is itself a source is reachable at lag 0 in 0 hops.
        min_lag: Smallest summed lag over such paths, i.e. the first step after
            the onset at which the target can respond. ``None`` if unreachable.
        min_hops: Fewest edges over such paths. ``None`` if unreachable. The
            path with the fewest hops need not have the smallest lag.
        direct_lags: Distinct lags of the edges from a source straight into the
            target, sorted. Empty when no source is a parent of the target.
    """

    reachable: bool
    min_lag: int | None
    min_hops: int | None
    direct_lags: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view with built-in values only.

        Returns:
            Dict with the four fields, ``direct_lags`` as a list.
        """
        return {
            "reachable": self.reachable,
            "min_lag": self.min_lag,
            "min_hops": self.min_hops,
            "direct_lags": list(self.direct_lags),
        }


def path_lag(graph: LaggedGraph, sources: Iterable[int], target: int) -> PathLag:
    """Measure the paths from the intervened columns to a queried column.

    Args:
        graph: The episode's graph.
        sources: Intervened graph nodes, e.g. ``episode.intervention.targets``.
        target: Queried graph node, e.g. one entry of ``episode.query_target``.

    Returns:
        The :class:`PathLag` summary.

    Raises:
        ValueError: If ``sources`` is empty or a node is outside the graph.
    """
    srcs = {operator.index(s) for s in sources}
    tgt = operator.index(target)
    if not srcs:
        raise ValueError("path_lag needs at least one source")
    if any(not 0 <= v < graph.n_nodes for v in (*srcs, tgt)):
        raise ValueError(f"sources {sorted(srcs)} or target {tgt} outside the graph")
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(graph.n_nodes)]
    for src, dst, lag in graph.edges:
        if src != dst:
            adjacency[src].append((dst, lag))
    direct = tuple(
        sorted({lag for src, dst, lag in graph.edges if src in srcs and dst == tgt and src != dst})
    )
    if tgt in srcs:
        return PathLag(True, 0, 0, direct)

    # Lags are non-negative, so Dijkstra from all sources at once gives the
    # smallest summed lag; lag-0 edges make a plain BFS by steps unsuitable.
    best = dict.fromkeys(srcs, 0)
    heap = [(0, s) for s in sorted(srcs)]
    while heap:
        lag_so_far, node = heapq.heappop(heap)
        if lag_so_far > best[node]:
            continue
        for nxt, lag in adjacency[node]:
            cand = lag_so_far + lag
            if cand < best.get(nxt, cand + 1):
                best[nxt] = cand
                heapq.heappush(heap, (cand, nxt))
    if tgt not in best:
        return PathLag(False, None, None, direct)

    hops = dict.fromkeys(srcs, 0)
    queue = deque(sorted(srcs))
    while queue:
        node = queue.popleft()
        for nxt, _ in adjacency[node]:
            if nxt not in hops:
                hops[nxt] = hops[node] + 1
                queue.append(nxt)
    return PathLag(True, best[tgt], hops[tgt], direct)


def load_graph_sidecar(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read a graph sidecar (JSON lines, optionally gzipped) keyed by episode.

    The sidecars published for the frozen suites hold one JSON object per
    episode with an ``"idx"`` (the episode's ``scm_id``), the episode's graph
    under ``"graph"`` and the path and audit fields described in the benchmark
    documentation.

    Args:
        path: ``.jsonl`` or ``.jsonl.gz`` file. Compression is detected from
            the file's first bytes, not its name.

    Returns:
        Map from ``idx`` to the record, with ``record["graph"]`` parsed into a
        :class:`LaggedGraph`.

    Raises:
        OSError: If the file cannot be read.
        KeyError: If a record has no ``"idx"`` or ``"graph"``.
        ValueError: If a line is not JSON, an ``idx`` repeats or a graph is
            inconsistent.
    """
    path = Path(path)
    raw = path.read_bytes()
    text = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    out: dict[int, dict[str, Any]] = {}
    for line in text.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        idx = int(record["idx"])
        if idx in out:
            raise ValueError(f"{path} lists idx {idx} twice")
        record["graph"] = LaggedGraph.from_dict(record["graph"])
        out[idx] = record
    return out
