"""Audit the adjustment sets behind the Identifiability structures by d-separation.

Backs the formulas appendix of the benchmark paper. For each named structure the
script prints the canonical column layout, every instantaneous and lagged cross
edge of the structure's temporal DAG, the adjustment set the shipped
``BackDoorOLS`` baseline conditions on, and whether candidate sets satisfy the
back-door criterion in an unrolled graph. It also tests the exclusion restriction
of ``X`` in the instrumental variable structure.

The classical formulas are written for one time slice. The released graphs carry
autoregressive self edges and lagged cross edges, so a set that is valid in the
slice graph can leave a lagged back-door path open. The audit unrolls the graph
over three time slices, places the onset at the middle slice, and checks
d-separation with the edges out of the treatment removed, which is the graphical
form of the back-door criterion.

Run with the development environment::

    python scripts/adjustment_set_audit.py

Building a :class:`~dotime.extended.TSCMPrior` draws no random numbers, so the
audit never touches an RNG stream.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from dotime.baselines import _back_door_columns, _canonical_summary_graph
from dotime.extended import TSCMPrior
from dotime.tscm_sampler import TSCMStructure

STRUCTURES = (
    "back_door",
    "observed_confounder",
    "confounder_mediator",
    "front_door",
    "mediator",
    "instrumental_variable",
    "bi_variate",
    "unobserved_confounder",
    "bow_graph",
)
ONSET = 1
SLICES = 3


def unroll(dag, topo: list[str], slices: int = SLICES) -> nx.DiGraph:
    """Unroll a temporal DAG over consecutive time slices.

    Args:
        dag: A :class:`~dotime.tscm_sampler.TemporalDAG` with ``G_0`` (the
            instantaneous graph) and ``G_lags`` (one adjacency matrix per lag).
        topo: Variable names in the order of the rows of the lag matrices.
        slices: Number of time slices to unroll.

    Returns:
        A directed graph whose node ``"X2"`` is variable ``X`` at slice ``2``.

    Raises:
        ValueError: If ``slices`` is smaller than one.
    """
    if slices < 1:
        raise ValueError("slices must be at least 1")
    graph = nx.DiGraph()
    inst = np.asarray(nx.to_numpy_array(dag.G_0, nodelist=topo))
    for s in range(slices):
        for i, j in zip(*np.nonzero(inst), strict=True):
            graph.add_edge(f"{topo[i]}{s}", f"{topo[j]}{s}")
        for lag, matrix in enumerate(dag.G_lags, start=1):
            if s - lag < 0:
                continue
            for i, j in zip(*np.nonzero(np.asarray(matrix)), strict=True):
                graph.add_edge(f"{topo[i]}{s - lag}", f"{topo[j]}{s}")
    return graph


def is_back_door_set(graph: nx.DiGraph, treatment: str, outcome: str, adjust: set[str]) -> bool:
    """Check the back-door criterion for one adjustment set.

    Args:
        graph: Unrolled graph from :func:`unroll`.
        treatment: Node name of the treatment, for example ``"A1"``.
        outcome: Node name of the outcome, for example ``"Y1"``.
        adjust: Node names of the candidate adjustment set.

    Returns:
        ``True`` when ``adjust`` d-separates the treatment from the outcome once
        the edges out of the treatment are deleted, which blocks every path into
        the treatment.

    Raises:
        nx.NodeNotFound: If a named node is not in the graph.
    """
    # Deleting the outgoing edges keeps exactly the paths that enter the
    # treatment, so d-separation in the pruned graph is the back-door criterion.
    pruned = graph.copy()
    pruned.remove_edges_from(list(graph.out_edges(treatment)))
    return nx.is_d_separator(pruned, {treatment}, {outcome}, adjust)


def instrument_checks(graph: nx.DiGraph, conditioning: set[str]) -> tuple[bool, bool]:
    """Relevance and exclusion of ``X`` as an instrument for ``A -> Y`` at the onset.

    Args:
        graph: Unrolled instrumental variable graph.
        conditioning: Node names conditioned on in both checks.

    Returns:
        ``(relevant, excluded)``. ``relevant`` is ``True`` when the instrument
        is d-connected to the treatment. ``excluded`` is ``True`` when the
        instrument is d-separated from the outcome after deleting the direct
        treatment to outcome edge, so no path bypasses the treatment.

    Raises:
        nx.NodeNotFound: If the onset nodes are not in the graph.
    """
    z, a, y = f"X{ONSET}", f"A{ONSET}", f"Y{ONSET}"
    relevant = not nx.is_d_separator(graph, {z}, {a}, conditioning)
    pruned = graph.copy()
    pruned.remove_edge(a, y)
    excluded = nx.is_d_separator(pruned, {z}, {y}, conditioning)
    return relevant, excluded


def main() -> None:
    """Print the audit for every structure."""
    for structure in STRUCTURES:
        names, _summary, hidden = _canonical_summary_graph(structure)
        prior = TSCMPrior(TSCMStructure(structure), pair_mode="counterfactual")
        dag = prior.dag
        topo = list(dag.topo_order)
        graph = unroll(dag, topo)
        inst = [f"{u}->{v}" for u, v in dag.G_0.edges()]
        lagged = [
            f"{topo[i]}(t-{lag})->{topo[j]}(t)"
            for lag, matrix in enumerate(dag.G_lags, start=1)
            for i, j in zip(*np.nonzero(np.asarray(matrix)), strict=True)
            if i != j
        ]
        _n, _a_col, _y_col, adjust_cols = _back_door_columns(structure)
        shipped = [names[c] for c in adjust_cols]
        print(f"\n### {structure}: columns={names} hidden={sorted(hidden)}")
        print(f"  instantaneous: {inst}")
        print(f"  lagged cross edges: {lagged} (plus a self edge on every variable)")
        print(f"  BackDoorOLS adjustment set (summary graph): {shipped}")
        treatment = f"A{ONSET}"
        parents = sorted(p for p in graph.predecessors(treatment) if not p.startswith("U"))
        observed = [v for v in topo if v not in hidden]
        candidates: dict[str, set[str]] = {
            "empty set": set(),
            "observed parents of A_T": set(parents),
            "pre-onset history": {f"{v}{ONSET - 1}" for v in observed},
            "{Y_T-1}": {f"Y{ONSET - 1}"},
        }
        if "X" in topo:
            candidates["{X_T, Y_T-1} (BackDoorOLS)"] = {f"X{ONSET}", f"Y{ONSET - 1}"}
        for offset in (0, 1):
            outcome = f"Y{ONSET + offset}"
            for label, adjust in candidates.items():
                ok = is_back_door_set(graph, treatment, outcome, adjust)
                print(f"  A_T -> Y_T+{offset}: back-door set {label:28s} valid={ok}")
        if structure == "instrumental_variable":
            for label, cond in {
                "nothing": set(),
                "{Y_T-1}": {f"Y{ONSET - 1}"},
                "{X_T-1}": {f"X{ONSET - 1}"},
                "pre-onset history": {f"{v}{ONSET - 1}" for v in observed},
            }.items():
                relevant, excluded = instrument_checks(graph, cond)
                print(
                    f"  X_T as instrument given {label:18s}: relevant={relevant} excluded={excluded}"
                )


if __name__ == "__main__":
    main()
