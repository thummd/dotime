"""Random DAG sampler used as the instantaneous-graph prior.

Reimplemented from ``Do-PFN-prior`` (``dopfnprior.causal_graph.graph_builder``)
so the package carries no submodule dependency.
:class:`~causaltime.temporal_graph.TemporalGraphBuilder` extends this with
lagged adjacencies.

Acyclicity is guaranteed by sampling edges from a strictly upper-triangular
Bernoulli mask over a random topological order. The RNG-determining operations
(the torch→numpy seed bridge, the permutation, the ``triu`` mask, the node
hiding) are kept faithful so sampled graphs reproduce the published results.

Attribution: Do-PFN (Oossen et al.).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch

__all__ = ["GraphBuilder"]


class GraphBuilder:
    """Sample random DAGs with hidden nodes and a designated target."""

    def __init__(self, num_nodes: int, edge_prob: float, dropout_prob: float) -> None:
        """
        Parameters
        ----------
        num_nodes:
            Number of nodes.
        edge_prob:
            Probability of an edge between any ordered pair ``i < j`` in a random
            topological order. Floored by ``2 / num_nodes**1.2`` to avoid very
            sparse small graphs.
        dropout_prob:
            Probability of marking a node hidden.
        """
        self.num_nodes = num_nodes
        edge_prob_min = 2 / (num_nodes**1.2)
        self.edge_prob = max(edge_prob_min, edge_prob)
        self.dropout_prob = dropout_prob

    def sample(self, generator: torch.Generator | None) -> nx.DiGraph:
        """Sample a random DAG, resampling until structural conditions hold.

        The returned graph relabels hidden nodes ``u{i}``, visible non-target
        nodes ``x{i}``, and the target node ``y``.
        """
        np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
        self.rng = np.random.default_rng(np_seed)

        n = int(self.num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 <= self.edge_prob <= 1.0):
            raise ValueError("edge_prob must be in [0, 1].")
        if not (0.0 <= self.dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1].")

        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))

        # Random topological order + strictly upper-triangular Bernoulli mask.
        perm = self.rng.permutation(n)
        mask = np.triu(self.rng.random((n, n)) < self.edge_prob, k=1)
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size:
            src = perm[i_idx]
            dst = perm[j_idx]
            graph.add_edges_from(zip(src.tolist(), dst.tolist(), strict=True))

        # Resample if there are no edges.
        if len(graph.edges) == 0:
            return self.sample(generator)

        # Hide some nodes.
        hidden_attr = {
            v: torch.rand(1, generator=generator) < self.dropout_prob for v in graph.nodes()
        }
        nx.set_node_attributes(graph, hidden_attr, name="hidden")

        visible_nodes = [v for v in graph.nodes if not graph.nodes[v]["hidden"]]
        if len(visible_nodes) < 2:
            return self.sample(generator)
        hidden_nodes = [v for v in graph.nodes if graph.nodes[v]["hidden"]]

        target_node = visible_nodes[-1]
        if graph.in_degree(target_node) == 0 or graph.out_degree(target_node) == 0:
            return self.sample(generator)

        renaming = {v: f"u{v}" for v in hidden_nodes}
        for v in visible_nodes:
            if v != target_node:
                renaming[v] = f"x{v}"
        renaming[target_node] = "y"
        return nx.relabel_nodes(graph, renaming)
