"""Temporal graph prior for CausalTimePrior.

This module extends Do-PFN's GraphBuilder to support temporal (time-lagged) DAGs.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from causaltimeprior._graph_builder import GraphBuilder


@dataclass
class TemporalDAG:
    """Temporal DAG with instantaneous and lagged edges.

    Attributes:
        G_0: Instantaneous (intra-slice) DAG from Do-PFN's GraphBuilder
        G_lags: List of adjacency matrices for lagged edges [G_1, G_2, ..., G_K]
        K: Maximum lag
        topo_order: Topological order of G_0 (for forward simulation)
    """

    G_0: nx.DiGraph
    G_lags: list[np.ndarray]
    K: int
    topo_order: list[str]


class TemporalGraphBuilder:
    """Builder for temporal causal graphs with time lags.

    Extends Do-PFN's GraphBuilder for instantaneous edges (G_0) and adds
    lagged edges (G_1, ..., G_K) with decay probability p * gamma^k.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_prob: float,
        dropout_prob: float,
        max_lag: int,
        gamma: float = 0.7,
    ):
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes (variables).
        edge_prob : float
            Base probability of an edge (for G_0 and G_1).
        dropout_prob : float
            Probability of making a node hidden.
        max_lag : int
            Maximum lag K.
        gamma : float
            Lag decay factor for edge probability: p_k = p * gamma^k.
        """
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.dropout_prob = dropout_prob
        self.max_lag = max_lag
        self.gamma = gamma

        # Instantaneous graph builder (reuse Do-PFN)
        self.graph_builder = GraphBuilder(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            dropout_prob=dropout_prob,
        )

    def sample(self, generator: torch.Generator | None) -> TemporalDAG:
        """Sample a temporal DAG.

        Parameters
        ----------
        generator : torch.Generator
            Random number generator for reproducibility.

        Returns
        -------
        TemporalDAG
            Sampled temporal DAG with instantaneous and lagged edges.
        """
        # Sample instantaneous DAG using Do-PFN's GraphBuilder
        G_0 = self.graph_builder.sample(generator)

        # Get topological order from G_0
        topo_order = list(nx.topological_sort(G_0))

        # Sample lagged edges G_1, ..., G_K
        G_lags = []
        np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
        rng = np.random.default_rng(np_seed)

        for k in range(1, self.max_lag + 1):
            # Edge probability decays with lag
            p_k = self.edge_prob * (self.gamma**k)

            # Sample adjacency matrix (no acyclicity constraint for lagged edges)
            G_k = (rng.random((self.num_nodes, self.num_nodes)) < p_k).astype(np.float32)
            G_lags.append(G_k)

        return TemporalDAG(
            G_0=G_0,
            G_lags=G_lags,
            K=self.max_lag,
            topo_order=topo_order,
        )
