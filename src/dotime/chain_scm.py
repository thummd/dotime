"""Simple chain SCM generator for DoTime.

Generates linear/nonlinear chain structures like A→B→C→D for better
generalization to simple sequential causal systems.
"""

import networkx as nx
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from dotime._sampling import TorchDistributionSampler
from dotime.temporal_graph import TemporalDAG
from dotime.temporal_mechanism import TemporalMechanism
from dotime.temporal_scm import TemporalSCM


class ChainSCMBuilder:
    """Builder for simple chain SCMs: A → B → C → ...

    These are simpler structures with sequential dependencies,
    helping the model generalize to chain-like benchmarks.
    """

    def __init__(
        self,
        activations: list[nn.Module],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        activations : List[nn.Module]
            List of activation functions (prefer Identity, tanh, relu for chains).
        device : torch.device
            Device for computation.
        """
        self.activations = activations
        self.device = device

    def sample(
        self,
        generator: torch.Generator | None = None,
        chain_length: int | None = None,
    ) -> TemporalSCM:
        """Sample a chain SCM.

        Parameters
        ----------
        generator : torch.Generator, optional
            RNG for reproducibility.
        chain_length : int, optional
            Length of chain (3-7). If None, sampled randomly.

        Returns
        -------
        TemporalSCM
            Chain-structured temporal SCM.
        """
        # Sample chain length
        if chain_length is None:
            chain_length = int(torch.randint(3, 8, (1,), generator=generator).item())

        # Create node names
        nodes = [f"X{i}" for i in range(chain_length)]

        # Build chain DAG: X0 → X1 → X2 → ...
        G_0 = nx.DiGraph()
        G_0.add_nodes_from(nodes)
        for i in range(chain_length - 1):
            G_0.add_edge(nodes[i], nodes[i + 1])

        # Add some lagged edges (with probability ~30%)
        max_lag = 2
        G_lags = []
        for _k in range(max_lag):
            G_k = np.zeros((chain_length, chain_length))
            # Each variable can depend on previous 1-2 time steps
            for i in range(1, chain_length):
                if torch.rand(1, generator=generator).item() < 0.3:
                    # Self-loop from lag k+1
                    G_k[i, i] = 1.0
            G_lags.append(G_k)

        dag = TemporalDAG(
            G_0=G_0,
            G_lags=G_lags,
            K=max_lag,
            topo_order=nodes,
        )

        # Create mechanisms (prefer simpler activations for chains)
        mechanisms = {}
        for v in nodes:
            # For chains, prefer Identity (linear), tanh, relu
            simple_activations = [
                a for a in self.activations if isinstance(a, (nn.Identity, nn.Tanh, nn.ReLU))
            ]
            if len(simple_activations) > 0:
                activation_idx = int(
                    torch.randint(0, len(simple_activations), (1,), generator=generator)
                )
                activation = simple_activations[activation_idx]
            else:
                activation_idx = int(
                    torch.randint(0, len(self.activations), (1,), generator=generator)
                )
                activation = self.activations[activation_idx]

            # Use moderate weights for stability
            mech = TemporalMechanism(
                node_names=nodes,
                activation=activation,
                num_lags=max_lag,
                device=self.device,
                generator=generator,
                sigma_w=0.8,  # Slightly smaller weights for chains
                sigma_b=0.3,  # Smaller bias
            )
            mechanisms[v] = mech

        # Create noise distributions (smaller noise for chains)
        noise = {}
        for v in nodes:
            noise_std = torch.rand(1, generator=generator).item() * 0.2 + 0.05  # [0.05, 0.25]
            noise[v] = TorchDistributionSampler(dist.Normal(loc=0.0, scale=noise_std))

        # Build SCM
        scm = TemporalSCM(dag, mechanisms, noise, device=self.device)

        return scm
