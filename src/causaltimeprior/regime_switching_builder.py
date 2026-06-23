"""Builder for regime-switching temporal SCMs."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from causaltimeprior.regime_switching import RegimeSwitchingTemporalSCM
from causaltimeprior.temporal_scm_builder import TemporalSCMBuilder
from causaltimeprior._sampling import ShiftedExponentialSampler


class RegimeSwitchingSCMBuilder:
    """Builder for regime-switching temporal SCMs.
    
    Samples R=2-3 regimes, each with its own DAG and mechanisms.
    Transition matrix P is sampled from Dirichlet prior (sticky regimes).
    """
    
    def __init__(
        self,
        num_nodes: int,
        max_lag: int,
        activations: List[nn.Module],
        gamma: float,
        sigma_w: float,
        sigma_b: float,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes (same across all regimes).
        max_lag : int
            Maximum lag K.
        activations : List[nn.Module]
            Activation functions to sample from.
        gamma : float
            Lag decay factor.
        sigma_w : float
            Mechanism weight std.
        sigma_b : float
            Mechanism bias std.
        device : torch.device
            Device for computation.
        """
        self.num_nodes = num_nodes
        self.max_lag = max_lag
        self.activations = activations
        self.gamma = gamma
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.device = device
    
    def sample(
        self,
        generator: Optional[torch.Generator] = None,
        num_regimes: Optional[int] = None,
    ) -> RegimeSwitchingTemporalSCM:
        """Sample a regime-switching temporal SCM.

        Parameters
        ----------
        generator : torch.Generator, optional
            RNG for reproducibility.
        num_regimes : int, optional
            Fix the number of regimes (e.g. for a regime-density benchmark tier).
            When ``None`` (default), it is sampled uniformly from ``{2, 3}``.

        Returns
        -------
        RegimeSwitchingTemporalSCM
            Sampled regime-switching SCM.
        """
        # Number of regimes: fixed when requested (benchmark tiers), else 2 or 3.
        if num_regimes is None:
            num_regimes = int(torch.randint(2, 4, (1,), generator=generator).item())
        
        # Sample transition matrix (sticky: high self-transition probability)
        # Dirichlet(alpha) with high alpha on diagonal
        transition_matrix = np.zeros((num_regimes, num_regimes))
        for i in range(num_regimes):
            # High concentration on staying in same regime
            alpha = np.ones(num_regimes) * 0.5
            alpha[i] = 9.0  # 90% probability of staying in regime i
            
            # Sample row from Dirichlet
            np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
            rng = np.random.default_rng(np_seed)
            transition_matrix[i] = rng.dirichlet(alpha)
        
        # Sample DAGs and mechanisms for each regime
        dags = []
        mechanisms_list = []
        
        # Create noise distributions (shared across regimes)
        root_std_dist = ShiftedExponentialSampler(rate=1.0, shift=0.1)
        non_root_std_dist = ShiftedExponentialSampler(rate=10.0, shift=0.01)
        
        # Fix node names to be consistent across all regimes
        # Use canonical names X0, X1, ..., X{N-1}
        canonical_node_names = [f"X{i}" for i in range(self.num_nodes)]
        
        # Sample all regimes with consistent node naming
        for r in range(num_regimes):
            edge_prob = float(torch.distributions.Beta(2, 5).sample().item())
            dropout_prob = float(torch.rand(1, generator=generator).item() * 0.3)
            
            scm_builder = TemporalSCMBuilder(
                num_nodes=self.num_nodes,
                max_lag=self.max_lag,
                edge_prob=edge_prob,
                dropout_prob=dropout_prob,
                gamma=self.gamma,
                activations=self.activations,
                root_std_dist=root_std_dist,
                non_root_std_dist=non_root_std_dist,
                root_mean=0.0,
                non_root_mean=0.0,
                sigma_w=self.sigma_w,
                sigma_b=self.sigma_b,
                device=self.device,
            )
            
            scm_r = scm_builder.sample(generator)
            
            # Remap node names to canonical names
            old_topo = scm_r.dag.topo_order
            node_mapping = {old_topo[i]: canonical_node_names[i] for i in range(len(old_topo))}
            
            # Remap DAG
            import networkx as nx
            from causaltimeprior.temporal_graph import TemporalDAG
            
            G_0_remapped = nx.DiGraph()
            G_0_remapped.add_nodes_from(canonical_node_names)
            for u, v in scm_r.dag.G_0.edges():
                G_0_remapped.add_edge(node_mapping[u], node_mapping[v])
            
            dag_remapped = TemporalDAG(
                G_0=G_0_remapped,
                G_lags=scm_r.dag.G_lags,
                K=scm_r.dag.K,
                topo_order=canonical_node_names,
            )
            
            # Remap mechanisms
            mechanisms_remapped = {canonical_node_names[i]: scm_r.mechanisms[old_topo[i]] 
                                   for i in range(len(old_topo))}
            
            dags.append(dag_remapped)
            mechanisms_list.append(mechanisms_remapped)
            
            # Use noise from first regime (shared across all)
            if r == 0:
                noise = {canonical_node_names[i]: scm_r.noise[old_topo[i]] 
                        for i in range(len(old_topo))}
        
        # Create regime-switching SCM
        rs_scm = RegimeSwitchingTemporalSCM(
            dags=dags,
            mechanisms=mechanisms_list,
            noise=noise,
            transition_matrix=transition_matrix,
            device=self.device,
        )
        
        # Add compatibility attributes for pipeline integration
        rs_scm._topo = dags[0].topo_order  # Use first regime's topology
        rs_scm.dag = dags[0]  # Use first regime's DAG as default
        
        return rs_scm