"""Builder for temporal SCMs with configurable hyperparameters."""

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

from dotime._sampling import DistributionSampler, TorchDistributionSampler
from dotime.temporal_graph import TemporalDAG, TemporalGraphBuilder
from dotime.temporal_mechanism import TemporalMechanism
from dotime.temporal_scm import TemporalSCM
from dotime.utils import LaplaceSampler, UniformNoiseSampler


class TemporalSCMBuilder:
    """Builder for temporal SCMs with configurable priors.

    Combines temporal graph sampling, mechanism sampling, and noise sampling
    to construct complete temporal SCMs.
    """

    def __init__(
        self,
        num_nodes: int,
        max_lag: int,
        edge_prob: float,
        dropout_prob: float,
        gamma: float,
        activations: list[nn.Module],
        root_std_dist: DistributionSampler,
        non_root_std_dist: DistributionSampler,
        root_mean: float = 0.0,
        non_root_mean: float = 0.0,
        sigma_w: float = 1.0,
        sigma_b: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters
        ----------
        num_nodes : int
            Number of nodes (variables).
        max_lag : int
            Maximum lag K.
        edge_prob : float
            Base edge probability.
        dropout_prob : float
            Probability of making a node hidden.
        gamma : float
            Lag decay factor.
        activations : List[nn.Module]
            List of activation functions to sample from.
        root_std_dist : DistributionSampler
            Distribution for root node noise std.
        non_root_std_dist : DistributionSampler
            Distribution for non-root node noise std.
        root_mean : float
            Mean for root node noise.
        non_root_mean : float
            Mean for non-root node noise.
        sigma_w : float
            Std for mechanism weights.
        sigma_b : float
            Std for mechanism biases.
        device : torch.device
            Device for computation.
        """
        self.num_nodes = num_nodes
        self.max_lag = max_lag
        self.edge_prob = edge_prob
        self.dropout_prob = dropout_prob
        self.gamma = gamma
        self.activations = activations
        self.root_std_dist = root_std_dist
        self.non_root_std_dist = non_root_std_dist
        self.root_mean = root_mean
        self.non_root_mean = non_root_mean
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.device = device

        # Graph builder
        self.graph_builder = TemporalGraphBuilder(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            dropout_prob=dropout_prob,
            max_lag=max_lag,
            gamma=gamma,
        )

    def sample(self, generator: torch.Generator | None = None) -> TemporalSCM:
        """Sample a complete temporal SCM.

        Parameters
        ----------
        generator : torch.Generator, optional
            RNG for reproducibility.

        Returns
        -------
        TemporalSCM
            Sampled temporal SCM.
        """
        # Sample temporal DAG
        dag = self.graph_builder.sample(generator)

        # Sample mechanisms
        mechanisms = self._create_mechanisms(dag, generator)

        # Sample noise distributions
        noise = self._create_noise_distributions(dag, generator)

        # Build SCM
        scm = TemporalSCM(dag, mechanisms, noise, device=self.device)

        return scm

    def _create_mechanisms(
        self, dag: TemporalDAG, generator: torch.Generator | None
    ) -> dict[str, TemporalMechanism]:
        """Create mechanisms for each variable.

        Parameters
        ----------
        dag : TemporalDAG
            Temporal DAG structure.
        generator : torch.Generator, optional
            RNG for reproducibility.

        Returns
        -------
        Dict[str, TemporalMechanism]
            Mechanisms for each variable.
        """
        nodes = dag.topo_order
        mechanisms = {}

        for v in nodes:
            # Sample activation function
            activation_idx = int(torch.randint(0, len(self.activations), (1,), generator=generator))
            activation = self.activations[activation_idx]

            # Create temporal mechanism
            mech = TemporalMechanism(
                node_names=nodes,
                activation=activation,
                num_lags=dag.K,
                device=self.device,
                generator=generator,
                sigma_w=self.sigma_w,
                sigma_b=self.sigma_b,
            )
            mechanisms[v] = mech

        return mechanisms

    def _create_noise_distributions(
        self, dag: TemporalDAG, generator: torch.Generator | None
    ) -> dict[str, DistributionSampler]:
        """Create noise distributions for each variable.

        Parameters
        ----------
        dag : TemporalDAG
            Temporal DAG structure.
        generator : torch.Generator, optional
            RNG for reproducibility.

        Returns
        -------
        Dict[str, DistributionSampler]
            Noise distributions for each variable.
        """
        # Identify root nodes (no instantaneous parents)
        root_nodes = [v for v in dag.topo_order if dag.G_0.in_degree(v) == 0]
        non_root_nodes = [v for v in dag.topo_order if dag.G_0.in_degree(v) > 0]

        noise = {}

        # Root nodes
        for v in root_nodes:
            std = self.root_std_dist.sample(generator)
            noise[v] = TorchDistributionSampler(dist.Normal(loc=self.root_mean, scale=std))

        # Non-root nodes
        for v in non_root_nodes:
            std = self.non_root_std_dist.sample(generator)

            # Sample noise distribution type
            noise_type = int(torch.randint(0, 3, (1,), generator=generator).item())

            if noise_type == 0:  # Gaussian
                noise[v] = TorchDistributionSampler(dist.Normal(loc=self.non_root_mean, scale=std))
            elif noise_type == 1:  # Uniform
                a = std * np.sqrt(3)  # Match variance
                noise[v] = UniformNoiseSampler(low=-a, high=a)
            else:  # Laplace
                b = std / np.sqrt(2)  # Match variance
                noise[v] = LaplaceSampler(loc=self.non_root_mean, scale=b)

        return noise
