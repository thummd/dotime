"""DoTime: Main orchestrator for sampling temporal SCMs with interventions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dotime._activations import Tanh, TanhReLU, TanhX2
from dotime._sampling import ShiftedExponentialSampler
from dotime.chain_scm import ChainSCMBuilder
from dotime.interventions import InterventionSampler, InterventionSpec
from dotime.regime_switching import RegimeSwitchingTemporalSCM
from dotime.regime_switching_builder import RegimeSwitchingSCMBuilder
from dotime.temporal_scm import TemporalSCM
from dotime.temporal_scm_builder import TemporalSCMBuilder
from dotime.utils import DEFAULT_CONFIG


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Cos(nn.Module):
    def forward(self, x):
        return torch.cos(x)


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class Square(nn.Module):
    def forward(self, x):
        return torch.pow(x, 2)


class DoTime:
    """
    Prior distribution over temporal SCMs with interventions.

    Main interface for generating synthetic causal time series data.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        seed: int = 42,
        chain_prob: float = 0.15,
        regime_switching_prob: float = 0.15,
    ):
        """
        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary. If None, uses DEFAULT_CONFIG.
        seed : int
            Random seed for reproducibility.
        chain_prob : float
            Probability of generating a chain SCM (default 0.15).
        regime_switching_prob : float
            Probability of generating a regime-switching SCM (default 0.15).
        """
        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG}
        if config is not None:
            self.config.update(config)

        self.seed = seed
        self.chain_prob = chain_prob
        self.regime_switching_prob = regime_switching_prob
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Activation functions (from paper + Do-PFN)
        self.activations = [
            nn.Identity(),  # Linear
            Tanh(),  # tanh
            TanhX2(),  # tanh(x^2)
            TanhReLU(),  # tanh(relu(x))
            nn.ReLU(),  # relu
            # Additional nonlinear functions from the paper
            Sin(),  # sin
            Cos(),  # cos
            Abs(),  # abs
            Square(),  # x^2
        ]

        # Chain SCM builder
        self.chain_builder = ChainSCMBuilder(
            activations=self.activations,
            device=self.config["device"],
        )

        # Regime-switching SCM builder (will be instantiated per sample)
        # since it depends on sampled N

    def sample_scm(self) -> TemporalSCM:
        """Sample a temporal SCM from the prior.

        Distribution:
        - chain_prob: chain SCMs
        - regime_switching_prob: regime-switching SCMs
        - remaining: diverse nonlinear SCMs

        Returns
        -------
        TemporalSCM
            Sampled temporal SCM (or compatible regime-switching SCM).
        """
        # Decide SCM type
        rand_val = torch.rand(1, generator=self.generator).item()

        if rand_val < self.chain_prob:
            # Sample chain SCM
            scm = self.chain_builder.sample(self.generator)
        elif rand_val < self.chain_prob + self.regime_switching_prob:
            # Sample regime-switching SCM
            N = int(
                torch.randint(3, self.config["N_max"] + 1, (1,), generator=self.generator).item()
            )
            K = int(
                torch.randint(1, self.config["K_max"] + 1, (1,), generator=self.generator).item()
            )

            rs_builder = RegimeSwitchingSCMBuilder(
                num_nodes=N,
                max_lag=K,
                activations=self.activations,
                gamma=self.config["gamma"],
                sigma_w=self.config["sigma_w"],
                sigma_b=self.config["sigma_b"],
                device=self.config["device"],
            )

            scm = rs_builder.sample(self.generator)
        else:
            # Sample diverse nonlinear SCM
            # Sample hyperparameters
            N = int(
                torch.randint(3, self.config["N_max"] + 1, (1,), generator=self.generator).item()
            )
            K = int(
                torch.randint(1, self.config["K_max"] + 1, (1,), generator=self.generator).item()
            )

            # Sample edge probability from Beta distribution
            alpha, beta = self.config["alpha"], self.config["beta"]
            edge_prob = float(torch.distributions.Beta(alpha, beta).sample().item())

            # Sample dropout probability
            dropout_prob = float(torch.rand(1, generator=self.generator).item() * 0.3)  # Up to 30%

            # Create noise distributions
            root_std_dist = ShiftedExponentialSampler(rate=1.0, shift=0.1)
            non_root_std_dist = ShiftedExponentialSampler(rate=10.0, shift=0.01)

            # Create SCM builder
            scm_builder = TemporalSCMBuilder(
                num_nodes=N,
                max_lag=K,
                edge_prob=edge_prob,
                dropout_prob=dropout_prob,
                gamma=self.config["gamma"],
                activations=self.activations,
                root_std_dist=root_std_dist,
                non_root_std_dist=non_root_std_dist,
                root_mean=self.config["root_mean"],
                non_root_mean=self.config["non_root_mean"],
                sigma_w=self.config["sigma_w"],
                sigma_b=self.config["sigma_b"],
                device=self.config["device"],
            )

            # Sample SCM
            scm = scm_builder.sample(self.generator)

        return scm

    def generate_pair(
        self,
        T: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, InterventionSpec, TemporalSCM]:
        """Generate a pair of observational and interventional time series.

        Parameters
        ----------
        T : int, optional
            Length of time series. If None, uses config default.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, InterventionSpec, TemporalSCM]
            (X_obs, X_int, intervention_spec, scm)
        """
        if T is None:
            T = self.config["T"]

        # Sample SCM
        scm = self.sample_scm()
        N = len(scm._topo)

        # Sample intervention
        intervention_sampler = InterventionSampler(
            N=N,
            T=T,
            generator=self.generator,
        )
        intervention = intervention_sampler.sample()

        # Generate observational data
        X_obs = scm.sample_observational(
            T=T,
            burn_in=self.config["burn_in"],
            generator=self.generator,
        )

        # Generate interventional data
        X_int = scm.sample_interventional(
            T=T,
            intervention=intervention,
            burn_in=self.config["burn_in"],
            generator=self.generator,
        )

        return X_obs, X_int, intervention, scm

    def generate_regime_pair(
        self,
        T: int | None = None,
        num_regimes: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor, InterventionSpec, RegimeSwitchingTemporalSCM]:
        """Generate a paired (obs, int) trajectory from a regime-switching SCM.

        Like :meth:`generate_pair` but forces a regime-switching SCM with a fixed
        number of regimes (for the regime-density benchmark tiers).
        """
        if T is None:
            T = self.config["T"]

        N = int(torch.randint(3, self.config["N_max"] + 1, (1,), generator=self.generator).item())
        K = int(torch.randint(1, self.config["K_max"] + 1, (1,), generator=self.generator).item())
        rs_builder = RegimeSwitchingSCMBuilder(
            num_nodes=N,
            max_lag=K,
            activations=self.activations,
            gamma=self.config["gamma"],
            sigma_w=self.config["sigma_w"],
            sigma_b=self.config["sigma_b"],
            device=self.config["device"],
        )
        scm = rs_builder.sample(self.generator, num_regimes=num_regimes)

        intervention = InterventionSampler(N=N, T=T, generator=self.generator).sample()
        X_obs = scm.sample_observational(
            T=T, burn_in=self.config["burn_in"], generator=self.generator
        )
        X_int = scm.sample_interventional(
            T=T, intervention=intervention, burn_in=self.config["burn_in"], generator=self.generator
        )
        return X_obs, X_int, intervention, scm

    def generate_dataset(
        self,
        n_scms: int,
        T: int | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, InterventionSpec]]:
        """Generate a dataset of paired observational/interventional time series.

        Parameters
        ----------
        n_scms : int
            Number of SCMs to sample.
        T : int, optional
            Length of time series. If None, uses config default.

        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, InterventionSpec]]
            List of (X_obs, X_int, intervention_spec) tuples.
        """
        dataset = []

        for i in range(n_scms):
            X_obs, X_int, intervention, _scm = self.generate_pair(T=T)
            dataset.append((X_obs, X_int, intervention))

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_scms} SCM pairs...")

        return dataset

    def generate_training_tuples(
        self,
        n_scms: int,
        T: int | None = None,
    ) -> list[tuple[torch.Tensor, list[int], list[int], Any, torch.Tensor]]:
        """Generate training tuples for PFN training.

        Format: (X_obs, targets, times, values, Y_int_tau)

        Parameters
        ----------
        n_scms : int
            Number of SCMs to sample.
        T : int, optional
            Length of time series. If None, uses config default.

        Returns
        -------
        List[Tuple[torch.Tensor, List[int], List[int], Any, torch.Tensor]]
            Training tuples suitable for PFN training.
        """
        if T is None:
            T = self.config["T"]

        training_data = []

        for i in range(n_scms):
            X_obs, X_int, intervention, _scm = self.generate_pair(T=T)

            # Extract target variable outcomes at intervention times
            target_idx = intervention.targets[0] if len(intervention.targets) > 0 else 0
            Y_int_tau = X_int[:, target_idx]

            training_data.append(
                (
                    X_obs,
                    intervention.targets,
                    intervention.times,
                    intervention.values,
                    Y_int_tau,
                )
            )

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_scms} training tuples...")

        return training_data
