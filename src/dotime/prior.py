"""DoTime: Main orchestrator for sampling temporal SCMs with interventions."""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from dotime._activations import Tanh, TanhReLU, TanhX2
from dotime._sampling import ShiftedExponentialSampler
from dotime.chain_scm import ChainSCMBuilder
from dotime.hardening import harden_scm, validate_hardening
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


_PAIR_MODES = ("interventional", "counterfactual")
# Salts the seed of the counterfactual noise generator, so that its stream is
# unrelated to self.generator even though both derive from the same seed.
_COUNTERFACTUAL_SALT = zlib.crc32(b"dotime.prior.counterfactual")


class DoTime:
    """
    Prior distribution over temporal SCMs with interventions.

    Main interface for generating synthetic causal time series data. Two opt-in
    options leave the default draws unchanged: ``config["N_min"]`` raises the
    smallest number of variables of a sampled graph, and
    ``generate_pair(pair_mode="counterfactual")`` shares one exogenous-noise
    realisation across the observational and interventional arms.
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
            Configuration dictionary. If None, uses DEFAULT_CONFIG. An optional
            ``"hardening"`` entry, for example
            ``{"unit_norm_rows": True, "spectral_rho": 0.9}``, rescales every
            sampled SCM so that large graphs simulate without diverging (see
            :mod:`dotime.hardening`). It is off by default, which keeps the
            released suites reproducible, and it draws no random numbers.
            An optional ``"N_min"`` entry, an int in ``[3, N_max]`` that is not
            part of ``DEFAULT_CONFIG``, is the smallest number of variables of a
            diverse or regime-switching SCM. Their size is uniform on
            ``[N_min, N_max]``, and the default of 3 draws exactly the numbers
            the prior drew before the option existed. Chain SCMs keep their own
            length of 3 to 7 variables whatever ``N_min`` and ``N_max`` are.
        seed : int
            Random seed for reproducibility.
        chain_prob : float
            Probability of generating a chain SCM (default 0.15).
        regime_switching_prob : float
            Probability of generating a regime-switching SCM (default 0.15).
            By default these SCMs reproduce v1.0.0, in which no mechanism reads
            its parents, so every variable is independent noise. Set
            ``config["regime_canonical_weights"] = True`` to give them live
            parent weights and zero any arm whose values exceed 500, as for the
            other SCMs. The flag draws the same random numbers either way.

        Raises
        ------
        TypeError
            If ``config["regime_canonical_weights"]`` is not a bool, if
            ``config["N_min"]`` is not an int, or if ``config["hardening"]`` is
            not a dict or holds a value of the wrong type.
        ValueError
            If ``config["N_min"]`` lies outside ``[3, N_max]``, or if
            ``config["hardening"]`` has unknown keys or a non-positive
            ``spectral_rho`` (see :func:`dotime.hardening.validate_hardening`).
        """
        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG}
        if config is not None:
            self.config.update(config)
        # Validated before any draw, so a bad config fails without touching RNG.
        self.hardening = validate_hardening(self.config.get("hardening"))
        # A bool is an int in Python, so it is refused explicitly instead of
        # being read as 0 or 1. The floor of 3 is structural: the graph builder
        # resamples until its target node has a parent and a child, which two
        # nodes can never satisfy.
        n_min = self.config.get("N_min", 3)
        if isinstance(n_min, bool) or not isinstance(n_min, int):
            raise TypeError(f"config['N_min'] must be an int, got {type(n_min).__name__}")
        if not 3 <= n_min <= self.config["N_max"]:
            raise ValueError(
                f"config['N_min'] must lie in [3, N_max={self.config['N_max']}], got {n_min}"
            )
        self.n_min = n_min

        self.seed = seed
        self.chain_prob = chain_prob
        self.regime_switching_prob = regime_switching_prob
        # A strict bool check, because a truthy string such as "false" would
        # otherwise switch the fix on. It runs before any draw.
        canonical = self.config.get("regime_canonical_weights", False)
        if not isinstance(canonical, bool):
            raise TypeError(
                f"config['regime_canonical_weights'] must be a bool, got {type(canonical).__name__}"
            )
        self.regime_canonical_weights = canonical
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        # Created on the first counterfactual pair, so interventional use never
        # pays for it (see _counterfactual_generator).
        self._cf_generator: torch.Generator | None = None

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
                torch.randint(
                    self.n_min, self.config["N_max"] + 1, (1,), generator=self.generator
                ).item()
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
                canonical_weights=self.regime_canonical_weights,
            )

            scm = rs_builder.sample(self.generator)
        else:
            # Sample diverse nonlinear SCM
            # Sample hyperparameters
            N = int(
                torch.randint(
                    self.n_min, self.config["N_max"] + 1, (1,), generator=self.generator
                ).item()
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

        if self.hardening is not None:
            harden_scm(scm, **self.hardening)
        return scm

    def _counterfactual_generator(self) -> torch.Generator:
        """Return the generator of the shared counterfactual noise, creating it on first use.

        It is seeded from ``np.random.SeedSequence([seed, salt])`` rather than
        from ``self.generator``, so a counterfactual pair takes exactly the SCM
        and intervention draws of an interventional pair with the same seed.
        The generator persists across calls, so successive pairs get fresh
        noise.

        Returns:
            The prior's counterfactual noise generator.
        """
        if self._cf_generator is None:
            # SeedSequence refuses negative entropy, and the mask keeps any
            # Python int seed usable.
            entropy = [self.seed & (2**64 - 1), _COUNTERFACTUAL_SALT]
            state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]
            self._cf_generator = torch.Generator().manual_seed(int(state))
        return self._cf_generator

    def generate_pair(
        self,
        T: int | None = None,
        pair_mode: str = "interventional",
    ) -> tuple[torch.Tensor, torch.Tensor, InterventionSpec, TemporalSCM]:
        """Generate a pair of observational and interventional time series.

        Parameters
        ----------
        T : int, optional
            Length of time series. If None, uses config default.
        pair_mode : str
            ``"interventional"`` (the default, which the released suites use)
            simulates the two arms with independent noise drawn from the global
            torch RNG. ``"counterfactual"`` draws one exogenous-noise
            realisation for the whole pair from a separate generator seeded from
            ``seed``, and freezes it on the SCM (see
            :meth:`TemporalSCM.freeze_noise`). The arms then agree exactly
            before the intervention onset, and their difference is a
            per-episode counterfactual effect. Both modes draw the same SCM and
            intervention from ``self.generator``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, InterventionSpec, TemporalSCM]
            (X_obs, X_int, intervention_spec, scm)

        Raises
        ------
        ValueError
            If ``pair_mode`` is unknown, or if it is ``"counterfactual"`` while
            ``regime_switching_prob`` is not zero. Regime-switching SCMs draw
            their noise step by step from the global numpy RNG, so their arms
            cannot share it.
        """
        # Validated before any draw, so a bad call leaves every RNG untouched.
        if pair_mode not in _PAIR_MODES:
            raise ValueError(
                f"pair_mode must be interventional or counterfactual, got {pair_mode!r}"
            )
        if pair_mode == "counterfactual" and self.regime_switching_prob != 0:
            raise ValueError(
                "pair_mode='counterfactual' needs regime_switching_prob=0, because "
                "regime-switching SCMs cannot share noise across arms; got "
                f"{self.regime_switching_prob}"
            )
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

        if pair_mode == "counterfactual":
            # After the intervention draw and from its own generator, so
            # self.generator ends in the state an interventional pair leaves.
            scm.freeze_noise(T + self.config["burn_in"], generator=self._counterfactual_generator())

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

        N = int(
            torch.randint(
                self.n_min, self.config["N_max"] + 1, (1,), generator=self.generator
            ).item()
        )
        K = int(torch.randint(1, self.config["K_max"] + 1, (1,), generator=self.generator).item())
        rs_builder = RegimeSwitchingSCMBuilder(
            num_nodes=N,
            max_lag=K,
            activations=self.activations,
            gamma=self.config["gamma"],
            sigma_w=self.config["sigma_w"],
            sigma_b=self.config["sigma_b"],
            device=self.config["device"],
            canonical_weights=self.regime_canonical_weights,
        )
        scm = rs_builder.sample(self.generator, num_regimes=num_regimes)
        if self.hardening is not None:
            harden_scm(scm, **self.hardening)

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
