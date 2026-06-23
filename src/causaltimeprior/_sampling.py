"""Distribution samplers for hyperparameter and noise sampling.

This module is a first-class reimplementation of the small sampling surface that
the CausalTimePrior base prior used from ``Do-PFN-prior`` (``dopfnprior.utils.sampling``).
It is reproduced here so that ``causaltimeprior`` is a self-contained package with
no git-submodule dependency.

Attribution
-----------
The sampler design (a generator-aware wrapper over :mod:`torch.distributions`,
the shifted-exponential / log-uniform variants, and the config-driven factory)
originates with **Do-PFN** (Oossen et al., *Do-PFN: In-Context Learning for
Causal Effect Estimation*). The random-number-generating operations are kept
behaviourally faithful so that trajectories sampled with a fixed seed reproduce
the published TSALM / FMSD numbers; an equivalence test pins this
(``tests/test_dopfn_equivalence.py``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Literal, overload

import torch
import torch.distributions as dist

__all__ = [
    "DistributionSampler",
    "FixedSampler",
    "TorchDistributionSampler",
    "ShiftedExponentialSampler",
    "DiscreteUniformSampler",
    "LogarithmicSampler",
    "DISTRIBUTION_FACTORIES",
    "build_samplers",
    "sample_parameters",
]


class DistributionSampler(ABC):
    """Abstract base class for distribution samplers."""

    @abstractmethod
    def sample_n(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Sample ``n`` values from this distribution."""

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return the log-probabilities of ``value``."""

    @abstractmethod
    def std(self) -> float:
        """Return the standard deviation of the distribution."""

    def sample(self, generator: torch.Generator | None = None) -> Any:
        """Sample a single scalar value."""
        return self.sample_n(1, generator).item()

    def sample_shape(
        self, shape: tuple[int, ...], generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Vectorized sampling for an arbitrary output shape."""
        n = int(math.prod(shape))
        flat = self.sample_n(n, generator=generator)
        return flat.reshape(shape)


class FixedSampler(DistributionSampler):
    """Sampler that always returns a fixed value."""

    def __init__(self, value: Any):
        self.value = value

    def sample_n(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.full((n,), self.value)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        equal = value == self.value
        return torch.where(
            equal,
            torch.zeros_like(value, dtype=torch.float32),
            torch.full_like(value, float("-inf"), dtype=torch.float32),
        )

    def std(self) -> float:
        return 0.0


class TorchDistributionSampler(DistributionSampler):
    """Generator-aware wrapper around a :class:`torch.distributions.Distribution`.

    The wrapper threads an explicit :class:`torch.Generator` through ``sample``
    (which the base ``torch.distributions`` API does not support) by temporarily
    swapping the global RNG state — preserving determinism without disturbing the
    caller's RNG.
    """

    def __init__(self, distribution: dist.Distribution):
        self.distribution = distribution
        # Allow log_prob of out-of-support values to return -inf instead of raising.
        self.distribution._validate_args = False

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(value)

    @torch.no_grad()
    def sample_n(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            return self.distribution.sample((n,))
        # Swap the global RNG state to the provided generator, sample, restore.
        saved = torch.get_rng_state()
        torch.set_rng_state(generator.get_state())
        try:
            value = self.distribution.sample((n,))
        finally:
            generator.set_state(torch.get_rng_state())
            torch.set_rng_state(saved)
        return value

    def std(self) -> float:
        return self.distribution.stddev.item()


class ShiftedExponentialSampler(DistributionSampler):
    """Exponential distribution shifted by a fixed amount."""

    def __init__(self, rate: float, shift: float):
        self.rate = rate
        self.shift = shift
        self._exp = TorchDistributionSampler(dist.Exponential(rate=rate))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        shifted = value - self.shift
        log_probs = self._exp.log_prob(shifted)
        return torch.where(
            shifted >= 0, log_probs, torch.full_like(value, float("-inf"), dtype=torch.float32)
        )

    def sample_n(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return self._exp.sample_n(n, generator) + self.shift

    def std(self) -> float:
        return self._exp.std()


class DiscreteUniformSampler(DistributionSampler):
    """Discrete uniform distribution over the integers ``[low, high]``."""

    def __init__(self, low: int, high: int):
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
        self.low = low
        self.high = high

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        in_range = (value >= self.low) & (value <= self.high)
        num_values = self.high - self.low + 1
        log_p = math.log(1.0 / num_values)
        return torch.where(
            in_range,
            torch.full_like(value, log_p, dtype=torch.float32),
            torch.full_like(value, float("-inf"), dtype=torch.float32),
        )

    def sample_n(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            return torch.randint(self.low, self.high + 1, (n,))
        saved = torch.get_rng_state()
        torch.set_rng_state(generator.get_state())
        try:
            values = torch.randint(self.low, self.high + 1, (n,))
        finally:
            generator.set_state(torch.get_rng_state())
            torch.set_rng_state(saved)
        return values

    def std(self) -> float:
        num_values = self.high - self.low + 1
        return math.sqrt((num_values**2 - 1) / 12)


class LogarithmicSampler(DistributionSampler):
    """Sample ``x`` in ``[low, high]`` such that ``log(x)`` is uniform."""

    def __init__(self, low: float, high: float):
        self.log_low = math.log(low)
        self.log_high = math.log(high)
        self._uniform = TorchDistributionSampler(
            dist.Uniform(low=self.log_low, high=self.log_high)
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -torch.log(value) - math.log(self.log_high - self.log_low)

    def sample_n(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.exp(self._uniform.sample_n(n, generator))

    def std(self) -> float:
        a = math.exp(self.log_low)
        b = math.exp(self.log_high)
        mean = (b - a) / (math.log(b) - math.log(a))
        mean_sq = ((b**2 - a**2) / 2) / (math.log(b) - math.log(a))
        return math.sqrt(mean_sq - mean**2)


DISTRIBUTION_FACTORIES = {
    "fixed": lambda p: FixedSampler(p["value"]),
    "uniform": lambda p: TorchDistributionSampler(dist.Uniform(low=p["low"], high=p["high"])),
    "normal": lambda p: TorchDistributionSampler(dist.Normal(loc=p["mean"], scale=p["std"])),
    "lognormal": lambda p: TorchDistributionSampler(dist.LogNormal(loc=p["mean"], scale=p["std"])),
    "exponential": lambda p: TorchDistributionSampler(dist.Exponential(rate=p["rate"])),
    "shifted_exponential": lambda p: ShiftedExponentialSampler(rate=p["rate"], shift=p["shift"]),
    "gamma": lambda p: TorchDistributionSampler(dist.Gamma(concentration=p["alpha"], rate=p["beta"])),
    "beta": lambda p: TorchDistributionSampler(
        dist.Beta(concentration1=p["alpha"], concentration0=p["beta"])
    ),
    "discrete_uniform": lambda p: DiscreteUniformSampler(p["low"], p["high"]),
    "logarithmic": lambda p: LogarithmicSampler(p["low"], p["high"]),
}


def build_samplers(
    config: dict[str, Any],
    config_name: str,
    expected_params: dict[str, Any] | None = None,
) -> dict[str, DistributionSampler]:
    """Build sampler objects from a configuration dict."""
    samplers: dict[str, DistributionSampler] = {}
    for param_name, param_config in config.items():
        if expected_params is not None and param_name not in expected_params:
            raise ValueError(f"Unknown {config_name} hyperparameter: {param_name}")

        if "value" in param_config and "distribution" not in param_config:
            samplers[param_name] = FixedSampler(param_config["value"])
            continue

        if "distribution" not in param_config:
            raise ValueError(
                f"Configuration for {config_name}.{param_name} must specify "
                "'distribution' or 'value'"
            )

        dist_type = param_config["distribution"]
        if dist_type not in DISTRIBUTION_FACTORIES:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        dist_params = param_config.get("distribution_parameters", {})
        if dist_type == "fixed":
            if "value" not in param_config:
                raise ValueError(f"Fixed distribution for {param_name} requires 'value' key")
            dist_params = {"value": param_config["value"]}

        try:
            samplers[param_name] = DISTRIBUTION_FACTORIES[dist_type](dist_params)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Error creating sampler for {config_name}.{param_name}: {exc}"
            ) from exc

    if expected_params is not None:
        missing = set(expected_params) - set(config)
        if missing:
            raise ValueError(f"Missing required {config_name} parameters: {missing}")

    return samplers


@overload
def sample_parameters(
    samplers: dict[str, DistributionSampler],
    generator: torch.Generator | None = None,
    return_log_prob: Literal[False] = False,
) -> dict[str, Any]: ...


@overload
def sample_parameters(
    samplers: dict[str, DistributionSampler],
    generator: torch.Generator | None = None,
    return_log_prob: Literal[True] = True,
) -> tuple[dict[str, Any], float]: ...


def sample_parameters(
    samplers: dict[str, DistributionSampler],
    generator: torch.Generator | None = None,
    return_log_prob: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], float]:
    """Sample one value per sampler, optionally returning the total log-prob."""
    sampled = {name: sampler.sample(generator) for name, sampler in samplers.items()}

    if not return_log_prob:
        return sampled

    total_log_prob = 0.0
    for name, sampler in samplers.items():
        total_log_prob += sampler.log_prob(torch.tensor([sampled[name]]))
    return sampled, total_log_prob
