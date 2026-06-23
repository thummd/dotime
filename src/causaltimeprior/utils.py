"""Utility functions for CausalTimePrior."""

import torch
import torch.distributions as dist
import numpy as np
from typing import Optional

from causaltimeprior._sampling import DistributionSampler


# ===== Stability Checks =====

def clip_values(x: torch.Tensor, min_val: float = -1000.0, max_val: float = 1000.0) -> torch.Tensor:
    """Clip tensor values to prevent numerical instability."""
    return torch.clamp(x, min_val, max_val)


def check_divergence(x: torch.Tensor, threshold: float = 1e6) -> bool:
    """Check if tensor contains diverged values (NaN, Inf, or very large values)."""
    if torch.isnan(x).any() or torch.isinf(x).any():
        return True
    if (torch.abs(x) > threshold).any():
        return True
    return False


# ===== Extra Noise Distribution Samplers =====

class UniformNoiseSampler(DistributionSampler):
    """Uniform noise distribution sampler."""
    
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        self.distribution = dist.Uniform(low=low, high=high)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(value)
    
    def sample_n(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = self.distribution.sample((n,))
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            value = self.distribution.sample((n,))
        return value
    
    def std(self) -> float:
        return self.distribution.stddev.item()


class LaplaceSampler(DistributionSampler):
    """Laplace noise distribution sampler."""
    
    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale
        self.distribution = dist.Laplace(loc=loc, scale=scale)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(value)
    
    def sample_n(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is not None:
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = self.distribution.sample((n,))
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            value = self.distribution.sample((n,))
        return value
    
    def std(self) -> float:
        return self.distribution.stddev.item()


# ===== Default Configuration =====

DEFAULT_CONFIG = {
    # Graph parameters
    'N_max': 10,          # Maximum number of variables
    'K_max': 3,           # Maximum number of lags
    'alpha': 2,           # Beta distribution parameter for edge probability (sparse graphs)
    'beta': 5,            # Beta distribution parameter for edge probability
    'gamma': 0.7,         # Lag decay factor
    
    # Mechanism parameters
    'sigma_w': 1.0,       # Weight std
    'sigma_b': 0.5,       # Bias std
    
    # Simulation parameters
    'T': 100,             # Default time series length
    'burn_in': 50,        # Burn-in steps
    
    # Noise parameters
    'root_mean': 0.0,
    'non_root_mean': 0.0,
    
    # Device
    'device': torch.device('cpu'),
    'dtype': torch.float32,
}