"""Temporal mechanisms for CausalTimePrior.

This module extends Do-PFN's SimpleMechanism to support time-lagged parents.
"""

import torch
from torch import nn, Tensor
from typing import Dict, List, Optional
import numpy as np

from causaltimeprior._mechanism import SimpleMechanism


class TemporalMechanism(nn.Module):
    """
    Temporal mechanism with support for both instantaneous and lagged parents.
    
    Extends Do-PFN's SimpleMechanism to handle time lags:
    f_i(Pa_instant, Pa_lag1, ..., Pa_lagK) = 
        activation(W_inst·Pa_inst + W_lag1·Pa_lag1 + ... + bias) + noise
    """
    
    def __init__(
        self,
        node_names: List[str],
        activation: nn.Module,
        num_lags: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        sigma_w: float = 1.0,
        sigma_b: float = 0.5,
    ):
        """
        Parameters
        ----------
        node_names : List[str]
            Names of all nodes in the SCM.
        activation : nn.Module
            Activation function to apply.
        num_lags : int
            Maximum number of lags K.
        device : torch.device
            Device for parameters.
        generator : torch.Generator, optional
            RNG for reproducibility.
        sigma_w : float
            Standard deviation for weight initialization.
        sigma_b : float
            Standard deviation for bias initialization.
        """
        super().__init__()
        self.generator = generator
        self.activation = activation
        self.device = device
        self.num_lags = num_lags
        
        # Instantaneous weights (like Do-PFN's SimpleMechanism)
        weights_instant = {}
        for v in node_names:
            initial_value = torch.randn(1, device=device, generator=generator) * sigma_w
            weights_instant[v] = nn.Parameter(initial_value)
        self.weights_instant = nn.ParameterDict(weights_instant)
        
        # Lagged weights for each lag k=1,...,K
        self.weights_lagged = nn.ModuleList()
        for k in range(num_lags):
            weights_k = {}
            for v in node_names:
                initial_value = torch.randn(1, device=device, generator=generator) * sigma_w
                weights_k[v] = nn.Parameter(initial_value)
            self.weights_lagged.append(nn.ParameterDict(weights_k))
        
        # Bias
        bias_value = torch.randn(1, device=device, generator=generator) * sigma_b
        self.bias = nn.Parameter(bias_value)
    
    def forward(
        self,
        parent_values_instant: Dict[str, Tensor],
        parent_values_lagged: List[Dict[str, Tensor]],
        eps: Tensor,
    ) -> Tensor:
        """
        Forward pass with both instantaneous and lagged parents.
        
        Parameters
        ----------
        parent_values_instant : Dict[str, Tensor]
            Current-time parent values {var_name: tensor}.
        parent_values_lagged : List[Dict[str, Tensor]]
            Lagged parent values, list of length K, where each element is
            a dict {var_name: tensor} for lag k.
        eps : Tensor
            Noise term.
            
        Returns
        -------
        Tensor
            Output value.
        """
        # If no parents, return noise only
        if len(parent_values_instant) == 0 and all(len(d) == 0 for d in parent_values_lagged):
            return eps
        
        # Instantaneous contribution
        weighted_instant = []
        for v, weight in self.weights_instant.items():
            if v in parent_values_instant:
                weighted_instant.append(parent_values_instant[v] * weight)
        
        # Lagged contributions
        weighted_lagged = []
        for k, (weights_k, parent_values_k) in enumerate(zip(self.weights_lagged, parent_values_lagged)):
            for v, weight in weights_k.items():
                if v in parent_values_k:
                    weighted_lagged.append(parent_values_k[v] * weight)
        
        # Combine all contributions
        all_weighted = weighted_instant + weighted_lagged
        if len(all_weighted) == 0:
            return eps
        
        combined = torch.sum(torch.stack(all_weighted), dim=0)
        return self.activation(combined + self.bias) + eps