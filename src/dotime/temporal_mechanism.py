"""Temporal mechanisms for DoTime.

This module extends Do-PFN's SimpleMechanism to support time-lagged parents.
"""

import torch
from torch import Tensor, nn


class TemporalMechanism(nn.Module):
    """
    Temporal mechanism with support for both instantaneous and lagged parents.

    Extends Do-PFN's SimpleMechanism to handle time lags:
    f_i(Pa_instant, Pa_lag1, ..., Pa_lagK) =
        activation(W_inst·Pa_inst + W_lag1·Pa_lag1 + ... + bias) + noise
    """

    def __init__(
        self,
        node_names: list[str],
        activation: nn.Module,
        num_lags: int,
        device: torch.device,
        generator: torch.Generator | None = None,
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
        for _k in range(num_lags):
            weights_k = {}
            for v in node_names:
                initial_value = torch.randn(1, device=device, generator=generator) * sigma_w
                weights_k[v] = nn.Parameter(initial_value)
            self.weights_lagged.append(nn.ParameterDict(weights_k))

        # Bias
        bias_value = torch.randn(1, device=device, generator=generator) * sigma_b
        self.bias = nn.Parameter(bias_value)

    def rename_nodes(self, mapping: dict[str, str]) -> None:
        """Re-key every weight from an old node name to a new one, in place.

        :meth:`forward` pairs each weight with a parent value by name, so a
        mechanism moved into an SCM with different node names has to be re-keyed
        with it. Otherwise it silently reads no parent at all and returns only its
        noise term. The ``Parameter`` objects are reused and their order is kept,
        so no random numbers are drawn and the renamed mechanism computes, bit for
        bit, what the original computed under the old names.

        Parameters
        ----------
        mapping : dict of str to str
            Old node name to new node name. It must cover every weight key and
            must not send two keys to the same name.

        Raises
        ------
        KeyError
            If a weight key has no entry in ``mapping``.
        ValueError
            If ``mapping`` sends two weight keys to the same new name.
        """
        # Validate every dict before replacing any, so a bad mapping leaves the
        # mechanism untouched.
        for weights in (self.weights_instant, *self.weights_lagged):
            missing = [v for v in weights if v not in mapping]
            if missing:
                raise KeyError(f"mapping has no entry for weight keys {missing}")
            renamed = [mapping[v] for v in weights]
            if len(set(renamed)) != len(renamed):
                raise ValueError(f"mapping sends two weight keys to one name: {renamed}")
        # Built from (key, value) pairs because ParameterDict re-sorts a plain
        # dict by key, and forward() sums the weighted parents in dict order.
        # Keeping the order keeps the floating-point sum identical.
        self.weights_instant = nn.ParameterDict(
            [(mapping[v], w) for v, w in self.weights_instant.items()]
        )
        self.weights_lagged = nn.ModuleList(
            nn.ParameterDict([(mapping[v], w) for v, w in weights_k.items()])
            for weights_k in self.weights_lagged
        )

    def forward(
        self,
        parent_values_instant: dict[str, Tensor],
        parent_values_lagged: list[dict[str, Tensor]],
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
        for _k, (weights_k, parent_values_k) in enumerate(
            zip(self.weights_lagged, parent_values_lagged, strict=False)
        ):
            for v, weight in weights_k.items():
                if v in parent_values_k:
                    weighted_lagged.append(parent_values_k[v] * weight)

        # Combine all contributions
        all_weighted = weighted_instant + weighted_lagged
        if len(all_weighted) == 0:
            return eps

        combined = torch.sum(torch.stack(all_weighted), dim=0)
        return self.activation(combined + self.bias) + eps
