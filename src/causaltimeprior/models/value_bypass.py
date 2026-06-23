"""Direct intervention_value → h_causal bypass modules.

The trace probe (scripts/trace_intervention_embedding.py) localized a
failure mode where the cross-attention's W_Q nullifies the intervention_value
direction in attention scores, so the model ignores the do-value magnitude
even though the input projection encodes it strongly. Both training-signal
mitigations (intervention_scale=10 and an aux MSE loss) failed to wake up
G_marker. These modules sidestep the cross-attn entirely by routing
intervention_value into h_causal via a direct learned path.

Both modules are zero-initialized so they start as identity (no behavior
change at step 0) — gradient signal then decides how much to lean on the
bypass vs the cross-attn.
"""

import torch
import torch.nn as nn


class ConcatValueBypass(nn.Module):
    """Concatenate intervention_value to h_causal and project back to E.

    Adds a learned residual: h' = h + W·[h, value]. The projection W is
    initialized to zero so the bypass is identity at step 0.
    """

    def __init__(self, embed_size: int):
        super().__init__()
        self.proj = nn.Linear(embed_size + 1, embed_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # h: (B, E), value: (B,)
        return h + self.proj(torch.cat([h, value.unsqueeze(-1)], dim=-1))


class FiLMValueBypass(nn.Module):
    """FiLM-style scale + shift of h_causal by intervention_value.

    h' = h * (1 + γ·v) + β·v, where γ, β ∈ R^E are learned from value.
    Zero-initialized γ, β so h' = h at step 0.
    """

    def __init__(self, embed_size: int):
        super().__init__()
        self.gamma = nn.Linear(1, embed_size)
        self.beta = nn.Linear(1, embed_size)
        for layer in (self.gamma, self.beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        v = value.unsqueeze(-1)  # (B, 1)
        return h * (1 + self.gamma(v)) + self.beta(v)


def make_value_bypass(mode: str, embed_size: int) -> nn.Module:
    if mode == "none":
        return None
    if mode == "concat":
        return ConcatValueBypass(embed_size)
    if mode == "film":
        return FiLMValueBypass(embed_size)
    raise ValueError(f"Unknown value_bypass mode: {mode!r}")
