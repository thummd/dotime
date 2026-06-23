"""Load a trained Do-Over-Time-PFN checkpoint into a ready-to-eval model.

Checkpoints are ``torch.save`` dicts with ``config`` (the ``DoOverTimePFN``
constructor kwargs), ``model_state_dict``, ``head_type``, and ``borders`` (for the
bar head). The ``config`` does not always record the quantile-head ``tau_levels``
(older checkpoints store ``None`` while the weights carry a fixed number of
levels), so this loader reads the actual levels back from the state dict before
constructing the model — otherwise ``load_state_dict`` size-mismatches.
"""

from __future__ import annotations

from pathlib import Path

import torch

from causaltimeprior.models.do_over_time_pfn import DoOverTimePFN

__all__ = ["load_dotpfn"]


def load_dotpfn(checkpoint: str | Path, device: str = "cpu") -> DoOverTimePFN:
    """Reconstruct a :class:`DoOverTimePFN` from a checkpoint and set it to eval."""
    ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    state = ckpt["model_state_dict"]

    # Recover the true quantile levels from the weights when the config omits them.
    if config.get("head_type") == "quantile" and not config.get("tau_levels"):
        tau = state.get("quantile_head.tau_levels")
        if tau is not None:
            config["tau_levels"] = tau.detach().cpu().tolist()

    model = DoOverTimePFN(**config)
    model.load_state_dict(state, strict=False)

    # Restore the bar distribution's borders for bar-head models.
    if ckpt.get("head_type") == "bar" and "borders" in ckpt and hasattr(model, "bar_head"):
        borders = ckpt["borders"]
        if hasattr(model.bar_head, "set_bar_distribution"):
            from pfns.model.bar_distribution import FullSupportBarDistribution

            model.bar_head.set_bar_distribution(FullSupportBarDistribution(borders), borders)

    model.to(device)
    model.eval()
    # Expose the padded variable width so callers can build matching batches.
    model.n_max = int(config.get("n_max", 41))
    return model
