"""Time schedule sampling for continuous-time trajectories.

A schedule is a 1-D float tensor of strictly increasing observation times
t_0 < t_1 < ... < t_{T-1}. The continuous-time simulator steps between
consecutive times via Euler-Maruyama and records the state at each t_i.

Three canonical schedules:

- ``regular_schedule(T, dt)``: uniform grid, reproduces AR(1) dynamics
  when combined with a suitably parameterised OU mechanism.
- ``jittered_schedule(T, dt, jitter)``: regular grid with i.i.d. uniform
  perturbation of each inter-observation gap.
- ``exponential_schedule(T, rate)``: Poisson point process with inter-
  arrival times ~ Exp(rate). Observations are irregularly spaced.

All functions return ``(times, dts)`` where ``times`` has shape ``(T,)``
and ``dts`` has shape ``(T - 1,)`` with ``dts[i] = times[i+1] - times[i]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def regular_schedule(
    T: int,
    dt: float = 1.0,
    t0: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform grid ``t_i = t0 + i * dt`` for ``i = 0, ..., T - 1``.

    Parameters
    ----------
    T : int
        Number of observation points. Must be at least 2.
    dt : float
        Spacing between consecutive observations. Must be positive.
    t0 : float
        Time of the first observation.
    device : torch.device, optional
        Device for the returned tensors.
    dtype : torch.dtype
        Dtype for the returned tensors.
    """
    if T < 2:
        raise ValueError(f"T must be at least 2, got {T}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    times = t0 + dt * torch.arange(T, device=device, dtype=dtype)
    dts = torch.full((T - 1,), float(dt), device=device, dtype=dtype)
    return times, dts


def jittered_schedule(
    T: int,
    dt: float = 1.0,
    jitter: float = 0.2,
    t0: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Regular grid with uniform-noise perturbation of each gap.

    Each inter-observation gap is ``dt * (1 + jitter * U)`` where
    ``U ~ Uniform(-1, 1)``. Guarantees strictly increasing times as long
    as ``0 <= jitter < 1``.
    """
    if T < 2:
        raise ValueError(f"T must be at least 2, got {T}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not 0.0 <= jitter < 1.0:
        raise ValueError(f"jitter must be in [0, 1), got {jitter}")

    u = torch.empty(T - 1, device=device, dtype=dtype)
    u.uniform_(-1.0, 1.0, generator=generator)
    dts = dt * (1.0 + jitter * u)
    times = torch.cumsum(
        torch.cat([torch.tensor([t0], device=device, dtype=dtype), dts]),
        dim=0,
    )
    return times, dts


def exponential_schedule(
    T: int,
    rate: float = 1.0,
    t0: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Poisson point process: inter-arrivals ~ Exp(rate).

    Expected gap between observations is ``1 / rate``; the produced
    schedule is irregular. Use this to model Theophylline-style
    clinically-sampled time series where observation times are not
    controlled.
    """
    if T < 2:
        raise ValueError(f"T must be at least 2, got {T}")
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")

    # Sample exp(1) via inverse CDF of uniform, then divide by rate
    u = torch.empty(T - 1, device=device, dtype=dtype)
    u.uniform_(0.0, 1.0, generator=generator)
    # clamp away from zero/one for numerical stability
    u = u.clamp(min=1e-6, max=1 - 1e-6)
    dts = -torch.log1p(-u) / rate
    times = torch.cumsum(
        torch.cat([torch.tensor([t0], device=device, dtype=dtype), dts]),
        dim=0,
    )
    return times, dts


def from_times(
    times: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrap an arbitrary 1-D tensor of observation times into (times, dts).

    Useful for replaying a fixed clinical schedule or for importing an
    external benchmark's sampling times (e.g. Theophylline PK).
    """
    if times.dim() != 1:
        raise ValueError(f"times must be 1-D, got shape {tuple(times.shape)}")
    if times.numel() < 2:
        raise ValueError(f"times must have at least 2 entries, got {times.numel()}")
    dts = times[1:] - times[:-1]
    if (dts <= 0).any():
        raise ValueError("times must be strictly increasing")
    return times, dts
