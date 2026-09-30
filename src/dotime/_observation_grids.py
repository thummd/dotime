"""Irregular observation grids for continuous suites (opt-in ``schedules`` key).

A continuous suite config may list observation schedules::

    schedules:
      - {name: regular, kind: regular}
      - {name: jittered, kind: jittered, dt: 1.0, jitter: 0.5, num_substeps: 2}
      - {name: poisson, kind: poisson, rate: 1.0, max_gap: 4.0, num_substeps: 4}

Episode ``idx`` uses ``schedules[idx % len(schedules)]``. A ``regular`` entry
takes no parameters and builds the episode exactly as ``dot-Continuous-v1``
does. The other kinds draw a grid here, from a generator of their own, and the
continuous prior replays it as a fixed grid. The prior's own generators then
start exactly as on the regular grid, so an irregular episode keeps the SCM,
intervention window and intervention value draws of the regular episode with
the same seed.

``dotime._build.make_episode`` is the only caller.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence

import numpy as np
import torch

__all__ = ["SCHEDULE_KINDS", "check_schedule", "draw_grid", "max_substep", "schedule_for"]

#: Schedule kinds a ``schedules`` entry may name.
SCHEDULE_KINDS = ("regular", "jittered", "poisson")

# Salts the grid generator so it shares no stream with any other generator
# derived from the episode seed.
_GRID_SALT = zlib.crc32(b"dotime.observation_grid")

# Smallest poisson gap. Times are stored in float32, whose spacing near t = 256
# is 3e-5, so a floor 30 times larger keeps every recorded gap positive. It
# removes about 0.1% of the Exp(1) mass.
_MIN_GAP = 1e-3

# Largest Euler sub-step. The prior's mean-reversion rates reach 2, and an Euler
# step of size h keeps a mean-reverting variable bounded only while
# rate * h <= 2. The frozen regular grid (h = 1) already sits on that edge, and
# continuous episodes are never retried.
_MAX_SUBSTEP = 1.0

_PARAMETERS = {
    "regular": (),
    "jittered": ("dt", "jitter", "num_substeps"),
    "poisson": ("rate", "max_gap", "num_substeps"),
}


def max_substep(entry: Mapping) -> float:
    """Largest Euler sub-step an entry's grids can produce.

    Args:
        entry: A validated non-regular ``schedules`` entry.

    Returns:
        The largest possible gap divided by ``num_substeps``.
    """
    if entry["kind"] == "jittered":
        largest = float(entry["dt"]) * (1.0 + float(entry["jitter"]))
    else:
        largest = float(entry["max_gap"])
    return largest / int(entry["num_substeps"])


def check_schedule(entry: Mapping) -> None:
    """Validate one ``schedules`` entry.

    Every parameter of a non-regular kind is required, so a release config
    records all the values its grids depend on.

    Args:
        entry: A mapping with ``name``, ``kind`` and the kind's parameters:
            ``dt``, ``jitter`` and ``num_substeps`` for ``jittered``, and
            ``rate``, ``max_gap`` and ``num_substeps`` for ``poisson``.

    Raises:
        ValueError: If the kind is unknown, the name is missing, a parameter is
            missing, unknown or out of range, or a sub-step can exceed 1.0.
    """
    kind = entry.get("kind")
    if kind not in SCHEDULE_KINDS:
        raise ValueError(f"schedule kind {kind!r} is not one of {SCHEDULE_KINDS}")
    if not entry.get("name"):
        raise ValueError(f"schedule {dict(entry)} has no name")
    expected = {"name", "kind", *_PARAMETERS[kind]}
    if set(entry) != expected:
        raise ValueError(
            f"schedule {entry['name']!r} of kind {kind!r} takes exactly {sorted(expected)}, "
            f"got {sorted(entry)}"
        )
    if kind == "regular":
        return
    substeps = entry["num_substeps"]
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError(f"schedule {entry['name']!r}: num_substeps must be a positive int")
    positive = ("dt",) if kind == "jittered" else ("rate", "max_gap")
    for key in positive:
        if not float(entry[key]) > 0:
            raise ValueError(f"schedule {entry['name']!r}: {key} must be positive")
    if kind == "jittered" and not 0.0 <= float(entry["jitter"]) < 1.0:
        raise ValueError(f"schedule {entry['name']!r}: jitter must be in [0, 1)")
    if kind == "poisson" and not float(entry["max_gap"]) > _MIN_GAP:
        raise ValueError(f"schedule {entry['name']!r}: max_gap must exceed {_MIN_GAP}")
    if max_substep(entry) > _MAX_SUBSTEP:
        raise ValueError(
            f"schedule {entry['name']!r} allows Euler sub-steps up to {max_substep(entry):.3g}; "
            f"raise num_substeps until they are at most {_MAX_SUBSTEP}"
        )


def schedule_for(schedules: Sequence[Mapping] | None, idx: int) -> Mapping | None:
    """The validated ``schedules`` entry of episode ``idx``.

    Assignment by index keeps the schedules balanced in every structure and
    draws no random number.

    Args:
        schedules: The suite's ``schedules`` list, or ``None`` if it sets none.
        idx: The episode's index in the suite.

    Returns:
        ``schedules[idx % len(schedules)]``, or ``None`` without schedules.

    Raises:
        ValueError: If the list is empty, names a schedule twice, or holds an
            invalid entry (see :func:`check_schedule`).
    """
    if schedules is None:
        return None
    if len(schedules) == 0:
        raise ValueError("schedules must list at least one schedule")
    for entry in schedules:
        check_schedule(entry)
    names = [entry["name"] for entry in schedules]
    if len(set(names)) != len(names):
        raise ValueError(f"schedule names must be unique, got {names}")
    return schedules[idx % len(schedules)]


def draw_grid(entry: Mapping, t_len: int, seed: int) -> torch.Tensor:
    """Draw the observation times of one episode on an irregular schedule.

    ``jittered`` gaps are ``dt * (1 + jitter * U)`` with ``U ~ Uniform(-1, 1)``.
    ``poisson`` gaps are ``Exp(rate)`` truncated to ``[0.001, max_gap]`` and
    drawn by inverse CDF, one uniform per gap, so the cap leaves no atom. The
    generator is ``np.random.default_rng([salt, seed])``, separate from the
    prior's generators. The grid starts at 0.

    Args:
        entry: A validated ``jittered`` or ``poisson`` entry.
        t_len: Number of observations ``T``.
        seed: The episode seed.

    Returns:
        Strictly increasing float32 times of shape ``(t_len,)``.

    Raises:
        ValueError: If ``entry`` is a ``regular`` entry, which draws no grid,
            or ``t_len`` is below 2.
    """
    if entry["kind"] not in ("jittered", "poisson"):
        raise ValueError(f"schedule kind {entry['kind']!r} draws no grid")
    if t_len < 2:
        raise ValueError(f"an observation grid needs at least 2 times, got {t_len}")
    rng = np.random.default_rng([_GRID_SALT, seed])
    if entry["kind"] == "jittered":
        u = rng.uniform(-1.0, 1.0, size=t_len - 1)
        gaps = float(entry["dt"]) * (1.0 + float(entry["jitter"]) * u)
    else:
        rate, cap = float(entry["rate"]), float(entry["max_gap"])
        hi, lo = np.exp(-rate * _MIN_GAP), np.exp(-rate * cap)
        gaps = -np.log(hi - rng.uniform(size=t_len - 1) * (hi - lo)) / rate
    # Accumulated in float64 and rounded once, so rounding does not compound
    # along the trajectory.
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    return torch.tensor(times, dtype=torch.float32)
