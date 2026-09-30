"""Seasonal and trend confounding drivers for the named-structure generator.

A driven structure label ``"<base>+<kind>_<visibility>"``, for example
``"back_door+seasonal_hidden"``, adds one exogenous driver ``D`` to a named
:class:`~dotime.tscm_sampler.TSCMStructure`. ``D`` is a deterministic function of
time, a sinusoid (``seasonal``) or a linear ramp (``trend``), whose parameters are
drawn once per episode. It is a root of the DAG with instantaneous edges ``D -> A``
and ``D -> Y``, and it enters both structural equations additively after the
activation::

    A_t = f_A(parents of A) + loading_A * D_t + eps_A(t)
    Y_t = f_Y(parents of Y) + loading_Y * D_t + eps_Y(t)

Under counterfactual pairing both arms share ``D``. It is exogenous, so ``do(A)``
cannot move it, and the arms still differ only through the intervention. An
``observed`` driver is released as a column whose future values are known in
advance, like a calendar feature. A ``hidden`` driver is zeroed in both released
arms, like the hidden confounder ``U``.

A hidden driver confounds ``A`` and ``Y`` through time. Because it is a
deterministic function of time, the effect stays identifiable in principle by
modelling time, for example with trend or seasonal terms as in an interrupted time
series. What it defeats are estimators that neither adjust for ``D`` nor model
time, and estimators that assume a stationary series.

Every driver draw comes from a generator derived from the episode seed under its
own salt, so the episode generator draws the base structure's mechanisms, noise
and intervention exactly as it does without a driver.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn

from dotime._sampling import DistributionSampler
from dotime.temporal_graph import TemporalDAG
from dotime.temporal_mechanism import TemporalMechanism
from dotime.temporal_scm import TemporalSCM
from dotime.tscm_sampler import TSCMStructure

__all__ = [
    "DRIVER_KINDS",
    "DRIVER_NODE",
    "DriverDraw",
    "DriverSpec",
    "driver_series",
    "parse_structure_label",
    "released_driver_series",
]

#: Driver time-series families, the ``<kind>`` part of a driven label.
DRIVER_KINDS = ("seasonal", "trend")
#: Name of the driver node in the driven DAG.
DRIVER_NODE = "D"
#: ``<visibility>`` part of a driven label -> whether ``D`` is released.
_VISIBILITIES = {"observed": True, "hidden": False}
_BASE_STRUCTURES = frozenset(s.value for s in TSCMStructure)
#: Range of the seasonal period in time steps.
SEASONAL_PERIOD_RANGE = (12.0, 48.0)
#: Range of each loading's magnitude before ``strength`` scales it.
LOADING_RANGE = (0.5, 1.0)
_DRIVER_SALT = zlib.crc32(b"dotime.drivers")


@dataclass(frozen=True)
class DriverSpec:
    """Which driver a driven structure label adds to its base structure.

    Args:
        kind: One of :data:`DRIVER_KINDS`.
        observed: Whether ``D`` is released as a column. A hidden ``D`` is
            zeroed in both released arms.
        strength: Scales both loadings. ``0.0`` switches the driver off but
            keeps every draw, which reproduces the base structure bit for bit.

    Raises:
        ValueError: If ``kind`` is unknown or ``strength`` is negative or not
            finite.
    """

    kind: str
    observed: bool
    strength: float = 1.0

    def __post_init__(self) -> None:
        """Validate the fields.

        Returns:
            None.

        Raises:
            ValueError: If ``kind`` is unknown or ``strength`` is negative or
                not finite.
        """
        if self.kind not in DRIVER_KINDS:
            raise ValueError(f"driver kind must be one of {DRIVER_KINDS}, got {self.kind!r}")
        if not (math.isfinite(self.strength) and self.strength >= 0.0):
            raise ValueError(f"driver strength must be finite and >= 0, got {self.strength!r}")

    @property
    def suffix(self) -> str:
        """The ``<kind>_<visibility>`` part of a driven label.

        Returns:
            For example ``"seasonal_hidden"``.

        Raises:
            Nothing. The fields were validated at construction.
        """
        return f"{self.kind}_{'observed' if self.observed else 'hidden'}"


def parse_structure_label(label: str) -> tuple[str, DriverSpec | None]:
    """Split a structure label into its base structure and its driver.

    A plain label such as ``"back_door"`` is returned unchanged and unvalidated,
    exactly as the generator received it before drivers existed. A driven label
    ``"<base>+<kind>_<visibility>"`` must name a current
    :class:`~dotime.tscm_sampler.TSCMStructure` value as its base.

    Args:
        label: A structure label, e.g. ``"bi_variate"`` or
            ``"back_door+trend_observed"``.

    Returns:
        ``(base, spec)``, where ``spec`` is ``None`` for a plain label and a
        :class:`DriverSpec` with ``strength=1.0`` otherwise.

    Raises:
        ValueError: If a driven label has an unknown base, kind or visibility,
            or more than one driver.
    """
    if "+" not in label:
        return label, None
    base, _, suffix = label.partition("+")
    kind, _, visibility = suffix.rpartition("_")
    if base not in _BASE_STRUCTURES or kind not in DRIVER_KINDS or visibility not in _VISIBILITIES:
        raise ValueError(
            f"malformed structure label {label!r}: expected '<base>+<kind>_<visibility>' "
            f"with base a TSCMStructure value, kind in {DRIVER_KINDS} and visibility in "
            f"{tuple(_VISIBILITIES)}"
        )
    return base, DriverSpec(kind=kind, observed=_VISIBILITIES[visibility])


def driver_generator(seed: int) -> torch.Generator:
    """Private generator for an episode's driver draws.

    Salting the episode seed gives the driver a stream of its own, so drawing it
    never advances the episode generator that samples the base structure.

    Args:
        seed: The episode seed, i.e. the ``seed`` of the ``TSCMPrior``. Negative
            seeds are folded into ``[0, 2**64)``.

    Returns:
        A CPU generator seeded from ``SeedSequence([seed, crc32(b"dotime.drivers")])``.

    Raises:
        TypeError: If ``seed`` is not an integer.
    """
    entropy = [seed & (2**64 - 1), _DRIVER_SALT]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]
    return torch.Generator().manual_seed(int(state))


def driver_series(kind: str, params: Mapping[str, float], total_t: int) -> torch.Tensor:
    """The driver's value at every simulated step, burn-in included.

    ``seasonal`` is ``sin(2 pi t / period + phase)`` and ``trend`` is
    ``direction * (2 t / (total_t - 1) - 1)``, a ramp from ``-direction`` to
    ``direction``. Both stay within ``[-1, 1]``.

    Args:
        kind: One of :data:`DRIVER_KINDS`.
        params: ``period`` and ``phase`` for ``seasonal``, ``direction``
            (``+1`` or ``-1``) for ``trend``.
        total_t: Number of simulated steps, burn-in plus released length.

    Returns:
        A float64 tensor of shape ``(total_t,)``.

    Raises:
        ValueError: If ``kind`` is unknown or ``total_t < 2``.
        KeyError: If ``params`` lacks a parameter of ``kind``.
    """
    if total_t < 2:
        raise ValueError(f"a driver series needs at least 2 steps, got {total_t}")
    t = torch.arange(total_t, dtype=torch.float64)
    if kind == "seasonal":
        return torch.sin(2.0 * math.pi * t / params["period"] + params["phase"])
    if kind == "trend":
        return params["direction"] * (2.0 * t / (total_t - 1) - 1.0)
    raise ValueError(f"driver kind must be one of {DRIVER_KINDS}, got {kind!r}")


def released_driver_series(driver: Mapping[str, Any], length: int) -> torch.Tensor:
    """Rebuild an episode's driver on its released rows from ``metadata["driver"]``.

    A hidden driver is zeroed in the released arms, so this is the only way to
    read it back, for example to fit an oracle that adjusts for it.

    Args:
        driver: An episode's ``metadata["driver"]``.
        length: Number of released rows ``T``.

    Returns:
        A float32 tensor of shape ``(length,)``, equal bit for bit to the
        released column of an observed driver.

    Raises:
        ValueError: If the recorded kind is unknown.
        KeyError: If ``driver`` lacks ``kind``, ``params`` or ``burn_in``.
    """
    burn_in = int(driver["burn_in"])
    series = driver_series(driver["kind"], driver["params"], burn_in + length)
    return series[burn_in:].to(torch.float32)


@dataclass(frozen=True)
class DriverDraw:
    """One episode's realisation of a driver.

    Args:
        spec: The driver that was drawn.
        params: Series parameters: ``period`` and ``phase``, or ``direction``.
        loading_a: Coefficient of ``D_t`` in the equation of ``A``.
        loading_y: Coefficient of ``D_t`` in the equation of ``Y``.
        confounding_sign: Sign of ``loading_a * loading_y``, ``0`` when
            ``spec.strength`` is ``0``.

    Raises:
        Nothing. The fields are not validated, :func:`draw_driver` builds
        consistent draws.
    """

    spec: DriverSpec
    params: dict[str, float]
    loading_a: float
    loading_y: float
    confounding_sign: int

    def series(self, total_t: int) -> torch.Tensor:
        """The drawn series over ``total_t`` simulated steps.

        Args:
            total_t: Burn-in plus released length.

        Returns:
            A float64 tensor of shape ``(total_t,)``.

        Raises:
            ValueError: If ``total_t < 2``.
        """
        return driver_series(self.spec.kind, self.params, total_t)

    def metadata(self, *, column: int, burn_in: int, generation_seed: int) -> dict[str, Any]:
        """JSON-able record of the draw for ``Episode.metadata["driver"]``.

        Args:
            column: Canonical column of ``D`` in the episode.
            burn_in: Simulated steps before the released window.
            generation_seed: Seed the driver generator was derived from.

        Returns:
            A dict of plain Python values: ``kind``, ``observed``, ``column``,
            ``strength``, ``params``, ``loadings`` (``A`` and ``Y``),
            ``confounding_sign``, ``known_future``, ``burn_in`` and
            ``generation_seed``.

        Raises:
            Nothing. Every value is converted to a plain Python type.
        """
        return {
            "kind": self.spec.kind,
            "observed": self.spec.observed,
            "column": int(column),
            "strength": float(self.spec.strength),
            "params": dict(self.params),
            "loadings": {"A": float(self.loading_a), "Y": float(self.loading_y)},
            "confounding_sign": int(self.confounding_sign),
            # D is exogenous and deterministic in time, so its values after the
            # onset are legitimately available to a forecaster.
            "known_future": self.spec.observed,
            "burn_in": int(burn_in),
            "generation_seed": int(generation_seed),
        }


def draw_driver(spec: DriverSpec, generator: torch.Generator) -> DriverDraw:
    """Draw one episode's series parameters and loadings.

    Args:
        spec: The driver to draw.
        generator: The driver generator from :func:`driver_generator`.

    Returns:
        The draw. Loadings are ``+-U[0.5, 1) * strength``, independently for
        ``A`` and ``Y``. A seasonal period is ``U[12, 48)`` and its phase
        ``U[0, 2 pi)``. A trend's direction is ``+1`` or ``-1`` with equal
        probability.

    Raises:
        RuntimeError: If ``generator`` is not a CPU generator.
    """
    # One fixed-size draw for both kinds, so a seed gives the same loadings
    # whether the label asks for a seasonal or a trend driver.
    u = torch.rand(6, generator=generator, dtype=torch.float64).tolist()
    low, high = LOADING_RANGE
    sign_a = 1 if u[0] < 0.5 else -1
    sign_y = 1 if u[2] < 0.5 else -1
    loading_a = sign_a * (low + (high - low) * u[1]) * spec.strength
    loading_y = sign_y * (low + (high - low) * u[3]) * spec.strength
    params: dict[str, float]
    if spec.kind == "seasonal":
        p_low, p_high = SEASONAL_PERIOD_RANGE
        params = {"period": p_low + (p_high - p_low) * u[4], "phase": 2.0 * math.pi * u[5]}
    else:
        params = {"direction": 1 if u[4] < 0.5 else -1}
    return DriverDraw(
        spec=spec,
        params=params,
        loading_a=loading_a,
        loading_y=loading_y,
        confounding_sign=sign_a * sign_y if spec.strength > 0.0 else 0,
    )


def driven_dag(base: TemporalDAG) -> TemporalDAG:
    """Prepend the driver node to a base structure's DAG.

    ``D`` becomes the first node of the topological order, a root with
    instantaneous edges ``D -> A`` and ``D -> Y``. Every lag matrix gains a zero
    first row and column, so ``D`` has no self-loop and no lagged edge.

    Args:
        base: The base structure's DAG. It must contain ``A`` and ``Y`` and no
            node named ``D``.

    Returns:
        The driven DAG. The base DAG is not modified.

    Raises:
        ValueError: If ``base`` lacks ``A`` or ``Y`` or already has a ``D``.
    """
    nodes = set(base.topo_order)
    if DRIVER_NODE in nodes or not {"A", "Y"} <= nodes:
        raise ValueError(f"cannot add a driver to a DAG over {base.topo_order}")
    g_0 = nx.DiGraph()
    g_0.add_node(DRIVER_NODE)
    g_0.add_nodes_from(base.G_0.nodes)
    g_0.add_edges_from(base.G_0.edges)
    g_0.add_edges_from([(DRIVER_NODE, "A"), (DRIVER_NODE, "Y")])
    g_lags = [np.pad(g_k, ((1, 0), (1, 0))) for g_k in base.G_lags]
    return TemporalDAG(G_0=g_0, G_lags=g_lags, K=base.K, topo_order=[DRIVER_NODE, *base.topo_order])


def driver_mechanism(num_lags: int, device: torch.device) -> TemporalMechanism:
    """The mechanism of ``D``, which returns its noise term, i.e. the series.

    Args:
        num_lags: Maximum lag ``K`` of the driven DAG.
        device: Device of the SCM.

    Returns:
        A mechanism without weights. ``D`` has no parents, so
        :meth:`TemporalMechanism.forward` returns its noise and the bias is
        never read.

    Raises:
        RuntimeError: If ``device`` is not a valid torch device.
    """
    # TemporalMechanism draws its bias at construction. A private generator keeps
    # that draw off both the episode generator and the global torch RNG.
    return TemporalMechanism(
        node_names=[],
        activation=nn.Identity(),
        num_lags=num_lags,
        device=device,
        generator=torch.Generator(device=device),
    )


class _SeriesSampler(DistributionSampler):
    """Noise sampler of ``D``: returns the episode's series and draws nothing.

    Args:
        series: The driver series over every simulated step.

    Raises:
        Nothing at construction.
    """

    def __init__(self, series: torch.Tensor) -> None:
        self.series = series

    def sample_n(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Return a copy of the series, leaving ``generator`` untouched.

        Args:
            n: Number of steps requested, which must equal the series length.
            generator: Ignored. The series is fixed for the episode.

        Returns:
            A copy, so a caller that edits the frozen noise cannot edit the series.

        Raises:
            ValueError: If ``n`` differs from the series length.
        """
        if n != self.series.numel():
            raise ValueError(f"driver series has {self.series.numel()} steps, {n} requested")
        return self.series.clone()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Not defined: the series is a fixed path, not a distribution over steps.

        Args:
            value: Ignored.

        Returns:
            Never returns.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("a driver series is deterministic and has no density")

    def std(self) -> float:
        """Spread of the per-episode noise given the draw, which is fixed.

        Returns:
            ``0.0``.

        Raises:
            Nothing.
        """
        return 0.0


def attach_driver(
    scm: TemporalSCM, dag: TemporalDAG, series: torch.Tensor, mechanism: TemporalMechanism
) -> TemporalSCM:
    """Wrap a sampled base SCM into the driven SCM over ``dag``.

    The base mechanisms and noise samplers are reused as they are. A base
    mechanism has no weight for ``D``, so the new edges ``D -> A`` and
    ``D -> Y`` change none of its arithmetic, and ``D`` reaches ``A`` and ``Y``
    only through :func:`add_forcing`.

    Args:
        scm: The base SCM drawn by ``TSCMSampler.sample``.
        dag: The driven DAG from :func:`driven_dag`.
        series: The driver series over every simulated step.
        mechanism: The mechanism of ``D`` from :func:`driver_mechanism`.

    Returns:
        A new SCM whose first node is ``D``.

    Raises:
        Nothing. A mismatch between ``dag`` and ``scm`` surfaces when the SCM
        is simulated.
    """
    return TemporalSCM(
        dag,
        {DRIVER_NODE: mechanism, **scm.mechanisms},
        {DRIVER_NODE: _SeriesSampler(series), **scm.noise},
        device=scm.device,
        dtype=scm.dtype,
    )


def add_forcing(noise: dict[str, torch.Tensor], draw: DriverDraw, series: torch.Tensor) -> None:
    """Add ``loading * series`` to the frozen noise of ``A`` and ``Y``.

    The mechanism adds its noise after the activation, so this makes ``D`` an
    additive term of both structural equations. The entries are replaced, never
    edited in place.

    Args:
        noise: The dict returned by ``TemporalSCM.freeze_noise``.
        draw: The episode's draw.
        series: The float64 driver series over every simulated step.

    Returns:
        None.

    Raises:
        KeyError: If ``noise`` has no entry for ``A`` or ``Y``.
    """
    for node, loading in (("A", draw.loading_a), ("Y", draw.loading_y)):
        noise[node] = noise[node] + (loading * series).to(noise[node].dtype)
