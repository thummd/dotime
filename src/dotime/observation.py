"""Observation layer: measurement error and missingness applied after simulation.

DoTime simulates latent trajectories, while real sensor logs are noisy,
quantized, censored at the edges of a sensor's range and full of gaps. This
module turns a latent :class:`~dotime.benchmarks.Episode` into an observed one.
The targets stay the latent true values, so an observed episode asks the same
causal question of worse data.

**Public surface**

- :class:`MeasurementModel` — additive noise at a signal-to-noise ratio,
  censoring at quantiles and quantization.
- :class:`MissingnessModel` — ``none``, ``mcar``, ``block`` or ``mnar`` gaps.
- :class:`ObservationModel` — one measurement and one missingness model, i.e.
  one cell of a factorial design, with ``from_dict`` / ``to_dict``.
- :func:`cells_from_config` — expand an ``observation:`` build config into its
  named cells.
- :func:`apply_observation` — observe one latent episode.
- :func:`impute_history` and :func:`impute_episode` — the imputation that
  :func:`dotime.evaluation.evaluate` gives models that are not mask-aware.
- :func:`require_finite_history` — the "impute first" guard of the reference
  evaluators.
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

if TYPE_CHECKING:
    from dotime.benchmarks import Episode

__all__ = [
    "MISSINGNESS_KINDS",
    "MeasurementModel",
    "MissingnessModel",
    "ObservationModel",
    "apply_observation",
    "cells_from_config",
    "impute_episode",
    "impute_history",
    "require_finite_history",
]

#: Missingness mechanisms understood by :class:`MissingnessModel`.
MISSINGNESS_KINDS = ("none", "mcar", "block", "mnar")

# Salt of the episode-derived observation RNG. The simulation seeds the global
# torch RNG, numpy's RandomState and per-prior generators from the same episode
# seed, so a salted SeedSequence keeps these streams independent of all of them.
_SALT = zlib.crc32(b"dotime.observation")

# One independent child stream per random component, in spawn order. Each
# component draws a fixed amount whatever the cell asks of it, so all cells of
# one latent episode share their random numbers (common random numbers across
# the factorial design). New components must be appended: SeedSequence children
# are indexed, so appending leaves every existing stream unchanged.
_COMPONENTS = ("noise", "cell", "block", "keep")

_MEASUREMENT_KEYS = ("snr", "censor_quantiles", "quantize_step")
_MISSINGNESS_KEYS = ("kind", "rate", "block_len", "mnar_quantile")
_CONFIG_KEYS = ("latent_per_structure", "measurement", "missingness")


def _positive(name: str, value: Any) -> float:
    """Validate a strictly positive, finite number.

    Args:
        name: Field name for the error message.
        value: The value to check.

    Returns:
        ``value`` as a float.

    Raises:
        ValueError: If ``value`` is not a finite number above zero.
    """
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
    return number


def _unknown_keys(where: str, data: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    """Refuse keys a config section does not define, so a typo cannot pass silently.

    Args:
        where: Name of the section for the error message.
        data: The section.
        allowed: Its valid keys.

    Raises:
        ValueError: If ``data`` has a key outside ``allowed``.
    """
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {where} keys {unknown}; expected a subset of {list(allowed)}")


@dataclass(frozen=True)
class MeasurementModel:
    """How a sensor reads a latent value: additive noise, censoring, quantization.

    Every scale is per column and relative to ``sd``, the column's latent
    standard deviation over the rows before the intervention onset (the
    population standard deviation, over the whole trajectory when fewer than
    two rows precede the onset). The steps run in the order noise, censoring,
    quantization. A column with ``sd == 0`` has no scale and is read exactly.

    Attributes:
        snr: Signal power over noise power. Gaussian noise with standard
            deviation ``sd / sqrt(snr)`` is added, so ``snr=10`` adds noise with
            a tenth of the signal's variance. ``None`` adds no noise.
        censor_quantiles: ``(lo, hi)`` quantiles of the column's latent
            pre-onset values. Observed values are clipped to the range between
            them, as a sensor saturates at the ends of its range. Either side
            may be ``None`` to censor one side only. ``None`` censors nothing.
        quantize_step: Resolution in units of ``sd``. Observed values are
            rounded to the nearest multiple of ``quantize_step * sd``, ties to
            even. ``None`` keeps full precision.
    """

    snr: float | None = None
    censor_quantiles: tuple[float | None, float | None] | None = None
    quantize_step: float | None = None

    def __post_init__(self) -> None:
        """Validate the fields and coerce sequences to tuples.

        Raises:
            ValueError: If ``snr`` or ``quantize_step`` is not a positive finite
                number, or ``censor_quantiles`` is not a pair of quantiles in
                ``[0, 1]`` with ``lo < hi``.
        """
        if self.snr is not None:
            object.__setattr__(self, "snr", _positive("snr", self.snr))
        if self.quantize_step is not None:
            object.__setattr__(
                self, "quantize_step", _positive("quantize_step", self.quantize_step)
            )
        if self.censor_quantiles is not None:
            bounds = tuple(self.censor_quantiles)
            if len(bounds) != 2:
                raise ValueError(
                    f"censor_quantiles must be (lo, hi), got {self.censor_quantiles!r}"
                )
            lo, hi = (None if b is None else float(b) for b in bounds)
            for q in (lo, hi):
                if q is not None and not 0.0 <= q <= 1.0:
                    raise ValueError(f"censor_quantiles must lie in [0, 1], got {bounds!r}")
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError(f"censor_quantiles needs lo < hi, got {bounds!r}")
            object.__setattr__(self, "censor_quantiles", (lo, hi))

    @property
    def is_identity(self) -> bool:
        """Whether the model reads every value exactly."""
        return self.snr is None and self.censor_quantiles is None and self.quantize_step is None


@dataclass(frozen=True)
class MissingnessModel:
    """Which observed cells go missing (become ``NaN``).

    Attributes:
        kind: ``"none"``; ``"mcar"``, where each cell is missing independently
            with probability ``rate``; ``"block"``, where each column has one
            contiguous gap with probability ``rate``, its length uniform on
            ``block_len`` and its start uniform over the rows where it fits;
            or ``"mnar"``, where each cell whose latent value exceeds the
            column's ``mnar_quantile`` quantile of latent pre-onset values is
            missing with probability ``rate``.
        rate: Probability in ``[0, 1]`` whose unit depends on ``kind`` (a cell,
            a column, a high cell). Must be 0 for ``"none"``.
        block_len: Inclusive ``(lo, hi)`` range of gap lengths in rows for
            ``"block"``, with ``1 <= lo <= hi``. A gap longer than the
            trajectory covers all of it.
        mnar_quantile: Threshold quantile in ``(0, 1)`` for ``"mnar"``.
    """

    kind: str = "none"
    rate: float = 0.0
    block_len: tuple[int, int] = (10, 40)
    mnar_quantile: float = 0.85

    def __post_init__(self) -> None:
        """Validate the fields and coerce ``block_len`` to a tuple of ints.

        Raises:
            ValueError: If ``kind`` is unknown, ``rate`` lies outside ``[0, 1]``
                or is nonzero for ``"none"``, ``block_len`` is not an ordered
                pair of positive integers, or ``mnar_quantile`` lies outside
                ``(0, 1)``.
        """
        if self.kind not in MISSINGNESS_KINDS:
            raise ValueError(f"kind must be one of {MISSINGNESS_KINDS}, got {self.kind!r}")
        rate = float(self.rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must lie in [0, 1], got {self.rate!r}")
        if self.kind == "none" and rate != 0.0:
            raise ValueError(f"kind 'none' drops no cells, so rate must be 0, got {self.rate!r}")
        object.__setattr__(self, "rate", rate)
        lengths = tuple(self.block_len)
        if (
            len(lengths) != 2
            or any(int(v) != v for v in lengths)
            or not 1 <= int(lengths[0]) <= int(lengths[1])
        ):
            raise ValueError(f"block_len must be integers 1 <= lo <= hi, got {self.block_len!r}")
        object.__setattr__(self, "block_len", (int(lengths[0]), int(lengths[1])))
        quantile = float(self.mnar_quantile)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"mnar_quantile must lie in (0, 1), got {self.mnar_quantile!r}")
        object.__setattr__(self, "mnar_quantile", quantile)


@dataclass(frozen=True)
class ObservationModel:
    """One cell of an observation design: a measurement and a missingness model.

    Attributes:
        measurement: How each value is read.
        missingness: Which values go missing.
        name: Label of the cell, e.g. ``"snr10+mcar10"``. It is recorded as the
            episode's ``obs_cell`` metadata. ``None`` for an unnamed model.
    """

    measurement: MeasurementModel = field(default_factory=MeasurementModel)
    missingness: MissingnessModel = field(default_factory=MissingnessModel)
    name: str | None = None

    @property
    def is_identity(self) -> bool:
        """Whether observing an episode leaves every value unchanged."""
        return self.measurement.is_identity and self.missingness.kind == "none"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationModel:
        """Build a model from its :meth:`to_dict` form or a config entry.

        Args:
            data: Mapping with optional ``name``, ``measurement`` (fields of
                :class:`MeasurementModel`) and ``missingness`` (fields of
                :class:`MissingnessModel`). Missing fields take their defaults
                and lists stand in for tuples, as YAML and JSON produce them.

        Returns:
            The model.

        Raises:
            ValueError: On an unknown key or an invalid field value.
        """
        _unknown_keys("observation", data, ("name", "measurement", "missingness"))
        measurement = dict(data.get("measurement") or {})
        missingness = dict(data.get("missingness") or {})
        _unknown_keys("measurement", measurement, _MEASUREMENT_KEYS)
        _unknown_keys("missingness", missingness, _MISSINGNESS_KEYS)
        name = data.get("name")
        return cls(
            measurement=MeasurementModel(**measurement),
            missingness=MissingnessModel(**missingness),
            name=None if name is None else str(name),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view of the model (round-trips via :meth:`from_dict`).

        Returns:
            Dict with ``name``, ``measurement`` and ``missingness``, every
            field spelled out.
        """
        meas, miss = self.measurement, self.missingness
        return {
            "name": self.name,
            "measurement": {
                "snr": meas.snr,
                "censor_quantiles": (
                    None if meas.censor_quantiles is None else list(meas.censor_quantiles)
                ),
                "quantize_step": meas.quantize_step,
            },
            "missingness": {
                "kind": miss.kind,
                "rate": miss.rate,
                "block_len": list(miss.block_len),
                "mnar_quantile": miss.mnar_quantile,
            },
        }


def cells_from_config(config: Mapping[str, Any]) -> list[ObservationModel]:
    """Expand an ``observation:`` build config into its named factorial cells.

    Args:
        config: Mapping with ``measurement`` and ``missingness`` sections, each
            mapping a level name to the fields of :class:`MeasurementModel` or
            :class:`MissingnessModel` (an empty entry means the defaults: no
            noise, no gaps). A missing section has the single level ``none``.
            The build key ``latent_per_structure`` is allowed and ignored here.

    Returns:
        One model per (measurement level, missingness level) pair,
        measurement-major in config order, named
        ``"<measurement level>+<missingness level>"``.

    Raises:
        ValueError: On an unknown key, an empty section or an invalid level.
    """
    _unknown_keys("observation config", config, _CONFIG_KEYS)
    sections = []
    for key in ("measurement", "missingness"):
        levels = config.get(key, {"none": {}})
        if not isinstance(levels, Mapping) or not levels:
            raise ValueError(f"observation {key} must map level names to settings, got {levels!r}")
        sections.append(levels)
    measurement, missingness = sections
    return [
        ObservationModel.from_dict(
            {"name": f"{m_name}+{k_name}", "measurement": m_cfg, "missingness": k_cfg}
        )
        for m_name, m_cfg in measurement.items()
        for k_name, k_cfg in missingness.items()
    ]


def _onset(episode: Episode) -> int:
    """First post-intervention row of an episode, clamped to ``[0, T]``.

    Args:
        episode: The episode.

    Returns:
        ``min(intervention.times)``, or ``T`` when the intervention has no
        times (the whole trajectory is history).
    """
    t_len = episode.length
    times = episode.intervention.times
    return min(max(int(min(times)), 0), t_len) if times else t_len


class _Scales(NamedTuple):
    """Per-column sensor parameters, fixed by the latent observational arm."""

    sd: np.ndarray
    noise_sd: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None
    step: np.ndarray | None


def _calibrate(reference: np.ndarray, measurement: MeasurementModel) -> _Scales:
    """Per-column noise scale, censoring bounds and quantization step.

    Args:
        reference: Latent observational rows that calibrate the sensor, shape
            ``(rows, N)``.
        measurement: The measurement model.

    Returns:
        The column-wise parameters, ``None`` for a step the model skips.
    """
    sd = reference.std(axis=0)
    noise_sd = None if measurement.snr is None else sd / math.sqrt(measurement.snr)
    lower = upper = None
    if measurement.censor_quantiles is not None:
        lo, hi = measurement.censor_quantiles
        lower = None if lo is None else np.quantile(reference, lo, axis=0)
        upper = None if hi is None else np.quantile(reference, hi, axis=0)
    step = None if measurement.quantize_step is None else measurement.quantize_step * sd
    return _Scales(sd, noise_sd, lower, upper, step)


def _read(x: np.ndarray, z: np.ndarray | None, scales: _Scales, live: np.ndarray) -> np.ndarray:
    """Read latent values through the sensor: noise, then censoring, then quantization.

    Args:
        x: Latent values, shape ``(rows, N)``.
        z: Standard normal draws aligned with ``x``, or ``None`` without noise.
        scales: Column-wise parameters from :func:`_calibrate`.
        live: Boolean ``(N,)`` mask of the columns to read. The others (all
            zero: hidden or diverged) are returned untouched.

    Returns:
        A new array with the observed values.
    """
    out = x.copy()
    if scales.noise_sd is not None and z is not None:
        out[:, live] += z[:, live] * scales.noise_sd[live]
    # A column that is constant before the onset has no scale: its censoring
    # range would collapse to one value and its quantization step to 0.
    scaled = live & (scales.sd > 0)
    if scales.lower is not None:
        out[:, scaled] = np.maximum(out[:, scaled], scales.lower[scaled])
    if scales.upper is not None:
        out[:, scaled] = np.minimum(out[:, scaled], scales.upper[scaled])
    if scales.step is not None:
        out[:, scaled] = np.round(out[:, scaled] / scales.step[scaled]) * scales.step[scaled]
    return out


def _missing_cells(
    latent: np.ndarray,
    reference: np.ndarray,
    missingness: MissingnessModel,
    rngs: dict[str, np.random.Generator],
) -> np.ndarray:
    """Draw the missingness mask of the observational arm.

    Args:
        latent: Latent observational trajectory, shape ``(T, N)``.
        reference: Its pre-onset rows, which set the MNAR thresholds.
        missingness: The missingness model.
        rngs: The episode's component generators.

    Returns:
        Boolean ``(T, N)`` array, ``True`` where a cell goes missing.
    """
    t_len, n_vars = latent.shape
    if missingness.kind == "none":
        return np.zeros((t_len, n_vars), dtype=bool)
    if missingness.kind in ("mcar", "mnar"):
        # MCAR and MNAR read the same uniforms, so at equal rates an MNAR gap
        # is an MCAR gap restricted to high values.
        missing = rngs["cell"].random((t_len, n_vars)) < missingness.rate
        if missingness.kind == "mnar":
            # Strictly above the threshold: MNAR drops high values only.
            missing &= latent > np.quantile(reference, missingness.mnar_quantile, axis=0)
        return missing
    rng = rngs["block"]
    has_gap = rng.random(n_vars) < missingness.rate
    lo, hi = missingness.block_len
    length = np.minimum(rng.integers(lo, hi + 1, size=n_vars), t_len)
    start = np.floor(rng.random(n_vars) * (t_len - length + 1)).astype(np.int64)
    rows = np.arange(t_len)[:, None]
    return has_gap & (rows >= start) & (rows < start + length)


def apply_observation(episode: Episode, model: ObservationModel, seed: int) -> Episode:
    """Observe a latent episode through a measurement and a missingness model.

    Every row of ``x_obs`` and the rows of ``x_int`` before the intervention
    onset are read with the same noise draws, the same sensor parameters and
    the same missingness mask, so shared-noise (counterfactual) arms still
    agree before the onset, missing cells included. ``x_int`` from the onset
    on, ``y_true`` and the ``y_oracle`` and ``y_causal_effect`` metadata stay
    latent: the targets are the true values. Missing cells are ``NaN``.

    Some cells are never missing. A query cell (``query_time_idx``,
    ``query_target``) of ``x_obs`` stays observed, and a column that is all
    zero in ``x_obs`` (a hidden variable or a diverged arm) is neither noised
    nor masked in either arm, nor is an all-zero ``x_int`` column. When the
    mask covers every pre-onset cell of a column, one of them, drawn at
    random, stays observed.

    The random numbers come from ``np.random.SeedSequence([salt, seed])``,
    spawned into one independent stream per component (noise, cell uniforms
    shared by MCAR and MNAR, blocks, the kept cell). Every cell of a design
    applied to one latent episode with its episode seed therefore shares its
    draws, and no global torch or numpy RNG state is read or advanced.

    Args:
        episode: A latent episode with finite ``x_obs`` and ``x_int``.
        model: The observation model.
        seed: The episode seed.

    Returns:
        A new episode. Its metadata adds ``observation`` (``model.to_dict()``),
        ``obs_cell`` (``model.name``), ``y_obs_latent`` (the latent ``x_obs``
        value at each query, which effect-scored direction accuracy reads) and
        ``obs_missing_frac`` (the fraction of missing ``x_obs`` cells among the
        columns that are not all zero, 0 when none is).

    Raises:
        ValueError: If ``x_obs`` or ``x_int`` is not finite, e.g. an episode
            that was already observed.
    """
    if not bool(torch.isfinite(episode.x_obs).all() and torch.isfinite(episode.x_int).all()):
        raise ValueError(
            f"episode {episode.scm_id} is not latent: its trajectories hold non-finite "
            "values, so it cannot be observed (again)"
        )
    x_lat = episode.x_obs.detach().cpu().numpy().astype(np.float64)
    xi_lat = episode.x_int.detach().cpu().numpy().astype(np.float64)
    t_len, n_vars = x_lat.shape
    onset = _onset(episode)
    reference = x_lat[:onset] if onset >= 2 else x_lat
    rows = episode.query_time_idx.numpy()
    cols = episode.query_target.reshape(-1).long().numpy()

    children = np.random.SeedSequence([_SALT, int(seed) & (2**64 - 1)]).spawn(len(_COMPONENTS))
    rngs = {
        name: np.random.default_rng(child)
        for name, child in zip(_COMPONENTS, children, strict=True)
    }

    live_obs = (x_lat != 0).any(axis=0)
    live_int = (xi_lat != 0).any(axis=0)
    missing = _missing_cells(x_lat, reference, model.missingness, rngs)
    missing[:, ~live_obs] = False
    missing[rows, cols] = False
    if onset > 0:
        # Drawn for every column, needed or not, so the kept cell of a column
        # does not depend on which other columns lost their history.
        pick = rngs["keep"].integers(0, onset, size=(n_vars,))
        lost = np.flatnonzero(missing[:onset].all(axis=0))
        missing[pick[lost], lost] = False

    scales = _calibrate(reference, model.measurement)
    z = None if model.measurement.snr is None else rngs["noise"].standard_normal((t_len, n_vars))
    x_obs = _read(x_lat, z, scales, live_obs)
    x_obs[missing] = np.nan
    # The sensor is calibrated on x_obs, so an x_int column is read only where
    # x_obs is alive too: a zeroed x_obs column would censor x_int to 0.
    live_hist = live_obs & live_int
    x_int = xi_lat.copy()
    history = _read(xi_lat[:onset], None if z is None else z[:onset], scales, live_hist)
    history[missing[:onset] & live_hist] = np.nan
    x_int[:onset] = history

    missing_frac = float(missing[:, live_obs].mean()) if live_obs.any() else 0.0
    dtype, device = episode.x_obs.dtype, episode.x_obs.device
    return dataclasses.replace(
        episode,
        x_obs=torch.as_tensor(x_obs, dtype=dtype, device=device),
        x_int=torch.as_tensor(x_int, dtype=episode.x_int.dtype, device=device),
        metadata={
            **episode.metadata,
            "observation": model.to_dict(),
            "obs_cell": model.name,
            "y_obs_latent": torch.as_tensor(x_lat[rows, cols], dtype=torch.float32),
            "obs_missing_frac": missing_frac,
        },
    )


def impute_history(x: torch.Tensor, onset: int) -> torch.Tensor:
    """Fill the missing cells of a ``(T, N)`` trajectory without reading the future.

    A missing cell takes the last observed value above it in its column
    (forward fill). A cell with no observation above it takes the mean of the
    column's observed cells before ``onset``, and 0 when there is none. Imputed
    rows before ``onset`` therefore depend only on observed rows before
    ``onset``, and an imputed row ``t >= onset`` only on observed rows up to
    ``t``.

    Args:
        x: Trajectory of shape ``(T, N)``. A non-finite value marks a missing
            cell.
        onset: First post-intervention row. The rows before it are the history
            whose observed mean fills leading gaps. Clamped to ``[0, T]``.

    Returns:
        A new finite tensor with the dtype and device of ``x``.

    Raises:
        ValueError: If ``x`` is not two-dimensional.
    """
    if x.ndim != 2:
        raise ValueError(f"impute_history expects a (T, N) trajectory, got shape {tuple(x.shape)}")
    values = x.detach().cpu().numpy().astype(np.float64)
    t_len, n_vars = values.shape
    seen = np.isfinite(values)
    last = np.where(seen, np.arange(t_len)[:, None], -1)
    np.maximum.accumulate(last, axis=0, out=last)
    filled = np.take_along_axis(values, np.maximum(last, 0), axis=0)
    history = min(max(int(onset), 0), t_len)
    counts = seen[:history].sum(axis=0)
    sums = np.where(seen[:history], values[:history], 0.0).sum(axis=0)
    fallback = np.divide(sums, counts, out=np.zeros(n_vars), where=counts > 0)
    filled = np.where(last >= 0, filled, fallback)
    return torch.as_tensor(filled, dtype=x.dtype, device=x.device)


def impute_episode(episode: Episode) -> Episode:
    """Impute an episode's missing cells for a model that cannot read a mask.

    Args:
        episode: The episode.

    Returns:
        ``episode`` itself when its ``x_obs`` is finite. Otherwise a copy whose
        ``x_obs`` and ``x_int`` went through :func:`impute_history` with the
        intervention onset. Targets and metadata are shared with ``episode``.
    """
    if bool(torch.isfinite(episode.x_obs).all()):
        return episode
    onset = _onset(episode)
    return dataclasses.replace(
        episode,
        x_obs=impute_history(episode.x_obs, onset),
        x_int=impute_history(episode.x_int, onset),
    )


def require_finite_history(episode: Episode, consumer: str) -> None:
    """Refuse an episode whose ``x_obs`` still has missing cells.

    Args:
        episode: The episode a model is about to read.
        consumer: Name of the reading code for the message, e.g.
            ``"dotime-eval-pfn"``.

    Raises:
        ValueError: If ``x_obs`` holds a non-finite cell. The message says to
            impute first.
    """
    n_missing = int((~torch.isfinite(episode.x_obs)).sum())
    if n_missing:
        raise ValueError(
            f"{consumer}: episode {episode.scm_id} has {n_missing} missing (NaN) x_obs cells; "
            "impute first with dotime.observation.impute_episode(episode), which "
            "dotime.evaluation.evaluate applies by default"
        )
