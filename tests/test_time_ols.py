"""TimeOLS: the unadjusted regression plus trend and seasonal terms.

A seasonal or trend driver D (dotime.drivers) is a function of time, so the
regression of Y_t on A_t, Y_(t-1) and time terms that span D blocks the path
A <- D -> Y even when D is hidden. These tests pin that TimeOLS removes the
omitted-driver bias that NaiveOLS carries, costs nothing without a driver, reads
no post-onset data, does not depend on the unit of the time axis, and falls back
to NaiveOLS where the window is too short to fit.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from dotime import baselines
from dotime._build import episode_specs, make_episode
from dotime.benchmarks import Episode
from dotime.interventions import InterventionSpec, InterventionType

yaml = pytest.importorskip("yaml")

_CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "release_config_seasonal_trend.yaml"


def _simulate(driver: str, t_len: int = 400, beta: float = 1.0, seed: int = 1) -> np.ndarray:
    """Linear bi_variate data [A, Y] with an optional driver D -> A, D -> Y that is not released.

    Args:
        driver: ``"none"``, ``"seasonal"`` (period 30) or ``"trend"``.
        t_len: Number of steps.
        beta: Causal effect of A_t on Y_t.
        seed: Noise seed.

    Returns:
        Array of shape ``(t_len, 2)``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(t_len)
    d = {
        "none": np.zeros(t_len),
        "seasonal": 2.0 * np.sin(2 * np.pi * t / 30 + 0.7),
        "trend": 3.0 * (t / t_len - 0.5),
    }[driver]
    x = np.zeros((t_len, 2))
    for s in range(1, t_len):
        a = 0.9 * d[s] + rng.normal()
        x[s] = (a, beta * a + 1.2 * d[s] + 0.3 * x[s - 1, 1] + 0.3 * rng.normal())
    return x


def _episode(
    x: np.ndarray, value: float, onset: int, query_row: int, obs_times: np.ndarray | None = None
) -> Episode:
    """A bi_variate episode with a hard do(A = ``value``) at ``onset``, querying Y at ``query_row``."""
    x_t = torch.as_tensor(x, dtype=torch.float32)
    return Episode(
        x_obs=x_t,
        x_int=x_t.clone(),
        intervention=InterventionSpec(
            targets=[0], times=[onset], intervention_type=InterventionType.HARD, values=value
        ),
        y_true=torch.zeros(1),
        query_target=torch.tensor([1]),
        query_time=torch.tensor([query_row / len(x)]),
        structure="bi_variate",
        metadata={"query_time_idx": [query_row]},
        obs_times=None if obs_times is None else torch.as_tensor(obs_times, dtype=torch.float64),
    )


def _slope(name: str, x: np.ndarray) -> float:
    """The baseline's estimated effect of a unit step in the do-value at the last step."""
    onset = len(x) - 1
    model = baselines.get(name)
    hi = model.predict(_episode(x, 1.0, onset, onset))
    lo = model.predict(_episode(x, 0.0, onset, onset))
    return float(hi - lo)


def test_time_ols_is_registered() -> None:
    assert "TimeOLS" in baselines.available()
    assert baselines.get("TimeOLS").name == "TimeOLS"


@pytest.mark.parametrize("driver", ["seasonal", "trend"])
def test_time_ols_removes_the_bias_of_a_hidden_time_driver(driver: str) -> None:
    """NaiveOLS reads D's push on A and Y as causal; the time terms absorb it."""
    x = _simulate(driver)
    assert _slope("NaiveOLS", x) - 1.0 > 0.08
    assert _slope("TimeOLS", x) == pytest.approx(1.0, abs=0.04)


def test_time_ols_matches_naive_ols_without_a_driver() -> None:
    x = _simulate("none")
    assert _slope("TimeOLS", x) == pytest.approx(_slope("NaiveOLS", x), abs=0.02)


@pytest.mark.parametrize("offset", [0, 5])
def test_time_ols_reads_no_post_onset_data(offset: int) -> None:
    """Changing every row from the onset on leaves the prediction unchanged."""
    x = _simulate("seasonal", t_len=300)
    onset = 250
    changed = x.copy()
    changed[onset:] = 99.0
    model = baselines.get("TimeOLS")
    a = model.predict(_episode(x, 1.5, onset, onset + offset))
    b = model.predict(_episode(changed, 1.5, onset, onset + offset))
    assert torch.equal(a, b)


def test_time_ols_does_not_depend_on_the_time_unit() -> None:
    """Rows 2 time units apart give the same prediction as rows 1 apart."""
    x = _simulate("seasonal", t_len=300)
    model = baselines.get("TimeOLS")
    plain = model.predict(_episode(x, 1.5, 299, 299))
    stretched = model.predict(_episode(x, 1.5, 299, 299, obs_times=2.0 * np.arange(300)))
    assert float(stretched) == pytest.approx(float(plain), abs=1e-3)


def test_time_ols_falls_back_to_naive_ols_on_a_short_window() -> None:
    x = _simulate("seasonal", t_len=40)
    ep = _episode(x, 1.5, 12, 12)
    assert float(baselines.get("TimeOLS").predict(ep)) == pytest.approx(
        float(baselines.get("NaiveOLS").predict(ep))
    )


def test_time_ols_runs_on_released_driven_labels() -> None:
    """One episode of every label of the dot-SeasonalTrend-v1 config gets a finite prediction."""
    config = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    cfg = dict(config["suites"]["dot-SeasonalTrend-v1"], episodes_per_structure=1, T=120)
    model = baselines.get("TimeOLS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for spec in episode_specs(cfg, int(cfg["seed"]), 1.0):
            pred = model.predict(make_episode(spec))
            assert torch.isfinite(pred).all(), spec["structure"]
