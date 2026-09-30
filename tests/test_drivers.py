"""Opt-in seasonal and trend confounding drivers (``dotime.drivers``).

A driven label ``"<base>+<kind>_<visibility>"`` prepends an exogenous driver D to
a named structure. These tests pin that the driver is opt-in and neutral to the
episode's random stream: strength 0 reproduces the base bit for bit, and the
episode generator ends where the base leaves it. They also pin that D equals its
series in both arms, that a hidden D is hidden like U, the column layout, the
back-door routing, the unsupported paths, and that every label of the release
config builds with JSON-able driver metadata.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dotime import _release_io, baselines
from dotime._build import episode_specs, make_episode
from dotime.benchmarks import Episode, SuiteMetadata
from dotime.drivers import (
    DRIVER_KINDS,
    DriverSpec,
    draw_driver,
    driver_generator,
    driver_series,
    parse_structure_label,
    released_driver_series,
)
from dotime.extended import ExtendedDoTime, TSCMPrior
from dotime.interventions import InterventionSpec, InterventionType
from dotime.tscm_sampler import TSCMStructure

CONFIG = Path(__file__).resolve().parents[1] / "scripts" / "release_config_seasonal_trend.yaml"
BASES = ("bi_variate", "back_door")
VISIBILITIES = ("observed", "hidden")
SUFFIXES = tuple(f"{k}_{v}" for k in DRIVER_KINDS for v in VISIBILITIES)
BACK_DOOR_FAMILY = {"back_door", "observed_confounder", "confounder_mediator"}


def _prior(base: str, seed: int, driver: DriverSpec | None = None) -> TSCMPrior:
    """A counterfactual ``TSCMPrior`` for ``base``, optionally driven.

    Args:
        base: A ``TSCMStructure`` value.
        seed: Episode seed.
        driver: The driver to add, or ``None`` for the base structure.

    Returns:
        The prior.
    """
    return TSCMPrior(TSCMStructure(base), seed=seed, pair_mode="counterfactual", driver=driver)


def _sample(label: str, seed: int = 5, t_len: int = 80) -> dict:
    """One counterfactual ``generate_sample`` dict of ``label``, seeded like the build.

    Args:
        label: A structure label, plain or driven.
        seed: Episode seed, also used for the global torch RNG.
        t_len: Number of released steps.

    Returns:
        The sample dict.
    """
    torch.manual_seed(seed)
    gen = ExtendedDoTime(tscm_structure=label, n_max=41, seed=seed, pair_mode="counterfactual")
    return gen.generate_sample(T=t_len)


def _canonical_names(structure: str) -> list[str]:
    """Variable name of each released column of ``structure``.

    Args:
        structure: A structure label, plain or driven.

    Returns:
        For example ``["A", "X", "D", "Y"]``.
    """
    return baselines._canonical_summary_graph(structure)[0]


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", [*(s.value for s in TSCMStructure), "rct_no_confounding"])
def test_plain_labels_pass_through_unchanged(label: str) -> None:
    """A label without ``+`` is returned as given, with no driver.

    Args:
        label: A plain structure label, including the legacy alias.
    """
    assert parse_structure_label(label) == (label, None)


@pytest.mark.parametrize("base", [*BASES, "front_door"])
@pytest.mark.parametrize("suffix", SUFFIXES)
def test_driven_labels_parse(base: str, suffix: str) -> None:
    """``<base>+<kind>_<visibility>`` yields the base and a full-strength spec.

    Args:
        base: The base structure.
        suffix: The ``<kind>_<visibility>`` part.
    """
    got_base, spec = parse_structure_label(f"{base}+{suffix}")
    kind, visibility = suffix.split("_")
    assert got_base == base
    assert spec == DriverSpec(kind=kind, observed=visibility == "observed", strength=1.0)
    assert spec.suffix == suffix


@pytest.mark.parametrize(
    "label",
    [
        "back_door+",
        "+seasonal_hidden",
        "back_door+seasonal",
        "back_door+weekly_hidden",
        "back_door+seasonal_visible",
        "back_door+seasonal_hidden+trend_observed",
        "no_such_structure+trend_observed",
        "rct_no_confounding+trend_observed",
        "back_door+Seasonal_hidden",
    ],
)
def test_malformed_driven_labels_raise(label: str) -> None:
    """Every part of a driven label is validated, and only one driver is allowed.

    Args:
        label: A malformed driven label.
    """
    with pytest.raises(ValueError, match="malformed structure label"):
        parse_structure_label(label)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "weekly", "observed": True}, "kind"),
        ({"kind": "trend", "observed": True, "strength": -0.5}, "strength"),
        ({"kind": "trend", "observed": True, "strength": math.nan}, "strength"),
    ],
)
def test_driver_spec_rejects_bad_fields(kwargs: dict, match: str) -> None:
    """``DriverSpec`` refuses an unknown kind and a negative or NaN strength.

    Args:
        kwargs: Constructor arguments.
        match: Expected fragment of the error message.
    """
    with pytest.raises(ValueError, match=match):
        DriverSpec(**kwargs)


# --------------------------------------------------------------------------- #
# Random streams
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kind", DRIVER_KINDS)
@pytest.mark.parametrize("base", BASES)
def test_strength_zero_reproduces_the_base_bit_for_bit(base: str, kind: str, seed: int) -> None:
    """With strength 0 both arms and the intervention equal the base structure's.

    D is prepended, so the driven arms carry the base columns shifted by one.

    Args:
        base: The base structure.
        kind: The driver kind.
        seed: Episode seed.
    """
    plain = _prior(base, seed)
    driven = _prior(base, seed, DriverSpec(kind, observed=True, strength=0.0))
    for _ in range(2):  # the second pair checks that the streams stay aligned
        x_obs, x_int, iv, _ = plain.generate_pair(T=60)
        d_obs, d_int, d_iv, _ = driven.generate_pair(T=60)
        assert torch.equal(d_obs[:, 1:], x_obs)
        assert torch.equal(d_int[:, 1:], x_int)
        assert d_iv.times == iv.times
        assert d_iv.values == iv.values
        assert [t - 1 for t in d_iv.targets] == iv.targets
    assert torch.equal(driven.gen.get_state(), plain.gen.get_state())


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("visibility", VISIBILITIES)
@pytest.mark.parametrize("base", BASES)
def test_episode_generator_ends_in_the_base_state(base: str, visibility: str, seed: int) -> None:
    """At full strength the driver moves the data but not the episode generator.

    Args:
        base: The base structure.
        visibility: Whether the driver is observed.
        seed: Episode seed.
    """
    plain = _prior(base, seed)
    driven = _prior(base, seed, DriverSpec("seasonal", observed=visibility == "observed"))
    global_state = torch.get_rng_state()
    for _ in range(2):
        x_obs, _, iv, _ = plain.generate_pair(T=60)
        d_obs, _, d_iv, _ = driven.generate_pair(T=60)
        assert torch.equal(driven.gen.get_state(), plain.gen.get_state())
        assert (d_iv.times, d_iv.values) == (iv.times, iv.values)
        assert not torch.equal(d_obs[:, 1:], x_obs)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_driver_draws_are_seeded_and_private() -> None:
    """The driver stream is a pure function of the seed and differs from the episode's."""
    spec = DriverSpec("seasonal", observed=True)
    first = draw_driver(spec, driver_generator(11))
    assert draw_driver(spec, driver_generator(11)) == first
    assert draw_driver(spec, driver_generator(12)) != first
    episode_gen = torch.Generator().manual_seed(11)
    assert not torch.equal(driver_generator(11).get_state(), episode_gen.get_state())
    # Negative seeds are folded into the unsigned range rather than rejected.
    assert draw_driver(spec, driver_generator(-1)) == draw_driver(spec, driver_generator(2**64 - 1))


def test_draws_stay_in_their_ranges() -> None:
    """Loadings, period, phase and direction follow the documented distributions."""
    for seed in range(200):
        gen = driver_generator(seed)
        season = draw_driver(DriverSpec("seasonal", observed=True), gen)
        trend = draw_driver(DriverSpec("trend", observed=False), gen)
        for draw in (season, trend):
            assert 0.5 <= abs(draw.loading_a) < 1.0
            assert 0.5 <= abs(draw.loading_y) < 1.0
            assert draw.confounding_sign == np.sign(draw.loading_a * draw.loading_y)
        assert 12.0 <= season.params["period"] < 48.0
        assert 0.0 <= season.params["phase"] < 2.0 * math.pi
        assert trend.params == {"direction": 1} or trend.params == {"direction": -1}
    off = draw_driver(DriverSpec("trend", observed=True, strength=0.0), driver_generator(0))
    assert (off.loading_a, off.loading_y, off.confounding_sign) == (0.0, 0.0, 0)


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("total_t", [2, 3, 250])
@pytest.mark.parametrize("direction", [1, -1])
def test_trend_is_a_bounded_ramp(total_t: int, direction: int) -> None:
    """The trend runs from ``-direction`` to ``direction`` and never leaves ``[-1, 1]``.

    Args:
        total_t: Number of simulated steps.
        direction: Sign of the ramp.
    """
    s = driver_series("trend", {"direction": direction}, total_t)
    assert s.dtype == torch.float64
    assert (float(s[0]), float(s[-1])) == (-direction, direction)
    assert float(s.abs().max()) <= 1.0
    assert bool((direction * s.diff() > 0).all())


def test_seasonal_is_a_bounded_sinusoid() -> None:
    """The seasonal driver stays in ``[-1, 1]`` and repeats after one period."""
    s = driver_series("seasonal", {"period": 24.0, "phase": 1.0}, 250)
    assert float(s.abs().max()) <= 1.0
    assert torch.allclose(s[24:], s[:-24], atol=1e-12)
    assert float(s[0]) == pytest.approx(math.sin(1.0))


def test_driver_series_rejects_bad_input() -> None:
    """Too short a series or an unknown kind raises."""
    with pytest.raises(ValueError, match="at least 2"):
        driver_series("trend", {"direction": 1}, 1)
    with pytest.raises(ValueError, match="kind"):
        driver_series("weekly", {}, 10)


# --------------------------------------------------------------------------- #
# Arms, visibility and layout
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", DRIVER_KINDS)
@pytest.mark.parametrize("base", BASES)
def test_observed_driver_column_is_its_series_in_both_arms(base: str, kind: str) -> None:
    """The released D column is the series after burn-in, identical in both arms.

    Args:
        base: The base structure.
        kind: The driver kind.
    """
    t_len = 80
    s = _sample(f"{base}+{kind}_observed", t_len=t_len)
    n, onset, col = int(s["num_vars"]), int(s["int_onset_idx"]), s["driver"]["column"]
    series = released_driver_series(s["driver"], t_len)
    assert torch.equal(s["X_obs_full"][:, col], series)
    assert torch.equal(s["X_int"][:, col], series)
    assert onset > 0
    assert torch.equal(s["X_obs_full"][:onset, :n], s["X_int"][:onset, :n])


@pytest.mark.parametrize("kind", DRIVER_KINDS)
@pytest.mark.parametrize("base", BASES)
def test_hidden_driver_is_simulated_then_hidden_like_u(base: str, kind: str) -> None:
    """A hidden D drives the simulation but is zeroed and masked on release.

    Args:
        base: The base structure.
        kind: The driver kind.
    """
    t_len = 80
    prior = _prior(base, 5, DriverSpec(kind, observed=False))
    x_obs, x_int, iv, _ = prior.generate_pair(T=t_len)
    series = released_driver_series(prior.last_driver, t_len)
    assert torch.equal(x_obs[:, 0], series)  # D is topo index 0 inside the simulation
    assert torch.equal(x_int[:, 0], series)
    onset = min(iv.times)
    assert torch.equal(x_obs[:onset], x_int[:onset])

    s = _sample(f"{base}+{kind}_hidden", t_len=t_len)
    col = s["driver"]["column"]
    assert s["driver"]["known_future"] is False
    for key in ("X_obs_full", "X_obs", "X_int"):
        assert not s[key][:, col].any(), key
    assert float(s["variable_mask"][col]) == 0.0

    seen = _sample(f"{base}+{kind}_observed", t_len=t_len)
    assert seen["X_obs_full"][:, col].any()
    assert float(seen["variable_mask"][col]) == 1.0
    assert seen["driver"]["known_future"] is True


@pytest.mark.parametrize("suffix", SUFFIXES)
@pytest.mark.parametrize("base", [s.value for s in TSCMStructure])
def test_driver_sits_just_before_the_outcome(base: str, suffix: str) -> None:
    """A driven layout is the base layout with D inserted before Y.

    A and the base middle columns keep their index, Y stays last, and the
    summary graph gains exactly ``D -> A`` and ``D -> Y``.

    Args:
        base: The base structure.
        suffix: The ``<kind>_<visibility>`` part.
    """
    b_names, b_summary, b_hidden = baselines._canonical_summary_graph(base)
    names, summary, hidden = baselines._canonical_summary_graph(f"{base}+{suffix}")
    assert names == [*b_names[:-1], "D", "Y"]
    assert set(summary.edges) == set(b_summary.edges) | {("D", "A"), ("D", "Y")}
    assert hidden == b_hidden | ({"D"} if suffix.endswith("hidden") else set())


@pytest.mark.parametrize("label", ["back_door+trend_observed", "bi_variate+seasonal_hidden"])
def test_sample_layout_and_metadata(label: str) -> None:
    """The sample puts A first, D at N-2 and Y last, and records a JSON-able driver.

    Args:
        label: A driven label.
    """
    seed = 5
    s = _sample(label, seed=seed)
    n = int(s["num_vars"])
    driver = s["driver"]
    assert n == len(_canonical_names(label))
    assert (int(s["intervention_target"]), int(s["query_target"]), driver["column"]) == (
        0,
        n - 1,
        n - 2,
    )
    assert json.loads(json.dumps(driver)) == driver
    assert set(driver) == {
        "kind",
        "observed",
        "column",
        "strength",
        "params",
        "loadings",
        "confounding_sign",
        "known_future",
        "burn_in",
        "generation_seed",
    }
    loadings = driver["loadings"]
    assert driver["confounding_sign"] == np.sign(loadings["A"] * loadings["Y"])
    assert (driver["burn_in"], driver["generation_seed"], driver["strength"]) == (50, seed, 1.0)
    assert driver["known_future"] == driver["observed"] == label.endswith("observed")


# --------------------------------------------------------------------------- #
# Back-door routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("back_door+seasonal_observed", ["X", "D"]),
        ("back_door+trend_observed", ["X", "D"]),
        ("observed_confounder+seasonal_observed", ["X", "D"]),
        ("confounder_mediator+trend_observed", ["X", "D"]),
        ("bi_variate+seasonal_observed", ["D"]),
        ("bi_variate+trend_observed", ["D"]),
        ("back_door+seasonal_hidden", ["X"]),
        ("bi_variate+trend_hidden", []),
    ],
)
def test_back_door_columns_on_driven_labels(label: str, expected: list[str]) -> None:
    """An observed D joins the back-door set, a hidden one never does.

    Args:
        label: A driven label.
        expected: Names of the adjustment columns.
    """
    names = _canonical_names(label)
    n_vars, a, y, adjust = baselines._back_door_columns(label)
    assert (n_vars, names[a], names[y]) == (len(names), "A", "Y")
    assert [names[c] for c in adjust] == expected


def test_back_door_ols_routing() -> None:
    """Plain labels route as before, and a driven label adjusts only when D is observed."""
    route = baselines.BackDoorOLSBaseline._adjusts
    for structure in TSCMStructure:
        assert route(structure.value) is (structure.value in BACK_DOOR_FAMILY)
        for suffix in SUFFIXES:
            expected = suffix.endswith("observed") and structure.value in {
                *BACK_DOOR_FAMILY,
                "bi_variate",
            }
            assert route(f"{structure.value}+{suffix}") is expected
    assert route("regime_2") is False
    with pytest.raises(ValueError, match="malformed"):
        route("back_door+weekly_observed")


def _driven_linear_episode(label: str, loading_y: float = -1.5) -> tuple[Episode, int]:
    """A linear bi-variate SCM confounded by a seasonal driver, as a 3-column episode.

    Columns are A, D, Y. ``D_t = sin(2 pi t / 24)``, ``A_t = D_t + noise`` and
    ``Y_t = A_t + loading_y * D_t + 0.3 Y_(t-1) + noise``, so the effect of
    ``A_t`` on ``Y_t`` is 1 while D biases an unadjusted regression.

    Args:
        label: Structure label of the episode.
        loading_y: Coefficient of D in the outcome equation.

    Returns:
        ``(episode, onset)`` with a hard do() on A at ``onset``.
    """
    rng = np.random.default_rng(0)
    t_len, onset = 2000, 1999
    x = np.zeros((t_len, 3))
    for t in range(1, t_len):
        d_t = math.sin(2 * math.pi * t / 24)
        a_t = d_t + 0.3 * rng.normal()
        y_t = a_t + loading_y * d_t + 0.3 * x[t - 1, 2] + 0.3 * rng.normal()
        x[t] = (a_t, d_t, y_t)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    episode = Episode(
        x_obs=x_t,
        x_int=x_t.clone(),
        intervention=InterventionSpec(
            targets=[0], times=[onset], intervention_type=InterventionType.HARD, values=0.0
        ),
        y_true=torch.zeros(1),
        query_target=torch.tensor([2]),
        query_time=torch.tensor([onset / t_len]),
        structure=label,
    )
    return episode, onset


def test_back_door_ols_adjusts_for_an_observed_driver() -> None:
    """Adjusting for an observed D recovers the effect that D confounds.

    The unadjusted regression on the same data has the wrong sign, so the
    recovery is due to the adjustment. On the hidden label, whose D column is
    zeroed as on release, the baseline takes the pre-onset mean.
    """
    episode, onset = _driven_linear_episode("bi_variate+seasonal_observed")
    x = episode.x_obs.numpy().astype(np.float64)
    unadjusted = baselines._ols_fit(
        np.column_stack([x[1:onset, 0], x[: onset - 1, 2]]), x[1:onset, 2]
    )
    assert unadjusted[1] < 0.0  # the confounding bias flips the sign of the naive slope

    model = baselines.get("BackDoorOLS")
    low = model.predict(episode)
    episode.intervention = InterventionSpec(
        targets=[0], times=[onset], intervention_type=InterventionType.HARD, values=1.0
    )
    high = model.predict(episode)
    # Predictions are affine in the do-value, so the step is the fitted A-coefficient.
    assert float(high - low) == pytest.approx(1.0, abs=0.1)

    hidden, onset = _driven_linear_episode("bi_variate+seasonal_hidden")
    hidden.x_obs[:, 1] = 0.0
    assert float(model.predict(hidden)) == float(hidden.x_obs.numpy()[:onset, 2].mean())


# --------------------------------------------------------------------------- #
# Unsupported paths
# --------------------------------------------------------------------------- #


def test_drivers_need_counterfactual_pairing() -> None:
    """Independent-noise pairing cannot share D across arms, so it is refused."""
    with pytest.raises(NotImplementedError, match="counterfactual"):
        TSCMPrior(TSCMStructure("back_door"), driver=DriverSpec("seasonal", observed=True))
    with pytest.raises(NotImplementedError, match="counterfactual"):
        ExtendedDoTime(tscm_structure="back_door+seasonal_observed")
    with pytest.raises(ValueError, match="malformed"):
        ExtendedDoTime(tscm_structure="back_door+weekly_hidden", pair_mode="counterfactual")


def test_generate_batch_refuses_drivers_before_any_draw() -> None:
    """The vectorized simulator has no drivers; the refusal leaves every stream alone."""
    gen = ExtendedDoTime(tscm_structure="bi_variate+trend_hidden", pair_mode="counterfactual")
    np_state = gen.rng.get_state()
    episode_state = gen.prior.gen.get_state()
    driver_state = gen.prior.driver_gen.get_state()
    with pytest.raises(NotImplementedError, match="generate_sample"):
        gen.generate_batch(2)
    assert np.array_equal(gen.rng.get_state()[1], np_state[1])
    assert gen.rng.get_state()[2] == np_state[2]
    assert torch.equal(gen.prior.gen.get_state(), episode_state)
    assert torch.equal(gen.prior.driver_gen.get_state(), driver_state)


# --------------------------------------------------------------------------- #
# Release config
# --------------------------------------------------------------------------- #


def _release_suite() -> tuple[int, dict]:
    """The top-level seed and the single suite of the seasonal/trend release config.

    Returns:
        ``(top_level_seed, suite_config)``.
    """
    config = yaml.safe_load(CONFIG.read_text())
    (suite,) = config["suites"].values()
    return int(config["seed"]), suite


def test_release_config_defines_the_suite() -> None:
    """One suite, seeded at the top-level seed + 1000, 10 labels of 1000 episodes."""
    top_seed, suite = _release_suite()
    assert int(suite["seed"]) == top_seed + 1000
    assert (suite["generator"], suite["pair_mode"], suite["T"]) == (
        "identifiability",
        "counterfactual",
        200,
    )
    labels = list(suite["structures"])
    assert labels == [
        "bi_variate",
        "back_door",
        "bi_variate+seasonal_observed",
        "bi_variate+trend_observed",
        "back_door+seasonal_observed",
        "back_door+trend_observed",
        "bi_variate+seasonal_hidden",
        "bi_variate+trend_hidden",
        "back_door+seasonal_hidden",
        "back_door+trend_hidden",
    ]
    assert len(episode_specs(suite, int(suite["seed"]), 1.0)) == 10_000


def test_every_release_label_builds_at_t60() -> None:
    """``make_episode`` builds each label with the driver recorded and JSON-able."""
    _, suite = _release_suite()
    small = {**suite, "T": 60, "episodes_per_structure": 1}
    for spec in episode_specs(small, int(suite["seed"]), 1.0):
        label = spec["structure"]
        episode = make_episode(spec)
        assert episode.structure == label
        assert episode.metadata["tier"] == suite["structures"][label]
        row = _release_io._episode_to_row(episode)
        stored = json.loads(row["metadata_json"])
        base, driver = parse_structure_label(label)
        if driver is None:
            assert "driver" not in episode.metadata
            continue
        record = episode.metadata["driver"]
        assert stored["driver"] == record
        col = record["column"]
        assert (episode.n_vars, col) == (len(_canonical_names(base)) + 1, episode.n_vars - 2)
        if driver.observed:
            series = released_driver_series(record, episode.length)
            assert torch.equal(episode.x_obs[:, col], series)
            assert torch.equal(episode.x_int[:, col], series)
        else:
            assert not episode.x_obs[:, col].any()
            assert not episode.x_int[:, col].any()


def test_driver_record_survives_the_release_roundtrip(tmp_path: Path) -> None:
    """Writing and reading a suite keeps ``metadata["driver"]`` exactly.

    Args:
        tmp_path: Pytest's temporary directory.
    """
    pytest.importorskip("pyarrow", reason="frozen-suite IO needs the evaluation extra")
    _, suite = _release_suite()
    small = {**suite, "T": 60, "episodes_per_structure": 1}
    specs = [s for s in episode_specs(small, int(suite["seed"]), 1.0) if "+" in s["structure"]]
    episodes = [make_episode(s) for s in specs[:2]]
    meta = SuiteMetadata(
        name="ST",
        version="1.0.0",
        zenodo_record_id="LOCAL",
        doi="",
        description="driver round-trip",
        n_episodes=len(episodes),
        query_time_encoding="index/T",
    )
    suite_dir = _release_io.write_suite(
        meta, episodes, tmp_path / "ST-1.0.0", package_version="0", seed=0
    )
    for orig, got in zip(episodes, _release_io.read_suite(meta, suite_dir), strict=True):
        assert got.metadata["driver"] == orig.metadata["driver"]
        assert got.structure == orig.structure


# --------------------------------------------------------------------------- #
# Global randomness
# --------------------------------------------------------------------------- #


def test_helpers_draw_no_global_randomness() -> None:
    """Parsing, series, column helpers and driven generation leave global RNGs alone."""
    baselines._back_door_columns.cache_clear()
    torch_state = torch.get_rng_state()
    _, np_keys, np_pos, *_ = np.random.get_state()
    np_keys = np_keys.copy()

    _, suite = _release_suite()
    for label in suite["structures"]:
        parse_structure_label(label)
        baselines._back_door_columns(label)
    for kind in DRIVER_KINDS:
        prior = _prior("back_door", 3, DriverSpec(kind, observed=False))
        prior.generate_pair(T=40)
        released_driver_series(prior.last_driver, 40)
        draw_driver(DriverSpec(kind, observed=True), driver_generator(3)).series(10)

    _, keys_after, pos_after, *_ = np.random.get_state()
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_keys, keys_after)
    assert np_pos == pos_after
