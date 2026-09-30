"""bow_graph in dot-Identifiability-v1 1.2.0, and the unadjusted NaiveOLS baseline.

``unobserved_confounder`` (hidden U -> A, U -> Y, no A -> Y) is identified with a
zero effect, so it cannot be the suite's non-identified case. ``bow_graph`` adds
the edge A -> Y: the effect is then nonzero and confounded along A <- U -> Y,
which no observed variable blocks. These tests pin that the two structures
differ by that edge alone, that the 1.2.0 config adds bow_graph without changing
any 1.1.0 episode except mediator's query, that the structural baselines decline
bow_graph, and that NaiveOLS is the unadjusted regression whose slope carries
the omitted-confounder bias that BackDoorOLS removes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from dotime import baselines
from dotime._build import episode_seed, episode_specs, make_episode
from dotime.benchmarks import Episode
from dotime.extended import ExtendedDoTime, TSCMPrior
from dotime.interventions import InterventionSpec, InterventionType
from dotime.reference.reference_table import CPU_BASELINES
from dotime.tscm_sampler import TSCMStructure

yaml = pytest.importorskip("yaml")

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# Measured on ExtendedDoTime seeds 0-39 at T=200: 22 bow_graph effects reach
# |effect| >= 0.1 and 11 are exactly zero (saturating mechanisms).
BOW_NONZERO_RATE = 22 / 40


def _release_cfg(name: str, **overrides: object) -> tuple[dict, int]:
    """The identifiability suite of a release config and the seed build_release gives it.

    Args:
        name: File name of a release config under ``scripts/``.
        **overrides: Suite keys to replace, e.g. ``T=60``.

    Returns:
        ``(suite_cfg, suite_seed)``. The seed is the suite's own ``seed:`` when
        it sets one, else the base seed + 1000 that ``build_release.py`` gives
        the first suite of a file.

    Raises:
        FileNotFoundError: If the config does not exist.
        KeyError: If it has no ``dot-Identifiability-v1`` suite.
    """
    cfg = yaml.safe_load((_SCRIPTS / name).read_text())
    suite = cfg["suites"]["dot-Identifiability-v1"]
    return {**suite, **overrides}, int(suite.get("seed", int(cfg["seed"]) + 1000))


def _make(spec: dict) -> Episode:
    """Build one release episode without leaking make_episode's warning filter.

    Args:
        spec: An episode spec from ``episode_specs``.

    Returns:
        The episode.

    Raises:
        ValueError: If the spec's ``kind`` is unknown.
    """
    with warnings.catch_warnings():
        return make_episode(spec)


def _assert_same_spec_but_mediator_offset(old: dict, new: dict) -> None:
    """Assert that a 1.1.0 spec and its 1.2.0 counterpart differ only in mediator's offset.

    Args:
        old: A spec from the 1.1.0 config.
        new: The spec at the same index from the 1.2.0 config.

    Raises:
        AssertionError: If they differ in any other key, or mediator's query is
            not moved from the onset to one step after it.
    """
    if old["structure"] == "mediator":
        assert (old["query_offset_range"], new["query_offset_range"]) == ((0, 0), (1, 1))
        old, new = {**old, "query_offset_range": None}, {**new, "query_offset_range": None}
    # 1.2.0 records the ground-truth graph in the metadata. The flag touches no
    # tensor or random stream (tests/test_graph_meta.py pins that), so it is
    # the one extra key a 1.2.0 spec may carry.
    assert new.pop("record_graph", None) is True
    assert old == new


def test_bow_graph_is_unobserved_confounder_plus_a_to_y() -> None:
    """Same seed: identical U, A, intervention and RNG state, and Y differs unless inert.

    Both structures sort their nodes as U, A, Y, and the sampler draws every
    node's weights, one per potential parent, in that order. So the two SCMs
    share every parameter and the frozen noise, and bow_graph merely reads the
    weight of A in Y's mechanism that unobserved_confounder draws and ignores.
    In practice Y then matches only when that mechanism never passes A on: its
    activation is identically zero, so Y is exactly its own noise.
    """
    inert = []
    for seed in range(6):
        runs = {}
        for structure in ("unobserved_confounder", "bow_graph"):
            torch.manual_seed(seed)
            prior = TSCMPrior(TSCMStructure(structure), seed=seed, pair_mode="counterfactual")
            assert prior.sampler._build_dag().topo_order == ["U", "A", "Y"]
            x_obs, x_int, iv, scm = prior.generate_pair(T=120)
            noise = scm._frozen_noise["Y"][prior.burn_in :]
            states = prior.gen.get_state(), torch.get_rng_state()
            runs[structure] = (x_obs, x_int, iv, noise, *states)
        uc_obs, uc_int, uc_iv, _, uc_gen, uc_global = runs["unobserved_confounder"]
        bow_obs, bow_int, bow_iv, noise, bow_gen, bow_global = runs["bow_graph"]
        # Topological columns: U = 0, A = 1, Y = 2.
        assert torch.equal(uc_obs[:, :2], bow_obs[:, :2])
        assert torch.equal(uc_int[:, :2], bow_int[:, :2])
        assert uc_iv.to_dict() == bow_iv.to_dict()
        assert torch.equal(uc_gen, bow_gen)
        assert torch.equal(uc_global, bow_global)
        if torch.equal(uc_obs[:, 2], bow_obs[:, 2]) and torch.equal(uc_int[:, 2], bow_int[:, 2]):
            assert torch.equal(bow_obs[:, 2], noise)
            assert torch.equal(bow_int[:, 2], noise)
            inert.append(seed)
    # Seed 2 draws a ReLU for Y whose input stays below zero, the saturation
    # that also zeroes about a third of the front_door effects.
    assert inert == [2]


def test_bow_graph_effect_is_mostly_nonzero_where_the_control_is_exactly_zero() -> None:
    """bow_graph moves Y after do(A) in most episodes, unobserved_confounder in none."""
    effects: dict[str, list[float]] = {"bow_graph": [], "unobserved_confounder": []}
    for seed in range(40):
        for structure, values in effects.items():
            torch.manual_seed(seed)
            gen = ExtendedDoTime(tscm_structure=structure, seed=seed, pair_mode="counterfactual")
            values.append(float(gen.generate_sample(T=200)["Y_causal_effect"]))
    rate = float(np.mean(np.abs(effects["bow_graph"]) >= 0.1))
    assert rate == pytest.approx(BOW_NONZERO_RATE, abs=0.1)
    assert effects["unobserved_confounder"] == [0.0] * 40


def test_v1_2_config_is_v1_1_plus_bow_graph_and_the_mediator_offset() -> None:
    """Every 1.1.0 (index, seed) pair survives, and only mediator's target moves."""
    v11, seed11 = _release_cfg("release_config_v1_1.yaml")
    v12, seed12 = _release_cfg("release_config_v1_2.yaml")
    assert seed11 == seed12

    full11, full12 = episode_specs(v11, seed11, 1.0), episode_specs(v12, seed12, 1.0)
    assert (len(full11), len(full12)) == (10_800, 12_150)
    for old, new in zip(full11, full12[:10_800], strict=True):
        _assert_same_spec_but_mediator_offset(old, new)
    appended = full12[10_800:]
    assert {(s["structure"], s["tier"]) for s in appended} == {("bow_graph", 3)}
    assert [s["idx"] for s in appended] == list(range(10_800, 12_150))

    small11, _ = _release_cfg("release_config_v1_1.yaml", T=60, episodes_per_structure=1)
    small12, _ = _release_cfg("release_config_v1_2.yaml", T=60, episodes_per_structure=1)
    specs11 = episode_specs(small11, seed11, 1.0)
    specs12 = episode_specs(small12, seed12, 1.0)
    assert len(specs11) == 8
    last = specs12[-1]
    assert (last["structure"], last["tier"], last["idx"]) == ("bow_graph", 3, 8)
    assert last["seed"] == episode_seed(seed12, 8)
    for old, new in zip(specs11, specs12[:-1], strict=True):
        _assert_same_spec_but_mediator_offset(old, new)
        ep_old, ep_new = _make(old), _make(new)
        assert torch.equal(ep_old.x_obs, ep_new.x_obs), old["structure"]
        assert torch.equal(ep_old.x_int, ep_new.x_int), old["structure"]
        same_target = torch.equal(ep_old.y_true, ep_new.y_true)
        assert same_target == (old["structure"] != "mediator"), old["structure"]


def test_structural_baselines_decline_bow_graph() -> None:
    """BackDoorOLS and IV2SLS return the pre-onset outcome mean on a release bow_graph episode."""
    cfg, seed = _release_cfg("release_config_v1_2.yaml", T=60, episodes_per_structure=1)
    ep = _make(episode_specs(cfg, seed, 1.0)[-1])
    assert ep.structure == "bow_graph"
    onset, y = ep.intervention.times[0], int(ep.query_target[0])
    mean = float(ep.x_obs[:onset, y].numpy().mean())
    for name in ("BackDoorOLS", "IV2SLS"):
        assert float(baselines.get(name).predict(ep)[0]) == pytest.approx(mean, abs=1e-6), name


def test_naive_ols_is_registered_and_scored_after_back_door_ols() -> None:
    """The registry serves NaiveOLS, and the reference table runs it next to BackDoorOLS."""
    assert "NaiveOLS" in baselines.available()
    assert baselines.get("NaiveOLS").name == "NaiveOLS"
    assert CPU_BASELINES.index("NaiveOLS") == CPU_BASELINES.index("BackDoorOLS") + 1


def _unadjusted_regression(ep: Episode) -> float:
    """NaiveOLS's definition, fit with lstsq: OLS of Y_t on [1, A_t, Y_(t-1)] before onset.

    Args:
        ep: A single-query episode whose pre-onset window has at least 4 steps.

    Returns:
        The fitted outcome at the do-value (the pre-onset mean of A when the
        intervention is not a scalar), averaged over the observed ``Y_(t-1)``.

    Raises:
        numpy.linalg.LinAlgError: If the least-squares fit does not converge.
    """
    x = ep.x_obs.double().numpy()
    a, y, onset = ep.intervention.targets[0], int(ep.query_target[0]), min(ep.intervention.times)
    design = np.column_stack([np.ones(onset - 1), x[1:onset, a], x[: onset - 1, y]])
    coef = np.linalg.lstsq(design, x[1:onset, y], rcond=None)[0]
    values = ep.intervention.values
    a_val = float(values) if isinstance(values, (int, float)) else float(x[1:onset, a].mean())
    return float(coef[0] + coef[1] * a_val + coef[2] * x[: onset - 1, y].mean())


def _release_episodes() -> dict[str, Episode]:
    """One bow_graph and one back_door 1.2.0 episode, and two generic-prior episodes.

    Returns:
        Episodes keyed by a label. ``generic_hard`` has a scalar do() on two
        targets and ``generic_time_varying`` a do() whose value is a function
        of time, so both do-value branches are covered.

    Raises:
        FileNotFoundError: If the 1.2.0 release config is missing.
    """
    cfg, seed = _release_cfg("release_config_v1_2.yaml", T=60, episodes_per_structure=1)
    ident = {s["structure"]: s for s in episode_specs(cfg, seed, 1.0)}
    generic = episode_specs({"generator": "generic", "n_episodes": 5, "T": 60}, 11, 1.0)
    return {
        "bow_graph": _make(ident["bow_graph"]),
        "back_door": _make(ident["back_door"]),
        "generic_hard": _make(generic[0]),
        "generic_time_varying": _make(generic[4]),
    }


def test_naive_ols_is_the_unadjusted_regression_on_every_structure() -> None:
    """NaiveOLS applies to named structures and the generic prior alike."""
    episodes = _release_episodes()
    assert episodes["generic_hard"].structure is None
    assert len(episodes["generic_hard"].intervention.targets) == 2
    assert callable(episodes["generic_time_varying"].intervention.values)
    model = baselines.get("NaiveOLS")
    for label, ep in episodes.items():
        expected = _unadjusted_regression(ep)
        assert float(model.predict(ep)[0]) == pytest.approx(expected, abs=1e-4), label


def test_naive_ols_falls_back_to_the_mean_on_a_short_window() -> None:
    """With fewer than 4 pre-onset steps there is nothing to fit, as for BackDoorOLS."""
    x = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    ep = Episode(
        x_obs=x,
        x_int=x.clone(),
        intervention=InterventionSpec(
            targets=[0], times=[3], intervention_type=InterventionType.HARD, values=-7.0
        ),
        y_true=torch.zeros(1),
        query_target=torch.tensor([2]),
        query_time=torch.tensor([0.3]),
        structure="bow_graph",
    )
    assert float(baselines.get("NaiveOLS").predict(ep)[0]) == pytest.approx(float(x[:3, 2].mean()))


def test_naive_ols_carries_the_omitted_confounder_bias() -> None:
    """On linear back_door data NaiveOLS's slope is beta + gamma * delta, BackDoorOLS's beta.

    Canonical columns [A, X, Y] with X -> A, X -> Y and A -> Y. Leaving X_t out
    of the regression of Y_t on [1, A_t, Y_(t-1)] adds gamma, the effect of X on
    Y, times delta, the A-coefficient of the auxiliary regression of X_t on that
    same design (the omitted-variable bias formula).
    """
    rng = np.random.default_rng(0)
    t_len, beta, gamma = 5000, 1.0, 1.0
    x = np.zeros((t_len, 3))
    for t in range(1, t_len):
        x_t = 0.5 * x[t - 1, 1] + rng.normal()
        a_t = 0.8 * x_t + rng.normal()
        y_t = beta * a_t + gamma * x_t + 0.3 * x[t - 1, 2] + 0.3 * rng.normal()
        x[t] = (a_t, x_t, y_t)
    onset = t_len - 1

    def episode(value: float) -> Episode:
        """The simulated trajectory with a hard do(A = ``value``) at its last step.

        Args:
            value: The do-value of the treatment A.

        Returns:
            A back_door episode whose single query targets Y at the onset.
        """
        x_t = torch.as_tensor(x, dtype=torch.float32)
        return Episode(
            x_obs=x_t,
            x_int=x_t.clone(),
            intervention=InterventionSpec(
                targets=[0], times=[onset], intervention_type=InterventionType.HARD, values=value
            ),
            y_true=torch.zeros(1),
            query_target=torch.tensor([2]),
            query_time=torch.tensor([onset / t_len]),
            structure="back_door",
        )

    # Predictions are affine in the do-value, so a unit step in v gives the slope.
    slope = {}
    for name in ("NaiveOLS", "BackDoorOLS"):
        model = baselines.get(name)
        slope[name] = float(model.predict(episode(1.0)) - model.predict(episode(0.0)))
    design = np.column_stack([np.ones(onset - 1), x[1:onset, 0], x[: onset - 1, 2]])
    delta = float(np.linalg.lstsq(design, x[1:onset, 1], rcond=None)[0][1])
    assert gamma * delta > 0.3
    assert slope["BackDoorOLS"] == pytest.approx(beta, abs=0.05)
    assert slope["NaiveOLS"] == pytest.approx(beta + gamma * delta, abs=0.05)
