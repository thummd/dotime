"""The ``"auto"`` direction-accuracy target (``dotime.evaluation.DEFAULT_DIR_TARGET``).

``"auto"`` scores the sign of the effect when the two arms of every episode
share their noise, which shows as bit-identical rows before the onset, and the
sign of the level otherwise. These tests pin the check on hand-built pairs, the
resolution and its warning, and what :func:`dotime.evaluation.evaluate` reports
in each case.
"""

from __future__ import annotations

import logging

import pytest
import torch

from dotime import baselines
from dotime.benchmarks import _SUITE_REGISTRY, BenchmarkSuite, episode_from_pair
from dotime.evaluation import (
    DEFAULT_DIR_TARGET,
    DIR_TARGET_MODES,
    NoiseCheck,
    check_shared_noise,
    describe_dir_target,
    evaluate,
    resolve_dir_target,
)
from dotime.interventions import InterventionSpec, InterventionType

T, N, ONSET = 40, 3, 20


def _pair(seed: int, *, shared: bool, effect: float = 1.0):
    """A hand-built pair whose arms share their noise or not.

    Args:
        seed: Seed of the observational arm.
        shared: Copy the observational arm before the onset (shared noise),
            or draw the interventional arm afresh (an independent twin).
        effect: Shift the interventional arm by this much from the onset on.

    Returns:
        ``(x_obs, x_int, intervention)``.
    """
    g = torch.Generator().manual_seed(seed)
    x_obs = torch.randn(T, N, generator=g)
    x_int = x_obs.clone() if shared else torch.randn(T, N, generator=g)
    x_int[ONSET:] += effect
    spec = InterventionSpec(
        targets=[0],
        times=list(range(ONSET, ONSET + 5)),
        intervention_type=InterventionType.HARD,
        values=1.5,
    )
    x_int[ONSET : ONSET + 5, 0] = 1.5
    return x_obs, x_int, spec


def _suite(shared: bool, n: int = 12, name: str = "dot-Generic-100k", version: str = "1.0.0"):
    """An in-memory suite of hand-built pairs.

    Args:
        shared: Whether the pairs share their noise.
        n: Number of episodes.
        name: Registered suite whose metadata the suite borrows.
        version: Version of that metadata.

    Returns:
        A :class:`~dotime.benchmarks.BenchmarkSuite`.
    """
    meta = _SUITE_REGISTRY[name].for_version(version)
    eps = [
        episode_from_pair(*_pair(i, shared=shared, effect=(-1.0) ** i), scm_id=i) for i in range(n)
    ]
    return BenchmarkSuite(meta, eps)


def test_the_default_is_auto_and_resolves_to_a_scored_target() -> None:
    assert DEFAULT_DIR_TARGET == "auto"
    assert DIR_TARGET_MODES == ("auto", "level", "effect")


def test_check_shared_noise_tells_the_pairings_apart() -> None:
    shared = check_shared_noise(_suite(shared=True))
    assert (shared.n_checked, shared.n_shared, shared.shared) == (12, 12, True)
    twins = check_shared_noise(_suite(shared=False))
    assert (twins.n_checked, twins.n_shared, twins.shared) == (12, 0, False)


def test_check_shared_noise_skips_zeroed_arms_and_counts_missing_cells_as_equal() -> None:
    x_obs, x_int, spec = _pair(0, shared=True)
    x_obs[3, 1] = x_int[3, 1] = float("nan")  # an observed suite masks both arms alike
    masked = episode_from_pair(x_obs, x_int, spec, scm_id=0)
    zeroed = episode_from_pair(*_pair(1, shared=False)[:1], torch.zeros(T, N), spec, scm_id=1)
    check = check_shared_noise([masked, zeroed])
    assert (check.n_checked, check.n_shared, check.n_zeroed, check.shared) == (1, 1, 1, True)
    # One differing cell before the onset is enough to call the pairing independent.
    x_obs2, x_int2, _ = _pair(2, shared=True)
    x_int2[ONSET - 1, 2] += 1e-6
    mixed = check_shared_noise([masked, episode_from_pair(x_obs2, x_int2, spec, scm_id=2)])
    assert (mixed.n_checked, mixed.n_shared, mixed.shared) == (2, 1, False)
    assert NoiseCheck(0, 0, 0, 0, 0).shared is False


def test_resolve_dir_target(caplog: pytest.LogCaptureFixture) -> None:
    shared, twins = NoiseCheck(5, 5, 5, 0, 0), NoiseCheck(5, 4, 0, 1, 0)
    assert resolve_dir_target("level") == "level"
    assert resolve_dir_target("effect", twins) == "effect"
    assert resolve_dir_target("auto", shared) == "effect"
    with caplog.at_level(logging.WARNING, logger="dotime.evaluation"):
        assert resolve_dir_target("auto", twins) == "level"
    assert "not a counterfactual effect" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dotime.evaluation"):
        resolve_dir_target("auto", twins, warn=False)
    assert caplog.text == ""
    with pytest.raises(ValueError, match="check_shared_noise"):
        resolve_dir_target("auto")
    with pytest.raises(ValueError, match="dir_target"):
        resolve_dir_target("sign")
    assert "1 with a zeroed arm" in describe_dir_target("auto", "level", twins)
    assert describe_dir_target("level", "level").endswith("(requested)")


def test_evaluate_scores_the_effect_on_shared_noise_and_reports_both() -> None:
    suite = _suite(shared=True)
    res = evaluate(baselines.get("Mean"), suite)
    assert (res.dir_target, res.dir_target_mode, res.pairs_share_noise) == ("effect", "auto", True)
    pooled = res.pooled
    assert {"dir_acc_level", "dir_acc_effect", "dir_n_valid_effect", "dir_acc_se_effect"} <= set(
        pooled
    )
    assert pooled["dir_acc"] == pooled["dir_acc_effect"]
    assert pooled["dir_n_valid"] == pooled["dir_n_valid_effect"]
    assert "acc_effect" in res.summary()
    assert res.to_dict()["pairs_share_noise"] is True
    # Asking for the level keeps the v1 number in dir_acc and still reports the effect.
    level = evaluate(baselines.get("Mean"), suite, dir_target="level")
    assert level.pooled["dir_acc"] == level.pooled["dir_acc_level"] == pooled["dir_acc_level"]
    assert level.pooled["dir_acc_effect"] == pooled["dir_acc_effect"]
    assert level.pooled["rmse"] == pooled["rmse"]


def test_evaluate_falls_back_to_the_level_on_independent_twins(
    caplog: pytest.LogCaptureFixture,
) -> None:
    suite = _suite(shared=False)
    with caplog.at_level(logging.WARNING, logger="dotime.evaluation"):
        res = evaluate(baselines.get("Mean"), suite)
    assert (res.dir_target, res.dir_target_mode, res.pairs_share_noise) == ("level", "auto", False)
    assert "dir_acc_effect" not in res.pooled
    assert res.pooled["dir_acc"] == res.pooled["dir_acc_level"]
    assert "not a counterfactual effect" in caplog.text
    assert "do not share their noise" in res.summary()
    # An explicit request still scores the effect, as before.
    effect = evaluate(baselines.get("Mean"), suite, dir_target="effect")
    assert effect.dir_target == "effect"
    assert effect.pooled["dir_acc"] == effect.pooled["dir_acc_effect"]


def test_auto_scores_the_level_on_the_archived_identifiability_files() -> None:
    # Whatever its arms look like, the 1.0.0 x_obs cannot give y_obs, so auto
    # must not raise where an explicit "effect" does.
    suite = _suite(shared=True, name="dot-Identifiability-v1", version="1.0.0")
    res = evaluate(baselines.get("Mean"), suite)
    assert res.dir_target == "level"
    assert "dir_acc_effect" not in res.pooled
    with pytest.raises(ValueError, match="realignment"):
        evaluate(baselines.get("Mean"), suite, dir_target="effect")


def test_target_qa_resolves_auto_from_the_episodes() -> None:
    from dotime.qa import target_qa

    shared = target_qa(list(_suite(shared=True)), group_by=None, log=None, raise_on_failure=False)
    twins = target_qa(list(_suite(shared=False)), group_by=None, log=None, raise_on_failure=False)
    assert (shared.dir_target, shared.check_effect) == ("effect", True)
    assert (twins.dir_target, twins.check_effect) == ("level", False)
