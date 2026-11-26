"""Tests for shared-noise pairs of regime-switching SCMs.

``RegimeSwitchingTemporalSCM.freeze_noise`` pre-draws the noise and the regime
path of a pair, so that the observational and the interventional arm differ
only through the intervention. Freezing is opt-in: an SCM that never freezes
draws exactly what it drew before, which the frozen fingerprints and
``tests/test_regime_canonical_weights.py`` pin.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import torch

from dotime import DoTime
from dotime.hardening import RECOMMENDED_HARDENING
from dotime.interventions import InterventionSpec, InterventionType
from dotime.regime_switching import RegimeSwitchingTemporalSCM

# Small repaired and stabilised SCMs: the mechanisms read their parents and do not diverge.
_REPAIRED = {"N_max": 5, "regime_canonical_weights": True, "hardening": RECOMMENDED_HARDENING}
_T, _BURN_IN, _ONSET, _END = 60, 20, 15, 40


def _scm(seed: int, repaired: bool = True, num_regimes: int = 3) -> RegimeSwitchingTemporalSCM:
    """Sample one regime-switching SCM from the prior.

    Args:
        seed: Seed of the prior and of both global generators.
        repaired: Whether the mechanisms read their parents.
        num_regimes: Number of regimes.

    Returns:
        The SCM, with nothing frozen.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = _REPAIRED if repaired else {"N_max": 5}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scm = DoTime(config=config, seed=seed).generate_regime_pair(T=_T, num_regimes=num_regimes)[
            3
        ]
    assert scm.frozen_regime_path() is None
    return scm


def _intervention(kind: InterventionType, target: int = 0) -> InterventionSpec:
    """An intervention of one type on one variable over the steps ``[_ONSET, _END]``.

    Args:
        kind: The intervention type.
        target: Column of the treated variable.

    Returns:
        The specification. A time-varying one ramps linearly over the window.
    """
    values = (lambda t: 0.1 * (t - _ONSET)) if kind == InterventionType.TIME_VARYING else 2.5
    return InterventionSpec(
        targets=[target],
        times=list(range(_ONSET, _END + 1)),
        intervention_type=kind,
        values=values,
    )


def _frozen_pair(scm, intervention, seed: int = 3):
    """Freeze a realisation and simulate both arms from it.

    Args:
        scm: The SCM.
        intervention: The intervention of the second arm.
        seed: Seed of the generator that draws the frozen realisation.

    Returns:
        ``(x_obs, x_int)``.
    """
    scm.freeze_noise(_T + _BURN_IN, generator=torch.Generator().manual_seed(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_obs = scm.sample_observational(_T, burn_in=_BURN_IN)
        x_int = scm.sample_interventional(_T, intervention, burn_in=_BURN_IN)
    return x_obs, x_int


@pytest.mark.parametrize(
    "kind", [InterventionType.HARD, InterventionType.SOFT, InterventionType.TIME_VARYING]
)
def test_frozen_arms_differ_only_through_the_intervention(kind):
    """The arms agree before the onset for every intervention type, and differ after it.

    A hard or time-varying intervention skips the noise draw of the treated
    variable in the unfrozen simulator, so this is the case freezing exists for.
    """
    moved = 0
    for seed in range(6):
        scm = _scm(seed)
        x_obs, x_int = _frozen_pair(scm, _intervention(kind))
        assert float(x_obs.abs().max()) > 0
        assert float(x_int.abs().max()) > 0
        assert torch.equal(x_obs[:_ONSET], x_int[:_ONSET])
        assert not torch.equal(x_obs[_ONSET : _END + 1, 0], x_int[_ONSET : _END + 1, 0])
        moved += bool((x_int[_ONSET:, 1:] - x_obs[_ONSET:, 1:]).abs().max() > 1e-6)
    # The treated variable has descendants in most sampled graphs.
    assert moved >= 3


def test_without_the_repair_only_the_treated_variable_moves():
    """Version 1.0.0 mechanisms read no parents, so nothing propagates under shared noise."""
    for seed in range(4):
        scm = _scm(seed, repaired=False)
        x_obs, x_int = _frozen_pair(scm, _intervention(InterventionType.HARD))
        assert torch.equal(x_obs[:, 1:], x_int[:, 1:])
        assert torch.equal(x_obs[:_ONSET, 0], x_int[:_ONSET, 0])
        assert torch.equal(x_obs[_END + 1 :, 0], x_int[_END + 1 :, 0])
        assert bool((x_int[_ONSET : _END + 1, 0] == 2.5).all())


def test_frozen_pair_depends_on_its_generator_only():
    """A frozen pair neither reads nor changes numpy's global generator."""
    scm = _scm(1)
    intervention = _intervention(InterventionType.HARD)
    np.random.seed(10)
    before = np.random.get_state()[1].copy()
    first = _frozen_pair(scm, intervention)
    assert np.array_equal(before, np.random.get_state()[1])
    np.random.seed(11)
    torch.manual_seed(12)
    second = _frozen_pair(scm, intervention)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    other = _frozen_pair(scm, intervention, seed=4)
    assert not torch.equal(first[0], other[0])


def test_both_arms_follow_the_frozen_regime_path():
    """The path that ``frozen_regime_path`` reports is the one both simulations take."""
    scm = _scm(2, num_regimes=5)
    scm.freeze_noise(_T + _BURN_IN, generator=torch.Generator().manual_seed(7))
    path = scm.frozen_regime_path(_BURN_IN)
    assert path.shape == (_T,)
    assert set(path.tolist()) <= set(range(5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, obs_path = scm.sample_observational(_T, burn_in=_BURN_IN, return_regimes=True)
        _, int_path = scm.sample_interventional(
            _T, _intervention(InterventionType.HARD), burn_in=_BURN_IN, return_regimes=True
        )
    assert np.array_equal(path, obs_path)
    assert np.array_equal(path, int_path)
    # The report is a copy: editing it must not change later simulations.
    path[:] = 0
    assert np.array_equal(scm.frozen_regime_path(_BURN_IN), obs_path)


def test_clearing_restores_the_unfrozen_stream():
    """Freezing and clearing leave no trace: the step-by-step draws are those of a fresh SCM."""

    def unfrozen(scm):
        np.random.seed(21)
        generator = torch.Generator().manual_seed(22)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return scm.sample_observational(_T, burn_in=_BURN_IN, generator=generator)

    fresh = unfrozen(_scm(5))
    scm = _scm(5)
    frozen_obs, _ = _frozen_pair(scm, _intervention(InterventionType.SOFT))
    scm.clear_noise()
    assert scm.frozen_regime_path() is None
    assert torch.equal(unfrozen(scm), fresh)
    assert not torch.equal(frozen_obs, fresh)


def test_frozen_length_must_match_the_simulation():
    """A simulation of another length fails instead of silently drawing fresh noise."""
    scm = _scm(0)
    with pytest.raises(ValueError, match="total_T must be positive"):
        scm.freeze_noise(0)
    scm.freeze_noise(_T + _BURN_IN, generator=torch.Generator().manual_seed(1))
    with pytest.raises(ValueError, match="frozen for 80 steps"):
        scm.sample_observational(_T, burn_in=_BURN_IN + 1)
    with pytest.raises(ValueError, match="frozen for 80 steps"):
        scm.sample_interventional(_T + 5, _intervention(InterventionType.HARD), burn_in=_BURN_IN)


def test_counterfactual_regime_pair_keeps_the_scm_and_the_intervention():
    """Both pair modes draw the same SCM and intervention, and only one shares the noise."""
    for seed in range(5):
        pairs = {}
        for mode in ("interventional", "counterfactual"):
            np.random.seed(seed)
            torch.manual_seed(seed)
            prior = DoTime(config=_REPAIRED, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pairs[mode] = prior.generate_regime_pair(T=_T, num_regimes=2, pair_mode=mode)
        (x_obs, x_int, iv, scm), (y_obs, y_int, iv_cf, scm_cf) = pairs.values()
        assert json.dumps(iv.to_dict(), sort_keys=True) == json.dumps(
            iv_cf.to_dict(), sort_keys=True
        )
        for dag, dag_cf in zip(scm.dags, scm_cf.dags, strict=True):
            assert set(dag.G_0.edges) == set(dag_cf.G_0.edges)
            assert all(np.array_equal(a, b) for a, b in zip(dag.G_lags, dag_cf.G_lags, strict=True))
        assert np.array_equal(scm.transition_matrix, scm_cf.transition_matrix)
        onset = min(iv.times)
        assert torch.equal(y_obs[:onset], y_int[:onset])
        assert not torch.equal(x_obs[:onset], x_int[:onset])
        assert scm.frozen_regime_path() is None
        assert scm_cf.frozen_regime_path(prior.config["burn_in"]).shape == (_T,)


def test_counterfactual_regime_pair_ignores_numpy_global_state():
    """A counterfactual pair is fixed by the prior's seed and the global torch seed.

    The prior samples parts of the SCM from the global torch generator, which the
    build seeds per episode. numpy's global generator, the source of the regime
    paths of an interventional pair, moves that pair and not the counterfactual one.
    """
    draws: dict[str, list] = {"interventional": [], "counterfactual": []}
    for mode, pairs in draws.items():
        for numpy_seed in (1, 2):
            np.random.seed(numpy_seed)
            torch.manual_seed(9)
            prior = DoTime(config=_REPAIRED, seed=9)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pairs.append(prior.generate_regime_pair(T=_T, num_regimes=3, pair_mode=mode))
    first, second = draws["counterfactual"]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    first, second = draws["interventional"]
    assert not torch.equal(first[0], second[0])


def test_regime_pair_mode_is_validated_before_any_draw():
    """An unknown pair mode leaves every generator untouched."""
    prior = DoTime(config=_REPAIRED, seed=4)
    state = prior.generator.get_state().clone()
    with pytest.raises(ValueError, match="pair_mode"):
        prior.generate_regime_pair(T=_T, pair_mode="shared")
    assert torch.equal(state, prior.generator.get_state())
