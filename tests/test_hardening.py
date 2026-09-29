"""Tests for the opt-in stability hardening of the generic prior."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from dotime import DoTime
from dotime._activations import TanhSquare
from dotime.hardening import (
    RECOMMENDED_HARDENING,
    _systems,
    companion_spectral_radius,
    harden_scm,
    validate_hardening,
)
from dotime.prior import Square
from dotime.regime_switching_builder import RegimeSwitchingSCMBuilder
from dotime.utils import DEFAULT_CONFIG


def _prior(seed: int, hardening=None, **overrides) -> DoTime:
    """Build a prior the way the release build does (global seed, then instance seed)."""
    torch.manual_seed(seed)
    cfg = {**DEFAULT_CONFIG, **overrides}
    if hardening is not None:
        cfg["hardening"] = hardening
    return DoTime(config=cfg, seed=seed)


def _diverged(x_obs: torch.Tensor, x_int: torch.Tensor) -> bool:
    """An episode is diverged when either arm was returned all-zero."""
    return float(x_obs.abs().max()) == 0.0 or float(x_int.abs().max()) == 0.0


def test_validate_hardening():
    assert validate_hardening(None) is None
    assert validate_hardening({}) is None
    assert validate_hardening({"spectral_rho": 1}) == {
        "unit_norm_rows": False,
        "spectral_rho": 1.0,
        "bounded_square": False,
    }
    assert validate_hardening(RECOMMENDED_HARDENING) == RECOMMENDED_HARDENING
    with pytest.raises(ValueError, match="unknown hardening keys"):
        validate_hardening({"spectral_radius": 0.9})
    for bad in (0, -0.5):
        with pytest.raises(ValueError, match="positive"):
            validate_hardening({"spectral_rho": bad})
    for bad in ({"spectral_rho": "0.9"}, {"spectral_rho": True}, {"unit_norm_rows": 1}):
        with pytest.raises(TypeError):
            validate_hardening(bad)
    with pytest.raises(TypeError):
        validate_hardening([("spectral_rho", 0.9)])
    # The prior validates at construction, before any draw.
    with pytest.raises(ValueError):
        DoTime(config={**DEFAULT_CONFIG, "hardening": {"bogus": True}})


@pytest.mark.parametrize("regime", [False, True])
def test_hardening_draws_the_same_randomness(regime):
    """Hardening rescales sampled weights only: same graphs, interventions and RNG use."""
    warnings.simplefilter("ignore", RuntimeWarning)
    for seed in range(4):
        runs = []
        for hardening in (None, RECOMMENDED_HARDENING):
            prior = _prior(seed, hardening, N_max=12, K_max=4)
            if regime:
                out = prior.generate_regime_pair(T=40, num_regimes=2)
            else:
                out = prior.generate_pair(T=40)
            runs.append((out, prior.generator.get_state(), torch.get_rng_state()))
        (_, _, iv0, scm0), gen0, glob0 = runs[0]
        (_, _, iv1, scm1), gen1, glob1 = runs[1]
        # to_dict() encodes scalar, tensor and time-varying profile values alike.
        assert iv0.to_dict() == iv1.to_dict()
        assert torch.equal(gen0, gen1)
        assert torch.equal(glob0, glob1)
        assert list(scm0._topo) == list(scm1._topo)


def test_hardening_invariants():
    """The hardened SCM meets the stated caps and has no unbounded activation left."""
    cap = RECOMMENDED_HARDENING["spectral_rho"]
    checked = 0
    for seed in range(8):
        scm = _prior(seed, RECOMMENDED_HARDENING, N_max=20, K_max=4).sample_scm()
        assert companion_spectral_radius(scm) <= cap + 1e-9
        for rows, _n, _k in _systems(scm):
            for row in rows:
                norm = math.sqrt(sum(float(w.detach()) ** 2 for w, _, _ in row))
                assert norm <= 1.0 + 1e-6
        mechs = (
            [m for regime in scm.mechanisms for m in regime.values()]
            if hasattr(scm, "_regime_parents")
            else list(scm.mechanisms.values())
        )
        assert not any(isinstance(m.activation, Square) for m in mechs)
        checked += 1
    assert checked == 8


def test_harden_scm_reports_one_system_per_regime():
    torch.manual_seed(0)
    builder = RegimeSwitchingSCMBuilder(
        num_nodes=6,
        max_lag=2,
        activations=DoTime(seed=0).activations,
        gamma=DEFAULT_CONFIG["gamma"],
        sigma_w=DEFAULT_CONFIG["sigma_w"],
        sigma_b=DEFAULT_CONFIG["sigma_b"],
    )
    scm = builder.sample(torch.Generator().manual_seed(0), num_regimes=3)
    reports = harden_scm(scm, **RECOMMENDED_HARDENING)
    assert len(reports) == 3
    assert all(r.rho_after <= RECOMMENDED_HARDENING["spectral_rho"] + 1e-9 for r in reports)


def test_hardening_stabilises_large_graphs():
    """Regression: large generic graphs that diverge unhardened simulate when hardened.

    Seeds 0-7 are a fixed range, not a selection. Unhardened, five of them
    diverge at this size. With the recommended hardening none may.
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    base, hardened = 0, 0
    for seed in range(8):
        x_obs, x_int, _, _ = _prior(seed, None, N_max=30, K_max=6).generate_pair(T=40)
        base += _diverged(x_obs, x_int)
        x_obs, x_int, _, _ = _prior(seed, RECOMMENDED_HARDENING, N_max=30, K_max=6).generate_pair(
            T=40
        )
        assert torch.isfinite(x_obs).all()
        assert torch.isfinite(x_int).all()
        hardened += _diverged(x_obs, x_int)
    assert base >= 4
    assert hardened == 0


def test_tanh_square_matches_square_near_zero_and_is_bounded():
    x = torch.linspace(-0.1, 0.1, 21)
    assert torch.allclose(TanhSquare()(x), x**2, rtol=0.01, atol=1e-6)
    big = TanhSquare()(torch.tensor([-50.0, 50.0]))
    # tanh saturates to exactly 1.0 in float32, so the bound is inclusive.
    assert bool((big <= 1.0).all())
