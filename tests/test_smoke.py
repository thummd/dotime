"""Foundational smoke tests: imports, prior invariants, registry, round-trip, eval."""

from __future__ import annotations

import pytest
import torch

import causaltimeprior as ctp

# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


def test_version_and_eager_core():
    assert ctp.__version__ == "0.1.0"
    for name in [
        "CausalTimePrior",
        "TemporalSCM",
        "TemporalDAG",
        "TemporalGraphBuilder",
        "TemporalMechanism",
        "TemporalSCMBuilder",
        "InterventionSpec",
        "InterventionType",
        "InterventionSampler",
        "RegimeSwitchingTemporalSCM",
        "RegimeSwitchingSCMBuilder",
        "DEFAULT_CONFIG",
    ]:
        assert hasattr(ctp, name), name


def test_lazy_submodules_resolve():
    for name in ["extended", "continuous", "data", "benchmarks", "baselines", "evaluation"]:
        assert getattr(ctp, name) is not None


# --------------------------------------------------------------------------- #
# Prior invariants
# --------------------------------------------------------------------------- #


def test_generate_pair_shapes_and_finiteness():
    prior = ctp.CausalTimePrior(seed=0)
    x_obs, x_int, intervention, _scm = prior.generate_pair(T=64)
    assert x_obs.shape == x_int.shape
    assert x_obs.shape[0] == 64
    assert x_obs.ndim == 2
    # Non-diverged trajectories must be finite (diverged ones are zeroed, also finite).
    assert torch.isfinite(x_obs).all()
    assert torch.isfinite(x_int).all()
    assert isinstance(intervention, ctp.InterventionSpec)


def test_intervention_targets_within_range():
    prior = ctp.CausalTimePrior(seed=1)
    x_obs, _x_int, intervention, _scm = prior.generate_pair(T=32)
    n = x_obs.shape[-1]
    for t in intervention.targets:
        assert 0 <= t < n


# --------------------------------------------------------------------------- #
# Baseline registry
# --------------------------------------------------------------------------- #


def test_baseline_registry_lists_expected():
    names = set(ctp.baselines.available())
    assert {"Zero", "Mean", "VAR-OLS", "Oracle"} <= names


def test_dependency_free_baselines_instantiate():
    for name in ["Zero", "Mean", "VAR-OLS", "Oracle"]:
        assert ctp.baselines.get(name) is not None


def test_unknown_baseline_raises():
    with pytest.raises(KeyError):
        ctp.baselines.get("does-not-exist")


# --------------------------------------------------------------------------- #
# Suite round-trip + evaluation
# --------------------------------------------------------------------------- #


def test_suite_roundtrip_shapes():
    suite = ctp.benchmarks.load_benchmark("CTP-Identifiability-v1")
    assert len(suite) > 0
    seen_structures = set()
    for ep in suite:
        assert ep.x_obs.shape == ep.x_int.shape
        assert ep.y_true.numel() == ep.query_target.numel() == ep.query_time.numel()
        if ep.structure is not None:
            seen_structures.add(ep.structure)
    # The identifiability suite should cover multiple named structures.
    assert len(seen_structures) >= 2


def test_oracle_is_exact_on_fallback():
    suite = ctp.benchmarks.load_benchmark("CTP-Generic-100k")
    results = ctp.evaluation.evaluate(ctp.baselines.get("Oracle"), suite)
    assert results.pooled["rmse"] == pytest.approx(0.0, abs=1e-5)
    assert results.pooled["mae"] == pytest.approx(0.0, abs=1e-5)
    # summary() and to_dict() must work (the CLI calls both).
    assert "Oracle" in results.summary()
    assert results.to_dict()["baseline"] == "Oracle"
