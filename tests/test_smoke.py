"""Foundational smoke tests: imports, prior invariants, registry, round-trip, eval."""

from __future__ import annotations

import pytest
import torch

import dotime as ctp

# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


def test_version_and_eager_core():
    assert ctp.__version__ == "0.1.1"
    for name in [
        "DoTime",
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
    prior = ctp.DoTime(seed=0)
    x_obs, x_int, intervention, _scm = prior.generate_pair(T=64)
    assert x_obs.shape == x_int.shape
    assert x_obs.shape[0] == 64
    assert x_obs.ndim == 2
    # Non-diverged trajectories must be finite (diverged ones are zeroed, also finite).
    assert torch.isfinite(x_obs).all()
    assert torch.isfinite(x_int).all()
    assert isinstance(intervention, ctp.InterventionSpec)


def test_intervention_targets_within_range():
    prior = ctp.DoTime(seed=1)
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


def _seed_local_suite(cache_dir, name, n=8):
    """Write a tiny suite into the loader's cache so load_benchmark reads it
    locally (the hosted suites are GBs; we don't download them in unit tests)."""
    pytest.importorskip("pyarrow")
    from dotime import _release_io
    from dotime.benchmarks import _SUITE_REGISTRY, episode_from_pair

    meta = _SUITE_REGISTRY[name]
    prior = ctp.DoTime(seed=0)
    structs = meta.structures or (None,)
    eps = [
        episode_from_pair(
            *prior.generate_pair(T=60)[:3], structure=structs[i % len(structs)], scm_id=i
        )
        for i in range(n)
    ]
    _release_io.write_suite(
        meta, eps, cache_dir / f"{name}-{meta.version}", package_version="test", seed=0
    )


def test_suite_roundtrip_shapes(tmp_path):
    # Seed a local cache copy and load it (no network / no GB download).
    _seed_local_suite(tmp_path, "dot-Identifiability-v1")
    suite = ctp.benchmarks.load_benchmark("dot-Identifiability-v1", cache_dir=tmp_path)
    assert len(suite) > 0
    seen_structures = set()
    for ep in suite:
        assert ep.x_obs.shape == ep.x_int.shape
        assert ep.y_true.numel() == ep.query_target.numel() == ep.query_time.numel()
        if ep.structure is not None:
            seen_structures.add(ep.structure)
    assert len(seen_structures) >= 2


def test_released_episodes_store_full_unmasked_xobs():
    # Identifiability episodes must store the FULL observational trajectory
    # (causal masking is a model-input transform, not part of the released data).
    from dotime.benchmarks import episode_from_sample
    from dotime.extended import ExtendedDoTime

    prior = ExtendedDoTime(tscm_structure="back_door", n_max=41, seed=0)
    ep = episode_from_sample(prior.generate_sample(T=80), structure="back_door")
    onset = min(ep.intervention.times)
    # At least some post-onset observational values are non-zero (i.e. not masked).
    assert bool((ep.x_obs[onset:] != 0).any())


def test_oracle_is_exact_on_loaded_suite(tmp_path):
    _seed_local_suite(tmp_path, "dot-Generic-100k")
    suite = ctp.benchmarks.load_benchmark("dot-Generic-100k", cache_dir=tmp_path)
    results = ctp.evaluation.evaluate(ctp.baselines.get("Oracle"), suite)
    assert results.pooled["rmse"] == pytest.approx(0.0, abs=1e-5)
    assert results.pooled["mae"] == pytest.approx(0.0, abs=1e-5)
    # summary() and to_dict() must work (the CLI calls both).
    assert "Oracle" in results.summary()
    assert results.to_dict()["baseline"] == "Oracle"


def test_scale_beyond_default_bounds():
    """N_max/K_max are config bounds, not architectural limits.

    Backs the Limitations claim that the generator scales past the frozen
    suites' DEFAULT_CONFIG (N<=10, K<=3) via a config override.
    """
    import warnings

    from dotime import DoTime
    from dotime.utils import DEFAULT_CONFIG

    cfg = {**DEFAULT_CONFIG, "N_max": 60, "K_max": 8}
    sizes = []
    for seed in range(8):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_obs, x_int, _, _ = DoTime(config=cfg, seed=seed).generate_pair(T=40)
        assert x_obs.shape[0] == 40
        assert x_obs.shape == x_int.shape
        sizes.append(x_obs.shape[1])
    # the override actually widens the sampled graph beyond the default cap
    assert max(sizes) > 10
