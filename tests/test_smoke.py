"""Foundational smoke tests: imports, prior invariants, registry, round-trip, eval."""

from __future__ import annotations

import re

import pytest
import torch

import dotime as ctp

# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


def test_version_and_eager_core():
    # Exact value is pinned by test_version_strings_agree, not duplicated here.
    assert re.fullmatch(r"\d+\.\d+\.\d+", ctp.__version__)
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


def test_results_report_direction_accuracy_uncertainty(tmp_path):
    """Direction accuracy ships with its (exact binomial) standard error.

    The suites score one query per episode, so there is no clustering to
    correct for and sqrt(p(1-p)/n_valid) is the right SE. Leaderboard
    submissions carry it, so a point estimate is never reported bare.
    """
    _seed_local_suite(tmp_path, "dot-Generic-100k")
    suite = ctp.benchmarks.load_benchmark("dot-Generic-100k", cache_dir=tmp_path)
    results = ctp.evaluation.evaluate(ctp.baselines.get("Mean"), suite)

    for group in [results.pooled, *results.per_structure.values()]:
        n_valid, acc, se = group["dir_n_valid"], group["dir_acc"], group["dir_acc_se"]
        assert 0 <= n_valid <= results.n_queries
        if n_valid > 0:
            assert se == pytest.approx((acc * (1 - acc) / n_valid) ** 0.5)

    assert "dir_acc_se" in results.summary()
    assert "dir_acc_se" in results.to_dict()["pooled"]


def test_reference_harness_imports_without_optional_extras():
    """`dotime.reference` must stay importable without tabpfn/chronos installed.

    The console scripts are declared unconditionally in the wheel metadata, so
    an eager third-party import here would break `pip install dotime` users.
    """
    import importlib

    for mod in ("dotime.reference", "dotime.reference.reference_table"):
        importlib.import_module(mod)
    # The dependency-gated ones import too; only *calling* them needs the extra.
    for mod in ("dotime.reference.tabpfn", "dotime.reference.chronos"):
        importlib.import_module(mod)


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


def test_version_strings_agree():
    """`__version__`, pyproject, and CITATION.cff must be bumped together.

    They are three hand-edited copies of one fact; a mismatch silently stamps
    the wrong `package_version` into every leaderboard submission JSON.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    citation = re.search(r"^version: (\S+)", (root / "CITATION.cff").read_text(), re.M)
    assert pyproject is not None
    assert citation is not None
    assert ctp.__version__ == pyproject.group(1) == citation.group(1)
