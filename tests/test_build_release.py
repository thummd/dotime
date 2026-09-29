"""Smoke test for the reproducible suite build (scripts/build_release.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow", reason="build_release writes parquet (evaluation extra)")
pytest.importorskip("yaml")

from dotime import _release_io, baselines, evaluation
from dotime.benchmarks import SuiteMetadata

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"


def _load_build_release():
    spec = importlib.util.spec_from_file_location("build_release", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_build_release_micro_all_suites(tmp_path):
    br = _load_build_release()
    # Tiny scale across all four suites.
    rc = br.main(["--scale", "0.0005", "--output-dir", str(tmp_path), "--timestamp", "T"])
    assert rc == 0

    run_dir = tmp_path / "T"
    build_manifest = json.loads((run_dir / "build_manifest.json").read_text())
    assert {s["name"] for s in build_manifest["suites"]} == {
        "dot-Identifiability-v1",
        "dot-RegimeSwitch-v1",
        "dot-Continuous-v1",
        "dot-Generic-100k",
    }
    assert "torch" in build_manifest
    assert "config_hash" in build_manifest

    # Every suite round-trips through the loader and Oracle is exact.
    for suite_dir in sorted(run_dir.glob("dot-*")):
        manifest = json.loads((suite_dir / "manifest.json").read_text())
        assert (suite_dir / "croissant.json").exists()
        meta = SuiteMetadata(
            name=manifest["name"],
            version=manifest["version"],
            zenodo_record_id="LOCAL",
            doi="",
            description="",
            n_episodes=manifest["n_episodes"],
            structures=tuple(manifest["structures"]),
        )
        suite = _release_io.read_suite(meta, suite_dir)
        assert len(suite) >= 1
        results = evaluation.evaluate(baselines.get("Oracle"), suite)
        assert results.pooled["rmse"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.slow
def test_build_is_reproducible_across_workers():
    # The v2 per-episode scheme must produce identical episodes regardless of the
    # worker count (the whole point of per-episode deterministic seeding).
    import torch

    from dotime._build import build_suite

    cfg = {"generator": "generic", "T": 60, "n_episodes": 12}
    seq = build_suite(cfg, 4242, 1.0, workers=1)
    par = build_suite(cfg, 4242, 1.0, workers=3)
    assert len(seq) == len(par) == 12
    assert all(
        torch.equal(a.x_obs, b.x_obs)
        and torch.equal(a.x_int, b.x_int)
        and torch.equal(a.y_true, b.y_true)
        for a, b in zip(seq, par, strict=True)
    )


@pytest.mark.slow
def test_identifiability_covers_all_eight_structures(tmp_path):
    br = _load_build_release()
    rc = br.main(
        [
            "--suite",
            "dot-Identifiability-v1",
            "--scale",
            "0.001",
            "--output-dir",
            str(tmp_path),
            "--timestamp",
            "T",
        ]
    )
    assert rc == 0
    suite_dir = next((tmp_path / "T").glob("dot-Identifiability-v1*"))
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    assert len(manifest["structures"]) == 8


def test_stability_retries_removes_divergence():
    """The opt-in stability_retries flag resamples diverged generic episodes.

    v1.0.0 (retries=0) ships ~30% zeroed (diverged) generic episodes. A pair is
    diverged when EITHER arm is all-zero: the arms are separate simulations, and
    about 1.4% of v1.0.0 Generic episodes have only one arm zeroed. The hardened
    build (retries>0) must remove both kinds, while retries=0 stays the
    unretried first draw, i.e. byte-identical to the release.
    """
    import warnings

    import torch

    from dotime import DoTime
    from dotime._build import episode_specs, make_episode

    cfg = {"generator": "generic", "n_episodes": 300, "T": 200, "seed": 20260714}

    def zeroed_arms(ep):
        return float(ep.x_obs.abs().max()) == 0.0, float(ep.x_int.abs().max()) == 0.0

    def build(retries):
        c = {**cfg, "stability_retries": retries}
        specs = episode_specs(c, c["seed"], 1.0)[:200]
        assert specs[0].get("stability_retries") == retries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            episodes = [make_episode(sp) for sp in specs]
        for ep in episodes:
            assert ep.metadata["diverged"] == any(zeroed_arms(ep)), ep.scm_id
        return specs, episodes

    specs, baseline = build(0)
    rate = sum(ep.metadata["diverged"] for ep in baseline) / len(baseline)
    assert rate > 0.10, f"expected sizable v1.0.0 divergence, got {rate:.2%}"
    # This config holds both half-diverged kinds (only x_obs zeroed, only x_int
    # zeroed), so the either-arm rule is exercised, not just the both-arm case.
    half = [i for i, ep in enumerate(baseline) if sum(zeroed_arms(ep)) == 1]
    assert {zeroed_arms(baseline[i]) for i in half} == {(True, False), (False, True)}
    # retries=0 must be the plain first draw of the per-episode seed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in half:
            torch.manual_seed(specs[i]["seed"])
            x_obs, x_int, _, _ = DoTime(seed=specs[i]["seed"]).generate_pair(T=cfg["T"])
            assert torch.equal(baseline[i].x_obs, x_obs)
            assert torch.equal(baseline[i].x_int, x_int)

    _, hardened = build(20)
    # deterministic build (fixed seed): retries=20 fully eliminates divergence.
    zeroed = [ep.scm_id for ep in hardened if any(zeroed_arms(ep))]
    assert zeroed == [], f"retries should eliminate divergence, still zeroed: {zeroed}"


def test_identifiability_retry_seed_stays_in_numpy_range():
    # ExtendedDoTime seeds numpy.random.RandomState (< 2**32); the perturbed
    # resample seed must never overflow it, even from the largest episode seed.
    from dotime._build import episode_seed, identifiability_retry_seed

    big = episode_seed(20261719, 10_799)
    assert identifiability_retry_seed(big, 0) == big
    for attempt in (1, 2, 3):
        s = identifiability_retry_seed(big, attempt)
        assert 0 <= s < 2**31
        assert s != big


def test_identifiability_specs_carry_per_structure_query_offsets():
    from dotime._build import episode_specs

    cfg = {
        "generator": "identifiability",
        "T": 40,
        "episodes_per_structure": 1,
        "structures": {"back_door": 1, "mediator": 2},
        "query_offsets": {"mediator": [1, 1]},
    }
    specs = {s["structure"]: s for s in episode_specs(cfg, 1, 1.0)}
    assert specs["back_door"]["query_offset_range"] == (0, 0)  # default: onset
    assert specs["mediator"]["query_offset_range"] == (1, 1)
