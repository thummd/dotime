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
def test_v2_build_is_reproducible_across_workers():
    # The v2 per-episode scheme must produce identical episodes regardless of the
    # worker count (the whole point of per-episode deterministic seeding).
    import torch

    from dotime._build import build_v2

    cfg = {"generator": "generic", "T": 60, "n_episodes": 12}
    seq = build_v2(cfg, 4242, 1.0, workers=1)
    par = build_v2(cfg, 4242, 1.0, workers=3)
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
