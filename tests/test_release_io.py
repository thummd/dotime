"""Tests for the frozen-suite on-disk schema and intervention serialization."""

from __future__ import annotations

import pytest
import torch

from causaltime import CausalTime, baselines, evaluation
from causaltime.interventions import (
    InterventionSpec,
    InterventionType,
    StepIntervention,
)

pytest.importorskip("pyarrow", reason="frozen-suite IO needs the evaluation extra")

from causaltime import _release_io
from causaltime.benchmarks import (
    SuiteMetadata,
    episode_from_pair,
)


def test_intervention_spec_roundtrip_scalar_and_profile():
    spec = InterventionSpec(
        targets=[1, 2], times=[10, 11, 12], intervention_type=InterventionType.HARD, values=3.5
    )
    back = InterventionSpec.from_dict(spec.to_dict())
    assert back.targets == spec.targets
    assert back.times == spec.times
    assert back.intervention_type == spec.intervention_type
    assert back.values == pytest.approx(3.5)

    prof = InterventionSpec(
        targets=[0],
        times=[5],
        intervention_type=InterventionType.TIME_VARYING,
        values=StepIntervention(step_time=7),
    )
    back2 = InterventionSpec.from_dict(prof.to_dict())
    assert isinstance(back2.values, StepIntervention)
    assert back2.values.step_time == 7


def test_intervention_spec_tensor_roundtrip():
    spec = InterventionSpec(
        targets=[0],
        times=[1, 2],
        intervention_type=InterventionType.SOFT,
        values=torch.tensor([1.0, 2.0, 3.0]),
    )
    back = InterventionSpec.from_dict(spec.to_dict())
    assert torch.allclose(back.values, spec.values)


def _make_suite(tmp_path, n=10):
    prior = CausalTime(seed=3)
    eps = []
    for i in range(n):
        x_obs, x_int, iv, _scm = prior.generate_pair(T=60)
        eps.append(
            episode_from_pair(
                x_obs, x_int, iv, structure="back_door" if i % 2 else "front_door", scm_id=i
            )
        )
    meta = SuiteMetadata(
        name="RT",
        version="1.0.0",
        zenodo_record_id="LOCAL",
        doi="",
        description="round-trip",
        n_episodes=n,
        structures=("back_door", "front_door"),
    )
    suite_dir = tmp_path / "RT-1.0.0"
    _release_io.write_suite(meta, eps, suite_dir, package_version="0.1.0", seed=3)
    return meta, eps, suite_dir


def test_suite_write_read_exact(tmp_path):
    meta, eps, suite_dir = _make_suite(tmp_path)
    assert (suite_dir / "manifest.json").exists()
    suite = _release_io.read_suite(meta, suite_dir)
    assert len(suite) == len(eps)
    for orig, got in zip(eps, suite, strict=True):
        assert torch.allclose(orig.x_obs, got.x_obs)
        assert torch.allclose(orig.x_int, got.x_int)
        assert torch.allclose(orig.y_true, got.y_true)
        assert orig.structure == got.structure
        assert orig.intervention.targets == got.intervention.targets


def test_oracle_exact_on_reloaded_suite(tmp_path):
    meta, _eps, suite_dir = _make_suite(tmp_path)
    suite = _release_io.read_suite(meta, suite_dir)
    results = evaluation.evaluate(baselines.get("Oracle"), suite)
    assert results.pooled["rmse"] == pytest.approx(0.0, abs=1e-5)


def test_checksum_mismatch_detected(tmp_path):
    meta, _eps, suite_dir = _make_suite(tmp_path)
    shard = suite_dir / "shard-0000.parquet"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _release_io.read_suite(meta, suite_dir)
