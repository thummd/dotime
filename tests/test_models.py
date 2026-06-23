"""Model consolidation + DoOverTimePFN inference path.

These tests need the ``[models]`` extra (``pfns``) and a local checkpoint; they
skip cleanly when either is absent so a core install is unaffected.
"""

from __future__ import annotations

import glob
import os

import pytest
import torch

pytest.importorskip("pfns", reason="model package needs the [models] extra")

from causaltimeprior.extended import ExtendedCausalTimePrior

_CKPTS = sorted(
    glob.glob(
        os.path.expanduser(
            "~/repos/do-over-time-pfn/checkpoints/**/do_over_time_pfn_best.pt"
        ),
        recursive=True,
    )
)
_needs_ckpt = pytest.mark.skipif(not _CKPTS, reason="no local DoOverTimePFN checkpoint")


def test_model_constructs_from_config():
    from causaltimeprior.models.do_over_time_pfn import DoOverTimePFN

    model = DoOverTimePFN(n_max=12, embed_size=64, n_encoder_layers=2, n_buckets=100)
    assert model.temporal_encoder.n_max == 12


def test_gdp_backend_raises_actionable_error():
    from causaltimeprior.models.encoder import TemporalEncoder

    with pytest.raises(ImportError, match=r"\[gdp\] extra"):
        TemporalEncoder(backend="gdp", embed_size=32, n_layers=1)


@_needs_ckpt
def test_load_dotpfn_and_predict_runs():
    from causaltimeprior import baselines, evaluation
    from causaltimeprior.benchmarks import (
        BenchmarkSuite,
        SuiteMetadata,
        episode_from_sample,
    )

    prior = ExtendedCausalTimePrior(tscm_structure="back_door", n_max=41, seed=0)
    episodes = [
        episode_from_sample(prior.generate_sample(T=80), structure="back_door", scm_id=i)
        for i in range(6)
    ]
    suite = BenchmarkSuite(
        SuiteMetadata("BD", "1.0.0", "LOCAL", "", "", len(episodes), structures=("back_door",)),
        episodes,
    )
    model = baselines.get("DoOverTimePFN", checkpoint=_CKPTS[0])
    results = evaluation.evaluate(model, suite)
    # The inference path runs and yields finite predictions (exact-number
    # reproduction is a separate, checkpoint-matched verification step).
    assert results.n_queries == len(episodes)
    for ep in suite:
        pred = model.predict(ep)
        assert torch.isfinite(pred).all()


def test_dotpfn_requires_checkpoint():
    from causaltimeprior import baselines

    with pytest.raises(ValueError, match="needs a trained checkpoint"):
        baselines.get("DoOverTimePFN")
