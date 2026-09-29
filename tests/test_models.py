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

from dotime.extended import ExtendedDoTime

_CKPTS = sorted(
    glob.glob(
        os.path.expanduser("~/repos/do-over-time-pfn/checkpoints/**/do_over_time_pfn_best.pt"),
        recursive=True,
    )
)
_needs_ckpt = pytest.mark.skipif(not _CKPTS, reason="no local DoOverTimePFN checkpoint")


def test_model_constructs_from_config():
    from dotime.models.do_over_time_pfn import DoOverTimePFN

    model = DoOverTimePFN(n_max=12, embed_size=64, n_encoder_layers=2, n_buckets=100)
    assert model.temporal_encoder.n_max == 12


def test_gdp_backend_raises_actionable_error():
    from dotime.models.encoder import TemporalEncoder

    with pytest.raises(ImportError, match=r"\[gdp\] extra"):
        TemporalEncoder(backend="gdp", embed_size=32, n_layers=1)


@_needs_ckpt
def test_load_dotpfn_and_predict_runs():
    from dotime import baselines, evaluation
    from dotime.benchmarks import (
        BenchmarkSuite,
        SuiteMetadata,
        episode_from_sample,
    )

    prior = ExtendedDoTime(tscm_structure="back_door", n_max=41, seed=0)
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
    from dotime import baselines

    with pytest.raises(ValueError, match="needs a trained checkpoint"):
        baselines.get("DoOverTimePFN")


@pytest.mark.parametrize("readout", ["mean", "mean_last", "attn"])
def test_readout_variants_construct_and_run(readout):
    """Recency-aware readouts (s11 checkpoints) must load and run here unchanged."""
    from dotime.models.do_over_time_pfn import DoOverTimePFN

    model = DoOverTimePFN(
        n_max=12,
        embed_size=32,
        n_heads=4,
        n_encoder_layers=1,
        n_buckets=50,
        head_type="quantile",
        readout=readout,
    ).eval()
    X = torch.randn(3, 40, 12)
    mask = torch.ones(3, 12)
    onset = torch.full((3,), 25, dtype=torch.long)
    h = model.temporal_encoder(X, mask, int_onset_idx=onset)
    assert h.shape == (3, 12, 32)
    assert torch.isfinite(h).all()
    assert model.temporal_encoder.readout == readout


def test_unknown_readout_rejected():
    from dotime.models.encoder import TemporalEncoder

    with pytest.raises(ValueError, match="readout"):
        TemporalEncoder(embed_size=32, n_layers=1, readout="last_only")


def test_token_lags_construct_and_run():
    """s11 checkpoints may carry token_lags; the vendored model must run them."""
    from dotime.models.do_over_time_pfn import DoOverTimePFN

    model = DoOverTimePFN(
        n_max=12,
        embed_size=32,
        n_heads=4,
        n_encoder_layers=1,
        n_buckets=50,
        head_type="quantile",
        readout="mean_last",
        token_lags=3,
    ).eval()
    assert model.temporal_encoder.expand_values.in_features == 4
    X = torch.randn(2, 40, 12)
    h = model.temporal_encoder(X, torch.ones(2, 12), int_onset_idx=torch.full((2,), 25))
    assert h.shape == (2, 12, 32)
    assert torch.isfinite(h).all()
    f = model.temporal_encoder._lagged_features(X)
    assert torch.equal(f[:, 7, :, 2], X[:, 5, :])


def test_quantile_head_per_horizon_routing():
    """K>0 routes each query to its horizon's projection; K=0 keeps the legacy keys."""
    from dotime.models.quantile_head import QuantileHead

    torch.manual_seed(0)
    legacy = QuantileHead(embed_size=16)
    assert {k.split(".")[0] for k in legacy.state_dict()} == {"tau_levels", "projection"}
    head = QuantileHead(embed_size=16, n_horizon_heads=3)
    h = torch.randn(5, 16)
    horizon = torch.tensor([0, 1, 2, 7, 1])  # 7 clamps to the last head
    out = head(h, horizon=horizon)
    for i, k in enumerate([0, 1, 2, 2, 1]):
        assert torch.allclose(out[i], head.horizon_projections[k](h[i : i + 1])[0])
    with pytest.raises(ValueError):
        head(h)


def test_model_routes_query_offset_to_horizon_heads():
    """DoOverTimePFN derives the query offset and routes it to the per-horizon head."""
    from dotime.models.do_over_time_pfn import DoOverTimePFN

    model = DoOverTimePFN(
        n_max=8,
        embed_size=16,
        n_heads=2,
        n_encoder_layers=1,
        n_cross_attn_heads=2,
        head_type="quantile",
        context_window=16,
        horizon_heads=4,
    )
    B, T, N = 3, 30, 8
    batch = {
        "X_obs_norm": torch.randn(B, T, N),
        "variable_mask": torch.ones(B, N, dtype=torch.bool),
        "intervention_target": torch.zeros(B, dtype=torch.long),
        "intervention_type": torch.zeros(B, dtype=torch.long),
        "intervention_value": torch.zeros(B),
        "intervention_time_start": torch.full((B,), 0.5),
        "intervention_time_end": torch.ones(B),
        "query_target": torch.full((B,), 7, dtype=torch.long),
        "query_time": torch.tensor([0.5, 0.6, 0.9]),
        "int_onset_idx": torch.full((B,), 15, dtype=torch.long),
    }
    out = model(batch)
    assert out.shape == (B, model.quantile_head.n_quantiles)
    assert torch.equal(model._query_offset(batch), torch.tensor([0, 3, 12]))


def test_horizon_mixers_route_each_query_to_its_own_path():
    """K mixers: every query goes through the mixer and head of its offset."""
    from dotime.models.do_over_time_pfn import DoOverTimePFN

    torch.manual_seed(0)
    model = DoOverTimePFN(
        n_max=8,
        embed_size=16,
        n_heads=2,
        n_encoder_layers=1,
        n_cross_attn_heads=2,
        head_type="quantile",
        context_window=16,
        horizon_mixers=4,
    ).eval()
    assert model.quantile_head.n_horizon_heads == 4
    assert model.cross_variable_mixer is None
    B, T, N = 4, 30, 8
    batch = {
        "X_obs_norm": torch.randn(B, T, N),
        "variable_mask": torch.ones(B, N, dtype=torch.bool),
        "intervention_target": torch.zeros(B, dtype=torch.long),
        "intervention_type": torch.zeros(B, dtype=torch.long),
        "intervention_value": torch.zeros(B),
        "intervention_time_start": torch.full((B,), 0.5),
        "intervention_time_end": torch.ones(B),
        "query_target": torch.full((B,), 7, dtype=torch.long),
        "query_time": torch.tensor([0.5, 0.6, 0.9, 0.5]),
        "int_onset_idx": torch.full((B,), 15, dtype=torch.long),
    }
    out = model(batch)
    h = model.encode(batch)
    off = model._query_offset(batch).clamp(0, 3)
    for i in range(B):
        k = int(off[i])
        hc = model.horizon_mixers[k](
            h_vars=h[i : i + 1],
            intervention_target=batch["intervention_target"][i : i + 1],
            intervention_type=batch["intervention_type"][i : i + 1],
            intervention_value=batch["intervention_value"][i : i + 1],
            intervention_time_start=batch["intervention_time_start"][i : i + 1],
            intervention_time_end=batch["intervention_time_end"][i : i + 1],
            query_target=batch["query_target"][i : i + 1],
            query_time=batch["query_time"][i : i + 1],
            variable_mask=batch["variable_mask"][i : i + 1],
            query_offset=off[i : i + 1],
        )
        ref = model.quantile_head.horizon_projections[k](hc)[0]
        assert torch.allclose(out[i], ref, atol=1e-5)
