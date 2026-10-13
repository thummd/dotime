"""The model options added by the do-over-time-pfn recovery program (October 2026).

The private trainer added, from wave W1 on, the treatment's normalised factual
onset value on the query token (``query_a_ref``) and a centred do-value, and in
wave W3 the in-context statistic tokens (``stat_tokens``), the joint-token
branch (``joint_branch``) and the learned role vectors (``role_embed``). The
vendored model must build and run every one of them so that pilot and s14
checkpoints load here strictly, and the reference scorer must feed a checkpoint
the input convention it was trained with. Every option is off by default and
silent at init, so legacy checkpoints and their predictions are untouched.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("pfns", reason="model package needs the [models] extra")

from dotime.models.do_over_time_pfn import DoOverTimePFN
from dotime.models.stat_tokens import InContextStatistics

_PILOT_CKPT = os.path.expanduser(
    "~/.cache/dotime-tools/pilot_s14/w3b_all_int_seed42/do_over_time_pfn_last.pt"
)


def _model(**kw):
    torch.manual_seed(0)
    args = dict(
        n_max=12,
        embed_size=32,
        n_heads=4,
        n_encoder_layers=1,
        n_cross_attn_heads=4,
        n_mixer_layers=1,
        head_type="quantile",
    )
    args.update(kw)
    return DoOverTimePFN(**args).eval()


def _batch(B=4, T=120, N=5, n_max=12, onset=90):
    """A synthetic normalised batch with the fields the model reads."""
    g = torch.Generator().manual_seed(7)
    x = torch.zeros(B, T, n_max)
    x[:, :, :N] = torch.randn(B, T, N, generator=g)
    x[:, onset:, :] = 0.0
    mask = torch.zeros(B, n_max)
    mask[:, :N] = 1.0
    return {
        "X_obs_norm": x,
        "variable_mask": mask,
        "int_onset_idx": torch.full((B,), onset),
        "intervention_target": torch.zeros(B, dtype=torch.long),
        "intervention_type": torch.zeros(B, dtype=torch.long),
        "intervention_value": torch.linspace(-1.5, 1.5, B),
        "intervention_time_start": torch.full((B,), onset / T),
        "intervention_time_end": torch.full((B,), (onset + 5) / T),
        "query_target": torch.full((B,), N - 1, dtype=torch.long),
        "query_time": torch.full((B,), (onset + 2) / T),
        "query_offset": torch.full((B,), 2),
        "a_ref_norm": torch.linspace(-0.5, 0.5, B),
    }


@pytest.mark.parametrize(
    "kw",
    [
        {"query_a_ref": True},
        {"stat_tokens": "ols_a"},
        {"stat_tokens": "ols_path", "role_embed": True},
        {"joint_branch": True, "joint_layers": 1},
    ],
)
def test_options_are_silent_at_init_and_round_trip_strictly(kw):
    batch = _batch()
    base = _model()
    with torch.no_grad():
        out_base = base(batch)
    model = _model(**kw)
    if not kw.get("query_a_ref"):
        # The wider query encoder of query_a_ref cannot take the base weights.
        model.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        out = model(batch)
    if kw.get("joint_branch"):
        # The joint token is appended and attended to from the start.
        assert model.encode(batch).shape == (4, 13, 32)
    elif kw.get("query_a_ref"):
        # A wider query encoder is a new random layer, so only the shape is fixed.
        assert model.cross_variable_mixer.query_encoder[0].in_features == 12 + 1 + 1
        assert out.shape == out_base.shape
    else:
        assert torch.allclose(out, out_base, atol=1e-6), f"{kw} must be silent at init"
    cfg = dict(
        n_max=12,
        embed_size=32,
        n_heads=4,
        n_encoder_layers=1,
        n_cross_attn_heads=4,
        n_mixer_layers=1,
        head_type="quantile",
        **kw,
    )
    DoOverTimePFN(**cfg).load_state_dict(model.state_dict(), strict=True)


def test_role_vectors_mark_the_queried_token():
    batch = _batch()
    model = _model(role_embed=True)
    with torch.no_grad():
        model.role_embed[1].normal_(0, 0.5)
        out = model(batch)
        other = dict(batch)
        other["query_target"] = torch.full((4,), 1, dtype=torch.long)
        assert not torch.allclose(model(other), out, atol=1e-6)
    assert _model().role_embed is None


def test_query_a_ref_reads_the_field_and_needs_it():
    batch = _batch()
    model = _model(query_a_ref=True)
    with torch.no_grad():
        out = model(batch)
        shifted = dict(batch)
        shifted["a_ref_norm"] = batch["a_ref_norm"] + 1.0
        assert not torch.allclose(model(shifted), out, atol=1e-6)
    missing = {k: v for k, v in batch.items() if k != "a_ref_norm"}
    with pytest.raises(ValueError, match="a_ref_norm"):
        model(missing)


def test_path_statistics_recover_a_planted_mediated_chain():
    torch.manual_seed(2)
    B, T, N = 2, 420, 5
    x = torch.randn(B, T, N)
    for t in range(1, T):
        x[:, t, 2] = 0.8 * x[:, t, 0] + 0.3 * x[:, t - 1, 2] + 0.05 * torch.randn(B)
        x[:, t, 4] = 0.6 * x[:, t, 2] + 0.3 * x[:, t - 1, 4] + 0.05 * torch.randn(B)
    stats = InContextStatistics(embed_size=8, n_lags=3, mode="ols_path")
    feats = stats.features(
        x,
        torch.ones(B, N),
        torch.full((B,), 400),
        torch.zeros(B, dtype=torch.long),
        torch.full((B,), 4, dtype=torch.long),
    )
    assert feats.shape == (B, N, 18)
    m = feats[0, 2]
    assert abs(m[0] - 0.8) < 0.05
    assert abs(m[stats.n_leg1] - 0.6) < 0.05
    assert feats[0, 4, -1] == 1.0
    assert feats[0, 0, -2] == 1.0


def test_loader_rejects_a_checkpoint_whose_weights_do_not_match_its_config(tmp_path):
    from dotime.models.loader import load_dotpfn

    model = _model(role_embed=True)
    state = {k: v for k, v in model.state_dict().items() if k != "role_embed"}
    path = tmp_path / "bad.pt"
    cfg = dict(
        n_max=12,
        embed_size=32,
        n_heads=4,
        n_encoder_layers=1,
        n_cross_attn_heads=4,
        n_mixer_layers=1,
        head_type="quantile",
        role_embed=True,
    )
    torch.save({"config": cfg, "model_state_dict": state, "head_type": "quantile"}, path)
    with pytest.raises(ValueError, match="missing"):
        load_dotpfn(path)
    good = tmp_path / "good.pt"
    torch.save(
        {
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "head_type": "quantile",
            "data": {"center_intervention_value": True},
        },
        good,
    )
    loaded = load_dotpfn(good)
    assert loaded.input_convention == {"center_intervention_value": True}
    assert load_dotpfn(path.with_name("good.pt")).role_embed is not None


def _episode():
    from dotime.benchmarks import episode_from_sample
    from dotime.extended import ExtendedDoTime

    prior = ExtendedDoTime(tscm_structure="back_door", n_max=41, seed=3)
    return episode_from_sample(prior.generate_sample(T=120), structure="back_door", scm_id=0)


def test_batch_builder_follows_the_checkpoint_convention():
    from dotime.reference.pfn import episode_to_batch_interp

    ep = _episode()
    onset = min(ep.intervention.times)
    a = ep.intervention.targets[0]
    pre = ep.x_obs[:onset, a]
    v = float(ep.intervention.values)
    legacy = episode_to_batch_interp(ep, 12, "cpu")
    centred = episode_to_batch_interp(ep, 12, "cpu", convention={"center_intervention_value": True})
    sd = max(float(pre.std()), 1e-4)
    mu = float(pre.mean())
    assert torch.allclose(legacy["intervention_value"], torch.tensor([v / sd]), atol=1e-5)
    assert torch.allclose(centred["intervention_value"], torch.tensor([(v - mu) / sd]), atol=1e-5)
    a_ref = float(ep.x_obs[onset, a])
    for b in (legacy, centred):
        assert torch.allclose(b["a_ref_norm"], torch.tensor([(a_ref - mu) / sd]), atol=1e-5)
    # Everything but the do-value is convention independent.
    for k in legacy:
        if k != "intervention_value":
            assert torch.equal(legacy[k], centred[k]), k
    obs = episode_to_batch_interp(ep, 12, "cpu", observational=True)
    assert float(obs["intervention_value"]) == 0.0
    assert torch.equal(obs["a_ref_norm"], legacy["a_ref_norm"])


@pytest.mark.skipif(not os.path.exists(_PILOT_CKPT), reason="no local pilot checkpoint")
def test_pilot_checkpoint_loads_strictly_and_scores_an_episode():
    from dotime.reference.pfn import PFNRef

    ref = PFNRef(_PILOT_CKPT)
    assert ref.model.stat_tokens == "ols_a"
    assert ref.model.query_a_ref
    assert ref.convention.get("center_intervention_value") is True
    pred = ref.predict(_episode())
    assert pred.shape == (1,)
    assert torch.isfinite(pred).all()
