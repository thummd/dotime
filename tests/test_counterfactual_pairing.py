"""Shared-noise (counterfactual) pairing of the discrete generator.

The v1.0.0 discrete suites drew independent noise for the two arms
(interventional twins). ``pair_mode="counterfactual"`` freezes one realisation
per episode and shares it, which must (a) leave the default path byte-identical,
(b) make the arms agree exactly before the intervention onset, (c) keep the
do-value in the canonical target column, (d) make ``Y_causal_effect`` the exact
per-episode difference, and (e) stay seed-deterministic.
"""

from __future__ import annotations

import pytest
import torch

from dotime.extended import ExtendedDoTime

STRUCTS = ("back_door", "front_door", "instrumental_variable", "unobserved_confounder")


def _sample(struct, pair_mode, seed=11, t_len=120):
    torch.manual_seed(seed)
    gen = ExtendedDoTime(tscm_structure=struct, n_max=41, seed=seed, pair_mode=pair_mode)
    return gen.generate_sample(T=t_len)


@pytest.mark.parametrize("struct", STRUCTS)
def test_counterfactual_arms_agree_before_onset(struct):
    s = _sample(struct, "counterfactual")
    n, onset = int(s["num_vars"]), int(s["int_onset_idx"])
    assert onset > 0
    assert torch.equal(s["X_obs_full"][:onset, :n], s["X_int"][:onset, :n])


@pytest.mark.parametrize("struct", STRUCTS)
def test_counterfactual_effect_and_do_value(struct):
    t_len = 120
    s = _sample(struct, "counterfactual", t_len=t_len)
    onset, tgt = int(s["int_onset_idx"]), int(s["intervention_target"])
    assert float(s["X_int"][onset, tgt]) == pytest.approx(
        float(s["intervention_value_raw"]), abs=1e-5
    )
    qt = int(s["query_target"])
    qti = min(round(float(s["query_time"]) * t_len), t_len - 1)
    expected = float(s["Y_true"]) - float(s["X_obs_full"][qti, qt])
    assert float(s["Y_causal_effect"]) == pytest.approx(expected, abs=1e-5)


def test_counterfactual_mode_is_seed_deterministic():
    a, b = _sample("back_door", "counterfactual"), _sample("back_door", "counterfactual")
    assert torch.equal(a["X_int"], b["X_int"])
    assert torch.equal(a["X_obs_full"], b["X_obs_full"])


def test_interventional_default_keeps_independent_draws():
    # The v1.0.0 semantics must survive unchanged: arms differ before the onset.
    s = _sample("back_door", "interventional")
    n, onset = int(s["num_vars"]), int(s["int_onset_idx"])
    assert not torch.equal(s["X_obs_full"][:onset, :n], s["X_int"][:onset, :n])


def test_pair_mode_validation():
    with pytest.raises(ValueError, match="pair_mode"):
        ExtendedDoTime(tscm_structure="back_door", pair_mode="bogus")
    with pytest.raises(NotImplementedError):
        ExtendedDoTime(tscm_structure=None, pair_mode="counterfactual")
