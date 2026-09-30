"""The observation layer (measurement error and missingness) and its build and evaluation wiring.

An observed episode must keep its latent targets, keep counterfactual arms in
agreement before the onset (missing cells included), never hide a query cell,
and draw its randomness from its own episode-seeded streams. Nothing may change
for the frozen suites or for evaluating a finite suite.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dotime import baselines, evaluation
from dotime._build import (
    _expand_observation_cells,
    _forward_opt_in,
    _simulate_episode,
    episode_seed,
    episode_specs,
    make_episode,
)
from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata
from dotime.interventions import InterventionSpec, InterventionType
from dotime.normalization import per_variable_normalize
from dotime.observation import (
    MeasurementModel,
    MissingnessModel,
    ObservationModel,
    apply_observation,
    cells_from_config,
    impute_episode,
    impute_history,
    require_finite_history,
)
from dotime.reference import reference_table

_ADDED_KEYS = {"observation", "obs_cell", "y_obs_latent", "obs_missing_frac", "latent_row"}
# The observation design of scripts/release_config_observed_v1.yaml.
_DESIGN = {
    "measurement": {"none": {}, "snr10": {"snr": 10}, "snr3": {"snr": 3}},
    "missingness": {
        "none": {"kind": "none"},
        "mcar10": {"kind": "mcar", "rate": 0.10},
        "block": {"kind": "block", "rate": 0.5, "block_len": [10, 40]},
        "mnar": {"kind": "mnar", "rate": 0.8, "mnar_quantile": 0.85},
    },
}


def _model(snr=None, kind="none", rate=0.0, **extra) -> ObservationModel:
    """Build an observation model from a few common settings.

    Args:
        snr: Signal-to-noise ratio, ``None`` for no noise.
        kind: Missingness kind.
        rate: Missingness rate.
        **extra: ``block_len``, ``mnar_quantile``, ``censor_quantiles`` or
            ``quantize_step``.

    Returns:
        The model.
    """
    meas = {k: extra.pop(k) for k in ("censor_quantiles", "quantize_step") if k in extra}
    return ObservationModel(
        MeasurementModel(snr=snr, **meas), MissingnessModel(kind=kind, rate=rate, **extra)
    )


def _synthetic(seed: int = 0, t_len: int = 200, n_vars: int = 5, onset: int = 100) -> Episode:
    """A hand-built counterfactual episode with a hidden (all-zero) column.

    Args:
        seed: Seed of the trajectory draw and the episode id.
        t_len: Number of rows.
        n_vars: Number of columns. Column 1 is hidden, the last is queried.
        onset: Intervention onset and query row.

    Returns:
        An episode whose arms agree before ``onset``.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t_len, n_vars)) + 0.3 * rng.standard_normal((t_len, n_vars)).cumsum(0)
    x[:, 1] = 0.0
    x_int = x.copy()
    x_int[onset:, 2:] += 1.5
    q = n_vars - 1
    y_true = torch.tensor([x_int[onset, q]], dtype=torch.float32)
    return Episode(
        x_obs=torch.tensor(x, dtype=torch.float32),
        x_int=torch.tensor(x_int, dtype=torch.float32),
        intervention=InterventionSpec(
            targets=[0], times=[onset], intervention_type=InterventionType.HARD, values=1.0
        ),
        y_true=y_true,
        query_target=torch.tensor([q]),
        query_time=torch.tensor([onset / t_len]),
        structure="back_door",
        scm_id=seed,
        metadata={
            "query_time_idx": [onset],
            "y_oracle": y_true,
            "y_causal_effect": y_true - float(x[onset, q]),
        },
    )


def _identifiability_spec(structure: str, row: int, t_len: int = 120) -> dict:
    """The build spec of one shared-noise identifiability episode.

    Args:
        structure: Structure name.
        row: Row index, which sets the episode seed.
        t_len: Trajectory length.

    Returns:
        A spec for :func:`dotime._build.make_episode`.
    """
    return {
        "kind": "identifiability",
        "idx": row,
        "seed": episode_seed(20261719, row),
        "T": t_len,
        "structure": structure,
        "tier": 1,
        "pair_mode": "counterfactual",
        "stability_retries": 3,
        "query_offset_range": (0, 0),
    }


def _nan_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors are equal, with NaN equal to NaN.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        ``True`` when the NaN positions and every other value agree.
    """
    return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(
        torch.nan_to_num(a, nan=0.0), torch.nan_to_num(b, nan=0.0)
    )


def _same_episode(a: Episode, b: Episode, *, ignore: set[str] | frozenset = frozenset()) -> bool:
    """Whether two episodes hold identical tensors, intervention and metadata.

    Args:
        a: First episode.
        b: Second episode.
        ignore: Metadata keys to leave out of the comparison.

    Returns:
        ``True`` when everything but the ignored metadata matches exactly.
    """
    tensors = ("x_obs", "x_int", "y_true", "query_target", "query_time")
    if not all(torch.equal(getattr(a, k), getattr(b, k)) for k in tensors):
        return False
    if a.intervention.to_dict() != b.intervention.to_dict() or a.structure != b.structure:
        return False
    keys = (set(a.metadata) | set(b.metadata)) - set(ignore)
    for key in keys:
        va, vb = a.metadata.get(key), b.metadata.get(key)
        if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
            if not torch.equal(torch.as_tensor(va), torch.as_tensor(vb)):
                return False
        elif va != vb:
            return False
    return True


# --------------------------------------------------------------------------- #
# The model and its config
# --------------------------------------------------------------------------- #


def test_model_round_trips_and_config_expands_into_named_cells():
    cells = cells_from_config(_DESIGN)
    names = [c.name for c in cells]
    assert names[:4] == ["none+none", "none+mcar10", "none+block", "none+mnar"]
    assert len(names) == len(set(names)) == 12
    assert cells[0].is_identity
    assert not any(c.is_identity for c in cells[1:])
    for cell in cells:
        assert ObservationModel.from_dict(cell.to_dict()) == cell
    full = _model(
        3, "block", 0.5, block_len=[2, 7], censor_quantiles=[None, 0.9], quantize_step=0.25
    )
    assert ObservationModel.from_dict(full.to_dict()) == full
    assert full.measurement.censor_quantiles == (None, 0.9)
    assert full.missingness.block_len == (2, 7)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: MeasurementModel(snr=0), "snr"),
        (lambda: MeasurementModel(quantize_step=float("inf")), "quantize_step"),
        (lambda: MeasurementModel(censor_quantiles=(0.9, 0.1)), "lo < hi"),
        (lambda: MeasurementModel(censor_quantiles=(0.1, 1.5)), r"\[0, 1\]"),
        (lambda: MissingnessModel(kind="burst"), "kind"),
        (lambda: MissingnessModel(kind="mcar", rate=1.5), "rate"),
        (lambda: MissingnessModel(kind="none", rate=0.1), "rate must be 0"),
        (lambda: MissingnessModel(kind="block", rate=0.5, block_len=(5, 2)), "block_len"),
        (lambda: MissingnessModel(kind="mnar", rate=0.5, mnar_quantile=1.0), "mnar_quantile"),
        (lambda: ObservationModel.from_dict({"noise": {}}), "unknown observation keys"),
        (lambda: ObservationModel.from_dict({"measurement": {"sd": 1}}), "unknown measurement"),
        (lambda: cells_from_config({"noise": {"snr3": {"snr": 3}}}), "unknown observation config"),
        (lambda: cells_from_config({"measurement": {}}), "measurement must map"),
    ],
)
def test_invalid_models_and_configs_are_refused(build, match):
    with pytest.raises(ValueError, match=match):
        build()


# --------------------------------------------------------------------------- #
# apply_observation
# --------------------------------------------------------------------------- #


def test_observation_is_deterministic_and_leaves_global_rngs_alone():
    ep = _synthetic()
    model = _model(3, "mcar", 0.3)
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    first, second = apply_observation(ep, model, 11), apply_observation(ep, model, 11)
    assert torch.equal(torch_state, torch.get_rng_state())
    after = np.random.get_state()
    assert np.array_equal(np_state[1], after[1])
    assert np_state[2:] == after[2:]
    assert _nan_equal(first.x_obs, second.x_obs)
    assert _nan_equal(first.x_int, second.x_int)
    other = apply_observation(ep, model, 12)
    assert not torch.equal(torch.isnan(first.x_obs), torch.isnan(other.x_obs))


def test_cells_of_one_latent_episode_share_their_draws():
    """Common random numbers: the factorial cells differ only in their settings."""
    ep = _synthetic(3)
    live = [0, 2, 3, 4]
    noise = {}
    for snr in (10.0, 3.0):
        obs = apply_observation(ep, _model(snr), 5).x_obs
        noise[snr] = (obs - ep.x_obs)[:, live].double()
    # Differences of float32 observations: exact up to float32 rounding.
    torch.testing.assert_close(noise[3.0], noise[10.0] * (10.0 / 3.0) ** 0.5, atol=1e-5, rtol=1e-5)
    mcar = torch.isnan(apply_observation(ep, _model(None, "mcar", 0.8), 5).x_obs)
    mnar = torch.isnan(apply_observation(ep, _model(None, "mnar", 0.8), 5).x_obs)
    assert bool(mnar.any())
    # MNAR reads MCAR's uniforms, so its gaps are MCAR's gaps at high values
    # (the fallback may keep a different pre-onset cell of a fully masked column).
    assert not bool((mnar[100:] & ~mcar[100:]).any())
    # The noise level does not move the mask.
    for kind in ("mcar", "block", "mnar"):
        masks = [
            torch.isnan(apply_observation(ep, _model(snr, kind, 0.5), 5).x_obs)
            for snr in (None, 10.0, 3.0)
        ]
        assert all(torch.equal(masks[0], m) for m in masks[1:])


def test_identity_model_reproduces_the_latent_tensors():
    ep = _synthetic()
    obs = apply_observation(ep, ObservationModel(name="none+none"), 1)
    assert _same_episode(obs, ep, ignore=_ADDED_KEYS)
    assert obs.metadata["obs_cell"] == "none+none"
    assert obs.metadata["obs_missing_frac"] == 0.0
    assert torch.equal(obs.metadata["y_obs_latent"], ep.x_obs[100, 4].reshape(1))


def test_arms_agree_before_onset_with_the_same_gaps():
    ep = _synthetic(2)
    onset = 100
    for model in (
        _model(3, "mcar", 0.3),
        _model(10, "block", 1.0),
        _model(None, "mnar", 0.9, censor_quantiles=(0.05, 0.95), quantize_step=0.5),
    ):
        obs = apply_observation(ep, model, 4)
        assert bool(torch.isnan(obs.x_obs).any())
        assert _nan_equal(obs.x_obs[:onset], obs.x_int[:onset])


def test_counterfactual_build_episode_keeps_arm_agreement_and_latent_targets():
    ep = make_episode(_identifiability_spec("front_door", 2700))
    onset = min(ep.intervention.times)
    assert torch.equal(ep.x_obs[:onset], ep.x_int[:onset])
    hidden = [j for j in range(ep.n_vars) if float(ep.x_obs[:, j].abs().max()) == 0.0]
    assert hidden, "front_door has a hidden confounder"
    for cell in cells_from_config(_DESIGN):
        obs = apply_observation(ep, cell, 99)
        assert _nan_equal(obs.x_obs[:onset], obs.x_int[:onset])
        assert torch.equal(obs.x_int[onset:], ep.x_int[onset:])
        assert torch.equal(obs.y_true, ep.y_true)
        assert torch.equal(obs.metadata["y_oracle"], ep.metadata["y_oracle"])
        assert torch.equal(obs.metadata["y_causal_effect"], ep.metadata["y_causal_effect"])
        for arm in (obs.x_obs, obs.x_int):
            assert bool((arm[:, hidden] == 0).all())


def test_query_cells_and_one_history_cell_per_column_survive_total_missingness():
    ep = _synthetic(4, onset=30)
    obs = apply_observation(ep, _model(None, "mcar", 1.0), 8)
    rows, cols = ep.query_time_idx, ep.query_target
    assert bool(torch.isfinite(obs.x_obs[rows, cols]).all())
    seen = torch.isfinite(obs.x_obs)
    assert bool(seen[:, 1].all()), "an all-zero column is never masked"
    for col in (0, 2, 3):
        assert int(seen[:30, col].sum()) == 1
        assert int(seen[:, col].sum()) == 1
    assert int(seen[:, 4].sum()) == 2  # its kept history cell and the query cell
    assert obs.metadata["obs_missing_frac"] == pytest.approx(1 - 5 / (4 * 200))


def test_realized_snr_and_mcar_rate_match_the_model():
    ratios, missing, cells = [], 0, 0
    for seed in range(6):
        ep = _synthetic(seed, t_len=300, n_vars=6)
        obs = apply_observation(ep, _model(4.0, "mcar", 0.2), seed)
        latent = ep.x_obs.double()
        diff = obs.x_obs.double() - latent
        for col in (0, 2, 3, 4, 5):
            signal = latent[:100, col].var(unbiased=False)
            seen = torch.isfinite(diff[:, col])
            ratios.append(float((diff[seen, col] ** 2).mean() / signal))
        live = torch.isnan(obs.x_obs[:, [0, 2, 3, 4, 5]])
        missing += int(live.sum())
        cells += live.numel()
    realized_snr = 1.0 / float(np.mean(ratios))
    assert realized_snr == pytest.approx(4.0, rel=0.08)
    assert missing / cells == pytest.approx(0.2, abs=0.02)


def test_block_gaps_are_single_contiguous_runs():
    lengths, with_gap = [], 0
    for seed in range(8):
        ep = _synthetic(seed, onset=120)
        obs = apply_observation(ep, _model(None, "block", 0.5, block_len=[10, 40]), seed)
        for col in (0, 2, 3, 4):
            gap = torch.nonzero(torch.isnan(obs.x_obs[:, col])).reshape(-1).tolist()
            if not gap:
                continue
            with_gap += 1
            span = gap[-1] - gap[0] + 1
            # Only the query cell (row 120 of the last column) may split a gap.
            holes = span - len(gap)
            assert holes == (1 if col == 4 and gap[0] < 120 < gap[-1] else 0)
            # A gap that ends at the query cell loses that one row.
            assert (9 if col == 4 else 10) <= span <= 40
            lengths.append(span)
        assert not bool(torch.isnan(obs.x_obs[:, 1]).any())
    assert 0.3 < with_gap / 32 < 0.7
    assert len(set(lengths)) > 5


def test_mnar_drops_only_high_values():
    dropped_high, high = 0, 0
    for seed in range(5):
        ep = _synthetic(seed)
        obs = apply_observation(ep, _model(None, "mnar", 0.6, mnar_quantile=0.8), seed)
        latent = ep.x_obs.double()
        threshold = torch.quantile(latent[:100], 0.8, dim=0)
        gone = torch.isnan(obs.x_obs)
        assert bool((latent[gone] > threshold.expand_as(latent)[gone]).all())
        is_high = (latent > threshold) & torch.tensor([True, False, True, True, True])
        dropped_high += int((gone & is_high).sum())
        high += int(is_high.sum())
    assert dropped_high / high == pytest.approx(0.6, abs=0.06)


def test_censoring_and_quantization_follow_the_pre_onset_scale():
    ep = _synthetic(6)
    latent = ep.x_obs.double()
    sd = latent[:100].std(dim=0, unbiased=False)
    censored = apply_observation(ep, _model(censor_quantiles=(0.1, 0.9)), 0).x_obs.double()
    for col in (0, 2, 3, 4):
        lo, hi = torch.quantile(latent[:100, col], torch.tensor([0.1, 0.9], dtype=torch.float64))
        expected = latent[:, col].clamp(lo, hi)
        torch.testing.assert_close(censored[:, col], expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(censored[:, 1], latent[:, 1])
    step = 0.5
    quantized = apply_observation(ep, _model(quantize_step=step), 0).x_obs.double()
    for col in (0, 2, 3, 4):
        units = quantized[:, col] / (step * sd[col])
        torch.testing.assert_close(units, units.round(), atol=1e-4, rtol=0)
        assert float((quantized[:, col] - latent[:, col]).abs().max()) <= step * sd[col] / 2 + 1e-5


def test_observing_twice_is_refused():
    obs = apply_observation(_synthetic(), _model(None, "mcar", 0.5), 0)
    with pytest.raises(ValueError, match="not latent"):
        apply_observation(obs, _model(), 0)


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #


def test_impute_history_fills_forward_then_with_the_history_mean():
    nan = float("nan")
    x = torch.tensor(
        [
            [nan, 1.0, nan],
            [2.0, nan, nan],
            [nan, 3.0, nan],
            [4.0, nan, nan],
            [nan, nan, nan],
        ]
    )
    out = impute_history(x, onset=3)
    expected = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
            [4.0, 3.0, 0.0],
        ]
    )
    assert torch.equal(out, expected)
    assert out.dtype == x.dtype
    with pytest.raises(ValueError, match="trajectory"):
        impute_history(x.reshape(-1), onset=3)


def test_imputation_never_reads_the_future():
    obs = apply_observation(_synthetic(7), _model(3, "mcar", 0.4), 7)
    x, onset = obs.x_obs, 100
    base = impute_history(x, onset)
    rng = np.random.default_rng(0)
    for cut in (onset, 130, 170):
        changed = x.clone()
        future = torch.isfinite(changed[cut:])
        changed[cut:][future] = torch.tensor(
            rng.normal(size=int(future.sum())) * 50, dtype=changed.dtype
        )
        changed[cut:][~future] = 7.0
        out = impute_history(changed, onset)
        assert torch.equal(out[:cut], base[:cut]), cut


def test_impute_episode_returns_a_finite_episode_unchanged():
    ep = _synthetic()
    assert impute_episode(ep) is ep
    obs = apply_observation(ep, _model(None, "block", 1.0), 3)
    filled = impute_episode(obs)
    assert bool(torch.isfinite(filled.x_obs).all() and torch.isfinite(filled.x_int).all())
    assert torch.equal(filled.x_obs[:100], filled.x_int[:100])
    assert filled.y_true is obs.y_true
    assert filled.metadata is obs.metadata


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _as_json(obj) -> str:
    """Serialize results so that NaN metrics compare equal.

    Args:
        obj: JSON-serializable results.

    Returns:
        The canonical JSON text.
    """
    return json.dumps(obj, sort_keys=True)


_META = SuiteMetadata(
    name="dot-Identifiability-v1",
    version="1.1.0",
    zenodo_record_id="LOCAL",
    doi="",
    description="test",
    n_episodes=0,
)
_CPU = ["Zero", "Mean", "AR1", "VAR-OLS", "BackDoorOLS", "IV2SLS", "Oracle"]


@pytest.fixture(scope="module")
def latent_episodes() -> list[Episode]:
    """Shared-noise identifiability episodes, two per structure.

    Returns:
        Eight latent episodes.
    """
    structures = ["back_door", "observed_confounder", "confounder_mediator", "front_door"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [
            make_episode(_identifiability_spec(s, 1350 * b + k, t_len=80))
            for b, s in enumerate(structures)
            for k in range(2)
        ]


@pytest.mark.parametrize("dir_target", ["level", "effect"])
def test_finite_suite_scores_are_unchanged(latent_episodes, dir_target, monkeypatch):
    suite = BenchmarkSuite(_META, latent_episodes)
    for name in _CPU:
        model = baselines.get(name)
        default = evaluation.evaluate(model, suite, dir_target=dir_target)
        plain = evaluation.evaluate(model, suite, dir_target=dir_target, impute=False)
        assert _as_json(default.to_dict()) == _as_json(plain.to_dict())
        assert "n_nonfinite" not in default.pooled
        excluded = evaluation.evaluate(model, suite, dir_target=dir_target, nonfinite="exclude")
        assert excluded.pooled.pop("n_nonfinite") == 0
        assert _as_json(excluded.pooled) == _as_json(default.pooled)
    with_imputation = [
        reference_table.run_baseline(name, latent_episodes, dir_target=dir_target) for name in _CPU
    ]
    # run_baseline imports impute_episode when called, so this swaps it out.
    monkeypatch.setattr("dotime.observation.impute_episode", lambda ep: ep)
    without = [
        reference_table.run_baseline(name, latent_episodes, dir_target=dir_target) for name in _CPU
    ]
    assert _as_json(with_imputation) == _as_json(without)


def test_observed_suite_is_imputed_scored_on_latent_targets_and_refuses_nan(latent_episodes):
    model = _model(3, "mcar", 0.3)
    observed = [apply_observation(ep, model, i) for i, ep in enumerate(latent_episodes)]
    suite = BenchmarkSuite(_META, observed)
    for name in _CPU:
        result = evaluation.evaluate(baselines.get(name), suite, dir_target="effect")
        assert all(np.isfinite(result.pooled[k]) for k in ("rmse", "mae"))
    oracle = evaluation.evaluate(baselines.get("Oracle"), suite, dir_target="effect")
    assert oracle.pooled["rmse"] == 0.0
    for obs, ep in zip(observed, latent_episodes, strict=True):
        assert torch.equal(evaluation.query_obs_levels(obs), evaluation.query_obs_levels(ep))
    with pytest.raises(ValueError, match=r"baseline 'Mean' returned 1 non-finite .* episode"):
        evaluation.evaluate(baselines.get("Mean"), suite, impute=False)
    rows = [reference_table.run_baseline(n, observed, dir_target="effect") for n in _CPU]
    assert all(np.isfinite(row["pooled_rmse"]) for row in rows)
    qa = reference_table.target_qa(observed, dir_target="effect")
    assert qa["y_obs_level"]["n"] == len(observed)


class _MaskAwareMean:
    """Mean of the observed pre-onset cells, reading the NaN mask itself."""

    name = "MaskAwareMean"
    mask_aware = True

    def __init__(self) -> None:
        """Start with no recorded inputs."""
        self.saw_nan = False

    def predict(self, episode: Episode) -> torch.Tensor:
        """Predict the queried column's mean over observed pre-onset cells.

        Args:
            episode: The episode, with its missing cells.

        Returns:
            One prediction per query.
        """
        onset = min(episode.intervention.times)
        history = episode.x_obs[:onset, episode.query_target]
        self.saw_nan |= bool(torch.isnan(history).any())
        return torch.nanmean(history, dim=0)


class _Abstainer:
    """Returns NaN for every other episode."""

    name = "Abstainer"

    def __init__(self) -> None:
        """Start counting calls."""
        self.calls = 0

    def predict(self, episode: Episode) -> torch.Tensor:
        """Predict the target, or NaN on every second call.

        Args:
            episode: The episode.

        Returns:
            One prediction per query.
        """
        self.calls += 1
        return episode.y_true.clone() if self.calls % 2 else torch.full_like(episode.y_true, np.nan)


def test_mask_aware_models_see_the_gaps(latent_episodes):
    observed = [apply_observation(ep, _model(None, "mcar", 0.5), 1) for ep in latent_episodes]
    model = _MaskAwareMean()
    result = evaluation.evaluate(model, BenchmarkSuite(_META, observed))
    assert model.saw_nan
    assert np.isfinite(result.pooled["rmse"])


def test_excluded_nonfinite_predictions_count_as_wrong_directions(latent_episodes):
    suite = BenchmarkSuite(_META, latent_episodes)
    with pytest.raises(ValueError, match="'Abstainer' returned 1 non-finite"):
        evaluation.evaluate(_Abstainer(), suite)
    result = evaluation.evaluate(_Abstainer(), suite, nonfinite="exclude")
    n = len(latent_episodes)
    assert result.pooled["n_nonfinite"] == n // 2
    assert result.pooled["rmse"] == 0.0
    targets = torch.cat([ep.y_true for ep in latent_episodes])
    scored = int((targets.abs() >= evaluation.DIR_ACC_EPS).sum())
    right = int((targets[0::2].abs() >= evaluation.DIR_ACC_EPS).sum())
    assert result.pooled["dir_n_valid"] == scored
    assert result.pooled["dir_acc"] == pytest.approx(right / scored)
    assert sum(g["n_nonfinite"] for g in result.per_structure.values()) == n // 2
    assert "Non-finite predictions: 4" in result.summary()
    with pytest.raises(ValueError, match="nonfinite"):
        evaluation.evaluate(_Abstainer(), suite, nonfinite="drop")


def test_direction_accuracy_counts_sign_zero_and_nan_as_wrong():
    preds = torch.tensor([0.0, float("nan"), 1.0, -1.0, 5.0])
    targets = torch.tensor([1.0, -1.0, 1.0, 1.0, 0.01])
    da = evaluation.direction_accuracy(preds, targets)
    assert (da["n_valid"], da["n_excluded"]) == (4, 1)
    assert da["accuracy"] == pytest.approx(0.25)


def test_reference_input_builders_ask_for_imputation():
    from dotime.reference import chronos, pfn, tabpfn

    obs = apply_observation(_synthetic(), _model(None, "mcar", 0.5), 0)
    for build in (
        lambda: pfn.episode_to_batch_interp(obs, 41, "cpu"),
        lambda: tabpfn._series(obs),
        lambda: chronos._episode_frames(obs, True),
        lambda: require_finite_history(obs, "test"),
    ):
        with pytest.raises(ValueError, match="impute first"):
            build()
    require_finite_history(impute_episode(obs), "test")


def test_normalization_obs_mask_hook():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(2.0, 3.0, size=(2, 50, 4)), dtype=torch.float32)
    var_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    onset = torch.tensor([30, 40])
    reference = per_variable_normalize(x, var_mask, int_onset_idx=onset)
    ones = per_variable_normalize(x, var_mask, int_onset_idx=onset, obs_mask=torch.ones_like(x))
    for a, b in zip(reference, ones, strict=True):
        assert torch.equal(a, b)
    observed = torch.tensor(rng.random(size=x.shape) > 0.3)
    gappy = torch.where(observed, x, torch.full_like(x, float("nan")))
    x_norm, means, _ = per_variable_normalize(
        gappy, var_mask, int_onset_idx=onset, obs_mask=observed.float()
    )
    assert bool(torch.isfinite(x_norm).all())
    assert bool((x_norm[~observed] == 0).all())
    history = observed[0, :30, 0]
    torch.testing.assert_close(means[0, 0], x[0, :30, 0][history].mean())


# --------------------------------------------------------------------------- #
# Build wiring
# --------------------------------------------------------------------------- #

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _observed_config(**overrides) -> dict:
    """The dot-Observed-v1 suite config, with overrides.

    Args:
        **overrides: Config keys to replace.

    Returns:
        The suite config.
    """
    rc = yaml.safe_load((_SCRIPTS / "release_config_observed_v1.yaml").read_text())
    return {**rc["suites"]["dot-Observed-v1"], **overrides}


@pytest.mark.parametrize(
    ("config", "scale"),
    [
        ("release_config.yaml", 0.0005),
        ("release_config_v1_1.yaml", 0.002),
        ("release_config_v1_2.yaml", 0.002),
    ],
)
def test_frozen_configs_keep_their_specs(config, scale):
    rc = yaml.safe_load((_SCRIPTS / config).read_text())
    for offset, cfg in enumerate(rc["suites"].values()):
        seed = int(rc["seed"]) + 1000 * (offset + 1)
        specs = episode_specs(cfg, seed, scale)
        # `pair_mode` is an opt-in key that the identifiability specs already carry,
        # so forwarding it yields equal copies rather than the same list object.
        assert _forward_opt_in(cfg, specs) == specs
        assert _expand_observation_cells(cfg, specs) is specs
        assert not any(key in spec for spec in specs for key in _ADDED_KEYS)
        assert [s["idx"] for s in specs] == list(range(len(specs)))
        assert [s["seed"] for s in specs] == [episode_seed(seed, i) for i in range(len(specs))]


@pytest.mark.parametrize(
    "spec",
    [
        {"kind": "generic", "idx": 3, "seed": episode_seed(11, 3), "T": 40},
        {"kind": "regime", "idx": 0, "seed": 5, "T": 40, "num_regimes": 2, "tier": 1},
        _identifiability_spec("mediator", 4050, t_len=40),
        {"kind": "continuous", "idx": 1, "seed": 9, "T": 40, "structure": "back_door"},
    ],
)
def test_make_episode_without_observation_is_the_simulation(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert _same_episode(make_episode(spec), _simulate_episode(spec))


def test_observed_suite_specs_keep_the_base_seeds():
    cfg = _observed_config()
    rc = yaml.safe_load((_SCRIPTS / "release_config_observed_v1.yaml").read_text())
    assert list(rc["suites"]) == ["dot-Observed-v1"]
    assert cfg["observation"] == {"latent_per_structure": 100, **_DESIGN}
    assert cfg["seed"] == int(rc["seed"]) + 1000 == 20261719
    v12 = yaml.safe_load((_SCRIPTS / "release_config_v1_2.yaml").read_text())
    assert list(cfg["structures"].items()) == list(
        v12["suites"]["dot-Identifiability-v1"]["structures"].items()
    )
    base = {k: v for k, v in cfg.items() if k not in ("version", "seed", "observation")}
    assert base == {
        k: v for k, v in v12["suites"]["dot-Identifiability-v1"].items() if k != "version"
    }

    specs = episode_specs(cfg, cfg["seed"], 1.0)
    base_specs = episode_specs(base, cfg["seed"], 1.0)
    assert len(specs) == 10_800
    assert [s["idx"] for s in specs] == list(range(10_800))
    cells = [c.name for c in cells_from_config(cfg["observation"])]
    assert [specs[900 * i]["obs_cell"] for i in range(12)] == cells
    latent_rows = [b * 1350 + k for b in range(9) for k in range(100)]
    for i, name in enumerate(cells):
        block = specs[900 * i : 900 * (i + 1)]
        assert {s["obs_cell"] for s in block} == {name}
        assert [s["latent_row"] for s in block] == latent_rows
        assert all(s["observation"]["name"] == name for s in block)
    for spec in specs[::97]:
        latent = base_specs[spec["latent_row"]]
        assert spec["seed"] == latent["seed"] == episode_seed(cfg["seed"], spec["latent_row"])
        assert {k: v for k, v in spec.items() if k not in _ADDED_KEYS | {"idx"}} == {
            k: v for k, v in latent.items() if k != "idx"
        }
    assert {s["query_offset_range"] for s in specs if s["structure"] == "mediator"} == {(1, 1)}


def test_none_cell_equals_the_base_suite_rows_and_other_cells_observe_them():
    cfg = _observed_config(T=60, episodes_per_structure=3)
    cfg["observation"] = {**cfg["observation"], "latent_per_structure": 1}
    base = {k: v for k, v in cfg.items() if k != "observation"}
    specs = episode_specs(cfg, cfg["seed"], 1.0)
    base_specs = episode_specs(base, cfg["seed"], 1.0)
    assert len(specs) == 12 * 9
    for spec in specs[:9]:
        assert spec["obs_cell"] == "none+none"
        observed, latent = make_episode(spec), make_episode(base_specs[spec["latent_row"]])
        assert _same_episode(observed, latent, ignore=_ADDED_KEYS)
        assert observed.scm_id == spec["idx"]
        assert latent.scm_id == spec["latent_row"]
        assert observed.metadata["latent_row"] == spec["latent_row"]
        assert observed.metadata["obs_cell"] == "none+none"
    spec = specs[-1]
    observed, latent = make_episode(spec), make_episode(base_specs[spec["latent_row"]])
    assert observed.metadata["obs_cell"] == "snr3+mnar"
    assert torch.equal(observed.y_true, latent.y_true)
    assert torch.equal(observed.metadata["y_obs_latent"], evaluation.query_obs_levels(latent))
    assert not torch.equal(observed.x_obs.nan_to_num(0.0), latent.x_obs)


def test_latent_per_structure_must_be_a_positive_integer():
    cfg = _observed_config(T=40, episodes_per_structure=2)
    for bad in (0, 1.5, "10"):
        cfg["observation"] = {**cfg["observation"], "latent_per_structure": bad}
        with pytest.raises(ValueError, match="latent_per_structure"):
            episode_specs(cfg, 1, 1.0)


@pytest.mark.slow
def test_micro_build_round_trips_missing_cells(tmp_path):
    """A small dot-Observed-v1 build writes NaN cells and reads them back exactly."""
    pytest.importorskip("pyarrow")
    import importlib.util
    import json

    from dotime import _release_io

    spec = importlib.util.spec_from_file_location("build_release", _SCRIPTS / "build_release.py")
    build_release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_release)
    config = _SCRIPTS / "release_config_observed_v1.yaml"
    args = ["--config", str(config), "--scale", "0.0015", "--workers", "2"]
    assert build_release.main([*args, "--output-dir", str(tmp_path), "--timestamp", "T"]) == 0
    suite_dir = tmp_path / "T" / "dot-Observed-v1-1.0.0"
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    assert manifest["n_episodes"] == 12 * 8 * 2
    meta = SuiteMetadata(
        name="dot-Observed-v1",
        version="1.0.0",
        zenodo_record_id="LOCAL",
        doi="",
        description="",
        n_episodes=manifest["n_episodes"],
        query_time_encoding="index/T",
    )
    suite = _release_io.read_suite(meta, suite_dir)
    cfg = _observed_config()
    specs = episode_specs(cfg, cfg["seed"], 0.0015)
    for spec in specs[::41]:
        built = make_episode(spec)
        loaded = suite[spec["idx"]]
        assert _nan_equal(loaded.x_obs, built.x_obs)
        assert _nan_equal(loaded.x_int, built.x_int)
        assert loaded.metadata["obs_cell"] == spec["obs_cell"]
        assert loaded.metadata["latent_row"] == spec["latent_row"]
    assert any(bool(torch.isnan(ep.x_obs).any()) for ep in suite)
    oracle = evaluation.evaluate(baselines.get("Oracle"), suite, dir_target="effect")
    assert oracle.pooled["rmse"] == pytest.approx(0.0, abs=1e-6)
    mean = evaluation.evaluate(baselines.get("Mean"), suite, dir_target="effect")
    assert np.isfinite(mean.pooled["rmse"])
