"""Tests for the configured regime build path behind ``dot-RegimeSwitch-v1`` 1.1.0.

A ``generator: regime`` suite that sets ``prior_config``, ``query_row`` or
``pair_mode: counterfactual`` is built by the ``regime_configured`` spec kind.
It is the only regime path that reads a prior config, the only one that seeds
numpy's global generator, which draws the regime path of each arm of an
interventional pair, and the only one that can share noise across the arms. A
suite that sets none of these keeps the specs and the episodes of the released
version 1.0.0, so every option here is opt-in.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dotime import DoTime
from dotime._build import build_suite, episode_seed, episode_specs, make_episode
from dotime.evaluation import check_shared_noise
from dotime.hardening import RECOMMENDED_HARDENING

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "scripts" / "release_config_regime_v1_1.yaml"
_V1_CONFIG = _ROOT / "scripts" / "release_config.yaml"
_REFERENCE = _ROOT / "results" / "reference"
_SUITE = "dot-RegimeSwitch-v1"
_SUITE_SEED = 20262719

# The repair and the stabilisation that the repaired dynamics need.
_REPAIRED = {"regime_canonical_weights": True, "hardening": RECOMMENDED_HARDENING}
# Twelve short episodes, four per regime density.
_SMALL = {"generator": "regime", "T": 80, "densities": {2: 1, 3: 2, 5: 3}, "n_episodes": 12}


def _build(cfg: dict) -> list:
    """Build every episode of a small suite config in this process.

    Args:
        cfg: A suite config with ``generator: regime``.

    Returns:
        The episodes in suite order.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [make_episode(spec) for spec in episode_specs(cfg, _SUITE_SEED, 1.0)]


def _abs_acf1(x: torch.Tensor) -> torch.Tensor:
    """Per-variable ``|corr(x[t], x[t-1])|`` of a ``(T, N)`` trajectory.

    Args:
        x: The trajectory.

    Returns:
        Tensor of shape ``(N,)``.
    """
    x = x - x.mean(0)
    return ((x[1:] * x[:-1]).sum(0) / (x * x).sum(0)).abs()


def test_legacy_regime_specs_are_unchanged():
    """Without an opt-in key the specs are those of the released version 1.0.0."""
    cfg = {"generator": "regime", "T": 60, "densities": {2: 1, 3: 2}, "n_episodes": 4}
    for retries in (0, 4):
        specs = episode_specs({**cfg, "stability_retries": retries}, 11, 1.0)
        assert specs == [
            {
                "kind": "regime",
                "idx": i,
                "seed": episode_seed(11, i),
                "T": 60,
                "num_regimes": density,
                "tier": tier,
                "stability_retries": retries,
            }
            for i, (density, tier) in enumerate([(2, 1), (2, 1), (3, 2), (3, 2)])
        ]
        keys = ["kind", "idx", "seed", "T", "num_regimes", "tier", "stability_retries"]
        assert all(list(s) == keys for s in specs)


def test_configured_regime_specs_forward_the_options():
    """Either opt-in key moves the suite to the configured kind and reaches every spec."""
    cfg = {**_SMALL, "prior_config": _REPAIRED, "query_row": "window_end", "record_graph": True}
    specs = episode_specs(cfg, 11, 1.0)
    assert {s["kind"] for s in specs} == {"regime_configured"}
    for key in ("prior_config", "query_row", "record_graph"):
        assert all(s[key] == cfg[key] for s in specs)
    assert [s["seed"] for s in specs] == [episode_seed(11, i) for i in range(12)]
    assert [s["num_regimes"] for s in specs] == [2] * 4 + [3] * 4 + [5] * 4
    only_query = episode_specs({**_SMALL, "query_row": "window_end"}, 11, 1.0)
    assert only_query[0]["kind"] == "regime_configured"
    assert "prior_config" not in only_query[0]
    # The graph record alone does not reconfigure the prior.
    assert episode_specs({**_SMALL, "record_graph": True}, 11, 1.0)[0]["kind"] == "regime"
    # An explicit interventional pair mode asks for what the legacy branch does.
    assert episode_specs({**_SMALL, "pair_mode": "interventional"}, 11, 1.0)[0]["kind"] == "regime"
    shared = episode_specs({**_SMALL, "pair_mode": "counterfactual"}, 11, 1.0)
    assert {s["kind"] for s in shared} == {"regime_configured"}
    assert all(s["pair_mode"] == "counterfactual" for s in shared)
    limited = episode_specs({**_SMALL, "amplitude_limit": 50}, 11, 1.0)
    assert {s["kind"] for s in limited} == {"regime_configured"}
    assert all(s["amplitude_limit"] == 50 for s in limited)
    # Only the regime builder acts on the limit, so another generator must not carry it.
    with pytest.raises(ValueError, match="amplitude_limit"):
        episode_specs({"generator": "generic", "n_episodes": 2, "amplitude_limit": 50}, 11, 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"prior_config": {"N_maxx": 5}},
        {"prior_config": [("N_max", 5)]},
        {"query_row": "last"},
        {"pair_mode": "shared"},
        {"amplitude_limit": 0},
        {"amplitude_limit": True},
        {"amplitude_limit": "50"},
        {"chain_prob": 0.0},
        {"regime_switching_prob": 1.0},
        {"latent": "drop"},
        {"tier_n_edges": [5]},
    ],
)
def test_regime_specs_reject_bad_options(bad):
    """An option the regime builder would ignore, or cannot honour, fails before any draw."""
    with pytest.raises(ValueError):
        episode_specs({**_SMALL, **bad}, 11, 1.0)


def test_configured_build_seeds_the_regime_paths():
    """A repaired episode does not depend on the state of numpy's global generator.

    The regime path of each arm comes from that generator. The control at the
    end shows that the path does move a repaired trajectory, so the equality
    holds because the builder seeds numpy, not because numpy is unused.
    """
    spec = episode_specs({**_SMALL, "prior_config": _REPAIRED}, _SUITE_SEED, 1.0)[3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.random.seed(1)
        first = make_episode(spec)
        np.random.seed(2)
        np.random.rand(5)
        second = make_episode(spec)
        assert not first.metadata["diverged"]
        assert torch.equal(first.x_obs, second.x_obs)
        assert torch.equal(first.x_int, second.x_int)
        drawn = []
        for numpy_seed in (1, 2):
            np.random.seed(numpy_seed)
            torch.manual_seed(spec["seed"])
            prior = DoTime(config=spec["prior_config"], seed=spec["seed"])
            drawn.append(prior.generate_regime_pair(T=spec["T"], num_regimes=2)[0])
    assert not torch.equal(drawn[0], drawn[1])


@pytest.mark.slow
@pytest.mark.parametrize("pair_mode", ["interventional", "counterfactual"])
def test_configured_build_is_independent_of_the_worker_count(pair_mode):
    """One worker and two workers build the same repaired suite, in either pair mode."""
    cfg = {
        **_SMALL,
        "prior_config": _REPAIRED,
        "query_row": "window_end",
        "pair_mode": pair_mode,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        serial = build_suite(cfg, _SUITE_SEED, 1.0, 1)
        pooled = build_suite(cfg, _SUITE_SEED, 1.0, 2)
    assert len(serial) == len(pooled) == 12
    for a, b in zip(serial, pooled, strict=True):
        assert torch.equal(a.x_obs, b.x_obs)
        assert torch.equal(a.x_int, b.x_int)
        assert torch.equal(a.y_true, b.y_true)


def test_repaired_episodes_have_memory_and_no_zeroed_arm():
    """Version 1.0.0 trajectories are white noise. Repaired ones carry lag structure."""
    legacy = _build({**_SMALL, "T": 160})
    repaired = _build(
        {**_SMALL, "T": 160, "stability_retries": 5, "prior_config": {**_REPAIRED, "N_max": 5}}
    )
    assert not any(ep.metadata["diverged"] for ep in repaired)
    assert all(float(ep.x_obs.abs().max()) > 0 for ep in repaired)
    assert all(float(ep.x_int.abs().max()) > 0 for ep in repaired)
    off = torch.cat([_abs_acf1(ep.x_obs) for ep in legacy if float(ep.x_obs.abs().max()) > 0])
    on = torch.cat([_abs_acf1(ep.x_obs) for ep in repaired])
    # White noise at T=160 has sd(acf1) of about 0.08, so 0.3 is close to 4 sd.
    assert float(off.max()) < 0.3
    assert float((on > 0.3).float().mean()) > 0.15


def test_window_end_query_on_a_regime_episode():
    """``query_row: window_end`` queries the last step of the intervention window."""
    cfg = {**_SMALL, "prior_config": _REPAIRED, "query_row": "window_end", "stability_retries": 5}
    for spec in episode_specs(cfg, _SUITE_SEED, 1.0)[::3]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ep = make_episode(spec)
            plain = make_episode({k: v for k, v in spec.items() if k != "query_row"})
        end = max(ep.intervention.times)
        assert ep.structure == f"regime_{spec['num_regimes']}"
        assert ep.metadata["n_regimes"] == spec["num_regimes"]
        assert ep.metadata["tier"] == spec["tier"]
        assert ep.metadata["query_row"] == "window_end"
        assert ep.metadata["query_time_idx"] == [end]
        assert float(ep.query_time[0]) == float(end)
        q = int(ep.query_target[0])
        assert q not in ep.intervention.targets
        assert torch.equal(ep.y_true, ep.x_int[end, q].reshape(1))
        # Without the key the same trajectories are queried at the last step.
        assert torch.equal(plain.x_int, ep.x_int)
        assert plain.metadata["query_time_idx"] == [spec["T"] - 1]
        assert "query_row" not in plain.metadata


def test_counterfactual_regime_episodes_share_noise_and_regime_path():
    """``pair_mode: counterfactual`` gives arms that differ only through the intervention."""
    cfg = {
        **_SMALL,
        "stability_retries": 5,
        "prior_config": {**_REPAIRED, "N_max": 5},
        "pair_mode": "counterfactual",
        "query_row": "window_end",
        "record_graph": True,
    }
    shared = _build(cfg)
    twins = _build({k: v for k, v in cfg.items() if k != "pair_mode"})
    check = check_shared_noise(shared)
    assert (check.n_checked, check.n_shared) == (12, 12)
    assert check_shared_noise(twins).n_shared == 0
    for ep, twin in zip(shared, twins, strict=True):
        assert ep.metadata["pair_mode"] == "counterfactual"
        assert twin.metadata["pair_mode"] == "interventional"
        assert not ep.metadata["diverged"]
        # Both modes draw the same regime graphs and the same intervention.
        assert ep.metadata["graph"]["regime_edges"] == twin.metadata["graph"]["regime_edges"]
        assert ep.intervention.times == twin.intervention.times
        # The shared path is recorded, and only where one exists.
        path = ep.metadata["graph"]["regime_path"]
        assert len(path) == cfg["T"]
        assert set(path) <= set(range(ep.metadata["n_regimes"]))
        assert "regime_path" not in twin.metadata["graph"]
        # With shared noise the gap between the arms is the effect, so the
        # query is the most affected untreated variable at the window's end.
        end = max(ep.intervention.times)
        effect = (ep.x_int[end] - ep.x_obs[end]).abs()
        effect[list(ep.intervention.targets)] = -1.0
        assert int(ep.query_target[0]) == int(effect.argmax())
        assert ep.metadata["query_time_idx"] == [end]
    # numpy's global generator, which draws the paths of the twins, plays no part here.
    spec = episode_specs(cfg, _SUITE_SEED, 1.0)[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.random.seed(123)
        again = make_episode(spec)
    assert torch.equal(again.x_obs, shared[2].x_obs)
    assert torch.equal(again.x_int, shared[2].x_int)


def test_amplitude_limit_resamples_large_episodes():
    """An episode with a value above ``amplitude_limit`` is resampled like a diverged one."""

    def peak(ep) -> float:
        """Largest magnitude in either arm of an episode.

        Args:
            ep: The episode.

        Returns:
            The peak absolute value.
        """
        return max(float(ep.x_obs.abs().max()), float(ep.x_int.abs().max()))

    cfg = {
        **_SMALL,
        "T": 160,
        "prior_config": {**_REPAIRED, "N_max": 5},
        "pair_mode": "counterfactual",
    }
    free = _build(cfg)
    peaks = [peak(ep) for ep in free]
    # The median of the first attempts, so that half of them exceed the limit.
    limit = sorted(peaks)[len(peaks) // 2]
    flagged = _build({**cfg, "amplitude_limit": limit})
    resampled = _build({**cfg, "amplitude_limit": limit, "stability_retries": 40})
    assert sum(p > limit for p in peaks) >= 4
    assert "amplitude_limit" not in free[0].metadata
    for first, kept, redrawn in zip(free, flagged, resampled, strict=True):
        assert kept.metadata["amplitude_limit"] == redrawn.metadata["amplitude_limit"] == limit
        # Without retries the first attempt stays and is flagged.
        assert torch.equal(first.x_int, kept.x_int)
        assert kept.metadata["over_amplitude_limit"] is (peak(first) > limit)
        # With retries no episode exceeds the limit, and the others keep their first attempt.
        assert peak(redrawn) <= limit
        assert redrawn.metadata["over_amplitude_limit"] is False
        assert torch.equal(first.x_int, redrawn.x_int) is (peak(first) <= limit)


def test_retries_resample_zeroed_episodes():
    """Without the stabilisation many repaired episodes diverge, and retries resample them."""
    unstable = {**_SMALL, "prior_config": {"N_max": 5, "regime_canonical_weights": True}}
    single = _build({**unstable, "stability_retries": 0})
    retried = _build({**unstable, "stability_retries": 40})
    assert sum(ep.metadata["diverged"] for ep in single) >= 3
    assert not any(ep.metadata["diverged"] for ep in retried)
    # An episode that did not diverge keeps its first attempt.
    for a, b in zip(single, retried, strict=True):
        if not a.metadata["diverged"]:
            assert torch.equal(a.x_int, b.x_int)


def test_repaired_episode_keeps_the_graph_and_intervention_of_the_legacy_draw():
    """The repair and the stabilisation draw no random numbers.

    A repaired episode therefore has the regime graphs and the intervention of
    the version 1.0.0 episode with the same seed. Only its mechanisms differ:
    they read their parents.
    """
    legacy = _build({**_SMALL, "record_graph": True})
    repaired = _build({**_SMALL, "record_graph": True, "prior_config": _REPAIRED})
    for old, new in zip(legacy, repaired, strict=True):
        assert old.x_obs.shape == new.x_obs.shape
        assert json.dumps(old.intervention.to_dict(), sort_keys=True) == json.dumps(
            new.intervention.to_dict(), sort_keys=True
        )
        assert old.metadata["graph"]["regime_edges"] == new.metadata["graph"]["regime_edges"]
        assert old.metadata["graph"]["reads_parents"] is False
        assert old.metadata["graph"]["edges"] == []
        assert new.metadata["graph"]["reads_parents"] is True
        assert new.metadata["graph"]["edges"]


def test_regime_v1_1_release_config():
    """Version 1.1.0 is version 1.0.0's recipe with the repair and shared noise."""
    config = yaml.safe_load(_CONFIG.read_text())
    assert list(config["suites"]) == [_SUITE]
    cfg = config["suites"][_SUITE]
    released = yaml.safe_load(_V1_CONFIG.read_text())
    old = released["suites"][_SUITE]
    assert (old["version"], cfg["version"]) == ("1.0.0", "1.1.0")
    # build_release.py seeds the n-th suite of a config with base + 1000 n.
    position = list(released["suites"]).index(_SUITE) + 1
    assert cfg["seed"] == released["seed"] + 1000 * position == _SUITE_SEED
    # The explicit seed also equals the derived one, as the fingerprint tests
    # expect of every release config.
    assert cfg["seed"] == config["seed"] + 1000
    for key in ("generator", "T", "densities", "n_episodes"):
        assert cfg[key] == old[key]
    assert cfg["prior_config"] == _REPAIRED
    assert cfg["pair_mode"] == "counterfactual"
    assert cfg["amplitude_limit"] == 50
    assert cfg["query_row"] == "window_end"
    assert cfg["record_graph"] is True
    assert cfg["stability_retries"] > 0
    new_specs = episode_specs(cfg, cfg["seed"], 1.0)
    old_specs = episode_specs(old, cfg["seed"], 1.0)
    assert len(new_specs) == len(old_specs) == 9999
    assert {s["kind"] for s in old_specs} == {"regime"}
    assert {s["kind"] for s in new_specs} == {"regime_configured"}
    block = lambda specs: [(s["seed"], s["num_regimes"], s["tier"]) for s in specs]  # noqa: E731
    assert block(new_specs) == block(old_specs)


def test_release_config_header_quotes_its_measurements():
    """Every number in the config's header comment is the number of an artifact.

    The header is hashed into the manifest of the released suite, so it cannot
    be corrected later. The design pilot must also belong to this config.
    """
    text = _CONFIG.read_text()
    header = re.sub(
        r"\s+",
        " ",
        " ".join(line.lstrip("# ") for line in text.splitlines() if line.startswith("#")),
    )
    cfg = yaml.safe_load(text)["suites"][_SUITE]
    pilot = json.loads((_REFERENCE / "regime_v1_1" / "design_pilot.json").read_text())
    keys_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    assert pilot["config_keys_sha256"] == keys_hash
    assert pilot["src_dirty"] is False
    summary, pooled = pilot["summary"], pilot["summary"]["pooled"]
    assert summary["n_episodes"] == 600
    assert summary["n_diverged"] == 0
    assert summary["n_reads_parents"] == 600
    assert summary["n_pre_onset_equal"] == 600
    median = summary["steps_after_window_median"]
    at_last = 100 * pooled["effect_at_last_step_share"]
    at_end = 100 * pooled["effect_at_window_end_share"]
    assert (
        f"In a pilot of 600 episodes the window closes a median {median:.1f} steps before T-1, "
        f"and {at_last:.1f}% of the counterfactual effects reach 0.1 in size at T-1, against "
        f"{at_end:.1f}% at the window's end" in header
    )
    weights = json.loads(
        (_REFERENCE / "regime_weights" / "regime_weights_measurement.json").read_text()
    )
    group = weights["configs"][0]["groups"]["regime_all"]
    assert group["on"]["n_episodes"] == group["on_hardened"]["n_episodes"] == 600
    zeroed = {k: round(group[k]["diverged_frac"] * 600) for k in ("on", "on_hardened")}
    assert (
        f"{zeroed['on']} of 600 measured episodes diverge without it and "
        f"{zeroed['on_hardened']} of 600 with it" in header
    )
    # The reason for the pairing: on the build with independent arms the mean was unbeaten.
    assert (
        "With independent arms no reference estimator beat the pre-onset mean on a build of "
        "this recipe" in header
    )
    rows = json.loads(
        (_REFERENCE / "regime_v1_1" / "independent_arms" / "regime_v1_1_cpu.json").read_text()
    )
    assert rows["pairs_share_noise"] is False
    rmse = {
        row["baseline"]: row["pooled_rmse"] for row in rows["rows"] if row["baseline"] != "Oracle"
    }
    assert min(rmse.values()) == rmse["Mean"]
    assert all(rmse[name] > rmse["Mean"] for name in ("AR1", "VAR-OLS", "NaiveOLS", "Zero"))
    # The amplitude limit and the tail of the build without it.
    tail = json.loads((_REFERENCE / "regime_v1_1" / "amplitude_tail.json").read_text())
    assert tail["build"]["amplitude_limit"] is None
    assert tail["build"]["pair_mode"] == "counterfactual"
    stats, levels = tail["summary"], tail["summary"]["quantile_levels"]
    peak_max = stats["peak_quantiles"][levels.index(1.0)]
    target_max = stats["target_abs_quantiles"][levels.index(1.0)]
    target_999 = stats["target_abs_quantiles"][levels.index(0.999)]
    assert cfg["amplitude_limit"] == 50
    assert (
        f"In a build without the limit {stats['peak_above']['50']} of {stats['n_episodes']:,} "
        f"episodes exceed 50, up to {peak_max:.1f}, and the largest target is {target_max:.1f}, "
        f"while 99.9% of the targets stay below {target_999:.1f}" in header
    )
