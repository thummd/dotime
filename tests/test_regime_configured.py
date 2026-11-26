"""Tests for the configured regime build path behind ``dot-RegimeSwitch-v1`` 1.1.0.

A ``generator: regime`` suite that sets ``prior_config`` or ``query_row`` is
built by the ``regime_configured`` spec kind. It is the only regime path that
reads a prior config, and the only one that seeds numpy's global generator,
which draws the regime path of each arm. A suite that sets neither option keeps
the specs and the episodes of the released version 1.0.0, so every option here
is opt-in.
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
    assert episode_specs({**_SMALL, "pair_mode": "interventional"}, 11, 1.0)[0]["kind"] == "regime"


@pytest.mark.parametrize(
    "bad",
    [
        {"prior_config": {"N_maxx": 5}},
        {"prior_config": [("N_max", 5)]},
        {"query_row": "last"},
        {"pair_mode": "counterfactual"},
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
def test_configured_build_is_independent_of_the_worker_count():
    """One worker and two workers build the same repaired suite."""
    cfg = {**_SMALL, "prior_config": _REPAIRED, "query_row": "window_end"}
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
    """Version 1.1.0 is version 1.0.0's recipe with the repair switched on."""
    config = yaml.safe_load(_CONFIG.read_text())
    assert list(config["suites"]) == [_SUITE]
    cfg = config["suites"][_SUITE]
    released = yaml.safe_load(_V1_CONFIG.read_text())
    old = released["suites"][_SUITE]
    assert (old["version"], cfg["version"]) == ("1.0.0", "1.1.0")
    # build_release.py seeds the n-th suite of a config with base + 1000 n.
    position = list(released["suites"]).index(_SUITE) + 1
    assert cfg["seed"] == released["seed"] + 1000 * position == _SUITE_SEED
    for key in ("generator", "T", "densities", "n_episodes"):
        assert cfg[key] == old[key]
    assert cfg["prior_config"] == _REPAIRED
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
    assert (summary["n_episodes"], summary["n_contrast"]) == (600, 600)
    assert summary["n_diverged"] == 0
    assert summary["n_reads_parents"] == 600
    median = summary["steps_after_window_median"]
    assert (
        f"In a pilot of 600 episodes the window closes a median {median:.1f} steps before T-1"
        in header
    )
    at_last = 100 * pooled["query_at_last_step_share"]
    at_end = 100 * pooled["query_at_window_end_share"]
    assert (
        f"least 0.1 in {at_last:.1f}% of the episodes at T-1 and in {at_end:.1f}% at the window's end"
        in header
    )
    most = 100 * pooled["query_is_most_affected_share"]
    assert f"most affected variable in {most:.1f}% of the episodes" in header
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
