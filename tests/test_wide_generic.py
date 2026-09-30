"""Tests for the generic-prior plumbing behind the wide suite ``dot-Wide-v1``.

Covers the prior's ``N_min`` and ``pair_mode="counterfactual"`` options, the
``generic_configured`` spec kind with its latent-variable drop, the release
config, and the per-suite seeding of ``scripts/build_release.py``. Every option
is opt-in, so the default paths must draw exactly the numbers they drew before.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from dotime import DoTime
from dotime._build import (
    _drop_latent_columns,
    arms_zeroed,
    episode_seed,
    episode_specs,
    make_episode,
)
from dotime.hardening import RECOMMENDED_HARDENING
from dotime.interventions import InterventionSpec, InterventionType
from dotime.regime_switching import RegimeSwitchingTemporalSCM

_ROOT = Path(__file__).resolve().parents[1]
_WIDE_CONFIG = _ROOT / "scripts" / "release_config_wide.yaml"

# A small configured suite: N in [6, 10] keeps each episode fast, and the
# recommended hardening keeps the arms from diverging.
_SMALL_CONFIGURED = {
    "generator": "generic",
    "n_episodes": 6,
    "T": 40,
    "stability_retries": 5,
    "pair_mode": "counterfactual",
    "chain_prob": 0.0,
    "regime_switching_prob": 0.0,
    "prior_config": {"N_min": 6, "N_max": 10, "K_max": 2, "hardening": RECOMMENDED_HARDENING},
    "latent": "drop",
    "tier_n_edges": [7, 9],
}


def _pair(seed: int, config=None, pair_mode: str = "interventional", **kwargs):
    """Draw one pair the way the release build does, with the global RNGs seeded.

    Regime-switching SCMs draw their noise from the global numpy RNG, so it is
    seeded too, which makes two calls with one seed comparable.

    Args:
        seed: Seed of the global torch and numpy RNGs and of the prior.
        config: The prior's config, or ``None`` for the defaults.
        pair_mode: Passed to :meth:`DoTime.generate_pair`.
        **kwargs: Further :class:`DoTime` arguments.

    Returns:
        ``(x_obs, x_int, intervention, scm, prior, global_torch_state)``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior = DoTime(config=config, seed=seed, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_obs, x_int, iv, scm = prior.generate_pair(T=40, pair_mode=pair_mode)
    return x_obs, x_int, iv, scm, prior, torch.get_rng_state()


def _load_build_release():
    """Import ``scripts/build_release.py`` as a module.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(
        "build_release", _ROOT / "scripts" / "build_release.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scm_class(scm) -> str:
    """Name the prior branch that sampled an SCM.

    Args:
        scm: An SCM returned by :meth:`DoTime.generate_pair`.

    Returns:
        ``"regime"``, ``"chain"`` (nodes ``X0, X1, ...`` in chain order) or
        ``"diverse"`` (nodes ``x{i}``, ``u{i}`` and ``y``).
    """
    if isinstance(scm, RegimeSwitchingTemporalSCM):
        return "regime"
    return "chain" if scm._topo[0] == "X0" else "diverse"


def test_n_min_three_draws_exactly_the_default_stream():
    # Seed 0 samples a diverse SCM, 3 a chain and 15, 16 and 31 regime-switching
    # SCMs, which covers every call site of the N draw.
    classes = set()
    for seed in (0, 3, 15, 16, 31):
        a = _pair(seed, {"N_max": 6})
        b = _pair(seed, {"N_max": 6, "N_min": 3})
        classes.add(_scm_class(a[3]))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
        assert a[2].to_dict() == b[2].to_dict()
        assert torch.equal(a[4].generator.get_state(), b[4].generator.get_state())
        assert torch.equal(a[5], b[5])
    assert classes == {"diverse", "chain", "regime"}
    runs = []
    for config in (None, {"N_min": 3}):
        np.random.seed(5)
        torch.manual_seed(5)
        prior = DoTime(config=config, seed=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_obs, x_int, iv, _ = prior.generate_regime_pair(T=40, num_regimes=2)
        runs.append((x_obs, x_int, iv.to_dict(), prior.generator.get_state()))
    assert torch.equal(runs[0][0], runs[1][0])
    assert torch.equal(runs[0][1], runs[1][1])
    assert runs[0][2] == runs[1][2]
    assert torch.equal(runs[0][3], runs[1][3])


@pytest.mark.parametrize("regime_switching_prob", [0.0, 1.0])
def test_n_min_bounds_the_graph_size(regime_switching_prob):
    config = {"N_min": 12, "N_max": 20, "K_max": 2}
    sizes = []
    for seed in range(10):
        torch.manual_seed(seed)
        prior = DoTime(
            config=config,
            seed=seed,
            chain_prob=0.0,
            regime_switching_prob=regime_switching_prob,
        )
        sizes.append(len(prior.sample_scm()._topo))
    assert all(12 <= n <= 20 for n in sizes), sizes
    assert len(set(sizes)) > 1, sizes
    torch.manual_seed(0)
    prior = DoTime(config={**config, "N_max": 13}, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_obs, _, _, _ = prior.generate_regime_pair(T=20, num_regimes=2)
    assert x_obs.shape[1] in (12, 13)


def test_n_min_validation():
    state = torch.get_rng_state()
    for bad in ("12", 12.0, True):
        with pytest.raises(TypeError, match="N_min"):
            DoTime(config={"N_min": bad})
    for bad in (2, 11):
        with pytest.raises(ValueError, match="N_min"):
            DoTime(config={"N_min": bad, "N_max": 10})
    assert torch.equal(torch.get_rng_state(), state)
    assert DoTime(config={"N_min": 10, "N_max": 10}).n_min == 10


def test_counterfactual_arms_agree_before_the_onset():
    config = {"N_max": 12, "K_max": 3, "hardening": RECOMMENDED_HARDENING}
    checked = 0
    for seed in range(8):
        x_obs, x_int, iv, *_ = _pair(
            seed, config, "counterfactual", chain_prob=0.3, regime_switching_prob=0.0
        )
        if arms_zeroed(x_obs, x_int):
            continue
        onset = min(iv.times)
        assert onset > 0
        assert torch.equal(x_obs[:onset], x_int[:onset])
        assert not torch.equal(x_obs, x_int)
        checked += 1
    assert checked >= 6


def test_counterfactual_mode_draws_the_interventional_scm_and_intervention():
    config = {"N_max": 12, "K_max": 3}
    for seed in range(6):
        a = _pair(seed, config, "interventional", chain_prob=0.3, regime_switching_prob=0.0)
        b = _pair(seed, config, "counterfactual", chain_prob=0.3, regime_switching_prob=0.0)
        assert a[2].to_dict() == b[2].to_dict()
        assert list(a[3]._topo) == list(b[3]._topo)
        assert torch.equal(a[4].generator.get_state(), b[4].generator.get_state())


def test_counterfactual_mode_is_deterministic():
    config = {"N_max": 12, "K_max": 3, "hardening": RECOMMENDED_HARDENING}
    for seed in range(3):
        a = _pair(seed, config, "counterfactual", regime_switching_prob=0.0)
        b = _pair(seed, config, "counterfactual", regime_switching_prob=0.0)
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])


def test_pair_mode_validation_happens_before_any_draw():
    prior = DoTime(seed=0)
    state = prior.generator.get_state()
    with pytest.raises(ValueError, match="pair_mode"):
        prior.generate_pair(T=40, pair_mode="bogus")
    # The default regime_switching_prob draws regime-switching SCMs.
    with pytest.raises(ValueError, match="regime_switching_prob"):
        prior.generate_pair(T=40, pair_mode="counterfactual")
    with pytest.raises(ValueError, match="regime_switching_prob"):
        DoTime(seed=0, regime_switching_prob=0.01).generate_pair(pair_mode="counterfactual")
    assert torch.equal(prior.generator.get_state(), state)


def test_legacy_generic_specs_are_unchanged():
    for cfg in (
        {"generator": "generic", "n_episodes": 3, "T": 60},
        {"generator": "generic", "n_episodes": 3, "T": 60, "stability_retries": 4},
    ):
        specs = episode_specs(cfg, 11, 1.0)
        assert specs == [
            {
                "kind": "generic",
                "idx": i,
                "seed": episode_seed(11, i),
                "T": 60,
                "stability_retries": cfg.get("stability_retries", 0),
            }
            for i in range(3)
        ]
        assert all(list(s) == ["kind", "idx", "seed", "T", "stability_retries"] for s in specs)


def test_configured_generic_specs_forward_only_the_keys_set():
    specs = episode_specs(_SMALL_CONFIGURED, 11, 1.0)
    assert {s["kind"] for s in specs} == {"generic_configured"}
    for key in ("prior_config", "pair_mode", "latent", "tier_n_edges", "chain_prob"):
        assert all(s[key] == _SMALL_CONFIGURED[key] for s in specs)
    assert [s["seed"] for s in specs] == [episode_seed(11, i) for i in range(6)]
    one = episode_specs({"generator": "generic", "n_episodes": 1, "latent": "drop"}, 11, 1.0)
    assert one[0]["kind"] == "generic_configured"
    assert "prior_config" not in one[0]
    assert "pair_mode" not in one[0]


@pytest.mark.parametrize(
    "bad",
    [
        {"prior_config": {"N_maxx": 12}},
        {"prior_config": [("N_max", 12)]},
        {"latent": "keep"},
        {"tier_n_edges": [30, 20]},
        {"tier_n_edges": [20, 20]},
        {"tier_n_edges": ["20"]},
        {"tier_n_edges": 20},
        {"pair_mode": "bogus"},
        {"pair_mode": "counterfactual"},
        {"pair_mode": "counterfactual", "regime_switching_prob": 0.1},
        {"chain_prob": 1.5},
        {"chain_prob": True},
    ],
)
def test_configured_generic_specs_reject_bad_options(bad):
    with pytest.raises(ValueError):
        episode_specs({"generator": "generic", "n_episodes": 2, **bad}, 11, 1.0)


def test_drop_latent_columns_keeps_intervened_latents_and_remaps_targets():
    x = torch.arange(12.0).reshape(3, 4)
    names = ["x0", "u1", "y", "u3"]
    iv = InterventionSpec(
        targets=[3, 0], times=[1], intervention_type=InterventionType.HARD, values=1.0
    )
    x_obs, x_int, remapped, latent = _drop_latent_columns(x, x + 1, iv, names)
    assert torch.equal(x_obs, x[:, [0, 2, 3]])
    assert torch.equal(x_int, x[:, [0, 2, 3]] + 1)
    assert remapped.targets == [2, 0]
    assert iv.targets == [3, 0]  # the caller's spec is not modified
    assert latent == {
        "mode": "drop",
        "columns": ["x0", "y", "u3"],
        "hidden": ["u1"],
        "n_vars_full": 4,
    }


def _full_pair(spec: dict):
    """Regenerate the pair of a configured spec before any column is dropped.

    Follows the documented retry rule of the configured generic branch.

    Args:
        spec: A ``generic_configured`` spec.

    Returns:
        ``(x_obs, x_int, intervention, scm)`` of the attempt that was kept.
    """
    for attempt in range(spec["stability_retries"] + 1):
        s = spec["seed"] if attempt == 0 else spec["seed"] * 100003 + attempt
        torch.manual_seed(s)
        prior = DoTime(
            config=spec["prior_config"],
            seed=s,
            chain_prob=spec["chain_prob"],
            regime_switching_prob=spec["regime_switching_prob"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = prior.generate_pair(T=spec["T"], pair_mode=spec["pair_mode"])
        if not arms_zeroed(out[0], out[1]):
            break
    return out


def test_latent_drop_releases_only_observed_columns():
    dropped = 0
    for spec in episode_specs(_SMALL_CONFIGURED, 11, 1.0):
        ep = make_episode(spec)
        x_obs, x_int, iv, scm = _full_pair(spec)
        names = list(scm._topo)
        latent = ep.metadata["latent"]
        keep = [names.index(c) for c in latent["columns"]]
        targets = [names[t] for t in iv.targets]
        assert latent["hidden"] == [
            n for i, n in enumerate(names) if n.startswith("u") and i not in iv.targets
        ]
        assert latent["n_vars_full"] == len(names) == x_obs.shape[1]
        assert keep == sorted(keep)
        assert torch.equal(ep.x_obs, x_obs[:, keep])
        assert torch.equal(ep.x_int, x_int[:, keep])
        assert [latent["columns"][t] for t in ep.intervention.targets] == targets
        # The query is the most affected released non-target column at the last row.
        row, col = ep.metadata["query_time_idx"][0], int(ep.query_target[0])
        assert row == ep.x_int.shape[0] - 1
        effect = (ep.x_int[row] - ep.x_obs[row]).abs()
        effect[ep.intervention.targets] = -1.0
        assert col == int(torch.argmax(effect))
        assert float(ep.y_true[0]) == float(ep.x_int[row, col])
        edges = _SMALL_CONFIGURED["tier_n_edges"]
        assert ep.metadata["tier"] == 1 + sum(e < len(names) for e in edges)
        assert ep.metadata["diverged"] is False
        assert ep.metadata["pair_mode"] == "counterfactual"
        meta = {k: v for k, v in ep.metadata.items() if k != "y_oracle"}
        assert json.loads(json.dumps(meta)) == meta
        dropped += len(latent["hidden"])
    assert dropped > 0


def test_configured_episode_without_latent_keeps_every_column():
    cfg = {**_SMALL_CONFIGURED, "n_episodes": 2}
    del cfg["latent"], cfg["tier_n_edges"]
    for spec in episode_specs(cfg, 11, 1.0):
        ep = make_episode(spec)
        x_obs, x_int, _, _ = _full_pair(spec)
        assert torch.equal(ep.x_obs, x_obs)
        assert torch.equal(ep.x_int, x_int)
        assert "latent" not in ep.metadata
        assert ep.metadata["tier"] == 1


def test_wide_release_config():
    config = yaml.safe_load(_WIDE_CONFIG.read_text())
    assert list(config["suites"]) == ["dot-Wide-v1"]
    cfg = config["suites"]["dot-Wide-v1"]
    assert cfg["seed"] == config["seed"] + 1000
    assert cfg["prior_config"]["hardening"] == RECOMMENDED_HARDENING
    assert cfg["prior_config"]["N_min"] == 12
    assert cfg["prior_config"]["N_max"] <= 40  # PFN checkpoints pad to 41 variables
    specs = episode_specs(cfg, cfg["seed"], 0.0001)
    assert specs[0]["kind"] == "generic_configured"


def test_build_release_honours_explicit_and_positional_suite_seeds(tmp_path):
    pytest.importorskip("pyarrow", reason="build_release writes parquet (evaluation extra)")
    tiny = {"generator": "generic", "T": 40, "n_episodes": 2}
    config = {
        "seed": 500,
        "suites": {
            "tiny-a": {"version": "1.0.0", **tiny},
            "tiny-b": {"version": "1.0.0", **tiny},
            "tiny-c": {
                "version": "1.0.0",
                "seed": 777,
                **{**_SMALL_CONFIGURED, "T": 40, "n_episodes": 2},
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    br = _load_build_release()
    expected = {"tiny-a": 1500, "tiny-b": 2500, "tiny-c": 777}
    runs = {"all": None, "tiny-b": "tiny-b", "tiny-c": "tiny-c"}
    for stamp, suite in runs.items():
        argv = ["--config", str(path), "--output-dir", str(tmp_path), "--timestamp", stamp]
        argv += ["--workers", "1"] + (["--suite", suite] if suite else [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert br.main(argv) == 0
        built = json.loads((tmp_path / stamp / "build_manifest.json").read_text())["suites"]
        assert {s["name"]: s["seed"] for s in built} == {
            s["name"]: expected[s["name"]] for s in built
        }
    manifest = json.loads((tmp_path / "tiny-b" / "tiny-b-1.0.0" / "manifest.json").read_text())
    assert manifest["seed"] == 2500
    assert "prior_config" not in manifest
    assert "latent" not in manifest
    manifest = json.loads((tmp_path / "tiny-c" / "tiny-c-1.0.0" / "manifest.json").read_text())
    assert manifest["seed"] == 777
    for key in ("prior_config", "chain_prob", "regime_switching_prob", "latent", "tier_n_edges"):
        assert manifest[key] == _SMALL_CONFIGURED[key]
    assert manifest["pair_mode"] == "counterfactual"


def test_query_row_window_end_queries_the_windows_last_step():
    """``query_row: window_end`` queries the intervention window's last step."""
    from dotime._build import episode_specs, make_episode

    cfg = yaml.safe_load(_WIDE_CONFIG.read_text())["suites"]["dot-Wide-v1"]
    assert cfg["query_row"] == "window_end"
    small = {**cfg, "T": 60, "n_episodes": 3}
    for spec in episode_specs(small, 20262002, 1.0):
        assert spec["query_row"] == "window_end"
        ep = make_episode(spec)
        end = max(ep.intervention.times)
        assert ep.metadata["query_time_idx"] == [end]
        assert float(ep.query_time[0]) == float(end)
        assert ep.metadata["query_row"] == "window_end"
        q = int(ep.query_target[0])
        assert q not in ep.intervention.targets
        assert torch.equal(ep.y_true, ep.x_int[end, q].reshape(1))
        # Without the key the query returns to the last step.
        plain = make_episode({k: v for k, v in spec.items() if k != "query_row"})
        assert plain.metadata["query_time_idx"] == [small["T"] - 1]
        assert "query_row" not in plain.metadata


def test_query_row_rejects_unknown_rules():
    """Only the window_end rule exists."""
    from dotime._build import episode_specs

    cfg = yaml.safe_load(_WIDE_CONFIG.read_text())["suites"]["dot-Wide-v1"]
    with pytest.raises(ValueError, match="query_row"):
        episode_specs({**cfg, "T": 40, "n_episodes": 1, "query_row": "last"}, 1, 1.0)
