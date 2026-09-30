"""Irregular observation grids for continuous suites (opt-in ``schedules``).

Released configs set no opt-in key, so their specs and episodes must stay
exactly as they were. An irregular episode must keep the SCM and intervention
draws of the regular episode with the same seed, record its grid, and resolve
every query to the row the generator queried. The references are the
generator's own outputs and the released dot-Continuous-v1 files, never the
code under test.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from dotime._build import _forward_opt_in, episode_seed, episode_specs, make_episode
from dotime._observation_grids import check_schedule, draw_grid, max_substep, schedule_for
from dotime.benchmarks import (
    QUERY_TIME_ENCODINGS,
    SuiteMetadata,
    episode_from_sample,
    query_time_to_index,
)
from dotime.continuous import ContinuousExtendedPrior
from dotime.evaluation import query_obs_levels

_ROOT = Path(__file__).resolve().parents[1]
_STRUCTURES = ["back_door", "front_door", "instrumental_variable"]
_JITTERED = {"name": "jittered", "kind": "jittered", "dt": 1.0, "jitter": 0.5, "num_substeps": 2}
_POISSON = {"name": "poisson", "kind": "poisson", "rate": 1.0, "max_gap": 4.0, "num_substeps": 4}
_SCHEDULES = [{"name": "regular", "kind": "regular"}, _JITTERED, _POISSON]
_FROZEN_KEYS = {"kind", "idx", "seed", "T", "structure"}


def _cfg(n: int = 12, t_len: int = 60, **extra) -> dict:
    return {
        "generator": "continuous",
        "T": t_len,
        "structures": _STRUCTURES,
        "n_episodes": n,
        **extra,
    }


def _build(specs) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return [make_episode(sp) for sp in specs]


def _yaml(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((_ROOT / "scripts" / name).read_text())


# --------------------------------------------------------------------------- #
# Frozen specs and the regular path
# --------------------------------------------------------------------------- #


def test_frozen_continuous_specs_are_unchanged():
    cfg = _yaml("release_config.yaml")["suites"]["dot-Continuous-v1"]
    specs = episode_specs(cfg, 20263719, 1.0)
    assert len(specs) == 9999
    assert all(set(sp) == _FROZEN_KEYS for sp in specs)
    assert [sp["seed"] for sp in specs] == [episode_seed(20263719, i) for i in range(9999)]
    assert [sp["structure"] for sp in specs] == [s for s in _STRUCTURES for _ in range(3333)]
    # Without opt-in keys the seam hands back the very same list.
    assert _forward_opt_in(cfg, specs) is specs

    # The frozen call is still the plain prior call, and records nothing new.
    for sp in (specs[0], specs[3333], specs[6666]):
        (ep,) = _build([{**sp, "T": 60}])
        torch.manual_seed(sp["seed"])  # as make_episode does
        s = ContinuousExtendedPrior(
            tscm_structure=sp["structure"], seed=sp["seed"]
        ).generate_sample(T=60)
        ref = episode_from_sample(s, structure=sp["structure"], scm_id=sp["idx"])
        assert torch.equal(ep.x_obs, ref.x_obs)
        assert torch.equal(ep.x_int, ref.x_int)
        assert torch.equal(ep.y_true, ref.y_true)
        assert "schedule" not in ep.metadata
        assert ep.obs_times is None


def test_opt_in_keys_reach_every_spec_and_nothing_else():
    plain = episode_specs(_cfg(), 7, 1.0)
    opted = episode_specs(_cfg(schedules=_SCHEDULES, record_obs_times=True), 7, 1.0)
    assert len(opted) == len(plain)
    for a, b in zip(plain, opted, strict=True):
        assert b == {**a, "schedules": _SCHEDULES, "record_obs_times": True}


def test_schedules_are_balanced_by_episode_index():
    specs = episode_specs(_cfg(n=30, t_len=40, schedules=_SCHEDULES), 11, 1.0)
    eps = _build(specs)
    names = [ep.metadata["schedule"] for ep in eps]
    assert names == [_SCHEDULES[sp["idx"] % 3]["name"] for sp in specs]
    for structure in _STRUCTURES:
        counts = Counter(n for n, ep in zip(names, eps, strict=True) if ep.structure == structure)
        assert sorted(counts) == ["jittered", "poisson", "regular"]
        assert max(counts.values()) - min(counts.values()) <= 1
    # record_obs_times is off here: grids are used but not stored.
    assert all(ep.obs_times is None for ep in eps)


# --------------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("entry", [_JITTERED, _POISSON], ids=["jittered", "poisson"])
def test_grids_are_strictly_increasing_with_bounded_substeps(entry):
    lo, hi = (0.5, 1.5) if entry["kind"] == "jittered" else (1e-3, 4.0)
    all_gaps = []
    for seed in range(300):
        times = draw_grid(entry, 200, seed)
        assert times.dtype == torch.float32
        assert times.shape == (200,)
        assert float(times[0]) == 0.0
        gaps = (times[1:] - times[:-1]).double()
        assert bool((gaps > 0).all())
        # float32 times near t = 256 are spaced 3e-5 apart.
        assert float(gaps.min()) >= lo - 1e-4
        assert float(gaps.max()) <= hi + 1e-4
        assert float(gaps.max()) / entry["num_substeps"] <= 1.0 + 1e-4
        all_gaps.append(gaps)
    assert max_substep(entry) <= 1.0
    gaps = torch.cat(all_gaps)
    if entry["kind"] == "poisson":
        # Exp(1) truncated to [0.001, 4]: mean 0.926, and the cap leaves no atom.
        assert float(gaps.mean()) == pytest.approx(0.926, abs=0.01)
        assert float((gaps > 3.99).double().mean()) < 1e-3
    else:
        assert float(gaps.mean()) == pytest.approx(1.0, abs=0.01)
    # A pure function of the entry and the episode seed.
    assert torch.equal(draw_grid(entry, 200, 5), draw_grid(entry, 200, 5))
    assert not torch.equal(draw_grid(entry, 200, 5), draw_grid(entry, 200, 6))


def test_episodes_record_their_grid():
    specs = episode_specs(_cfg(n=9, t_len=50, schedules=_SCHEDULES, record_obs_times=True), 3, 1.0)
    for sp, ep in zip(specs, _build(specs), strict=True):
        assert ep.obs_times.dtype == torch.float64
        assert bool((ep.obs_times[1:] > ep.obs_times[:-1]).all())
        entry = _SCHEDULES[sp["idx"] % 3]
        if entry["kind"] == "regular":
            assert torch.equal(ep.obs_times, torch.arange(50, dtype=torch.float64))
        else:
            assert torch.equal(ep.obs_times, draw_grid(entry, 50, sp["seed"]).double())


# --------------------------------------------------------------------------- #
# Same SCM draws as the regular episode
# --------------------------------------------------------------------------- #


def test_irregular_episodes_keep_the_regular_episodes_scm_draws(monkeypatch):
    drawn = []
    original = ContinuousExtendedPrior._sample_scm_context

    def spy(self):
        ctx = original(self)
        drawn.append(
            [
                (m.theta, m.sigma, m.parent_weights.clone(), tuple(m.parents))
                for m in ctx.scm.mechanisms
            ]
        )
        return ctx

    monkeypatch.setattr(ContinuousExtendedPrior, "_sample_scm_context", spy)
    base = episode_specs(_cfg(n=6, t_len=80), 20263719, 1.0)
    for sp in base:
        eps = _build([{**sp, "schedules": [entry]} for entry in _SCHEDULES])
        regular, *irregular = drawn[-3:]
        for scm in irregular:
            assert [m[:2] for m in scm] == [m[:2] for m in regular]
            assert [m[3] for m in scm] == [m[3] for m in regular]
            assert all(torch.equal(a[2], b[2]) for a, b in zip(scm, regular, strict=True))
        # The window, kind and value draws come from the prior's numpy
        # generator in the same order, so the intervention is the same too.
        assert len({ep.intervention.values for ep in eps}) == 1
        assert len({tuple(ep.intervention.targets) for ep in eps}) == 1
        assert [ep.metadata["schedule"] for ep in eps] == ["regular", "jittered", "poisson"]
        assert not torch.equal(eps[0].x_obs, eps[1].x_obs)


# --------------------------------------------------------------------------- #
# Query rows
# --------------------------------------------------------------------------- #


def _generator_sample(sp: dict, entry: dict) -> dict:
    """The prior's own sample for an irregular spec, built independently."""
    times = draw_grid(entry, sp["T"], sp["seed"])
    torch.manual_seed(sp["seed"])
    return ContinuousExtendedPrior(
        tscm_structure=sp["structure"],
        seed=sp["seed"],
        schedule="fixed",
        fixed_times=times,
        dt=float(times[-1] - times[0]) / (sp["T"] - 1),
        num_substeps=entry["num_substeps"],
    ).generate_sample(T=sp["T"])


@pytest.mark.parametrize("entry", [_JITTERED, _POISSON], ids=["jittered", "poisson"])
def test_irregular_query_rows_are_exact(entry, tmp_path):
    pytest.importorskip("pyarrow")
    from dotime import _release_io

    specs = episode_specs(_cfg(n=12, t_len=80, schedules=[entry], record_obs_times=True), 29, 1.0)
    eps = _build(specs)
    off_grid = 0
    for sp, ep in zip(specs, eps, strict=True):
        s = _generator_sample(sp, entry)
        rows = ep.query_time_idx.tolist()
        assert torch.equal(ep.obs_times, s["times"].double())
        assert ep.obs_times[rows[0]] == s["t_query"].double()
        assert torch.equal(query_obs_levels(ep), s["Y_obs"].reshape(-1))
        assert torch.equal(ep.x_int[rows, ep.query_target], ep.y_true)
        resolved = query_time_to_index(ep.query_time, ep.length, "time/span", times=ep.obs_times)
        assert resolved == rows
        # No fraction of T resolves an irregular grid.
        pos = float(ep.query_time[0]) * (ep.length - 1)
        off_grid += abs(pos - round(pos)) > 1e-3
    assert off_grid > 0

    # Files without recorded rows resolve them from the declared encoding and
    # the stored grid; a wrong declaration fails loudly.
    for ep in eps:
        del ep.metadata["query_time_idx"]
    meta = SuiteMetadata(
        name="irr",
        version="1.0.0",
        zenodo_record_id="LOCAL",
        doi="",
        description="",
        n_episodes=len(eps),
        query_time_encoding="time/span",
    )
    suite_dir = _release_io.write_suite(meta, eps, tmp_path / "irr", package_version="t", seed=0)
    for sp, got in zip(specs, _release_io.read_suite(meta, suite_dir), strict=True):
        s = _generator_sample(sp, entry)
        assert torch.equal(query_obs_levels(got), s["Y_obs"].reshape(-1))
        assert torch.equal(got.obs_times, s["times"].double())
    wrong = dataclasses.replace(meta, query_time_encoding="index/(T-1)")
    with pytest.raises(ValueError, match="not a whole row"):
        _release_io.read_suite(wrong, suite_dir)


def test_time_span_encoding():
    assert QUERY_TIME_ENCODINGS[-1] == "time/span"
    times = torch.tensor([0.0, 0.5, 2.0, 2.25, 4.0], dtype=torch.float64)
    frac = ((times - times[0]) / 4.0).float()
    assert query_time_to_index(frac, 5, "time/span", times=times) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="needs the episode's observation times"):
        query_time_to_index(frac, 5, "time/span")
    with pytest.raises(ValueError, match="not an observation time"):
        query_time_to_index([0.3], 5, "time/span", times=times)
    with pytest.raises(ValueError, match="observation times for T=4"):
        query_time_to_index(frac, 4, "time/span", times=times)
    # On a regular grid it is index / (T - 1).
    regular = torch.arange(200, dtype=torch.float64)
    qt = torch.tensor([r / 199 for r in (0, 1, 150, 199)], dtype=torch.float32)
    assert query_time_to_index(qt, 200, "time/span", times=regular) == [0, 1, 150, 199]
    assert query_time_to_index(qt, 200, "index/(T-1)") == [0, 1, 150, 199]
    # ...and it is strict about index / T fractions, even near the start.
    with pytest.raises(ValueError, match="not an observation time"):
        query_time_to_index([1 / 200], 200, "time/span", times=regular)


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #


def test_fixed_schedule_contract():
    times = torch.tensor([0.0, 0.7, 1.9, 2.0, 3.5])
    with pytest.raises(ValueError, match="fixed_times"):
        ContinuousExtendedPrior(schedule="fixed")
    with pytest.raises(ValueError, match="fixed_times"):
        ContinuousExtendedPrior(fixed_times=times)
    with pytest.raises(ValueError, match="strictly increasing"):
        ContinuousExtendedPrior(schedule="fixed", fixed_times=times.flip(0))

    prior = ContinuousExtendedPrior(schedule="fixed", fixed_times=times.double(), seed=1)
    state = prior._np_rng.get_state()[1].copy()
    assert prior.sample_T() == 5
    assert np.array_equal(prior._np_rng.get_state()[1], state)  # no draw
    with pytest.raises(ValueError, match="T=6"):
        prior.generate_sample(T=6)
    s = prior.generate_sample()
    assert torch.equal(s["times"], times)
    s["times"][0] = -1.0
    assert torch.equal(prior.generate_sample()["times"], times)


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"name": "x", "kind": "exponential", "rate": 1.0}, "kind"),
        ({"kind": "regular"}, "no name"),
        ({"name": "r", "kind": "regular", "dt": 1.0}, "takes exactly"),
        ({"name": "j", "kind": "jittered", "dt": 1.0, "jitter": 0.5}, "takes exactly"),
        ({**_JITTERED, "jitter": 1.0}, "jitter"),
        ({**_JITTERED, "dt": 0.0}, "dt must be positive"),
        ({**_JITTERED, "num_substeps": 1}, "sub-steps up to 1.5"),
        ({**_JITTERED, "num_substeps": 2.0}, "positive int"),
        ({**_POISSON, "num_substeps": 3}, "sub-steps up to 1.33"),
        ({**_POISSON, "max_gap": 0.0005}, "max_gap"),
        ({**_POISSON, "rate": -1.0}, "rate must be positive"),
    ],
)
def test_invalid_schedules_are_refused(entry, match):
    with pytest.raises(ValueError, match=match):
        check_schedule(entry)


def test_schedule_lists_are_validated_whole():
    assert schedule_for(None, 5) is None
    assert schedule_for(_SCHEDULES, 8) is _POISSON
    with pytest.raises(ValueError, match="at least one"):
        schedule_for([], 0)
    with pytest.raises(ValueError, match="unique"):
        schedule_for([_JITTERED, _JITTERED], 0)
    # A bad entry fails every episode, not only the ones assigned to it.
    with pytest.raises(ValueError, match="sub-steps"):
        schedule_for([{"name": "regular", "kind": "regular"}, {**_POISSON, "num_substeps": 1}], 0)
    with pytest.raises(ValueError, match="draws no grid"):
        draw_grid(_SCHEDULES[0], 10, 0)


# --------------------------------------------------------------------------- #
# The prepared release config
# --------------------------------------------------------------------------- #


def test_irregular_release_config():
    config = _yaml("release_config_continuous_irregular.yaml")
    ((name, cfg),) = config["suites"].items()
    assert name == "dot-ContinuousIrregular-v1"
    assert cfg["version"] == "1.0.0"
    # One suite, seeded explicitly: top-level + 1000, which is also what
    # build_release.py derives for the first suite, and dot-Continuous-v1's seed.
    assert cfg["seed"] == config["seed"] + 1000 == 20263719
    assert cfg["generator"] == "continuous"
    assert cfg["T"] == 200
    assert cfg["structures"] == _STRUCTURES
    assert cfg["n_episodes"] == 10000
    assert cfg["record_obs_times"] is True
    assert [e["kind"] for e in cfg["schedules"]] == ["regular", "jittered", "poisson"]
    for entry in cfg["schedules"]:
        check_schedule(entry)
    specs = episode_specs(cfg, cfg["seed"], 1.0)
    assert len(specs) == 9999
    frozen = episode_specs(_yaml("release_config.yaml")["suites"]["dot-Continuous-v1"], 20263719, 1)
    assert [{k: sp[k] for k in _FROZEN_KEYS} for sp in specs] == frozen


def test_regular_third_reproduces_the_released_continuous_suite():
    """Regular episodes of the irregular suite are the dot-Continuous-v1 rows."""
    pq = pytest.importorskip("pyarrow.parquet")
    cache = Path.home() / ".cache" / "dotime" / "dot-Continuous-v1-1.0.0"
    if not (cache / "shard-0001.parquet").exists():
        pytest.skip("dot-Continuous-v1 1.0.0 is not cached")
    cfg = _yaml("release_config_continuous_irregular.yaml")["suites"]["dot-ContinuousIrregular-v1"]
    specs = episode_specs(cfg, cfg["seed"], 1.0)
    picks = [0, 3, 2997, 3333, 4998, 5001, 6666, 9996]  # every structure, both shards
    eps = _build([specs[i] for i in picks])
    shards = [pq.read_table(cache / f"shard-000{k}.parquet") for k in (0, 1)]
    for i, ep in zip(picks, eps, strict=True):
        assert ep.metadata["schedule"] == "regular"
        row = shards[i // 5000].slice(i % 5000, 1).to_pylist()[0]
        assert row["scm_id"] == i
        length, n_vars = row["length"], row["n_vars"]
        assert torch.equal(ep.x_obs, torch.tensor(row["x_obs"]).float().reshape(length, n_vars))
        assert torch.equal(ep.x_int, torch.tensor(row["x_int"]).float().reshape(length, n_vars))
        assert torch.equal(ep.y_true, torch.tensor(row["y_true"]).float())
        assert torch.equal(ep.query_target, torch.tensor(row["query_target"]))
        assert torch.equal(ep.query_time, torch.tensor(row["query_time"]).float())
        assert torch.equal(ep.obs_times, torch.arange(200, dtype=torch.float64))


def test_micro_build_of_the_irregular_release_config(tmp_path):
    """build_release.py writes the prepared config as a loadable schema-2 suite."""
    import importlib.util
    import json

    pytest.importorskip("pyarrow")
    pytest.importorskip("yaml")
    from dotime import _release_io, baselines, evaluation

    script = _ROOT / "scripts" / "build_release.py"
    spec = importlib.util.spec_from_file_location("build_release", script)
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)
    config = _ROOT / "scripts" / "release_config_continuous_irregular.yaml"
    argv = ["--config", str(config), "--scale", "0.001", "--output-dir", str(tmp_path)]
    assert br.main([*argv, "--timestamp", "T", "--workers", "1"]) == 0

    suite_dir = tmp_path / "T" / "dot-ContinuousIrregular-v1-1.0.0"
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "2"
    assert manifest["seed"] == 20263719
    croissant = json.loads((suite_dir / "croissant.json").read_text())
    ids = [f["@id"] for f in croissant["cr:recordSet"]["field"]]
    assert ids[-2:] == ["obs_times", "obs_mask"]

    meta = SuiteMetadata(
        name=manifest["name"],
        version=manifest["version"],
        zenodo_record_id="LOCAL",
        doi="",
        description="",
        n_episodes=manifest["n_episodes"],
        query_time_encoding="time/span",
    )
    suite = _release_io.read_suite(meta, suite_dir)
    assert len(suite) == 9
    assert [ep.metadata["schedule"] for ep in suite] == ["regular", "jittered", "poisson"] * 3
    for ep in suite:
        assert ep.obs_mask is None
        assert bool((ep.obs_times[1:] > ep.obs_times[:-1]).all())
        rows = query_time_to_index(ep.query_time, ep.length, "time/span", times=ep.obs_times)
        assert rows == ep.query_time_idx.tolist()
    results = evaluation.evaluate(baselines.get("Oracle"), suite)
    assert results.pooled["rmse"] == pytest.approx(0.0, abs=1e-6)
