"""Query rows: every consumer must read the row the generator actually queried.

The generators encode ``Episode.query_time`` differently. ``ExtendedDoTime``
(identifiability) stores ``index / T``, the continuous prior stores normalized
observation time (``index / (T - 1)`` on its regular grid), and the generic and
regime paths store the step itself. ``evaluation.query_obs_levels`` used to read
every fraction as ``index / T``, which on ``dot-Continuous-v1`` returned the
observational level one step late for every query in the second half of the
trajectory and so corrupted effect-scored direction accuracy.

The reference value in these tests is the generator's own ``Y_obs``, the factual
observational value it computed at its own query index, so the checks do not
reuse the code under test.
"""

from __future__ import annotations

import pytest
import torch

from dotime.benchmarks import (
    _SUITE_REGISTRY,
    QUERY_TIME_ENCODINGS,
    Episode,
    episode_from_sample,
    query_time_to_index,
)
from dotime.continuous import ContinuousExtendedPrior
from dotime.evaluation import query_obs_levels
from dotime.extended import ExtendedDoTime
from dotime.interventions import InterventionSpec, InterventionType

CONTINUOUS = ("back_door", "front_door", "instrumental_variable")


def _continuous(structure, schedule="regular", n=6, t_len=120, seed=7):
    gen = ContinuousExtendedPrior(tscm_structure=structure, schedule=schedule, seed=seed)
    samples = [gen.generate_sample(T=t_len) for _ in range(n)]
    return [(s, episode_from_sample(s, structure=structure)) for s in samples]


def _identifiability(structure, offsets=(0, 0), n=4, t_len=80, seed=3):
    torch.manual_seed(seed)
    gen = ExtendedDoTime(tscm_structure=structure, n_max=41, seed=seed, query_offset_range=offsets)
    samples = [gen.generate_sample(T=t_len) for _ in range(n)]
    return [(s, episode_from_sample(s, structure=structure)) for s in samples]


@pytest.mark.parametrize("structure", CONTINUOUS)
def test_continuous_obs_level_is_read_at_the_true_query_row(structure):
    pairs = _continuous(structure)
    old_reads_late = 0
    for s, ep in pairs:
        # Regular dt=1 grid starting at 0: the absolute query time is the row.
        true_row = round(float(s["t_query"]))
        assert int(ep.query_time_idx[0]) == true_row
        assert torch.equal(query_obs_levels(ep), s["Y_obs"].reshape(-1))
        old_reads_late += query_time_to_index(ep.query_time, ep.length) != [true_row]
    # Guard: the sampled queries include rows the old index / T reading missed,
    # so this test fails on the pre-fix helper.
    assert old_reads_late > 0


@pytest.mark.parametrize("schedule", ["jittered", "exponential"])
def test_continuous_irregular_schedules_resolve_exactly(schedule):
    # No fraction-based encoding can serve an irregular grid; the row recorded
    # from the sample's own time grid can.
    for s, ep in _continuous("back_door", schedule=schedule, n=4):
        assert torch.equal(query_obs_levels(ep), s["Y_obs"].reshape(-1))


@pytest.mark.parametrize(
    ("structure", "offsets"),
    [("back_door", (0, 0)), ("front_door", (0, 0)), ("mediator", (0, 20))],
)
def test_identifiability_obs_level_matches_generator_and_old_reading(structure, offsets):
    for s, ep in _identifiability(structure, offsets):
        assert torch.equal(query_obs_levels(ep), s["Y_obs"].reshape(-1))
        # Behaviour is unchanged for this suite: index / T was always right here.
        assert ep.query_time_idx.tolist() == query_time_to_index(ep.query_time, ep.length)


def test_frozen_files_resolve_rows_from_the_suite_declaration(tmp_path):
    """Frozen v1 files predate ``query_time_idx``; the loader must resolve it."""
    pytest.importorskip("pyarrow")
    from dotime import _release_io

    cases = [
        ("dot-Continuous-v1", _continuous("back_door") + _continuous("front_door")),
        ("dot-Identifiability-v1", _identifiability("back_door", offsets=(0, 30))),
    ]
    for name, pairs in cases:
        for _s, ep in pairs:
            del ep.metadata["query_time_idx"]  # as in the released parquet shards
        meta = _SUITE_REGISTRY[name]
        suite_dir = tmp_path / name
        _release_io.write_suite(
            meta, [ep for _s, ep in pairs], suite_dir, package_version="t", seed=0
        )
        loaded = _release_io.read_suite(meta, suite_dir)
        for (s, _ep), got in zip(pairs, loaded, strict=True):
            assert torch.equal(query_obs_levels(got), s["Y_obs"].reshape(-1))

    # A wrong declaration fails loudly instead of reading a neighbouring row.
    wrong = _SUITE_REGISTRY["dot-Identifiability-v1"]
    with pytest.raises(ValueError, match="not a whole row"):
        _release_io.read_suite(wrong, tmp_path / "dot-Continuous-v1")


def test_registry_declares_every_suites_encoding():
    declared = {name: meta.query_time_encoding for name, meta in _SUITE_REGISTRY.items()}
    assert declared == {
        "dot-Identifiability-v1": "index/T",
        "dot-RegimeSwitch-v1": "step",
        "dot-Continuous-v1": "index/(T-1)",
        "dot-Generic-100k": "step",
    }
    # Pinning an earlier release keeps the declaration.
    pinned = _SUITE_REGISTRY["dot-Identifiability-v1"].for_version("1.0.0")
    assert pinned.query_time_encoding == "index/T"


def test_query_time_to_index_encodings_and_guards():
    t_len, row = 200, 150
    assert query_time_to_index([row], t_len, "step") == [row]
    assert query_time_to_index([row / t_len], t_len, "index/T") == [row]
    assert query_time_to_index([row / (t_len - 1)], t_len, "index/(T-1)") == [row]
    # The undeclared guess reads a continuous fraction one row late (the bug)...
    assert query_time_to_index([row / (t_len - 1)], t_len) == [row + 1]
    # ...but is unchanged for steps and index / T fractions.
    assert query_time_to_index([row, row / t_len], t_len) == [row, row]
    assert query_time_to_index([250.0, -3.0], t_len, "step") == [t_len - 1, 0]
    assert set(QUERY_TIME_ENCODINGS) == {"step", "index/T", "index/(T-1)"}
    with pytest.raises(ValueError, match="unknown query_time encoding"):
        query_time_to_index([0.5], t_len, "fraction")
    with pytest.raises(ValueError, match="not a whole row"):
        query_time_to_index([row / (t_len - 1)], t_len, "index/T")
    with pytest.raises(ValueError, match="length must be positive"):
        query_time_to_index([0.0], 0, "step")


def test_recorded_rows_must_match_the_queries():
    x = torch.zeros(10, 3)
    ep = Episode(
        x_obs=x,
        x_int=x,
        intervention=InterventionSpec(
            targets=[0], times=[4], intervention_type=InterventionType.HARD, values=1.0
        ),
        y_true=torch.tensor([0.0]),
        query_target=torch.tensor([1]),
        query_time=torch.tensor([0.5]),
        metadata={"query_time_idx": [4, 5]},
    )
    with pytest.raises(ValueError, match="query_time_idx"):
        query_obs_levels(ep)
    ep.metadata["query_time_idx"] = [10]
    with pytest.raises(ValueError, match="query_time_idx"):
        query_obs_levels(ep)


def test_chronos_horizon_ends_at_the_true_query_row():
    pytest.importorskip("pandas")
    from dotime.reference.chronos import _episode_frames

    for s, ep in _continuous("back_door", n=4):
        onset = min(ep.intervention.times)
        _ctx, _future, horizon = _episode_frames(ep, use_covariate=False)
        assert onset + horizon - 1 == round(float(s["t_query"]))
