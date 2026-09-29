"""Per-arm target QA (``dotime.qa``) and its hooks in the build, the CLIs and the loader.

The rule behind it: seeds guard against variance, not against a
systematically corrupted target, so every build, benchmark run and training
loader logs and asserts the nonzero fraction, mean and variance of the
observational level, the interventional level and their difference before
anything is trusted.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx
import numpy as np
import pytest
import torch

from dotime.benchmarks import _SUITE_REGISTRY, BenchmarkSuite, Episode
from dotime.interventions import InterventionSpec, InterventionType
from dotime.qa import (
    ARMS,
    QAThresholds,
    TargetQAError,
    arm_stats,
    batch_target_qa,
    effect_lag,
    is_null_effect,
    target_qa,
)
from dotime.tscm_sampler import TSCMSampler, TSCMStructure

ONSET, T_LEN = 20, 30


def _episode(
    structure: str | None = "back_door",
    *,
    effect: float = 1.0,
    y_true: float | None = None,
    offset: int = 0,
    seed: int = 0,
    scm_id: int = 0,
) -> Episode:
    """A one-query episode with a chosen effect at a chosen query offset.

    Args:
        structure: Structure label.
        effect: ``y_true - y_obs`` at the query, unless ``y_true`` is given.
        y_true: Overrides the target level.
        offset: Query row minus the intervention onset ``ONSET``.
        seed: Seed of the random trajectory.
        scm_id: Episode id.

    Returns:
        An episode querying the last of three variables at ``ONSET + offset``,
        with that row recorded as ``query_time_idx``.
    """
    x = torch.randn(T_LEN, 3, generator=torch.Generator().manual_seed(seed))
    row = ONSET + offset
    target = float(x[row, 2]) + effect if y_true is None else y_true
    x_int = x.clone()
    x_int[row, 2] = target
    return Episode(
        x_obs=x,
        x_int=x_int,
        intervention=InterventionSpec([0], [ONSET], InterventionType.HARD, 1.0),
        y_true=torch.tensor([target]),
        query_target=torch.tensor([2]),
        query_time=torch.tensor([float(row)]),
        structure=structure,
        scm_id=scm_id,
        metadata={"query_time_idx": [row], "y_oracle": torch.tensor([target])},
    )


def _episodes(structure: str | None, n: int, **kwargs: Any) -> list[Episode]:
    """``n`` episodes of one structure with distinct trajectories.

    Args:
        structure: Structure label.
        n: Number of episodes.
        **kwargs: Passed to :func:`_episode`.

    Returns:
        Episodes with seeds and ids ``0 .. n - 1``.
    """
    return [_episode(structure, seed=i, scm_id=i, **kwargs) for i in range(n)]


# --------------------------------------------------------------------------- #
# Statistics and exemptions
# --------------------------------------------------------------------------- #


def test_arm_stats_counts_nonfinite_values_and_summarises_the_finite_ones() -> None:
    st = arm_stats([0.0, 1.0, 3.0, float("nan"), float("inf")])
    assert (st["n"], st["n_nonfinite"]) == (5, 2)
    assert st["nonzero_frac"] == pytest.approx(2 / 3)
    assert st["mean"] == pytest.approx(4 / 3)
    assert st["var"] == pytest.approx(np.var([0.0, 1.0, 3.0]))
    assert st["abs_max"] == 3.0
    for values in ([], [float("nan")]):
        st = arm_stats(values)
        assert st["nonzero_frac"] is st["mean"] is st["var"] is st["abs_max"] is None
    json.dumps(arm_stats([float("nan")]), allow_nan=False)


def _reachable(structure: TSCMStructure, offset: int) -> bool:
    """Whether ``A`` at the onset reaches ``Y`` ``offset`` steps later in the unrolled DAG.

    Independent of :func:`effect_lag`: it unrolls the temporal DAG over time
    (instantaneous edges within a step, ``G_lags[k]`` edges across ``k + 1``
    steps, as ``TemporalSCM`` reads them) and asks networkx for a path.

    Args:
        structure: A named structure.
        offset: Query offset after the onset.

    Returns:
        ``True`` if a directed path from ``(A, 0)`` to ``(Y, offset)`` exists.
    """
    dag = TSCMSampler(structure)._build_dag()
    unrolled = nx.DiGraph()
    for t in range(offset + 1):
        unrolled.add_nodes_from((v, t) for v in dag.topo_order)
        unrolled.add_edges_from(((u, t), (v, t)) for u, v in dag.G_0.edges())
        for k, g_k in enumerate(dag.G_lags):
            if t - k - 1 >= 0:
                for j, i in zip(*np.nonzero(g_k), strict=True):
                    unrolled.add_edge((dag.topo_order[j], t - k - 1), (dag.topo_order[i], t))
    return nx.has_path(unrolled, ("A", 0), ("Y", offset))


@pytest.mark.parametrize("structure", list(TSCMStructure), ids=lambda s: s.value)
def test_null_effect_exemption_matches_dag_reachability(structure: TSCMStructure) -> None:
    """Every named structure, bow_graph included, is exempt exactly where A cannot reach Y."""
    for offset in range(4):
        assert is_null_effect(structure.value, offset) is not _reachable(structure, offset), offset


def test_exemptions_of_the_released_structures() -> None:
    """The cases the rebuttal cites, spelled out."""
    for control in ("observed_confounder", "unobserved_confounder"):
        assert effect_lag(control) is None
        assert is_null_effect(control, 0)
        assert is_null_effect(control, 50)
    assert effect_lag("mediator") == 1
    assert is_null_effect("mediator", 0)
    assert not is_null_effect("mediator", 1)
    assert effect_lag("bow_graph") == 0
    assert not is_null_effect("bow_graph", 0)
    # A lagged structure is not exempt when the offset is unknown.
    assert not is_null_effect("mediator", None)
    # The legacy label of the v1.0.0 files is bi_variate.
    assert effect_lag("rct_no_confounding") == effect_lag("bi_variate") == 0


@pytest.mark.parametrize("structure", [None, "regime_2", "no_such_structure", ""])
def test_unknown_structures_are_never_exempt(structure: str | None) -> None:
    assert not is_null_effect(structure, 0)
    if structure is not None:
        with pytest.raises(ValueError, match="not a named TSCM structure"):
            effect_lag(structure)


# --------------------------------------------------------------------------- #
# target_qa on episodes
# --------------------------------------------------------------------------- #


def test_clean_targets_pass_with_groups_exemptions_and_a_json_report() -> None:
    lines: list[str] = []
    episodes = _episodes("back_door", 12) + _episodes("observed_confounder", 12, effect=0.0)
    report = target_qa(episodes, dir_target="effect", log=lines.append)
    assert report.passed
    assert report.problems == []
    assert report.notes == []
    assert sorted(report.groups) == ["back_door", "observed_confounder"]
    assert report.pooled["n"] == 24
    assert report.pooled["n_null_effect_exempt"] == 12
    assert report.pooled["effect_checked"]["n"] == 12
    assert report.pooled["effect_checked"]["nonzero_frac"] == 1.0
    assert report.pooled["effect"]["mean"] == pytest.approx(0.5, abs=1e-6)
    assert all(s["asserted"] for s in report.groups.values())
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["passed"] is True
    assert payload["check_effect"] is True
    assert set(ARMS) <= set(payload["pooled"])
    assert any("observed_confounder" in line and "exempt" in line for line in lines)
    assert lines[-1].startswith("[target QA] passed")


def test_an_all_zero_level_arm_fails_and_names_the_arm() -> None:
    episodes = _episodes("back_door", 12, y_true=0.0)
    with pytest.raises(TargetQAError, match="y_int_level") as info:
        target_qa(episodes, log=None)
    assert isinstance(info.value, RuntimeError)
    assert not info.value.report.passed
    report = target_qa(episodes, raise_on_failure=False, log=None)
    assert any("pooled y_int_level: nonzero_frac" in p for p in report.problems)
    assert any("back_door y_int_level" in p for p in report.problems)


def test_small_groups_are_reported_but_not_asserted() -> None:
    good = _episodes("front_door", 20)
    small_bad = _episodes("back_door", QAThresholds().min_group_n - 1, y_true=0.0)
    report = target_qa(good + small_bad, log=None)
    assert report.passed
    assert not report.groups["back_door"]["asserted"]
    assert any(n.startswith("back_door y_int_level") for n in report.notes)
    big_bad = _episodes("back_door", QAThresholds().min_group_n, y_true=0.0)
    report = target_qa(good + big_bad, raise_on_failure=False, log=None)
    assert not report.passed
    assert all(p.startswith("back_door") for p in report.problems)


def test_the_pooled_arms_are_asserted_from_two_queries() -> None:
    one = [_episode(y_true=0.0)]
    assert target_qa(one, log=None).passed
    with pytest.raises(TargetQAError, match="pooled y_int_level"):
        target_qa(one * 2, log=None)
    with pytest.raises(TargetQAError, match="no queries"):
        target_qa([], log=None)


def test_a_nonfinite_target_fails() -> None:
    episodes = _episodes(None, 12)
    episodes[3] = _episode(None, y_true=float("nan"), seed=3)
    with pytest.raises(TargetQAError, match="not finite"):
        target_qa(episodes, log=None)


def test_the_effect_check_follows_the_structure_dag() -> None:
    """A zero effect fails where the DAG allows one, and is exempt where it cannot exist."""
    zero_back_door = _episodes("back_door", 12, effect=0.0)
    assert target_qa(zero_back_door, dir_target="level", log=None).passed  # ignores the effect
    with pytest.raises(TargetQAError, match="back_door effect"):
        target_qa(zero_back_door, dir_target="effect", log=None)
    # mediator: A(t-1) -> M(t) -> Y(t), so a query at the onset has no effect yet.
    assert target_qa(_episodes("mediator", 12, effect=0.0), dir_target="effect", log=None).passed
    with pytest.raises(TargetQAError, match="mediator effect"):
        target_qa(_episodes("mediator", 12, effect=0.0, offset=1), dir_target="effect", log=None)
    # Unlabelled episodes are never exempt.
    with pytest.raises(TargetQAError, match="pooled effect"):
        target_qa(_episodes(None, 12, effect=0.0), dir_target="effect", log=None)


def test_obs_levels_override_query_obs_levels_and_are_validated() -> None:
    episodes = _episodes(None, 4)
    report = target_qa(episodes, obs_levels=[[0.0]] * 4, raise_on_failure=False, log=None)
    assert report.pooled["y_obs_level"]["nonzero_frac"] == 0.0
    assert not report.passed
    with pytest.raises(ValueError, match="obs_levels"):
        target_qa(episodes, obs_levels=[[0.0]] * 3, log=None)
    with pytest.raises(ValueError, match="observational levels"):
        target_qa(episodes, obs_levels=[[0.0, 1.0]] * 4, log=None)
    with pytest.raises(ValueError, match="dir_target"):
        target_qa(episodes, dir_target="sign", log=None)


def test_reference_table_wrapper_keeps_its_contract() -> None:
    """``dotime-eval-reference``'s target_qa keeps its signature, keys and RuntimeError."""
    from dotime.reference.reference_table import TARGET_QA_MIN_NONZERO
    from dotime.reference.reference_table import target_qa as reference_target_qa

    out = reference_target_qa(_episodes("back_door", 12), None, "effect")
    assert {"y_obs_level", "y_int_level", "effect", "min_nonzero_frac"} <= set(out)
    assert out["min_nonzero_frac"] == TARGET_QA_MIN_NONZERO == 0.5
    assert out["passed"] is True
    assert "back_door" in out["groups"]
    with pytest.raises(RuntimeError, match="y_int_level"):
        reference_target_qa(_episodes("back_door", 12, y_true=0.0))


# --------------------------------------------------------------------------- #
# batch_target_qa and the training loader
# --------------------------------------------------------------------------- #


def _loader(**kwargs: Any):
    """A small back_door loader, as the prefetch tests build it.

    Args:
        **kwargs: Overrides for ``TemporalInterventionDataLoader``.

    Returns:
        The loader.
    """
    from dotime.data import TemporalInterventionDataLoader

    config = {"num_steps": 6, "batch_size": 16, "tscm_structure": "back_door", "seed": 3}
    return TemporalInterventionDataLoader(**{**config, **kwargs})


def _zero_targets(loader: Any, key: str = "Y_true") -> None:
    """Make every batch of a loader's prior carry an all-zero target field.

    Args:
        loader: A single-structure loader.
        key: The batch field to zero.
    """
    generate = loader.prior.generate_batch

    def _zeroed(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        batch = generate(*args, **kwargs)
        batch[key] = torch.zeros_like(batch[key])
        return batch

    loader.prior.generate_batch = _zeroed


def test_batch_target_qa_reads_raw_targets_and_offsets() -> None:
    batch = {
        "X_obs": torch.zeros(4, 50, 3),
        "Y_true": torch.tensor([1.0, 2.0, -1.0, 0.5]),
        "Y_obs": torch.tensor([1.0, 1.5, -2.0, 0.25]),
        "query_time": torch.tensor([20 / 50, 21 / 50, 20 / 50, 22 / 50]),
        "int_onset_idx": torch.tensor([20, 20, 20, 20]),
    }
    batch["Y_causal_effect"] = batch["Y_true"] - batch["Y_obs"]
    report = batch_target_qa(batch, structure="mediator", target_key="Y_causal_effect", log=None)
    assert report.label == "mediator"
    # Offsets 0, 1, 0, 2: the two onset queries of a mediator cannot carry an effect.
    assert report.pooled["n_null_effect_exempt"] == 2
    assert report.passed


@pytest.mark.parametrize("prefetch", [0, 2])
def test_loader_raises_on_all_zero_targets_unless_opted_out(prefetch: int) -> None:
    loader = _loader(prefetch=prefetch)
    _zero_targets(loader)
    with pytest.raises(TargetQAError, match="back_door y_int_level"):
        list(loader)
    loader = _loader(prefetch=prefetch, target_qa=False)
    _zero_targets(loader)
    assert len(list(loader)) == 6


def _seeded_batches(**kwargs: Any) -> list[dict[str, torch.Tensor]]:
    """Every batch of a fresh loader, with the global RNGs reset first.

    A diverged sample is replaced through ``generate_sample``, which draws from
    the global torch RNG, so two loaders with the same seed only agree when the
    global state is reset too, target QA or not.

    Args:
        **kwargs: Overrides for :func:`_loader`.

    Returns:
        The loader's batches.
    """
    import random

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    return list(_loader(**kwargs))


def test_loader_batches_are_bit_identical_with_target_qa_on_or_off() -> None:
    for kwargs in (
        {},
        {"tscm_structure": None, "tscm_structures": ["back_door", "front_door"], "batch_size": 8},
    ):
        on = _seeded_batches(**kwargs)
        off = _seeded_batches(target_qa=False, **kwargs)
        assert len(on) == len(off) == 6
        for got, want in zip(on, off, strict=True):
            assert got.keys() == want.keys()
            for key in got:
                assert torch.equal(got[key], want[key]), key


def test_loader_logs_through_logging_and_exempts_null_effect_structures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Training on the effect of a structure without an A -> Y path is not an error."""
    caplog.set_level(logging.WARNING, logger="dotime.data")
    loader = _loader(
        tscm_structure="observed_confounder",
        pair_mode="counterfactual",
        target_key="Y_causal_effect",
    )
    batches = list(loader)
    effect = torch.cat([b["Y_causal_effect"] for b in batches])
    assert bool((effect == 0).all()), "shared noise and no A -> Y path give a zero effect"
    lines = [r.getMessage() for r in caplog.records if r.name == "dotime.data"]
    assert any(line.startswith("[target QA] passed") for line in lines)
    assert any("cannot carry an effect" in line for line in lines)
    loader = _loader(target_key="Y_causal_effect")
    _zero_targets(loader, "Y_causal_effect")
    with pytest.raises(TargetQAError, match="back_door effect"):
        list(loader)


# --------------------------------------------------------------------------- #
# CLI hooks, with a stubbed load_benchmark
# --------------------------------------------------------------------------- #


def _stub_load(episodes: list[Episode]):
    """A ``load_benchmark`` stand-in serving copies of fixed episodes.

    Args:
        episodes: The episodes every load returns.

    Returns:
        A function with the signature of :func:`~dotime.benchmarks.load_benchmark`.
    """

    def _load(name: str, version: str = "latest", **_: Any) -> BenchmarkSuite:
        return BenchmarkSuite(_SUITE_REGISTRY[name].for_version(version), list(episodes))

    return _load


@pytest.mark.parametrize("zeroed", [False, True])
def test_dotime_benchmark_asserts_and_records_target_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, zeroed: bool
) -> None:
    import dotime.benchmarks
    from dotime.cli import benchmark_main

    episodes = _episodes(None, 12, **({"y_true": 0.0} if zeroed else {}))
    monkeypatch.setattr(dotime.benchmarks, "load_benchmark", _stub_load(episodes))
    out = tmp_path / "bench.json"
    argv = ["--suite", "dot-Generic-100k", "--baseline", "Mean", "--json-out", str(out)]
    if zeroed:
        with pytest.raises(TargetQAError, match="y_int_level"):
            benchmark_main(argv)
        assert not out.exists()
        argv += ["--target-qa", "warn"]
    assert benchmark_main(argv) == 0
    qa = json.loads(out.read_text())["target_qa"]
    assert qa["passed"] is not zeroed
    assert qa["pooled"]["n"] == 12


@pytest.mark.parametrize("zeroed", [False, True])
def test_dotime_eval_submission_asserts_and_records_target_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, zeroed: bool
) -> None:
    from dotime.reference import submission

    episodes = _episodes("back_door", 12, **({"y_true": 0.0} if zeroed else {}))
    monkeypatch.setattr(submission, "load_benchmark", _stub_load(episodes))
    out = tmp_path / "submission.json"
    argv = ["--suite", "dot-Identifiability-v1", "--baseline", "Mean", "--out", str(out)]
    argv += ["--dir-target", "effect"]
    if zeroed:
        with pytest.raises(TargetQAError):
            submission.main(argv)
        argv += ["--target-qa", "warn"]
    assert submission.main(argv) == 0
    payload = json.loads(out.read_text())
    assert payload["target_qa"]["passed"] is not zeroed
    assert payload["target_qa"]["dir_target"] == payload["dir_target"] == "effect"


def test_pfn_evaluator_checks_the_scored_episodes_before_loading_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dotime.reference import pfn

    built: list[str] = []

    class _Model:
        """Stands in for a PFN checkpoint. Predicts zero."""

        def predict(self, ep: Episode) -> torch.Tensor:
            return torch.zeros(1)

    def _build(checkpoint: str, **_: Any) -> _Model:
        built.append(checkpoint)
        return _Model()

    monkeypatch.setattr(pfn, "PFNRef", _build)
    argv = ["--suite", "dot-Identifiability-v1", "--ckpt-int", "i.pt", "--ckpt-obs", "o.pt"]
    argv += ["--device", "cpu", "--out", str(tmp_path / "pfn.json")]
    monkeypatch.setattr(pfn, "load_benchmark", _stub_load(_episodes("back_door", 12, y_true=0.0)))
    with pytest.raises(TargetQAError):
        pfn.main(argv)
    assert built == []
    monkeypatch.setattr(pfn, "load_benchmark", _stub_load(_episodes("back_door", 12)))
    pfn.main(argv)
    assert built == ["i.pt", "o.pt"]
    assert json.loads((tmp_path / "pfn.json").read_text())["target_qa"]["passed"] is True


def test_tabpfn_and_chronos_check_their_subsample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pandas")
    from dotime.reference import chronos, tabpfn

    class _Regressor:
        """Stands in for ``TabPFNRegressor``. Predicts zeros."""

        def fit(self, design: np.ndarray, target: np.ndarray) -> _Regressor:
            return self

        def predict(self, design: np.ndarray) -> np.ndarray:
            return np.zeros(len(design))

    class _Pipeline:
        """Stands in for a Chronos-2 pipeline. Forecasts zeros."""

        def predict_df(self, context_df: Any, **kwargs: Any) -> Any:
            import pandas as pd

            return pd.DataFrame({"predictions": np.zeros(kwargs["prediction_length"])})

    monkeypatch.setattr(tabpfn, "_regressor", lambda: _Regressor)
    monkeypatch.setenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    monkeypatch.setattr(chronos, "_load_pipeline", lambda model_id, device: _Pipeline())
    for module in (tabpfn, chronos):
        out = tmp_path / f"{module.__name__}.json"
        argv = ["--suite", "dot-Identifiability-v1", "--device", "cpu", "--out", str(out)]
        bad = _stub_load(_episodes("back_door", 12, y_true=0.0))
        monkeypatch.setattr(module, "load_benchmark", bad)
        with pytest.raises(TargetQAError):
            module.main(argv)
        module.main([*argv, "--target-qa", "warn"])
        assert json.loads(out.read_text())["target_qa"]["passed"] is False
        monkeypatch.setattr(module, "load_benchmark", _stub_load(_episodes("back_door", 12)))
        module.main(argv)
        assert json.loads(out.read_text())["target_qa"]["passed"] is True


# --------------------------------------------------------------------------- #
# The release build hook
# --------------------------------------------------------------------------- #


def _build_release(monkeypatch: pytest.MonkeyPatch, episodes: list[Episode]) -> SimpleNamespace:
    """Load scripts/build_release.py with ``build_suite`` stubbed to return fixed episodes.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        episodes: What every suite build returns.

    Returns:
        The loaded script module.
    """
    pytest.importorskip("pyarrow")
    pytest.importorskip("yaml")
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_release.py"
    spec = importlib.util.spec_from_file_location("build_release_qa", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "build_suite", lambda cfg, seed, scale, workers: list(episodes))
    return module


@pytest.mark.parametrize("mode", ["enforce", "warn", "off"])
@pytest.mark.parametrize("zeroed", [False, True])
def test_build_release_asserts_target_qa_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, zeroed: bool
) -> None:
    episodes = _episodes(None, 12, **({"y_true": 0.0} if zeroed else {}))
    br = _build_release(monkeypatch, episodes)
    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 1\nsuites:\n  dot-Generic-100k:\n"
        "    version: '9.9.9'\n    generator: generic\n    T: 30\n    n_episodes: 12\n"
    )
    argv = ["--config", str(config), "--output-dir", str(tmp_path), "--timestamp", "T"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rc = br.main([*argv, "--workers", "1", "--target-qa", mode])
    build_manifest = json.loads((tmp_path / "T" / "build_manifest.json").read_text())
    assert build_manifest["target_qa"] == mode
    suite_dir = tmp_path / "T" / "dot-Generic-100k-9.9.9"
    if zeroed and mode == "enforce":
        assert rc == 1
        assert not suite_dir.exists()
        assert build_manifest["suites"] == []
        assert build_manifest["target_qa_failed"]["target_qa"]["passed"] is False
        return
    assert rc == 0
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    if mode == "off":
        assert manifest["target_qa"] == {"mode": "off"}
        return
    assert manifest["target_qa"]["mode"] == mode
    assert manifest["target_qa"]["passed"] is not zeroed
    assert manifest["target_qa"]["check_effect"] is True
    assert manifest["target_qa"]["pooled"]["n"] == 12
    assert build_manifest["suites"][0]["target_qa"]["passed"] is not zeroed
    assert math.isfinite(manifest["target_qa"]["pooled"]["y_int_level"]["var"])
