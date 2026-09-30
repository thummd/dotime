"""``--version`` and ``--realignment`` on the reference evaluators.

TabPFN, Chronos-2, the reference baselines and the Do-Over-Time-PFN all read
``x_obs`` by canonical column, but the archived ``dot-Identifiability-v1`` 1.0.0
files store ``x_obs`` in topological order. These tests serve small 1.0.0-style
``back_door`` episodes through each evaluator's ``main`` and record which
columns a stub model receives, so neither TabPFN, Chronos nor a PFN checkpoint
is needed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from dotime.benchmarks import _SUITE_REGISTRY, BenchmarkSuite, Episode
from dotime.interventions import InterventionSpec, InterventionType
from dotime.reference import chronos, pfn, reference_table, tabpfn
from dotime.reference._realignment import load_realignment, realign_episodes

SUITE = "dot-Identifiability-v1"
# back_door's released permutation (canonical index -> stored column): the
# 1.0.0 files hold canonical [A, X, Y] as stored columns [X, A, Y].
PERM = (1, 0, 2)
A, X, Y = 0, 1, 2
T_LEN, ONSET, QUERY_STEP, DO_VALUE = 60, 40, 50, 1.5
RELEASED_SIDECAR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "reference"
    / "dot-Identifiability-v1.0.0_realignment.jsonl"
)


def _canonical_x(n_vars: int = 3) -> np.ndarray:
    """A trajectory whose columns can be told apart by value.

    Args:
        n_vars: Number of columns.

    Returns:
        ``(T_LEN, n_vars)`` float32 array whose column ``j`` is ``100 (j + 1) + t``.
    """
    t = np.arange(T_LEN, dtype=np.float32)
    return np.stack([100.0 * (j + 1) + t for j in range(n_vars)], axis=1)


def _episode(
    perm: tuple[int, ...] = PERM,
    scm_id: int = 7,
    y_true: float = 0.25,
    n_vars: int = 3,
    offset: float = 0.0,
) -> Episode:
    """A single-query episode stored the way the 1.0.0 files store it.

    Args:
        perm: Canonical index to stored column. Stored column ``perm[c]``
            holds canonical variable ``c``.
        scm_id: Episode id, the key into the realignment sidecar.
        y_true: The episode's target, which the sidecar fingerprints.
        n_vars: Number of variables.
        offset: Added to every value of both trajectories, so that episodes
            can differ in their observational levels.

    Returns:
        An episode with a hard do(A = ``DO_VALUE``) at ``ONSET`` that queries
        the last canonical variable at ``QUERY_STEP``.
    """
    canonical = _canonical_x(n_vars) + offset
    stored = np.empty_like(canonical)
    stored[:, list(perm)] = canonical
    return Episode(
        x_obs=torch.from_numpy(stored),
        x_int=torch.from_numpy(canonical),
        intervention=InterventionSpec(
            targets=[A], times=[ONSET], intervention_type=InterventionType.HARD, values=DO_VALUE
        ),
        y_true=torch.tensor([y_true]),
        query_target=torch.tensor([n_vars - 1]),
        query_time=torch.tensor([float(QUERY_STEP)]),
        structure="back_door",
        scm_id=scm_id,
    )


def _row(ep: Episode, **overrides: Any) -> dict[str, Any]:
    """The sidecar row that describes an episode built by :func:`_episode`.

    Args:
        ep: The episode the row should describe.
        **overrides: Fields to replace, e.g. to simulate another suite version.

    Returns:
        A row with every field the released sidecar carries.
    """
    row = {
        "idx": ep.scm_id,
        "structure": ep.structure,
        "n_vars": ep.n_vars,
        "canonical_perm": list(PERM),
        "hidden_canonical": [],
        "query_target": int(ep.query_target[0]),
        "query_time_idx": QUERY_STEP,
        "y_true_regen": float(ep.y_true[0]),
        "y_true_match": True,
        "y_obs_corrected": 0.0,
        "y_effect_corrected": 0.0,
    }
    row.update(overrides)
    return row


def _write_sidecar(path: Path, rows: list[dict[str, Any]]) -> Path:
    """Write rows as a JSONL sidecar.

    Args:
        path: Destination file.
        rows: One JSON object per line.

    Returns:
        ``path``.
    """
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _stub_load_benchmark(
    episodes: Callable[[], list[Episode]], calls: list[tuple[str, str]]
) -> Callable[..., BenchmarkSuite]:
    """A ``load_benchmark`` stand-in that records each load.

    The stub resolves ``version`` through the real registry, so ``"latest"``
    maps to whatever version the registry currently serves.

    Args:
        episodes: Builds the episodes of each load. It is called per load, so
            a realigned copy from one run never reaches the next.
        calls: Receives the ``(name, version)`` of every load.

    Returns:
        A function with the signature of :func:`~dotime.benchmarks.load_benchmark`.
    """

    def _load(name: str, version: str = "latest", **_: Any) -> BenchmarkSuite:
        calls.append((name, version))
        return BenchmarkSuite(_SUITE_REGISTRY[name].for_version(version), episodes())

    return _load


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Serve one :func:`_episode` from the TabPFN and Chronos-2 ``load_benchmark``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The ``(name, version)`` of every load, in call order.
    """
    calls: list[tuple[str, str]] = []
    load = _stub_load_benchmark(lambda: [_episode()], calls)
    monkeypatch.setattr(tabpfn, "load_benchmark", load)
    monkeypatch.setattr(chronos, "load_benchmark", load)
    return calls


@pytest.fixture
def fits(monkeypatch: pytest.MonkeyPatch) -> list[tuple[np.ndarray, np.ndarray]]:
    """Swap TabPFN for a stub that records every ``(design, target)`` it is fit on.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The list the stub appends to, in fit order.
    """
    recorded: list[tuple[np.ndarray, np.ndarray]] = []

    class _RecordingRegressor:
        """Stands in for ``TabPFNRegressor``. Predicts zeros."""

        def fit(self, design: np.ndarray, target: np.ndarray) -> _RecordingRegressor:
            recorded.append((np.asarray(design), np.asarray(target)))
            return self

        def predict(self, design: np.ndarray) -> np.ndarray:
            return np.zeros(len(design))

    monkeypatch.setattr(tabpfn, "_regressor", lambda: _RecordingRegressor)
    # main() calls os.environ.setdefault on this key. Setting it through
    # monkeypatch first keeps that from leaking into later tests.
    monkeypatch.setenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    return recorded


@pytest.fixture
def forecasts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Swap the Chronos pipeline loader for a stub that records each forecast call.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        One ``{"context", "future", "prediction_length"}`` dict per
        ``predict_df`` call, in call order.
    """
    recorded: list[dict[str, Any]] = []

    class _RecordingPipeline:
        """Stands in for a Chronos-2 pipeline. Forecasts zeros."""

        def predict_df(
            self,
            context_df: Any,
            *,
            target: str,
            prediction_length: int,
            quantile_levels: list[float],
            future_df: Any = None,
        ) -> Any:
            # The evaluator builds its frames with pandas before calling this,
            # so pandas is importable whenever the stub runs.
            import pandas as pd

            recorded.append(
                {
                    "context": context_df.copy(),
                    "future": None if future_df is None else future_df.copy(),
                    "prediction_length": prediction_length,
                }
            )
            return pd.DataFrame({"predictions": np.zeros(prediction_length)})

    monkeypatch.setattr(chronos, "_load_pipeline", lambda model_id, device: _RecordingPipeline())
    return recorded


# --------------------------------------------------------------------------- #
# Sidecar loading and checks
# --------------------------------------------------------------------------- #


def test_released_sidecar_matches_the_loader_and_this_layout() -> None:
    """The released sidecar loads, covers every episode, and uses ``PERM`` for back_door."""
    if not RELEASED_SIDECAR.exists():
        pytest.skip("the released sidecar ships with the repository, not the sdist")
    rows = load_realignment(RELEASED_SIDECAR)
    assert sorted(rows) == list(range(10_800))
    back_door = {tuple(r["canonical_perm"]) for r in rows.values() if r["structure"] == "back_door"}
    assert back_door == {PERM}


def test_realign_permutes_to_canonical_and_zeroes_hidden() -> None:
    """A front_door-style row puts every variable back in place and zeroes the hidden U."""
    perm = (1, 0, 2, 3)
    ep = _episode(perm=perm, n_vars=4)
    row = _row(ep, canonical_perm=list(perm), hidden_canonical=[1])
    (fixed,) = realign_episodes([ep], {ep.scm_id: row})
    expected = _canonical_x(4)
    expected[:, 1] = 0.0
    np.testing.assert_array_equal(fixed.x_obs.numpy(), expected)
    assert fixed.y_true is ep.y_true


@pytest.mark.parametrize(
    "override",
    [
        {"y_true_regen": -0.5},  # same id, other target: e.g. a 1.1.0 episode
        {"n_vars": 4},
        {"query_target": 1},
    ],
)
def test_realign_refuses_a_row_built_from_another_episode(override: dict[str, Any]) -> None:
    """A row whose fingerprint disagrees with the episode is an error, not a silent permute.

    Args:
        override: The row field that disagrees with the episode.
    """
    ep = _episode()
    with pytest.raises(ValueError, match=r"not built from this episode.*--version 1\.0\.0"):
        realign_episodes([ep], {ep.scm_id: _row(ep, **override)})


@pytest.mark.parametrize("override", [{"canonical_perm": [0, 0, 2]}, {"hidden_canonical": [3]}])
def test_realign_refuses_a_malformed_row(override: dict[str, Any]) -> None:
    """A permutation that repeats a column or a hidden index out of range is rejected.

    Args:
        override: The malformed row field.
    """
    ep = _episode()
    with pytest.raises(ValueError, match="malformed"):
        realign_episodes([ep], {ep.scm_id: _row(ep, **override)})


def test_realign_refuses_an_episode_without_a_row() -> None:
    """Unlike ``dotime-eval-reference``, a missing row does not pass the episode through."""
    with pytest.raises(ValueError, match="no row for episode 7"):
        realign_episodes([_episode(scm_id=7)], {})


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (['{"idx": 1, "n_vars": 3}'], "lacks"),
        (["not json"], "not valid JSON"),
    ],
)
def test_load_refuses_incomplete_or_invalid_lines(
    tmp_path: Path, lines: list[str], match: str
) -> None:
    """Lines without the realignment fields, or not JSON at all, are rejected.

    Args:
        tmp_path: The pytest temporary directory.
        lines: The sidecar's lines.
        match: A fragment the error message must contain.
    """
    path = tmp_path / "sidecar.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=match):
        load_realignment(path)


def test_load_refuses_duplicate_ids(tmp_path: Path) -> None:
    """A repeated ``idx`` would silently shadow the first row, so it is an error.

    Args:
        tmp_path: The pytest temporary directory.
    """
    ep = _episode()
    path = _write_sidecar(tmp_path / "sidecar.jsonl", [_row(ep), _row(ep)])
    with pytest.raises(ValueError, match="duplicate realignment row for idx 7"):
        load_realignment(path)


# --------------------------------------------------------------------------- #
# The evaluators
# --------------------------------------------------------------------------- #


def _args(tmp_path: Path, realign: bool, version: str | None = "1.0.0") -> list[str]:
    """Command-line arguments for either evaluator.

    Args:
        tmp_path: Where to write the sidecar and the result JSON.
        realign: Whether to pass ``--realignment`` with a sidecar that matches
            :func:`_episode`.
        version: Value of ``--version``, or ``None`` to omit the flag.

    Returns:
        The argument list, writing results to ``tmp_path / "out.json"``.
    """
    argv = ["--suite", SUITE, "--out", str(tmp_path / "out.json")]
    if version is not None:
        argv += ["--version", version]
    if realign:
        sidecar = _write_sidecar(tmp_path / "sidecar.jsonl", [_row(_episode())])
        argv += ["--realignment", str(sidecar)]
    return argv


def _assert_provenance(tmp_path: Path, version: str, realign: bool) -> None:
    """Check the suite version and realignment fields of the result JSON.

    Args:
        tmp_path: The directory :func:`_args` wrote the result to.
        version: The resolved suite version the JSON must record.
        realign: Whether the run was realigned.
    """
    result = json.loads((tmp_path / "out.json").read_text())
    assert result["suite_version"] == version
    assert result["realigned"] is realign
    assert result["realignment_sidecar"] == ("sidecar.jsonl" if realign else None)


@pytest.mark.parametrize("realign", [True, False])
def test_tabpfn_fits_on_the_columns_it_is_given(
    tmp_path: Path, loads: list, fits: list, realign: bool
) -> None:
    """Realigned, the outcome model sees [A_t, X_t, Y_(t-1)]. Otherwise it sees stored columns.

    Without realignment the treatment slot holds the confounder X and the
    outcome slot holds Y only by luck of the permutation, which is what the
    July 2026 1.0.0 run fed TabPFN on back_door.

    Args:
        tmp_path: The pytest temporary directory.
        loads: Suite loads recorded by the stub ``load_benchmark``.
        fits: Designs and targets recorded by the stub regressor.
        realign: Whether to pass the sidecar.
    """
    tabpfn.main(_args(tmp_path, realign))
    x = _canonical_x() if realign else _episode().x_obs.numpy()
    expected = np.column_stack([x[1:ONSET, A], x[1:ONSET, X], x[: ONSET - 1, Y]])
    assert len(fits) == 2  # one outcome model per arm, int then obs
    for design, target in fits:
        np.testing.assert_array_equal(design, expected)
        np.testing.assert_array_equal(target, x[1:ONSET, Y])
    if not realign:
        np.testing.assert_array_equal(fits[0][0][:, 0], _canonical_x()[1:ONSET, X])
    assert loads == [(SUITE, "1.0.0")]
    _assert_provenance(tmp_path, "1.0.0", realign)


@pytest.mark.parametrize("realign", [True, False])
def test_chronos_forecasts_from_the_columns_it_is_given(
    tmp_path: Path, loads: list, forecasts: list, realign: bool
) -> None:
    """Realigned, Chronos-2 sees A as the covariate and Y as the target.

    Args:
        tmp_path: The pytest temporary directory.
        loads: Suite loads recorded by the stub ``load_benchmark``.
        forecasts: Forecast calls recorded by the stub pipeline.
        realign: Whether to pass the sidecar.
    """
    pytest.importorskip("pandas")
    chronos.main(_args(tmp_path, realign))
    x = _canonical_x() if realign else _episode().x_obs.numpy()
    int_call, obs_call = forecasts
    np.testing.assert_array_equal(int_call["context"]["actuator"].to_numpy(), x[:ONSET, A])
    np.testing.assert_array_equal(int_call["context"]["target"].to_numpy(), x[:ONSET, Y])
    np.testing.assert_array_equal(
        int_call["future"]["actuator"].to_numpy(), np.full(QUERY_STEP - ONSET + 1, DO_VALUE)
    )
    np.testing.assert_array_equal(obs_call["context"]["target"].to_numpy(), x[:ONSET, Y])
    assert "actuator" not in obs_call["context"]
    assert obs_call["future"] is None
    assert loads == [(SUITE, "1.0.0")]
    _assert_provenance(tmp_path, "1.0.0", realign)


def test_check_predictions_counts_and_refuses_all_nonfinite() -> None:
    """An arm with some NaN predictions is counted; one with none finite stops the run."""
    from dotime.reference._scoring import check_predictions

    assert check_predictions("arm", np.array([0.5, float("nan"), 1.0])) == 1
    assert check_predictions("arm", np.array([0.5, 1.0])) == 0
    with pytest.raises(SystemExit, match="every prediction is non-finite"):
        check_predictions("arm", np.array([float("nan"), float("inf")]))


@pytest.mark.parametrize("evaluator", [tabpfn, chronos], ids=["tabpfn", "chronos"])
@pytest.mark.parametrize("dir_target", ["level", "effect"])
def test_dir_target_selects_the_headline_and_both_scores_are_written(
    tmp_path: Path, loads: list, fits: list, forecasts: list, evaluator: Any, dir_target: str
) -> None:
    """``--dir-target`` fills ``dir_acc`` from the chosen score; both scores land in the JSON.

    The stubs predict zero, so the level score has no valid sign to match and the
    effect score compares ``-y_obs`` with ``y_true - y_obs``: the two differ, which
    is what makes the selection observable.

    Args:
        tmp_path: The pytest temporary directory.
        loads: Keeps the suite loader stubbed.
        fits: Keeps TabPFN stubbed.
        forecasts: Keeps Chronos stubbed.
        evaluator: The evaluator module under test.
        dir_target: The flag value.
    """
    evaluator.main([*_args(tmp_path, realign=False), "--dir-target", dir_target])
    result = json.loads((tmp_path / "out.json").read_text())
    assert result["dir_target"] == dir_target
    arm = next(k for k in result if k.endswith("_int"))
    scores = result[arm]
    for name in ("level", "effect"):
        assert {f"dir_acc_{name}", f"dir_n_valid_{name}", f"dir_acc_se_{name}"} <= set(scores)
    assert scores["dir_acc"] == scores[f"dir_acc_{dir_target}"]
    assert scores["dir_n_valid"] == scores[f"dir_n_valid_{dir_target}"]


@pytest.mark.parametrize("evaluator", [tabpfn, chronos], ids=["tabpfn", "chronos"])
def test_default_version_is_latest_and_the_resolved_one_is_recorded(
    tmp_path: Path, loads: list, fits: list, forecasts: list, evaluator: Any
) -> None:
    """Without ``--version`` the registry's current version loads and lands in the JSON.

    Args:
        tmp_path: The pytest temporary directory.
        loads: Suite loads recorded by the stub ``load_benchmark``.
        fits: Keeps TabPFN stubbed.
        forecasts: Keeps Chronos stubbed. Without pandas every Chronos
            forecast takes the evaluator's mean fallback, which is enough here.
        evaluator: The evaluator module under test.
    """
    evaluator.main(_args(tmp_path, realign=False, version=None))
    assert loads == [(SUITE, "latest")]
    _assert_provenance(tmp_path, _SUITE_REGISTRY[SUITE].version, realign=False)


@pytest.mark.parametrize("evaluator", [tabpfn, chronos], ids=["tabpfn", "chronos"])
def test_a_sidecar_from_another_version_stops_the_run_before_any_fit(
    tmp_path: Path, loads: list, fits: list, forecasts: list, evaluator: Any
) -> None:
    """A 1.0.0 sidecar against episodes with other targets fails before the model runs.

    Args:
        tmp_path: The pytest temporary directory.
        loads: Suite loads recorded by the stub ``load_benchmark``.
        fits: Designs recorded by the stub regressor.
        forecasts: Forecast calls recorded by the stub pipeline.
        evaluator: The evaluator module under test.
    """
    ep = _episode()
    sidecar = _write_sidecar(tmp_path / "sidecar.jsonl", [_row(ep, y_true_regen=-0.5)])
    argv = ["--suite", SUITE, "--realignment", str(sidecar), "--out", str(tmp_path / "out.json")]
    with pytest.raises(ValueError, match="not built from this episode"):
        evaluator.main(argv)
    assert fits == []
    assert forecasts == []
    assert not (tmp_path / "out.json").exists()


# --------------------------------------------------------------------------- #
# The full-suite evaluators: dotime-eval-reference and dotime-eval-pfn
# --------------------------------------------------------------------------- #

# Every stub model predicts STUB_PRED. Against the sidecar's observational
# levels PAIR_Y_OBS it gets the effect sign of both episodes wrong. Against the
# levels stored in x_obs (350 and 1350) it gets both right. The effect accuracy
# therefore shows which of the two the evaluator subtracted.
STUB_PRED = 0.5
PAIR_Y_TRUE = (0.25, -0.75)
PAIR_Y_OBS = (0.4, -0.2)
PAIR_OFFSETS = (0.0, 1000.0)

FULL_SUITE = pytest.mark.parametrize("evaluator", [reference_table, pfn], ids=["reference", "pfn"])


def _pair() -> list[Episode]:
    """Two episodes that differ in id, target and trajectory level.

    ``dotime-eval-reference`` asserts that each target arm varies before it
    scores anything, so a one-episode suite never reaches its baselines.

    Returns:
        Episodes 7 and 8, built by :func:`_episode` from ``PAIR_Y_TRUE`` and
        ``PAIR_OFFSETS``.
    """
    return [
        _episode(scm_id=scm_id, y_true=y_true, offset=offset)
        for scm_id, y_true, offset in zip((7, 8), PAIR_Y_TRUE, PAIR_OFFSETS, strict=True)
    ]


def _pair_rows(**overrides: Any) -> list[dict[str, Any]]:
    """Sidecar rows that describe :func:`_pair`, with observational levels ``PAIR_Y_OBS``.

    Args:
        **overrides: Fields to replace in both rows, as in :func:`_row`.

    Returns:
        One row per episode of :func:`_pair`, in the same order.
    """
    return [
        _row(ep, **{"y_obs_corrected": y_obs, **overrides})
        for ep, y_obs in zip(_pair(), PAIR_Y_OBS, strict=True)
    ]


@pytest.fixture
def pair_loads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Serve :func:`_pair` from the full-suite evaluators' ``load_benchmark``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The ``(name, version)`` of every load, in call order.
    """
    calls: list[tuple[str, str]] = []
    load = _stub_load_benchmark(_pair, calls)
    monkeypatch.setattr(reference_table, "load_benchmark", load)
    monkeypatch.setattr(pfn, "load_benchmark", load)
    return calls


@pytest.fixture
def model_inputs(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Swap every model of the full-suite evaluators for a stub that records its input.

    ``dotime-eval-reference`` resolves baselines through its module's
    ``baselines`` attribute and ``dotime-eval-pfn`` builds ``PFNRef``, so both
    are replaced there. No checkpoint is opened.

    Args:
        monkeypatch: The pytest monkeypatch fixture.

    Returns:
        The ``x_obs`` of every episode a stub predicted, in call order.
    """
    seen: list[np.ndarray] = []

    class _RecordingModel:
        """Stands in for a baseline or a PFN checkpoint. Predicts ``STUB_PRED``."""

        def predict(self, ep: Episode) -> torch.Tensor:
            seen.append(ep.x_obs.numpy().copy())
            return torch.tensor([STUB_PRED])

    def _build(*_args: Any, **_kwargs: Any) -> _RecordingModel:
        return _RecordingModel()

    monkeypatch.setattr(reference_table, "baselines", SimpleNamespace(get=_build))
    monkeypatch.setattr(pfn, "PFNRef", _build)
    return seen


def _full_suite_args(
    evaluator: Any, tmp_path: Path, rows: list[dict[str, Any]] | None, version: str | None = "1.0.0"
) -> list[str]:
    """Command-line arguments for ``dotime-eval-reference`` or ``dotime-eval-pfn``.

    Both runs score the effect sign, so each one reads the observational level.

    Args:
        evaluator: The ``reference_table`` or the ``pfn`` module.
        tmp_path: Where to write the sidecar and the result JSON.
        rows: Sidecar rows to pass with ``--realignment``, or ``None`` to omit it.
        version: Value of ``--version``, or ``None`` to omit the flag.

    Returns:
        The argument list, writing results to ``tmp_path / "out.json"``.
    """
    argv = ["--suite", SUITE, "--dir-target", "effect", "--out", str(tmp_path / "out.json")]
    # The model_inputs stubs ignore the baseline name and the checkpoint paths.
    if evaluator is reference_table:
        argv += ["--baselines", "Stub"]
    else:
        argv += ["--ckpt-int", "int.pt", "--ckpt-obs", "obs.pt"]
    if version is not None:
        argv += ["--version", version]
    if rows is not None:
        argv += ["--realignment", str(_write_sidecar(tmp_path / "sidecar.jsonl", rows))]
    return argv


def _effect_accuracies(evaluator: Any, tmp_path: Path) -> list[float]:
    """Effect-sign accuracy of every scored model in the result JSON.

    Args:
        evaluator: The evaluator module that wrote the result.
        tmp_path: The directory :func:`_full_suite_args` wrote the result to.

    Returns:
        One accuracy per stub baseline of ``dotime-eval-reference``, or one
        per arm of ``dotime-eval-pfn``, interventional first.
    """
    result = json.loads((tmp_path / "out.json").read_text())
    if evaluator is reference_table:
        # A baseline that raises becomes a row with an "error" key instead of
        # failing the run, so check that the stub was actually scored.
        assert all("error" not in row for row in result["rows"]), result["rows"]
        return [row["dir_acc"] for row in result["rows"]]
    return [result[arm]["dir_acc_effect"] for arm in ("PFN_int", "PFN_obs")]


@FULL_SUITE
@pytest.mark.parametrize("realign", [True, False])
def test_full_suite_models_see_canonical_columns_and_the_sidecar_y_obs(
    tmp_path: Path, pair_loads: list, model_inputs: list, evaluator: Any, realign: bool
) -> None:
    """Realigned, every model reads canonical columns and the effect uses ``y_obs_corrected``.

    Args:
        tmp_path: The pytest temporary directory.
        pair_loads: Suite loads recorded by the stub ``load_benchmark``.
        model_inputs: ``x_obs`` recorded by the stub models.
        evaluator: The evaluator module under test.
        realign: Whether to pass the sidecar.
    """
    evaluator.main(_full_suite_args(evaluator, tmp_path, _pair_rows() if realign else None))
    expected = [
        _canonical_x() + offset if realign else ep.x_obs.numpy()
        for ep, offset in zip(_pair(), PAIR_OFFSETS, strict=True)
    ]
    n_models = 1 if evaluator is reference_table else 2  # the PFN has an int and an obs arm
    assert len(model_inputs) == n_models * len(expected)
    for seen, want in zip(model_inputs, expected * n_models, strict=True):
        np.testing.assert_array_equal(seen, want)
    assert _effect_accuracies(evaluator, tmp_path) == [0.0 if realign else 1.0] * n_models
    assert pair_loads == [(SUITE, "1.0.0")]
    _assert_provenance(tmp_path, "1.0.0", realign)


@FULL_SUITE
def test_full_suite_default_version_is_latest_and_the_resolved_one_is_recorded(
    tmp_path: Path, pair_loads: list, model_inputs: list, evaluator: Any
) -> None:
    """Without ``--version`` the registry's current version loads and lands in the JSON.

    Args:
        tmp_path: The pytest temporary directory.
        pair_loads: Suite loads recorded by the stub ``load_benchmark``.
        model_inputs: Keeps the models stubbed.
        evaluator: The evaluator module under test.
    """
    evaluator.main(_full_suite_args(evaluator, tmp_path, rows=None, version=None))
    assert pair_loads == [(SUITE, "latest")]
    _assert_provenance(tmp_path, _SUITE_REGISTRY[SUITE].version, realign=False)


@FULL_SUITE
def test_full_suite_a_sidecar_from_another_version_stops_the_run_before_any_model(
    tmp_path: Path, pair_loads: list, model_inputs: list, evaluator: Any
) -> None:
    """The 1.0.0 sidecar without ``--version`` fails instead of scoring the wrong suite.

    The registry serves 1.1.0 by default, which reuses the 1.0.0 episode ids
    with other targets. Rows used to be applied by id alone, which re-permuted
    the already canonical 1.1.0 columns without an error.

    Args:
        tmp_path: The pytest temporary directory.
        pair_loads: Suite loads recorded by the stub ``load_benchmark``.
        model_inputs: ``x_obs`` recorded by the stub models.
        evaluator: The evaluator module under test.
    """
    argv = _full_suite_args(evaluator, tmp_path, _pair_rows(y_true_regen=-0.5), version=None)
    with pytest.raises(ValueError, match=r"not built from this episode.*--version 1\.0\.0"):
        evaluator.main(argv)
    assert model_inputs == []
    assert not (tmp_path / "out.json").exists()


@FULL_SUITE
def test_full_suite_an_episode_without_a_row_stops_the_run_before_any_model(
    tmp_path: Path, pair_loads: list, model_inputs: list, evaluator: Any
) -> None:
    """An episode missing from the sidecar is an error, no longer scored unrealigned.

    Args:
        tmp_path: The pytest temporary directory.
        pair_loads: Suite loads recorded by the stub ``load_benchmark``.
        model_inputs: ``x_obs`` recorded by the stub models.
        evaluator: The evaluator module under test.
    """
    with pytest.raises(ValueError, match="no row for episode 8"):
        evaluator.main(_full_suite_args(evaluator, tmp_path, _pair_rows()[:1]))
    assert model_inputs == []
    assert not (tmp_path / "out.json").exists()


@FULL_SUITE
def test_full_suite_logs_the_realigned_episodes_not_the_sidecar_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pair_loads: list,
    model_inputs: list,
    evaluator: Any,
) -> None:
    """A row without an episode is ignored and not counted as realigned.

    ``dotime-eval-reference`` used to log the sidecar's size as the realigned count.

    Args:
        tmp_path: The pytest temporary directory.
        capsys: The pytest stdout capture fixture.
        pair_loads: Suite loads recorded by the stub ``load_benchmark``.
        model_inputs: Keeps the models stubbed.
        evaluator: The evaluator module under test.
    """
    rows = [*_pair_rows(), _row(_episode(scm_id=99))]
    evaluator.main(_full_suite_args(evaluator, tmp_path, rows))
    assert "realigned x_obs of 2 episodes with sidecar.jsonl" in capsys.readouterr().out
