"""The TabPFN evaluator reads its columns off each structure's DAG, not positions.

Until the fix, the back-door branch adjusted for every column other than the
treatment A and the outcome Y. On ``confounder_mediator`` (canonical columns
A, X, M, Y) that set holds the mediator M, a descendant of A, which blocks the
causal path A -> M -> Y. The front-door branch took the first such column as
its mediator, which on ``front_door`` (columns A, U, M, Y) is the hidden
confounder U. A stub regressor records what TabPFN would be fit on, so these
tests need no TabPFN install.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import numpy as np
import pytest
import torch

from dotime import baselines
from dotime.benchmarks import Episode
from dotime.interventions import InterventionSpec, InterventionType
from dotime.reference import tabpfn
from dotime.tscm_sampler import TSCMStructure

# Released x_obs column order of the structures TabPFN adjusts on. The frozen
# suites are laid out this way, so this is a contract, not an implementation
# detail.
LAYOUTS = {
    "back_door": ["A", "X", "Y"],
    "observed_confounder": ["A", "X", "Y"],
    "confounder_mediator": ["A", "X", "M", "Y"],
    "mediator": ["A", "M", "Y"],
    "front_door": ["A", "U", "M", "Y"],
}
ONSET = 40


def _episode(structure: str, query: int | None = None, n_vars: int | None = None) -> Episode:
    """A random trajectory in a structure's canonical layout with a hard do() on A.

    Args:
        structure: Structure label of the episode.
        query: Queried column. Defaults to the outcome Y, the last column.
        n_vars: Number of columns. Defaults to the structure's canonical count.

    Returns:
        A single-query episode with its intervention on column 0 at ``ONSET``.
    """
    n = len(LAYOUTS[structure]) if n_vars is None else n_vars
    x = torch.as_tensor(np.random.default_rng(0).normal(size=(60, n)), dtype=torch.float32)
    return Episode(
        x_obs=x,
        x_int=x.clone(),
        intervention=InterventionSpec(
            targets=[0], times=[ONSET], intervention_type=InterventionType.HARD, values=1.5
        ),
        y_true=torch.zeros(1),
        query_target=torch.tensor([n - 1 if query is None else query]),
        query_time=torch.tensor([ONSET]),
        structure=structure,
    )


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
    return recorded


@pytest.mark.parametrize("structure", sorted(LAYOUTS))
def test_canonical_layout_matches_the_released_columns(structure: str) -> None:
    """The graph helper names the columns in the order the suites store them.

    Args:
        structure: A structure TabPFN adjusts on.
    """
    names, _, hidden = baselines._canonical_summary_graph(structure)
    assert names == LAYOUTS[structure]
    assert hidden == ({"U"} if structure == "front_door" else set())


@pytest.mark.parametrize(
    ("structure", "legacy_matches"),
    [("back_door", True), ("observed_confounder", True), ("confounder_mediator", False)],
)
def test_back_door_set_is_the_confounder(structure: str, legacy_matches: bool) -> None:
    """The adjustment set is {X}, and differs from the positional rule only with a mediator.

    Args:
        structure: A back-door-family structure.
        legacy_matches: Whether the old rule (every column except A and Y)
            already gave {X}, so this structure's predictions must not change.
    """
    names = LAYOUTS[structure]
    n_vars, a, y, adjust = baselines._back_door_columns(structure)
    assert (n_vars, names[a], names[y]) == (len(names), "A", "Y")
    assert [names[c] for c in adjust] == ["X"]
    legacy = tuple(v for v in range(n_vars) if v not in (a, y))
    assert (adjust == legacy) is legacy_matches


@pytest.mark.parametrize(("structure", "legacy_pick"), [("mediator", "M"), ("front_door", "U")])
def test_front_door_mediator_is_m(structure: str, legacy_pick: str) -> None:
    """The mediator is M. The positional rule picked the hidden U on front_door.

    Args:
        structure: A front-door-family structure.
        legacy_pick: The variable the old rule (first column other than A and
            Y) selected as the mediator.
    """
    names = LAYOUTS[structure]
    n_vars, a, y, mediator = baselines._front_door_columns(structure)
    assert (n_vars, names[a], names[y], names[mediator]) == (len(names), "A", "Y", "M")
    legacy = next(v for v in range(n_vars) if v not in (a, y))
    assert names[legacy] == legacy_pick


@pytest.mark.parametrize(
    ("helper", "structure", "match"),
    [
        (baselines._front_door_columns, "back_door", "exactly one"),
        (baselines._back_door_columns, "no_such_structure", "no_such_structure"),
    ],
)
def test_column_helpers_reject_what_they_cannot_derive(
    helper: Callable[[str], tuple[int, ...]], structure: str, match: str
) -> None:
    """A structure without a mediator, or an unknown name, raises instead of guessing.

    Args:
        helper: The column helper under test.
        structure: A structure the helper cannot derive columns for.
        match: A fragment the error message must contain.
    """
    with pytest.raises(ValueError, match=match):
        helper(structure)


@pytest.mark.parametrize("structure", ["back_door", "observed_confounder", "confounder_mediator"])
def test_back_door_branch_fits_on_a_x_and_lagged_y(structure: str, fits: list) -> None:
    """TabPFN's outcome model sees [A_t, X_t, Y_(t-1)] and never the mediator.

    Args:
        structure: A back-door-family structure.
        fits: Designs and targets recorded by the stub regressor.
    """
    ep = _episode(structure)
    x = ep.x_obs.numpy()
    col = LAYOUTS[structure].index
    tabpfn.predict(ep)
    ((design, target),) = fits
    expected = np.column_stack(
        [x[1:ONSET, col("A")], x[1:ONSET, col("X")], x[0 : ONSET - 1, col("Y")]]
    )
    np.testing.assert_array_equal(design, expected)
    np.testing.assert_array_equal(target, x[1:ONSET, col("Y")])


@pytest.mark.parametrize("structure", ["mediator", "front_door"])
def test_front_door_branch_fits_on_the_mediator(structure: str, fits: list) -> None:
    """Both front-door models use M: ``M_t ~ A_t`` and ``Y_t ~ [M_t, A_t]``.

    Args:
        structure: A front-door-family structure.
        fits: Designs and targets recorded by the stub regressor.
    """
    ep = _episode(structure)
    x = ep.x_obs.numpy()
    col = LAYOUTS[structure].index
    tabpfn.predict(ep)
    (m_design, m_target), (y_design, y_target) = fits
    np.testing.assert_array_equal(m_design, x[1:ONSET, [col("A")]])
    np.testing.assert_array_equal(m_target, x[1:ONSET, col("M")])
    np.testing.assert_array_equal(
        y_design, np.column_stack([x[1:ONSET, col("M")], x[1:ONSET, col("A")]])
    )
    np.testing.assert_array_equal(y_target, x[1:ONSET, col("Y")])


@pytest.mark.parametrize("structure", ["confounder_mediator", "front_door"])
def test_non_outcome_query_takes_the_mean_fallback(structure: str, fits: list) -> None:
    """Roles are derived for the outcome, so a query of another column gets the mean.

    Args:
        structure: A structure whose adjustment would otherwise run.
        fits: Designs and targets recorded by the stub regressor.
    """
    ep = _episode(structure, query=LAYOUTS[structure].index("M"))
    expected = float(ep.x_obs[:ONSET, LAYOUTS[structure].index("M")].mean())
    assert tabpfn.predict(ep) == pytest.approx(expected)
    assert fits == []


def test_non_canonical_layout_raises(fits: list) -> None:
    """An episode whose column count contradicts its structure raises, not guesses.

    Args:
        fits: Designs and targets recorded by the stub regressor.
    """
    with pytest.raises(ValueError, match="canonical columns"):
        tabpfn.predict(_episode("confounder_mediator", n_vars=3))
    assert fits == []


def test_column_helpers_leave_global_rng_untouched() -> None:
    """Deriving columns builds a TSCMPrior, which must not draw random numbers."""
    baselines._back_door_columns.cache_clear()
    baselines._front_door_columns.cache_clear()
    torch_state = torch.get_rng_state()
    _, np_keys, np_pos, *_ = np.random.get_state()
    np_keys = np_keys.copy()
    for structure in TSCMStructure:
        baselines._back_door_columns(structure.value)
        # Most structures have no unique mediator. The call must still leave
        # the RNG alone on its way to raising.
        with contextlib.suppress(ValueError):
            baselines._front_door_columns(structure.value)
    _, keys_after, pos_after, *_ = np.random.get_state()
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_keys, keys_after)
    assert np_pos == pos_after
