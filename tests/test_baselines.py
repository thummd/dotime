"""BackDoorOLS adjusts for each structure's back-door set, never for a descendant of A.

Until the fix it adjusted for every column other than the treatment and the
outcome. On ``confounder_mediator`` (canonical columns A, X, M, Y) that set holds
the mediator M, which blocks the only causal path A -> M -> Y, and the baseline
scored 0.499 effect-sign accuracy there on dot-Identifiability-v1 1.1.0.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dotime import baselines
from dotime.benchmarks import Episode
from dotime.extended import TSCMPrior
from dotime.interventions import InterventionSpec, InterventionType
from dotime.tscm_sampler import TSCMStructure


def _canonical_names(structure: str) -> list[str]:
    """Variable names in the released column order of a named structure.

    Args:
        structure: A ``TSCMStructure`` value.

    Returns:
        Names such as ``["A", "X", "M", "Y"]``, one per ``x_obs`` column.
    """
    prior = TSCMPrior(TSCMStructure(structure))
    topo = prior.sampler._build_dag().topo_order
    return [topo[t] for t in prior.canonical_perm]


def _episode(x: np.ndarray, structure: str, onset: int, value: float) -> Episode:
    """Wrap a trajectory as a single-query episode with a hard do() at ``onset``.

    Args:
        x: Observational trajectory of shape ``(T, N)``, canonical columns.
        structure: Structure label of the episode.
        onset: Intervention time, which also ends the pre-intervention fit window.
        value: The do-value of the treatment (column 0).

    Returns:
        An episode whose single query targets the last column at ``onset``.
    """
    x_t = torch.as_tensor(x, dtype=torch.float32)
    return Episode(
        x_obs=x_t,
        x_int=x_t.clone(),
        intervention=InterventionSpec(
            targets=[0], times=[onset], intervention_type=InterventionType.HARD, values=value
        ),
        y_true=torch.zeros(1),
        query_target=torch.tensor([x.shape[1] - 1]),
        query_time=torch.tensor([onset / x.shape[0]]),
        structure=structure,
    )


@pytest.mark.parametrize(
    ("structure", "legacy_set_was_valid"),
    [("back_door", True), ("observed_confounder", True), ("confounder_mediator", False)],
)
def test_back_door_set_is_the_confounder(structure: str, legacy_set_was_valid: bool) -> None:
    """The adjustment set is {X} and excludes every descendant of A.

    Args:
        structure: A back-door-family structure.
        legacy_set_was_valid: Whether the old rule (every column except A and Y)
            already gave {X}. It did for back_door and observed_confounder, so
            their predictions must not change.
    """
    names = _canonical_names(structure)
    n_vars, a, y, adjust = baselines._back_door_columns(structure)
    assert n_vars == len(names)
    assert (names[a], names[y]) == ("A", "Y")
    assert [names[c] for c in adjust] == ["X"]
    legacy = tuple(v for v in range(n_vars) if v not in (a, y))
    assert (adjust == legacy) is legacy_set_was_valid


def test_confounder_mediator_recovers_the_mediated_effect() -> None:
    """The fitted A-coefficient is the total effect through M, not zero.

    Linear SCM with canonical columns [A, X, M, Y]: X -> A -> M -> Y plus
    X(t-1) -> Y(t) and Y(t-1) -> Y(t). The total effect of A_t on Y_t is
    1.0 * 1.0 = 1.0. A carries no autoregression here, so {X_t, Y_(t-1)} also
    blocks every back-door path of the time-unrolled graph. Adjusting for M
    as well would drive the coefficient to zero.
    """
    rng = np.random.default_rng(0)
    t_len = 3000
    x = np.zeros((t_len, 4))
    for t in range(1, t_len):
        x_t = 0.5 * x[t - 1, 1] + rng.normal()
        a_t = 0.8 * x_t + rng.normal()
        m_t = 1.0 * a_t + 0.3 * rng.normal()
        y_t = 1.0 * m_t + 0.5 * x[t - 1, 1] + 0.3 * x[t - 1, 3] + 0.3 * rng.normal()
        x[t] = (a_t, x_t, m_t, y_t)
    model = baselines.get("BackDoorOLS")
    # Predictions are affine in the do-value, so their difference over a unit
    # step in v is exactly the fitted A-coefficient.
    lo = model.predict(_episode(x, "confounder_mediator", t_len - 1, 0.0))
    hi = model.predict(_episode(x, "confounder_mediator", t_len - 1, 1.0))
    assert float(hi - lo) == pytest.approx(1.0, abs=0.1)


def test_back_door_rejects_a_non_canonical_layout() -> None:
    """A back-door-family episode with the wrong column count raises, not guesses."""
    x = np.zeros((50, 3))  # confounder_mediator has four canonical columns
    with pytest.raises(ValueError, match="canonical columns"):
        baselines.get("BackDoorOLS").predict(_episode(x, "confounder_mediator", 40, 1.0))


def test_back_door_columns_leave_global_rng_untouched() -> None:
    """Deriving the columns builds a TSCMPrior, which must not draw random numbers."""
    baselines._back_door_columns.cache_clear()
    torch_state = torch.get_rng_state()
    _, np_keys, np_pos, *_ = np.random.get_state()
    np_keys = np_keys.copy()
    for structure in TSCMStructure:
        baselines._back_door_columns(structure.value)
    _, keys_after, pos_after, *_ = np.random.get_state()
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(np_keys, keys_after)
    assert np_pos == pos_after


# --------------------------------------------------------------------------- #
# LaggedBackDoorOLS: the lag-one adjustment set of the unrolled graph
# --------------------------------------------------------------------------- #


def _order_one_scm(effect: float, x_inst: float, seed: int = 0, t_len: int = 4000) -> np.ndarray:
    """Linear order-one SCM in the canonical columns ``[A, X, Y]`` of the back-door family.

    ``X`` is autoregressive, ``A`` depends on ``X_t`` and on ``A_{t-1}``, and
    ``Y`` depends on ``A_t`` (by ``effect``), ``X_t`` (by ``x_inst``),
    ``X_{t-1}`` and ``Y_{t-1}``. With ``effect = x_inst = 0`` it is the
    ``observed_confounder`` structure, otherwise ``back_door``. Either way the
    unrolled graph holds the path ``A_t <- A_{t-1} <- X_{t-1} -> Y_t``, which
    the contemporaneous set ``{X_t, Y_{t-1}}`` leaves open.

    Args:
        effect: Coefficient of ``A_t`` in ``Y_t``.
        x_inst: Coefficient of ``X_t`` in ``Y_t``.
        seed: Seed of the noise.
        t_len: Number of rows.

    Returns:
        Trajectory of shape ``(t_len, 3)``.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((t_len, 3))
    for t in range(1, t_len):
        x_t = 0.5 * x[t - 1, 1] + rng.normal()
        a_t = 0.8 * x_t + 0.6 * x[t - 1, 0] + rng.normal()
        y_t = effect * a_t + x_inst * x_t + 0.7 * x[t - 1, 1] + 0.5 * x[t - 1, 2] + 0.3 * rng.normal()
        x[t] = (a_t, x_t, y_t)
    return x


def _unit_step(name: str, x: np.ndarray, structure: str) -> float:
    """The change of a baseline's prediction over a unit step of the do-value.

    Args:
        name: A registered baseline.
        x: Trajectory in canonical columns.
        structure: Structure label of the episode.

    Returns:
        ``pred(do 1) - pred(do 0)`` queried at the last row, i.e. the fitted
        treatment coefficient of an estimator that is affine in the do-value.
    """
    model = baselines.get(name)
    onset = x.shape[0] - 1
    lo = model.predict(_episode(x, structure, onset, 0.0))
    hi = model.predict(_episode(x, structure, onset, 1.0))
    return float(hi - lo)


def test_lagged_back_door_closes_the_lag_one_path() -> None:
    """On the null structure the contemporaneous set reports an effect and the lagged set does not."""
    x = _order_one_scm(effect=0.0, x_inst=0.0)
    assert abs(_unit_step("BackDoorOLS", x, "observed_confounder")) > 0.1
    assert abs(_unit_step("LaggedBackDoorOLS", x, "observed_confounder")) < 0.05
    assert abs(_unit_step("LaggedBackDoorOLS-L3", x, "observed_confounder")) < 0.05


def test_lagged_back_door_recovers_the_effect_under_autoregression() -> None:
    """With a unit effect the lagged set recovers it, the contemporaneous set overshoots."""
    x = _order_one_scm(effect=1.0, x_inst=0.5)
    assert _unit_step("LaggedBackDoorOLS", x, "back_door") == pytest.approx(1.0, abs=0.05)
    assert _unit_step("BackDoorOLS", x, "back_door") - 1.0 > 0.03


def test_lagged_back_door_declines_like_its_parent() -> None:
    """Outside the back-door family both variants predict the pre-onset mean, whatever the do-value."""
    x = np.random.default_rng(1).normal(size=(300, 4))  # front_door columns [A, U, M, Y]
    for name in ("LaggedBackDoorOLS", "LaggedBackDoorOLS-L3"):
        model = baselines.get(name)
        lo = float(model.predict(_episode(x, "front_door", 250, 0.0)))
        hi = float(model.predict(_episode(x, "front_door", 250, 1.0)))
        assert lo == hi
        assert lo == pytest.approx(x[:250, 3].mean(), abs=1e-4)


def test_lagged_back_door_falls_back_on_short_windows() -> None:
    """With fewer rows than coefficients the estimator predicts the mean instead of a ridge guess."""
    x = _order_one_scm(effect=1.0, x_inst=0.5, t_len=12)
    # Three lags on [A, X, Y] need 13 coefficients, and an onset at 10 leaves 7 rows.
    pred = baselines.get("LaggedBackDoorOLS-L3").predict(_episode(x, "back_door", 10, 5.0))
    assert float(pred) == pytest.approx(x[:10, 2].mean(), abs=1e-4)


def test_lagged_back_door_registry_names_and_lags() -> None:
    """Both variants are registered, carry their lag, and reject a lag below one."""
    assert {"LaggedBackDoorOLS", "LaggedBackDoorOLS-L3"} <= set(baselines.available())
    assert baselines.get("LaggedBackDoorOLS").lag == 1
    three = baselines.get("LaggedBackDoorOLS-L3")
    assert (three.lag, three.name) == (3, "LaggedBackDoorOLS-L3")
    with pytest.raises(ValueError, match="lag >= 1"):
        baselines.LaggedBackDoorOLSBaseline(lag=0)
