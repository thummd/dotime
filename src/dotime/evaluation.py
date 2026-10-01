"""Evaluation harness for DoTime benchmark suites.

This module ports the metric functions and aggregation helpers from the
Do-Over-Time-PFN evaluation code (``dotime/eval/metrics.py`` and the
``scripts/tscm_identifiability.py`` reference harness) into a single
dependency-light surface (torch + numpy only — R² is computed directly rather
than via scikit-learn so it stays in the core install).

**Public surface**

- metric functions: :func:`compute_rmse`, :func:`compute_mae`,
  :func:`compute_nmse`, :func:`compute_r2`.
- :func:`direction_accuracy` — sign-consistent accuracy, near-zero targets excluded.
- :func:`bootstrap_ci` — bootstrap mean/std/CI over per-sample values.
- :func:`check_shared_noise`, :func:`resolve_dir_target` and
  :func:`describe_dir_target` — what direction accuracy scores when the
  default ``"auto"`` is in force (see :data:`DEFAULT_DIR_TARGET`).
- :func:`evaluate` — run a baseline over a suite, aggregating pooled and
  per-structure metrics.
- :class:`Results` — holds the aggregated metrics with ``.summary()`` and
  ``.to_dict()``.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from dotime.observation import impute_episode

if TYPE_CHECKING:
    from dotime.baselines import Baseline
    from dotime.benchmarks import BenchmarkSuite, Episode

__all__ = [
    "DEFAULT_DIR_TARGET",
    "DIR_ACC_EPS",
    "DIR_TARGETS",
    "DIR_TARGET_MODES",
    "NONFINITE_MODES",
    "NoiseCheck",
    "Results",
    "add_dir_target_argument",
    "bootstrap_ci",
    "check_shared_noise",
    "compute_mae",
    "compute_nmse",
    "compute_r2",
    "compute_rmse",
    "describe_dir_target",
    "direction_accuracy",
    "evaluate",
    "resolve_dir_target",
]

_LOG = logging.getLogger(__name__)

# Near-zero targets are ambiguous for sign-based direction accuracy and are
# excluded from that metric (reported separately).
DIR_ACC_EPS = 0.1

# What the direction-accuracy sign test can score (see DEFAULT_DIR_TARGET).
DIR_TARGETS = ("level", "effect")

# What a caller may ask for: a target of DIR_TARGETS, or "auto", which
# resolve_dir_target turns into one of them from the scored episodes.
DIR_TARGET_MODES = ("auto", *DIR_TARGETS)

# What evaluate() does with a non-finite prediction. "raise" names the baseline
# and episode, because a NaN would otherwise poison every pooled metric.
# "exclude" leaves it out of the level metrics, which need a number, but still
# scores it as a wrong sign, so abstaining never raises direction accuracy.
NONFINITE_MODES = ("raise", "exclude")

# The one switch for what direction accuracy scores by default. Set it to one of:
#
#   "auto"    "effect" when the two arms of every scored episode share their
#             noise, "level" (with a logged warning) when they do not; see
#             check_shared_noise. Shared noise makes y_true - y_obs the
#             episode's counterfactual effect: dot-Identifiability-v1 from
#             1.1.0, dot-Continuous-v1 and every 2026-10 suite. The v1.0.0
#             discrete suites (Identifiability, RegimeSwitch, Generic-100k)
#             draw the interventional arm with its own noise, so there
#             y_true - y_obs adds a second noise draw to the effect, and the
#             level keeps the v1 protocol.
#   "level"   sign(y_pred) vs sign(y_true): the sign of the interventional
#             level. The v1 paper protocol; reproduces the published v1 tables.
#             A positive level can come from a negative effect on a positive
#             baseline, so this mostly rewards predicting where the series
#             already sits, not the direction of the intervention.
#   "effect"  sign(y_pred - y_obs) vs sign(y_true - y_obs), with y_obs the
#             observational level at the query: the sign of the causal effect,
#             i.e. whether the model gets the intervention's direction right.
#             evaluate() refuses it on the misaligned dot-Identifiability-v1
#             1.0.0 files (use dotime-eval-reference --realignment for those).
#
# Only direction accuracy changes; RMSE, MAE, NMSE and R^2 are the same either
# way. evaluate(), Results, dotime.qa.target_qa and the --dir-target option of
# dotime-benchmark, dotime-eval-submission, dotime-eval-reference,
# dotime-eval-pfn, dotime-eval-tabpfn and dotime-eval-chronos all read this
# line. A single call or run can still override it with dir_target= /
# --dir-target, and tests/test_smoke.py fails if a default is hard-coded
# anywhere else. Every result records the target it scored ("level" or
# "effect"), the requested mode and whether the pairs share their noise, and
# reports both scores wherever the effect is a counterfactual effect.
DEFAULT_DIR_TARGET = "auto"


# --------------------------------------------------------------------------- #
# Pointwise metrics
# --------------------------------------------------------------------------- #


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Root mean squared error."""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean absolute error."""
    return torch.mean(torch.abs(predictions - targets)).item()


def compute_nmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Normalized MSE: ``MSE / Var(targets)``.

    Equals 1.0 for a predict-the-mean baseline, <1.0 when better, >1.0 worse.
    Returns NaN when there are fewer than two targets or the variance is ~0.
    """
    if targets.numel() < 2:
        return float("nan")
    mse = torch.mean((predictions - targets) ** 2)
    var = torch.var(targets, unbiased=False)
    if var < 1e-8:
        return float("nan")
    return (mse / var).item()


def compute_r2(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Coefficient of determination, ``1 - SS_res / SS_tot``.

    Computed directly (no scikit-learn) so it stays in the core install.
    Returns NaN when the target variance is ~0.
    """
    targets = targets.float()
    predictions = predictions.float()
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return (1.0 - ss_res / ss_tot).item()


def realign_episode(episode, canonical_perm, hidden_canonical=()):
    """Return a copy of an episode with its ``x_obs`` columns realigned.

    Repairs the archived ``dot-Identifiability-v1`` (v1.0.0) episodes, whose
    released ``x_obs`` is in topological order while ``x_int``/``query_target``
    are canonical, and whose hidden variables were not zeroed (v1 erratum).

    Args:
        episode: The episode to repair.
        canonical_perm: ``canonical_idx -> topo_idx`` permutation from the
            realignment sidecar.
        hidden_canonical: Canonical indices of hidden variables to zero out.

    Returns:
        A new :class:`~dotime.benchmarks.Episode` with realigned ``x_obs`` and,
        when the episode has one, ``obs_mask`` permuted with it; every other
        field is shared with the input episode.
    """
    import dataclasses

    perm = torch.as_tensor(list(canonical_perm), dtype=torch.long)
    x = episode.x_obs.index_select(1, perm).clone()
    for h in hidden_canonical:
        x[:, int(h)] = 0.0
    # The mask describes x_obs cell by cell, so it must move with its columns.
    # Zeroed hidden columns keep their entries: hidden variables are stored as
    # zeros, not as missing values.
    mask = episode.obs_mask
    if mask is not None:
        mask = mask.index_select(1, perm).clone()
    return dataclasses.replace(episode, x_obs=x, obs_mask=mask)


def query_obs_levels(episode) -> torch.Tensor:
    """Observational level of the queried variable at each query's row.

    Used to score direction accuracy on the *causal effect*
    (``y_true - y_obs``) instead of the interventional level: subtracting the
    same observational level from prediction and target leaves RMSE unchanged
    but makes the sign test measure the effect direction.

    The row comes from :attr:`~dotime.benchmarks.Episode.query_time_idx`,
    which resolves each suite's declared ``query_time`` encoding. The suites
    disagree (``dot-Identifiability-v1`` stores ``index / T``,
    ``dot-Continuous-v1`` stores ``index / (T - 1)``), so a fraction cannot be
    decoded here without knowing which suite wrote it.

    An episode from the observation layer (:mod:`dotime.observation`) records
    the latent level as ``metadata["y_obs_latent"]``, and that is what this
    returns: its ``x_obs`` cell at the query is a noisy measurement, while the
    effect is defined on the latent values, like ``y_true``.

    Args:
        episode: A benchmark :class:`~dotime.benchmarks.Episode`.

    Returns:
        Tensor of shape ``(n_queries,)`` with ``metadata["y_obs_latent"]``
        when recorded, otherwise ``x_obs[query_time_idx, query_target]`` per
        query.

    Raises:
        ValueError: If the episode records query rows, or latent levels, that
            do not match its queries (see
            :attr:`~dotime.benchmarks.Episode.query_time_idx`).

    .. warning::
        For the archived ``dot-Identifiability-v1`` (v1.0.0) files this reads a
        possibly *misaligned* column: the released ``x_obs`` is in topological
        order while ``query_target`` is canonical (v1 erratum). Use the
        released realignment sidecar for that suite; later suite versions and
        ``dot-Continuous-v1`` are correctly aligned.
    """
    latent = episode.metadata.get("y_obs_latent")
    if latent is not None:
        levels = torch.as_tensor(latent, dtype=torch.float32).reshape(-1)
        n_queries = episode.query_target.numel()
        if levels.numel() != n_queries:
            raise ValueError(
                f"episode {episode.scm_id} records {levels.numel()} y_obs_latent values "
                f"for {n_queries} queries"
            )
        return levels
    rows = episode.query_time_idx
    cols = episode.query_target.reshape(-1).long()
    return episode.x_obs[rows, cols].to(torch.float32)


def direction_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = DIR_ACC_EPS
) -> dict[str, float | int]:
    """Sign-consistent direction accuracy, excluding near-zero targets.

    A query is scored when ``|target| >= eps``. A target closer to zero has no
    reliable sign, so it is excluded and counted in ``n_excluded``. A scored
    query is right when ``sign(pred) == sign(target)``. A prediction with sign
    0 (exactly zero) or a non-finite prediction has no sign to match and
    counts as wrong.

    Args:
        preds: Predictions.
        targets: Targets, the same shape as ``preds``.
        eps: Threshold on ``|target|`` below which a query is excluded.

    Returns:
        Dict with ``accuracy`` (the fraction of scored queries whose sign
        matches, NaN when none is scored), ``n_valid`` (scored queries) and
        ``n_excluded``.
    """
    if preds.numel() == 0:
        return {"accuracy": float("nan"), "n_valid": 0, "n_excluded": 0}
    mask = targets.abs() >= eps
    n_valid = int(mask.sum().item())
    n_excluded = int(preds.numel() - n_valid)
    if n_valid == 0:
        return {"accuracy": float("nan"), "n_valid": 0, "n_excluded": n_excluded}
    acc = (preds[mask].sign() == targets[mask].sign()).float().mean().item()
    return {"accuracy": acc, "n_valid": n_valid, "n_excluded": n_excluded}


def bootstrap_ci(
    values: Iterable[float], n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float, float]:
    """Bootstrap ``(mean, std, ci_low, ci_high)`` over per-sample values.

    Uses the percentile method at confidence ``1 - alpha``. Returns NaNs for an
    empty input; a degenerate ``(v, 0, v, v)`` for a single value.
    """
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, 0.0, v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    boot_means = arr[idx].mean(axis=1)
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(arr.mean()), float(arr.std()), ci_low, ci_high


# --------------------------------------------------------------------------- #
# What direction accuracy scores under "auto"
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NoiseCheck:
    """Whether the two arms of a set of episodes share their noise.

    Arms that share one noise realisation are bit-identical before the
    intervention onset, while an interventional arm drawn with its own noise
    differs there almost surely. Pre-onset agreement therefore tells the two
    pairings apart from the episodes alone, with no metadata. On the released
    suites the separation is complete: the v1.0.0 discrete suites agree in none
    of their checked episodes, and every shared-noise suite in all of them.

    Attributes:
        n_episodes: Episodes inspected.
        n_checked: Episodes with at least one pre-onset row and no zeroed arm.
        n_shared: Checked episodes whose arms agree before the onset.
        n_zeroed: Episodes skipped because an arm is all zero, the build's
            mark of a diverged simulation, which says nothing about the noise.
        n_no_history: Episodes skipped because the intervention starts at the
            first row.
    """

    n_episodes: int
    n_checked: int
    n_shared: int
    n_zeroed: int
    n_no_history: int

    @property
    def shared(self) -> bool:
        """Whether at least one episode was checked and every checked one agrees.

        Returns:
            The verdict that :func:`resolve_dir_target` acts on.
        """
        return self.n_checked > 0 and self.n_shared == self.n_checked


def _arms_agree(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-for-bit equality that counts NaN cells in the same places as equal.

    Args:
        a: One arm's rows.
        b: The other arm's rows.

    Returns:
        Whether the two blocks are identical, missing cells included.
    """
    if a.shape != b.shape:
        return False
    # An observed suite masks the same cells in both arms, and NaN != NaN, so
    # the masks are compared first and the values only where both are present.
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    return bool(torch.equal(nan_a, nan_b) and torch.equal(a[~nan_a], b[~nan_b]))


def _all_zero(x: torch.Tensor) -> bool:
    """Whether every present cell of an arm is zero.

    Args:
        x: One arm, possibly with missing (NaN) cells.

    Returns:
        True for an arm the build zeroed after a diverged simulation.
    """
    return not bool(torch.any(torch.nan_to_num(x) != 0))


def check_shared_noise(episodes: Iterable[Episode]) -> NoiseCheck:
    """Check whether the two arms of every episode share their noise.

    Args:
        episodes: The episodes about to be scored, e.g. a
            :class:`~dotime.benchmarks.BenchmarkSuite`.

    Returns:
        The counts, with :attr:`NoiseCheck.shared` as the verdict.
    """
    n = checked = shared = zeroed = no_history = 0
    for ep in episodes:
        n += 1
        if _all_zero(ep.x_obs) or _all_zero(ep.x_int):
            zeroed += 1
            continue
        times = list(ep.intervention.times)
        onset = int(min(times)) if times else 0
        if onset <= 0:
            no_history += 1
            continue
        checked += 1
        shared += _arms_agree(ep.x_obs[:onset], ep.x_int[:onset])
    return NoiseCheck(n, checked, shared, zeroed, no_history)


def describe_dir_target(mode: str, target: str, noise: NoiseCheck | None = None) -> str:
    """One line saying what direction accuracy scores and why.

    Args:
        mode: The requested mode, one of :data:`DIR_TARGET_MODES`.
        target: The resolved target, ``"level"`` or ``"effect"``.
        noise: The :func:`check_shared_noise` result behind an ``"auto"`` choice.

    Returns:
        A sentence for logs and command-line output.
    """
    what = f"direction accuracy scores the sign of the {target}"
    if mode != "auto":
        return f"{what} (requested)"
    if noise is None:
        return f"{what} (auto)"
    detail = (
        f"the arms of {noise.n_shared:,} of {noise.n_checked:,} checked episodes "
        "agree before the onset"
    )
    skipped = []
    if noise.n_zeroed:
        skipped.append(f"{noise.n_zeroed:,} with a zeroed arm")
    if noise.n_no_history:
        skipped.append(f"{noise.n_no_history:,} without pre-onset rows")
    if skipped:
        detail += " (" + " and ".join(skipped) + " skipped)"
    if target == "level":
        detail += (
            ", so y_true - y_obs is not a counterfactual effect and the level keeps the v1 protocol"
        )
    return f"{what} (auto: {detail})"


def resolve_dir_target(
    dir_target: str, noise: NoiseCheck | None = None, *, warn: bool = True
) -> str:
    """Turn a direction-target mode into the target that is scored.

    Args:
        dir_target: One of :data:`DIR_TARGET_MODES`.
        noise: :func:`check_shared_noise` of the scored episodes. Only
            ``"auto"`` reads it.
        warn: Log a warning when ``"auto"`` falls back to the level, so a run
            that cannot score the effect says so. Command-line tools print the
            same sentence themselves and pass ``False``.

    Returns:
        ``"level"`` or ``"effect"``.

    Raises:
        ValueError: If ``dir_target`` is not a mode, or ``"auto"`` comes
            without a ``noise`` check.
    """
    if dir_target not in DIR_TARGET_MODES:
        raise ValueError(f"dir_target must be one of {DIR_TARGET_MODES}, got {dir_target!r}")
    if dir_target != "auto":
        return dir_target
    if noise is None:
        raise ValueError('dir_target="auto" needs check_shared_noise() of the scored episodes')
    if noise.shared:
        return "effect"
    if warn:
        _LOG.warning(describe_dir_target("auto", "level", noise))
    return "level"


# --------------------------------------------------------------------------- #
# Aggregated results container
# --------------------------------------------------------------------------- #


@dataclass
class Results:
    """Aggregated evaluation results for one baseline on one suite."""

    suite: str
    baseline: str
    n_episodes: int
    n_queries: int
    pooled: dict[str, float]
    per_structure: dict[str, dict[str, float]] = field(default_factory=dict)
    # The target dir_acc scored ("level" or "effect"); "auto" only on a
    # Results built by hand, since evaluate() always records what it resolved.
    dir_target: str = DEFAULT_DIR_TARGET
    dir_target_mode: str = DEFAULT_DIR_TARGET
    pairs_share_noise: bool | None = None

    def to_dict(self) -> dict:
        """JSON-serializable view of the results."""
        return {
            "suite": self.suite,
            "baseline": self.baseline,
            "n_episodes": self.n_episodes,
            "n_queries": self.n_queries,
            "dir_target": self.dir_target,
            "dir_target_mode": self.dir_target_mode,
            "pairs_share_noise": self.pairs_share_noise,
            "pooled": self.pooled,
            "per_structure": self.per_structure,
        }

    def summary(self) -> str:
        """Human-readable results table."""
        lines = [
            f"Suite:    {self.suite}",
            f"Baseline: {self.baseline}",
            f"Episodes: {self.n_episodes}   Queries: {self.n_queries}",
            f"dir_acc scores the sign of the {self.dir_target} (mode: {self.dir_target_mode})",
        ]
        if self.pairs_share_noise is False:
            lines.append(
                "The arms do not share their noise, so y_true - y_obs is not a "
                "counterfactual effect."
            )
        if "n_nonfinite" in self.pooled:
            lines.append(
                f"Non-finite predictions: {self.pooled['n_nonfinite']} "
                "(left out of the level metrics, wrong for dir_acc)"
            )
        lines.append("")
        cols = ["rmse", "mae", "nmse", "r2", "dir_acc", "dir_acc_se", "dir_acc_level"]
        if "dir_acc_effect" in self.pooled:
            cols.append("dir_acc_effect")
        # Short headers keep the 11-character columns aligned.
        names = {"dir_acc_level": "acc_level", "dir_acc_effect": "acc_effect"}
        header = f"{'group':<22}" + "".join(f"{names.get(c, c):>11}" for c in cols)
        lines.append(header)
        lines.append("-" * len(header))

        def _row(name: str, m: dict[str, float]) -> str:
            cells = []
            for c in cols:
                v = m.get(c, float("nan"))
                cells.append(f"{v:>11.4f}" if isinstance(v, (int, float)) else f"{v:>11}")
            return f"{name:<22}" + "".join(cells)

        lines.append(_row("pooled", self.pooled))
        for struct in sorted(self.per_structure):
            lines.append(_row(struct, self.per_structure[struct]))
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #

_DEFAULT_METRICS: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "rmse": compute_rmse,
    "mae": compute_mae,
    "nmse": compute_nmse,
    "r2": compute_r2,
}


def add_dir_target_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--dir-target`` option, defaulting to :data:`DEFAULT_DIR_TARGET`.

    Every command-line tool that scores direction accuracy takes its option
    from here, so none of them can drift from :func:`evaluate`.

    Args:
        parser: The parser to add the option to.
    """
    parser.add_argument(
        "--dir-target",
        choices=DIR_TARGET_MODES,
        default=DEFAULT_DIR_TARGET,
        help="What direction accuracy scores: the sign of the interventional level "
        "(the v1 protocol), of the causal effect y - y_obs, or 'auto', the effect "
        "when the two arms of every episode share their noise and the level "
        f"otherwise (default: {DEFAULT_DIR_TARGET}).",
    )


def _aggregate(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metrics,
    obs: torch.Tensor | None = None,
    count_nonfinite: bool = False,
    *,
    dir_target: str,
) -> dict[str, float]:
    """Level metrics and direction accuracy of one group of queries.

    Args:
        preds: Predictions, one per query.
        targets: Targets aligned with ``preds``.
        metrics: Level-space metrics by name.
        obs: Observational levels. When given, the effect is scored too
            (``preds - obs`` against ``targets - obs``).
        count_nonfinite: Report ``n_nonfinite``, the number of non-finite
            predictions (``nonfinite="exclude"``).
        dir_target: The target that fills ``dir_acc``, ``dir_n_valid`` and
            ``dir_acc_se``. ``"effect"`` needs ``obs``.

    Returns:
        Metric name to value, plus the direction-accuracy fields of the level,
        of the effect when ``obs`` is given, and of ``dir_target`` unprefixed.
    """
    finite = torch.isfinite(preds)
    if bool(finite.all()):
        out = {name: fn(preds, targets) for name, fn in metrics.items()}
    else:
        # Level metrics need a number: they run over the finite predictions,
        # while direction accuracy below still scores the others as wrong.
        out = {name: fn(preds[finite], targets[finite]) for name, fn in metrics.items()}
    # Subtracting the same observational level from both sides leaves every
    # level metric above unchanged but turns the sign of the level into the
    # sign of the effect, so one prediction pass yields both scores.
    scores = {"level": direction_accuracy(preds, targets)}
    if obs is not None:
        scores["effect"] = direction_accuracy(preds - obs, targets - obs)
    for name, da in scores.items():
        # The suites score one query per episode, so the binomial standard
        # error is exact (no clustering). ``n_valid`` excludes near-zero
        # targets, which carry no sign to score.
        n_valid = int(da["n_valid"])
        p = da["accuracy"]
        out[f"dir_acc_{name}"] = p
        out[f"dir_n_valid_{name}"] = n_valid
        out[f"dir_acc_se_{name}"] = (
            math.sqrt(p * (1.0 - p) / n_valid) if n_valid > 0 and p == p else float("nan")
        )
    out["dir_acc"] = out[f"dir_acc_{dir_target}"]
    out["dir_n_valid"] = out[f"dir_n_valid_{dir_target}"]
    out["dir_acc_se"] = out[f"dir_acc_se_{dir_target}"]
    if count_nonfinite:
        out["n_nonfinite"] = int((~finite).sum())
    return out


def evaluate(
    model: Baseline,
    suite: BenchmarkSuite,
    metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] | None = None,
    dir_target: str = DEFAULT_DIR_TARGET,
    *,
    impute: bool = True,
    nonfinite: str = "raise",
) -> Results:
    """Evaluate a baseline over every episode of a suite.

    Calls ``model.predict(episode)`` for each episode, pools predictions and
    ground-truth targets across all queries, and reports pooled and
    per-structure metrics.

    Episodes of an observed suite (:mod:`dotime.observation`) have missing
    (``NaN``) cells. A model that cannot read them gets the episode through
    :func:`dotime.observation.impute_episode`, which returns a finite episode
    unchanged, so imputation never alters a latent suite's results. A model
    whose ``mask_aware`` attribute is true gets the ``NaN`` cells as they are.

    Args:
        model: The baseline to evaluate.
        suite: The benchmark suite.
        metrics: Level-space metrics by name. Defaults to RMSE, MAE, NMSE, R^2.
        dir_target: What ``dir_acc`` scores: ``"level"`` (the sign of the
            interventional level, the v1 protocol), ``"effect"`` (the sign of
            ``y - y_obs`` at the query, read with :func:`query_obs_levels`), or
            ``"auto"``, the effect when the arms of every episode share their
            noise (:func:`check_shared_noise`) and the level otherwise. The
            level metrics are the same either way, and both direction scores are
            reported wherever the effect is a counterfactual effect. Defaults to
            :data:`DEFAULT_DIR_TARGET`.
        impute: Impute missing cells for models that are not ``mask_aware``.
        nonfinite: ``"raise"`` stops at the first non-finite prediction.
            ``"exclude"`` leaves non-finite predictions out of the level
            metrics, scores them as wrong in ``dir_acc`` and reports their
            number as ``n_nonfinite`` in the pooled and per-structure metrics.

    Returns:
        The pooled and per-structure metrics, with the scored target, the
        requested mode and the noise verdict recorded on the result.

    Raises:
        ValueError: If ``dir_target`` or ``nonfinite`` is unknown, if the model
            returns the wrong number of predictions or, with
            ``nonfinite="raise"``, a non-finite one, or if ``"effect"`` is asked
            of the archived ``dot-Identifiability-v1`` 1.0.0 files, whose
            ``x_obs`` columns are misaligned. Score those with
            ``dotime-eval-reference --dir-target effect --realignment <sidecar>``.
    """
    metrics = metrics or _DEFAULT_METRICS
    if dir_target not in DIR_TARGET_MODES:
        raise ValueError(f"dir_target must be one of {DIR_TARGET_MODES}, got {dir_target!r}")
    if nonfinite not in NONFINITE_MODES:
        raise ValueError(f"nonfinite must be one of {NONFINITE_MODES}, got {nonfinite!r}")
    noise = check_shared_noise(suite)
    misaligned = suite.meta.name == "dot-Identifiability-v1" and suite.meta.version == "1.0.0"
    # The archived 1.0.0 x_obs cannot give y_obs, so "auto" scores the level
    # there whatever the arms look like; only an explicit "effect" is refused.
    scored = (
        "level" if dir_target == "auto" and misaligned else resolve_dir_target(dir_target, noise)
    )
    if scored == "effect" and misaligned:
        raise ValueError(
            "dot-Identifiability-v1 1.0.0 ships x_obs in topological order, so y_obs "
            "cannot be read from it directly. Load 1.1.0, or score 1.0.0 with "
            "dotime-eval-reference --dir-target effect --realignment "
            "results/reference/dot-Identifiability-v1.0.0_realignment.jsonl"
        )
    # The effect is scored where it is a counterfactual effect, and wherever it
    # is asked for. On independent-noise twins it would mostly measure the
    # second noise draw, so "auto" and "level" leave it out there.
    effect = scored == "effect" or (noise.shared and not misaligned)
    use_imputation = impute and not getattr(model, "mask_aware", False)
    count_nonfinite = nonfinite == "exclude"

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_obs: list[torch.Tensor] = []
    by_struct: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]] = {}

    n_episodes = 0
    for ep in suite:
        seen = impute_episode(ep) if use_imputation else ep
        pred = torch.as_tensor(model.predict(seen), dtype=torch.float32).reshape(-1)
        target = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1)
        if pred.numel() != target.numel():
            raise ValueError(
                f"baseline {getattr(model, 'name', model)!r} returned {pred.numel()} "
                f"predictions for {target.numel()} queries in episode {ep.scm_id}"
            )
        n_bad = int((~torch.isfinite(pred)).sum())
        if n_bad and not count_nonfinite:
            raise ValueError(
                f"baseline {getattr(model, 'name', model)!r} returned {n_bad} non-finite "
                f"prediction(s) for episode {ep.scm_id}; keep impute=True for episodes with "
                "missing cells, or pass nonfinite='exclude' to score them as errors"
            )
        obs = query_obs_levels(ep).reshape(-1) if effect else None
        all_preds.append(pred)
        all_targets.append(target)
        if obs is not None:
            all_obs.append(obs)
        if ep.structure is not None:
            by_struct.setdefault(ep.structure, []).append((pred, target, obs))
        n_episodes += 1

    if not all_preds:
        raise ValueError(f"suite {suite.meta.name!r} contains no episodes")

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    per_structure = {
        struct: _aggregate(
            torch.cat([p for p, _, _ in rows]),
            torch.cat([t for _, t, _ in rows]),
            metrics,
            torch.cat([o for _, _, o in rows if o is not None]) if effect else None,
            count_nonfinite,
            dir_target=scored,
        )
        for struct, rows in by_struct.items()
    }

    return Results(
        suite=suite.meta.name,
        baseline=getattr(model, "name", type(model).__name__),
        n_episodes=n_episodes,
        n_queries=int(preds.numel()),
        pooled=_aggregate(
            preds,
            targets,
            metrics,
            torch.cat(all_obs) if effect else None,
            count_nonfinite,
            dir_target=scored,
        ),
        per_structure=per_structure,
        dir_target=scored,
        dir_target_mode=dir_target,
        pairs_share_noise=noise.shared,
    )
