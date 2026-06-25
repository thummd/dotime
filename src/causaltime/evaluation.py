"""Evaluation harness for CausalTime benchmark suites.

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
- :func:`evaluate` — run a baseline over a suite, aggregating pooled and
  per-structure metrics.
- :class:`Results` — holds the aggregated metrics with ``.summary()`` and
  ``.to_dict()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from causaltime.baselines import Baseline
    from causaltime.benchmarks import BenchmarkSuite

__all__ = [
    "DIR_ACC_EPS",
    "Results",
    "bootstrap_ci",
    "compute_mae",
    "compute_nmse",
    "compute_r2",
    "compute_rmse",
    "direction_accuracy",
    "evaluate",
]

# Near-zero targets are ambiguous for sign-based direction accuracy and are
# excluded from that metric (reported separately).
DIR_ACC_EPS = 0.1


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


def direction_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, eps: float = DIR_ACC_EPS
) -> dict[str, float | int]:
    """Sign-consistent direction accuracy, excluding near-zero targets.

    Returns a dict with ``accuracy`` (fraction of ``|target| >= eps`` samples
    whose predicted sign matches), ``n_valid`` and ``n_excluded``.
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

    def to_dict(self) -> dict:
        """JSON-serializable view of the results."""
        return {
            "suite": self.suite,
            "baseline": self.baseline,
            "n_episodes": self.n_episodes,
            "n_queries": self.n_queries,
            "pooled": self.pooled,
            "per_structure": self.per_structure,
        }

    def summary(self) -> str:
        """Human-readable results table."""
        lines = [
            f"Suite:    {self.suite}",
            f"Baseline: {self.baseline}",
            f"Episodes: {self.n_episodes}   Queries: {self.n_queries}",
            "",
        ]
        cols = ["rmse", "mae", "nmse", "r2", "dir_acc"]
        header = f"{'group':<22}" + "".join(f"{c:>10}" for c in cols)
        lines.append(header)
        lines.append("-" * len(header))

        def _row(name: str, m: dict[str, float]) -> str:
            cells = []
            for c in cols:
                v = m.get(c, float("nan"))
                cells.append(f"{v:>10.4f}" if isinstance(v, (int, float)) else f"{v:>10}")
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


def _aggregate(preds: torch.Tensor, targets: torch.Tensor, metrics) -> dict[str, float]:
    out = {name: fn(preds, targets) for name, fn in metrics.items()}
    out["dir_acc"] = direction_accuracy(preds, targets)["accuracy"]
    return out


def evaluate(
    model: Baseline,
    suite: BenchmarkSuite,
    metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] | None = None,
) -> Results:
    """Evaluate a baseline over every episode of a suite.

    Calls ``model.predict(episode)`` for each episode, pools predictions and
    ground-truth targets across all queries, and reports pooled and
    per-structure metrics.
    """
    metrics = metrics or _DEFAULT_METRICS

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    by_struct: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    n_episodes = 0
    for ep in suite:
        pred = torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1)
        target = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1)
        if pred.numel() != target.numel():
            raise ValueError(
                f"baseline {getattr(model, 'name', model)!r} returned {pred.numel()} "
                f"predictions for {target.numel()} queries in episode {ep.scm_id}"
            )
        all_preds.append(pred)
        all_targets.append(target)
        if ep.structure is not None:
            by_struct.setdefault(ep.structure, []).append((pred, target))
        n_episodes += 1

    if not all_preds:
        raise ValueError(f"suite {suite.meta.name!r} contains no episodes")

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    per_structure = {
        struct: _aggregate(
            torch.cat([p for p, _ in pairs]),
            torch.cat([t for _, t in pairs]),
            metrics,
        )
        for struct, pairs in by_struct.items()
    }

    return Results(
        suite=suite.meta.name,
        baseline=getattr(model, "name", type(model).__name__),
        n_episodes=n_episodes,
        n_queries=int(preds.numel()),
        pooled=_aggregate(preds, targets, metrics),
        per_structure=per_structure,
    )
