"""Direction accuracy on the level and on the effect for the out-of-band evaluators.

The TabPFN and Chronos-2 evaluators score one prediction per episode. They
used to report direction accuracy on the interventional level only (the v1
protocol). This helper scores both the level and the causal effect, so a
result JSON carries the two side by side and ``--dir-target`` only decides
which one fills the ``dir_acc`` field that the tables read.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from dotime.evaluation import DIR_TARGETS, direction_accuracy, query_obs_levels


def observational_levels(
    episodes: Sequence, sidecar_levels: list[list[float]] | None
) -> np.ndarray:
    """Factual level of each episode's first query.

    Args:
        episodes: The scored episodes, in scoring order.
        sidecar_levels: Levels from a realignment sidecar
            (:func:`dotime.reference._realignment.sidecar_obs_levels`), or
            ``None`` to read them with :func:`dotime.evaluation.query_obs_levels`.

    Returns:
        One float per episode.
    """
    if sidecar_levels is not None:
        return np.array([float(lv[0]) for lv in sidecar_levels], dtype=np.float64)
    return np.array(
        [float(torch.as_tensor(query_obs_levels(ep)).reshape(-1)[0]) for ep in episodes],
        dtype=np.float64,
    )


def direction_scores(
    preds: np.ndarray, tgts: np.ndarray, y_obs: np.ndarray, dir_target: str
) -> dict[str, float | int | str]:
    """Direction accuracy on the level and on the effect, with binomial errors.

    Args:
        preds: Predicted interventional levels.
        tgts: Released interventional levels (``y_true``).
        y_obs: Factual levels at the query.
        dir_target: Which score fills ``dir_acc``: ``"level"`` compares
            ``sign(pred)`` with ``sign(y_true)``, ``"effect"`` compares
            ``sign(pred - y_obs)`` with ``sign(y_true - y_obs)``.

    Returns:
        ``dir_acc``, ``dir_n_valid`` and ``dir_acc_se`` for the selected
        target, ``dir_target`` itself, and the same three numbers for both
        targets under ``dir_acc_level`` and ``dir_acc_effect`` prefixes.

    Raises:
        ValueError: If ``dir_target`` is unknown.
    """
    if dir_target not in DIR_TARGETS:
        raise ValueError(f"dir_target must be one of {DIR_TARGETS}, got {dir_target!r}")
    p, t = torch.from_numpy(preds).float(), torch.from_numpy(tgts).float()
    o = torch.from_numpy(y_obs).float()
    out: dict[str, float | int | str] = {"dir_target": dir_target}
    for name, pp, tt in (("level", p, t), ("effect", p - o, t - o)):
        da = direction_accuracy(pp, tt)
        n_valid = int(da["n_valid"])
        acc = float(da["accuracy"])
        se = (acc * (1 - acc) / n_valid) ** 0.5 if n_valid else float("nan")
        out[f"dir_acc_{name}"] = acc
        out[f"dir_n_valid_{name}"] = n_valid
        out[f"dir_acc_se_{name}"] = se
        if name == dir_target:
            out.update(dir_acc=acc, dir_n_valid=n_valid, dir_acc_se=se)
    return out


def check_predictions(tag: str, preds: np.ndarray) -> int:
    """Count non-finite predictions of one arm and refuse an arm that has none finite.

    A broken optional dependency can return NaN for every episode without
    raising (a transformers release outside ``chronos-forecasting``'s pin
    re-initialises the weights at random on load), and the result JSON would
    then carry NaN errors as if they were measurements.

    Args:
        tag: Arm name, for the message.
        preds: Predicted levels of the arm.

    Returns:
        The number of non-finite predictions.

    Raises:
        SystemExit: If no prediction is finite.
    """
    n_bad = int((~np.isfinite(preds)).sum())
    if n_bad == len(preds):
        raise SystemExit(
            f"{tag}: every prediction is non-finite. Check the optional dependencies "
            "(chronos-forecasting pins transformers<5 and huggingface_hub<1.0; TabPFN "
            "needs its weights) before trusting any number from this run."
        )
    if n_bad:
        print(f"  {tag}: {n_bad} of {len(preds)} predictions are non-finite", flush=True)
    return n_bad
