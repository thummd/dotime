"""Load and apply the ``dot-Identifiability-v1`` 1.0.0 realignment sidecar.

The archived 1.0.0 files store ``x_obs`` in topological column order, while
``intervention.targets`` and ``query_target`` index the canonical order, and
they leave hidden variables unmasked (datasheet erratum in
``docs/benchmarks.md``). The released JSONL sidecar
``results/reference/dot-Identifiability-v1.0.0_realignment.jsonl`` holds one
row per episode with the permutation and hidden columns that repair ``x_obs``,
plus the ``y_true`` it regenerated from the episode's seed.

A sidecar row is only valid for the episode it was built from. Suite 1.1.0
reuses the same episode ids but already stores canonical ``x_obs``, so
permuting it again would scramble the columns without any error. The loader
therefore checks every row against its episode before realigning it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dotime.benchmarks import Episode
from dotime.evaluation import realign_episode

__all__ = ["load_realignment", "realign_episodes"]

# The sidecar stores y_true as the float64 image of the released float32
# value, so a matching episode agrees exactly (verified on all 10,800 1.0.0
# episodes). The tolerance only absorbs float formatting. Suites whose
# targets differ, such as 1.1.0, differ by far more than this.
_Y_TRUE_TOL = 1e-6

_VERSION_HINT = (
    "The released sidecar repairs dot-Identifiability-v1 1.0.0 only. Pass "
    "--version 1.0.0, or drop --realignment for 1.1.0 and later, whose x_obs "
    "is already canonical."
)


def load_realignment(path: str | Path) -> dict[int, dict[str, Any]]:
    """Read a realignment sidecar into a map from episode id to its row.

    Args:
        path: The JSONL sidecar. Each line is one JSON object with at least
            ``idx``, ``n_vars``, ``canonical_perm``, ``hidden_canonical``,
            ``query_target`` and ``y_true_regen``.

    Returns:
        ``{idx: row}``, keyed like :attr:`~dotime.benchmarks.Episode.scm_id`.

    Raises:
        OSError: If ``path`` cannot be read.
        ValueError: If a line is not valid JSON, lacks one of the fields above,
            or repeats an ``idx``. A repeated id would silently shadow an
            earlier row.
    """
    required = (
        "idx",
        "n_vars",
        "canonical_perm",
        "hidden_canonical",
        "query_target",
        "y_true_regen",
    )
    rows: dict[int, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{lineno}: realignment row lacks {missing}")
            idx = int(row["idx"])
            if idx in rows:
                raise ValueError(f"{path}:{lineno}: duplicate realignment row for idx {idx}")
            rows[idx] = row
    return rows


def _check_row(ep: Episode, row: dict[str, Any]) -> None:
    """Check that a sidecar row was built from this episode and is well formed.

    Args:
        ep: The episode about to be realigned.
        row: The sidecar row stored under ``ep.scm_id``.

    Raises:
        ValueError: If the row's variable count, query target or regenerated
            ``y_true`` disagree with the episode, or if its permutation or
            hidden columns are not valid for that many variables.
    """
    n_vars = ep.n_vars
    queries = [int(q) for q in ep.query_target.reshape(-1).tolist()]
    targets = [float(v) for v in ep.y_true.reshape(-1).tolist()]
    y_regen = float(row["y_true_regen"])
    if (
        int(row["n_vars"]) != n_vars
        or queries != [int(row["query_target"])]
        or len(targets) != 1
        or not math.isclose(targets[0], y_regen, rel_tol=_Y_TRUE_TOL, abs_tol=_Y_TRUE_TOL)
    ):
        raise ValueError(
            f"realignment row {ep.scm_id} was not built from this episode. Episode: "
            f"n_vars={n_vars}, query_target={queries}, y_true={targets}. Sidecar: "
            f"n_vars={row['n_vars']}, query_target={row['query_target']}, "
            f"y_true_regen={y_regen}. {_VERSION_HINT}"
        )
    perm = [int(c) for c in row["canonical_perm"]]
    hidden = [int(h) for h in row["hidden_canonical"]]
    # realign_episode trusts its inputs: a repeated index would duplicate a
    # column and an out-of-range hidden index would raise deep inside torch.
    if sorted(perm) != list(range(n_vars)) or any(not 0 <= h < n_vars for h in hidden):
        raise ValueError(
            f"realignment row {ep.scm_id} is malformed: canonical_perm={perm} and "
            f"hidden_canonical={hidden} must index {n_vars} variables"
        )


def realign_episodes(
    episodes: Iterable[Episode], realignment: dict[int, dict[str, Any]]
) -> list[Episode]:
    """Realign every episode's ``x_obs`` to canonical order, after checking each row.

    Unlike the lenient lookup in ``dotime-eval-reference``, an episode without
    a row is an error, not a pass-through. The published sidecar covers all
    10,800 episodes, so a miss means the sidecar belongs to another suite.

    Args:
        episodes: Episodes of the suite the sidecar was built for.
        realignment: Rows from :func:`load_realignment`.

    Returns:
        New episodes whose ``x_obs`` is permuted to canonical order with hidden
        columns zeroed. Every other field is shared with the input episode.

    Raises:
        ValueError: If an episode has no row, or its row fails the checks of
            :func:`_check_row`.
    """
    out = []
    for ep in episodes:
        # scm_id is Optional on Episode; None can never match an integer key.
        row = realignment.get(ep.scm_id) if ep.scm_id is not None else None
        if row is None:
            raise ValueError(
                f"realignment sidecar has no row for episode {ep.scm_id} "
                f"(structure {ep.structure!r}). {_VERSION_HINT}"
            )
        _check_row(ep, row)
        out.append(realign_episode(ep, row["canonical_perm"], row["hidden_canonical"]))
    return out
