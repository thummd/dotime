"""Fingerprints of frozen-suite rows, for checking that released suites still regenerate.

A released suite is rebuilt from its config and seeds by
:func:`dotime._build.make_episode`. A change to a default code path that draws
different random numbers, or applies different arithmetic to them, changes
every future rebuild without any error. Two fingerprints pin a row:

* :func:`row_hashes` hashes every column of the row that
  :func:`dotime._release_io._episode_to_row` writes. It is exact, so it holds
  only on the platform, library versions and CPU kernels that recorded it
  (:func:`reference_env`): float results legitimately differ in the last bit
  between CPUs and library builds.
* :func:`portable_summary` keeps what the random draws alone determine, so it
  must hold everywhere: shapes, the intervention, the query and the RNG-only
  metadata. :func:`summary_mismatches` compares two summaries.

``scripts/fingerprint_frozen_suites.py`` records both for stratified rows of
every released suite version in ``tests/data/frozen_fingerprints.json``, which
``tests/test_frozen_fingerprints.py`` regenerates.
"""

from __future__ import annotations

import hashlib
import math
import platform
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from dotime.benchmarks import Episode

__all__ = [
    "RNG_ONLY_METADATA",
    "portable_summary",
    "reference_env",
    "row_hashes",
    "summary_mismatches",
]

#: Metadata keys whose values are fixed by the spec and the random draws alone.
#: The others (``diverged``, ``y_causal_effect``) are outcomes of the float
#: simulation, which may differ in the last bit between CPUs, so the portable
#: summary records only that those keys exist.
RNG_ONLY_METADATA = frozenset(
    {
        "tier",
        "n_regimes",
        "pair_mode",
        "query_offset_range",
        "query_time_idx",
        "self_query",
        "query_in_window",
        "window_end_idx",
        "fallback",
    }
)


def reference_env() -> dict[str, str]:
    """Describe the environment that exact row hashes are valid for.

    Returns:
        ``platform`` (operating system and machine), the ``torch`` and
        ``numpy`` versions and the ``cpu_capability`` of torch's CPU kernels
        (for example ``"AVX512"``), which selects the vectorized code paths
        that produce the float results.
    """
    return {
        "platform": f"{platform.system()}-{platform.machine()}",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }


def _is_int(value: object) -> bool:
    """Whether ``value`` is an integer scalar that is not a bool.

    Args:
        value: Any object.

    Returns:
        ``True`` for Python and numpy integers, ``False`` for bools and
        everything else.
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _column_bytes(name: str, value: object) -> bytes:
    """Canonical bytes of one row column.

    Args:
        name: The column name, used in error messages only.
        value: A string, an integer, a float or a list of numbers, as
            :func:`dotime._release_io._episode_to_row` writes them and pyarrow
            reads them back.

    Returns:
        UTF-8 for strings, little-endian int64 for integers and integer lists,
        little-endian float64 for floats and float lists.

    Raises:
        TypeError: If the value, or an element of a list, is none of these.
            Bools are refused so that a new flag column needs an explicit
            encoding rather than silently hashing as an integer.
    """
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (list, tuple)):
        # The column's dtype follows its elements: tensor.tolist() and pyarrow
        # both return Python ints for integer columns and floats for float ones.
        if all(_is_int(v) for v in value):
            return np.asarray(value, dtype="<i8").tobytes()
        if all(_is_int(v) or isinstance(v, (float, np.floating)) for v in value):
            return np.asarray(value, dtype="<f8").tobytes()
        raise TypeError(f"column {name!r} holds a list with non-numeric elements")
    if _is_int(value):
        return np.asarray([value], dtype="<i8").tobytes()
    if isinstance(value, (float, np.floating)):
        return np.asarray([value], dtype="<f8").tobytes()
    raise TypeError(f"column {name!r} has unsupported type {type(value).__name__}")


def row_hashes(row: Mapping[str, object]) -> dict[str, str]:
    """SHA-256 of every column of a suite row.

    The encoding is fixed per type rather than per column, so a released row
    read back from parquet and a freshly generated row hash identically when
    their values are identical: the float32 tensors of an episode become the
    same float64 values in both.

    Args:
        row: A row as :func:`dotime._release_io._episode_to_row` returns it,
            or a parquet row of a released suite.

    Returns:
        ``{column: hex digest}`` in the row's column order.

    Raises:
        TypeError: If a column holds a value :func:`_column_bytes` cannot encode.
    """
    return {
        name: hashlib.sha256(_column_bytes(name, value)).hexdigest() for name, value in row.items()
    }


def _round_rel(value: float) -> float:
    """Round a float to seven significant digits.

    Args:
        value: Any float, including infinities and NaN.

    Returns:
        ``value`` with a relative rounding error below ``1e-6``, which removes
        the last-bit differences between CPUs from a stored summary.
    """
    return float(f"{value:.6e}")


def _rounded(value: Any) -> Any:
    """Recursively round every float of a JSON-like value.

    Args:
        value: A scalar, tensor, list, tuple or dict.

    Returns:
        The same structure with floats passed through :func:`_round_rel`,
        tensors converted to lists and tuples to lists.

    Raises:
        TypeError: If a leaf is not a bool, int, float, string or ``None``.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value is None or isinstance(value, (bool, str)) or _is_int(value):
        return value
    if isinstance(value, (float, np.floating)):
        return _round_rel(float(value))
    if isinstance(value, Mapping):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    raise TypeError(f"cannot summarise a value of type {type(value).__name__}")


def _ranges(steps: Sequence[int]) -> list[list[int]]:
    """Compress intervention steps into inclusive ``[first, last]`` runs.

    Args:
        steps: The intervention's time steps, in the order it stores them.

    Returns:
        One ``[first, last]`` pair per run of consecutive steps. A generic
        window of 150 steps becomes a single pair, which keeps the stored
        summaries small.
    """
    runs: list[list[int]] = []
    for step in (int(s) for s in steps):
        if runs and step == runs[-1][1] + 1:
            runs[-1][1] = step
        else:
            runs.append([step, step])
    return runs


def portable_summary(ep: Episode) -> dict[str, Any]:
    """Summarise what the random draws of an episode determine.

    Args:
        ep: A generated or loaded episode.

    Returns:
        A JSON-able dict with the ``structure`` label, ``scm_id``, ``n_vars``
        and ``length``, the intervention (targets, steps as runs, type and
        values rounded to seven significant digits), the query targets and
        rounded query times, the sorted metadata keys a suite row stores and
        the values of the keys in :data:`RNG_ONLY_METADATA`.

    Raises:
        TypeError: If the intervention or an RNG-only metadata value holds a
            type that has no JSON form.
    """
    spec = ep.intervention.to_dict()
    # y_oracle never reaches a suite row (_episode_to_row drops it).
    keys = sorted(k for k in ep.metadata if k != "y_oracle")
    return {
        "structure": ep.structure,
        "scm_id": ep.scm_id,
        "n_vars": ep.n_vars,
        "length": ep.length,
        "intervention": {
            "targets": [int(t) for t in spec["targets"]],
            "times": _ranges(spec["times"]),
            "type": spec["intervention_type"],
            "values": _rounded(spec["values"]),
        },
        "query_target": [int(q) for q in ep.query_target.reshape(-1).tolist()],
        "query_time": [_round_rel(float(q)) for q in ep.query_time.reshape(-1).tolist()],
        "metadata_keys": keys,
        "metadata": {k: _rounded(ep.metadata[k]) for k in keys if k in RNG_ONLY_METADATA},
    }


def summary_mismatches(
    expected: Any,
    actual: Any,
    *,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-7,
    path: str = "summary",
) -> list[str]:
    """List where two portable summaries disagree.

    Floats are compared with a tolerance because the stored summary is rounded
    and because a platform computes some values differently: the float32
    trajectories of time-varying interventions go through the platform's
    ``sin`` and ``exp``, and macOS arm64 and x86 Linux disagree on them by up
    to about 1e-5 relative, which the rounding of :func:`portable_summary`
    (seven significant digits) does not hide. Everything else, including every
    integer, must be equal.

    Args:
        expected: The stored summary (or a part of it).
        actual: The regenerated summary (or the matching part).
        rel_tol: Relative tolerance for float leaves, ten times the largest
            cross-platform difference seen so far.
        abs_tol: Absolute tolerance for float leaves near zero.
        path: Dotted location of ``expected``, used in the messages.

    Returns:
        One ``"path: expected != actual"`` message per disagreement, empty when
        the summaries match.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            return [f"{path}: keys {sorted(expected)} != {sorted(actual)}"]
        out: list[str] = []
        for key in expected:
            out += summary_mismatches(
                expected[key], actual[key], rel_tol=rel_tol, abs_tol=abs_tol, path=f"{path}.{key}"
            )
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            out += summary_mismatches(e, a, rel_tol=rel_tol, abs_tol=abs_tol, path=f"{path}[{i}]")
        return out
    floats = isinstance(expected, float) or isinstance(actual, float)
    numbers = all(_is_int(v) or isinstance(v, float) for v in (expected, actual))
    if floats and numbers:
        e, a = float(expected), float(actual)
        if (math.isnan(e) and math.isnan(a)) or math.isclose(
            e, a, rel_tol=rel_tol, abs_tol=abs_tol
        ):
            return []
    elif type(expected) is type(actual) and expected == actual:
        return []
    return [f"{path}: {expected!r} != {actual!r}"]
