#!/usr/bin/env python
"""Reference rows of a released suite, pooled and split by one design factor.

``dotime-eval-reference`` pools a suite. The camera-ready paper also reports the
CPU baselines per level of one factor: the lag of the shortest path from the
treatment to the query on dot-Wide-v1, and the observation schedule of
dot-ContinuousIrregular-v1 without self-queries. This script scores each level
with the functions behind ``dotime-eval-reference``
(:func:`dotime.reference.reference_table.run_baseline`), so a pooled row here and
a row of the reference table are the same computation.

Two gates run before anything is written. The per-arm target statistics are
logged and asserted, as in every evaluator. With ``--reference``, the pooled
rows must equal the rows of an existing ``dotime-eval-reference`` file.

Usage::

    python group_rows.py --suite dot-Wide-v1 --version 1.0.0 --group min_lag \
        --reference ../../wide/wide_cpu_effect.json --out ../wide_per_lag.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dotime.benchmarks import Episode, load_benchmark
from dotime.evaluation import check_shared_noise, query_obs_levels
from dotime.reference.reference_table import CPU_BASELINES, run_baseline, target_qa

# The bins of audit_2026-09/scripts/lag_breakdown.py, so the two breakdowns
# (dot-Generic-100k there, dot-Wide-v1 here) can be read side by side.
LAG_BINS = ("0", "1", "2", "3", ">=4", "unreachable")
# Fields that must agree with the reference file. The bootstrap interval is left
# out because it is seeded per call and carries no information about the rows.
GATE_FIELDS = ("n_episodes", "pooled_rmse", "dir_acc", "dir_n_valid")
KEPT_FIELDS = (
    "n_episodes",
    "pooled_rmse",
    "rmse_ci95",
    "dir_acc",
    "dir_n_valid",
    "dir_acc_se",
    "dir_target",
)


def lag_bin(episode: Episode) -> str:
    """Bin of the shortest treatment-to-query path lag of one episode.

    Args:
        episode: An episode built with ``record_graph: true``, whose
            ``metadata["graph"]["path"]`` holds one entry per query.

    Returns:
        ``"0"`` to ``"3"``, ``">=4"`` or ``"unreachable"``.

    Raises:
        ValueError: If the episode has no recorded graph or more than one query.
    """
    graph = episode.metadata.get("graph")
    if not isinstance(graph, dict) or "path" not in graph:
        raise ValueError(f"episode {episode.scm_id} carries no recorded graph")
    paths = graph["path"]
    if len(paths) != 1:
        raise ValueError(f"episode {episode.scm_id} has {len(paths)} queries, expected 1")
    min_lag = paths[0]["min_lag"]
    if min_lag is None:
        return "unreachable"
    return str(min_lag) if min_lag < 4 else ">=4"


def group_key(episode: Episode, group: str) -> str:
    """Level of the grouping factor for one episode.

    Args:
        episode: The episode.
        group: ``"min_lag"``, ``"structure"`` or a metadata key.

    Returns:
        The level as a string.

    Raises:
        ValueError: If ``group`` is ``"min_lag"`` and the episode has no graph.
        KeyError: If ``group`` is a metadata key that the episode lacks.
    """
    if group == "min_lag":
        return lag_bin(episode)
    if group == "structure":
        return str(episode.structure)
    return str(episode.metadata[group])


def effect_summary(episodes: Sequence[Episode]) -> dict[str, float]:
    """Size of the counterfactual effect at the query over a set of episodes.

    Args:
        episodes: Shared-noise episodes with one query each.

    Returns:
        The median absolute effect and the share of episodes whose absolute
        effect reaches the 0.1 threshold of the direction metric.
    """
    effect = np.array(
        [
            float(np.asarray(ep.y_true, dtype=np.float64).reshape(-1)[0])
            - float(query_obs_levels(ep).reshape(-1)[0])
            for ep in episodes
        ]
    )
    return {
        "abs_effect_median": float(np.median(np.abs(effect))),
        "share_abs_effect_ge_0.1": float(np.mean(np.abs(effect) >= 0.1)),
    }


def strict_json(value: Any) -> Any:
    """Replace non-finite floats by ``None`` throughout a nested result.

    Args:
        value: A JSON-serialisable value, possibly nested.

    Returns:
        The same structure with every NaN or infinite float turned into ``None``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    return value


def score(episodes: Sequence[Episode], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Score every baseline on one set of episodes.

    Args:
        episodes: The episodes of one group, or the whole suite.
        names: Baseline names of :mod:`dotime.baselines`.

    Returns:
        ``{baseline: row}`` with the fields in ``KEPT_FIELDS``.
    """
    # One noise check per group instead of one per baseline: it reads every
    # episode's pre-onset rows and would otherwise dominate the run time.
    noise = check_shared_noise(episodes)
    rows = {}
    for name in names:
        row = run_baseline(name, episodes, dir_target="effect", noise=noise)
        rows[name] = {k: row[k] for k in KEPT_FIELDS}
    return rows


def assert_reproduces(pooled: dict[str, dict[str, Any]], reference: Path) -> None:
    """Check the pooled rows against an existing ``dotime-eval-reference`` file.

    Args:
        pooled: ``{baseline: row}`` from :func:`score` on the whole suite.
        reference: Path of the reference JSON.

    Raises:
        SystemExit: If a baseline is missing from the reference or a gated field
            differs.
    """
    ref = {r["baseline"]: r for r in json.loads(reference.read_text())["rows"]}
    for name, row in pooled.items():
        if name not in ref:
            raise SystemExit(f"gate: {name} is not in {reference}")
        for field in GATE_FIELDS:
            ours, theirs = row[field], ref[name][field]
            # Same functions, same episodes, same machine: the values agree to
            # the last bit, and the tolerance only absorbs JSON round-tripping.
            if not np.isclose(ours, theirs, rtol=0.0, atol=1e-12):
                raise SystemExit(f"gate: {name}.{field} = {ours!r}, reference has {theirs!r}")
    print(f"gate: pooled rows reproduce {reference.name} ({', '.join(GATE_FIELDS)})")


def main() -> None:
    """Score the baselines pooled and per group, and write one JSON file.

    Raises:
        SystemExit: If the arms do not share their noise, a gate fails, or a
            level of the factor is not one of the expected lag bins.
        RuntimeError: If the per-arm target statistics fail their checks.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--group", required=True, help="'min_lag', 'structure' or a metadata key.")
    ap.add_argument("--exclude-self-queries", action="store_true")
    ap.add_argument("--reference", type=Path, help="dotime-eval-reference JSON to reproduce.")
    ap.add_argument("--baselines", nargs="+", default=list(CPU_BASELINES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    episodes = list(load_benchmark(args.suite, version=args.version))
    n_loaded = len(episodes)
    if args.exclude_self_queries:
        episodes = [ep for ep in episodes if not ep.is_self_query]
    # Step-zero target check, pooled and per structure. It raises on a zeroed or
    # constant arm, so no accuracy below is computed on a corrupted target.
    qa = target_qa(episodes, dir_target="effect")
    for arm in ("y_obs_level", "y_int_level", "effect"):
        s = qa[arm]
        print(
            f"[target QA] {arm}: n={s['n']} nonzero={s['nonzero_frac']:.4f} "
            f"mean={s['mean']:.4f} var={s['var']:.4f}",
            flush=True,
        )
    if not check_shared_noise(episodes).shared:
        raise SystemExit("the arms do not share their noise, so the effect sign is not scored")

    pooled = score(episodes, args.baselines)
    if args.reference is not None:
        assert_reproduces(pooled, args.reference)

    groups: dict[str, list[Episode]] = {}
    for ep in episodes:
        groups.setdefault(group_key(ep, args.group), []).append(ep)
    if args.group == "min_lag":
        unknown = set(groups) - set(LAG_BINS)
        if unknown:
            raise SystemExit(f"unexpected lag bins: {sorted(unknown)}")
        order = [b for b in LAG_BINS if b in groups]
    else:
        order = sorted(groups)

    out: dict[str, Any] = {
        "suite": args.suite,
        "suite_version": args.version,
        "group": args.group,
        "exclude_self_queries": args.exclude_self_queries,
        "protocol": "dotime.reference.reference_table.run_baseline(dir_target='effect')",
        "n_loaded": n_loaded,
        "n_episodes": len(episodes),
        "reference": args.reference.name if args.reference else None,
        "target_qa": qa,
        "pooled": {"baselines": pooled, **effect_summary(episodes)},
        "groups": {},
    }
    for key in order:
        eps = groups[key]
        rows = score(eps, args.baselines)
        out["groups"][key] = {"n_episodes": len(eps), "baselines": rows, **effect_summary(eps)}
        print(
            f"[{key}] n={len(eps)} "
            + " ".join(f"{n}={rows[n]['dir_acc']:.3f}" for n in args.baselines),
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # A group without a scored episode (an unreachable query has an effect of
    # exactly zero) has no accuracy. It is written as null, not as the NaN
    # literal, so that strict JSON parsers can read the file.
    args.out.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
