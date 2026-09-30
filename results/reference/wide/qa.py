#!/usr/bin/env python
"""Quality check of the wide generic suite dot-Wide-v1 before it is built in full.

Builds the first episodes of ``scripts/release_config_wide.yaml`` through the
release path (``dotime._build.episode_specs`` and ``make_episode``). Episode
seeds depend only on the suite seed and the episode index, so these are the
first episodes of the full build. The script asserts that no episode diverged,
that every simulated graph has 12 to 40 variables, that both arms agree exactly
before the intervention onset (shared noise), and the per-arm target statistics
of the reference harness (``target_qa``). It logs the distributions of the graph
size and of the number of hidden variables, the saturation share, the size of
the counterfactual effect at the query and the time per episode, and projects
the time of the full build.

    PYTHONPATH=src python results/reference/wide/qa.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

from dotime import __version__
from dotime._build import episode_specs, make_episode
from dotime.reference.reference_table import target_qa

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG = _ROOT / "scripts" / "release_config_wide.yaml"
_OUT = Path(__file__).with_name("wide_qa.json")
_SUITE = "dot-Wide-v1"
# The simulator clips every value to +-1000, so |x| >= 999 marks a variable
# pinned at the clip rather than a genuine level.
_SATURATION = 999.0
# The full build is planned on 15 workers and has to fit in 2 hours.
_PLANNED_WORKERS = 15
_BUDGET_HOURS = 2.0


def _timed_episode(spec: dict):
    """Build one episode and time it (runs in a worker).

    Args:
        spec: A spec from :func:`dotime._build.episode_specs`.

    Returns:
        ``(episode, seconds)``.
    """
    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.perf_counter()
    episode = make_episode(spec)
    return episode, time.perf_counter() - t0


def _quantiles(values) -> dict[str, float]:
    """Summarise a sample by its mean and a few quantiles.

    Args:
        values: 1-D sequence of numbers.

    Returns:
        Dict with ``mean``, ``min``, ``p10``, ``median``, ``p90`` and ``max``.
    """
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "min": float(v.min()),
        "p10": float(np.quantile(v, 0.1)),
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.9)),
        "max": float(v.max()),
    }


def _histogram(values) -> dict[str, int]:
    """Count the occurrences of each integer value, in ascending order.

    Args:
        values: Sequence of ints.

    Returns:
        ``{str(value): count}``, sorted by value so the JSON reads in order.
    """
    return {str(k): v for k, v in sorted(Counter(int(x) for x in values).items())}


def _effect_record(ep) -> dict:
    """Describe the counterfactual effect of one episode at and before its query.

    Args:
        ep: A built :class:`~dotime.benchmarks.Episode`.

    Returns:
        Dict with the query effect ``y_true - y_obs``, its size over the
        queried variable's pre-onset observational standard deviation (``None``
        when that deviation is zero), whether the query row lies inside the
        intervention window, the number of steps from the window's last step to
        the query, and the largest effect over released non-target variables at
        the window's last step.
    """
    row, col = ep.metadata["query_time_idx"][0], int(ep.query_target[0])
    onset, end = min(ep.intervention.times), max(ep.intervention.times)
    effect = float(ep.y_true[0] - ep.x_obs[row, col])
    sd = float(ep.x_obs[:onset, col].std())
    at_end = (ep.x_int[end] - ep.x_obs[end]).abs()
    at_end[ep.intervention.targets] = 0.0
    return {
        "effect_at_query": effect,
        "effect_at_query_sd": abs(effect) / sd if sd > 0 else None,
        "query_in_window": onset <= row <= end,
        "steps_after_window": row - end,
        "max_effect_at_window_end": float(at_end.max()),
    }


def main(argv: list[str] | None = None) -> int:
    """Build the QA episodes, assert the invariants and write the JSON summary.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        0 on success.

    Raises:
        AssertionError: If an episode diverged, a graph size leaves [12, 40],
            or the arms differ before the onset.
        RuntimeError: If the per-arm target statistics fail (see
            :func:`dotime.reference.reference_table.target_qa`).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=_OUT)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())["suites"][_SUITE]
    seed = int(cfg["seed"])
    specs = episode_specs(cfg, seed, args.n_episodes / cfg["n_episodes"])
    load_before = os.getloadavg()
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        results = pool.map(_timed_episode, specs, chunksize=1)
    wall = time.perf_counter() - t0
    load_after = os.getloadavg()
    episodes = [ep for ep, _ in results]
    seconds = [sec for _, sec in results]

    n_full = [ep.metadata["latent"]["n_vars_full"] for ep in episodes]
    n_hidden = [len(ep.metadata["latent"]["hidden"]) for ep in episodes]
    diverged = [ep.scm_id for ep in episodes if ep.metadata["diverged"]]
    assert not diverged, f"diverged episodes: {diverged}"
    assert all(12 <= n <= 40 for n in n_full), _histogram(n_full)
    for ep in episodes:
        onset = min(ep.intervention.times)
        assert torch.equal(ep.x_obs[:onset], ep.x_int[:onset]), ep.scm_id
        latent = ep.metadata["latent"]
        assert ep.x_obs.shape[1] + len(latent["hidden"]) == latent["n_vars_full"], ep.scm_id
    targets = target_qa(episodes, dir_target="effect")

    saturated = [
        bool((ep.x_obs.abs() >= _SATURATION).any() or (ep.x_int.abs() >= _SATURATION).any())
        for ep in episodes
    ]
    entries = sum(ep.x_obs.numel() + ep.x_int.numel() for ep in episodes)
    saturated_entries = sum(
        int((ep.x_obs.abs() >= _SATURATION).sum() + (ep.x_int.abs() >= _SATURATION).sum())
        for ep in episodes
    )
    effects = [_effect_record(ep) for ep in episodes]
    abs_query = [abs(e["effect_at_query"]) for e in effects]
    tiers = [ep.metadata["tier"] for ep in episodes]
    mean_sec = float(np.mean(seconds))
    projected_hours = cfg["n_episodes"] * mean_sec / _PLANNED_WORKERS / 3600
    summary = {
        "suite": _SUITE,
        "version": cfg["version"],
        "config": str(args.config.resolve().relative_to(_ROOT)),
        "suite_seed": seed,
        "n_episodes": len(episodes),
        "package_version": __version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "workers": args.workers,
        "load_average_before_after": [load_before[0], load_after[0]],
        "diverged": len(diverged),
        "n_vars_full_histogram": _histogram(n_full),
        "n_vars_released_histogram": _histogram(ep.x_obs.shape[1] for ep in episodes),
        "n_hidden_histogram": _histogram(n_hidden),
        "hidden_share_of_variables": float(np.sum(n_hidden) / np.sum(n_full)),
        "tier_histogram": _histogram(tiers),
        "intervention_types": dict(
            Counter(ep.intervention.intervention_type.value for ep in episodes)
        ),
        "saturated_episode_share": float(np.mean(saturated)),
        "saturated_entry_share": saturated_entries / entries,
        "pre_onset_arms_equal": True,
        "target_qa": targets,
        "effect_at_query": {
            "abs": _quantiles(abs_query),
            "share_abs_ge_0.01": float(np.mean([a >= 0.01 for a in abs_query])),
            "share_abs_ge_0.1": float(np.mean([a >= 0.1 for a in abs_query])),
            "share_ge_0.1_pre_onset_sd": float(
                np.mean([(e["effect_at_query_sd"] or 0.0) >= 0.1 for e in effects])
            ),
            "query_in_window_share": float(np.mean([e["query_in_window"] for e in effects])),
            "steps_after_window": _quantiles([e["steps_after_window"] for e in effects]),
        },
        "max_effect_at_window_end": {
            "abs": _quantiles([e["max_effect_at_window_end"] for e in effects]),
            "share_abs_ge_0.1": float(
                np.mean([e["max_effect_at_window_end"] >= 0.1 for e in effects])
            ),
        },
        "seconds_per_episode": _quantiles(seconds),
        "mean_seconds_per_episode_by_tier": {
            str(t): float(np.mean([s for s, u in zip(seconds, tiers, strict=True) if u == t]))
            for t in sorted(set(tiers))
        },
        "wall_seconds": wall,
        "projected_full_build_hours": {
            "episodes": cfg["n_episodes"],
            "workers": _PLANNED_WORKERS,
            "hours": projected_hours,
            "within_budget": projected_hours <= _BUDGET_HOURS,
        },
        "episodes": [
            {
                "idx": ep.scm_id,
                "n_vars_full": nf,
                "n_hidden": nh,
                "tier": ep.metadata["tier"],
                "seconds": round(sec, 3),
                # Significant digits, so that tiny effects are not rounded to zero.
                **{k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in e.items()},
            }
            for ep, nf, nh, sec, e in zip(episodes, n_full, n_hidden, seconds, effects, strict=True)
        ],
    }
    args.out.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    print(
        f"[wide QA] {len(episodes)} episodes, 0 diverged, N {min(n_full)}-{max(n_full)}, "
        f"hidden share {summary['hidden_share_of_variables']:.3f}, "
        f"{mean_sec:.2f} s/episode, full build ~{projected_hours:.2f} h on "
        f"{_PLANNED_WORKERS} workers -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
