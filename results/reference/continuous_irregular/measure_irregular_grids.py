"""Stability of irregular observation grids against the frozen dot-Continuous-v1.

For each structure, the first ``--n`` episode indices of that structure's block
in the suite (suite seed 20263719, 3,333 episodes per structure) are built on
every candidate schedule. All candidates therefore share seeds, SCMs and
interventions, and the comparison is paired. Candidates that pass
``check_schedule`` are built by ``dotime._build.make_episode``, exactly as a
release build would. The two ``unguarded`` candidates allow Euler sub-steps
above 1.0, which ``check_schedule`` refuses, and are built with the same calls
past the guard to show what it prevents.

Usage::

    PYTHONPATH=src python results/reference/continuous_irregular/measure_irregular_grids.py \\
        --cache ~/.cache/dotime/dot-Continuous-v1-1.0.0 --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_CONFIG = _HERE.parents[2] / "scripts" / "release_config_continuous_irregular.yaml"
_SUITE_SEED, _PER, _T = 20263719, 3333, 200
_STRUCTURES = ("back_door", "front_door", "instrumental_variable")


def candidates() -> dict[str, tuple[dict, bool]]:
    """Schedules to measure, keyed by label.

    Returns:
        ``label -> (schedules entry, guarded)``. The release config's three
        entries come first, then alternatives, the regular grid with Euler
        sub-steps (a ``jittered`` entry with ``jitter`` 0), and two unguarded
        entries with sub-steps above 1.0.
    """
    suite = yaml.safe_load(_CONFIG.read_text())["suites"]["dot-ContinuousIrregular-v1"]
    out = {entry["name"]: (entry, True) for entry in suite["schedules"]}
    jit = {"kind": "jittered", "dt": 1.0}
    poi = {"kind": "poisson", "rate": 1.0}
    out.update(
        {
            "jittered_0.9": ({"name": "j", **jit, "jitter": 0.9, "num_substeps": 2}, True),
            "poisson_cap3": ({"name": "p", **poi, "max_gap": 3.0, "num_substeps": 3}, True),
            "regular_2_substeps": ({"name": "r2", **jit, "jitter": 0.0, "num_substeps": 2}, True),
            "regular_4_substeps": ({"name": "r4", **jit, "jitter": 0.0, "num_substeps": 4}, True),
            "unguarded_jittered_1_substep": (
                {"name": "j1", **jit, "jitter": 0.5, "num_substeps": 1},
                False,
            ),
            "unguarded_poisson_1_substep": (
                {"name": "p1", **poi, "max_gap": 4.0, "num_substeps": 1},
                False,
            ),
        }
    )
    return out


def episode_stats(x_obs: np.ndarray, x_int: np.ndarray, y_true: float, y_obs: float) -> dict:
    """Per-episode statistics shared by the built and the frozen episodes.

    Args:
        x_obs: Observational trajectory ``(T, N)``.
        x_int: Interventional trajectory ``(T, N)``.
        y_true: Interventional level at the query.
        y_obs: Observational level at the query.

    Returns:
        Finiteness, the largest ``|x|`` over both arms, and the query's levels.
    """
    finite = bool(np.isfinite(x_obs).all() and np.isfinite(x_int).all())
    both = np.abs(np.concatenate([x_obs.ravel(), x_int.ravel()]))
    return {
        "finite": finite,
        "maxabs": float(both.max()) if finite else float("inf"),
        "y_true": y_true,
        "y_obs": y_obs,
    }


def build(job: tuple[str, str, int]) -> dict:
    """Build one episode and return its statistics (runs in a worker).

    Args:
        job: ``(candidate label, structure, episode index)``.

    Returns:
        The episode's statistics, labelled.
    """
    import torch

    from dotime._build import episode_seed, make_episode
    from dotime._observation_grids import draw_grid
    from dotime.benchmarks import episode_from_sample
    from dotime.continuous import ContinuousExtendedPrior

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    label, structure, idx = job
    entry, guarded = candidates()[label]
    seed = episode_seed(_SUITE_SEED, idx)
    start = time.perf_counter()
    if guarded:
        spec = {"kind": "continuous", "idx": idx, "seed": seed, "T": _T, "structure": structure}
        ep = make_episode({**spec, "schedules": [entry]})
    else:
        # make_episode's own calls, minus check_schedule.
        torch.manual_seed(seed)
        times = draw_grid(entry, _T, seed)
        prior = ContinuousExtendedPrior(
            tscm_structure=structure,
            seed=seed,
            schedule="fixed",
            fixed_times=times,
            dt=float(times[-1] - times[0]) / (_T - 1),
            num_substeps=entry["num_substeps"],
        )
        ep = episode_from_sample(prior.generate_sample(T=_T), structure=structure, scm_id=idx)
    row, target = int(ep.query_time_idx[0]), int(ep.query_target[0])
    stats = episode_stats(
        ep.x_obs.numpy(), ep.x_int.numpy(), float(ep.y_true[0]), float(ep.x_obs[row, target])
    )
    return {
        **stats,
        "label": label,
        "structure": structure,
        "idx": idx,
        "seconds": time.perf_counter() - start,
        "x_obs": ep.x_obs.numpy().tobytes().hex() if label == "regular" else None,
    }


def frozen_episodes(cache: Path) -> dict[int, dict]:
    """Statistics of every released dot-Continuous-v1 1.0.0 episode.

    Args:
        cache: The cached suite directory.

    Returns:
        ``scm_id -> statistics``, plus the raw ``x_obs`` bytes for the paired
        identity check.
    """
    import pyarrow.parquet as pq

    out = {}
    for shard in sorted(cache.glob("shard-*.parquet")):
        cols = pq.read_table(shard).to_pydict()
        for i in range(len(cols["scm_id"])):
            length, n_vars = cols["length"][i], cols["n_vars"][i]
            x_obs = np.asarray(cols["x_obs"][i], dtype=np.float32).reshape(length, n_vars)
            x_int = np.asarray(cols["x_int"][i], dtype=np.float32).reshape(length, n_vars)
            # dot-Continuous-v1 declares index / (T - 1).
            row = round(cols["query_time"][i][0] * (length - 1))
            y_obs = float(x_obs[row, cols["query_target"][i][0]])
            stats = episode_stats(x_obs, x_int, cols["y_true"][i][0], y_obs)
            stats.update(structure=cols["structure"][i], x_obs=x_obs.tobytes().hex())
            out[cols["scm_id"][i]] = stats
    return out


def summarize(rows: list[dict]) -> dict:
    """Aggregate episode statistics and assert the per-arm target statistics.

    Args:
        rows: Episode statistics of one group.

    Returns:
        Finite fraction, ``max |x|`` quantiles, effect quantiles and the
        per-arm target statistics (nonzero fraction, mean, variance).

    Raises:
        AssertionError: If a group is empty, a level arm is less than half
            nonzero, or an arm's target statistics are not finite.
    """
    assert rows, "empty group"
    maxabs = np.array([r["maxabs"] for r in rows])
    y_true = np.array([r["y_true"] for r in rows])
    y_obs = np.array([r["y_obs"] for r in rows])
    arms = {"y_int": y_true, "y_obs": y_obs, "effect": y_true - y_obs}
    targets = {
        name: {
            "nonzero_frac": float((np.abs(v) > 1e-6).mean()),
            "mean": float(v.mean()),
            "var": float(v.var()),
        }
        for name, v in arms.items()
    }
    for name in ("y_int", "y_obs"):
        assert targets[name]["nonzero_frac"] > 0.5, (name, targets[name])
    for name, t in targets.items():
        assert np.isfinite([t["mean"], t["var"]]).all(), (name, t)
    effect = np.abs(arms["effect"])
    out = {
        "n": len(rows),
        "finite_frac": float(np.mean([r["finite"] for r in rows])),
        "maxabs_gt10_frac": float((maxabs > 10).mean()),
        "maxabs_median": float(np.median(maxabs)),
        "maxabs_p99": float(np.quantile(maxabs, 0.99)),
        "maxabs_max": float(maxabs.max()),
        "effect_abs_median": float(np.median(effect)),
        "effect_zero_frac": float((effect < 1e-6).mean()),
        "targets": targets,
    }
    if "seconds" in rows[0]:
        out["seconds_mean"] = float(np.mean([r["seconds"] for r in rows]))
    return out


def main() -> None:
    """Measure every candidate and write ``irregular_grids_measurement.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="Episodes per structure.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_HERE / "irregular_grids_measurement.json")
    args = parser.parse_args()

    labels = list(candidates())
    jobs = [
        (label, s, k * _PER + j)
        for label in labels
        for k, s in enumerate(_STRUCTURES)
        for j in range(args.n)
    ]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(build, jobs, chunksize=8))
    print(f"built {len(rows)} episodes in {time.perf_counter() - start:.0f} s", flush=True)

    frozen = frozen_episodes(args.cache)
    regular = [r for r in rows if r["label"] == "regular"]
    identical = sum(r["x_obs"] == frozen[r["idx"]]["x_obs"] for r in regular)
    report: dict = {
        "suite_seed": _SUITE_SEED,
        "episodes_per_structure": args.n,
        "config": str(_CONFIG.relative_to(_HERE.parents[2])),
        "schedules": {label: entry for label, (entry, _) in candidates().items()},
        "regular_x_obs_identical_to_release": f"{identical}/{len(regular)}",
        "groups": {},
    }
    for s in _STRUCTURES:
        rel = [v for v in frozen.values() if v["structure"] == s]
        report["groups"][f"{s} | released 1.0.0, all"] = summarize(rel)
        for label in labels:
            group = [r for r in rows if r["label"] == label and r["structure"] == s]
            report["groups"][f"{s} | {label}"] = summarize(group)
    report["groups"]["all | released 1.0.0, all"] = summarize(list(frozen.values()))
    for label in labels:
        report["groups"][f"all | {label}"] = summarize([r for r in rows if r["label"] == label])
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "groups"}, indent=2))
    for key, g in report["groups"].items():
        print(
            f"{key:55s} >10: {g['maxabs_gt10_frac']:6.1%}  p99 {g['maxabs_p99']:7.2f}  "
            f"max {g['maxabs_max']:9.3g}  zero effect {g['effect_zero_frac']:5.1%}"
        )


if __name__ == "__main__":
    main()
