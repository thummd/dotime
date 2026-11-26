#!/usr/bin/env python
"""Where does an intervention still act in dot-RegimeSwitch-v1 version 1.1.0?

Version 1.1.0 pairs its arms under shared noise and a shared regime path, so the
gap between the arms of an episode is the effect of its intervention and
nothing else. This script reads that effect off the first episodes of every
regime-density block and reports how much of it is left at the last step and at
the last step of the intervention window. The stabilised dynamics of the suite
are contractive, and most windows close long before the last step, so the two
query rules differ in whether the benchmark has anything to score.

It builds the audited episodes from the release config, exactly as
``scripts/build_release.py`` does, once with the config's query rule and once
with the query at the last step. Per-arm target statistics are logged and
asserted first (``dotime.qa.target_qa``), as the repository requires of every
generation check.

    python results/reference/regime_v1_1/scripts/effect_persistence.py \\
        --per-block 200 --workers 10 --out results/reference/regime_v1_1/design_pilot.json

The version of this script for independent arms, which had to simulate each
interventional arm twice under common random numbers, is in the history of the
repository (commit 488ebc7). Its results are in ``../independent_arms/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = ROOT / "scripts" / "release_config_regime_v1_1.yaml"
SUITE = "dot-RegimeSwitch-v1"


def suite_config(path: Path) -> tuple[dict[str, Any], int]:
    """Read the suite config and its seed from a release config.

    Args:
        path: The release config.

    Returns:
        ``(cfg, suite_seed)`` for :data:`SUITE`.

    Raises:
        KeyError: If the config does not define the suite or its seed.
    """
    cfg = yaml.safe_load(path.read_text())["suites"][SUITE]
    return cfg, int(cfg["seed"])


def audit_specs(cfg: dict[str, Any], seed: int, per_block: int | None) -> list[dict[str, Any]]:
    """The specs to audit: the first episodes of every regime-density block.

    Args:
        cfg: The suite config.
        seed: The suite seed.
        per_block: Episodes per density block, or ``None`` for all of them.

    Returns:
        Specs in suite order.
    """
    from dotime._build import episode_specs

    specs = episode_specs(cfg, seed, 1.0)
    if per_block is None:
        return specs
    kept: dict[int, int] = {}
    out = []
    for spec in specs:
        if kept.get(spec["num_regimes"], 0) < per_block:
            kept[spec["num_regimes"]] = kept.get(spec["num_regimes"], 0) + 1
            out.append(spec)
    return out


def audit_episode(spec: dict[str, Any]) -> dict[str, Any]:
    """Read the effect of one episode's intervention off its two arms.

    Args:
        spec: A ``regime_configured`` spec of the suite with ``pair_mode: counterfactual``.

    Returns:
        The episode's record: build facts (resampled, divergence, intervention
        type, window, regime switches), whether the arms agree before the
        onset, and the absolute effect on the queried variable under both
        query rules.

    Raises:
        AssertionError: If the two query rules do not share their trajectories.
    """
    from dotime._build import make_episode

    warnings.simplefilter("ignore", RuntimeWarning)
    torch.set_num_threads(1)
    built = make_episode(spec)
    # The same pair queried at the last step, where the most affected variable
    # may be another one.
    last_rule = make_episode({k: v for k, v in spec.items() if k != "query_row"})
    assert torch.equal(last_rule.x_obs, built.x_obs)
    assert torch.equal(last_rule.x_int, built.x_int)
    first_attempt = make_episode({**spec, "stability_retries": 0})
    x_obs, x_int, iv = built.x_obs, built.x_int, built.intervention
    t_len, n = x_obs.shape
    onset, window_end = min(iv.times), max(iv.times)
    path = np.asarray(built.metadata["graph"]["regime_path"])
    q_end, q_last = int(built.query_target[0]), int(last_rule.query_target[0])
    return {
        "idx": int(spec["idx"]),
        "num_regimes": int(spec["num_regimes"]),
        "resampled": not torch.equal(first_attempt.x_int, x_int),
        "diverged": bool(built.metadata["diverged"]),
        "type": iv.intervention_type.name,
        "n_vars": int(n),
        "n_targets": len(iv.targets),
        "onset": int(onset),
        "window_end": int(window_end),
        "steps_after_window": int(t_len - 1 - window_end),
        "pre_onset_equal": bool(torch.equal(x_obs[:onset], x_int[:onset])),
        "reads_parents": bool(built.metadata["graph"]["reads_parents"]),
        "query_reachable": bool(built.metadata["graph"]["path"][0]["reachable"]),
        "query_min_lag": built.metadata["graph"]["path"][0]["min_lag"],
        "regime_switches": int((np.diff(path) != 0).sum()),
        "regimes_visited": len(set(path.tolist())),
        "effect_at_window_end": float((x_int[window_end, q_end] - x_obs[window_end, q_end]).abs()),
        "effect_at_last_step": float((x_int[t_len - 1, q_last] - x_obs[t_len - 1, q_last]).abs()),
    }


def summarise(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Aggregate the per-episode records.

    Args:
        rows: Records of :func:`audit_episode`.
        threshold: Effect threshold of the benchmark.

    Returns:
        Counts and shares, pooled, per intervention type and per regime density.
    """
    kept = [r for r in rows if not r["diverged"]]

    def block(sub: list[dict[str, Any]]) -> dict[str, Any]:
        """Effect statistics of one subset of the records."""
        out: dict[str, Any] = {"n": len(sub)}
        for key in ("effect_at_window_end", "effect_at_last_step"):
            values = np.asarray([r[key] for r in sub])
            out[f"{key}_share"] = float(np.mean(values >= threshold)) if len(sub) else None
            out[f"{key}_median"] = float(np.median(values)) if len(sub) else None
            out[f"{key}_zero_share"] = float(np.mean(values == 0.0)) if len(sub) else None
        return out

    gap = np.asarray([r["steps_after_window"] for r in kept])
    return {
        "n_episodes": len(rows),
        "n_resampled": sum(r["resampled"] for r in rows),
        "n_diverged": len(rows) - len(kept),
        "n_reads_parents": sum(r["reads_parents"] for r in rows),
        "n_pre_onset_equal": sum(r["pre_onset_equal"] for r in rows),
        "intervention_types": {
            t: sum(r["type"] == t for r in kept) for t in ("HARD", "SOFT", "TIME_VARYING")
        },
        "steps_after_window_median": float(np.median(gap)),
        "window_closes_at_last_step_share": float(np.mean(gap == 0)),
        "query_reachable_share": float(np.mean([r["query_reachable"] for r in kept])),
        "regime_switches_median": float(np.median([r["regime_switches"] for r in kept])),
        "no_regime_switch_share": float(np.mean([r["regime_switches"] == 0 for r in kept])),
        "threshold": threshold,
        "pooled": block(kept),
        "by_type": {
            t: block([r for r in kept if r["type"] == t]) for t in ("HARD", "SOFT", "TIME_VARYING")
        },
        "by_num_regimes": {
            str(d): block([r for r in kept if r["num_regimes"] == d])
            for d in sorted({r["num_regimes"] for r in rows})
        },
    }


def assert_targets(specs: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    """Log and assert the per-arm target statistics of the audited episodes.

    Args:
        specs: The specs to build.
        workers: Worker processes.

    Returns:
        The QA report as a dict.

    Raises:
        dotime.qa.TargetQAError: If a target check fails.
    """
    from dotime._build import make_episode
    from dotime.qa import target_qa

    with Pool(workers) as pool:
        episodes = pool.map(make_episode, specs, chunksize=16)
    return target_qa(episodes, dir_target="effect", group_by="structure").to_dict()


def main(argv: list[str] | None = None) -> None:
    """Run the audit and write its JSON.

    Args:
        argv: Command line, ``sys.argv[1:]`` by default.

    Raises:
        SystemExit: If the suite config does not pair its arms under shared noise.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument(
        "--per-block", type=int, default=None, help="Episodes per density, all if unset."
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--threshold", type=float, default=None, help="Default: evaluation.DIR_ACC_EPS."
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import dotime
    from dotime import evaluation

    threshold = float(evaluation.DIR_ACC_EPS if args.threshold is None else args.threshold)
    cfg, seed = suite_config(args.config)
    if cfg.get("pair_mode") != "counterfactual":
        raise SystemExit(
            "this audit reads the effect off shared-noise arms: pair_mode is not counterfactual"
        )
    specs = audit_specs(cfg, seed, args.per_block)
    print(f"[effect_persistence] {len(specs)} episodes of {SUITE} {cfg['version']}, seed {seed}")
    qa = assert_targets(specs, args.workers)
    with Pool(args.workers) as pool:
        rows = pool.map(audit_episode, specs, chunksize=8)
    summary = summarise(rows, threshold)
    # The audit only means something on repaired episodes whose arms share their noise.
    assert summary["n_reads_parents"] == len(rows), "an episode does not read its parents"
    assert summary["n_pre_onset_equal"] == len(rows), (
        "the arms of an episode differ before the onset"
    )
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()
    result = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
        "dotime_file": str(Path(dotime.__file__).resolve().relative_to(ROOT)),
        "suite": SUITE,
        "version": cfg["version"],
        "config": str(args.config.resolve().relative_to(ROOT)),
        "config_keys_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[
            :16
        ],
        "suite_seed": seed,
        "per_block": args.per_block,
        "target_qa": {k: qa[k] for k in ("mode", "passed", "problems") if k in qa},
        "summary": summary,
        "episodes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    pooled = summary["pooled"]
    print(
        f"[effect_persistence] resampled {summary['n_resampled']}, diverged {summary['n_diverged']}, "
        f"arms equal before the onset in {summary['n_pre_onset_equal']} of {len(rows)}"
    )
    print(
        f"[effect_persistence] effect on the query >= {threshold}: last step "
        f"{pooled['effect_at_last_step_share']:.3f} (median {pooled['effect_at_last_step_median']:.4f}), "
        f"window end {pooled['effect_at_window_end_share']:.3f} "
        f"(median {pooled['effect_at_window_end_median']:.4f})"
    )
    print(f"[effect_persistence] wrote {args.out}")


if __name__ == "__main__":
    main()
