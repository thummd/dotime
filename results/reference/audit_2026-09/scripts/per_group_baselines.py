#!/usr/bin/env python
"""Effect-sign accuracy of the CPU baselines per group of a suite.

The reference table pools a suite and splits it by structure. The 2026-10 suites
carry a second design factor in their metadata, the observation cell of
dot-Observed-v1, the grid schedule of dot-ContinuousIrregular-v1 or the driven
label of dot-SeasonalTrend-v1, and this script scores each level of that factor
with the packaged protocol (:func:`dotime.evaluation.evaluate`, effect target,
imputation for models that cannot read masks).

Usage::

    python per_group_baselines.py --suite dot-Observed-v1 --version 1.0.0 \
        --group obs_cell --out ../observed_per_cell.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotime import baselines
from dotime.benchmarks import BenchmarkSuite, load_benchmark
from dotime.evaluation import evaluate
from dotime.qa import target_qa

BASELINES = ("Mean", "AR1", "VAR-OLS", "NaiveOLS", "BackDoorOLS", "IV2SLS")


def main() -> None:
    """Score every baseline on every group and write one JSON file.

    Raises:
        SystemExit: If the suite's targets fail the per-arm QA.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--group", required=True, help="Metadata key, or 'structure'.")
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    suite = load_benchmark(args.suite, version=args.version)
    episodes = list(suite)
    # The same gate as every evaluator: pooled and per-structure target statistics.
    report = target_qa(episodes, dir_target="effect")
    if not report.passed:
        raise SystemExit(f"target QA failed: {report.problems}")
    groups: dict[str, list] = {}
    for ep in episodes:
        key = ep.structure if args.group == "structure" else str(ep.metadata.get(args.group))
        groups.setdefault(key, []).append(ep)
    out = {
        "suite": args.suite,
        "version": args.version,
        "group": args.group,
        "protocol": "dotime.evaluation.evaluate(dir_target='effect', impute=True)",
        "n_episodes": len(episodes),
        "groups": {},
    }
    for key, eps in groups.items():
        sub = BenchmarkSuite(suite.meta, eps)
        rows = {}
        for name in args.baselines:
            res = evaluate(baselines.get(name), sub, dir_target="effect", impute=True)
            rows[name] = {
                k: v for k, v in res.pooled.items() if k in ("dir_acc", "dir_n_valid", "rmse")
            }
        out["groups"][key] = {"n_episodes": len(eps), "baselines": rows}
        print(
            f"[{key}] n={len(eps)} "
            + " ".join(f"{n}={rows[n].get('dir_acc', float('nan')):.3f}" for n in args.baselines),
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
