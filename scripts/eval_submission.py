#!/usr/bin/env python
"""Evaluate a model against a DoTime suite and emit a submission JSON.

This is the reproducible submission path for the public leaderboard. A model is
any object exposing ``predict(episode) -> Tensor`` (the :class:`Baseline`
protocol); reference it by import path ``module:attr`` (a class or zero-arg
factory), or by a registered baseline name.

Usage
-----
    # a registered baseline
    python scripts/eval_submission.py --suite dot-Identifiability-v1 --baseline VAR-OLS \\
        --out submission.json
    # a custom model class
    python scripts/eval_submission.py --suite dot-Continuous-v1 \\
        --model mypkg.models:MyModel --name MyModel --out submission.json

The output JSON conforms to the schema in ``docs/leaderboard.md`` (suite, model,
pooled + per-structure metrics, package version) and can be submitted as-is.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from dotime import __version__, baselines, evaluation
from dotime.benchmarks import load_benchmark


def _load_model(model_path: str | None, baseline: str | None, name: str | None):
    if baseline:
        return baselines.get(baseline), name or baseline
    if not model_path:
        raise SystemExit("pass either --baseline NAME or --model module:attr")
    module_name, _, attr = model_path.partition(":")
    if not attr:
        raise SystemExit("--model must be 'module:attr'")
    obj = getattr(importlib.import_module(module_name), attr)
    model = obj() if callable(obj) else obj
    return model, name or getattr(model, "name", attr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--baseline", default=None, help="A registered baseline name.")
    parser.add_argument("--model", default=None, help="Custom model as 'module:attr'.")
    parser.add_argument("--name", default=None, help="Display name for the leaderboard.")
    parser.add_argument("--out", type=Path, default=Path("submission.json"))
    args = parser.parse_args(argv)

    model, name = _load_model(args.model, args.baseline, args.name)
    suite = load_benchmark(args.suite)
    results = evaluation.evaluate(model, suite)

    payload = results.to_dict()
    payload["model"] = name
    payload["package_version"] = __version__
    payload["schema"] = "ctp-submission/1"
    args.out.write_text(json.dumps(payload, indent=2))
    print(results.summary())
    print(f"\n[eval_submission] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
