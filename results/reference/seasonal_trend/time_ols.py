#!/usr/bin/env python
"""TimeOLS on dot-SeasonalTrend-v1: does modelling time recover hidden drivers?

Builds dot-SeasonalTrend-v1 1.0.0 exactly as the release build would (the
config's own per-episode specs and suite seed, through
``dotime._build.make_episode``, one torch thread per worker process) and scores
Mean, NaiveOLS, BackDoorOLS and TimeOLS per label with the released protocol,
``dotime.evaluation.evaluate(dir_target="effect", impute=True)``.

Before anything is scored it asserts the per-arm target statistics of every
label (``dotime.qa.target_qa``, effect checked). On the full build it then
gates on the released rows: the Mean, NaiveOLS and BackDoorOLS accuracies and
``n_valid`` of every label must equal ``seasonal_trend_per_label.json`` exactly,
which shows the local episodes are the released ones.

Per label it reports each estimator's effect-sign accuracy with its binomial
standard error, and TimeOLS minus NaiveOLS with a paired episode bootstrap 95%
interval (2,000 resamples of the valid episodes, ``default_rng(20261001)`` per
label, as in ``../detection_power_2026-10``). The per-episode signs are
recomputed in float32 and asserted to reproduce ``evaluate``'s accuracy.

The official score compares ``pred(v) - y_obs`` with the true effect, so it
rewards a good forecast of the factual level at the query as well as a good
effect estimate. To separate the two it also reports the sign of each
estimator's own effect, ``pred(do A = v) - pred(do A = a_ref)`` with ``a_ref``
the factual treatment at the intervened row, on the same valid episodes (a zero
contrast names no direction and counts as wrong), as the exploratory table of
``../detection_power_2026-10`` does.

    PYTHONPATH=src python results/reference/seasonal_trend/time_ols.py --workers 16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "scripts" / "release_config_seasonal_trend.yaml"
RELEASED = Path(__file__).with_name("seasonal_trend_per_label.json")
OUT = Path(__file__).with_name("seasonal_trend_time_ols.json")
SUITE = "dot-SeasonalTrend-v1"
ESTIMATORS = ("Mean", "NaiveOLS", "BackDoorOLS", "TimeOLS")
GATED = ("Mean", "NaiveOLS", "BackDoorOLS")
BOOT_N, BOOT_SEED = 2000, 20261001


def _init_worker() -> None:
    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)


def _make(spec: dict):
    from dotime._build import make_episode

    return make_episode(spec)


def _signs_correct(model, episodes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-episode effect-sign correctness, validity and own-effect correctness.

    Args:
        model: A baseline with ``predict(episode)``.
        episodes: Single-query episodes.

    Returns:
        ``(correct, valid, own)`` boolean arrays over the episodes: the official
        sign test as ``direction_accuracy`` scores it, ``|effect| >= 0.1``, and
        the sign of ``pred(do v) - pred(do a_ref)`` against the true effect.
    """
    from dotime.evaluation import DIR_ACC_EPS, query_obs_levels
    from dotime.observation import impute_episode

    correct, valid, own = [], [], []
    for ep in episodes:
        scored = ep if getattr(model, "mask_aware", False) else impute_episode(ep)
        pred = torch.as_tensor(model.predict(scored), dtype=torch.float32).reshape(-1)
        a_ref = float(ep.x_obs[min(ep.intervention.times), ep.intervention.targets[0]])
        at_ref = dataclasses.replace(
            scored, intervention=dataclasses.replace(scored.intervention, values=a_ref)
        )
        pred_ref = torch.as_tensor(model.predict(at_ref), dtype=torch.float32).reshape(-1)
        tgt = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1)
        obs = query_obs_levels(ep).reshape(-1)
        eff_t, eff_p = tgt - obs, pred - obs
        valid.append(bool(eff_t.abs()[0] >= DIR_ACC_EPS))
        correct.append(bool(eff_p.sign()[0] == eff_t.sign()[0]))
        own.append(bool((pred - pred_ref).sign()[0] == eff_t.sign()[0]))
    return np.asarray(correct), np.asarray(valid), np.asarray(own)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--scale", type=float, default=1.0, help="Episode-count scale; 1.0 = full suite."
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    from dotime import baselines
    from dotime._build import episode_specs
    from dotime.benchmarks import _SUITE_REGISTRY, BenchmarkSuite
    from dotime.evaluation import evaluate
    from dotime.qa import target_qa

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    suite_cfg = config["suites"][SUITE]
    suite_seed = int(suite_cfg["seed"])
    specs = episode_specs(suite_cfg, suite_seed, args.scale)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        episodes = list(pool.map(_make, specs, chunksize=max(1, len(specs) // (args.workers * 8))))
    build_s = time.time() - t0
    print(f"built {len(episodes)} episodes in {build_s:.0f} s")

    report = target_qa(episodes, dir_target="effect", group_by="structure", log=print)
    labels = list(suite_cfg["structures"])
    by_label = {lab: [ep for ep in episodes if ep.structure == lab] for lab in labels}

    meta = _SUITE_REGISTRY[SUITE]
    results: dict[str, dict] = {lab: {"n_episodes": len(eps)} for lab, eps in by_label.items()}
    correct: dict[str, dict[str, np.ndarray]] = {lab: {} for lab in labels}
    valid: dict[str, np.ndarray] = {}
    for name in ESTIMATORS:
        model = baselines.get(name)
        official = evaluate(model, BenchmarkSuite(meta, episodes), dir_target="effect", impute=True)
        for lab in labels:
            c, v, o = _signs_correct(model, by_label[lab])
            got = official.per_structure[lab]
            n_valid = int(v.sum())
            acc = float(c[v].mean()) if n_valid else float("nan")
            assert n_valid == got["dir_n_valid"], (name, lab, n_valid, got["dir_n_valid"])
            assert math.isclose(acc, got["dir_acc"], abs_tol=1e-6), (name, lab, acc, got["dir_acc"])
            correct[lab][name] = c
            valid.setdefault(lab, v)
            results[lab][name] = {
                "dir_acc": got["dir_acc"],
                "dir_acc_se": got["dir_acc_se"],
                "dir_n_valid": got["dir_n_valid"],
                "rmse": got["rmse"],
                "own_effect_sign_acc": float(o[v].mean()) if n_valid else float("nan"),
            }
        print(f"scored {name}")

    gate = "skipped (scale < 1)"
    if args.scale == 1.0:
        released = json.loads(RELEASED.read_text(encoding="utf-8"))["groups"]
        for lab in labels:
            for name in GATED:
                ref = released[lab]["baselines"][name]
                got = results[lab][name]
                assert got["dir_n_valid"] == ref["dir_n_valid"], (lab, name)
                assert math.isclose(got["dir_acc"], ref["dir_acc"], abs_tol=1e-7), (lab, name)
        gate = f"{', '.join(GATED)} reproduce seasonal_trend_per_label.json exactly on every label"
    print(f"gate: {gate}")

    for lab in labels:
        v = valid[lab]
        a = correct[lab]["TimeOLS"][v].astype(float)
        b = correct[lab]["NaiveOLS"][v].astype(float)
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(a), size=(BOOT_N, len(a)))
        boots = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        results[lab]["TimeOLS_minus_NaiveOLS"] = {
            "diff": float(a.mean() - b.mean()),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        }

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()
    out = {
        "suite": SUITE,
        "version": suite_cfg["version"],
        "suite_seed": suite_seed,
        "scale": args.scale,
        "n_episodes": len(episodes),
        "commit": commit,
        "protocol": "dotime.evaluation.evaluate(dir_target='effect', impute=True)",
        "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "paired": True},
        "gate": gate,
        "target_qa_passed": report.passed,
        "build_seconds": round(build_s, 1),
        "labels": results,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"{'label':32s} " + " ".join(f"{n:>11s}" for n in ESTIMATORS) + "   TimeOLS-Naive [95% CI]"
    )
    for lab in labels:
        row = results[lab]
        d = row["TimeOLS_minus_NaiveOLS"]
        accs = " ".join(f"{row[n]['dir_acc']:11.3f}" for n in ESTIMATORS)
        print(f"{lab:32s} {accs}   {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print("own-effect sign accuracy (pred(do v) - pred(do a_ref)):")
    for lab in labels:
        own = " ".join(f"{results[lab][n]['own_effect_sign_acc']:11.3f}" for n in ESTIMATORS)
        print(f"{lab:32s} {own}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
