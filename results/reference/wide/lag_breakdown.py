#!/usr/bin/env python
"""Per-lag breakdown of effect-sign accuracy on dot-Wide-v1.

dot-Generic-100k cannot answer how estimators fare across lag configurations:
its arms are independent noise draws, so its effect field measures regression to
the mean (see ``../audit_2026-09/generic_lag_breakdown.json``). dot-Wide-v1 pairs
its arms with shared noise, queries at the last step of the intervention window
and records the ground-truth lagged graph of every episode, so there the split
by lag is meaningful.

Builds dot-Wide-v1 1.0.0 exactly as the release build would (the config's own
per-episode specs and suite seed, through ``dotime._build.make_episode``, one
torch thread per worker process). It asserts the per-arm target statistics
(``dotime.qa.target_qa``, effect checked), then scores the CPU baselines in the
float32 arithmetic of ``dotime.reference.reference_table.run_baseline``. On the
full build it gates on the released rows: ``dir_n_valid`` and ``dir_acc`` of
every gated baseline must equal ``wide_cpu_effect.json`` exactly, and pooled
RMSE must agree to a relative 1e-4, which shows the local episodes are the
released ones. The RMSE tolerance is the one the frozen fingerprints use
(``dotime._fingerprint``): the hardening's spectral scaling and the
time-varying interventions go through the platform's linear algebra, sin and
exp, which differ across platforms by about 1e-5 relative. The largest RMSE
difference is recorded. Only then is the breakdown written. ``--cache`` keeps the
scored arrays, so a rerun skips the half-hour build.

Episodes are cut by the smallest summed lag from an intervened column to the
query (``metadata["graph"]["path"]``: 0, 1, 2, 3, >=4 or unreachable), by the
sampled maximum lag ``k_sampled`` (1 to 8), and by the lag order the graph
actually uses, ``k_eff``. Each cell reports the episode count, the share with
``|effect| >= 0.1``, the median ``|effect|``, and per baseline the effect-sign
accuracy with its ``n_valid`` and binomial standard error. TimeOLS is scored
too but not gated, since the released rows predate it.

    PYTHONPATH=src python results/reference/wide/lag_breakdown.py --workers 16
"""

from __future__ import annotations

import argparse
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

RMSE_REL_TOL = 1e-4

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "scripts" / "release_config_wide.yaml"
RELEASED = Path(__file__).with_name("wide_cpu_effect.json")
OUT = Path(__file__).with_name("wide_lag_breakdown.json")
SUITE = "dot-Wide-v1"
GATED = ("Zero", "Mean", "AR1", "VAR-OLS", "NaiveOLS")
ESTIMATORS = (*GATED, "TimeOLS")
EFFECT_EPS = 0.1
MIN_LAG_BINS = ("0", "1", "2", "3", ">=4", "unreachable")


def _init_worker() -> None:
    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)


def _make(spec: dict):
    from dotime._build import make_episode

    return make_episode(spec)


def _predict_shard(task: tuple[str, list]) -> np.ndarray:
    """One baseline's float32 predictions on a shard, as ``run_baseline`` makes them."""
    from dotime import baselines
    from dotime.observation import impute_episode

    name, episodes = task
    model = baselines.get(name)
    impute = not getattr(model, "mask_aware", False)
    out = []
    for ep in episodes:
        seen = impute_episode(ep) if impute else ep
        out.append(torch.as_tensor(model.predict(seen), dtype=torch.float32).reshape(-1).numpy())
    return np.concatenate(out)


def _min_lag(ep) -> int | None:
    """Smallest summed lag from any intervened column to the query, ``None`` if unreachable."""
    lags = [p["min_lag"] for p in ep.metadata["graph"]["path"] if p["reachable"]]
    return min(lags) if lags else None


def _lag_bin(min_lag: int | None) -> str:
    if min_lag is None:
        return "unreachable"
    return str(min_lag) if min_lag < 4 else ">=4"


def _cell(mask: np.ndarray, preds: dict[str, np.ndarray], tgt: np.ndarray, obs: np.ndarray) -> dict:
    """Counts, effect sizes and per-baseline effect-sign accuracy over ``mask``."""
    from dotime.evaluation import direction_accuracy

    eff = (tgt - obs)[mask]
    cell: dict = {
        "n_episodes": int(mask.sum()),
        "effect_ge_0.1_frac": float(np.mean(np.abs(eff) >= EFFECT_EPS)) if mask.any() else None,
        "median_abs_effect": float(np.median(np.abs(eff))) if mask.any() else None,
    }
    for name, pred in preds.items():
        da = direction_accuracy(
            torch.from_numpy(pred[mask] - obs[mask]), torch.from_numpy(tgt[mask] - obs[mask])
        )
        n, p = int(da["n_valid"]), float(da["accuracy"])
        cell[name] = {
            "dir_acc": p if n else None,
            "dir_n_valid": n,
            "dir_acc_se": math.sqrt(p * (1 - p) / n) if n else None,
        }
    return cell


def _build_and_score(specs: list[dict], workers: int):
    """Build the episodes, assert their targets, and score every estimator.

    Args:
        specs: The suite's per-episode specs.
        workers: Worker processes.

    Returns:
        ``(tgt, obs, preds, min_lag, k_sampled, k_eff, build_seconds, qa_passed)``.
    """
    from dotime.evaluation import query_obs_levels
    from dotime.qa import target_qa

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        episodes = list(pool.map(_make, specs, chunksize=max(1, len(specs) // (workers * 8))))
        build_s = time.time() - t0
        print(f"built {len(episodes)} episodes in {build_s:.0f} s", flush=True)
        report = target_qa(episodes, dir_target="effect", group_by=None, log=print)

        tgt = np.concatenate(
            [torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).numpy() for ep in episodes]
        )
        obs = np.concatenate([query_obs_levels(ep).cpu().numpy() for ep in episodes])
        shard = max(1, len(episodes) // (workers * 4))
        chunks = [episodes[i : i + shard] for i in range(0, len(episodes), shard)]
        preds: dict[str, np.ndarray] = {}
        for name in ESTIMATORS:
            preds[name] = np.concatenate(
                list(pool.map(_predict_shard, [(name, c) for c in chunks]))
            )
            print(f"scored {name}", flush=True)

    min_lag = [_lag_bin(_min_lag(ep)) for ep in episodes]
    k_sampled = [int(ep.metadata["graph"]["k_sampled"]) for ep in episodes]
    k_eff = [ep.metadata["graph"]["k_eff"] for ep in episodes]
    return tgt, obs, preds, min_lag, k_sampled, k_eff, build_s, report.passed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0, help="Episode-count scale; 1.0 = full.")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cache", type=Path, default=None, help="npz of the scored arrays.")
    args = ap.parse_args(argv)

    from dotime._build import episode_specs
    from dotime.evaluation import direction_accuracy
    from dotime.reference.reference_table import _pooled_rmse

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    suite_cfg = config["suites"][SUITE]
    suite_seed = int(suite_cfg["seed"])
    specs = episode_specs(suite_cfg, suite_seed, args.scale)
    if args.cache is not None and args.cache.exists():
        z = np.load(args.cache, allow_pickle=False)
        assert int(z["n"]) == len(specs), "cache is from a different scale"
        tgt, obs = z["tgt"], z["obs"]
        preds = {name: z[f"pred_{name}"] for name in ESTIMATORS}
        min_lag = [str(v) for v in z["min_lag"]]
        k_sampled = [int(v) for v in z["k_sampled"]]
        k_eff = [None if v < 0 else int(v) for v in z["k_eff"]]
        build_s, qa_passed = float(z["build_s"]), bool(z["qa_passed"])
        print(f"loaded {len(tgt)} scored episodes from {args.cache}", flush=True)
    else:
        (tgt, obs, preds, min_lag, k_sampled, k_eff, build_s, qa_passed) = _build_and_score(
            specs, args.workers
        )
        if args.cache is not None:
            np.savez(
                args.cache,
                n=len(specs),
                tgt=tgt,
                obs=obs,
                min_lag=np.asarray(min_lag),
                k_sampled=np.asarray(k_sampled),
                k_eff=np.asarray([-1 if k is None else k for k in k_eff]),
                build_s=build_s,
                qa_passed=qa_passed,
                **{f"pred_{name}": preds[name] for name in ESTIMATORS},
            )

    pooled = {}
    for name in ESTIMATORS:
        da = direction_accuracy(torch.from_numpy(preds[name] - obs), torch.from_numpy(tgt - obs))
        pooled[name] = {
            "pooled_rmse": _pooled_rmse(preds[name], tgt),
            "dir_acc": da["accuracy"],
            "dir_n_valid": da["n_valid"],
        }

    gate = "skipped (scale < 1)"
    rmse_rel_diff: dict[str, float] = {}
    if args.scale == 1.0:
        released = {r["baseline"]: r for r in json.loads(RELEASED.read_text())["rows"]}
        for name in GATED:
            ref, got = released[name], pooled[name]
            rmse_rel_diff[name] = abs(got["pooled_rmse"] - ref["pooled_rmse"]) / ref["pooled_rmse"]
            print(
                f"{name:9s} n_valid {got['dir_n_valid']} vs {ref['dir_n_valid']}, "
                f"dir_acc {got['dir_acc']:.7f} vs {ref['dir_acc']:.7f}, "
                f"rmse {got['pooled_rmse']:.7f} vs {ref['pooled_rmse']:.7f} "
                f"(rel {rmse_rel_diff[name]:.1e})",
                flush=True,
            )
        for name in GATED:
            ref, got = released[name], pooled[name]
            assert got["dir_n_valid"] == ref["dir_n_valid"], name
            assert math.isclose(got["dir_acc"], ref["dir_acc"], abs_tol=1e-7), name
            assert rmse_rel_diff[name] <= RMSE_REL_TOL, name
        gate = (
            f"{', '.join(GATED)} reproduce wide_cpu_effect.json: dir_n_valid and dir_acc "
            f"exactly, pooled RMSE within a relative {RMSE_REL_TOL:g} "
            f"(largest {max(rmse_rel_diff.values()):.1e})"
        )
    print(f"gate: {gate}", flush=True)

    splits = {
        "by_min_lag": (np.asarray(min_lag), list(MIN_LAG_BINS)),
        "by_k_sampled": (np.asarray(k_sampled), sorted(set(k_sampled))),
        "by_k_eff": (np.asarray(k_eff, dtype=object), sorted({k for k in k_eff if k is not None})),
    }
    breakdown: dict[str, dict] = {"all": _cell(np.ones(len(tgt), bool), preds, tgt, obs)}
    for split, (keys, bins) in splits.items():
        breakdown[split] = {str(b): _cell(keys == b, preds, tgt, obs) for b in bins}

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()
    out = {
        "suite": SUITE,
        "version": suite_cfg["version"],
        "suite_seed": suite_seed,
        "scale": args.scale,
        "n_episodes": len(tgt),
        "commit": commit,
        "protocol": "direction_accuracy(pred - y_obs, y_true - y_obs), float32, as run_baseline",
        "gate": gate,
        "rmse_rel_diff_vs_released": rmse_rel_diff,
        "target_qa_passed": qa_passed,
        "build_seconds": round(build_s, 1),
        "pooled": pooled,
        "breakdown": breakdown,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"{'min lag':12s} {'n':>6s} {'eff>=.1':>8s} " + " ".join(f"{n:>9s}" for n in ESTIMATORS))
    for b, cell in breakdown["by_min_lag"].items():
        accs = " ".join(
            f"{cell[n]['dir_acc']:9.3f}" if cell[n]["dir_acc"] is not None else f"{'n/a':>9s}"
            for n in ESTIMATORS
        )
        frac = cell["effect_ge_0.1_frac"]
        print(f"{b:12s} {cell['n_episodes']:6d} {frac if frac is not None else 0:8.3f} {accs}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
