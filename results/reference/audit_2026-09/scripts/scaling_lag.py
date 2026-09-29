#!/usr/bin/env python
"""Generator scaling and a per-lag baseline breakdown.

Samples fresh episodes from the released generic prior (``DoTime.generate_pair``,
the path that built dot-Generic-100k) under config overrides of (N_max, K_max),
with the same per-episode seeding and ``stability_retries`` semantics as
``dotime._build.make_episode``. The in-memory SCM supplies the realised number
of variables N and lag order K, which the released suites do not store.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CONFIGS = [(10, 3), (10, 8), (60, 8)]
BASELINES = ["Zero", "Mean", "AR1", "VAR-OLS"]
SUITE_SEED = 20260928


def _lag_order(scm) -> tuple[int | None, int | None, str]:
    """Return (sampled K, deepest lag with an edge, SCM class name).

    Args:
        scm: The sampled SCM returned by ``generate_pair``.

    Returns:
        Tuple of the sampled lag order, the effective lag order, and the class.
    """
    name = type(scm).__name__
    k = getattr(scm, "_K", None)
    g = getattr(scm, "_G_lags", None)
    k_eff = None
    if g is not None:
        nz = [i + 1 for i, gk in enumerate(g) if float(np.asarray(gk).sum()) > 0]
        k_eff = max(nz) if nz else 0
    return (int(k) if k is not None else None), k_eff, name


def one(spec: dict) -> dict:
    """Generate and score one episode (runs in a worker process).

    Args:
        spec: Dict with idx, n_max, k_max, retries and T.

    Returns:
        Per-episode record with N, K, divergence flag, timing, targets and
        baseline predictions.
    """
    import torch

    from dotime import DoTime, baselines
    from dotime.benchmarks import episode_from_pair
    from dotime.evaluation import query_obs_levels
    from dotime.utils import DEFAULT_CONFIG

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    cfg = {**DEFAULT_CONFIG, "N_max": spec["n_max"], "K_max": spec["k_max"]}
    seed = (SUITE_SEED * 1_000_003 + spec["idx"]) & 0x7FFFFFFF
    t0 = time.perf_counter()
    attempts = 0
    for attempt in range(spec["retries"] + 1):
        s = seed if attempt == 0 else seed * 100003 + attempt
        torch.manual_seed(s)
        x_obs, x_int, iv, scm = DoTime(config=cfg, seed=s).generate_pair(T=spec["T"])
        attempts = attempt + 1
        diverged = float(x_obs.abs().max()) == 0.0 and float(x_int.abs().max()) == 0.0
        if attempt == spec["retries"] or not diverged:
            break
    gen_s = time.perf_counter() - t0
    k, k_eff, cls = _lag_order(scm)
    rec = {
        "idx": spec["idx"],
        "N": int(x_obs.shape[1]),
        "K": k,
        "K_eff": k_eff,
        "cls": cls,
        "diverged": bool(diverged),
        "attempts": attempts,
        "gen_s": gen_s,
    }
    if not diverged:
        ep = episode_from_pair(x_obs, x_int, iv, scm_id=spec["idx"])
        rec["y_int"] = float(ep.y_true.reshape(-1)[0])
        rec["y_obs"] = float(query_obs_levels(ep)[0])
        preds = {}
        for b in BASELINES:
            try:
                preds[b] = float(torch.as_tensor(baselines.get(b).predict(ep)).reshape(-1)[0])
            except Exception as ex:  # record, keep going
                preds[b] = None
                rec.setdefault("errors", {})[b] = str(ex)[:120]
        rec["pred"] = preds
    return rec


def arm_stats(x: np.ndarray) -> dict:
    """Nonzero fraction, mean and variance of one target arm.

    Args:
        x: 1-D array of targets.

    Returns:
        Summary dict.
    """
    return {
        "n": int(x.size),
        "nonzero_frac": float(np.mean(x != 0)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
    }


def score(recs: list[dict]) -> dict:
    """Per-baseline RMSE, NMSE and level-direction accuracy on non-diverged records.

    Args:
        recs: Worker records.

    Returns:
        Dict keyed by baseline name.
    """
    ok = [r for r in recs if not r["diverged"]]
    if len(ok) < 20:
        return {"n": len(ok)}
    y = np.array([r["y_int"] for r in ok])
    out = {"n": len(ok)}
    for b in BASELINES:
        pr = np.array([r["pred"].get(b) if r["pred"].get(b) is not None else np.nan for r in ok])
        m = np.isfinite(pr)
        mse = float(np.mean((pr[m] - y[m]) ** 2))
        valid = np.abs(y[m]) >= 0.1
        out[b] = {
            "n_scored": int(m.sum()),
            "rmse": float(np.sqrt(mse)),
            "nmse": mse / float(np.var(y[m])),
            "dir_level": float(np.mean(np.sign(pr[m][valid]) == np.sign(y[m][valid])))
            if valid.any()
            else None,
        }
    return out


def main() -> None:
    """Run every configuration, assert target QA, and write the JSON output.

    Raises:
        AssertionError: If the non-diverged interventional or observational
            targets of a configuration fail the nonzero-fraction floor.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--out", type=Path, default=Path("results/reference/audit_2026-09/scaling_lag.json")
    )
    args = ap.parse_args()
    results = {"suite_seed": SUITE_SEED, "T": args.T, "n_per_config": args.n, "runs": []}
    with Pool(args.workers) as pool:
        for n_max, k_max in CONFIGS:
            for retries in (0, 3):
                t0 = time.perf_counter()
                specs = [
                    {"idx": i, "n_max": n_max, "k_max": k_max, "retries": retries, "T": args.T}
                    for i in range(args.n)
                ]
                recs = pool.map(one, specs, chunksize=4)
                wall = time.perf_counter() - t0
                ok = [r for r in recs if not r["diverged"]]
                y_int = np.array([r["y_int"] for r in ok])
                y_obs = np.array([r["y_obs"] for r in ok])
                qa = {
                    "y_int": arm_stats(y_int),
                    "y_obs": arm_stats(y_obs),
                    "effect": arm_stats(y_int - y_obs),
                }
                print(f"[QA] N_max={n_max} K_max={k_max} retries={retries}: {qa}")
                assert qa["y_int"]["nonzero_frac"] >= 0.5, qa
                assert qa["y_obs"]["nonzero_frac"] >= 0.5, qa
                by_k = {}
                for k in sorted({r["K"] for r in recs if r["K"] is not None}):
                    rk = [r for r in recs if r["K"] == k and r["cls"] == "TemporalSCM"]
                    if not rk:
                        continue
                    by_k[str(k)] = {
                        "n": len(rk),
                        "zeroed_frac": float(np.mean([r["diverged"] for r in rk])),
                        "scores": score(rk),
                    }
                Ns = [r["N"] for r in recs]
                run = {
                    "N_max": n_max,
                    "K_max": k_max,
                    "stability_retries": retries,
                    "wall_s": wall,
                    "workers": args.workers,
                    "mean_gen_s_per_episode": float(np.mean([r["gen_s"] for r in recs])),
                    "zeroed_frac": float(np.mean([r["diverged"] for r in recs])),
                    "N_range": [int(min(Ns)), int(max(Ns))],
                    "N_median": float(np.median(Ns)),
                    "class_counts": {
                        c: sum(r["cls"] == c for r in recs)
                        for c in sorted({r["cls"] for r in recs})
                    },
                    "mean_attempts": float(np.mean([r["attempts"] for r in recs])),
                    "target_qa": qa,
                    "scores_all": score(recs),
                    "by_K": by_k,
                    "baseline_errors": sum(1 for r in recs if r.get("errors")),
                }
                results["runs"].append(run)
                print(
                    f"N_max={n_max:2d} K_max={k_max} retries={retries}: zeroed={run['zeroed_frac']:.3f} "
                    f"wall={wall:.1f}s gen/ep={run['mean_gen_s_per_episode']:.3f}s N={run['N_range']} "
                    f"attempts={run['mean_attempts']:.2f} byK_zeroed="
                    + ",".join(f"{k}:{v['zeroed_frac']:.2f}" for k, v in by_k.items())
                )
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
