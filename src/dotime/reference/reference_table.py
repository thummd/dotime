#!/usr/bin/env python
"""Release-scale reference-table eval for the DoTime paper (Table 3).

Runs the registered CPU baselines (and, if a checkpoint is given, the
Do-Over-Time-PFN) over a full suite, reporting pooled RMSE and direction
accuracy with an episode-cluster bootstrap CI on the pooled RMSE.

    dotime-eval-reference --suite dot-Identifiability-v1 --out ident.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dotime import baselines
from dotime.benchmarks import load_benchmark
from dotime.evaluation import direction_accuracy

CPU_BASELINES = ["Zero", "Mean", "AR1", "VAR-OLS", "BackDoorOLS", "IV2SLS", "Oracle"]


def _pooled_rmse(pred: np.ndarray, tgt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - tgt) ** 2)))


def _cluster_bootstrap_rmse(ep_pred, ep_tgt, n_boot=1000, seed=0):
    """Episode-cluster bootstrap CI for pooled RMSE."""
    rng = np.random.default_rng(seed)
    m = len(ep_pred)
    # precompute per-episode summed sq error and count for fast pooling
    sse = np.array([float(np.sum((p - t) ** 2)) for p, t in zip(ep_pred, ep_tgt, strict=True)])
    cnt = np.array([len(t) for t in ep_tgt], dtype=np.float64)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        boot[b] = np.sqrt(sse[idx].sum() / cnt[idx].sum())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def run_baseline(name, suite_episodes, checkpoint=None, device="cpu"):
    if name == "DoOverTimePFN":
        model = baselines.get(name, checkpoint=checkpoint, device=device)
    else:
        model = baselines.get(name)
    ep_pred, ep_tgt = [], []
    for ep in suite_episodes:
        p = torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1).cpu().numpy()
        t = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).cpu().numpy()
        ep_pred.append(p)
        ep_tgt.append(t)
    pred = np.concatenate(ep_pred)
    tgt = np.concatenate(ep_tgt)
    rmse = _pooled_rmse(pred, tgt)
    lo, hi = _cluster_bootstrap_rmse(ep_pred, ep_tgt)
    da = direction_accuracy(torch.from_numpy(pred), torch.from_numpy(tgt))
    return {
        "baseline": name,
        "n_episodes": len(ep_pred),
        "n_queries": int(pred.size),
        "pooled_rmse": rmse,
        "rmse_ci95": [lo, hi],
        "dir_acc": da["accuracy"],
        "dir_n_valid": da["n_valid"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--baselines", nargs="+", default=CPU_BASELINES)
    ap.add_argument("--pfn-checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    episodes = list(load_benchmark(args.suite))
    print(f"[{args.suite}] loaded {len(episodes)} episodes in {time.time() - t0:.1f}s")

    rows = []
    todo = list(args.baselines)
    if args.pfn_checkpoint:
        todo += ["DoOverTimePFN"]
    for name in todo:
        t = time.time()
        try:
            row = run_baseline(name, episodes, checkpoint=args.pfn_checkpoint, device=args.device)
        except Exception as ex:  # keep going; report the failure
            print(f"  {name:14s} FAILED: {ex}")
            rows.append({"baseline": name, "error": str(ex)})
            continue
        rows.append(row)
        print(
            f"  {name:14s} RMSE={row['pooled_rmse']:8.3f} "
            f"[{row['rmse_ci95'][0]:.3f},{row['rmse_ci95'][1]:.3f}] "
            f"dir_acc={row['dir_acc']:.3f}  ({time.time() - t:.1f}s)"
        )

    out = {"suite": args.suite, "n_episodes": len(episodes), "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
