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
from dotime.evaluation import direction_accuracy, query_obs_levels, realign_episode

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


def _episode_obs_levels(ep, realignment):
    """Per-query observational levels, preferring the realignment sidecar.

    Args:
        ep: The episode being scored.
        realignment: Optional ``{scm_id: y_obs}`` map from the released v1
            realignment sidecar (required for archived suites whose ``x_obs``
            is column-misaligned; see the datasheet erratum).

    Returns:
        1-D numpy array of observational levels, one per query.
    """
    if realignment is not None and ep.scm_id in realignment:
        return np.asarray([realignment[ep.scm_id]["y_obs_corrected"]], dtype=np.float32)
    return query_obs_levels(ep).cpu().numpy()


def run_baseline(
    name, suite_episodes, checkpoint=None, device="cpu", dir_target="level", realignment=None
):
    if name == "DoOverTimePFN":
        model = baselines.get(name, checkpoint=checkpoint, device=device)
    else:
        model = baselines.get(name)
    ep_pred, ep_tgt, ep_obs = [], [], []
    for ep in suite_episodes:
        p = torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1).cpu().numpy()
        t = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).cpu().numpy()
        ep_pred.append(p)
        ep_tgt.append(t)
        ep_obs.append(_episode_obs_levels(ep, realignment) if dir_target == "effect" else None)
    pred = np.concatenate(ep_pred)
    tgt = np.concatenate(ep_tgt)
    # RMSE is always level-space (subtracting y_obs from both sides would not
    # change it anyway); dir_target only changes what the sign test scores.
    rmse = _pooled_rmse(pred, tgt)
    lo, hi = _cluster_bootstrap_rmse(ep_pred, ep_tgt)
    if dir_target == "effect":
        obs = np.concatenate(ep_obs)
        da = direction_accuracy(torch.from_numpy(pred - obs), torch.from_numpy(tgt - obs))
    else:
        da = direction_accuracy(torch.from_numpy(pred), torch.from_numpy(tgt))
    return {
        "baseline": name,
        "n_episodes": len(ep_pred),
        "n_queries": int(pred.size),
        "pooled_rmse": rmse,
        "rmse_ci95": [lo, hi],
        "dir_acc": da["accuracy"],
        "dir_n_valid": da["n_valid"],
        "dir_target": dir_target,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--baselines", nargs="+", default=CPU_BASELINES)
    ap.add_argument("--pfn-checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--dir-target",
        choices=["level", "effect"],
        default="level",
        help="What the direction-accuracy sign test scores: the interventional "
        "level (v1 paper protocol) or the causal effect y_true - y_obs.",
    )
    ap.add_argument(
        "--realignment",
        type=Path,
        default=None,
        help="JSONL realignment sidecar mapping episodes to corrected y_obs "
        "(needed for archived suites with misaligned x_obs columns).",
    )
    args = ap.parse_args()
    realignment = None
    if args.realignment:
        realignment = {}
        with args.realignment.open() as fh:
            for line in fh:
                row = json.loads(line)
                realignment[int(row["idx"])] = row

    t0 = time.time()
    episodes = list(load_benchmark(args.suite))
    print(f"[{args.suite}] loaded {len(episodes)} episodes in {time.time() - t0:.1f}s")
    if realignment is not None:
        # Repair the archived x_obs (column order + hidden zeroing) so
        # baselines read the variable they claim to read.
        episodes = [
            realign_episode(
                ep,
                realignment[ep.scm_id]["canonical_perm"],
                realignment[ep.scm_id]["hidden_canonical"],
            )
            if ep.scm_id in realignment
            else ep
            for ep in episodes
        ]
        print(f"[{args.suite}] realigned x_obs for {len(realignment)} episodes")

    rows = []
    todo = list(args.baselines)
    if args.pfn_checkpoint:
        todo += ["DoOverTimePFN"]
    for name in todo:
        t = time.time()
        try:
            row = run_baseline(
                name,
                episodes,
                checkpoint=args.pfn_checkpoint,
                device=args.device,
                dir_target=args.dir_target,
                realignment=realignment,
            )
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
