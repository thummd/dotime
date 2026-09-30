#!/usr/bin/env python
"""Release-scale reference-table eval for the DoTime paper (Table 3).

Runs the registered CPU baselines (and, if a checkpoint is given, the
Do-Over-Time-PFN) over a full suite, reporting pooled RMSE and direction
accuracy with an episode-cluster bootstrap CI on the pooled RMSE.

    dotime-eval-reference --suite dot-Identifiability-v1 --out ident.json

The archived 1.0.0 Identifiability files store ``x_obs`` in topological order,
so pin that version together with the realignment sidecar. Its rows also
supply the observational level that ``--dir-target effect`` subtracts:

    dotime-eval-reference --suite dot-Identifiability-v1 --version 1.0.0 \
        --realignment results/reference/dot-Identifiability-v1.0.0_realignment.jsonl \
        --dir-target effect --out ident_effect_realigned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dotime import baselines, qa
from dotime.benchmarks import load_benchmark
from dotime.evaluation import direction_accuracy, query_obs_levels
from dotime.reference._realignment import load_realignment, realign_episodes

CPU_BASELINES = ["Zero", "Mean", "AR1", "VAR-OLS", "BackDoorOLS", "NaiveOLS", "IV2SLS", "Oracle"]
# Same floor as the training-side step-zero check: a level arm that is mostly
# zero means a masked or diverged target, not data.
TARGET_QA_MIN_NONZERO = qa.QAThresholds().min_level_nonzero_frac


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
    """Per-query observational levels, from the realignment sidecar when one is given.

    Args:
        ep: The episode being scored.
        realignment: Optional ``{scm_id: row}`` map from
            :func:`~dotime.reference._realignment.load_realignment`. Each row's
            ``y_obs_corrected`` is the observational level regenerated for the
            archived 1.0.0 suite, whose ``x_obs`` is column-misaligned (see the
            datasheet erratum).

    Returns:
        1-D numpy array of observational levels, one per query.

    Raises:
        KeyError: If ``realignment`` is given but has no row for ``ep``.
            :func:`main` realigns every scored episode first, which refuses
            such an episode with a clearer message.
    """
    if realignment is not None:
        # No fallback to x_obs: for a 1.0.0 episode that column may be another
        # variable, and one effect score would mix two sources of y_obs.
        return np.asarray([realignment[ep.scm_id]["y_obs_corrected"]], dtype=np.float32)
    return query_obs_levels(ep).cpu().numpy()


def target_qa(episodes, realignment=None, dir_target="level"):
    """Log and assert per-arm target statistics before any baseline is scored.

    A thin wrapper over :func:`dotime.qa.target_qa`, which asserts the arms
    pooled and per structure. The observational arm is read exactly as in
    scoring, from the realignment sidecar when one is given.

    Args:
        episodes: The episodes that will be scored.
        realignment: Optional ``{scm_id: row}`` realignment sidecar map, used for
            the observational level exactly as in scoring.
        dir_target: ``"level"`` or ``"effect"``. An effect-scored run also
            asserts the effect arm on the queries that can carry an effect.

    Returns:
        Dict with the pooled ``y_obs_level``, ``y_int_level`` and ``effect``
        statistics (:func:`dotime.qa.arm_stats`), the ``min_nonzero_frac``
        floor, and every key of :meth:`dotime.qa.QAReport.to_dict`.

    Raises:
        RuntimeError: A :class:`dotime.qa.TargetQAError` if an arm is
            non-finite, a level arm has zero variance or a nonzero fraction
            below ``TARGET_QA_MIN_NONZERO``, or an effect-scored run has too
            few nonzero effects, pooled or in a structure.
    """
    episodes = list(episodes)
    obs = None
    if realignment is not None:
        obs = [_episode_obs_levels(ep, realignment) for ep in episodes]
    report = qa.target_qa(episodes, obs_levels=obs, dir_target=dir_target)
    pooled = {arm: report.pooled[arm] for arm in qa.ARMS}
    return {**pooled, "min_nonzero_frac": TARGET_QA_MIN_NONZERO, **report.to_dict()}


def run_baseline(
    name, suite_episodes, checkpoint=None, device="cpu", dir_target="level", realignment=None
):
    if name == "DoOverTimePFN":
        model = baselines.get(name, checkpoint=checkpoint, device=device)
    else:
        model = baselines.get(name)
    ep_pred, ep_tgt = [], []
    ep_obs: list[np.ndarray] = []
    for ep in suite_episodes:
        p = torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1).cpu().numpy()
        t = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).cpu().numpy()
        ep_pred.append(p)
        ep_tgt.append(t)
        if dir_target == "effect":
            ep_obs.append(_episode_obs_levels(ep, realignment))
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


def main(argv: list[str] | None = None) -> None:
    """Score the reference baselines, and optionally the PFN, on a full suite.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``, which is how
            the ``dotime-eval-reference`` console script calls it.

    Raises:
        SystemExit: On invalid arguments.
        OSError: If the ``--realignment`` sidecar cannot be read.
        ValueError: If the sidecar is malformed or does not describe the
            evaluated episodes, e.g. a 1.0.0 sidecar against suite 1.1.0.
        RuntimeError: If the targets fail :func:`target_qa`.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument(
        "--version",
        default="latest",
        help="Suite version to load, e.g. 1.0.0 (default: the registry's current version).",
    )
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
        "--exclude-self-queries",
        action="store_true",
        help="Drop episodes whose query targets the intervened variable (continuous "
        "suite: ~1/3 of episodes; in-window hard self-queries equal the do-value).",
    )
    ap.add_argument(
        "--realignment",
        type=Path,
        default=None,
        help="JSONL realignment sidecar for dot-Identifiability-v1 1.0.0: permutes "
        "x_obs to canonical order, zeroes hidden variables and supplies the "
        "corrected y_obs for --dir-target effect. Every evaluated episode must "
        "match its row, so pair it with --version 1.0.0.",
    )
    args = ap.parse_args(argv)
    # Read the sidecar first: a bad path should fail before a suite download.
    realignment = load_realignment(args.realignment) if args.realignment is not None else None

    t0 = time.time()
    suite = load_benchmark(args.suite, version=args.version)
    episodes = list(suite)
    print(
        f"[{args.suite} {suite.meta.version}] loaded {len(episodes)} episodes "
        f"in {time.time() - t0:.1f}s"
    )
    if args.exclude_self_queries:
        n0 = len(episodes)
        episodes = [ep for ep in episodes if not ep.is_self_query]
        print(f"[{args.suite}] excluded {n0 - len(episodes)} self-query episodes")
    if realignment is not None:
        # Repair the archived x_obs (column order + hidden zeroing) so
        # baselines read the variable they claim to read. Every episode must
        # match its row, so a sidecar from another suite version stops here.
        episodes = realign_episodes(episodes, realignment)
        print(
            f"[{args.suite}] realigned x_obs of {len(episodes)} episodes "
            f"with {args.realignment.name}"
        )

    qa = target_qa(episodes, realignment, args.dir_target)
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

    out = {
        "suite": args.suite,
        "suite_version": suite.meta.version,
        "realigned": realignment is not None,
        # File name only: an absolute path would leak the machine's layout
        # into a released result JSON.
        "realignment_sidecar": args.realignment.name if realignment is not None else None,
        "n_episodes": len(episodes),
        "exclude_self_queries": args.exclude_self_queries,
        "target_qa": qa,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
