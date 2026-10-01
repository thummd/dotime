"""Chronos-2 int/obs baseline on the dot-* suites (Table 3).

Mirrors ``dotime/eval/baselines/chronos2.py`` (branch liam/add-baseline-eval):

- interventional (``use_covariate=True``): the treatment variable is a known
  past covariate whose FUTURE values are pinned to the intervention value —
  the closest a purely observational forecaster gets to conditioning on do(A=v);
- observational (``use_covariate=False``): univariate forecast of the outcome
  from its own pre-intervention history (no intervention information at all).

Forecast horizon runs from the intervention onset to the query step; the
prediction at the query step is compared against episode ``y_true``.
Direction accuracy is written on the interventional level and on the causal
effect ``y_true - y_obs`` for every arm; ``--dir-target`` picks which one fills
``dir_acc`` (default ``level``, the v1 protocol).

    dotime-eval-chronos --suite dot-Identifiability-v1 \
        --per-structure 60 --device cuda:0 --out chronos_ident.json

The treatment and outcome columns are read from ``x_obs`` by canonical index.
The archived 1.0.0 Identifiability files store ``x_obs`` in topological order,
so pin that version together with the realignment sidecar:

    dotime-eval-chronos --suite dot-Identifiability-v1 --version 1.0.0 \
        --realignment results/reference/dot-Identifiability-v1.0.0_realignment.jsonl \
        --per-structure 60 --device cuda:0 --out chronos_ident_realigned.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dotime.benchmarks import load_benchmark
from dotime.evaluation import (
    add_dir_target_argument,
    check_shared_noise,
    describe_dir_target,
    resolve_dir_target,
)
from dotime.qa import target_qa
from dotime.reference._realignment import (
    load_realignment,
    realign_episodes,
    sidecar_obs_levels,
)
from dotime.reference._scoring import (
    check_predictions,
    direction_scores,
    observational_levels,
)

_FREQ = "s"
_T0_ISO = "2000-01-01"


def _episode_frames(ep, use_covariate):
    from dotime.observation import require_finite_history

    # Checked before pandas loads: a NaN context would be forecast silently.
    require_finite_history(ep, "dotime-eval-chronos")
    # pandas ships with the `baselines` extra, not the core package.
    import pandas as pd

    t0 = pd.Timestamp(_T0_ISO)
    x = ep.x_obs.detach().cpu().numpy()
    t_len, _n_vars = x.shape
    a = ep.intervention.targets[0] if ep.intervention.targets else 0
    y = int(ep.query_target[0])
    onset = min(ep.intervention.times) if ep.intervention.times else t_len
    # The row comes from the episode, which knows its suite's query_time
    # encoding: a bare fraction cannot tell index / T from index / (T - 1).
    q_idx = min(max(int(ep.query_time_idx[0]), onset), t_len - 1)
    horizon = q_idx - onset + 1
    a_val = (
        float(ep.intervention.values)
        if isinstance(ep.intervention.values, (int, float))
        else float(x[:onset, a].mean())
    )

    ctx = {
        "item_id": "ep",
        "timestamp": pd.date_range(t0, periods=onset, freq=_FREQ),
        "target": x[:onset, y].astype(np.float32),
    }
    if use_covariate:
        ctx["actuator"] = x[:onset, a].astype(np.float32)
    context_df = pd.DataFrame(ctx)

    future_df = None
    if use_covariate:
        future_df = pd.DataFrame(
            {
                "item_id": "ep",
                "timestamp": pd.date_range(
                    t0 + pd.Timedelta(seconds=onset), periods=horizon, freq=_FREQ
                ),
                "actuator": np.full(horizon, a_val, dtype=np.float32),
            }
        )
    return context_df, future_df, horizon


def predict(pipeline, ep, use_covariate):
    context_df, future_df, horizon = _episode_frames(ep, use_covariate)
    if len(context_df) < 8:
        return float(context_df["target"].mean())
    kwargs = dict(prediction_length=horizon, quantile_levels=[0.5])
    if future_df is not None:
        pred = pipeline.predict_df(context_df, future_df=future_df, target="target", **kwargs)
    else:
        pred = pipeline.predict_df(context_df, target="target", **kwargs)
    return float(pred["predictions"].to_numpy()[-1])


def _load_pipeline(model_id, device):
    """Load a pretrained Chronos pipeline, importing Chronos lazily.

    Chronos is an optional dependency, so the import happens on first use,
    as TabPFN's does in :func:`dotime.reference.tabpfn._regressor`.

    Args:
        model_id: Hugging Face model id, e.g. ``"amazon/chronos-2"``.
        device: Device map handed to ``from_pretrained``, e.g. ``"cuda:0"``.

    Returns:
        The loaded pipeline. Only its ``predict_df`` method is used.

    Raises:
        SystemExit: If Chronos is not installed.
    """
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise SystemExit(
            "Chronos is required for this evaluator: pip install 'dotime[baselines]'"
        ) from exc
    return BaseChronosPipeline.from_pretrained(model_id, device_map=device)


def main(argv: list[str] | None = None) -> None:
    """Score the Chronos-2 int/obs pair on a per-structure subsample of a suite.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``, which is how
            the ``dotime-eval-chronos`` console script calls it.

    Raises:
        SystemExit: On invalid arguments, or if Chronos is not installed.
        OSError: If the ``--realignment`` sidecar cannot be read.
        ValueError: If the sidecar is malformed or does not describe the
            evaluated episodes, e.g. a 1.0.0 sidecar against suite 1.1.0.
        dotime.qa.TargetQAError: If the evaluated targets fail target QA and
            ``--target-qa`` is ``enforce``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument(
        "--version",
        default="latest",
        help="Suite version to load, e.g. 1.0.0 (default: the registry's current version).",
    )
    ap.add_argument("--per-structure", type=int, default=60)
    ap.add_argument("--max-total", type=int, default=600)
    ap.add_argument("--model-id", default="amazon/chronos-2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--realignment",
        type=Path,
        default=None,
        help="JSONL realignment sidecar for dot-Identifiability-v1 1.0.0: permutes "
        "x_obs to canonical order and zeroes hidden variables. Every evaluated "
        "episode must match its row, so pair it with --version 1.0.0.",
    )
    add_dir_target_argument(ap)
    ap.add_argument(
        "--target-qa",
        choices=["enforce", "warn"],
        default="enforce",
        help="Log and assert per-arm target statistics of the evaluated episodes "
        "before any model runs (dotime.qa). 'warn' reports a failure and scores anyway.",
    )
    args = ap.parse_args(argv)
    # Read the sidecar first: a bad path should fail before a model load or
    # a suite download.
    realignment = load_realignment(args.realignment) if args.realignment is not None else None

    pipeline = _load_pipeline(args.model_id, args.device)

    suite = load_benchmark(args.suite, version=args.version)
    byst = defaultdict(list)
    for ep in suite:
        byst[ep.structure].append(ep)
    samp = [e for eps in byst.values() for e in eps[: args.per_structure]][: args.max_total]
    print(
        f"[{args.suite} v{suite.meta.version}] {len(samp)} episodes across {len(byst)} structures"
    )
    if realignment is not None:
        # The subsample is chosen by structure label and suite order alone, so
        # realigning after it scores the same episodes and checks only those.
        samp = realign_episodes(samp, realignment)
        print(f"  realigned x_obs of {len(samp)} episodes with {args.realignment.name}")
    # The subsample is what gets scored, so it is what gets checked.
    sidecar_levels = sidecar_obs_levels(samp, realignment)
    noise = check_shared_noise(samp)
    dir_target = resolve_dir_target(args.dir_target, noise, warn=False)
    print(f"  {describe_dir_target(args.dir_target, dir_target, noise)}")
    qa_report = target_qa(
        samp,
        obs_levels=sidecar_levels,
        dir_target=dir_target,
        raise_on_failure=args.target_qa == "enforce",
    )
    # The same factual levels the QA saw score the effect direction below,
    # where it is a counterfactual effect or was asked for.
    with_effect = dir_target == "effect" or noise.shared
    y_obs = observational_levels(samp, sidecar_levels) if with_effect else None

    out = {
        "suite": args.suite,
        "suite_version": suite.meta.version,
        "realigned": realignment is not None,
        # File name only: an absolute path would leak the machine's layout
        # into a released result JSON.
        "realignment_sidecar": args.realignment.name if realignment is not None else None,
        "n": len(samp),
        "dir_target": dir_target,
        "dir_target_mode": args.dir_target,
        "pairs_share_noise": noise.shared,
        "target_qa": qa_report.to_dict(),
        "model_id": args.model_id,
    }
    for tag, cov in [("Chronos_int", True), ("Chronos_obs", False)]:
        t0 = time.time()
        pred_list: list[float] = []
        tgt_list: list[float] = []
        for i, ep in enumerate(samp):
            try:
                pred_list.append(predict(pipeline, ep, cov))
            except Exception:
                pred_list.append(float(ep.x_obs[:, int(ep.query_target[0])].mean()))
            tgt_list.append(float(ep.y_true.reshape(-1)[0]))
            if (i + 1) % 100 == 0:
                print(f"  {tag} {i + 1}/{len(samp)} ({time.time() - t0:.0f}s)")
        preds = np.array(pred_list)
        tgts = np.array(tgt_list)
        n_nonfinite = check_predictions(tag, preds)
        rmse = float(np.sqrt(np.nanmean((preds - tgts) ** 2)))
        rng = np.random.default_rng(0)
        se = (preds - tgts) ** 2
        boot = np.array(
            [np.sqrt(se[rng.integers(0, len(se), len(se))].mean()) for _ in range(1000)]
        )
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        scores = direction_scores(preds, tgts, y_obs, dir_target)
        out[tag] = {"pooled_rmse": rmse, "rmse_ci95": ci, "n_nonfinite": n_nonfinite, **scores}
        print(
            f"{tag}  RMSE={rmse:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] dir_acc={scores['dir_acc']:.3f} ({dir_target}) "
            f"({time.time() - t0:.0f}s)"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
