"""Chronos-2 int/obs baseline on the dot-* suites (Table 3).

Mirrors ``dotime/eval/baselines/chronos2.py`` (branch liam/add-baseline-eval):

- interventional (``use_covariate=True``): the treatment variable is a known
  past covariate whose FUTURE values are pinned to the intervention value —
  the closest a purely observational forecaster gets to conditioning on do(A=v);
- observational (``use_covariate=False``): univariate forecast of the outcome
  from its own pre-intervention history (no intervention information at all).

Forecast horizon runs from the intervention onset to the query step; the
prediction at the query step is compared against episode ``y_true``.

    dotime-eval-chronos --suite dot-Identifiability-v1 \
        --per-structure 60 --device cuda:0 --out chronos_ident.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dotime.benchmarks import load_benchmark
from dotime.evaluation import direction_accuracy

_FREQ = "s"
_T0_ISO = "2000-01-01"


def _episode_frames(ep, use_covariate):
    # pandas ships with the `baselines` extra, not the core package.
    import pandas as pd

    t0 = pd.Timestamp(_T0_ISO)
    x = ep.x_obs.detach().cpu().numpy()
    t_len, _n_vars = x.shape
    a = ep.intervention.targets[0] if ep.intervention.targets else 0
    y = int(ep.query_target[0])
    onset = min(ep.intervention.times) if ep.intervention.times else t_len
    qt = float(ep.query_time[0])
    q_idx = round(qt * t_len) if qt <= 1.0 else round(qt)
    q_idx = min(max(q_idx, onset), t_len - 1)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--per-structure", type=int, default=60)
    ap.add_argument("--max-total", type=int, default=600)
    ap.add_argument("--model-id", default="amazon/chronos-2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise SystemExit(
            "Chronos is required for this evaluator: pip install 'dotime[baselines]'"
        ) from exc

    pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=args.device)

    allep = list(load_benchmark(args.suite))
    byst = defaultdict(list)
    for ep in allep:
        byst[ep.structure].append(ep)
    samp = [e for eps in byst.values() for e in eps[: args.per_structure]][: args.max_total]
    print(f"[{args.suite}] {len(samp)} episodes across {len(byst)} structures")

    out = {"suite": args.suite, "n": len(samp), "model_id": args.model_id}
    for tag, cov in [("Chronos_int", True), ("Chronos_obs", False)]:
        t0 = time.time()
        preds, tgts = [], []
        for i, ep in enumerate(samp):
            try:
                preds.append(predict(pipeline, ep, cov))
            except Exception:
                preds.append(float(ep.x_obs[:, int(ep.query_target[0])].mean()))
            tgts.append(float(ep.y_true.reshape(-1)[0]))
            if (i + 1) % 100 == 0:
                print(f"  {tag} {i + 1}/{len(samp)} ({time.time() - t0:.0f}s)")
        preds = np.array(preds)
        tgts = np.array(tgts)
        rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
        da = direction_accuracy(torch.from_numpy(preds).float(), torch.from_numpy(tgts).float())
        rng = np.random.default_rng(0)
        se = (preds - tgts) ** 2
        boot = np.array(
            [np.sqrt(se[rng.integers(0, len(se), len(se))].mean()) for _ in range(1000)]
        )
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        _nv = da["n_valid"]
        _se = (da["accuracy"] * (1 - da["accuracy"]) / _nv) ** 0.5 if _nv else float("nan")
        out[tag] = {
            "pooled_rmse": rmse,
            "rmse_ci95": ci,
            "dir_acc": da["accuracy"],
            "dir_n_valid": _nv,
            "dir_acc_se": _se,
        }
        print(
            f"{tag}  RMSE={rmse:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] dir_acc={da['accuracy']:.3f} "
            f"({time.time() - t0:.0f}s)"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
