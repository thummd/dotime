"""TabPFN adjustment baseline for the dot-* suites (Table 3).

Mirrors Jake Robertson's do-over-time-pfn TabPFN baselines
(``scripts/baselines.py``): a back-door adjustment using two TabPFN
regressors (model_x: p(X_t|X_{t-1}); model_y: p(Y_t|A_t,X_t,Y_{t-1})),
MC-integrated over the confounder, and a front-door variant. Ported to the
dotime Episode API; falls back to the pre-intervention outcome mean on
structures where the adjustment assumptions do not hold (as BackDoorOLS does),
so every episode gets a prediction.

TabPFN is expensive, so we evaluate on a stratified subsample.

    python scripts/eval_tabpfn_baseline.py --suite dot-Identifiability-v1 \
        --per-structure 60 --device cuda:0 --out results/reference/tabpfn_ident.json
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor

from dotime.benchmarks import load_benchmark
from dotime.evaluation import direction_accuracy

BACK_DOOR = {"back_door", "observed_confounder", "confounder_mediator"}
FRONT_DOOR = {"front_door", "mediator"}


def _series(ep):
    x = ep.x_obs.detach().cpu().numpy()
    t_len, n = x.shape
    a = ep.intervention.targets[0] if ep.intervention.targets else 0
    y = int(ep.query_target[0])
    onset = min(ep.intervention.times) if ep.intervention.times else t_len
    fit_end = max(2, min(onset, t_len))
    a_val = (float(ep.intervention.values)
             if isinstance(ep.intervention.values, (int, float)) else None)
    return x, n, a, y, fit_end, a_val


def _mean_pred(ep):
    x, n, a, y, fit_end, _ = _series(ep)
    return float(x[:fit_end, y].mean())


def _backdoor_tabpfn(ep, n_mc=100, observational=False):
    x, n, a, y, fit_end, a_val = _series(ep)
    adj = [v for v in range(n) if v not in (a, y)]
    if len(adj) < 1 or fit_end < 8 or a_val is None:
        return _mean_pred(ep)
    xcov = x[1:fit_end, adj]
    a_t = x[1:fit_end, a]
    y_prev = x[0:fit_end - 1, y]
    y_t = x[1:fit_end, y]
    # model_y: Y_t ~ [A_t, X_t..., Y_{t-1}]
    my = TabPFNRegressor()
    my.fit(np.column_stack([a_t, xcov, y_prev]), y_t)
    # MC over observed confounder rows; plug do(A=a_val) (int) or the last
    # observed A as stand-in for the natural A_t (obs), per Jake's
    # BackDoorTabPFNObservational. Y_{t-1}=last obs.
    plug = float(x[fit_end - 1, a]) if observational else a_val
    y_last = float(x[fit_end - 1, y])
    Xq = np.column_stack([
        np.full(len(xcov), plug), xcov, np.full(len(xcov), y_last)])
    return float(np.mean(my.predict(Xq)))


def _frontdoor_tabpfn(ep, n_mc=100, observational=False):
    # front-door: mediator M between A and Y. Use all non-(A,Y) as candidate M.
    x, n, a, y, fit_end, a_val = _series(ep)
    med = [v for v in range(n) if v not in (a, y)]
    if len(med) < 1 or fit_end < 8 or a_val is None:
        return _mean_pred(ep)
    m_idx = med[0]
    a_t = x[1:fit_end, a]
    m_t = x[1:fit_end, m_idx]
    y_t = x[1:fit_end, y]
    # model_m: M_t ~ A_t ; model_y: Y_t ~ [M_t, A_t]
    mm = TabPFNRegressor(); mm.fit(a_t.reshape(-1, 1), m_t)
    myd = TabPFNRegressor(); myd.fit(np.column_stack([m_t, a_t]), y_t)
    plug = float(x[fit_end - 1, a]) if observational else a_val
    m_do = mm.predict(np.array([[plug]]))
    m_samp = np.full(len(a_t), float(m_do[0]))
    Xq = np.column_stack([m_samp, a_t])
    return float(np.mean(myd.predict(Xq)))


def predict(ep, observational=False):
    if ep.structure in BACK_DOOR:
        return _backdoor_tabpfn(ep, observational=observational)
    if ep.structure in FRONT_DOOR:
        return _frontdoor_tabpfn(ep, observational=observational)
    return _mean_pred(ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--per-structure", type=int, default=60)
    ap.add_argument("--max-total", type=int, default=600)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import os
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    allep = list(load_benchmark(args.suite))
    byst = defaultdict(list)
    for ep in allep:
        byst[ep.structure].append(ep)
    samp = []
    for st, eps in byst.items():
        samp += eps[:args.per_structure]
    samp = samp[:args.max_total]
    print(f"[{args.suite}] {len(samp)} episodes across {len(byst)} structures")

    out = {"suite": args.suite, "n": len(samp)}
    for tag, obs in [("TabPFN_int", False), ("TabPFN_obs", True)]:
        t0 = time.time()
        preds, tgts = [], []
        for i, ep in enumerate(samp):
            preds.append(predict(ep, observational=obs))
            tgts.append(float(ep.y_true.reshape(-1)[0]))
            if (i + 1) % 100 == 0:
                print(f"  {tag} {i+1}/{len(samp)}  ({time.time()-t0:.0f}s)")
        preds = np.array(preds); tgts = np.array(tgts)
        rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
        da = direction_accuracy(torch.from_numpy(preds).float(), torch.from_numpy(tgts).float())
        rng = np.random.default_rng(0)
        se = (preds - tgts) ** 2
        boot = np.array([np.sqrt(se[rng.integers(0, len(se), len(se))].mean()) for _ in range(1000)])
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        _nv = da["n_valid"]; _se = (da["accuracy"]*(1-da["accuracy"])/_nv)**0.5 if _nv else float("nan")
        out[tag] = {"pooled_rmse": rmse, "rmse_ci95": ci, "dir_acc": da["accuracy"],
                    "dir_n_valid": _nv, "dir_acc_se": _se}
        print(f"{tag}  RMSE={rmse:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] dir_acc={da['accuracy']:.3f}  "
              f"({time.time()-t0:.0f}s total)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
