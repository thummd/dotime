"""TabPFN adjustment baseline for the dot-* suites (Table 3).

Mirrors the TabPFN baselines of the Do-Over-Time-PFN training codebase
(``scripts/baselines.py``): a back-door adjustment using two TabPFN
regressors (model_x: p(X_t|X_{t-1}); model_y: p(Y_t|A_t,X_t,Y_{t-1})),
MC-integrated over the confounder, and a front-door variant. Ported to the
dotime Episode API; falls back to the pre-intervention outcome mean on
structures where the adjustment assumptions do not hold (as BackDoorOLS does),
so every episode gets a prediction.

TabPFN is expensive, so we evaluate on a stratified subsample.

    dotime-eval-tabpfn --suite dot-Identifiability-v1 \
        --per-structure 60 --device cuda:0 --out tabpfn_ident.json

The evaluator indexes ``x_obs`` by canonical column. The archived 1.0.0
Identifiability files store it in topological order, so pin that version
together with the realignment sidecar:

    dotime-eval-tabpfn --suite dot-Identifiability-v1 --version 1.0.0 \
        --realignment results/reference/dot-Identifiability-v1.0.0_realignment.jsonl \
        --per-structure 60 --device cuda:0 --out tabpfn_ident_realigned.json
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
from dotime.reference._realignment import load_realignment, realign_episodes


def _regressor():
    """Import TabPFN lazily: it is an optional dependency (``baselines`` extra)."""
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise SystemExit(
            "TabPFN is required for this evaluator: pip install 'dotime[baselines]'"
        ) from exc
    return TabPFNRegressor


BACK_DOOR = {"back_door", "observed_confounder", "confounder_mediator"}
FRONT_DOOR = {"front_door", "mediator"}


def _series(ep):
    x = ep.x_obs.detach().cpu().numpy()
    t_len, n = x.shape
    a = ep.intervention.targets[0] if ep.intervention.targets else 0
    y = int(ep.query_target[0])
    onset = min(ep.intervention.times) if ep.intervention.times else t_len
    fit_end = max(2, min(onset, t_len))
    a_val = (
        float(ep.intervention.values) if isinstance(ep.intervention.values, (int, float)) else None
    )
    return x, n, a, y, fit_end, a_val


def _mean_pred(ep):
    x, _n, _a, y, fit_end, _ = _series(ep)
    return float(x[:fit_end, y].mean())


def _backdoor_tabpfn(ep, n_mc=100, observational=False):
    x, n, a, y, fit_end, a_val = _series(ep)
    adj = [v for v in range(n) if v not in (a, y)]
    if len(adj) < 1 or fit_end < 8 or a_val is None:
        return _mean_pred(ep)
    xcov = x[1:fit_end, adj]
    a_t = x[1:fit_end, a]
    y_prev = x[0 : fit_end - 1, y]
    y_t = x[1:fit_end, y]
    # model_y: Y_t ~ [A_t, X_t..., Y_{t-1}]
    my = _regressor()()
    my.fit(np.column_stack([a_t, xcov, y_prev]), y_t)
    # MC over observed confounder rows; plug do(A=a_val) (int) or the last
    # observed A as stand-in for the natural A_t (obs), per Jake's
    # BackDoorTabPFNObservational. Y_{t-1}=last obs.
    plug = float(x[fit_end - 1, a]) if observational else a_val
    y_last = float(x[fit_end - 1, y])
    Xq = np.column_stack([np.full(len(xcov), plug), xcov, np.full(len(xcov), y_last)])
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
    _R = _regressor()
    mm = _R()
    mm.fit(a_t.reshape(-1, 1), m_t)
    myd = _R()
    myd.fit(np.column_stack([m_t, a_t]), y_t)
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


def main(argv: list[str] | None = None) -> None:
    """Score the TabPFN int/obs pair on a per-structure subsample of a suite.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``, which is how
            the ``dotime-eval-tabpfn`` console script calls it.

    Raises:
        SystemExit: On invalid arguments, or if TabPFN is not installed.
        OSError: If the ``--realignment`` sidecar cannot be read.
        ValueError: If the sidecar is malformed or does not describe the
            evaluated episodes, e.g. a 1.0.0 sidecar against suite 1.1.0.
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
    args = ap.parse_args(argv)
    # Read the sidecar first: a bad path should fail before a suite download.
    realignment = load_realignment(args.realignment) if args.realignment is not None else None

    import os

    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    suite = load_benchmark(args.suite, version=args.version)
    byst = defaultdict(list)
    for ep in suite:
        byst[ep.structure].append(ep)
    samp = []
    for eps in byst.values():
        samp += eps[: args.per_structure]
    samp = samp[: args.max_total]
    print(
        f"[{args.suite} v{suite.meta.version}] {len(samp)} episodes across {len(byst)} structures"
    )
    if realignment is not None:
        # The subsample is chosen by structure label and suite order alone, so
        # realigning after it scores the same episodes and checks only those.
        samp = realign_episodes(samp, realignment)
        print(f"  realigned x_obs of {len(samp)} episodes with {args.realignment.name}")

    out = {
        "suite": args.suite,
        "suite_version": suite.meta.version,
        "realigned": realignment is not None,
        # File name only: an absolute path would leak the machine's layout
        # into a released result JSON.
        "realignment_sidecar": args.realignment.name if realignment is not None else None,
        "n": len(samp),
    }
    for tag, obs in [("TabPFN_int", False), ("TabPFN_obs", True)]:
        t0 = time.time()
        pred_list: list[float] = []
        tgt_list: list[float] = []
        for i, ep in enumerate(samp):
            pred_list.append(predict(ep, observational=obs))
            tgt_list.append(float(ep.y_true.reshape(-1)[0]))
            if (i + 1) % 100 == 0:
                print(f"  {tag} {i + 1}/{len(samp)}  ({time.time() - t0:.0f}s)")
        preds = np.array(pred_list)
        tgts = np.array(tgt_list)
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
            f"{tag}  RMSE={rmse:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] dir_acc={da['accuracy']:.3f}  "
            f"({time.time() - t0:.0f}s total)"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
