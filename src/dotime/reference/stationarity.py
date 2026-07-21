"""Stationarity diagnostics for the generic prior (backs Appendix B of the paper).

The convergence result assumes a reduced-form spectral radius $\\rho < 1$, which
the released v1.0.0 suites deliberately do not enforce. This module measures how
often that assumption actually holds, and whether violating it is what produces
the divergence disclosed in the Limitations section. Two independent checks:

(A) **rho diagnostic.** For each sampled SCM, build the reduced-form VAR
    companion matrix from the weight matrices alone -- no simulation -- and take
    its spectral radius. Reports the distribution of rho and cross-tabulates it
    against divergence.

(B) **burn-in moments.** For episodes that did *not* diverge, compare
    per-variable mean and variance on the first vs the last 50 retained steps.
    If burn-in is sufficient, the two windows agree.

Usage::

    dotime-diagnose-stationarity --n-episodes 2000 --out stationarity.json
    dotime-diagnose-stationarity --n-episodes 2000 --activations identity

``--activations identity`` restricts every mechanism to a linear activation,
the condition under which the companion rho is the exact stability criterion
rather than an upper bound. Because rho depends on the sampled weights alone,
the two runs cover the same SCMs and differ only in the nonlinearity, making
the comparison a paired one. Released outputs:
``results/reference/stationarity_diagnostic.json`` and
``results/reference/stationarity_diagnostic_identity.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dotime import DoTime
from dotime.utils import DEFAULT_CONFIG


def companion_rho(scm) -> tuple[float | None, list[str] | None]:
    """Return ``(rho, activation_names)`` for one SCM, computed from weights only.

    Mirrors the reduced form used by the hardening path in ``batched_tscm``:
    ``A_k = (I - W_inst)^{-1} W_lag[k]``, assembled into a block companion
    matrix. The weight for node ``v`` from parent ``u`` at lag ``k`` is
    ``mechanisms[v].weights_lagged[k-1][u]`` (``weights_instant[u]`` for
    ``k = 0``), matching ``TemporalMechanism.forward``.

    Returns ``(None, None)`` for SCMs that have no single companion matrix --
    regime-switching and chain SCMs. That is not a failure: a switching system
    genuinely has no one reduced form, which is exactly why the convergence
    theorem does not cover it.

    The mechanism computes ``activation(W x + b) + eps``, so this linear rho is
    exact only for ``Identity`` activations. Every other activation in the prior
    except ``Square`` is non-expansive near the origin, making rho an upper
    bound for those; the caller can use the returned activation names to isolate
    the exact subset.
    """
    topo = list(getattr(scm, "_topo", []))
    mechs = getattr(scm, "mechanisms", None)
    inst_pairs = getattr(scm, "_instant_parent_pairs", None)
    lag_pairs = getattr(scm, "_lagged_parent_pairs", None)
    if not topo or not mechs or inst_pairs is None or lag_pairs is None:
        return None, None

    n = len(topo)
    max_lag = max((len(lag_pairs[i]) for i in range(n)), default=1) or 1
    w_inst = np.zeros((n, n))
    w_lag = [np.zeros((n, n)) for _ in range(max_lag)]
    acts: list[str] = []

    for i, v in enumerate(topo):
        m = mechs[v]
        acts.append(type(m.activation).__name__)
        for pname, pidx in inst_pairs[i]:
            w_inst[i, pidx] = float(m.weights_instant[pname].detach().cpu())
        for k, pairs_k in enumerate(lag_pairs[i]):
            for pname, pidx in pairs_k:
                w_lag[k][i, pidx] = float(m.weights_lagged[k][pname].detach().cpu())

    try:
        inv = np.linalg.inv(np.eye(n) - w_inst)
    except np.linalg.LinAlgError:
        return float("inf"), acts

    dim = n * max_lag
    c = np.zeros((dim, dim))
    c[:n, :] = np.hstack([inv @ wl for wl in w_lag])
    if max_lag > 1:
        c[n:, : n * (max_lag - 1)] = np.eye(n * (max_lag - 1))
    return float(np.max(np.abs(np.linalg.eigvals(c)))), acts


def _pct(x) -> float:
    return 100.0 * float(np.mean(x)) if len(x) else float("nan")


def diagnose(
    n_episodes: int = 2000,
    t_len: int = 200,
    seed0: int = 20260719,
    activations: str = "all",
) -> dict:
    """Run both diagnostics over ``n_episodes`` freshly sampled generic SCMs.

    ``activations="identity"`` restricts every mechanism to a linear activation.
    On that subset the companion rho is the *exact* stability criterion rather
    than an upper bound, so it isolates the assumption the theorem actually
    states. The restriction needs no change to the prior: the generic and regime
    builders are constructed per sample and read ``DoTime.activations`` at
    generate time, and mutating the list in place also reaches the
    chain builder, which captures it at construction.
    """
    rhos: list[float] = []
    diverged: list[bool] = []
    all_linear: list[bool] = []
    early_m, late_m, early_v, late_v = [], [], [], []

    for i in range(n_episodes):
        seed = seed0 + i
        torch.manual_seed(seed)
        prior = DoTime(seed=seed)
        if activations == "identity":
            prior.activations[:] = [nn.Identity()]
        elif activations != "all":
            raise ValueError(f"unknown activations mode {activations!r}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # handled divergence
            x_obs, x_int, _iv, scm = prior.generate_pair(T=t_len)

        # A diverged episode is returned as all zeros in both arms.
        div = float(x_obs.abs().max()) == 0.0 and float(x_int.abs().max()) == 0.0
        diverged.append(div)

        r, acts = companion_rho(scm)
        rhos.append(np.nan if r is None else r)
        all_linear.append(bool(acts) and all(a == "Identity" for a in acts))

        if not div:
            # x_obs is the intervention-free arm and is already post-burn-in,
            # so these windows probe *residual* non-stationarity.
            x = x_obs.detach().cpu().numpy()
            early_m.append(x[:50].mean(axis=0))
            late_m.append(x[-50:].mean(axis=0))
            early_v.append(x[:50].var(axis=0))
            late_v.append(x[-50:].var(axis=0))

    rho = np.array(rhos, dtype=float)
    div = np.array(diverged, dtype=bool)
    ok = np.isfinite(rho)

    out: dict = {
        "n_episodes": n_episodes,
        "t_len": t_len,
        "seed0": seed0,
        "activations": activations,
        "burn_in": int(DEFAULT_CONFIG.get("burn_in", 50)),
        "rho": {
            "n_measurable": int(ok.sum()),
            "note": (
                "Regime-switching and chain SCMs have no single companion "
                "matrix and are excluded from the rho statistics by design."
            ),
            "median": float(np.median(rho[ok])) if ok.any() else None,
            "p90": float(np.percentile(rho[ok], 90)) if ok.any() else None,
            "max": float(np.max(rho[ok])) if ok.any() else None,
            "pct_ge_1": _pct(rho[ok] >= 1.0),
        },
        "divergence": {
            "pct_all": _pct(div),
            "pct_measurable_subset": _pct(div[ok]),
            "pct_diverged_given_rho_ge_1": _pct(div[ok][rho[ok] >= 1.0]),
            "pct_diverged_given_rho_lt_1": _pct(div[ok][rho[ok] < 1.0]),
            "pct_rho_ge_1_given_diverged": (
                _pct(rho[ok & div] >= 1.0) if (ok & div).any() else None
            ),
        },
    }

    lin = ok & np.array(all_linear, dtype=bool)
    if lin.any() and activations != "identity":
        out["rho_identity_only"] = {
            "n": int(lin.sum()),
            "note": "All-Identity SCMs: the linear rho is exact, not an upper bound.",
            "median": float(np.median(rho[lin])),
            "pct_ge_1": _pct(rho[lin] >= 1.0),
            "pct_diverged": _pct(div[lin]),
        }

    if early_m:
        em, lm = np.concatenate(early_m), np.concatenate(late_m)
        ev, lv = np.concatenate(early_v), np.concatenate(late_v)
        fin = np.isfinite(em) & np.isfinite(lm) & np.isfinite(ev) & np.isfinite(lv)
        em, lm, ev, lv = em[fin], lm[fin], ev[fin], lv[fin]

        ratio = lv / np.where(ev > 1e-12, ev, np.nan)
        ratio = ratio[np.isfinite(ratio)]
        # Pooled means are dominated by a heavy tail of large-level episodes, so
        # compare standardized drift per variable rather than the pooled mean.
        drift = np.abs(lm - em) / np.sqrt(np.where(ev > 1e-12, ev, np.nan))
        drift = drift[np.isfinite(drift)]

        out["burn_in_moments"] = {
            "n_nondiverged_episodes": len(early_m),
            "n_variables": len(em),
            "var_ratio_median": float(np.median(ratio)),
            "var_ratio_p10": float(np.percentile(ratio, 10)),
            "var_ratio_p90": float(np.percentile(ratio, 90)),
            "pct_var_ratio_within_2x": _pct((ratio > 0.5) & (ratio < 2.0)),
            "std_drift_median": float(np.median(drift)),
            "std_drift_p90": float(np.percentile(drift, 90)),
            "pct_std_drift_lt_0p5": _pct(drift < 0.5),
            "median_abs_mean_early": float(np.median(np.abs(em))),
            "median_abs_mean_late": float(np.median(np.abs(lm))),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-episodes", type=int, default=2000)
    parser.add_argument("--t-len", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=20260719)
    parser.add_argument(
        "--activations",
        choices=("all", "identity"),
        default="all",
        help="'identity' restricts every mechanism to a linear activation, the "
        "subset on which the companion rho is exact rather than an upper bound.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = diagnose(args.n_episodes, args.t_len, args.seed0, args.activations)
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
