#!/usr/bin/env python
"""Measure the generic-prior hardening option against the unhardened prior.

For each (N_max, K_max) configuration, episodes are sampled exactly as
``dotime._build.make_episode`` samples the generic suite (per-episode seed,
global torch seed, ``DoTime(...).generate_pair``) under three variants: no
hardening, the two weight knobs only (``unit_norm_rows`` and
``spectral_rho=0.9``), and ``RECOMMENDED_HARDENING``, which adds
``bounded_square``. Because hardening draws no random numbers, all variants see
the same graphs, interventions and noise, so the comparison is paired.

Per variant it reports the share of episodes with either arm zeroed, the share
of surviving episodes that hit the +-1000 clip, the surviving graph sizes,
per-arm target statistics (asserted, per the repository rule), the shared-noise
causal effect size, and the sampling time.

    python results/reference/hardening/measure_hardening.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SUITE_SEED = 20260929
VARIANTS = {
    "base": None,
    "weights_only": {"unit_norm_rows": True, "spectral_rho": 0.9},
    "recommended": {"unit_norm_rows": True, "spectral_rho": 0.9, "bounded_square": True},
}


def one(spec: dict) -> dict:
    """Sample one episode per variant and summarise it (runs in a worker).

    Parameters
    ----------
    spec : dict
        ``idx``, ``n_max``, ``k_max`` and ``T``.

    Returns
    -------
    dict
        Per-variant record with divergence, saturation, targets, effect size
        and timing.
    """
    import torch

    from dotime import DoTime
    from dotime.benchmarks import episode_from_pair
    from dotime.evaluation import query_obs_levels
    from dotime.hardening import companion_spectral_radius
    from dotime.utils import DEFAULT_CONFIG

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    seed = (SUITE_SEED * 1_000_003 + spec["idx"]) & 0x7FFFFFFF
    out = {"idx": spec["idx"]}
    for name, hard in VARIANTS.items():
        cfg = {**DEFAULT_CONFIG, "N_max": spec["n_max"], "K_max": spec["k_max"]}
        if hard is not None:
            cfg["hardening"] = hard
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        prior = DoTime(config=cfg, seed=seed)
        xo, xi, iv, scm = prior.generate_pair(T=spec["T"])
        sec = time.perf_counter() - t0
        zo, zi = float(xo.abs().max()) == 0.0, float(xi.abs().max()) == 0.0
        rec = {
            "N": int(xo.shape[1]),
            "cls": type(scm).__name__,
            "diverged": bool(zo or zi),
            "sec": sec,
        }
        if not rec["diverged"]:
            rec["saturated"] = bool((xo.abs() >= 999).any() or (xi.abs() >= 999).any())
            ep = episode_from_pair(xo, xi, iv, scm_id=spec["idx"])
            rec["y_int"] = float(ep.y_true.reshape(-1)[0])
            rec["y_obs"] = float(query_obs_levels(ep)[0])
            rec["sd_obs"] = float(np.median(xo.std(0).numpy()))
            if hasattr(scm, "freeze_noise"):
                rec["rho"] = companion_spectral_radius(scm)
                # True causal effect: rerun both arms on one noise realisation.
                burn = cfg["burn_in"]
                scm.freeze_noise(
                    spec["T"] + burn, generator=torch.Generator().manual_seed(seed + 7)
                )
                co = scm.sample_observational(T=spec["T"], burn_in=burn)
                ci = scm.sample_interventional(T=spec["T"], intervention=iv, burn_in=burn)
                scm.clear_noise()
                if float(co.abs().max()) > 0 and float(ci.abs().max()) > 0:
                    onset = min(iv.times)
                    d = (ci[onset:] - co[onset:]).abs()
                    for tg in iv.targets:
                        d[:, tg] = 0.0
                    rec["effect"] = float((d.max(0).values / co.std(0).clamp(min=1e-8)).max())
        out[name] = rec
    return out


def arm_stats(x: np.ndarray) -> dict:
    """Nonzero fraction, mean and variance of one target arm.

    Parameters
    ----------
    x : numpy.ndarray
        Targets of surviving episodes.

    Returns
    -------
    dict
        Summary statistics.
    """
    return {
        "n": int(x.size),
        "nonzero_frac": float(np.mean(x != 0)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
    }


def summarise(recs: list[dict], name: str) -> dict:
    """Aggregate one variant over all episodes of a configuration.

    Parameters
    ----------
    recs : list of dict
        Worker outputs.
    name : str
        A key of ``VARIANTS``.

    Returns
    -------
    dict
        Aggregate statistics.

    Raises
    ------
    AssertionError
        If either target arm of the surviving episodes is mostly zero.
    """
    r = [x[name] for x in recs]
    ok = [x for x in r if not x["diverged"]]
    yi = np.array([x["y_int"] for x in ok])
    yo = np.array([x["y_obs"] for x in ok])
    qa = {"y_int": arm_stats(yi), "y_obs": arm_stats(yo), "effect": arm_stats(yi - yo)}
    assert qa["y_int"]["nonzero_frac"] >= 0.5, qa
    assert qa["y_obs"]["nonzero_frac"] >= 0.5, qa
    eff = np.array([x["effect"] for x in ok if "effect" in x])
    big = [x for x in r if x["N"] > 20]
    by_cls = {}
    for c in sorted({x["cls"] for x in r}):
        rc = [x for x in r if x["cls"] == c]
        by_cls[c] = {"n": len(rc), "diverged_frac": float(np.mean([x["diverged"] for x in rc]))}
    return {
        "n_episodes": len(r),
        "diverged_frac": float(np.mean([x["diverged"] for x in r])),
        "diverged_frac_N_gt_20": float(np.mean([x["diverged"] for x in big])) if big else None,
        "saturated_frac_of_survivors": float(np.mean([x["saturated"] for x in ok])) if ok else None,
        "median_N_all": float(np.median([x["N"] for x in r])),
        "median_N_survivors": float(np.median([x["N"] for x in ok])) if ok else None,
        "median_sd_obs": float(np.median([x["sd_obs"] for x in ok])) if ok else None,
        "effect_n": int(eff.size),
        "effect_median": float(np.median(eff)) if eff.size else None,
        "effect_frac_ge_0p1": float(np.mean(eff >= 0.1)) if eff.size else None,
        "rho_max": float(max((x["rho"] for x in ok if "rho" in x), default=float("nan"))),
        "mean_sec": float(np.mean([x["sec"] for x in r])),
        "by_class": by_cls,
        "target_qa": qa,
    }


def main() -> None:
    """Run both configurations and write ``hardening_measurement.json``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("hardening_measurement.json")
    )
    args = ap.parse_args()
    runs = [(10, 3, 500), (60, 8, 300)]
    result = {"suite_seed": SUITE_SEED, "T": args.T, "variants": VARIANTS, "configs": []}
    with Pool(args.workers) as pool:
        for n_max, k_max, n in runs:
            t0 = time.perf_counter()
            specs = [{"idx": i, "n_max": n_max, "k_max": k_max, "T": args.T} for i in range(n)]
            recs = pool.map(one, specs, chunksize=2)
            entry = {"N_max": n_max, "K_max": k_max, "n": n, "wall_s": time.perf_counter() - t0}
            for name in VARIANTS:
                entry[name] = summarise(recs, name)
                s = entry[name]
                print(
                    f"N_max={n_max} K_max={k_max} {name:12s} diverged={s['diverged_frac']:.3f} "
                    f"(N>20: {s['diverged_frac_N_gt_20']}) saturated={s['saturated_frac_of_survivors']:.3f} "
                    f"medN all/surv={s['median_N_all']:.0f}/{s['median_N_survivors']:.0f} sd={s['median_sd_obs']:.3g} "
                    f"effect>=0.1={s['effect_frac_ge_0p1']} (n={s['effect_n']}) rho_max={s['rho_max']:.3f} sec={s['mean_sec']:.2f}"
                )
                print(f"   by class: {s['by_class']}")
            result["configs"].append(entry)
            args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
