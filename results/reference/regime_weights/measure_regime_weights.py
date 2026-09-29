#!/usr/bin/env python
"""Measure the opt-in regime-switching fix ``regime_canonical_weights``.

In v1.0.0 no regime-switching mechanism reads its parents. Its weights are keyed
by per-regime node names (``x3``, ``u1``, ``y``) while the SCM passes parent
values under canonical names (``X0..X{N-1}``), so every variable is independent
noise. ``DoTime(config={"regime_canonical_weights": True})`` re-keys the
weights and zeroes any arm whose values exceed 500. This script measures what
the fix does to the two affected suites.

Episodes are sampled as ``dotime._build.make_episode`` samples them, with each
suite's released seed scheme (``episode_seed(suite_seed, idx)``, the global
torch seed, then ``DoTime(seed=...)``):

``regime``
    ``generate_regime_pair(T, num_regimes=d)``, the dot-RegimeSwitch-v1 path,
    using the released index blocks for d = 2, 3 and 5.
``generic_regime_share``
    ``generate_pair(T)`` for the dot-Generic-100k indices whose first draw
    selects a regime-switching SCM (the ``regime_switching_prob=0.15`` share).

The release build does not seed the global numpy RNG, which draws the regime
paths. That is harmless for v1.0.0, whose output ignores the regime path, but
not with the fix, so this script seeds it per episode.

Each episode runs with the flag off and on. The flag draws no random numbers,
so the runs are paired (same graphs, weights, noise, regime paths and
interventions), which the script asserts through the RNG states they leave
behind. Per arm it records whether the arm was zeroed, the largest magnitude
in the simulated buffer (burn-in included), whether that hit the +-1000 clip
(the arm would have saturated silently without the new check), and whether the
sparse every-50-steps check of ``TemporalSCM`` would have caught it. Per
variant it reports the per-arm target statistics (asserted, per the repository
rule) and the lag-1 structure of the observational arm.

    python results/reference/regime_weights/measure_regime_weights.py --workers 15

If ``dotime.hardening`` is importable, the flag is also measured together with
``RECOMMENDED_HARDENING``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
import warnings
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# From the v1.0.0 manifests (build_release: base seed 20260719 + 1000 * position).
SUITE_SEEDS = {"regime": 20262719, "generic": 20264719}
# dot-RegimeSwitch-v1 index blocks: (num_regimes, first index), 3333 per density.
REGIME_BLOCKS = ((2, 0), (3, 3333), (5, 6666))
# DoTime.sample_scm picks a regime-switching SCM when its first draw lands here
# (chain_prob, chain_prob + regime_switching_prob).
REGIME_SHARE = (0.15, 0.30)
CLIP_HIT = 999.0

_BUFFERS: list[dict] = []


def _install_hook() -> None:
    """Record every simulated buffer's peak magnitude before the divergence check.

    Runs once per worker process. The hook only reads the buffer and then calls
    the original check, so the sampled trajectories are unchanged.
    """
    import torch

    from dotime.regime_switching import RegimeSwitchingTemporalSCM

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    original = RegimeSwitchingTemporalSCM._diverged

    def hooked(self, buffer):
        a = buffer.abs()
        # TemporalSCM checks row t - 1 at every t that is a positive multiple of 50.
        rows = [t - 1 for t in range(50, buffer.shape[0], 50)]
        _BUFFERS.append(
            {
                "buf_max": float(a.max()),
                "sparse_catch": bool(rows) and bool((a[rows].amax(dim=1) > 500).any()),
            }
        )
        return original(self, buffer)

    RegimeSwitchingTemporalSCM._diverged = hooked


def _digest(*arrays) -> str:
    """SHA-256 over the raw bytes of tensors and arrays.

    Parameters
    ----------
    *arrays : torch.Tensor or numpy.ndarray
        States to fingerprint.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def _lag_structure(x: np.ndarray) -> dict:
    """Lag-1 autocorrelation and cross-correlations of one trajectory.

    Parameters
    ----------
    x : numpy.ndarray
        Trajectory of shape ``(T, N)``.

    Returns
    -------
    dict
        ``acf1``: per-variable ``|corr(x_i[t], x_i[t-1])|``. ``xcorr0`` and
        ``xcorr1``: ``|corr(x_i[t], x_j[t])|`` for ``i < j`` and
        ``|corr(x_i[t], x_j[t-1])|`` for ``i != j``. Constant columns are skipped.
    """
    x = x[:, x.std(0) > 0]
    n = x.shape[1]
    if n == 0:
        return {"acf1": [], "xcorr0": [], "xcorr1": []}

    def z(a):
        return (a - a.mean(0)) / a.std(0).clip(1e-12)

    now, prev = z(x[1:]), z(x[:-1])
    c1 = np.abs(now.T @ prev / now.shape[0])
    c0 = np.abs(np.corrcoef(x, rowvar=False)) if n > 1 else np.ones((1, 1))
    return {
        "acf1": np.diag(c1).tolist(),
        "xcorr0": c0[np.triu_indices(n, k=1)].tolist(),
        "xcorr1": c1[~np.eye(n, dtype=bool)].tolist(),
    }


def _arm(x, buf: dict) -> dict:
    """Summarise one returned arm together with its simulated buffer.

    Parameters
    ----------
    x : torch.Tensor
        The arm as returned, shape ``(T, N)``.
    buf : dict
        The hook's record for the buffer behind it.

    Returns
    -------
    dict
        Zeroing, peak magnitude, clip and sparse-check flags.
    """
    return {
        "zeroed": float(x.abs().max()) == 0.0,
        "buf_max": buf["buf_max"],
        "clip_hit": buf["buf_max"] >= CLIP_HIT,
        "over_500": buf["buf_max"] > 500,
        "sparse_catch": buf["sparse_catch"],
    }


def one(spec: dict) -> dict:
    """Sample one episode under every requested variant (runs in a worker).

    Parameters
    ----------
    spec : dict
        ``path``, ``idx``, ``seed``, ``num_regimes`` (regime path), ``n_max``,
        ``k_max``, ``T`` and ``variants``.

    Returns
    -------
    dict
        Per-variant record: arm flags, targets, lag structure, RNG fingerprint.

    Raises
    ------
    AssertionError
        If the episode is not a regime-switching SCM, or if the regime SCM did
        not run its divergence check once per arm.
    """
    import torch

    from dotime import DoTime
    from dotime.benchmarks import episode_from_pair
    from dotime.evaluation import query_obs_levels
    from dotime.utils import DEFAULT_CONFIG

    out = {k: spec[k] for k in ("path", "idx", "num_regimes", "n_max", "k_max")}
    for name, (flag, hardening) in spec["variants"].items():
        config = {
            **DEFAULT_CONFIG,
            "N_max": spec["n_max"],
            "K_max": spec["k_max"],
            "regime_canonical_weights": flag,
        }
        if hardening is not None:
            config["hardening"] = hardening
        np.random.seed(spec["seed"])
        torch.manual_seed(spec["seed"])
        _BUFFERS.clear()
        t0 = time.perf_counter()
        prior = DoTime(config=config, seed=spec["seed"])
        if spec["path"] == "regime":
            xo, xi, iv, scm = prior.generate_regime_pair(
                T=spec["T"], num_regimes=spec["num_regimes"]
            )
        else:
            xo, xi, iv, scm = prior.generate_pair(T=spec["T"])
        sec = time.perf_counter() - t0
        assert type(scm).__name__ == "RegimeSwitchingTemporalSCM", type(scm).__name__
        assert len(_BUFFERS) == 2, len(_BUFFERS)
        ep = episode_from_pair(xo, xi, iv)
        rec = {
            "N": int(xo.shape[1]),
            "K": int(scm.dags[0].K),
            "R": int(scm.num_regimes),
            "sec": sec,
            "rng": _digest(
                prior.generator.get_state().numpy(),
                torch.get_rng_state().numpy(),
                np.random.get_state()[1],
            ),
            "iv": json.dumps(iv.to_dict(), sort_keys=True),
            "obs": _arm(xo, _BUFFERS[0]),
            "int": _arm(xi, _BUFFERS[1]),
            "y_int": float(ep.y_true.reshape(-1)[0]),
            "y_obs": float(query_obs_levels(ep)[0]),
        }
        if not rec["obs"]["zeroed"]:
            rec["lag"] = _lag_structure(xo.numpy().astype(np.float64))
        out[name] = rec
    return out


def _stats(x: np.ndarray) -> dict:
    """Nonzero fraction, mean and variance of one target arm.

    Parameters
    ----------
    x : numpy.ndarray
        Target values.

    Returns
    -------
    dict
        Summary statistics (``None`` entries when ``x`` is empty).
    """
    if x.size == 0:
        return {"n": 0, "nonzero_frac": None, "mean": None, "var": None}
    return {
        "n": int(x.size),
        "nonzero_frac": float(np.mean(x != 0)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
    }


def _frac(flags: list[bool]) -> float | None:
    """Mean of a list of flags, ``None`` when it is empty."""
    return float(np.mean(flags)) if flags else None


def summarise(recs: list[dict], name: str) -> dict:
    """Aggregate one variant over a group of episodes.

    Parameters
    ----------
    recs : list of dict
        Worker outputs that ran this variant.
    name : str
        Variant key.

    Returns
    -------
    dict
        Divergence, saturation, target and lag-structure statistics.

    Raises
    ------
    AssertionError
        If a surviving target arm is not almost surely nonzero, has zero
        variance, or is not finite, or if a zeroed arm left a nonzero target.
    """
    r = [x[name] for x in recs]
    ok = [x for x in r if not (x["obs"]["zeroed"] or x["int"]["zeroed"])]
    arms = [a for x in r for a in (x["obs"], x["int"])]
    y_int_all = np.array([x["y_int"] for x in r])
    y_obs_all = np.array([x["y_obs"] for x in r])
    y_int = np.array([x["y_int"] for x in ok])
    y_obs = np.array([x["y_obs"] for x in ok])
    qa = {
        "all_episodes": {
            "y_int": _stats(y_int_all),
            "y_obs": _stats(y_obs_all),
            "effect": _stats(y_int_all - y_obs_all),
        },
        "survivors": {
            "y_int": _stats(y_int),
            "y_obs": _stats(y_obs),
            "effect": _stats(y_int - y_obs),
        },
    }
    # A zeroed arm must leave an exactly-zero target, otherwise the zeroing
    # accounting above is wrong.
    assert all(x["y_int"] == 0.0 for x in r if x["int"]["zeroed"])
    assert all(x["y_obs"] == 0.0 for x in r if x["obs"]["zeroed"])
    if len(ok) >= 10:
        for arm in ("y_int", "y_obs"):
            s = qa["survivors"][arm]
            assert s["nonzero_frac"] >= 0.99, (name, arm, s)
            assert s["var"] > 0, (name, arm, s)
            assert np.isfinite(s["mean"]), (name, arm, s)
            assert np.isfinite(s["var"]), (name, arm, s)
    lag = [x["lag"] for x in r if "lag" in x]
    acf1 = np.array([v for d in lag for v in d["acf1"]])
    xc0 = np.array([v for d in lag for v in d["xcorr0"]])
    xc1 = np.array([v for d in lag for v in d["xcorr1"]])
    over = [x["obs"]["over_500"] or x["int"]["over_500"] for x in r]
    missed = [any(a["over_500"] and not a["sparse_catch"] for a in (x["obs"], x["int"])) for x in r]
    out = {
        "n_episodes": len(r),
        "diverged_frac": _frac([x["obs"]["zeroed"] or x["int"]["zeroed"] for x in r]),
        "obs_zeroed_frac": _frac([x["obs"]["zeroed"] for x in r]),
        "int_zeroed_frac": _frac([x["int"]["zeroed"] for x in r]),
        "over_500_frac": _frac(over),
        "clip_hit_frac": _frac([x["obs"]["clip_hit"] or x["int"]["clip_hit"] for x in r]),
        "arm_clip_hit_frac": _frac([a["clip_hit"] for a in arms]),
        "sparse_check_would_miss_frac": _frac(missed),
        "sparse_check_would_miss_of_over_500": (sum(missed) / sum(over)) if any(over) else None,
        "clip_hit_among_survivors": _frac(
            [x["obs"]["clip_hit"] or x["int"]["clip_hit"] for x in ok]
        ),
        "median_buf_max_survivors": float(
            np.median([max(x["obs"]["buf_max"], x["int"]["buf_max"]) for x in ok])
        )
        if ok
        else None,
        "median_N_all": float(np.median([x["N"] for x in r])),
        "median_N_survivors": float(np.median([x["N"] for x in ok])) if ok else None,
        "acf1": {
            "n_vars": int(acf1.size),
            "median": float(np.median(acf1)) if acf1.size else None,
            "p90": float(np.quantile(acf1, 0.9)) if acf1.size else None,
            "frac_gt_0p3": float(np.mean(acf1 > 0.3)) if acf1.size else None,
        },
        "xcorr0_median": float(np.median(xc0)) if xc0.size else None,
        "xcorr1_median": float(np.median(xc1)) if xc1.size else None,
        "mean_sec": float(np.mean([x["sec"] for x in r])),
        "target_qa": qa,
    }
    by_r = {}
    for rr in sorted({x["R"] for x in r}):
        rs = [x for x in r if x["R"] == rr]
        by_r[str(rr)] = {
            "n": len(rs),
            "diverged_frac": _frac([x["obs"]["zeroed"] or x["int"]["zeroed"] for x in rs]),
        }
    out["by_num_regimes"] = by_r
    by_n = {}
    for lo, hi in ((3, 10), (11, 30), (31, 60)):
        rs = [x for x in r if lo <= x["N"] <= hi]
        if rs:
            by_n[f"N{lo}-{hi}"] = {
                "n": len(rs),
                "diverged_frac": _frac([x["obs"]["zeroed"] or x["int"]["zeroed"] for x in rs]),
            }
    out["by_N"] = by_n
    return out


def _check_pairing(recs: list[dict]) -> int:
    """Assert that all variants of an episode drew identical randomness.

    Parameters
    ----------
    recs : list of dict
        Worker outputs.

    Returns
    -------
    int
        Number of episodes with at least two variants compared.

    Raises
    ------
    AssertionError
        If the RNG states or interventions of two variants differ.
    """
    checked = 0
    for x in recs:
        runs = [v for k, v in x.items() if isinstance(v, dict) and "rng" in v]
        if len(runs) > 1:
            assert len({v["rng"] for v in runs}) == 1, ("rng", x["path"], x["idx"])
            assert len({v["iv"] for v in runs}) == 1, ("iv", x["path"], x["idx"])
            assert len({(v["N"], v["K"], v["R"]) for v in runs}) == 1, ("shape", x["idx"])
            checked += 1
    return checked


def _regime_share_indices(n: int) -> list[int]:
    """First ``n`` dot-Generic-100k indices whose SCM is regime-switching.

    Parameters
    ----------
    n : int
        How many indices to return.

    Returns
    -------
    list of int
        Episode indices, ascending.
    """
    import torch

    from dotime._build import episode_seed

    out, idx = [], 0
    while len(out) < n:
        g = torch.Generator()
        g.manual_seed(episode_seed(SUITE_SEEDS["generic"], idx))
        if REGIME_SHARE[0] <= torch.rand(1, generator=g).item() < REGIME_SHARE[1]:
            out.append(idx)
        idx += 1
    return out


def main() -> None:
    """Run both configurations and write the measurement JSON."""
    from dotime._build import episode_seed

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--scale", type=float, default=1.0, help="Scale every episode count.")
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("regime_weights_measurement.json")
    )
    ap.add_argument("--episodes-out", type=Path, default=None, help="Optional per-episode JSONL.")
    args = ap.parse_args()

    variants: dict = {"off": (False, None), "on": (True, None)}
    if importlib.util.find_spec("dotime.hardening") is not None:
        from dotime.hardening import RECOMMENDED_HARDENING

        variants["on_hardened"] = (True, dict(RECOMMENDED_HARDENING))
    # (N_max, K_max, regime episodes per density, generic regime-share episodes,
    # episodes per group that also run the flag-off control). The flag-off
    # control is cheap at the default size but costs ~20 s per pair at 60/8.
    runs = [(10, 3, 200, 300, None), (60, 8, 50, 90, 20)]

    def scaled(n):
        return max(1, round(n * args.scale))

    import dotime

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "src"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[3],
        ).stdout.strip()
    )
    result = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": head,
        "src_dirty": dirty,
        # Repo-relative, so the committed JSON carries no machine-local path, and
        # a dotime imported from outside this checkout shows up as "../...".
        "dotime_file": os.path.relpath(dotime.__file__, Path(__file__).parents[3]),
        "T": args.T,
        "suite_seeds": SUITE_SEEDS,
        "numpy_global_seed": "np.random.seed(episode_seed) per episode",
        "variants": {
            k: {"regime_canonical_weights": f, "hardening": h} for k, (f, h) in variants.items()
        },
        "configs": [],
    }
    all_recs = []
    with Pool(args.workers, initializer=_install_hook) as pool:
        for n_max, k_max, per_d, n_gen, off_n in runs:
            t0 = time.perf_counter()
            specs = []
            groups = [
                (
                    f"regime_d{d}",
                    "regime",
                    d,
                    [start + k for k in range(scaled(per_d))],
                    SUITE_SEEDS["regime"],
                )
                for d, start in REGIME_BLOCKS
            ]
            groups.append(
                (
                    "generic_regime_share",
                    "generic",
                    None,
                    _regime_share_indices(scaled(n_gen)),
                    SUITE_SEEDS["generic"],
                )
            )
            for gname, path, d, idxs, suite_seed in groups:
                for j, idx in enumerate(idxs):
                    run_off = off_n is None or j < scaled(off_n)
                    specs.append(
                        {
                            "group": gname,
                            "path": path,
                            "idx": idx,
                            "seed": episode_seed(suite_seed, idx),
                            "num_regimes": d,
                            "n_max": n_max,
                            "k_max": k_max,
                            "T": args.T,
                            "variants": {
                                k: v for k, v in variants.items() if k != "off" or run_off
                            },
                        }
                    )
            recs = pool.map(one, specs, chunksize=1)
            for s, rec in zip(specs, recs, strict=True):
                rec["group"] = s["group"]
            paired = _check_pairing(recs)
            entry = {
                "N_max": n_max,
                "K_max": k_max,
                "n_specs": len(specs),
                "paired_episodes_checked": paired,
                "wall_s": time.perf_counter() - t0,
                "groups": {},
            }
            for gname in [g[0] for g in groups] + ["regime_all"]:
                grecs = [
                    x
                    for x in recs
                    if x["group"] == gname or (gname == "regime_all" and x["path"] == "regime")
                ]
                entry["groups"][gname] = {}
                for name in variants:
                    have = [x for x in grecs if name in x]
                    if have:
                        entry["groups"][gname][name] = summarise(have, name)
                    s = entry["groups"][gname].get(name)
                    if s:
                        print(
                            f"N_max={n_max} K_max={k_max} {gname:21s} {name:11s} n={s['n_episodes']:4d} "
                            f"diverged={s['diverged_frac']:.3f} clip_hit={s['clip_hit_frac']:.3f} "
                            f"sparse_miss={s['sparse_check_would_miss_frac']:.3f} "
                            f"acf1>0.3={s['acf1']['frac_gt_0p3']} medN={s['median_N_all']:.0f}/{s['median_N_survivors']} "
                            f"sec={s['mean_sec']:.2f}",
                            flush=True,
                        )
            result["configs"].append(entry)
            all_recs += recs
            args.out.write_text(json.dumps(result, indent=2))
    if args.episodes_out is not None:
        with args.episodes_out.open("w") as fh:
            for x in all_recs:
                slim = {k: v for k, v in x.items() if not isinstance(v, dict)}
                for name in variants:
                    if name in x:
                        slim[name] = {k: v for k, v in x[name].items() if k != "lag"}
                fh.write(json.dumps(slim) + "\n")


if __name__ == "__main__":
    main()
