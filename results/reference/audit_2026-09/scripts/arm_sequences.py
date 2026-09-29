"""Per-arm divergence accounting by regeneration from the published seeds.

Each task regenerates one episode's retry sequence exactly as the generic
branch of ``dotime._build.make_episode`` would (attempt seed ``seed`` then
``seed * 100003 + attempt``, global torch seed reset per attempt) and records,
per attempt, whether the observational and/or interventional arm came back
all-zero. The sequence stops at the first attempt with neither arm zeroed, so
one sequence answers every (gate, R) question: the both-arm gate always stops
no later than the either-arm gate.

Modes (each gated on reproducing an already-published number):
  release       make_episode(retries=0) vs the cached release file, bit for bit,
                plus retry sequences for the one-arm rows and a random sample.
  testcfg       the tests/test_build_release.py configuration behind the
                paper's "0/200 zeroed" claim.
  stationarity  the dotime-diagnose-stationarity sample behind Appendix B.
  scaling       the scaling_lag.py configurations.

Nothing here modifies the package; it is analysis only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np


def _hash(arr: np.ndarray) -> str:
    """sha256 of an array's raw bytes (dtype and sign of zero included).

    Args:
        arr: Array to hash.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _episode_hashes(x_obs, x_int, y_true, query_target, query_time, intervention: dict) -> dict:
    """Field hashes that define bit-identity of an episode's released content.

    Args:
        x_obs, x_int, y_true, query_target, query_time: Array-likes.
        intervention: The intervention dict (``InterventionSpec.to_dict`` form).

    Returns:
        Dict of field name to hex digest.
    """
    return {
        "x_obs": _hash(np.asarray(x_obs, dtype=np.float32).reshape(-1)),
        "x_int": _hash(np.asarray(x_int, dtype=np.float32).reshape(-1)),
        "y_true": _hash(np.asarray(y_true, dtype=np.float32).reshape(-1)),
        "query_target": _hash(np.asarray(query_target, dtype=np.int64).reshape(-1)),
        "query_time": _hash(np.asarray(query_time, dtype=np.float32).reshape(-1)),
        "intervention": hashlib.sha256(
            json.dumps(json.loads(json.dumps(intervention)), sort_keys=True).encode()
        ).hexdigest(),
    }


def _moments(x: np.ndarray) -> dict:
    """First/last 50-step per-variable mean and variance (stationarity check).

    Args:
        x: Observational trajectory, shape (T, N).

    Returns:
        Dict of lists, mirroring ``dotime.reference.stationarity.diagnose``.
    """
    return {
        "em": x[:50].mean(axis=0).tolist(),
        "lm": x[-50:].mean(axis=0).tolist(),
        "ev": x[:50].var(axis=0).tolist(),
        "lv": x[-50:].var(axis=0).tolist(),
    }


def run_task(task: dict) -> dict:
    """Regenerate one episode's retry sequence (runs in a worker process).

    Args:
        task: Dict with ``seed``, ``T``, ``r_max`` and optional ``config``
            (DEFAULT_CONFIG overrides), ``activations`` ("all"/"identity"),
            ``build_spec`` (a make_episode spec for the bit-identity check) and
            ``stationarity`` (record rho and burn-in moments at attempt 0).

    Returns:
        Dict with the per-attempt sequence and, if requested, field hashes of
        the make_episode output and stationarity data.
    """
    import torch
    from torch import nn

    from dotime import DoTime
    from dotime.utils import DEFAULT_CONFIG

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    out: dict = {"key": task["key"]}

    # One make_episode call per requested stability_retries value, so a single
    # task checks both the retries=0 reproduction and the hardened path.
    build_specs = task.get("build_specs") or (
        [task["build_spec"]] if task.get("build_spec") else []
    )
    if build_specs:
        from dotime._build import make_episode

        out["builds"] = {}
        for bspec in build_specs:
            ep = make_episode(dict(bspec))
            out["builds"][str(int(bspec.get("stability_retries", 0)))] = {
                "hashes": _episode_hashes(
                    ep.x_obs.numpy(),
                    ep.x_int.numpy(),
                    ep.y_true.numpy(),
                    ep.query_target.numpy(),
                    ep.query_time.numpy(),
                    ep.intervention.to_dict(),
                ),
                "metadata": {k: v for k, v in ep.metadata.items() if k != "y_oracle"},
                "zero": [float(ep.x_obs.abs().max()) == 0.0, float(ep.x_int.abs().max()) == 0.0],
            }
        first = out["builds"][str(int(build_specs[0].get("stability_retries", 0)))]
        out["build_hashes"], out["build_metadata"], out["build_zero"] = (
            first["hashes"],
            first["metadata"],
            first["zero"],
        )

    if task.get("kind", "generic") != "generic":
        return out  # regime rows: bit-identity only (no zeroed arms in the file)

    cfg = {**DEFAULT_CONFIG, **task["config"]} if task.get("config") else None
    seq = []
    seed = task["seed"]
    for attempt in range(task["r_max"] + 1):
        s = seed if attempt == 0 else seed * 100003 + attempt
        t0 = time.perf_counter()
        torch.manual_seed(s)
        prior = DoTime(config=cfg, seed=s)
        if task.get("activations", "all") == "identity":
            prior.activations[:] = [nn.Identity()]
        x_obs, x_int, _iv, scm = prior.generate_pair(T=task["T"])
        k_lag = getattr(scm, "_K", None)  # same source as scaling_lag.py's _lag_order
        rec = {
            "obs_zero": float(x_obs.abs().max()) == 0.0,
            "int_zero": float(x_int.abs().max()) == 0.0,
            "N": int(x_obs.shape[1]),
            "K": int(k_lag) if k_lag is not None else None,
            "cls": type(scm).__name__,
            "gen_s": time.perf_counter() - t0,
            # Tensor hashes let the parent check that make_episode(retries=R)
            # ships exactly the attempt the either-arm gate predicts.
            "h_obs": _hash(x_obs.detach().cpu().numpy().astype(np.float32).reshape(-1)),
            "h_int": _hash(x_int.detach().cpu().numpy().astype(np.float32).reshape(-1)),
        }
        if task.get("stationarity") and attempt == 0:
            from dotime.reference.stationarity import companion_rho

            r, acts = companion_rho(scm)
            rec["rho"] = None if r is None else float(r)
            rec["all_linear"] = bool(acts) and all(a == "Identity" for a in acts)
            rec["moments"] = _moments(x_obs.detach().cpu().numpy())
        seq.append(rec)
        if not (rec["obs_zero"] or rec["int_zero"]):
            break
    out["seq"] = seq
    return out


def category(rec: dict) -> str:
    """Classify one attempt by which arms are zeroed.

    Args:
        rec: Per-attempt record with ``obs_zero`` and ``int_zero``.

    Returns:
        One of "both", "obs_only", "int_only", "neither".
    """
    o, i = rec["obs_zero"], rec["int_zero"]
    return "both" if (o and i) else "obs_only" if o else "int_only" if i else "neither"


def outcome(seq: list[dict], gate: str, r: int) -> tuple[int, dict]:
    """Final attempt a build with ``gate`` and ``r`` retries would ship.

    Args:
        seq: Recorded attempt sequence (stops at the first clean attempt).
        gate: "both" (current _build rule) or "either" (proposed rule).
        r: stability_retries.

    Returns:
        (attempt index, attempt record) that the build keeps.

    Raises:
        ValueError: If the sequence is too short to decide (r beyond r_max).
    """
    for k, rec in enumerate(seq):
        clean = (
            not (rec["obs_zero"] and rec["int_zero"])
            if gate == "both"
            else category(rec) == "neither"
        )
        if k == r or clean:
            return k, rec
    raise ValueError(f"sequence of length {len(seq)} cannot decide r={r}")


def summarize_sequences(results: list[dict], rs: tuple[int, ...]) -> dict:
    """Category counts of shipped attempts for both gates and each R.

    Args:
        results: Worker outputs with ``seq``.
        rs: Retry counts to evaluate.

    Returns:
        Nested dict gate -> R -> {counts, mean_attempts, N_median}.
    """
    summ: dict = {}
    for gate in ("both", "either"):
        summ[gate] = {}
        for r in rs:
            cats = {"both": 0, "obs_only": 0, "int_only": 0, "neither": 0}
            att, ns = [], []
            for res in results:
                k, rec = outcome(res["seq"], gate, r)
                cats[category(rec)] += 1
                att.append(k + 1)
                ns.append(rec["N"])
            summ[gate][str(r)] = {
                "counts": cats,
                "either_zeroed": cats["both"] + cats["obs_only"] + cats["int_only"],
                "mean_attempts": float(np.mean(att)),
                "N_median": float(np.median(ns)),
            }
    return summ


def pool_map(tasks: list[dict], workers: int) -> list[dict]:
    """Run tasks on a fork pool of single-threaded workers, with progress.

    Args:
        tasks: Task dicts for :func:`run_task`.
        workers: Process count.

    Returns:
        Results in task order.
    """
    res: dict = {}
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=2), 1):
            res[r["key"]] = r
            if n % 200 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} done ({time.time() - t0:.0f}s)", flush=True)
    return [res[t["key"]] for t in tasks]


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def _file_rows(suite_dir: Path, rows: list[int], shard_size: int = 5000) -> dict[int, dict]:
    """Read selected rows of a released suite (md5-verified by the scan step).

    Args:
        suite_dir: Suite directory.
        rows: Global row indices.
        shard_size: Rows per shard in the release layout.

    Returns:
        Map of global row index to column dict.
    """
    import pyarrow.parquet as pq

    manifest = json.loads((suite_dir / "manifest.json").read_text())
    out: dict[int, dict] = {}
    by_shard: dict[int, list[int]] = {}
    for r in rows:
        by_shard.setdefault(r // shard_size, []).append(r)
    for si, rs in by_shard.items():
        t = pq.read_table(suite_dir / manifest["shards"][si]["file"]).take(
            [r % shard_size for r in rs]
        )
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        for j, r in enumerate(rs):
            out[r] = {c: cols[c][j] for c in cols}
    return out


def mode_release(args) -> dict:
    """Bit-identity of make_episode(retries=0) vs the release, plus sequences.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary dict.
    """
    from dotime._build import episode_specs

    scan = json.loads(Path(args.scan_json).read_text())["dot-Generic-100k-1.0.0"]
    one_arm = {r["row"]: r["category"] for r in scan["one_arm_rows"]}
    rng = np.random.default_rng(args.sample_seed)
    rand = sorted(int(x) for x in rng.choice(100_000, size=args.n_random, replace=False))
    rows = sorted(set(one_arm) | set(rand))
    gen_cfg = {"generator": "generic", "T": 200, "n_episodes": 100_000, "stability_retries": 0}
    specs = episode_specs(gen_cfg, args.generic_seed, 1.0)
    tasks = [
        {
            "key": f"g{r}",
            "seed": specs[r]["seed"],
            "T": 200,
            "r_max": args.r_max,
            "build_spec": specs[r],
        }
        for r in rows
    ]
    # RegimeSwitch: bit-identity sample only (the file has no zeroed arm).
    reg_cfg = {
        "generator": "regime",
        "T": 200,
        "densities": {2: 1, 3: 2, 5: 3},
        "n_episodes": 10_000,
    }
    reg_specs = episode_specs(reg_cfg, args.regime_seed, 1.0)
    assert len(reg_specs) == 9999
    reg_rows = sorted(int(x) for x in rng.choice(9999, size=args.n_regime, replace=False))
    tasks += [{"key": f"r{r}", "kind": "regime", "build_spec": reg_specs[r]} for r in reg_rows]
    print(
        f"release mode: {len(rows)} generic rows ({len(one_arm)} one-arm, {len(rand)} random), "
        f"{len(reg_rows)} regime rows",
        flush=True,
    )
    results = pool_map(tasks, args.workers)

    g_file = _file_rows(Path(args.generic_dir), rows)
    r_file = _file_rows(Path(args.regime_dir), reg_rows)
    mism, meta_diff = [], {}
    per_row = []
    for t, res in zip(tasks, results, strict=True):
        r = int(t["key"][1:])
        f = (g_file if t["key"][0] == "g" else r_file)[r]
        fh = _episode_hashes(
            f["x_obs"],
            f["x_int"],
            f["y_true"],
            f["query_target"],
            f["query_time"],
            json.loads(f["intervention_json"]),
        )
        bad = [k for k in fh if fh[k] != res["build_hashes"][k]]
        if bad:
            mism.append({"key": t["key"], "fields": bad})
        fmeta = json.loads(f["metadata_json"]) if f["metadata_json"] else {}
        extra = {k: v for k, v in res["build_metadata"].items() if fmeta.get(k, object()) != v}
        sig = json.dumps(sorted(extra), sort_keys=True)
        meta_diff[sig] = meta_diff.get(sig, 0) + 1
        if t["key"][0] == "g":
            per_row.append(
                {
                    "row": r,
                    "set": ("one_arm+" if r in one_arm else "")
                    + ("random" if r in set(rand) else ""),
                    "file_category": one_arm.get(r),
                    "build_zero": res["build_zero"],
                    "build_metadata_diverged": res["build_metadata"].get("diverged"),
                    "seq": res["seq"],
                }
            )
    # Consistency: attempt 0 of each sequence must match the build's zero flags.
    seq0_mismatch = [
        p["row"]
        for p in per_row
        if [p["seq"][0]["obs_zero"], p["seq"][0]["int_zero"]] != p["build_zero"]
    ]
    one = [p for p in per_row if p["file_category"]]
    rnd = [p for p in per_row if "random" in p["set"]]
    rs = (0, 1, 2, 3, 5, 10, 20)
    return {
        "n_generic_rows": len(rows),
        "n_regime_rows": len(reg_rows),
        "bit_identity_mismatches": mism,
        "metadata_keys_differing_from_file": meta_diff,
        "seq0_vs_build_zero_mismatch": seq0_mismatch,
        "one_arm_file_category_vs_regen": sum(
            1
            for p in one
            if category({"obs_zero": p["build_zero"][0], "int_zero": p["build_zero"][1]})
            != p["file_category"]
        ),
        "one_arm_build_flag_diverged_true": sum(1 for p in one if p["build_metadata_diverged"]),
        "one_arm_sequences": summarize_sequences(one, rs),
        "random_sample_sequences": summarize_sequences(rnd, rs),
        "per_row": per_row,
    }


def mode_verify(args) -> dict:
    """Post-change check of make_episode against the release and the prototype.

    For every row: retries=0 must reproduce the released fields bit for bit
    with ``diverged`` equal to the either-arm rule on the file's tensors, and
    retries=R must ship exactly the attempt (by tensor hash) that the
    either-arm gate predicts from the independently regenerated sequence.

    Args:
        args: Parsed CLI arguments (uses ``build_retries``).

    Returns:
        Summary dict of mismatch counts and shipped-category counts.
    """
    from dotime._build import episode_specs

    r = args.build_retries
    scan = json.loads(Path(args.scan_json).read_text())["dot-Generic-100k-1.0.0"]
    one_arm = {x["row"]: x["category"] for x in scan["one_arm_rows"]}
    rng = np.random.default_rng(args.sample_seed)  # same draws as mode_release
    rand = sorted(int(x) for x in rng.choice(100_000, size=args.n_random, replace=False))
    rows = sorted(set(one_arm) | set(rand))
    specs = episode_specs(
        {"generator": "generic", "T": 200, "n_episodes": 100_000}, args.generic_seed, 1.0
    )
    reg_specs = episode_specs(
        {"generator": "regime", "T": 200, "densities": {2: 1, 3: 2, 5: 3}, "n_episodes": 10_000},
        args.regime_seed,
        1.0,
    )
    reg_rows = sorted(int(x) for x in rng.choice(9999, size=args.n_regime, replace=False))

    def both(spec):
        return [{**spec, "stability_retries": 0}, {**spec, "stability_retries": r}]

    tasks = [
        {
            "key": f"g{i}",
            "seed": specs[i]["seed"],
            "T": 200,
            "r_max": r,
            "build_specs": both(specs[i]),
        }
        for i in rows
    ]
    tasks += [
        {"key": f"r{i}", "kind": "regime", "build_specs": both(reg_specs[i])} for i in reg_rows
    ]
    print(
        f"verify: {len(rows)} generic + {len(reg_rows)} regime rows, retries 0 and {r}", flush=True
    )
    results = pool_map(tasks, args.workers)
    g_file = _file_rows(Path(args.generic_dir), rows)
    r_file = _file_rows(Path(args.regime_dir), reg_rows)
    summ = {
        "r": r,
        "retries0_field_mismatch": 0,
        "retries0_flag_vs_either_rule_mismatch": 0,
        "retriesR_vs_predicted_attempt_mismatch": 0,
        "retriesR_flag_mismatch": 0,
        "regime_retriesR_changed": 0,
        "flag_true_retries0": {"one_arm": 0, "random": 0},
        "shipped_retriesR": {"one_arm": {}, "random": {}},
    }
    for t, res in zip(tasks, results, strict=True):
        i = int(t["key"][1:])
        f = (g_file if t["key"][0] == "g" else r_file)[i]
        fh = _episode_hashes(
            f["x_obs"],
            f["x_int"],
            f["y_true"],
            f["query_target"],
            f["query_time"],
            json.loads(f["intervention_json"]),
        )
        b0, br = res["builds"]["0"], res["builds"][str(r)]
        if fh != b0["hashes"]:
            summ["retries0_field_mismatch"] += 1
        file_zero = [not np.any(np.asarray(f["x_obs"])), not np.any(np.asarray(f["x_int"]))]
        if b0["metadata"]["diverged"] != any(file_zero):
            summ["retries0_flag_vs_either_rule_mismatch"] += 1
        if br["metadata"]["diverged"] != any(br["zero"]):
            summ["retriesR_flag_mismatch"] += 1
        if t["key"][0] == "r":
            summ["regime_retriesR_changed"] += int(br["hashes"] != fh)
            continue
        _, rec = outcome(res["seq"], "either", r)
        if (br["hashes"]["x_obs"], br["hashes"]["x_int"]) != (rec["h_obs"], rec["h_int"]):
            summ["retriesR_vs_predicted_attempt_mismatch"] += 1
        for name, member in (("one_arm", i in one_arm), ("random", i in set(rand))):
            if member:
                summ["flag_true_retries0"][name] += int(b0["metadata"]["diverged"])
                c = category({"obs_zero": br["zero"][0], "int_zero": br["zero"][1]})
                summ["shipped_retriesR"][name][c] = summ["shipped_retriesR"][name].get(c, 0) + 1
    return summ


def mode_testcfg(args) -> dict:
    """The regression test's 200-episode configuration (paper's "0/200").

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary dict.
    """
    from dotime._build import episode_specs

    cfg = {"generator": "generic", "n_episodes": 300, "T": 200, "seed": 20260714}
    specs = episode_specs(cfg, cfg["seed"], 1.0)[:200]
    tasks = [
        {"key": f"t{i}", "seed": s["seed"], "T": 200, "r_max": args.r_max}
        for i, s in enumerate(specs)
    ]
    results = pool_map(tasks, args.workers)
    return {
        "sequences": summarize_sequences(results, (0, 3, 20)),
        "rows": [{"i": i, "seq": r["seq"]} for i, r in enumerate(results)],
    }


def mode_stationarity(args) -> dict:
    """Recount the Appendix B divergence statistics under both rules.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary dict per activation mode and rule.
    """

    def pct(x):
        return 100.0 * float(np.mean(x)) if len(x) else float("nan")

    out = {}
    for acts in ("all", "identity"):
        tasks = [
            {
                "key": f"{acts}{i}",
                "seed": 20260719 + i,
                "T": 200,
                "r_max": 0,
                "activations": acts,
                "stationarity": True,
            }
            for i in range(2000)
        ]
        print(f"stationarity {acts}", flush=True)
        results = pool_map(tasks, args.workers)
        recs = [r["seq"][0] for r in results]
        rho = np.array([np.nan if r["rho"] is None else r["rho"] for r in recs])
        ok = np.isfinite(rho)
        lin = ok & np.array([r["all_linear"] for r in recs])
        o = np.array([r["obs_zero"] for r in recs])
        i_ = np.array([r["int_zero"] for r in recs])
        res = {
            "n": len(recs),
            "n_measurable": int(ok.sum()),
            "n_rho_lt_1": int((ok & (rho < 1)).sum()),
            "counts": {
                "both": int((o & i_).sum()),
                "obs_only": int((o & ~i_).sum()),
                "int_only": int((~o & i_).sum()),
                "neither": int((~o & ~i_).sum()),
            },
            "one_arm_by_rho": {
                "rho_ge_1": int((ok & (rho >= 1) & (o ^ i_)).sum()),
                "rho_lt_1": int((ok & (rho < 1) & (o ^ i_)).sum()),
                "rho_undefined": int((~ok & (o ^ i_)).sum()),
            },
        }
        for rule, div in (("both", o & i_), ("either", o | i_), ("obs_arm", o)):
            d = {
                "pct_all": pct(div),
                "pct_measurable_subset": pct(div[ok]),
                "pct_diverged_given_rho_ge_1": pct(div[ok][rho[ok] >= 1.0]),
                "pct_diverged_given_rho_lt_1": pct(div[ok][rho[ok] < 1.0]),
                "n_diverged_given_rho_lt_1": int(div[ok][rho[ok] < 1.0].sum()),
                "pct_rho_ge_1_given_diverged": pct(rho[ok & div] >= 1.0)
                if (ok & div).any()
                else None,
            }
            if acts == "all":
                d["identity_only_pct_diverged"] = pct(div[lin])
            em, lm, ev, lv = [], [], [], []
            for r, dv in zip(recs, div, strict=True):
                if not dv:
                    m = r["moments"]
                    em += m["em"]
                    lm += m["lm"]
                    ev += m["ev"]
                    lv += m["lv"]
            # float32 like diagnose() (np.concatenate of float32 numpy windows),
            # so the both-arm rule reproduces the published medians exactly.
            em, lm, ev, lv = (np.array(a, dtype=np.float32) for a in (em, lm, ev, lv))
            fin = np.isfinite(em) & np.isfinite(lm) & np.isfinite(ev) & np.isfinite(lv)
            em, lm, ev, lv = em[fin], lm[fin], ev[fin], lv[fin]
            ratio = lv / np.where(ev > 1e-12, ev, np.nan)
            ratio = ratio[np.isfinite(ratio)]
            drift = np.abs(lm - em) / np.sqrt(np.where(ev > 1e-12, ev, np.nan))
            drift = drift[np.isfinite(drift)]
            d["burn_in_moments"] = {
                "n_nondiverged_episodes": int((~div).sum()),
                "n_variables": len(em),
                "var_ratio_median": float(np.median(ratio)),
                "pct_var_ratio_within_2x": pct((ratio > 0.5) & (ratio < 2.0)),
                "std_drift_median": float(np.median(drift)),
                "pct_std_drift_lt_0p5": pct(drift < 0.5),
                "median_abs_mean_early": float(np.median(np.abs(em))),
                "median_abs_mean_late": float(np.median(np.abs(lm))),
            }
            res[rule] = d
        out[acts] = res
    return out


def mode_scaling(args) -> dict:
    """Recount the scaling_lag.py divergence rates under both rules.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Summary dict per (N_max, K_max).
    """
    out = {}
    for n_max, k_max in ((10, 3), (10, 8), (60, 8)):
        tasks = [
            {
                "key": f"s{n_max}_{k_max}_{i}",
                "seed": (20260928 * 1_000_003 + i) & 0x7FFFFFFF,
                "T": 200,
                "r_max": 3,
                "config": {"N_max": n_max, "K_max": k_max},
            }
            for i in range(args.n_scaling)
        ]
        print(f"scaling N_max={n_max} K_max={k_max}", flush=True)
        results = pool_map(tasks, args.workers)
        out[f"{n_max}_{k_max}"] = {
            "sequences": summarize_sequences(results, (0, 3)),
            "rows": [r["seq"] for r in results],
        }
        Path(args.out).write_text(json.dumps(out))  # checkpoint after each config
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["release", "verify", "testcfg", "stationarity", "scaling"])
    ap.add_argument("--build-retries", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--r-max", type=int, default=20)
    ap.add_argument("--scan-json")
    ap.add_argument(
        "--generic-dir", default=os.path.expanduser("~/.cache/dotime/dot-Generic-100k-1.0.0")
    )
    ap.add_argument(
        "--regime-dir", default=os.path.expanduser("~/.cache/dotime/dot-RegimeSwitch-v1-1.0.0")
    )
    ap.add_argument("--generic-seed", type=int, default=20264719)
    ap.add_argument("--regime-seed", type=int, default=20262719)
    ap.add_argument("--n-random", type=int, default=2000)
    ap.add_argument("--n-regime", type=int, default=200)
    ap.add_argument("--n-scaling", type=int, default=1000)
    ap.add_argument("--sample-seed", type=int, default=20260929)
    args = ap.parse_args()
    t0 = time.time()
    res = {
        "release": mode_release,
        "verify": mode_verify,
        "testcfg": mode_testcfg,
        "stationarity": mode_stationarity,
        "scaling": mode_scaling,
    }[args.mode](args)
    res["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out} in {res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
