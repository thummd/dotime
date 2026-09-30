#!/usr/bin/env python
"""Target QA and baseline measurements for dot-SeasonalTrend-v1.

Builds the first ``--per-label`` episodes of every label of
``scripts/release_config_seasonal_trend.yaml`` exactly as the release build
would: the suite's own per-episode specs and seed, through
``dotime._build.make_episode``, in worker processes with one torch thread each.
Per label it logs, asserts where the repository rule asks for it, and writes to
JSON:

- per-arm target statistics (nonzero fraction, mean and variance) of ``y_obs``
  (the observational level at the query), ``y_int`` (``y_true``) and
  ``effect = y_int - y_obs``, asserted before any other number is computed;
- the diverged share (``metadata["diverged"]``);
- the driver's variance share in A and Y, ``var(x - x_twin) / var(x)`` over the
  released observational arm. ``x_twin`` is the same episode rebuilt at driver
  strength 0, which reproduces the base structure bit for bit, so the share
  isolates what D adds. The rebuilt driven pair is asserted equal to the
  released episode;
- effect-sign accuracy of the contrast ``pred(do A=v) - pred(do A=a_obs(t_q))``
  against ``y_true - y_obs`` over episodes with ``|effect| >= 0.1``, for
  BackDoorOLS, for NaiveOLS when it is registered, and for the same OLS design
  with three adjustment sets: none, the base structure's back-door set (which
  ignores D), and that set plus D. On hidden labels D is rebuilt from
  ``metadata["driver"]``, which makes the last one an oracle;
- level RMSE of Mean, AR1, VAR-OLS and BackDoorOLS against ``y_true``.

    PYTHONPATH=src python results/reference/seasonal_trend/qa.py --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "scripts" / "release_config_seasonal_trend.yaml"
OUT = Path(__file__).with_name("seasonal_trend_qa.json")
LEVEL_BASELINES = ("Mean", "AR1", "VAR-OLS", "BackDoorOLS")
OLS_VARIANTS = ("ols_unadjusted", "ols_base_set", "ols_with_D")
# Same threshold as evaluation.direction_accuracy: smaller effects have no reliable sign.
EFFECT_EPS = 0.1
# ExtendedDoTime.generate_sample makes up to 20 attempts per episode.
MAX_ATTEMPTS = 20


def _accepted(x_obs: torch.Tensor, x_int: torch.Tensor) -> bool:
    """ExtendedDoTime.generate_sample's acceptance rule for one generated pair.

    Args:
        x_obs: Observational arm.
        x_int: Interventional arm.

    Returns:
        Whether generate_sample keeps the pair instead of drawing another.
    """
    return bool(
        not torch.isnan(x_obs).any()
        and not torch.isnan(x_int).any()
        and x_obs.abs().max() < 10
        and x_int.abs().max() < 10
    )


def _twin_pairs(spec: dict) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Rebuild a driven episode's observational arm and its strength-0 twin.

    Mirrors make_episode's stability retries and generate_sample's attempts,
    with a driven and a strength-0 TSCMPrior stepped in lockstep. Strength 0
    draws exactly what the driven prior draws, so the twin is the released
    episode with the driver switched off.

    Args:
        spec: A make_episode spec of a driven label.

    Returns:
        ``(x_obs, x_obs_twin, canonical_perm)``: both observational arms in topo
        order and the permutation from canonical columns to topo indices.

    Raises:
        ValueError: If the spec's label has no driver.
    """
    from dotime._build import arms_zeroed, identifiability_retry_seed
    from dotime.drivers import parse_structure_label
    from dotime.extended import TSCMPrior
    from dotime.tscm_sampler import TSCMStructure

    base, driver = parse_structure_label(spec["structure"])
    if driver is None:
        raise ValueError(f"{spec['structure']!r} has no driver")
    retries = int(spec.get("stability_retries", 0))
    for attempt in range(retries + 1):
        seed = identifiability_retry_seed(spec["seed"], attempt)
        torch.manual_seed(seed)
        priors = [
            TSCMPrior(TSCMStructure(base), seed=seed, pair_mode="counterfactual", driver=d)
            for d in (driver, dataclasses.replace(driver, strength=0.0))
        ]
        for _ in range(MAX_ATTEMPTS):
            (x_obs, x_int, _, _), (twin_obs, _, _, _) = (
                p.generate_pair(T=spec["T"]) for p in priors
            )
            if _accepted(x_obs, x_int):
                break
        if attempt == retries or not arms_zeroed(x_obs, x_int):
            break
    return x_obs, twin_obs, priors[0].canonical_perm


def _share(x: torch.Tensor, twin: torch.Tensor) -> float:
    """Fraction of a series' variance that the driver adds.

    Args:
        x: The driven series.
        twin: The same series with the driver switched off.

    Returns:
        ``var(x - twin) / var(x)``, NaN for a constant ``x``.
    """
    x64, t64 = x.double(), twin.double()
    total = float(x64.var())
    return float((x64 - t64).var()) / total if total > 0 else math.nan


def _ols_contrast(x: np.ndarray, adj: list[int], onset: int, v: float, a_obs: float) -> float:
    """Effect contrast of BackDoorOLS's design, ``Y_t ~ A_t + adj_t + Y_(t-1)``.

    Columns are canonical: the treatment is column 0 and the outcome the last.
    The fit uses the pre-onset rows, as BackDoorOLS does.

    Args:
        x: Observational trajectory ``(T, N)``.
        adj: Adjustment columns.
        onset: Intervention onset.
        v: The do-value.
        a_obs: Observational treatment at the query.

    Returns:
        ``coef_A * (v - a_obs)``, or NaN when fewer than 4 pre-onset rows exist.
    """
    from dotime.baselines import _ols_fit

    fit_end = max(2, min(onset, x.shape[0]))
    if fit_end < 4:
        return math.nan
    y = x.shape[1] - 1
    covariates = x[1:fit_end, adj] if adj else np.empty((fit_end - 1, 0))
    design = np.column_stack([x[1:fit_end, 0], covariates, x[0 : fit_end - 1, y]])
    return float(_ols_fit(design, x[1:fit_end, y])[1] * (v - a_obs))


def one(spec: dict) -> dict:
    """Build one release episode and measure it (runs in a worker).

    Args:
        spec: A make_episode spec from the release config.

    Returns:
        The per-episode record: targets, divergence, driver shares, contrasts
        and level predictions.

    Raises:
        AssertionError: If the rebuilt twin does not reproduce the released
            episode, or BackDoorOLS disagrees with its own design on an
            observed-driver label.
    """
    from dotime import baselines
    from dotime._build import make_episode
    from dotime.drivers import parse_structure_label, released_driver_series
    from dotime.interventions import InterventionSpec, InterventionType

    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    ep = make_episode(spec)
    label = spec["structure"]
    base, driver = parse_structure_label(label)
    onset = min(ep.intervention.times)
    t_q = int(ep.query_time_idx[0])
    y_col = int(ep.query_target[0])
    y_int = float(ep.y_true[0])
    y_obs = float(ep.x_obs[t_q, y_col])
    effect = float(ep.metadata["y_causal_effect"][0])
    assert abs(effect - (y_int - y_obs)) < 1e-5, (label, effect, y_int - y_obs)
    rec: dict = {
        "label": label,
        "diverged": bool(ep.metadata["diverged"]),
        "y_obs": y_obs,
        "y_int": y_int,
        "effect": effect,
    }

    v = float(ep.intervention.values)
    a_obs = float(ep.x_obs[t_q, 0])
    at_a_obs = dataclasses.replace(
        ep,
        intervention=InterventionSpec(
            targets=list(ep.intervention.targets),
            times=list(ep.intervention.times),
            intervention_type=InterventionType.HARD,
            values=a_obs,
        ),
    )
    rec["level"] = {name: float(baselines.get(name).predict(ep)[0]) for name in LEVEL_BASELINES}
    contrast_models = ["BackDoorOLS"] + (
        ["NaiveOLS"] if "NaiveOLS" in baselines.available() else []
    )
    rec["contrast"] = {}
    for name in contrast_models:
        model = baselines.get(name)
        rec["contrast"][name] = float(model.predict(ep)[0] - model.predict(at_a_obs)[0])

    names, _, hidden = baselines._canonical_summary_graph(label)
    base_names = baselines._canonical_summary_graph(base)[0]
    base_set = [names.index(base_names[c]) for c in baselines._back_door_columns(base)[3]]
    x = ep.x_obs.detach().cpu().numpy()
    rec["contrast"]["ols_unadjusted"] = _ols_contrast(x, [], onset, v, a_obs)
    rec["contrast"]["ols_base_set"] = _ols_contrast(x, base_set, onset, v, a_obs)
    rec["share"] = None
    if driver is None:
        return rec

    d_col = names.index("D")
    x_with_d = x.copy()
    # A hidden D is zeroed on release; the oracle reads it back from the metadata.
    x_with_d[:, d_col] = released_driver_series(ep.metadata["driver"], ep.length).numpy()
    rec["contrast"]["ols_with_D"] = _ols_contrast(x_with_d, [*base_set, d_col], onset, v, a_obs)
    if driver.observed:
        # BackDoorOLS returns float32 levels (|level| < 10), so its contrast carries
        # up to ~2e-6 of rounding that the float64 coefficient does not.
        assert math.isclose(
            rec["contrast"]["BackDoorOLS"], rec["contrast"]["ols_with_D"], rel_tol=0.0, abs_tol=1e-5
        ), (label, rec["contrast"])

    x_obs, twin_obs, perm = _twin_pairs(spec)
    for col, name in enumerate(names):
        released = ep.x_obs[:, col]
        if name in hidden:
            assert not released.any(), (label, name)
            continue
        assert torch.equal(released, x_obs[:, perm[col]]), (label, name, spec["idx"])
    rec["share"] = {
        v_name: _share(x_obs[:, perm[names.index(v_name)]], twin_obs[:, perm[names.index(v_name)]])
        for v_name in ("A", "Y")
    }
    return rec


def _target_stats(values: np.ndarray) -> dict:
    """Nonzero fraction, mean and variance of one arm's targets.

    Args:
        values: Per-episode target values.

    Returns:
        ``{"nonzero_frac", "mean", "var", "n"}``.
    """
    return {
        "nonzero_frac": float(np.mean(values != 0.0)),
        "mean": float(np.mean(values)),
        "var": float(np.var(values)),
        "n": int(values.size),
    }


def _sign_accuracy(contrast: np.ndarray, effect: np.ndarray) -> dict:
    """Effect-sign accuracy with a bootstrap 95% interval.

    A zero or NaN contrast counts as wrong, since it names no direction.

    Args:
        contrast: Predicted effect contrasts.
        effect: True effects ``y_true - y_obs``.

    Returns:
        ``{"accuracy", "ci95", "n_valid", "n_zero_contrast"}`` over episodes
        with ``|effect| >= 0.1``.
    """
    from dotime.evaluation import bootstrap_ci

    valid = np.abs(effect) >= EFFECT_EPS
    hits = (np.sign(np.nan_to_num(contrast[valid])) == np.sign(effect[valid])).astype(float)
    mean, _, low, high = bootstrap_ci(hits.tolist())
    return {
        "accuracy": mean,
        "ci95": [low, high],
        "n_valid": int(valid.sum()),
        "n_zero_contrast": int(np.sum(np.nan_to_num(contrast[valid]) == 0.0)),
    }


def _quartiles(values: list[float]) -> dict:
    """Median and interquartile range.

    Args:
        values: Per-episode values.

    Returns:
        ``{"median", "q25", "q75"}``.
    """
    q25, median, q75 = np.nanquantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    return {"median": float(median), "q25": float(q25), "q75": float(q75)}


def summarise(records: list[dict], label: str, tier: int) -> dict:
    """Aggregate and assert one label's episodes.

    Args:
        records: The label's per-episode records.
        label: The structure label.
        tier: The label's tier in the config.

    Returns:
        The label's summary.

    Raises:
        AssertionError: If a target arm is less than half nonzero, has a
            non-finite or zero-variance level, or no effect is nonzero.
    """
    arms = {key: np.array([r[key] for r in records]) for key in ("y_obs", "y_int", "effect")}
    targets = {key: _target_stats(values) for key, values in arms.items()}
    for key in ("y_obs", "y_int"):
        stats = targets[key]
        assert stats["nonzero_frac"] >= 0.5, (label, key, stats)
        assert math.isfinite(stats["mean"]), (label, key, stats)
        assert stats["var"] > 0.0, (label, key, stats)
    assert targets["effect"]["nonzero_frac"] > 0.0, (label, targets["effect"])
    out: dict = {
        "tier": tier,
        "n": len(records),
        "diverged_share": float(np.mean([r["diverged"] for r in records])),
        "targets": targets,
    }
    shares = [r["share"] for r in records if r["share"] is not None]
    if shares:
        out["driver_variance_share"] = {v: _quartiles([s[v] for s in shares]) for v in ("A", "Y")}
    contrasts = sorted({k for r in records for k in r["contrast"]})
    out["effect_sign_accuracy"] = {
        name: _sign_accuracy(np.array([r["contrast"][name] for r in records]), arms["effect"])
        for name in contrasts
    }
    out["level_rmse"] = {
        name: float(
            np.sqrt(np.mean((np.array([r["level"][name] for r in records]) - arms["y_int"]) ** 2))
        )
        for name in LEVEL_BASELINES
    }
    return out


def main(argv: list[str] | None = None) -> int:
    """Build the QA episodes, summarise them per label and write the JSON.

    Args:
        argv: Command-line arguments, ``None`` for ``sys.argv``.

    Returns:
        The process exit code.

    Raises:
        AssertionError: If a target or reproduction check fails.
    """
    from dotime import __version__
    from dotime._build import episode_specs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    config_text = CONFIG.read_text()
    config = yaml.safe_load(config_text)
    (suite_name, suite), *rest = config["suites"].items()
    assert not rest, "the seasonal/trend config holds one suite"
    suite_seed = int(suite["seed"])
    # The release build's own specs, so these are the first episodes of each label.
    specs = [
        s
        for s in episode_specs(suite, suite_seed, 1.0)
        if s["idx"] % int(suite["episodes_per_structure"]) < args.per_label
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, specs, chunksize=4))

    labels = {}
    for label, tier in suite["structures"].items():
        labels[label] = summarise([r for r in records if r["label"] == label], label, tier)
        s = labels[label]
        share = s.get("driver_variance_share")
        print(
            f"{label:30s} diverged={s['diverged_share']:.3f} "
            f"y_int nz={s['targets']['y_int']['nonzero_frac']:.2f} "
            f"var={s['targets']['y_int']['var']:.3f} effect var={s['targets']['effect']['var']:.3f}"
            + (f" share A={share['A']['median']:.2f} Y={share['Y']['median']:.2f}" if share else "")
        )
        print(
            "    sign acc "
            + " ".join(f"{k}={v['accuracy']:.2f}" for k, v in s["effect_sign_accuracy"].items())
            + " | rmse "
            + " ".join(f"{k}={v:.3f}" for k, v in s["level_rmse"].items())
        )

    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    # The numbers depend on the generator and the config, not on this results
    # directory, so a clean src/ and scripts/ pins them to git_commit.
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "scripts"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    out = {
        "config": {
            "suite": suite_name,
            "version": suite["version"],
            "config_file": str(CONFIG.relative_to(REPO)),
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "suite_seed": suite_seed,
            "T": int(suite["T"]),
            "per_label": args.per_label,
            "episodes": len(records),
            "workers": args.workers,
            "effect_eps": EFFECT_EPS,
            "package_version": __version__,
            "git_commit": commit,
            "src_and_scripts_clean": not dirty,
            "naive_ols_registered": any("NaiveOLS" in r["contrast"] for r in records),
        },
        "labels": labels,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
