#!/usr/bin/env python
"""Where does an intervention still act in dot-RegimeSwitch-v1 version 1.1.0?

The two arms of a regime-switching episode are independent draws, so the gap
between them mixes the effect of the intervention with noise and with two
different regime paths. This script measures the effect itself, which the
released pair cannot show. For every audited episode it rebuilds the SCM and the
intervention exactly as the build draws them and simulates the interventional
arm twice under common random numbers: the same torch generator state and the
same numpy seed, hence the same noise and the same regime path.

* A soft intervention still draws the treated variable's noise, so its
  reference run is the observational one.
* A hard or time-varying intervention skips that draw, so its reference run is
  a hard intervention on the same targets and steps at the pre-onset mean of
  the first target. Both runs then skip the same draws.

The absolute difference of the two runs is the causal contrast of the
intervention against its reference. The script reports the share of episodes in
which it reaches the effect threshold of the benchmark (0.1) at the last step
and at the last step of the intervention window, for any untreated variable and
for the variable that each query rule selects.

It reads the suite from the release config, so the audited episodes are the
ones ``scripts/build_release.py`` builds. Per-arm target statistics are logged
and asserted first (``dotime.qa.target_qa``), as the repository requires of
every generation check.

    python results/reference/regime_v1_1/scripts/effect_persistence.py \\
        --per-block 200 --workers 10 --out results/reference/regime_v1_1/design_pilot.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = ROOT / "scripts" / "release_config_regime_v1_1.yaml"
SUITE = "dot-RegimeSwitch-v1"
# Fixed seeds of the two common-random-number runs. Any pair works, because both
# runs of one episode use the same pair.
CRN_NUMPY_SEED, CRN_TORCH_SEED = 77, 78


def suite_config(path: Path) -> tuple[dict[str, Any], int]:
    """Read the suite config and its seed from a release config.

    Args:
        path: The release config.

    Returns:
        ``(cfg, suite_seed)`` for :data:`SUITE`.

    Raises:
        KeyError: If the config does not define the suite or its seed.
    """
    cfg = yaml.safe_load(path.read_text())["suites"][SUITE]
    return cfg, int(cfg["seed"])


def audit_specs(cfg: dict[str, Any], seed: int, per_block: int | None) -> list[dict[str, Any]]:
    """The specs to audit: the first episodes of every regime-density block.

    Args:
        cfg: The suite config.
        seed: The suite seed.
        per_block: Episodes per density block, or ``None`` for all of them.

    Returns:
        Specs in suite order.
    """
    from dotime._build import episode_specs

    specs = episode_specs(cfg, seed, 1.0)
    if per_block is None:
        return specs
    kept: dict[int, int] = {}
    out = []
    for spec in specs:
        if kept.get(spec["num_regimes"], 0) < per_block:
            kept[spec["num_regimes"]] = kept.get(spec["num_regimes"], 0) + 1
            out.append(spec)
    return out


def _draw_pair(spec: dict[str, Any]) -> tuple[Any, Any, Any, Any, int]:
    """Redraw the SCM, the intervention and both arms of a built episode.

    Mirrors the retry loop of ``dotime._build._make_configured_regime_episode``,
    which keeps the first attempt without a zeroed arm.

    Args:
        spec: A ``regime_configured`` spec.

    Returns:
        ``(x_obs, x_int, intervention, scm, attempt)`` of the kept attempt.
    """
    from dotime import DoTime
    from dotime._build import arms_zeroed

    seed, retries = spec["seed"], int(spec.get("stability_retries", 0))
    for attempt in range(retries + 1):
        s = seed if attempt == 0 else seed * 100003 + attempt
        torch.manual_seed(s)
        np.random.seed(s & 0xFFFFFFFF)
        prior = DoTime(config=spec.get("prior_config"), seed=s)
        x_obs, x_int, iv, scm = prior.generate_regime_pair(
            T=spec["T"], num_regimes=spec["num_regimes"]
        )
        if attempt == retries or not arms_zeroed(x_obs, x_int):
            break
    return x_obs, x_int, iv, scm, attempt


def audit_episode(spec: dict[str, Any]) -> dict[str, Any]:
    """Measure the causal contrast of one episode under common random numbers.

    Args:
        spec: A ``regime_configured`` spec of the suite.

    Returns:
        The episode's record: build facts (attempt, divergence, intervention
        type, window), the query variable under both query rules with its
        reachability in the recorded graph, and the contrasts.

    Raises:
        AssertionError: If the redrawn pair differs from the built episode, or
            if the two common-random-number runs differ before the onset.
    """
    from dotime._build import make_episode
    from dotime.interventions import InterventionType
    from dotime.utils import DEFAULT_CONFIG

    warnings.simplefilter("ignore", RuntimeWarning)
    torch.set_num_threads(1)
    built = make_episode(spec)
    # The same trajectories under the other query rule: only the query moves.
    last_rule = make_episode({k: v for k, v in spec.items() if k != "query_row"})
    x_obs, x_int, iv, scm, attempt = _draw_pair(spec)
    assert torch.equal(x_obs, built.x_obs), "redrawn observational arm differs from the build"
    assert torch.equal(x_int, built.x_int), "redrawn interventional arm differs from the build"
    assert torch.equal(last_rule.x_int, built.x_int)
    t_len, n = x_obs.shape
    onset, window_end = min(iv.times), max(iv.times)
    q_built, q_last = int(built.query_target[0]), int(last_rule.query_target[0])
    record: dict[str, Any] = {
        "idx": int(spec["idx"]),
        "num_regimes": int(spec["num_regimes"]),
        "attempt": attempt,
        "diverged": bool(built.metadata["diverged"]),
        "type": iv.intervention_type.name,
        "n_vars": int(n),
        "n_targets": len(iv.targets),
        "onset": int(onset),
        "window_end": int(window_end),
        "steps_after_window": int(t_len - 1 - window_end),
        "query_time": int(built.metadata["query_time_idx"][0]),
        "query_reachable": bool(built.metadata["graph"]["path"][0]["reachable"]),
        "query_min_lag": built.metadata["graph"]["path"][0]["min_lag"],
        "last_rule_query_reachable": bool(last_rule.metadata["graph"]["path"][0]["reachable"]),
        "reads_parents": bool(built.metadata["graph"]["reads_parents"]),
        "y_int": float(built.y_true.reshape(-1)[0]),
        "y_obs": float(built.x_obs[int(built.metadata["query_time_idx"][0]), q_built]),
    }
    if record["diverged"]:
        return record
    burn_in = DEFAULT_CONFIG["burn_in"]

    def run(intervention: Any) -> torch.Tensor:
        """Simulate one arm from the fixed common-random-number state."""
        np.random.seed(CRN_NUMPY_SEED)
        generator = torch.Generator().manual_seed(CRN_TORCH_SEED)
        if intervention is None:
            return scm.sample_observational(t_len, burn_in=burn_in, generator=generator)
        return scm.sample_interventional(t_len, intervention, burn_in=burn_in, generator=generator)

    treated = run(iv)
    if iv.intervention_type == InterventionType.SOFT:
        reference = run(None)
    else:
        level = float(treated[:onset, iv.targets[0]].mean())
        reference = run(
            dataclasses.replace(iv, intervention_type=InterventionType.HARD, values=level)
        )
    if float(treated.abs().max()) == 0.0 or float(reference.abs().max()) == 0.0:
        record["contrast"] = None  # a common-random-number run diverged
        return record
    assert torch.equal(treated[:onset], reference[:onset]), "runs differ before the onset"
    effect = (treated - reference).abs()
    others = [j for j in range(n) if j not in iv.targets]
    most_affected = max(others, key=lambda j: float(effect[window_end, j]))
    record["contrast"] = {
        "any_at_last_step": float(effect[t_len - 1, others].max()),
        "any_at_window_end": float(effect[window_end, others].max()),
        "query_at_last_step": float(effect[t_len - 1, q_last]),
        "query_at_window_end": float(effect[window_end, q_built]),
        "query_is_most_affected": bool(q_built == most_affected),
    }
    return record


def _share(values: list[float], threshold: float) -> float | None:
    """Share of values at or above a threshold.

    Args:
        values: The values.
        threshold: The threshold.

    Returns:
        The share, or ``None`` for an empty list.
    """
    return float(np.mean(np.asarray(values) >= threshold)) if values else None


def summarise(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Aggregate the per-episode records.

    Args:
        rows: Records of :func:`audit_episode`.
        threshold: Effect threshold of the benchmark.

    Returns:
        Counts and shares, pooled and per intervention type.
    """
    kept = [r for r in rows if not r["diverged"]]
    con = [r for r in kept if r.get("contrast")]

    def block(sub: list[dict[str, Any]]) -> dict[str, Any]:
        """Shares of one subset of contrast records."""
        out: dict[str, Any] = {"n": len(sub)}
        for key in (
            "any_at_last_step",
            "any_at_window_end",
            "query_at_last_step",
            "query_at_window_end",
        ):
            values = [r["contrast"][key] for r in sub]
            out[f"{key}_share"] = _share(values, threshold)
            out[f"{key}_median"] = float(np.median(values)) if values else None
        out["query_is_most_affected_share"] = (
            float(np.mean([r["contrast"]["query_is_most_affected"] for r in sub])) if sub else None
        )
        return out

    gap = [r["steps_after_window"] for r in kept]
    return {
        "n_episodes": len(rows),
        "n_resampled": sum(r["attempt"] > 0 for r in rows),
        "n_diverged": len(rows) - len(kept),
        "n_reads_parents": sum(r["reads_parents"] for r in rows),
        "intervention_types": {
            t: sum(r["type"] == t for r in kept) for t in ("HARD", "SOFT", "TIME_VARYING")
        },
        "steps_after_window_median": float(np.median(gap)),
        "window_closes_at_last_step_share": float(np.mean(np.asarray(gap) == 0)),
        "query_reachable_share": float(np.mean([r["query_reachable"] for r in kept])),
        "last_rule_query_reachable_share": float(
            np.mean([r["last_rule_query_reachable"] for r in kept])
        ),
        "n_contrast": len(con),
        "n_contrast_run_diverged": len(kept) - len(con),
        "threshold": threshold,
        "pooled": block(con),
        "by_type": {
            t: block([r for r in con if r["type"] == t]) for t in ("HARD", "SOFT", "TIME_VARYING")
        },
        "by_num_regimes": {
            str(d): block([r for r in con if r["num_regimes"] == d])
            for d in sorted({r["num_regimes"] for r in rows})
        },
    }


def assert_targets(specs: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    """Log and assert the per-arm target statistics of the audited episodes.

    Args:
        specs: The specs to build.
        workers: Worker processes.

    Returns:
        The QA report as a dict.

    Raises:
        dotime.qa.TargetQAError: If a target check fails.
    """
    from dotime._build import make_episode
    from dotime.qa import target_qa

    with Pool(workers) as pool:
        episodes = pool.map(make_episode, specs, chunksize=16)
    return target_qa(episodes, dir_target="level", group_by="structure").to_dict()


def main(argv: list[str] | None = None) -> None:
    """Run the audit and write its JSON.

    Args:
        argv: Command line, ``sys.argv[1:]`` by default.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument(
        "--per-block", type=int, default=None, help="Episodes per density, all if unset."
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--threshold", type=float, default=None, help="Default: evaluation.DIR_ACC_EPS."
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import dotime
    from dotime import evaluation

    threshold = float(evaluation.DIR_ACC_EPS if args.threshold is None else args.threshold)
    cfg, seed = suite_config(args.config)
    specs = audit_specs(cfg, seed, args.per_block)
    assert {s["kind"] for s in specs} == {"regime_configured"}, "the config is not a configured one"
    print(f"[effect_persistence] {len(specs)} episodes of {SUITE} {cfg['version']}, seed {seed}")
    qa = assert_targets(specs, args.workers)
    with Pool(args.workers) as pool:
        rows = pool.map(audit_episode, specs, chunksize=8)
    summary = summarise(rows, threshold)
    # Every audited episode must be a repaired one, or the audit says nothing.
    assert summary["n_reads_parents"] == len(rows), "an episode does not read its parents"
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()
    result = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
        "dotime_file": str(Path(dotime.__file__).resolve().relative_to(ROOT)),
        "suite": SUITE,
        "version": cfg["version"],
        "config": str(args.config.resolve().relative_to(ROOT)),
        "config_keys_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[
            :16
        ],
        "suite_seed": seed,
        "per_block": args.per_block,
        "reference_runs": {
            "SOFT": "observational run",
            "HARD": "hard intervention at the pre-onset mean of the first target",
            "TIME_VARYING": "hard intervention at the pre-onset mean of the first target",
            "numpy_seed": CRN_NUMPY_SEED,
            "torch_seed": CRN_TORCH_SEED,
        },
        "target_qa": {k: qa[k] for k in ("mode", "passed", "problems") if k in qa},
        "summary": summary,
        "episodes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    pooled = summary["pooled"]
    print(
        f"[effect_persistence] resampled {summary['n_resampled']}, diverged {summary['n_diverged']}, "
        f"contrasts {summary['n_contrast']}"
    )
    print(
        f"[effect_persistence] query moved by >= {threshold}: last step "
        f"{pooled['query_at_last_step_share']:.3f}, window end {pooled['query_at_window_end_share']:.3f}; "
        f"any variable: {pooled['any_at_last_step_share']:.3f} / {pooled['any_at_window_end_share']:.3f}; "
        f"query is the most affected variable {pooled['query_is_most_affected_share']:.3f}"
    )
    print(f"[effect_persistence] wrote {args.out}")


if __name__ == "__main__":
    main()
