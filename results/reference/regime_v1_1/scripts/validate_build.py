#!/usr/bin/env python
"""Validate a build of dot-RegimeSwitch-v1 version 1.1.0 before it is released.

Seeds and manifests show that a build is reproducible. They do not show that
its targets mean what the suite claims, which is how version 1.0.0 shipped as
independent noise. This script therefore checks the written files against the
claims of the release config, one by one, and fails on the first that does not
hold.

1. Files: every shard matches its md5, and the manifest names this config
   (``config_hash``), the repaired prior, the shared-noise pairing, the query
   rule and a passed target QA.
2. Targets: per-arm statistics of the stored episodes, the effect included,
   logged and asserted (``dotime.qa.target_qa``).
3. Episodes: none has a zeroed arm or a value above the amplitude limit, all
   read their parents, and the two arms of every episode agree exactly before
   the onset. Every query sits at the last
   step of its intervention window on the untreated variable that the
   intervention moves most, and ``y_true`` is the interventional value there.
   A query that the recorded graph cannot reach from the treatment has an
   effect of exactly zero.
4. Regeneration: a random sample of stored episodes equals the episodes that
   the config and the seeds rebuild in this process, the recorded regime path
   included. The build used many workers.
5. Repair: the observational arms have lag structure, which white noise lacks,
   and a sampled episode keeps the regime graphs and the intervention of the
   version 1.0.0 episode with the same seed unless it was resampled.

    python results/reference/regime_v1_1/scripts/validate_build.py \\
        --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \\
        --out results/reference/regime_v1_1/build_validation.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "scripts" / "release_config_regime_v1_1.yaml"
V1_CONFIG = ROOT / "scripts" / "release_config.yaml"
SUITE = "dot-RegimeSwitch-v1"
# White noise at T=200 has sd(acf1) of about 0.07, so 0.3 is more than 4 sd.
ACF_LEVEL = 0.3
# Share of variables above ACF_LEVEL that a repaired suite must exceed. The
# measurement of the repair with the hardening found 0.31 on 600 episodes.
ACF_SHARE_FLOOR = 0.15


def abs_acf1(x: torch.Tensor) -> np.ndarray:
    """Per-variable ``|corr(x[t], x[t-1])|`` of a ``(T, N)`` trajectory.

    Args:
        x: The trajectory.

    Returns:
        Array of shape ``(N,)``. A constant variable gives ``nan``.
    """
    a = x.double().numpy()
    a = a - a.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs((a[1:] * a[:-1]).sum(0) / (a * a).sum(0))


def check_files(suite_dir: Path, config_text: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Check the shards and the manifest against the release config.

    Args:
        suite_dir: The built suite directory.
        config_text: Text of the release config.
        cfg: The suite's entry in it.

    Returns:
        The manifest.

    Raises:
        AssertionError: If a shard or a manifest entry does not match.
    """
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    for shard in manifest["shards"]:
        digest = hashlib.md5((suite_dir / shard["file"]).read_bytes()).hexdigest()
        assert digest == shard["md5"], f"{shard['file']} does not match its md5"
    assert manifest["name"] == SUITE
    assert manifest["version"] == cfg["version"]
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:16]
    assert manifest["config_hash"] == config_hash, "the build is not from this config"
    assert manifest["seed"] == cfg["seed"]
    assert cfg["pair_mode"] == "counterfactual", "this validation is for shared-noise arms"
    for key in (
        "prior_config",
        "pair_mode",
        "query_row",
        "record_graph",
        "stability_retries",
        "amplitude_limit",
    ):
        assert manifest[key] == cfg[key], f"manifest {key} differs from the config"
    assert manifest["target_qa"]["passed"] is True, "the build's target QA did not pass"
    return manifest


def check_episodes(episodes: list, cfg: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Check the claims that hold for every stored episode.

    Args:
        episodes: The stored episodes in suite order.
        cfg: The suite config.
        threshold: Effect threshold of the benchmark.

    Returns:
        Counts and summary statistics of the suite.

    Raises:
        AssertionError: If an episode breaks a claim.
    """
    densities = {int(k): int(v) for k, v in cfg["densities"].items()}
    per = cfg["n_episodes"] // len(densities)
    assert len(episodes) == per * len(densities)
    labels = Counter(ep.structure for ep in episodes)
    assert labels == {f"regime_{d}": per for d in densities}, labels
    types: Counter = Counter()
    lags: Counter = Counter()
    n_vars: Counter = Counter()
    gaps, effects, switches, peaks = [], [], [], []
    limit = float(cfg["amplitude_limit"])
    effect_by_type: dict[str, list[float]] = {}
    reachable = unreachable_with_effect = 0
    acf_above = acf_total = 0
    for row, ep in enumerate(episodes):
        d = int(ep.structure.split("_")[1])
        meta = ep.metadata
        assert meta["n_regimes"] == d, row
        assert meta["tier"] == densities[d], row
        assert meta["pair_mode"] == "counterfactual", row
        assert meta["diverged"] is False, f"episode {row} is flagged as diverged"
        assert float(ep.x_obs.abs().max()) > 0, f"episode {row} has a zeroed observational arm"
        assert float(ep.x_int.abs().max()) > 0, f"episode {row} has a zeroed interventional arm"
        assert bool(torch.isfinite(ep.x_obs).all()), row
        assert bool(torch.isfinite(ep.x_int).all()), row
        assert ep.x_obs.shape == ep.x_int.shape, row
        assert ep.x_obs.shape[0] == cfg["T"], row
        peak = max(float(ep.x_obs.abs().max()), float(ep.x_int.abs().max()))
        assert peak <= limit, f"episode {row} peaks at {peak:.1f}, above the limit {limit}"
        assert meta["amplitude_limit"] == limit, row
        assert meta["over_amplitude_limit"] is False, row
        peaks.append(peak)
        assert meta["graph"]["reads_parents"] is True, f"episode {row} does not read its parents"
        onset, end = min(ep.intervention.times), max(ep.intervention.times)
        assert torch.equal(ep.x_obs[:onset], ep.x_int[:onset]), (
            f"the arms of episode {row} differ before the onset"
        )
        path = np.asarray(meta["graph"]["regime_path"])
        assert path.shape == (cfg["T"],), row
        assert set(path.tolist()) <= set(range(d)), row
        assert meta["query_row"] == "window_end", row
        assert meta["query_time_idx"] == [end], row
        q = int(ep.query_target[0])
        assert q not in ep.intervention.targets, row
        assert float(ep.y_true[0]) == float(ep.x_int[end, q]), row
        # With shared noise the gap between the arms is the effect, so the
        # query must be the untreated variable with the largest gap.
        gap = (ep.x_int[end] - ep.x_obs[end]).abs()
        gap[list(ep.intervention.targets)] = -1.0
        assert q == int(gap.argmax()), f"episode {row} does not query its most affected variable"
        effect = float(gap[q])
        graph_path = meta["graph"]["path"][0]
        if graph_path["reachable"]:
            reachable += 1
        else:
            # The recorded graph, the simulator and the pairing have to agree:
            # what the treatment cannot reach, it cannot move.
            unreachable_with_effect += effect != 0.0
        lags[
            "unreachable" if graph_path["min_lag"] is None else str(min(graph_path["min_lag"], 4))
        ] += 1
        kind = ep.intervention.intervention_type.name
        types[kind] += 1
        effect_by_type.setdefault(kind, []).append(effect)
        effects.append(effect)
        n_vars[int(ep.x_obs.shape[1])] += 1
        gaps.append(cfg["T"] - 1 - end)
        switches.append(int((np.diff(path) != 0).sum()))
        acf = abs_acf1(ep.x_obs)
        acf_above += int(np.nansum(acf > ACF_LEVEL))
        acf_total += int(np.isfinite(acf).sum())
    assert unreachable_with_effect == 0, (
        f"{unreachable_with_effect} queries move although the graph cannot reach them"
    )
    share = acf_above / acf_total
    assert share > ACF_SHARE_FLOOR, f"only {share:.3f} of the variables have lag structure"
    values = np.asarray(effects)
    targets = np.asarray([float(ep.y_true[0]) for ep in episodes])
    return {
        "n_episodes": len(episodes),
        "amplitude_limit": limit,
        "peak_max": float(np.max(peaks)),
        "peak_quantile_999": float(np.quantile(peaks, 0.999)),
        "target_abs_max": float(np.abs(targets).max()),
        "target_variance": float(targets.var()),
        "target_variance_without_10_largest": float(
            np.delete(targets, np.argsort(-np.abs(targets))[:10]).var()
        ),
        "per_density": dict(labels),
        "intervention_types": dict(types),
        "n_vars": {str(k): v for k, v in sorted(n_vars.items())},
        "n_arms_equal_before_onset": len(episodes),
        "query_reachable_share": reachable / len(episodes),
        "query_min_lag": {k: lags[k] for k in sorted(lags)},
        "steps_after_window_median": float(np.median(gaps)),
        "regime_switches_median": float(np.median(switches)),
        "no_regime_switch_share": float(np.mean(np.asarray(switches) == 0)),
        "effect_threshold": threshold,
        "effect_share_at_or_above_threshold": float(np.mean(values >= threshold)),
        "effect_median": float(np.median(values)),
        "effect_zero_share": float(np.mean(values == 0.0)),
        "effect_share_by_type": {
            k: float(np.mean(np.asarray(v) >= threshold)) for k, v in sorted(effect_by_type.items())
        },
        "acf1_above_level_share": share,
        "acf1_level": ACF_LEVEL,
    }


def check_regeneration(
    episodes: list, cfg: dict[str, Any], n_sample: int, seed: int
) -> dict[str, Any]:
    """Rebuild a random sample of episodes and compare them with the stored ones.

    Also rebuilds each sampled episode with the version 1.0.0 recipe and checks
    that the repair kept its regime graphs and its intervention, which holds
    whenever the stored episode is the first attempt of its seed.

    Args:
        episodes: The stored episodes in suite order.
        cfg: The suite config.
        n_sample: Number of episodes to rebuild.
        seed: Seed of the sample.

    Returns:
        The sampled rows and the counts of the comparison with version 1.0.0.

    Raises:
        AssertionError: If a rebuilt episode differs from the stored one, or if
            a first-attempt episode differs from its version 1.0.0 draw.
    """
    from dotime._build import episode_specs, make_episode

    specs = episode_specs(cfg, int(cfg["seed"]), 1.0)
    old_cfg = yaml.safe_load(V1_CONFIG.read_text())["suites"][SUITE]
    old_specs = episode_specs({**old_cfg, "record_graph": True}, int(cfg["seed"]), 1.0)
    rows = sorted(np.random.default_rng(seed).choice(len(specs), n_sample, replace=False).tolist())
    same_as_v1 = resampled = 0
    for row in rows:
        stored = episodes[row]
        rebuilt = make_episode(specs[row])
        assert torch.equal(stored.x_obs, rebuilt.x_obs.to(stored.x_obs.dtype)), row
        assert torch.equal(stored.x_int, rebuilt.x_int.to(stored.x_int.dtype)), row
        assert torch.equal(stored.y_true, rebuilt.y_true.to(stored.y_true.dtype)), row
        assert int(stored.query_target[0]) == int(rebuilt.query_target[0]), row
        assert (
            stored.metadata["graph"]["regime_path"] == rebuilt.metadata["graph"]["regime_path"]
        ), row
        # First attempt or resample? Rebuild without retries and compare.
        first = make_episode({**specs[row], "stability_retries": 0})
        if not torch.equal(first.x_int, rebuilt.x_int):
            resampled += 1
            continue
        old = make_episode(old_specs[row])
        same_iv = json.dumps(old.intervention.to_dict(), sort_keys=True) == json.dumps(
            rebuilt.intervention.to_dict(), sort_keys=True
        )
        same_graph = (
            old.metadata["graph"]["regime_edges"] == rebuilt.metadata["graph"]["regime_edges"]
        )
        assert same_iv, f"episode {row} has another intervention than its version 1.0.0 draw"
        assert same_graph, f"episode {row} has other regime graphs than its version 1.0.0 draw"
        assert old.metadata["graph"]["reads_parents"] is False
        same_as_v1 += 1
    return {
        "n_sample": n_sample,
        "sample_seed": seed,
        "rows": rows,
        "n_equal_to_stored": len(rows),
        "n_first_attempt_with_v1_graph_and_intervention": same_as_v1,
        "n_resampled_in_sample": resampled,
    }


def main(argv: list[str] | None = None) -> None:
    """Run every check and write the validation record.

    Args:
        argv: Command line, ``sys.argv[1:]`` by default.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-regenerate", type=int, default=300)
    ap.add_argument("--sample-seed", type=int, default=20261127)
    args = ap.parse_args(argv)

    import dotime
    from dotime import _release_io, benchmarks, evaluation
    from dotime.qa import target_qa

    warnings.simplefilter("ignore", RuntimeWarning)
    torch.set_num_threads(1)
    config_text = CONFIG.read_text()
    cfg = yaml.safe_load(config_text)["suites"][SUITE]
    suite_dir = args.suite_dir.resolve()
    manifest = check_files(suite_dir, config_text, cfg)
    print(f"[validate] files and manifest of {suite_dir.name}: ok")
    meta = dataclasses.replace(
        benchmarks._SUITE_REGISTRY[SUITE], version=cfg["version"], zenodo_record_id="LOCAL"
    )
    episodes = list(_release_io.read_suite(meta, suite_dir))
    noise = evaluation.check_shared_noise(episodes)
    assert noise.n_shared == noise.n_checked == len(episodes), noise
    qa = target_qa(episodes, dir_target="effect", group_by="structure").to_dict()
    summary = check_episodes(episodes, cfg, float(evaluation.DIR_ACC_EPS))
    print(
        f"[validate] {summary['n_episodes']} episodes: no zeroed arm, all read their parents, arms "
        f"equal before the onset, most affected variable queried at the window's end"
    )
    print(
        f"[validate] largest value {summary['peak_max']:.1f} (limit {summary['amplitude_limit']:.0f}), "
        f"largest target {summary['target_abs_max']:.1f}, target variance "
        f"{summary['target_variance']:.3f} ({summary['target_variance_without_10_largest']:.3f} "
        f"without the 10 largest)"
    )
    print(
        f"[validate] effect >= {summary['effect_threshold']} in "
        f"{summary['effect_share_at_or_above_threshold']:.3f} of the queries (median "
        f"{summary['effect_median']:.3f}, exactly zero in {summary['effect_zero_share']:.3f}); "
        f"acf1 > {ACF_LEVEL} in {summary['acf1_above_level_share']:.3f} of the variables"
    )
    regeneration = check_regeneration(episodes, cfg, args.n_regenerate, args.sample_seed)
    print(
        f"[validate] {regeneration['n_equal_to_stored']} sampled episodes regenerate exactly, "
        f"{regeneration['n_resampled_in_sample']} of them were resampled"
    )
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
        "build": {
            "suite_dir": suite_dir.name,
            "config_hash": manifest["config_hash"],
            "package_version": manifest["package_version"],
            "seed": manifest["seed"],
            "pair_mode": manifest["pair_mode"],
            "shards": [{"file": s["file"], "md5": s["md5"]} for s in manifest["shards"]],
            "target_qa_passed": manifest["target_qa"]["passed"],
        },
        "target_qa": {k: qa[k] for k in ("mode", "passed", "problems") if k in qa},
        "episodes": summary,
        "regeneration": regeneration,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"[validate] all checks passed, wrote {args.out}")


if __name__ == "__main__":
    main()
