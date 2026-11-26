#!/usr/bin/env python
"""How large do the trajectories of a regime-switching build get?

The repaired regime-switching SCMs are stabilised regime by regime
(``dotime.hardening``). That bounds each regime on its own and not a system
that switches between regimes, so a build without an amplitude limit holds a
few trajectories that grow far beyond the others without reaching the
divergence threshold of 500. This script measures that tail on a built suite:
the peak magnitude of every episode, the size of the targets, and how much of
the variance of the level target the largest targets carry.

    python results/reference/regime_v1_1/scripts/amplitude_tail.py \\
        --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \\
        --out results/reference/regime_v1_1/amplitude_tail.json

It was run on a build of the shared-noise recipe without ``amplitude_limit``
(the release config of commit bbccbf2). The limit of the release config is
chosen from its result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SUITE = "dot-RegimeSwitch-v1"
BOUNDS = (20, 50, 100, 200)
QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)


def tail_summary(episodes: list, n_largest: int = 10) -> dict[str, Any]:
    """Summarise the peaks of the trajectories and the size of the targets.

    Args:
        episodes: The stored episodes.
        n_largest: Number of largest targets to set aside for the variance comparison.

    Returns:
        Quantiles and counts above :data:`BOUNDS` for the per-episode peak and
        for the target, the variance of the target with and without its
        ``n_largest`` largest values, and the rows of the largest peaks.
    """
    peak = np.asarray(
        [max(float(ep.x_obs.abs().max()), float(ep.x_int.abs().max())) for ep in episodes]
    )
    target = np.asarray([float(ep.y_true.reshape(-1)[0]) for ep in episodes])
    size = np.abs(target)
    trimmed = np.delete(target, np.argsort(-size)[:n_largest])
    order = np.argsort(-peak)[:n_largest]
    return {
        "n_episodes": len(episodes),
        "quantile_levels": list(QUANTILES),
        "peak_quantiles": [float(q) for q in np.quantile(peak, QUANTILES)],
        "peak_above": {str(b): int((peak > b).sum()) for b in BOUNDS},
        "target_abs_quantiles": [float(q) for q in np.quantile(size, QUANTILES)],
        "target_abs_above": {str(b): int((size > b).sum()) for b in BOUNDS},
        "target_variance": float(target.var()),
        "target_variance_without_largest": float(trimmed.var()),
        "n_largest": n_largest,
        "largest_peaks": [
            {
                "row": int(i),
                "structure": episodes[i].structure,
                "n_vars": int(episodes[i].x_obs.shape[1]),
                "peak": float(peak[i]),
                "target": float(target[i]),
            }
            for i in order
        ],
    }


def main(argv: list[str] | None = None) -> None:
    """Measure the tail of a built suite and write the record.

    Args:
        argv: Command line, ``sys.argv[1:]`` by default.

    Raises:
        ValueError: If a shard of the build does not match its digest.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from dotime import _release_io, benchmarks

    suite_dir = args.suite_dir.resolve()
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    for shard in manifest["shards"]:
        digest = hashlib.md5((suite_dir / shard["file"]).read_bytes()).hexdigest()
        if digest != shard["md5"]:
            raise ValueError(f"{shard['file']} does not match its manifest digest")
    meta = dataclasses.replace(
        benchmarks._SUITE_REGISTRY[SUITE], version=manifest["version"], zenodo_record_id="LOCAL"
    )
    episodes = list(_release_io.read_suite(meta, suite_dir))
    summary = tail_summary(episodes)
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()
    result = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git("rev-parse", "HEAD"),
        "build": {
            "suite_dir": suite_dir.name,
            "version": manifest["version"],
            "config_hash": manifest["config_hash"],
            "pair_mode": manifest.get("pair_mode"),
            "amplitude_limit": manifest.get("amplitude_limit"),
            "shards_md5": [shard["md5"] for shard in manifest["shards"]],
        },
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        f"[amplitude_tail] {summary['n_episodes']} episodes, peaks above 50: "
        f"{summary['peak_above']['50']}, largest peak {summary['peak_quantiles'][-1]:.1f}, largest "
        f"target {summary['target_abs_quantiles'][-1]:.1f}, target variance "
        f"{summary['target_variance']:.3f} ({summary['target_variance_without_largest']:.3f} without "
        f"the {summary['n_largest']} largest)"
    )
    print(f"[amplitude_tail] wrote {args.out}")


if __name__ == "__main__":
    main()
