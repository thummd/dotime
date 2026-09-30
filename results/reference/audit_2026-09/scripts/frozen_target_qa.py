"""Per-arm target QA of every cached frozen suite version with the default thresholds.

Loads each released suite version from the md5-verified cache and runs
``dotime.qa.target_qa`` pooled and per structure (regime density for
RegimeSwitch) with ``dir_target="effect"``, so the observational level, the
interventional level and, where the structure can carry one, the effect are
all asserted. Identifiability 1.0.0 archives ``x_obs`` in topological column
order, so its observational levels come from the realignment sidecar, whose
rows are first checked against every episode.

If a version fails, the script prints the report and exits with status 1
without writing anything: the thresholds are not loosened to make a suite pass.

    python results/reference/audit_2026-09/scripts/frozen_target_qa.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotime import __version__
from dotime.benchmarks import load_benchmark
from dotime.qa import QAThresholds, target_qa
from dotime.reference._realignment import load_realignment, realign_episodes, sidecar_obs_levels

_ROOT = Path(__file__).resolve().parents[4]
_OUT = _ROOT / "results" / "reference" / "audit_2026-09" / "frozen_target_qa.json"
_SIDECAR = _ROOT / "results" / "reference" / "dot-Identifiability-v1.0.0_realignment.jsonl"
VERSIONS = (
    ("dot-Identifiability-v1", "1.0.0"),
    ("dot-Identifiability-v1", "1.1.0"),
    ("dot-RegimeSwitch-v1", "1.0.0"),
    ("dot-Continuous-v1", "1.0.0"),
    ("dot-Generic-100k", "1.0.0"),
)


def check_version(name: str, version: str) -> dict:
    """Run target QA on one cached suite version.

    Args:
        name: Suite name.
        version: Released version, loaded from the cache without a download.

    Returns:
        Dict with ``n_episodes``, ``y_obs_source``, ``seconds`` and the
        ``report`` (:meth:`dotime.qa.QAReport.to_dict`).

    Raises:
        ValueError: If the Identifiability 1.0.0 sidecar does not match its
            episodes.
    """
    t0 = time.time()
    episodes = list(load_benchmark(name, version=version))
    obs_levels, source = None, "dotime.evaluation.query_obs_levels"
    if (name, version) == ("dot-Identifiability-v1", "1.0.0"):
        sidecar = load_realignment(_SIDECAR)
        # Checks every row against its episode (variable count, query target,
        # y_true) before trusting the row's observational level.
        realign_episodes(episodes, sidecar)
        obs_levels, source = (
            sidecar_obs_levels(episodes, sidecar),
            f"{_SIDECAR.name} y_obs_corrected",
        )
    print(f"== {name} {version}: {len(episodes)} episodes, y_obs from {source}", flush=True)
    report = target_qa(episodes, obs_levels=obs_levels, dir_target="effect", raise_on_failure=False)
    return {
        "n_episodes": len(episodes),
        "y_obs_source": source,
        "seconds": round(time.time() - t0, 1),
        "report": report.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    """Check every version and write the JSON only if all of them pass.

    Args:
        argv: Command-line arguments, ``sys.argv`` by default.

    Returns:
        0 when every version passes, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_OUT)
    args = ap.parse_args(argv)
    results = {}
    for name, version in VERSIONS:
        results[f"{name}-{version}"] = check_version(name, version)
        gc.collect()  # one suite in memory at a time (Generic is 100k episodes)
    failed = [k for k, v in results.items() if not v["report"]["passed"]]
    if failed:
        for key in failed:
            print(f"FAILED {key}:", *results[key]["report"]["problems"], sep="\n  ")
        print("nothing written: fix the suite or report it, do not loosen the thresholds")
        return 1
    payload = {
        "generated_by": "results/reference/audit_2026-09/scripts/frozen_target_qa.py",
        "package_version": __version__,
        "dir_target": "effect",
        "thresholds": asdict(QAThresholds()),
        "all_passed": True,
        "versions": results,
    }
    args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"all {len(results)} versions pass; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
