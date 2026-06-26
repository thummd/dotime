#!/usr/bin/env python
"""Reproducible build of the frozen DoTime benchmark suites.

This is the reproducibility anchor cited in the paper: a single committed,
config-driven script that regenerates every released suite with fixed seeds and
records the package version + hardware in a self-describing, timestamped output
directory.

Usage
-----
    python scripts/build_release.py                       # all suites, full scale
    python scripts/build_release.py --suite dot-Generic-100k
    python scripts/build_release.py --scale 0.001         # tiny smoke build
    python scripts/build_release.py --output-dir output

Each suite is written under ``<output-dir>/<timestamp>/<suite>-<version>/`` as
parquet shards + ``manifest.json`` (the canonical schema from
``dotime._release_io``), plus a per-suite Croissant ``croissant.json``
and a top-level ``build_manifest.json`` recording the config hash, seed, package
version, and hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from dotime import __version__, _release_io
from dotime._build import build_suite
from dotime.benchmarks import SuiteMetadata

_CONFIG_PATH = Path(__file__).with_name("release_config.yaml")


# Suite generation uses the per-episode deterministic-seeding scheme in
# dotime._build (build_suite): each episode index gets an independent seed, so the
# output is identical regardless of the worker count, and generation parallelises
# cleanly across processes. The worker lives in the package (not this script) so it
# is importable by multiprocessing workers under both fork and spawn.


# --------------------------------------------------------------------------- #
# Croissant metadata
# --------------------------------------------------------------------------- #


def croissant_metadata(meta: SuiteMetadata, manifest: dict) -> dict:
    """Minimal Croissant JSON-LD descriptor for one suite."""
    return {
        "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/"},
        "@type": "Dataset",
        "name": meta.name,
        "version": meta.version,
        "description": meta.description,
        "license": f"https://spdx.org/licenses/{meta.license}.html",
        "citation": "DoTime: synthetic interventional/counterfactual time-series suites.",
        "cr:schemaVersion": manifest["schema_version"],
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": shard["file"],
                "contentSize": None,
                "md5": shard["md5"],
                "encodingFormat": "application/vnd.apache.parquet",
            }
            for shard in manifest["shards"]
        ],
        "cr:recordSet": {
            "@type": "cr:RecordSet",
            "field": [
                {
                    "@id": "x_obs",
                    "dataType": "cr:Float",
                    "description": "Observational trajectory (T*N row-major).",
                },
                {
                    "@id": "x_int",
                    "dataType": "cr:Float",
                    "description": "Interventional trajectory (T*N row-major).",
                },
                {
                    "@id": "y_true",
                    "dataType": "cr:Float",
                    "description": "Exact interventional outcome at the query.",
                },
                {
                    "@id": "structure",
                    "dataType": "cr:Text",
                    "description": "Identification structure label.",
                },
                {"@id": "tier", "dataType": "cr:Integer", "description": "Difficulty tier."},
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _suite_metadata(name: str, cfg: dict, n_episodes: int) -> SuiteMetadata:
    structures: tuple[str, ...] = ()
    if cfg["generator"] == "identifiability":
        structures = tuple(cfg["structures"].keys())
    return SuiteMetadata(
        name=name,
        version=cfg["version"],
        zenodo_record_id="LOCAL",
        doi="",
        description=f"{name} ({cfg['generator']} generator).",
        n_episodes=n_episodes,
        structures=structures,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH)
    parser.add_argument("--suite", default=None, help="Build only this suite.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale every episode count.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--timestamp", default=None, help="Override the output timestamp dir.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes. 0 (default) auto-selects ~all CPU cores. "
        "Generation uses per-episode deterministic seeding, so the output is "
        "identical regardless of the worker count.",
    )
    args = parser.parse_args(argv)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    config_text = args.config.read_text()
    config = yaml.safe_load(config_text)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:16]
    base_seed = int(config["seed"])

    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    suites = config["suites"]
    names = [args.suite] if args.suite else list(suites)
    built = []
    for offset, name in enumerate(names):
        if name not in suites:
            raise SystemExit(f"unknown suite {name!r}; available: {list(suites)}")
        cfg = suites[name]
        seed = base_seed + 1000 * (offset + 1)
        print(
            f"[build_release] generating {name} (scale={args.scale}, seed={seed}, "
            f"workers={workers}) ...",
            flush=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # handled SCM divergence
            episodes = build_suite(cfg, seed, args.scale, workers)

        meta = _suite_metadata(name, cfg, len(episodes))
        suite_dir = run_dir / f"{name}-{meta.version}"
        _release_io.write_suite(
            meta,
            episodes,
            suite_dir,
            package_version=__version__,
            seed=seed,
            extra_manifest={
                "generator": cfg["generator"],
                "config_hash": config_hash,
                "scale": args.scale,
                "scheme": "perepisode",
            },
        )
        manifest = json.loads((suite_dir / "manifest.json").read_text())
        (suite_dir / "croissant.json").write_text(
            json.dumps(croissant_metadata(meta, manifest), indent=2)
        )
        print(f"[build_release]   wrote {len(episodes)} episodes -> {suite_dir}", flush=True)
        built.append({"name": name, "n_episodes": len(episodes), "dir": suite_dir.name})

    build_manifest = {
        "created_utc": stamp,
        "package_version": __version__,
        "config_hash": config_hash,
        "base_seed": base_seed,
        "scale": args.scale,
        "scheme": "perepisode",
        "workers": workers,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "suites": built,
    }
    (run_dir / "build_manifest.json").write_text(json.dumps(build_manifest, indent=2))
    print(f"[build_release] done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
