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

Before a suite is written, its per-arm target statistics (observational level,
interventional level and effect at each query) are logged and asserted, pooled
and per structure, and recorded in both manifests (``--target-qa``).
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
from dotime._build import _OPT_IN_SPEC_KEYS, build_suite
from dotime.benchmarks import SuiteMetadata
from dotime.qa import target_qa

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


def suite_seed(cfg: dict, base_seed: int, position: int) -> int:
    """Seed of one suite: its own ``seed`` when it sets one, else one derived from its place.

    The derived seed depends on the suite's position in the config, not in the
    list being built, so ``--suite X`` builds X with the seed that a full build
    gives it.

    Args:
        cfg: The suite's config.
        base_seed: The config's top-level ``seed``.
        position: Index of the suite in the config's ``suites`` mapping.

    Returns:
        ``cfg["seed"]`` if set, otherwise ``base_seed + 1000 * (position + 1)``.

    Raises:
        TypeError: If ``cfg["seed"]`` is neither a number nor a string.
        ValueError: If ``cfg["seed"]`` is a string that is not an integer.
    """
    if "seed" in cfg:
        return int(cfg["seed"])
    return base_seed + 1000 * (position + 1)


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


def suite_target_qa(name: str, episodes: list, mode: str) -> dict:
    """Log and assert the per-arm target statistics of one built suite.

    Every arm is asserted, the effect included, because a released suite is
    scored on both the interventional level and the effect. Groups are the
    suite's structures (identifiability, continuous) or regime densities.

    Args:
        name: Suite name, used to prefix the log lines.
        episodes: The generated episodes, before they are written.
        mode: ``"enforce"``, ``"warn"`` or ``"off"``.

    Returns:
        ``{"mode": mode}`` for ``"off"``, otherwise the mode plus
        :meth:`dotime.qa.QAReport.to_dict`, which is what the manifests record.
    """
    if mode == "off":
        return {"mode": mode}
    report = target_qa(
        episodes,
        dir_target="effect",
        raise_on_failure=False,
        log=lambda line: print(f"[build_release] {name} {line}", flush=True),
    )
    return {"mode": mode, **report.to_dict()}


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
    parser.add_argument(
        "--stability-retries",
        type=int,
        default=None,
        help="Override each suite's stability_retries. When either arm of a "
        "generic, regime or identifiability episode comes back all-zero "
        "(diverged), resample it deterministically up to this many times instead "
        "of releasing the zeroed episode. Continuous episodes are never resampled. "
        "0 reproduces the 1.0.0 suites. Omit the flag to keep each config's own "
        "value (release_config_v1_1.yaml uses 3). ~20 drives the divergence rate "
        "to zero.",
    )
    parser.add_argument(
        "--target-qa",
        choices=["enforce", "warn", "off"],
        default="enforce",
        help="Per-arm target QA of each suite, pooled and per structure, after "
        "generation and before writing (dotime.qa.target_qa). 'enforce' (default) "
        "stops before writing a failing suite and exits with status 1, 'warn' records "
        "the failure and writes the suite anyway, 'off' skips the check. The report "
        "goes into each manifest.json and build_manifest.json, never the shards.",
    )
    args = parser.parse_args(argv)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    config_text = args.config.read_text()
    config = yaml.safe_load(config_text)
    # A CLI override changes what is generated, so it has to change the
    # provenance hash too -- otherwise two different builds claim one hash.
    hash_text = config_text
    if args.stability_retries is not None:
        hash_text += f"\n# --stability-retries={args.stability_retries}\n"
    config_hash = hashlib.sha256(hash_text.encode()).hexdigest()[:16]
    base_seed = int(config["seed"])

    stamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    suites = config["suites"]
    names = [args.suite] if args.suite else list(suites)
    built = []
    failed = None
    for name in names:
        if name not in suites:
            raise SystemExit(f"unknown suite {name!r}; available: {list(suites)}")
        cfg = suites[name]
        if args.stability_retries is not None:
            cfg = {**cfg, "stability_retries": args.stability_retries}
        seed = suite_seed(cfg, base_seed, list(suites).index(name))
        print(
            f"[build_release] generating {name} (scale={args.scale}, seed={seed}, "
            f"workers={workers}, stability_retries={cfg.get('stability_retries', 0)}) ...",
            flush=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # handled SCM divergence
            episodes = build_suite(cfg, seed, args.scale, workers)
        # Checked in memory before anything is written: seeds guard against
        # variance, not against a systematically corrupted target.
        qa = suite_target_qa(name, episodes, args.target_qa)
        if qa.get("passed") is False and args.target_qa == "enforce":
            print(
                f"[build_release] {name}: target QA failed, not writing it "
                "(--target-qa warn writes it anyway)",
                flush=True,
            )
            failed = {"name": name, "target_qa": qa}
            break

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
                "pair_mode": cfg.get("pair_mode", "interventional"),
                "stability_retries": int(cfg.get("stability_retries", 0)),
                "config_hash": config_hash,
                "scale": args.scale,
                "scheme": "perepisode",
                # Every opt-in key reaches the episode specs, so the manifest
                # records each one a suite sets. Legacy suites set none and
                # keep their manifest keys.
                **{k: cfg[k] for k in _OPT_IN_SPEC_KEYS if k in cfg},
                "target_qa": qa,
            },
        )
        manifest = json.loads((suite_dir / "manifest.json").read_text())
        (suite_dir / "croissant.json").write_text(
            json.dumps(croissant_metadata(meta, manifest), indent=2)
        )
        print(f"[build_release]   wrote {len(episodes)} episodes -> {suite_dir}", flush=True)
        built.append(
            {
                "name": name,
                "seed": seed,
                "n_episodes": len(episodes),
                "dir": suite_dir.name,
                "target_qa": {k: qa[k] for k in ("mode", "passed", "problems") if k in qa},
            }
        )

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
        "target_qa": args.target_qa,
        "suites": built,
    }
    if failed is not None:
        build_manifest["target_qa_failed"] = failed
    (run_dir / "build_manifest.json").write_text(json.dumps(build_manifest, indent=2))
    if failed is not None:
        print(f"[build_release] stopped: target QA failed for {failed['name']} -> {run_dir}")
        return 1
    print(f"[build_release] done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
