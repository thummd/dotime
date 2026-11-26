#!/usr/bin/env python
"""Score the reference baselines on a local build of dot-RegimeSwitch-v1 1.1.0.

``dotime-eval-reference`` loads a suite through the registry, which names the
released version only. This wrapper points the registry of the running process
at a local build directory and then calls the same harness, so the rows are the
ones the command line tool will print once the version is registered. Nothing
is downloaded and the installed registry is not changed.

    python results/reference/regime_v1_1/scripts/reference_rows.py \\
        --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \\
        --out results/reference/regime_v1_1/regime_v1_1_cpu.json

The harness logs and asserts the per-arm target statistics before it scores
(``dotime.reference.reference_table.target_qa``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
from pathlib import Path


def verified_manifest(suite_dir: Path) -> dict:
    """Read the manifest of a build and check every shard against its digest.

    Args:
        suite_dir: A suite directory written by ``scripts/build_release.py``.

    Returns:
        The manifest.

    Raises:
        FileNotFoundError: If the manifest or a shard is missing.
        ValueError: If a shard does not match its recorded md5.
    """
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    for shard in manifest["shards"]:
        digest = hashlib.md5((suite_dir / shard["file"]).read_bytes()).hexdigest()
        if digest != shard["md5"]:
            raise ValueError(f"{shard['file']} does not match its manifest digest")
    return manifest


def main(argv: list[str] | None = None) -> None:
    """Register the local build for this process and run the reference harness.

    Args:
        argv: Command line, ``sys.argv[1:]`` by default.

    Raises:
        SystemExit: If the build is not a newer version of a registered suite.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baselines", nargs="+", default=None)
    args = ap.parse_args(argv)

    from dotime import benchmarks
    from dotime.reference import reference_table

    suite_dir = args.suite_dir.resolve()
    manifest = verified_manifest(suite_dir)
    name, version = manifest["name"], manifest["version"]
    released = benchmarks._SUITE_REGISTRY.get(name)
    if released is None or released.version == version:
        raise SystemExit(f"{name} {version} is not a new version of a registered suite")
    # hf_repo_id is cleared so that a missing cache entry can never turn into a
    # download of the released version under the new version's name.
    local = dataclasses.replace(
        released,
        version=version,
        zenodo_record_id="LOCAL",
        hf_repo_id="",
        n_episodes=int(manifest["n_episodes"]),
        prior_versions=((released.version, released.zenodo_record_id), *released.prior_versions),
    )
    benchmarks._SUITE_REGISTRY[name] = local
    with tempfile.TemporaryDirectory() as cache:
        os.symlink(suite_dir, Path(cache) / f"{name}-{version}")
        os.environ["DOTIME_CACHE"] = cache
        cli = ["--suite", name, "--version", version, "--out", str(args.out)]
        if args.baselines:
            cli += ["--baselines", *args.baselines]
        reference_table.main(cli)
    # The harness does not know where the episodes came from, so record it here.
    result = json.loads(args.out.read_text())
    result["build"] = {
        "suite_dir": suite_dir.name,
        "version": version,
        "config_hash": manifest.get("config_hash"),
        "package_version": manifest.get("package_version"),
        "shards_md5": [shard["md5"] for shard in manifest["shards"]],
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
