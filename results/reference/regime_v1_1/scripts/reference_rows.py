#!/usr/bin/env python
"""Score the reference baselines on a local build of dot-RegimeSwitch-v1 1.1.0.

``dotime-eval-reference`` loads a suite through the registry, which names the
released version only. This wrapper points the registry of the running process
at a local build directory and then calls the same harness, so the rows are the
ones the command line tool will print once the version is registered. Nothing
is downloaded and the installed registry is not changed.

    python results/reference/regime_v1_1/scripts/reference_rows.py \\
        --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \\
        --out results/reference/regime_v1_1/regime_v1_1_cpu.json \\
        --per-lag-out results/reference/regime_v1_1/regime_v1_1_per_lag.json

With ``--per-lag-out`` it also runs the per-group script of the November audit
(``results/reference/audit_2026-11/scripts/group_rows.py``), which splits the
rows by the shortest lag from the treatment to the query and first checks that
its pooled rows equal the reference file written here.

Both harnesses log and assert the per-arm target statistics before they score.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import runpy
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GROUP_ROWS = ROOT / "results" / "reference" / "audit_2026-11" / "scripts" / "group_rows.py"


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


@contextlib.contextmanager
def registered_locally(suite_dir: Path, manifest: dict) -> Iterator[tuple[str, str]]:
    """Make ``load_benchmark`` of this process serve a local build.

    Args:
        suite_dir: The build directory.
        manifest: Its manifest.

    Yields:
        ``(name, version)`` of the build, loadable for the duration of the block.

    Raises:
        SystemExit: If the build is not a newer version of a registered suite.
    """
    from dotime import benchmarks

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
    previous_cache = os.environ.get("DOTIME_CACHE")
    benchmarks._SUITE_REGISTRY[name] = local
    try:
        with tempfile.TemporaryDirectory() as cache:
            os.symlink(suite_dir, Path(cache) / f"{name}-{version}")
            os.environ["DOTIME_CACHE"] = cache
            yield name, version
    finally:
        benchmarks._SUITE_REGISTRY[name] = released
        if previous_cache is None:
            os.environ.pop("DOTIME_CACHE", None)
        else:
            os.environ["DOTIME_CACHE"] = previous_cache


def main(argv: list[str] | None = None) -> None:
    """Register the local build for this process and run the harnesses.

    Args:
        argv: Command line, ``sys.argv[1:]`` by default.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-lag-out", type=Path, default=None)
    ap.add_argument("--baselines", nargs="+", default=None)
    args = ap.parse_args(argv)

    from dotime.reference import reference_table

    suite_dir = args.suite_dir.resolve()
    manifest = verified_manifest(suite_dir)
    written = [args.out]
    with registered_locally(suite_dir, manifest) as (name, version):
        cli = ["--suite", name, "--version", version, "--out", str(args.out)]
        if args.baselines:
            cli += ["--baselines", *args.baselines]
        reference_table.main(cli)
        if args.per_lag_out is not None:
            # The audit script is a command line tool, so it is run as one. Its
            # --reference gate compares its pooled rows with the file above.
            group_cli = [
                "--suite", name, "--version", version, "--group", "min_lag",
                "--reference", str(args.out), "--out", str(args.per_lag_out),
            ]  # fmt: skip
            if args.baselines:
                group_cli += ["--baselines", *args.baselines]
            saved_argv = sys.argv
            sys.argv = [str(GROUP_ROWS), *group_cli]
            try:
                runpy.run_path(str(GROUP_ROWS), run_name="__main__")
            finally:
                sys.argv = saved_argv
            written.append(args.per_lag_out)
    # The harnesses do not know where the episodes came from, so record it here.
    build = {
        "suite_dir": suite_dir.name,
        "version": manifest["version"],
        "config_hash": manifest.get("config_hash"),
        "package_version": manifest.get("package_version"),
        "shards_md5": [shard["md5"] for shard in manifest["shards"]],
    }
    for path in written:
        result = json.loads(path.read_text())
        result["build"] = build
        path.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
