#!/usr/bin/env python
"""Archive built suites to Zenodo and mint a DOI per suite (the citable record).

Hugging Face is the discovery mirror; Zenodo is the archive of record whose DOI
goes in the paper and `_SUITE_REGISTRY`. This uploads each suite directory from a
``build_release.py`` run as a Zenodo deposition (real author block — the D&B track
is single-blind) and prints the reserved DOI; review and publish in the Zenodo UI,
then backfill `zenodo_record_id`/`doi` into `causaltimeprior.benchmarks`.

Usage
-----
    export ZENODO_TOKEN=...          # personal access token (deposit:write)
    python scripts/upload_zenodo.py --run-dir output/<timestamp> --namespace thummd
    python scripts/upload_zenodo.py --run-dir output/<ts> --sandbox   # test on sandbox.zenodo.org

Dependency-light: stdlib ``urllib`` only.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def _base(sandbox: bool) -> str:
    return "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"


def _req(
    method: str, url: str, token: str, *, data: bytes | None = None, content_type: str | None = None
):
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _metadata(manifest: dict) -> dict:
    return {
        "metadata": {
            "title": f"CausalTimePrior — {manifest['name']} (v{manifest['version']})",
            "upload_type": "dataset",
            "description": (
                f"Frozen evaluation suite '{manifest['name']}' from CausalTimePrior "
                f"(KDD 2027 Datasets & Benchmarks). {manifest['n_episodes']} episodes; "
                "parquet shards + manifest + Croissant metadata. Generated reproducibly "
                "by scripts/build_release.py."
            ),
            "license": "cc-by-4.0",
            "version": manifest["version"],
            "keywords": ["causal inference", "time series", "benchmark", "interventional"],
        }
    }


def upload_suite(suite_dir: Path, token: str, base: str) -> tuple[str, str]:
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    dep = _req(
        "POST",
        f"{base}/deposit/depositions",
        token,
        data=json.dumps({}).encode(),
        content_type="application/json",
    )
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]

    for f in sorted(suite_dir.iterdir()):
        if f.is_file():
            with f.open("rb") as fh:
                _req("PUT", f"{bucket}/{f.name}", token, data=fh.read())

    _req(
        "PUT",
        f"{base}/deposit/depositions/{dep_id}",
        token,
        data=json.dumps(_metadata(manifest)).encode(),
        content_type="application/json",
    )
    reserved = _req("GET", f"{base}/deposit/depositions/{dep_id}", token)
    doi = reserved.get("metadata", {}).get("prereserve_doi", {}).get("doi", "(reserve in UI)")
    print(
        f"[zenodo] {manifest['name']}: deposition {dep_id}, reserved DOI {doi} "
        f"(review + publish at {base.replace('/api', '')}/deposit/{dep_id})"
    )
    return str(dep_id), doi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="build_release output dir.")
    parser.add_argument(
        "--sandbox", action="store_true", help="Use sandbox.zenodo.org for testing."
    )
    parser.add_argument("--token", default=None, help="Zenodo token (else $ZENODO_TOKEN).")
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set $ZENODO_TOKEN or pass --token")
    base = _base(args.sandbox)

    results = {}
    for suite_dir in sorted(args.run_dir.glob("CTP-*")):
        if (suite_dir / "manifest.json").exists():
            dep_id, doi = upload_suite(suite_dir, token, base)
            results[suite_dir.name] = {"deposition": dep_id, "doi": doi}
    (args.run_dir / "zenodo_depositions.json").write_text(json.dumps(results, indent=2))
    print(
        f"[zenodo] wrote {args.run_dir / 'zenodo_depositions.json'}; publish each deposition in the UI."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
