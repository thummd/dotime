#!/usr/bin/env python
"""Archive built suites to Zenodo and mint a DOI per suite (the citable record).

Hugging Face is the discovery mirror; Zenodo is the archive of record whose DOI
goes in the paper and `_SUITE_REGISTRY`. This uploads each suite directory from a
``build_release.py`` run as a Zenodo deposition (real author block — the D&B track
is single-blind) and prints the reserved DOI; review and publish in the Zenodo UI,
then backfill `zenodo_record_id`/`doi` into `dotime.benchmarks`. With ``--publish``
the depositions are published at once (irreversible).

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


# The author block of every DoTime suite record, in the order of the paper
# (CITATION.cff, preferred-citation). zenodo_update.py imports it, so new
# records and new versions of existing ones carry the same creators.
CREATORS = [
    {"name": "Thumm, Dennis", "affiliation": "National University of Singapore"},
    {"name": "Anthony, Billy Tim", "affiliation": "National University of Singapore"},
    {"name": "Chen, Ying", "affiliation": "National University of Singapore"},
]
# Fields Zenodo requires before it publishes a deposition.
REQUIRED = ("title", "upload_type", "description", "creators")


def _metadata(manifest: dict) -> dict:
    """Zenodo metadata of one suite version, built from its manifest.

    Every field that names the version or its size comes from the manifest, so
    a new version never inherits the previous version's title or episode count.

    Args:
        manifest: The suite's ``manifest.json``.

    Returns:
        ``{"metadata": {...}}`` with title, upload type, description, creators,
        license, version and keywords.
    """
    name, version = manifest["name"], manifest["version"]
    about = ""
    try:
        from dotime.benchmarks import _SUITE_REGISTRY

        if name in _SUITE_REGISTRY:
            about = _SUITE_REGISTRY[name].description.strip() + " "
    except ImportError:  # the package is optional for this script
        pass
    return {
        "metadata": {
            "title": f"DoTime: {name} (v{version})",
            "upload_type": "dataset",
            "description": (
                f"Frozen evaluation suite '{name}' from DoTime, version {version}. {about}"
                f"{manifest['n_episodes']} episodes; parquet shards + manifest + Croissant "
                "metadata. Generated reproducibly by scripts/build_release.py."
            ),
            "creators": CREATORS,
            "license": "cc-by-4.0",
            "version": version,
            "keywords": ["causal inference", "time series", "benchmark", "interventional"],
        }
    }


def check_metadata(md: dict) -> None:
    """Refuse metadata that Zenodo would not publish, before anything is created.

    Args:
        md: The ``metadata`` dict of a deposition.

    Raises:
        SystemExit: If a required field is missing or empty.
    """
    missing = [k for k in REQUIRED if not md.get(k)]
    if missing:
        raise SystemExit(f"[zenodo] metadata lacks {missing}; nothing was created")


def upload_suite(suite_dir: Path, token: str, base: str, publish: bool = False) -> dict:
    """Create a Zenodo deposition for one built suite and optionally publish it.

    Args:
        suite_dir: A suite directory written by ``build_release.py``.
        token: Zenodo personal access token.
        base: API base URL (production or sandbox).
        publish: Publish the deposition right away. Publishing is irreversible,
            so the default only reserves the DOI for review in the UI.

    Returns:
        ``deposition`` and ``doi`` (reserved or final), plus ``record_id``,
        ``concept_doi`` and ``published`` when the deposition was published.
    """
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    check_metadata(_metadata(manifest)["metadata"])
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
                # Zenodo's bucket API requires an explicit content type (415 otherwise).
                _req(
                    "PUT",
                    f"{bucket}/{f.name}",
                    token,
                    data=fh.read(),
                    content_type="application/octet-stream",
                )

    _req(
        "PUT",
        f"{base}/deposit/depositions/{dep_id}",
        token,
        data=json.dumps(_metadata(manifest)).encode(),
        content_type="application/json",
    )
    reserved = _req("GET", f"{base}/deposit/depositions/{dep_id}", token)
    doi = reserved.get("metadata", {}).get("prereserve_doi", {}).get("doi", "(reserve in UI)")
    result = {"deposition": str(dep_id), "doi": doi, "published": False}
    if publish:
        pub = _req("POST", f"{base}/deposit/depositions/{dep_id}/actions/publish", token)
        result.update(
            doi=pub.get("doi", doi),
            record_id=str(pub.get("record_id") or pub.get("id") or dep_id),
            concept_doi=pub.get("conceptdoi", ""),
            published=True,
        )
        print(
            f"[zenodo] {manifest['name']}: published record {result['record_id']}, "
            f"DOI {result['doi']}, concept DOI {result['concept_doi']}"
        )
    else:
        print(
            f"[zenodo] {manifest['name']}: deposition {dep_id}, reserved DOI {doi} "
            f"(review + publish at {base.replace('/api', '')}/deposit/{dep_id})"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="build_release output dir.")
    parser.add_argument(
        "--sandbox", action="store_true", help="Use sandbox.zenodo.org for testing."
    )
    parser.add_argument("--token", default=None, help="Zenodo token (else $ZENODO_TOKEN).")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish each new deposition immediately (irreversible) instead of "
        "leaving it for review in the UI.",
    )
    args = parser.parse_args(argv)

    token = args.token or os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set $ZENODO_TOKEN or pass --token")
    base = _base(args.sandbox)

    from dotime.benchmarks import _SUITE_REGISTRY

    results = {}
    for suite_dir in sorted(args.run_dir.glob("dot-*")):
        if not (suite_dir / "manifest.json").exists():
            continue
        name = json.loads((suite_dir / "manifest.json").read_text())["name"]
        meta = _SUITE_REGISTRY.get(name)
        # A suite that already has a Zenodo concept record gets a new *version*
        # through zenodo_update.py; a second concept record would split its DOI.
        if meta is not None and (
            meta.zenodo_record_id not in ("", "TODO", "LOCAL") or meta.prior_versions
        ):
            print(f"[zenodo] skip {name}: it has a concept record, use zenodo_update.py")
            continue
        results[suite_dir.name] = upload_suite(suite_dir, token, base, publish=args.publish)
    (args.run_dir / "zenodo_depositions.json").write_text(json.dumps(results, indent=2))
    tail = "" if args.publish else "; publish each deposition in the UI"
    print(f"[zenodo] wrote {args.run_dir / 'zenodo_depositions.json'}{tail}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
