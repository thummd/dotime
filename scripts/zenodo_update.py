#!/usr/bin/env python
"""Publish a NEW VERSION of each already-published Zenodo suite record.

Zenodo records are immutable once published, so updating a suite's data means
minting a new *version* under the same concept DOI. This reads the current record
id per suite from ``dotime.benchmarks._SUITE_REGISTRY``, and for each suite
directory in ``--run-dir`` it: creates a new draft version, replaces the files
with the freshly-built ones, ensures the author block, and publishes — yielding a
new version-DOI (the concept DOI is stable, so the paper/registry can cite that).

Usage
-----
    export ZENODO_TOKEN=...
    python scripts/zenodo_update.py --run-dir output/<timestamp>
    python scripts/zenodo_update.py --run-dir output/<ts> --no-publish   # dry-ish

Writes ``<run-dir>/zenodo_versions.json`` mapping suite -> new record id +
version/concept DOIs. Dependency-light (stdlib urllib).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_BASE = "https://zenodo.org/api"
_CREATORS = [
    {"name": "Thumm, Dennis", "affiliation": "National University of Singapore"},
    {"name": "Anthony, Billy Tim", "affiliation": "National University of Singapore"},
    {"name": "Chen, Ying", "affiliation": "National University of Singapore"},
]


def _req(method, url, token, *, data=None, content_type=None, raw=False):
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
        return body if raw else (json.loads(body) if body else {})


def update_suite(suite_dir: Path, old_record_id: str, token: str, publish: bool) -> dict:
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    # 1. New draft version from the published record.
    dep = _req("POST", f"{_BASE}/deposit/depositions/{old_record_id}/actions/newversion", token)
    draft = _req("GET", dep["links"]["latest_draft"], token)
    draft_id, bucket = draft["id"], draft["links"]["bucket"]

    # 2. Remove the files copied from the previous version.
    for f in draft.get("files", []):
        fid = f.get("id") or f.get("file_id")
        _req("DELETE", f"{_BASE}/deposit/depositions/{draft_id}/files/{fid}", token)

    # 3. Upload the freshly-built files.
    for f in sorted(suite_dir.iterdir()):
        if f.is_file():
            with f.open("rb") as fh:
                _req(
                    "PUT",
                    f"{bucket}/{f.name}",
                    token,
                    data=fh.read(),
                    content_type="application/octet-stream",
                )

    # 4. Ensure metadata (creators + version + a supersedes note).
    md = draft["metadata"]
    md["creators"] = _CREATORS
    md["version"] = manifest["version"]
    md["notes"] = "Per-episode deterministic generation (scheme=perepisode)."
    _req(
        "PUT",
        f"{_BASE}/deposit/depositions/{draft_id}",
        token,
        data=json.dumps({"metadata": md}).encode(),
        content_type="application/json",
    )

    doi = draft.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    if publish:
        pub = _req("POST", f"{_BASE}/deposit/depositions/{draft_id}/actions/publish", token)
        doi = pub.get("doi", doi)
        print(
            f"[zenodo] {manifest['name']}: published new version {draft_id}, DOI {doi}", flush=True
        )
    else:
        print(f"[zenodo] {manifest['name']}: draft {draft_id} ready (not published)", flush=True)
    return {"record_id": str(draft_id), "version_doi": doi}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--no-publish", action="store_true", help="Create drafts but don't publish."
    )
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    token = args.token or os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("set $ZENODO_TOKEN or pass --token")

    from dotime.benchmarks import _SUITE_REGISTRY

    out = {}
    for suite_dir in sorted(args.run_dir.glob("dot-*")):
        if not (suite_dir / "manifest.json").exists():
            continue
        name = json.loads((suite_dir / "manifest.json").read_text())["name"]
        meta = _SUITE_REGISTRY.get(name)
        if meta is None or meta.zenodo_record_id in ("", "TODO", "LOCAL"):
            print(f"[zenodo] skip {name}: no existing record id in registry", file=sys.stderr)
            continue
        try:
            out[name] = update_suite(suite_dir, meta.zenodo_record_id, token, not args.no_publish)
        except urllib.error.HTTPError as e:
            print(f"[zenodo] {name}: HTTP {e.code} {e.read()[:200]!r}", file=sys.stderr)
            raise
    (args.run_dir / "zenodo_versions.json").write_text(json.dumps(out, indent=2))
    print(f"[zenodo] wrote {args.run_dir / 'zenodo_versions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
