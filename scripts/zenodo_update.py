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

A rerun skips a version whose title is already published, and a draft is
published only when its files have exactly the local checksums. ``--plan``
reads the account and prints what a run would do, writing nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

# upload_zenodo.py sits next to this script, which ``python scripts/...`` puts
# on sys.path, so both scripts build a version's metadata the same way.
from upload_zenodo import (
    _metadata,
    _plain,
    check_draft_files,
    check_metadata,
    find_published,
    local_files,
    put_file,
)

_BASE = "https://zenodo.org/api"


def _req(method, url, token, *, data=None, content_type=None, raw=False):
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as resp:
        body = resp.read()
        return body if raw else (json.loads(body) if body else {})


def update_suite(
    suite_dir: Path, old_record_id: str, token: str, publish: bool, plan: bool = False
) -> dict:
    """Publish one suite directory as a new version of its Zenodo concept record.

    Args:
        suite_dir: A suite directory written by ``build_release.py``.
        old_record_id: A published record of the concept to add the version to.
        token: Zenodo personal access token.
        publish: Publish the new version (irreversible) instead of leaving a draft.
        plan: Only print what a run would do; nothing is created or changed.

    Returns:
        ``record_id`` and ``version_doi`` of the new or already published
        version, or ``{}`` in plan mode when the version is not published yet.

    Raises:
        SystemExit: If the draft does not match the local build before publishing.
        urllib.error.HTTPError: If Zenodo refuses a request.
    """
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    name = manifest["name"]
    # 0. A version whose title is already published is never published twice,
    #    so the release can be rerun after a later step fails.
    done = find_published(_BASE, token, _metadata(manifest)["metadata"]["title"])
    if done is not None:
        rid = str(done.get("record_id") or done["id"])
        print(f"[zenodo] {name}: published as record {rid}, skip", flush=True)
        return {"record_id": rid, "version_doi": done.get("doi", "")}
    if plan:
        print(
            f"[zenodo] {name}: create a new version of record {old_record_id} and upload "
            "the files that differ from it",
            flush=True,
        )
        return {}
    # 1. New draft version from the published record. Zenodo keeps one
    #    unpublished new version per record, so a rerun gets the same draft back.
    dep = _req("POST", f"{_BASE}/deposit/depositions/{old_record_id}/actions/newversion", token)
    draft = _req("GET", dep["links"]["latest_draft"], token)
    draft_id, bucket = draft["id"], draft["links"]["bucket"]

    # 2. Keep files that already match this build, remove the others (the
    #    previous version's files or a partial upload).
    local = local_files(suite_dir)
    keep = set()
    for f in draft.get("files", []):
        file_name = f.get("filename")
        if file_name in local and local[file_name] == _plain(f.get("checksum")):
            keep.add(file_name)
            continue
        fid = f.get("id") or f.get("file_id")
        _req("DELETE", f"{_BASE}/deposit/depositions/{draft_id}/files/{fid}", token)

    # 3. Upload the rest of the freshly built files.
    for f in sorted(suite_dir.iterdir()):
        if f.is_file() and f.name not in keep:
            put_file(bucket, f, token)

    # 4. Ensure metadata (creators + version + a supersedes note).
    md = draft["metadata"]
    # A new version starts as a copy of the previous one, whose title and
    # description named the old version and its episode count.
    md.update(_metadata(manifest)["metadata"])
    md["notes"] = "Per-episode deterministic generation (scheme=perepisode)."
    check_metadata(md)
    _req(
        "PUT",
        f"{_BASE}/deposit/depositions/{draft_id}",
        token,
        data=json.dumps({"metadata": md}).encode(),
        content_type="application/json",
    )

    doi = draft.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
    check_draft_files(_BASE, token, draft_id, local)
    if publish:
        pub = _req("POST", f"{_BASE}/deposit/depositions/{draft_id}/actions/publish", token)
        doi = pub.get("doi", doi)
        print(f"[zenodo] {name}: published new version {draft_id}, DOI {doi}", flush=True)
    else:
        print(f"[zenodo] {name}: draft {draft_id} ready (not published)", flush=True)
    return {"record_id": str(draft_id), "version_doi": doi}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--no-publish", action="store_true", help="Create drafts but don't publish."
    )
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Read the account and print what a run would do; change and write nothing.",
    )
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
        record_id = "" if meta is None else meta.zenodo_record_id
        if record_id in ("", "TODO", "LOCAL") and meta is not None and meta.prior_versions:
            # The registry lists the version being released as LOCAL until it is
            # minted, so the new version hangs off the newest published record.
            record_id = meta.prior_versions[0][1]
        if record_id in ("", "TODO", "LOCAL"):
            print(f"[zenodo] skip {name}: no existing record id in registry", file=sys.stderr)
            continue
        try:
            out[name] = update_suite(
                suite_dir, record_id, token, not args.no_publish, plan=args.plan
            )
        except urllib.error.HTTPError as e:
            print(f"[zenodo] {name}: HTTP {e.code} {e.read()[:200]!r}", file=sys.stderr)
            raise
    if args.plan:
        print("[zenodo] --plan: nothing was changed or written.")
        return 0
    (args.run_dir / "zenodo_versions.json").write_text(json.dumps(out, indent=2))
    print(f"[zenodo] wrote {args.run_dir / 'zenodo_versions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
