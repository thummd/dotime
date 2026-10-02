#!/usr/bin/env python
"""Archive built suites to Zenodo and mint a DOI per suite (the citable record).

Hugging Face is the discovery mirror; Zenodo is the archive of record whose DOI
goes in the paper and `_SUITE_REGISTRY`. This uploads each suite directory from a
``build_release.py`` run as a Zenodo deposition (real author block — the D&B track
is single-blind) and prints the reserved DOI; review and publish in the Zenodo UI,
then backfill `zenodo_record_id`/`doi` into `dotime.benchmarks`. With ``--publish``
the depositions are published at once (irreversible).

The script can be rerun after a failure. A suite whose title is already
published is reported and skipped, an unpublished draft that holds the suite's
manifest is resumed, files already in it with the same checksum are not sent
again, and every file upload is retried on a dropped connection.

Usage
-----
    export ZENODO_TOKEN=...          # personal access token (deposit:write)
    python scripts/upload_zenodo.py --run-dir output/<timestamp> --namespace thummd
    python scripts/upload_zenodo.py --run-dir output/<ts> --sandbox   # test on sandbox.zenodo.org

Dependency-light: stdlib ``urllib`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.error
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def _md5(path: Path) -> str:
    """MD5 of a file, read in blocks.

    Args:
        path: The file.

    Returns:
        The hex digest, as Zenodo reports file checksums.
    """
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _plain(checksum: str | None) -> str:
    """A Zenodo checksum without its ``md5:`` prefix."""
    return (checksum or "").removeprefix("md5:")


def list_depositions(base: str, token: str, status: str) -> list[dict]:
    """Every deposition of the account with a status, all versions included.

    Args:
        base: API base URL.
        token: Zenodo token.
        status: ``"draft"`` or ``"published"``.

    Returns:
        The depositions as the API lists them.
    """
    out: list[dict] = []
    page = 1
    while True:
        batch = _req(
            "GET",
            f"{base}/deposit/depositions?status={status}&size=100&page={page}&all_versions=true",
            token,
        )
        if not batch:
            return out
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def find_published(base: str, token: str, title: str) -> dict | None:
    """The published deposition with exactly this title, if there is one.

    Titles name the suite and its version, so a match means this version is
    already archived and must not be published a second time.

    Args:
        base: API base URL.
        token: Zenodo token.
        title: The title :func:`_metadata` gives the version.

    Returns:
        The deposition, or ``None``.
    """
    for dep in list_depositions(base, token, "published"):
        if (dep.get("metadata") or {}).get("title") == title or dep.get("title") == title:
            return dep
    return None


def find_resumable_draft(base: str, token: str, manifest_md5: str) -> dict | None:
    """An unpublished draft that already holds this suite's manifest.

    A failed upload leaves its draft behind with the files that completed; the
    manifest's checksum identifies the suite version it belongs to.

    Args:
        base: API base URL.
        token: Zenodo token.
        manifest_md5: MD5 of the local ``manifest.json``.

    Returns:
        The draft with its current ``files`` and ``links``, or ``None``.
    """
    for dep in list_depositions(base, token, "draft"):
        full = _req("GET", f"{base}/deposit/depositions/{dep['id']}", token)
        files = full.get("files") or []
        if any(
            f.get("filename") == "manifest.json" and _plain(f.get("checksum")) == manifest_md5
            for f in files
        ):
            return full
    return None


def put_file(bucket: str, path: Path, token: str, attempts: int = 3) -> None:
    """Stream one file into a deposition bucket, retrying a dropped connection.

    Args:
        bucket: The deposition's bucket URL.
        path: The file.
        token: Zenodo token.
        attempts: How many times to try before giving up.

    Raises:
        urllib.error.HTTPError: On a client error (4xx), which a retry would
            only repeat, or when the last attempt fails.
        urllib.error.URLError: When the last attempt fails.
    """
    size = path.stat().st_size
    for attempt in range(1, attempts + 1):
        t0 = time.time()
        try:
            with path.open("rb") as fh:
                # Zenodo's bucket API requires an explicit content type (415
                # otherwise); the length lets urllib stream the file in blocks.
                req = urllib.request.Request(
                    f"{bucket}/{path.name}",
                    data=fh,
                    method="PUT",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(size),
                    },
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts:
                raise
            err: Exception = exc
        except (urllib.error.URLError, ConnectionError, TimeoutError, ssl.SSLError) as exc:
            if attempt == attempts:
                raise
            err = exc
        else:
            dt = max(time.time() - t0, 1e-9)
            print(
                f"[zenodo]   {path.name}: {size / 1e6:.1f} MB in {dt:.0f} s "
                f"({size / 1e6 / dt:.2f} MB/s)",
                flush=True,
            )
            return
        wait = 60 * attempt
        print(
            f"[zenodo]   {path.name}: attempt {attempt} of {attempts} failed ({err}); "
            f"retrying in {wait} s",
            flush=True,
        )
        time.sleep(wait)


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
    name = manifest["name"]
    md = _metadata(manifest)["metadata"]
    check_metadata(md)
    done = find_published(base, token, md["title"])
    if done is not None:
        result = {
            "deposition": str(done["id"]),
            "doi": done.get("doi", ""),
            "record_id": str(done.get("record_id") or done["id"]),
            "concept_doi": done.get("conceptdoi", ""),
            "published": True,
        }
        print(f"[zenodo] {name}: already published as record {result['record_id']}, skipped")
        return result
    draft = find_resumable_draft(base, token, _md5(suite_dir / "manifest.json"))
    if draft is not None:
        dep_id, bucket = draft["id"], draft["links"]["bucket"]
        have = {f["filename"]: _plain(f.get("checksum")) for f in draft.get("files") or []}
        print(f"[zenodo] {name}: resuming draft {dep_id} ({len(have)} files already there)")
    else:
        dep = _req(
            "POST",
            f"{base}/deposit/depositions",
            token,
            data=json.dumps({}).encode(),
            content_type="application/json",
        )
        dep_id, bucket, have = dep["id"], dep["links"]["bucket"], {}
        print(f"[zenodo] {name}: created draft {dep_id}")

    for f in sorted(suite_dir.iterdir()):
        if not f.is_file():
            continue
        if have.get(f.name) == _md5(f):
            print(f"[zenodo]   {f.name}: already in the draft")
            continue
        put_file(bucket, f, token)

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
        # Written after every suite, so a later failure keeps the earlier ids.
        (args.run_dir / "zenodo_depositions.json").write_text(json.dumps(results, indent=2))
    (args.run_dir / "zenodo_depositions.json").write_text(json.dumps(results, indent=2))
    tail = "" if args.publish else "; publish each deposition in the UI"
    print(f"[zenodo] wrote {args.run_dir / 'zenodo_depositions.json'}{tail}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
