#!/usr/bin/env python
"""Mirror built suites (and optionally a model checkpoint) to the Hugging Face Hub.

The Hub is the primary discovery surface (Zenodo remains the citable archive).
Each suite directory produced by ``build_release.py`` becomes a HF *dataset* repo
``<namespace>/<suite-name>`` with a ``v<version>`` git tag, so the package's
``load_benchmark`` can pull a pinned revision via ``snapshot_download``.

Usage
-----
    huggingface-cli login            # or pass --token / set HF_TOKEN
    python scripts/upload_huggingface.py --run-dir output/<timestamp> --namespace thummd
    python scripts/upload_huggingface.py --checkpoint best.pt --namespace thummd \\
        --model-repo do-over-time-pfn

Needs the ``hf`` extra: ``pip install 'dotime[hf]'``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path


def _api(token: str | None):
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit("this script needs the 'hf' extra: pip install 'dotime[hf]'") from exc
    return HfApi(token=token or os.environ.get("HF_TOKEN"))


_DATASET_CARD = """---
license: cc-by-4.0
tags: [causal-inference, time-series, benchmark, dotime]
---

# {name}

A frozen evaluation suite from **DoTime** (KDD 2027 Datasets & Benchmarks).

- Episodes: {n_episodes}
- Schema: parquet shards + `manifest.json` (md5-checksummed), Croissant metadata.
- Load with:

```python
from dotime.benchmarks import load_benchmark
suite = load_benchmark("{name}")   # pulls this repo at tag v{version}
```

Generated reproducibly by `scripts/build_release.py`. Zenodo DOI is the citable
archive of record.
"""


def upload_suite(api, suite_dir: Path, namespace: str, private: bool) -> str:
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    name, version = manifest["name"], manifest["version"]
    repo_id = f"{namespace}/{name}"
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    card = suite_dir / "README.md"
    card.write_text(
        _DATASET_CARD.format(name=name, version=version, n_episodes=manifest["n_episodes"])
    )
    # delete_patterns removes any stale shards from a prior upload so the repo
    # contents are exactly this build (the version tag must point at fresh data).
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(suite_dir),
        delete_patterns=["*.parquet", "*.json"],
    )
    # create_tag(exist_ok=True) does NOT move an existing tag, so when re-hosting
    # we delete it first to re-point v<version> at the new commit.
    tag = f"v{version}"
    with contextlib.suppress(Exception):  # tag may not exist yet (first upload)
        api.delete_tag(repo_id, repo_type="dataset", tag=tag)
    api.create_tag(repo_id, repo_type="dataset", tag=tag)
    print(f"[hf] uploaded {name} -> https://huggingface.co/datasets/{repo_id} (tag {tag})")
    return repo_id


def upload_checkpoint(api, checkpoint: Path, namespace: str, model_repo: str, private: bool) -> str:
    repo_id = f"{namespace}/{model_repo}"
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=checkpoint.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[hf] uploaded {checkpoint.name} -> https://huggingface.co/{repo_id}")
    return repo_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, help="build_release output dir (uploads every suite in it)."
    )
    parser.add_argument("--suite-dir", type=Path, help="A single suite directory to upload.")
    parser.add_argument("--checkpoint", type=Path, help="A model checkpoint to upload.")
    parser.add_argument("--namespace", required=True, help="HF user or org, e.g. 'thummd'.")
    parser.add_argument(
        "--model-repo", default="do-over-time-pfn", help="Model repo name for the checkpoint."
    )
    parser.add_argument("--private", action="store_true", help="Create private repos.")
    parser.add_argument("--token", default=None, help="HF token (else $HF_TOKEN / cached login).")
    args = parser.parse_args(argv)

    api = _api(args.token)
    repos = []
    if args.run_dir:
        for suite_dir in sorted(args.run_dir.glob("dot-*")):
            if (suite_dir / "manifest.json").exists():
                repos.append(upload_suite(api, suite_dir, args.namespace, args.private))
    if args.suite_dir:
        repos.append(upload_suite(api, args.suite_dir, args.namespace, args.private))
    if args.checkpoint:
        repos.append(
            upload_checkpoint(api, args.checkpoint, args.namespace, args.model_repo, args.private)
        )

    if not repos:
        raise SystemExit("nothing to upload: pass --run-dir, --suite-dir, and/or --checkpoint")
    print(f"[hf] done: {len(repos)} repo(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
