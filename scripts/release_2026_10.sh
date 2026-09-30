#!/usr/bin/env bash
# Release the 2026-10 suites: verify the shard digests offline, mirror every suite to
# Hugging Face, publish dot-Identifiability-v1 1.2.0 as a new version of its Zenodo
# concept record and publish the four new suites as new Zenodo records.
#
# The script reads only $HF_TOKEN and $ZENODO_TOKEN from the environment and never
# prints them. Run it from the repository root once the run directory is ready:
#
#   set -a; . ./.env; set +a; bash scripts/release_2026_10.sh <run-dir>
#
# <run-dir> holds one directory (or symlink) per suite, named like build_release.py
# writes them (dot-<Suite>-<version>), each with its manifest.json and shards. The
# script stops at the first failure, so a rerun after a fix continues safely:
# upload_huggingface.py re-points the version tag, zenodo_update.py and
# upload_zenodo.py skip nothing but refuse to duplicate a published record only
# through the registry check below, so do not rerun the Zenodo steps after they
# have published (their JSON outputs in <run-dir> record what was published).
set -euo pipefail

RUN_DIR="${1:?usage: release_2026_10.sh <run-dir>}"
PY="${PY:-python}"
: "${HF_TOKEN:?HF_TOKEN is not set (source the .env file first)}"
: "${ZENODO_TOKEN:?ZENODO_TOKEN is not set (source the .env file first)}"

echo "[release] run dir: $RUN_DIR"
echo "[release] suites: $(ls -d "$RUN_DIR"/dot-* | xargs -n1 basename | tr '\n' ' ')"

# 1. Offline digest check: every shard listed in a manifest must be present and match.
"$PY" - "$RUN_DIR" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
bad = 0
for suite_dir in sorted(run_dir.glob("dot-*")):
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    for shard in manifest["shards"]:
        path = suite_dir / shard["file"]
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        ok = digest == shard["md5"]
        bad += not ok
        print(f"[release] {suite_dir.name}/{shard['file']}: {'ok' if ok else 'MD5 MISMATCH'}")
    qa = manifest.get("target_qa", {})
    if qa and not qa.get("passed", True):
        print(f"[release] {suite_dir.name}: target QA did not pass"); bad += 1
if bad:
    sys.exit(f"[release] {bad} problem(s); nothing uploaded")
PYEOF

# 2. Hugging Face mirror (dataset repo per suite, tag v<version>).
"$PY" scripts/upload_huggingface.py --run-dir "$RUN_DIR" --namespace thummd

# 3. Zenodo: new version of the existing Identifiability concept record ...
"$PY" scripts/zenodo_update.py --run-dir "$RUN_DIR"
# ... and new records for the suites that have none yet (skips registered ones).
"$PY" scripts/upload_zenodo.py --run-dir "$RUN_DIR" --publish

# 4. Summary for the registry update (no secrets).
"$PY" - "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out = {}
for name in ("zenodo_versions.json", "zenodo_depositions.json"):
    p = run_dir / name
    if p.exists():
        out[name] = json.loads(p.read_text())
summary = run_dir / "release_2026_10_summary.json"
summary.write_text(json.dumps(out, indent=2))
print("[release] summary:")
print(json.dumps(out, indent=2))
print(f"[release] written to {summary}")
PYEOF
echo "[release] done"
