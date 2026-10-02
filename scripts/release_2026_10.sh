#!/usr/bin/env bash
# Release the 2026-10 suites: verify the shard digests offline, mirror every suite to
# Hugging Face, publish dot-Identifiability-v1 1.2.0 as a new version of its Zenodo
# concept record and publish the four new suites as new Zenodo records.
#
# Run it from the repository root once the run directory is ready, first with --check:
#
#   set -a; . ./.env; set +a; bash scripts/release_2026_10.sh --check <run-dir> [<s13-checkpoint-dir>]
#   set -a; . ./.env; set +a; bash scripts/release_2026_10.sh <run-dir> [<s13-checkpoint-dir>]
#
# --check finds the tokens and verifies every digest, then stops before any upload.
# --zenodo-only skips the Hugging Face uploads, e.g. to resume after a Zenodo failure.
# A rerun is safe: versions already published on Zenodo are skipped by title, an
# unpublished draft of a suite is resumed and files already in it are not sent again.
# The tokens may be stored under any of the usual names (see pick below); the script
# passes them on as HF_TOKEN and ZENODO_TOKEN, which the upload scripts read, and never
# prints a value, only the name it used.
#
# <run-dir> holds one directory (or symlink) per suite, named like build_release.py
# writes them (dot-<Suite>-<version>), each with its manifest.json and shards. The
# script stops at the first failure and can then be rerun with the same arguments:
# upload_huggingface.py re-points the version tag, and the Zenodo scripts skip what
# is already published. Their JSON outputs in <run-dir> record what was published.
set -euo pipefail

CHECK=0
ZENODO_ONLY=0
while [ "${1:-}" = "--check" ] || [ "${1:-}" = "--zenodo-only" ]; do
    [ "$1" = "--check" ] && CHECK=1
    [ "$1" = "--zenodo-only" ] && ZENODO_ONLY=1
    shift
done
RUN_DIR="${1:?usage: release_2026_10.sh [--check] [--zenodo-only] <run-dir> [<s13-checkpoint-dir>]}"
S13_DIR="${2:-}"
PY="${PY:-python}"

# pick <target> <name>...: export the first non-empty variable among the names as
# <target>. Only the name is printed.
pick() {
    local target=$1
    shift
    local name
    for name in "$@"; do
        if [ -n "${!name:-}" ]; then
            export "$target=${!name}"
            echo "[release] $target: taken from \$$name"
            return 0
        fi
    done
    return 1
}
missing=0
pick HF_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HUGGINGFACE_TOKEN \
    HUGGING_FACE_TOKEN HF_API_TOKEN HF_API_KEY HUGGINGFACE_API_TOKEN HF_WRITE_TOKEN || missing=1
pick ZENODO_TOKEN ZENODO_TOKEN ZENODO_ACCESS_TOKEN ZENODO_API_TOKEN ZENODO_API_KEY \
    ZENODO_PERSONAL_TOKEN ZENODO_PAT || missing=1
if [ "$missing" = 1 ]; then
    echo "[release] a token is missing. Exported variables whose names mention HF, HUGGING or"
    echo "[release] ZENODO (names only, no values):"
    compgen -e | grep -iE 'hf|hugging|zenodo' | sed 's/^/[release]   /' || echo "[release]   none"
    echo "[release] Set HF_TOKEN and ZENODO_TOKEN, e.g. HF_TOKEN=\$YOUR_NAME on the command line."
    exit 1
fi

echo "[release] run dir: $RUN_DIR"
echo "[release] suites: $(ls -d "$RUN_DIR"/dot-* | xargs -n1 basename | tr '\n' ' ')"
if [ -n "$S13_DIR" ]; then
    echo "[release] s13 checkpoint folders: $(ls -d "$S13_DIR"/s13ho_* | wc -l) in $S13_DIR"
fi

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

# 1b. The s13 folder: every file in MANIFEST.sha256 present and matching.
if [ -n "$S13_DIR" ]; then
    (cd "$S13_DIR" && sha256sum --quiet -c MANIFEST.sha256) \
        || { echo "[release] s13 manifest mismatch; nothing uploaded"; exit 1; }
    echo "[release] s13 manifest: all files match"
fi

if [ "$CHECK" = 1 ]; then
    echo "[release] --check passed: tokens found, digests verified. Nothing was uploaded."
    exit 0
fi

if [ "$ZENODO_ONLY" = 0 ]; then
    # 2. Hugging Face mirror (dataset repo per suite, tag v<version>).
    "$PY" scripts/upload_huggingface.py --run-dir "$RUN_DIR" --namespace thummd

    # 2b. The pre-registered s13 checkpoints, when a directory is given as the second
    #     argument: one folder per run with do_over_time_pfn_last.pt, cmd.txt,
    #     train.log and step_losses.csv.
    if [ -n "$S13_DIR" ]; then
        "$PY" scripts/upload_huggingface.py --checkpoint-dir "$S13_DIR" --path-in-repo s13 \
            --namespace thummd --model-repo do-over-time-pfn
    fi
else
    echo "[release] --zenodo-only: Hugging Face uploads skipped"
fi

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
