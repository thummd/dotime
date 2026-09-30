#!/usr/bin/env python
"""Score the s13 checkpoints on dot-Identifiability-v1 for the pre-registration.

This is the scoring step fixed by ``PREREG.md``. It loads every s13 checkpoint
with a strict key check, runs the packaged ``PFNRef`` protocol on every episode
of the suite (the observational arms with every intervention feature zeroed),
and writes one row per (episode, arm, seed) with the prediction, the target,
the factual level at the query and the effect. The analysis is a separate
step (``analyze_prereg.py``), so the predictions are graded once and can be
re-graded without recomputing them.

Usage (from the repository root, after ``git archive`` of the s13 runs)::

    python results/reference/prereg_2026-09/scripts/score_prereg.py \
        --run-root /data/s13/runs --out-dir results/reference/prereg_2026-09 \
        --suite-dir output/<stamp>/dot-Identifiability-v1-1.2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ARMS = ("int", "B", "A")
SEEDS = (42, 43, 44)
CHECKPOINT = "do_over_time_pfn_last.pt"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Args:
        path: The file to hash.

    Returns:
        The hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_load_check(path: Path) -> int:
    """Load a checkpoint into a fresh model with strict key matching.

    ``dotime.models.loader.load_dotpfn`` loads with ``strict=False``, which
    would score a partially loaded model without any error if the packaged
    architecture and the checkpoint disagreed.

    Args:
        path: The checkpoint file.

    Returns:
        The number of tensors in the checkpoint's state dict.

    Raises:
        RuntimeError: If keys are missing or unexpected.
    """
    from dotime.models.loader import load_dotpfn

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # load_dotpfn repairs the config (quantile levels) and builds the packaged
    # architecture; the key comparison below is the strict check it skips.
    model = load_dotpfn(path, device="cpu")
    ckpt_keys, model_keys = set(ckpt["model_state_dict"]), set(model.state_dict())
    missing, unexpected = sorted(model_keys - ckpt_keys), sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(f"{path}: missing keys {missing}, unexpected keys {unexpected}")
    return len(ckpt["model_state_dict"])


def find_checkpoint(run_root: Path, tag: str) -> Path | None:
    """Return the checkpoint of a run, preferring a registered ``_r2`` relaunch.

    Args:
        run_root: Directory holding ``checkpoints/<tag>/``.
        tag: The run tag, e.g. ``s13ho_all_int_seed42``.

    Returns:
        The checkpoint path, or ``None`` if neither the run nor its relaunch has one.
    """
    for candidate in (f"{tag}_r2", tag):
        p = run_root / "checkpoints" / candidate / CHECKPOINT
        if p.exists():
            return p
    return None


def load_suite(args):
    """Load the scoring suite from a build directory or from the registry.

    Args:
        args: Parsed command-line arguments.

    Returns:
        ``(episodes, manifest)``, the episodes in file order and the manifest dict.

    Raises:
        SystemExit: If neither ``--suite-dir`` nor a registered version is given.
    """
    from dotime import _release_io
    from dotime.benchmarks import load_benchmark

    if args.suite_dir is not None:
        suite_dir = Path(args.suite_dir)
        episodes = list(_release_io.read_suite(suite_dir))
    else:
        suite = load_benchmark("dot-Identifiability-v1", version=args.version)
        episodes = list(suite)
        from dotime.benchmarks import _cache_root

        suite_dir = _cache_root(None) / f"dot-Identifiability-v1-{suite.meta.version}"
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    return episodes, manifest


def main() -> None:
    """Score every s13 checkpoint and write the predictions and provenance.

    Raises:
        RuntimeError: If a checkpoint fails the strict key check.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suite-dir", type=Path, default=None, help="An unregistered build.")
    ap.add_argument("--version", default="1.2.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--limit", type=int, default=None, help="Episodes per run (smoke only).")
    args = ap.parse_args()

    import dotime
    from dotime.evaluation import query_obs_levels
    from dotime.reference.pfn import PFNRef

    episodes, manifest = load_suite(args)
    if args.limit:
        episodes = episodes[: args.limit]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    onset = np.array([min(ep.intervention.times) for ep in episodes])
    query_idx = np.array([int(ep.query_time_idx[0]) for ep in episodes])
    y_true = np.array([float(ep.y_true[0]) for ep in episodes], dtype=np.float64)
    y_obs = np.array([float(query_obs_levels(ep)[0]) for ep in episodes], dtype=np.float64)
    base = {
        "scm_id": np.array([ep.scm_id for ep in episodes]),
        "structure": np.array([ep.structure for ep in episodes]),
        "onset": onset,
        "query_idx": query_idx,
        "y_true": y_true,
        "y_obs": y_obs,
    }
    rows = []
    provenance = {
        "suite": "dot-Identifiability-v1",
        "suite_version": manifest.get("version"),
        "suite_shard_md5": {s["file"]: s["md5"] for s in manifest.get("shards", [])},
        "n_episodes": len(episodes),
        "checkpoint_file": args.checkpoint,
        "dotime_version": dotime.__version__,
        "dotime_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
        ).stdout.strip(),
        "runs": {},
    }
    for arm in ARMS:
        for seed in SEEDS:
            tag = f"s13ho_all_{arm}_seed{seed}"
            path = find_checkpoint(args.run_root, tag)
            if path is None or path.name != args.checkpoint:
                alt = path.with_name(args.checkpoint) if path is not None else None
                path = alt if alt is not None and alt.exists() else None
            if path is None:
                provenance["runs"][tag] = {"status": "missing"}
                print(f"[{tag}] missing", flush=True)
                continue
            n_keys = strict_load_check(path)
            model = PFNRef(str(path), device=args.device, observational=(arm != "int"))
            t0 = time.time()
            pred = np.array(
                [
                    float(torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1)[0])
                    for ep in episodes
                ],
                dtype=np.float64,
            )
            provenance["runs"][tag] = {
                "status": "ok",
                "path": str(path.relative_to(args.run_root)),
                "sha256": sha256(path),
                "n_keys": n_keys,
                "seconds": round(time.time() - t0, 1),
            }
            print(f"[{tag}] {len(pred)} predictions in {time.time() - t0:.0f} s", flush=True)
            for i in range(len(episodes)):
                rows.append(
                    {
                        "tag": tag,
                        "arm": arm,
                        "seed": seed,
                        **{k: v[i] for k, v in base.items()},
                        "pred": pred[i],
                    }
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in rows]
    )
    pq.write_table(table, args.out_dir / "s13_predictions.parquet")
    (args.out_dir / "s13_scoring_provenance.json").write_text(json.dumps(provenance, indent=1))
    print(f"wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
