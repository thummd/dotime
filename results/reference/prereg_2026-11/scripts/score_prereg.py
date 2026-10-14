#!/usr/bin/env python
"""Score the s14 checkpoints on dot-Identifiability-v1 for the pre-registration.

This is the scoring step fixed by ``PREREG.md`` of this folder. It loads every
s14 checkpoint with a strict key check, runs the packaged ``PFNRef`` protocol
on every episode of the suite (the observational arms with every intervention
feature zeroed) under the input convention recorded in the checkpoint, and
writes one row per (episode, arm, seed) with the prediction, the target, the
factual level at the query and the effect. For the interventional arms it also
records the prediction at the mirrored do-value ``2 a_ref - v`` and at the
anchor ``a_ref`` (the treatment's factual onset value), which the registered
mechanism probes (S10) read. The analysis is a separate step
(``analyze_prereg.py``), so the predictions are graded once and can be
re-graded without recomputing them.

The script is the s13 scorer (``../../prereg_2026-09/scripts/score_prereg.py``)
with the run series, seeds and arms as options, the probe columns, and the
per-episode treatment statistics that the in-support subset (S11) needs.

Usage (from the repository root, after the nine runs exist)::

    python results/reference/prereg_2026-11/scripts/score_prereg.py \\
        --run-root /data/dennis/s14/runs --out-dir results/reference/prereg_2026-11 \\
        --version 1.2.0 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ARMS = ("int", "B", "A")
SEEDS = (42, 43, 44)
CHECKPOINT = "do_over_time_pfn_last.pt"
SERIES = "s14ho_all"
IN_SUPPORT_SD = 3.0


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

    ``dotime.models.loader.load_dotpfn`` refuses mismatched checkpoints since
    the October 2026 mirror; this repeats the comparison so that the scoring
    record does not depend on that behaviour.

    Args:
        path: The checkpoint file.

    Returns:
        The number of tensors in the checkpoint's state dict.

    Raises:
        RuntimeError: If keys are missing or unexpected.
    """
    from dotime.models.loader import load_dotpfn

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = load_dotpfn(path, device="cpu")
    ckpt_keys, model_keys = set(ckpt["model_state_dict"]), set(model.state_dict())
    missing, unexpected = sorted(model_keys - ckpt_keys), sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(f"{path}: missing keys {missing}, unexpected keys {unexpected}")
    return len(ckpt["model_state_dict"])


def find_checkpoint(run_root: Path, tag: str, checkpoint: str) -> Path | None:
    """Return the checkpoint of a run, preferring a registered ``_r2`` relaunch.

    Args:
        run_root: Directory holding ``checkpoints/<tag>/``.
        tag: The run tag, e.g. ``s14ho_all_int_seed42``.
        checkpoint: The checkpoint file name.

    Returns:
        The checkpoint path, or ``None`` if neither the run nor its relaunch has one.
    """
    for candidate in (f"{tag}_r2", tag):
        p = run_root / "checkpoints" / candidate / checkpoint
        if p.exists():
            return p
    return None


def load_suite(args):
    """Load the scoring suite from a build directory or from the registry.

    Args:
        args: Parsed command-line arguments.

    Returns:
        ``(episodes, manifest)``, the episodes in file order and the manifest dict.
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


def column_stats(x_obs: torch.Tensor, onset: int, col: int) -> tuple[float, float]:
    """Pre-onset mean and standard deviation of one column, as the batch builder uses them.

    Args:
        x_obs: The episode's observational trajectory ``(T, N)``.
        onset: The intervention onset row.
        col: The column.

    Returns:
        ``(mean, std)``; ``(0.0, 1.0)`` when fewer than two pre-onset rows exist.
    """
    pre = x_obs[:onset, col] if onset > 0 else x_obs[:, col]
    if pre.numel() > 1:
        return float(pre.mean().item()), float(pre.std().item())
    return 0.0, 1.0


def predict_at(ref, episode, batch: dict, v_raw: float, mu: float, sd: float) -> float:
    """Level prediction of a loaded ``PFNRef`` at another raw do-value.

    Reuses the batch built for the episode and replaces the do-value under the
    checkpoint's convention, exactly as ``episode_to_batch_interp`` forms it.

    Args:
        ref: The ``PFNRef`` instance.
        episode: The episode (for the queried column).
        batch: The batch built for this episode by ``episode_to_batch_interp``.
        v_raw: The raw do-value to predict at.
        mu: Pre-onset mean of the treatment.
        sd: Pre-onset standard deviation of the treatment.

    Returns:
        The predicted level in raw units.
    """
    centred = bool(ref.convention.get("center_intervention_value", False))
    b = dict(batch)
    value = (v_raw - (mu if centred else 0.0)) / max(sd, 1e-4)
    b["intervention_value"] = torch.tensor([value], dtype=torch.float32, device=ref.device)
    out = ref.model(b)
    head = getattr(ref.model, "quantile_head", None) or getattr(ref.model, "bar_head", None)
    pred_norm = head.predict_mean(out).reshape(-1)
    q = int(episode.query_target[0])
    return float(
        (pred_norm * batch["_norm_stds"][0, q] + batch["_norm_means"][0, q]).reshape(-1)[0]
    )


def main() -> None:
    """Score every s14 checkpoint and write the predictions and provenance.

    Raises:
        RuntimeError: If a checkpoint fails the strict key check, or if the
            do-value rebuilt for the probes does not reproduce the batch's own.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--suite-dir", type=Path, default=None, help="An unregistered build.")
    ap.add_argument("--version", default="1.2.0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--series", default=SERIES, help="Run tag prefix: <series>_<arm>_seed<seed>.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out-prefix", default="s14")
    ap.add_argument("--limit", type=int, default=None, help="Episodes per run (smoke only).")
    args = ap.parse_args()

    import dotime
    from dotime.evaluation import query_obs_levels
    from dotime.reference.pfn import PFNRef, episode_to_batch_interp

    episodes, manifest = load_suite(args)
    if args.limit:
        episodes = episodes[: args.limit]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    onset = np.array([min(ep.intervention.times) for ep in episodes])
    treat = [int(ep.intervention.targets[0]) for ep in episodes]
    query = [int(ep.query_target[0]) for ep in episodes]
    stats_a = [
        column_stats(ep.x_obs, int(o), a) for ep, o, a in zip(episodes, onset, treat, strict=True)
    ]
    stats_y = [
        column_stats(ep.x_obs, int(o), q) for ep, o, q in zip(episodes, onset, query, strict=True)
    ]
    do_value = np.array([float(ep.intervention.values) for ep in episodes], dtype=np.float64)
    a_ref = np.array(
        [float(ep.x_obs[int(o), a]) for ep, o, a in zip(episodes, onset, treat, strict=True)],
        dtype=np.float64,
    )
    mu_a = np.array([s[0] for s in stats_a])
    sd_a = np.array([s[1] for s in stats_a])
    base = {
        "scm_id": np.array([ep.scm_id for ep in episodes]),
        "structure": np.array([ep.structure for ep in episodes]),
        "onset": onset,
        "query_idx": np.array([int(ep.query_time_idx[0]) for ep in episodes]),
        "y_true": np.array([float(ep.y_true[0]) for ep in episodes], dtype=np.float64),
        "y_obs": np.array([float(query_obs_levels(ep)[0]) for ep in episodes], dtype=np.float64),
        "do_value": do_value,
        "a_ref": a_ref,
        "mu_A": mu_a,
        "sd_A": sd_a,
        "mu_Y": np.array([s[0] for s in stats_y]),
        "sd_Y": np.array([s[1] for s in stats_y]),
        "in_support": np.abs(do_value - mu_a) <= IN_SUPPORT_SD * np.maximum(sd_a, 1e-4),
    }
    rows = []
    provenance = {
        "series": args.series,
        "seeds": args.seeds,
        "arms": args.arms,
        "suite": "dot-Identifiability-v1",
        "suite_version": manifest.get("version"),
        "suite_shard_md5": {s["file"]: s["md5"] for s in manifest.get("shards", [])},
        "n_episodes": len(episodes),
        "checkpoint_file": args.checkpoint,
        "in_support_sd": IN_SUPPORT_SD,
        "dotime_version": dotime.__version__,
        "dotime_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
        ).stdout.strip(),
        "torch_version": torch.__version__,
        "device": args.device,
        "host": platform.node(),
        "runs": {},
    }
    for arm in args.arms:
        for seed in args.seeds:
            tag = f"{args.series}_{arm}_seed{seed}"
            path = find_checkpoint(args.run_root, tag, args.checkpoint)
            if path is None:
                provenance["runs"][tag] = {"status": "missing"}
                print(f"[{tag}] missing", flush=True)
                continue
            n_keys = strict_load_check(path)
            ref = PFNRef(str(path), device=args.device, observational=(arm != "int"))
            t0 = time.time()
            pred = np.empty(len(episodes))
            pred_mirror = np.full(len(episodes), np.nan)
            pred_anchor = np.full(len(episodes), np.nan)
            with torch.no_grad():
                for i, ep in enumerate(episodes):
                    pred[i] = float(
                        torch.as_tensor(ref.predict(ep), dtype=torch.float32).reshape(-1)[0]
                    )
                    if arm != "int":
                        continue
                    batch = episode_to_batch_interp(
                        ep, ref.n_max, ref.device, False, ref.convention
                    )
                    mu, sd = stats_a[i]
                    # The probe path must reproduce the registered path at the episode's own do-value.
                    same = predict_at(ref, ep, batch, do_value[i], mu, sd)
                    if not np.isclose(same, pred[i], atol=1e-5, rtol=0):
                        raise RuntimeError(
                            f"{tag} scm {ep.scm_id}: probe path {same!r} != predict {pred[i]!r}"
                        )
                    pred_mirror[i] = predict_at(ref, ep, batch, 2 * a_ref[i] - do_value[i], mu, sd)
                    pred_anchor[i] = predict_at(ref, ep, batch, a_ref[i], mu, sd)
            cfg = ref.model
            provenance["runs"][tag] = {
                "status": "ok",
                "path": str(path.relative_to(args.run_root)),
                "sha256": sha256(path),
                "n_keys": n_keys,
                "seconds": round(time.time() - t0, 1),
                "input_convention": ref.convention,
                "model_options": {
                    "query_a_ref": bool(getattr(cfg, "query_a_ref", False)),
                    "stat_tokens": getattr(cfg, "stat_tokens", "none"),
                    "role_embed": getattr(cfg, "role_embed", None) is not None,
                    "joint_branch": bool(getattr(cfg, "joint_branch", False)),
                },
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
                        "pred_mirror": pred_mirror[i],
                        "pred_anchor": pred_anchor[i],
                    }
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(
        [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in rows]
    )
    pq.write_table(table, args.out_dir / f"{args.out_prefix}_predictions.parquet")
    (args.out_dir / f"{args.out_prefix}_scoring_provenance.json").write_text(
        json.dumps(provenance, indent=1)
    )
    print(f"wrote {len(rows)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
