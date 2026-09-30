"""Per-lag breakdown of the published dot-Generic-100k CPU baseline rows.

Reviewer o2Tx asked how models fare across lag configurations. The graph sidecar
(``graph_sidecar.py``) gives every episode's SCM family, sampled maximum lag K
and the smallest summed lag from the intervened columns to the queried column,
so the published CPU rows can be split by them.

Gate: with no filter, every baseline must reproduce results/reference/generic.json
(pooled RMSE, dir_acc and dir_n_valid) exactly, with the float32 arithmetic of
``dotime.reference.reference_table.run_baseline`` and episodes in release order.
Only then is the breakdown written. Before scoring, the per-arm target statistics
(nonzero fraction, mean, variance of the observational level, the
interventional level and the effect) are logged and asserted with the checks of
``reference_table.target_qa``.

Each cell reports the query count, pooled RMSE, and direction accuracy scored on
the interventional level and on the effect ``y - y_obs``, each with its number
of scored queries (``|target| >= 0.1``) and binomial standard error, for every
CPU baseline, over all episodes and over the episodes where neither arm is
zeroed. Episodes are cut by SCM family, sampled K, min lag in {0, 1, 2, 3, >=4,
unreachable}, family by min lag, and steps from the end of the intervention
window to the query.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from dotime import _release_io, baselines
from dotime.benchmarks import _SUITE_REGISTRY
from dotime.evaluation import direction_accuracy, query_obs_levels
from dotime.graph_meta import load_graph_sidecar
from dotime.reference.reference_table import CPU_BASELINES, TARGET_QA_MIN_NONZERO, _arm_stats

_REPO = Path(__file__).resolve().parents[4]
_SUITE = "dot-Generic-100k"
_STEP_BINS = (
    (0, 0, "0"),
    (1, 9, "1-9"),
    (10, 49, "10-49"),
    (50, 99, "50-99"),
    (100, None, ">=100"),
)


def _lag_bin(min_lag: int | None) -> str:
    """Bin label of a min lag.

    Args:
        min_lag: Smallest summed lag from treatment to query, ``None`` when the
            query is unreachable.

    Returns:
        ``"0"`` to ``"3"``, ``">=4"`` or ``"unreachable"``.
    """
    if min_lag is None:
        return "unreachable"
    return str(min_lag) if min_lag < 4 else ">=4"


def _step_bin(steps: int) -> str:
    """Bin label of the steps from the window end to the query.

    Args:
        steps: ``query_row - window_end``; 0 means the window is still active.

    Returns:
        One of the ``_STEP_BINS`` labels.
    """
    for lo, hi, label in _STEP_BINS:
        if steps >= lo and (hi is None or steps <= hi):
            return label
    raise ValueError(f"negative steps after the window: {steps}")


def _score_shard(task: tuple[str, str, int]) -> dict:
    """Predict every CPU baseline on one md5-verified release shard.

    Args:
        task: ``(suite_dir, shard_file, expected_md5)``.

    Returns:
        Dict of arrays in release order: ``idx``, ``tgt``, ``obs`` (float32
        levels), ``zero_obs``, ``zero_int`` and ``pred`` of shape
        ``(len(CPU_BASELINES), rows)``.

    Raises:
        ValueError: If the shard fails its md5 check or an episode has more
            than one query.
    """
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    suite_dir, shard_file, md5 = task
    path = Path(suite_dir) / shard_file
    if _release_io._md5(path) != md5:
        raise ValueError(f"md5 mismatch {path}")
    models = [baselines.get(b) for b in CPU_BASELINES]
    encoding = _SUITE_REGISTRY[_SUITE].query_time_encoding
    table = pq.read_table(path)
    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    out: dict[str, list] = {"idx": [], "tgt": [], "obs": [], "zero_obs": [], "zero_int": []}
    preds: list[list[np.ndarray]] = [[] for _ in CPU_BASELINES]
    for r in range(table.num_rows):
        # Read with the suite's declared query_time encoding, as load_benchmark
        # does, so every baseline sees the rows the published run saw.
        ep = _release_io._row_to_episode({c: cols[c][r] for c in cols}, encoding)
        tgt = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).numpy()
        if tgt.size != 1:
            raise ValueError(f"episode {ep.scm_id} has {tgt.size} queries")
        out["idx"].append(ep.scm_id)
        out["tgt"].append(tgt)
        out["obs"].append(query_obs_levels(ep).cpu().numpy())
        out["zero_obs"].append(float(ep.x_obs.abs().max()) == 0.0)
        out["zero_int"].append(float(ep.x_int.abs().max()) == 0.0)
        for j, model in enumerate(models):
            p = torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1).cpu().numpy()
            preds[j].append(p)
    return {
        "idx": np.asarray(out["idx"], dtype=np.int64),
        "tgt": np.concatenate(out["tgt"]),
        "obs": np.concatenate(out["obs"]).astype(np.float32),
        "zero_obs": np.asarray(out["zero_obs"]),
        "zero_int": np.asarray(out["zero_int"]),
        "pred": np.stack([np.concatenate(p) for p in preds]),
    }


def _direction(acc_dict: dict) -> dict:
    """Direction-accuracy entry with a binomial standard error.

    Args:
        acc_dict: Output of ``dotime.evaluation.direction_accuracy``.

    Returns:
        Dict with ``acc``, ``n_valid`` and ``se``; ``None`` instead of NaN when
        nothing is scoreable, so the JSON stays strict.
    """
    n = int(acc_dict["n_valid"])
    if n == 0:
        return {"acc": None, "n_valid": 0, "se": None}
    acc = float(acc_dict["accuracy"])
    return {"acc": acc, "n_valid": n, "se": math.sqrt(acc * (1.0 - acc) / n)}


def _cell(pred: np.ndarray, tgt: np.ndarray, obs: np.ndarray) -> dict:
    """Pooled RMSE and level/effect direction accuracy of one baseline on one cell.

    Args:
        pred: float32 predictions.
        tgt: float32 targets (interventional levels).
        obs: float32 observational levels at the queries.

    Returns:
        Dict with ``rmse``, ``level`` and ``effect``.
    """
    if pred.size == 0:
        return {
            "rmse": None,
            "level": _direction({"n_valid": 0}),
            "effect": _direction({"n_valid": 0}),
        }
    return {
        "rmse": float(np.sqrt(np.mean((pred - tgt) ** 2))),
        "level": _direction(direction_accuracy(torch.from_numpy(pred), torch.from_numpy(tgt))),
        "effect": _direction(
            direction_accuracy(torch.from_numpy(pred - obs), torch.from_numpy(tgt - obs))
        ),
    }


def _target_qa(tgt: np.ndarray, obs: np.ndarray) -> dict:
    """Log and assert per-arm target statistics, as ``reference_table.target_qa``.

    Args:
        tgt: Interventional levels.
        obs: Observational levels.

    Returns:
        ``_arm_stats`` of ``y_obs_level``, ``y_int_level`` and ``effect``.

    Raises:
        RuntimeError: If an arm is non-finite, a level arm has zero variance or
            too few nonzero values, or the effect is zero everywhere.
    """
    arms = {"y_obs_level": obs, "y_int_level": tgt, "effect": tgt - obs}
    stats = {name: _arm_stats(values) for name, values in arms.items()}
    for name, st in stats.items():
        print(
            f"[target QA] {name:11s} n={st['n']} nonzero_frac={st['nonzero_frac']:.4f} "
            f"mean={st['mean']:.4f} var={st['var']:.4f}",
            flush=True,
        )
    problems = [name for name, values in arms.items() if not np.isfinite(values).all()]
    for name in ("y_obs_level", "y_int_level"):
        st = stats[name]
        if st["var"] <= 0.0 or st["nonzero_frac"] < TARGET_QA_MIN_NONZERO:
            problems.append(name)
    if stats["effect"]["nonzero_frac"] == 0.0:
        problems.append("effect")
    if problems:
        raise RuntimeError(f"target QA failed: {problems}")
    return stats


def main() -> None:
    """Gate on the published rows, then write the per-lag breakdown."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--suite-dir", type=Path, default=Path.home() / ".cache" / "dotime" / f"{_SUITE}-1.0.0"
    )
    ap.add_argument(
        "--published", type=Path, default=_REPO / "results" / "reference" / "generic.json"
    )
    ap.add_argument(
        "--sidecar",
        type=Path,
        default=_REPO / "results" / "reference" / "dot-Generic-100k-v1.0.0_graph.jsonl.gz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "generic_lag_breakdown.json",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shards", type=int, default=None, help="first N shards only (smoke run)")
    args = ap.parse_args()
    t0 = time.time()

    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    shards = manifest["shards"][: args.shards]
    tasks = [(str(args.suite_dir), s["file"], s["md5"]) for s in shards]
    with get_context("fork").Pool(args.workers) as pool:
        parts = pool.map(_score_shard, tasks, chunksize=1)
    idx = np.concatenate([p["idx"] for p in parts])
    tgt = np.concatenate([p["tgt"] for p in parts])
    obs = np.concatenate([p["obs"] for p in parts])
    pred = np.concatenate([p["pred"] for p in parts], axis=1)
    zero_either = np.concatenate([p["zero_obs"] | p["zero_int"] for p in parts])
    if not np.array_equal(idx, np.arange(idx.size)):
        raise ValueError("episodes are not in release order")
    print(f"scored {idx.size} episodes in {time.time() - t0:.0f}s", flush=True)
    qa = {
        "all": _target_qa(tgt, obs),
        "drop_either_zeroed": _target_qa(tgt[~zero_either], obs[~zero_either]),
    }

    gate = {}
    published = {r["baseline"]: r for r in json.loads(args.published.read_text())["rows"]}
    for j, name in enumerate(CPU_BASELINES):
        rmse = float(np.sqrt(np.mean((pred[j] - tgt) ** 2)))
        da = direction_accuracy(torch.from_numpy(pred[j]), torch.from_numpy(tgt))
        pub = published[name]
        gate[name] = {
            "pooled_rmse": rmse,
            "dir_acc": da["accuracy"],
            "dir_n_valid": da["n_valid"],
            "reproduces_published": rmse == pub["pooled_rmse"]
            and da["accuracy"] == pub["dir_acc"]
            and da["n_valid"] == pub["dir_n_valid"],
        }
    gate_passed = idx.size == manifest["n_episodes"] and all(
        g["reproduces_published"] for g in gate.values()
    )
    out: dict = {
        "suite": _SUITE,
        "version": manifest["version"],
        "n_episodes": int(idx.size),
        "published": str(args.published.resolve().relative_to(_REPO)),
        "gate_passed": gate_passed,
        "gate": gate,
    }
    if not gate_passed:
        args.out.write_text(json.dumps(out, indent=1) + "\n")
        raise SystemExit(f"gate failed, breakdown not written: {json.dumps(gate)}")

    sidecar = load_graph_sidecar(args.sidecar)
    if sorted(sidecar) != idx.tolist():
        raise ValueError(f"{args.sidecar} does not cover the {idx.size} released episodes")
    unverified = [i for i, rec in sidecar.items() if not rec["verified"]]
    if unverified:
        raise ValueError(f"{len(unverified)} sidecar rows are not verified, e.g. {unverified[:5]}")
    recs = [sidecar[i] for i in idx.tolist()]
    keys = {
        "by_family": [r["family"] for r in recs],
        "by_k_sampled": [str(r["graph"].k_sampled) for r in recs],
        "by_min_lag": [_lag_bin(r["min_lag"]) for r in recs],
        "by_family_x_min_lag": [f"{r['family']}|{_lag_bin(r['min_lag'])}" for r in recs],
        "by_steps_after_window": [_step_bin(r["steps_after_window"]) for r in recs],
    }
    out["target_qa"] = qa
    out["sidecar"] = str(args.sidecar.resolve().relative_to(_REPO))
    out["baselines"] = CPU_BASELINES
    out["min_lag_bins"] = ["0", "1", "2", "3", ">=4", "unreachable"]
    out["steps_after_window_bins"] = [label for _, _, label in _STEP_BINS]
    out["filters"] = {}
    for filt, keep in (("all", np.ones(idx.size, bool)), ("drop_either_zeroed", ~zero_either)):
        tables: dict = {"n_episodes": int(keep.sum())}
        for dim, labels in keys.items():
            lab = np.asarray(labels)
            table = {}
            for value in sorted(set(labels)):
                mask = keep & (lab == value)
                table[value] = {
                    "n_episodes": int(mask.sum()),
                    "rows": {
                        name: _cell(pred[j][mask], tgt[mask], obs[mask])
                        for j, name in enumerate(CPU_BASELINES)
                    },
                }
            tables[dim] = table
        out["filters"][filt] = tables
    out["seconds"] = time.time() - t0
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
