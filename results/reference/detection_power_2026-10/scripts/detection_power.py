#!/usr/bin/env python
"""Which estimator classes dot-Identifiability-v1 separates, structure by structure.

Scores naive, association, do-SVAR, identification-aware, graph-routed and
oracle estimators (see ``estimators.py``) with the official effect-sign
protocol, plus level RMSE, level-sign accuracy, the false-effect rate on
null-effect structures, and paired-bootstrap separations. Run from the
repository root::

    python results/reference/detection_power_2026-10/scripts/detection_power.py --self-test
    python results/reference/detection_power_2026-10/scripts/detection_power.py --version 1.1.0
    python results/reference/detection_power_2026-10/scripts/detection_power.py --version 1.2.0

The gates run in order and are asserted: a 1.1.0 run first reproduces the
released CPU reference rows exactly (gate 1, ``ident_v1_1_gate.json``);
``--self-test`` checks every estimator on synthetic linear Gaussian data
(gate 2, ``self_test.json``); a run on a later version requires both and then
checks that every structure other than ``mediator`` scores exactly as on
1.1.0, with the oracle at 1 (gate 3). A suite version that is built but not
registered yet loads with ``--suite-dir <build>/dot-Identifiability-v1-<version>``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import subprocess
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import estimators as est
import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

import dotime
from dotime import _release_io
from dotime.baselines import _canonical_summary_graph
from dotime.benchmarks import _SUITE_REGISTRY, Episode, load_benchmark
from dotime.evaluation import DIR_ACC_EPS, direction_accuracy, query_obs_levels
from dotime.interventions import InterventionSpec, InterventionType
from dotime.reference.reference_table import _arm_stats, target_qa

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ANALYSIS_DIR.parents[2]
SUITE = "dot-Identifiability-v1"
GATE1_VERSION = "1.1.0"
REF_EFFECT = REPO_ROOT / "results/reference/v1_1/ident_cpu_effect_backdoor_fix.json"
REF_LEVEL = REPO_ROOT / "results/reference/v1_1/ident_cpu_level_backdoor_fix.json"
BOOT_SEED = 20261001
N_BOOT = 2000
# A predicted effect below the direction-accuracy threshold is not a call.
FALSE_EFFECT_THRESHOLD = DIR_ACC_EPS
# Structures whose query moves between 1.1.0 and 1.2.0 (mediator: offset 0 -> 1).
GATE3_CHANGED = ("mediator",)

# Pre-registered in README.md: the estimators whose identification assumptions
# hold on each structure's summary graph. do-SVAR is valid only where the
# canonical column order is causal and nothing hidden confounds A and Y.
VALID = {
    "bi_variate": ("NaiveOLS", "do-SVAR"),
    "back_door": ("BackDoorOLS",),
    "observed_confounder": ("BackDoorOLS",),
    "mediator": ("NaiveOLS", "do-SVAR", "FrontDoorOLS"),
    "front_door": ("FrontDoorOLS",),
    "confounder_mediator": ("BackDoorOLS", "FrontDoorOLS"),
    "instrumental_variable": ("IV2SLS",),
    "unobserved_confounder": (),
    "bow_graph": (),
}

# Pre-registered cross-structure contrast: bow_graph is bi_variate plus the
# hidden confounder U, so the accuracy an estimator loses between the two is
# what confounding by U costs it.
CROSS_STRUCTURE = (("bi_variate", "bow_graph"),)
# Thresholds of the pre-registered predictions (README, "Pre-registration").
NAIVE_BAND = (0.40, 0.60)
MIN_N_VALID = 100
MIN_FALSE_EFFECT = 0.10

SELF_TEST_T = 20_000
SELF_TEST_SEED = 20261001
RECOVER_TOL = 0.05
BIAS_MIN = 0.2


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _finite(x: float) -> float | None:
    """JSON-safe float: ``None`` for NaN.

    Args:
        x: A float.

    Returns:
        ``x``, or ``None`` if it is NaN.
    """
    return None if x != x else float(x)


def _binomial_se(p: float | None, n: int) -> float | None:
    """Binomial standard error of a proportion.

    Args:
        p: The proportion, or ``None``.
        n: Its denominator.

    Returns:
        ``sqrt(p (1 - p) / n)``, or ``None`` when undefined.
    """
    if p is None or n <= 0:
        return None
    return math.sqrt(p * (1.0 - p) / n)


def _write_json(path: Path, obj: Any) -> None:
    """Write strict JSON (NaN is refused, so undefined values must be ``None``).

    Args:
        path: Destination.
        obj: JSON-serialisable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    print(f"wrote {path}", flush=True)


def output_tag(suite: str, version: str) -> str:
    """File-name stem of a run's outputs.

    Args:
        suite: Suite name.
        version: Suite version.

    Returns:
        ``ident_v1_1`` for dot-Identifiability-v1 1.1.0, ``ident_v1_2`` for 1.2.0,
        otherwise ``<suite>_<version>`` with dots replaced.
    """
    if suite == SUITE:
        major, minor = version.split(".")[:2]
        return f"ident_v{major}_{minor}"
    return f"{suite}_{version}".replace(".", "_")


def code_provenance() -> dict[str, Any]:
    """Commit and cleanliness of the code that produced an output.

    Returns:
        Git commit, whether ``src`` or the analysis scripts differ from it,
        the package version, and whether NaiveOLS is registered.
    """
    scripts = str((ANALYSIS_DIR / "scripts").relative_to(REPO_ROOT))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", scripts],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, None
    return {
        "git_commit": commit,
        "code_dirty": None if status is None else bool(status),
        "dotime_version": dotime.__version__,
        "naive_ols_registered": "NaiveOLS" in dotime.baselines.available(),
    }


def _check_dotime_source() -> None:
    """Refuse to run against a ``dotime`` other than this checkout's.

    Raises:
        RuntimeError: If ``dotime`` is imported from elsewhere, e.g. the main
            checkout that an editable install points at from a worktree.
    """
    here = (REPO_ROOT / "src").resolve()
    if not Path(dotime.__file__).resolve().is_relative_to(here):
        raise RuntimeError(
            f"dotime is imported from {Path(dotime.__file__).parent}, not {here}; "
            f"run with PYTHONPATH={here}"
        )


# --------------------------------------------------------------------------- #
# Loading and target QA
# --------------------------------------------------------------------------- #


def load_episodes(
    suite: str,
    version: str,
    cache_dir: Path | None,
    suite_dir: Path | None,
    limit_per_structure: int | None,
) -> list[Episode]:
    """Load a suite's episodes and check they are what the analysis assumes.

    Args:
        suite: Suite name.
        version: Suite version.
        cache_dir: Cache root for :func:`~dotime.benchmarks.load_benchmark`.
        suite_dir: A local build directory to read instead of the registry,
            for a version that is built but not registered yet.
        limit_per_structure: Keep only the first episodes of each structure.

    Returns:
        The episodes, in suite order.

    Raises:
        ValueError: If the suite is the misaligned 1.0.0 release, a fallback
            suite, incomplete, or has episodes the estimators cannot take.
    """
    if suite == SUITE and version == "1.0.0":
        raise ValueError("1.0.0 ships x_obs in topological order; score 1.1.0 or later")
    if suite_dir is not None:
        manifest = json.loads((suite_dir / "manifest.json").read_text())
        if (manifest["name"], manifest["version"]) != (suite, version):
            raise ValueError(
                f"{suite_dir.name} holds {manifest['name']} {manifest['version']}, "
                f"not {suite} {version}"
            )
        meta = dataclasses.replace(
            _SUITE_REGISTRY[suite],
            version=version,
            n_episodes=int(manifest["n_episodes"]),
            structures=tuple(manifest["structures"]),
        )
        loaded = _release_io.read_suite(meta, suite_dir)
    else:
        loaded = load_benchmark(suite, version=version, cache_dir=cache_dir)
    episodes = list(loaded)
    if loaded.meta.version != version:
        raise ValueError(f"asked for {suite} {version}, loaded {loaded.meta.version}")
    # load_benchmark silently generates a tiny stand-in when a registered
    # version has no files; scoring that would be scoring noise.
    if any(ep.metadata.get("fallback") for ep in episodes):
        raise ValueError(f"{suite} {version} has no files; the loader generated a stand-in")
    if len(episodes) != loaded.meta.n_episodes:
        raise ValueError(f"{len(episodes)} episodes, the metadata says {loaded.meta.n_episodes}")
    for ep in episodes:
        if ep.structure is None:
            raise ValueError(f"episode {ep.scm_id} has no structure label")
        est.do_value(ep)
        est.query_col(ep)
        if int(ep.query_time_idx.reshape(-1)[0]) < est.onset_row(ep):
            raise ValueError(f"episode {ep.scm_id} queries before its onset")
    if limit_per_structure is not None:
        seen: dict[str, int] = {}
        kept = []
        for ep in episodes:
            s = str(ep.structure)
            if seen.get(s, 0) < limit_per_structure:
                kept.append(ep)
                seen[s] = seen.get(s, 0) + 1
        episodes = kept
    return episodes


@dataclasses.dataclass
class EpisodeColumns:
    """Per-episode fields every metric needs, as aligned numpy arrays.

    Attributes:
        scm_id: Episode ids.
        structure: Structure labels.
        onset: Intervened row.
        query_idx: Query row.
        v: Do-value of the episode.
        a_ref: Factual treatment at the intervened row.
        y_true: Interventional level at the query, float32 as scored.
        y_obs: Factual level at the query, float32 as scored.
    """

    scm_id: np.ndarray
    structure: np.ndarray
    onset: np.ndarray
    query_idx: np.ndarray
    v: np.ndarray
    a_ref: np.ndarray
    y_true: np.ndarray
    y_obs: np.ndarray

    @classmethod
    def from_episodes(cls, episodes: Sequence[Episode]) -> EpisodeColumns:
        """Extract the columns, with the float32 targets of ``run_baseline``.

        Args:
            episodes: The scored episodes.

        Returns:
            The aligned columns.
        """
        return cls(
            scm_id=np.array([int(ep.scm_id or 0) for ep in episodes], dtype=np.int64),
            structure=np.array([ep.structure for ep in episodes], dtype=object),
            onset=np.array([est.onset_row(ep) for ep in episodes], dtype=np.int64),
            query_idx=np.array(
                [int(ep.query_time_idx.reshape(-1)[0]) for ep in episodes], dtype=np.int64
            ),
            v=np.array([est.do_value(ep) for ep in episodes], dtype=np.float64),
            a_ref=np.array([est.reference_do_value(ep) for ep in episodes], dtype=np.float64),
            y_true=np.concatenate(
                [
                    torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).numpy()
                    for ep in episodes
                ]
            ),
            y_obs=np.concatenate([query_obs_levels(ep).cpu().numpy() for ep in episodes]),
        )

    def structures(self) -> list[str]:
        """Structure labels in order of first appearance.

        Returns:
            The distinct labels.
        """
        return list(dict.fromkeys(self.structure.tolist()))


def structure_qa(cols: EpisodeColumns) -> dict[str, dict[str, Any]]:
    """Per-structure target statistics, logged and asserted before any scoring.

    Seeds guard against variance, not against a corrupted target, so every run
    records what each structure is scored against. The level arms must be
    mostly nonzero with positive variance. The effect arm may be all zero: the
    null-effect structures are zero by design, and ``mediator`` queried at the
    onset is zero because its only path from A to Y is lagged.

    Args:
        cols: The episode columns.

    Returns:
        ``{structure: {arm: stats, ...}}`` with the exactly-zero effect
        fraction and the query offsets.

    Raises:
        RuntimeError: If a level arm fails the floor on some structure.
    """
    out: dict[str, dict[str, Any]] = {}
    problems = []
    for s in cols.structures():
        m = cols.structure == s
        eff = cols.y_true[m] - cols.y_obs[m]
        arms = {"y_obs_level": cols.y_obs[m], "y_int_level": cols.y_true[m], "effect": eff}
        stats: dict[str, Any] = {name: _arm_stats(values) for name, values in arms.items()}
        stats["exactly_zero_effect_frac"] = float(np.mean(eff == 0.0))
        stats["query_offsets"] = sorted(set((cols.query_idx[m] - cols.onset[m]).tolist()))
        for name in ("y_obs_level", "y_int_level"):
            st = stats[name]
            if st["var"] <= 0.0 or st["nonzero_frac"] < 0.5:
                problems.append(f"{s}/{name} {st}")
        print(
            f"[structure QA] {s:22s} n={int(m.sum())} "
            + " ".join(
                f"{k}:nz={v['nonzero_frac']:.3f},mean={v['mean']:.3f},var={v['var']:.3f}"
                for k, v in stats.items()
                if isinstance(v, dict)
            )
            + f" zero_effect={stats['exactly_zero_effect_frac']:.3f} offsets={stats['query_offsets']}",
            flush=True,
        )
        out[s] = stats
    if problems:
        raise RuntimeError(f"structure QA failed: {problems}")
    return out


# --------------------------------------------------------------------------- #
# Prediction (parallel over episode chunks)
# --------------------------------------------------------------------------- #

_WORKER_ESTIMATORS: dict[str, est.Estimator | None] = {}


def _init_worker(names: tuple[str, ...]) -> None:
    """Build the estimators once per worker process.

    Args:
        names: Estimators to build.
    """
    # One core per worker: the pool provides the parallelism.
    torch.set_num_threads(1)
    _WORKER_ESTIMATORS.clear()
    _WORKER_ESTIMATORS.update(est.build_estimators(names))


def _predict_chunk(episodes: list[Episode]) -> dict[str, np.ndarray]:
    """Predict every estimator at ``v`` and at ``a_ref`` on a chunk of episodes.

    Args:
        episodes: The chunk.

    Returns:
        ``{name: (len(episodes), 2) float32}`` with the level at ``do(v)`` and
        at ``do(a_ref)``; NaN where the estimator is pending.
    """
    out = {}
    for name, model in _WORKER_ESTIMATORS.items():
        arr = np.full((len(episodes), 2), np.nan, dtype=np.float32)
        if model is not None:
            for i, ep in enumerate(episodes):
                if isinstance(model, est.GraphRouter) and model.is_pending(ep.structure):
                    continue
                arr[i, 0] = model.predict(ep, est.do_value(ep))
                arr[i, 1] = model.predict(ep, est.reference_do_value(ep))
        out[name] = arr
    return out


def predict_all(
    episodes: list[Episode], names: Sequence[str], workers: int
) -> dict[str, np.ndarray]:
    """Predict every estimator on every episode.

    Per-episode predictions are independent, so the result does not depend on
    the worker count or chunking.

    Args:
        episodes: The scored episodes.
        names: Estimators to run.
        workers: Worker processes; ``<= 1`` runs in this process.

    Returns:
        ``{name: (n_episodes, 2) float32}``, see :func:`_predict_chunk`.
    """
    if workers <= 1:
        _init_worker(tuple(names))
        parts = [_predict_chunk(episodes)]
    else:
        # Workers must not multiply threads: numpy and torch read these at
        # import, and spawned children inherit this environment.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"
        size = max(1, len(episodes) // (workers * 8))
        chunks = [episodes[i : i + size] for i in range(0, len(episodes), size)]
        # Spawn rather than fork: pyarrow has started threads in this process.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(tuple(names),)
        ) as pool:
            parts = list(pool.map(_predict_chunk, chunks))
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def cell_metrics(
    pred: np.ndarray, pred_ref: np.ndarray, y_true: np.ndarray, y_obs: np.ndarray
) -> dict[str, Any]:
    """Every metric of one estimator on one group of episodes.

    All arithmetic is float32, as in ``reference_table.run_baseline``, so the
    pooled numbers reproduce the released rows bit for bit.

    Args:
        pred: Predicted level at ``do(v)``.
        pred_ref: Predicted level at ``do(a_ref)``.
        y_true: Interventional level.
        y_obs: Factual level.

    Returns:
        Effect-sign accuracy with ``n_valid`` and binomial SE, level RMSE,
        level-sign accuracy, the rate of predicted effects of at least the
        threshold (the false-effect rate on null structures), and the fraction
        of episodes whose prediction ignores the do-value. The exploratory
        ``contrast_*`` keys score the sign of the estimator's own effect,
        ``pred - pred_ref``, on the same valid episodes: over all of them (an
        abstention, a zero contrast, counts as wrong) and over its calls.
        The exploratory ``effect_rmse`` is the RMSE of that contrast against
        the true effect over every episode, null ones included.
    """
    eff = direction_accuracy(torch.from_numpy(pred - y_obs), torch.from_numpy(y_true - y_obs))
    lvl = direction_accuracy(torch.from_numpy(pred), torch.from_numpy(y_true))
    calls = np.abs(pred - pred_ref) >= FALSE_EFFECT_THRESHOLD
    e_acc, l_acc = _finite(eff["accuracy"]), _finite(lvl["accuracy"])
    call_rate = float(np.mean(calls))
    # The official score charges the level forecast too: an estimator that
    # knows the effect still misses when |mean(Y_pre) - y_obs| exceeds it. The
    # contrast drops that level term and keeps only the effect direction.
    contrast = pred - pred_ref
    con = direction_accuracy(torch.from_numpy(contrast), torch.from_numpy(y_true - y_obs))
    made = contrast != 0.0
    con_calls = direction_accuracy(
        torch.from_numpy(contrast[made]), torch.from_numpy((y_true - y_obs)[made])
    )
    c_acc, cc_acc = _finite(con["accuracy"]), _finite(con_calls["accuracy"])
    return {
        "n_episodes": int(pred.size),
        "effect_sign_acc": e_acc,
        "effect_n_valid": int(eff["n_valid"]),
        "effect_sign_se": _binomial_se(e_acc, int(eff["n_valid"])),
        "level_rmse": float(np.sqrt(np.mean((pred - y_true) ** 2))),
        "level_sign_acc": l_acc,
        "level_n_valid": int(lvl["n_valid"]),
        "effect_call_rate": call_rate,
        "effect_call_se": _binomial_se(call_rate, int(pred.size)),
        "v_invariant_frac": float(np.mean(pred == pred_ref)),
        "contrast_sign_acc": c_acc,
        "contrast_sign_se": _binomial_se(c_acc, int(con["n_valid"])),
        "contrast_n_calls": int(con_calls["n_valid"]),
        "contrast_sign_acc_calls": cc_acc,
        "contrast_sign_se_calls": _binomial_se(cc_acc, int(con_calls["n_valid"])),
        "effect_rmse": float(np.sqrt(np.mean((contrast - (y_true - y_obs)) ** 2))),
    }


def sign_correct(
    pred: np.ndarray, y_true: np.ndarray, y_obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode effect-sign validity and correctness, as direction_accuracy scores them.

    Args:
        pred: Predicted level at ``do(v)``.
        y_true: Interventional level.
        y_obs: Factual level.

    Returns:
        ``(valid, correct)`` boolean arrays over the episodes.
    """
    p = torch.from_numpy(pred - y_obs)
    t = torch.from_numpy(y_true - y_obs)
    return (t.abs() >= DIR_ACC_EPS).numpy(), (p.sign() == t.sign()).numpy()


def _ci(samples: np.ndarray) -> list[float]:
    """Percentile 95% interval.

    Args:
        samples: Bootstrap replicates.

    Returns:
        ``[2.5%, 97.5%]`` quantiles.
    """
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return [float(lo), float(hi)]


def structure_contrasts(
    correct: dict[str, np.ndarray],
    naive: Sequence[str],
    id_aware: Sequence[str],
    others: Sequence[str],
    router: str | None,
    router_peers: Sequence[str],
) -> dict[str, Any]:
    """Paired episode-bootstrap contrasts of effect-sign accuracy on one structure.

    One resample of the structure's valid episodes (``default_rng(BOOT_SEED)``,
    :data:`N_BOOT` replicates) is shared by every estimator, so differences are
    paired. A fresh generator per structure keeps each structure's intervals
    independent of which other structures the suite contains. Maxima are taken
    inside every replicate, so the intervals include the selection of the best
    estimator of each class.

    Args:
        correct: Per-episode correctness on the valid episodes, by estimator.
        naive: Naive estimators.
        id_aware: Identification-aware estimators that apply to the structure.
        others: Further estimators to contrast with the best naive one.
        router: The router, if it is scored on this structure.
        router_peers: Valid estimators to contrast the router with.

    Returns:
        ``separation`` (best identification-aware minus best naive),
        ``vs_best_naive`` per estimator and ``router_vs_valid`` per peer, each
        with the point difference and its 95% interval.
    """
    n = len(next(iter(correct.values())))
    if not naive:
        return {"n_valid": n, "reason": "no naive estimator scored"}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    point = {k: float(c.mean()) for k, c in correct.items()}
    boot = {k: c[idx].mean(axis=1) for k, c in correct.items()}
    best_naive = max(naive, key=lambda k: point[k])
    naive_boot = np.max([boot[k] for k in naive], axis=0)
    out: dict[str, Any] = {"n_valid": n}
    if id_aware:
        best_id = max(id_aware, key=lambda k: point[k])
        diff = np.max([boot[k] for k in id_aware], axis=0) - naive_boot
        out["separation"] = {
            "best_identification_aware": best_id,
            "best_naive": best_naive,
            "diff": point[best_id] - point[best_naive],
            "ci95": _ci(diff),
        }
    else:
        out["separation"] = {"diff": None, "reason": "no identification-aware estimator applies"}
    out["vs_best_naive"] = {
        k: {"diff": point[k] - point[best_naive], "ci95": _ci(boot[k] - naive_boot)} for k in others
    }
    out["router_vs_valid"] = (
        {
            k: {"diff": point[router] - point[k], "ci95": _ci(boot[router] - boot[k])}
            for k in router_peers
        }
        if router is not None
        else {}
    )
    return out


def score(
    cols: EpisodeColumns,
    preds: dict[str, np.ndarray],
    models: dict[str, est.Estimator | None],
) -> dict[str, Any]:
    """Score every estimator per structure and pooled, with the structure contrasts.

    Args:
        cols: The episode columns.
        preds: Predictions from :func:`predict_all`.
        models: The estimators, ``None`` where pending.

    Returns:
        ``{"structures": ..., "estimators": ..., "contrasts": ...}``.
    """
    structures = cols.structures()
    info: dict[str, dict[str, Any]] = {}
    for s in structures:
        _names, summary, _hidden = _canonical_summary_graph(s)
        m = cols.structure == s
        zero = bool(np.all(cols.y_true[m] - cols.y_obs[m] == 0.0))
        dag_null = not nx.has_path(summary, "A", "Y")
        r = est.route(s)
        info[s] = {
            "n_episodes": int(m.sum()),
            "null_effect": dag_null or zero,
            "null_reason": "dag" if dag_null else ("query offset" if zero else None),
            "route": {"estimator": r.estimator, "rule": r.rule, "identified": r.identified},
            "valid_estimators": list(VALID.get(s, ())),
        }

    table: dict[str, Any] = {}
    for name, model in models.items():
        row: dict[str, Any] = {"class": est.CLASS_OF[name], "per_structure": {}}
        if model is None:
            row.update(status="pending", pooled=None)
            row["per_structure"] = {s: {"status": "pending"} for s in structures}
            table[name] = row
            continue
        p = preds[name]
        pending_s = [
            s for s in structures if isinstance(model, est.GraphRouter) and model.is_pending(s)
        ]
        for s in structures:
            if s in pending_s:
                row["per_structure"][s] = {"status": "pending"}
                continue
            m = cols.structure == s
            cell = cell_metrics(p[m, 0], p[m, 1], cols.y_true[m], cols.y_obs[m])
            uses_v = model.uses_do_value(s)
            if isinstance(model, est.GraphRouter) and est.route(s).estimator is None:
                label = "rule 3"
            else:
                label = "estimated" if uses_v else "trivial"
            cell.update(status="ok", uses_do_value=uses_v, false_effect_label=label)
            if not uses_v and cell["v_invariant_frac"] != 1.0:
                raise AssertionError(f"{name} declines {s} but its prediction moves with v")
            row["per_structure"][s] = cell
        if pending_s:
            row.update(status="partial", pooled=None, pending_structures=pending_s)
        else:
            row.update(status="ok", pooled=cell_metrics(p[:, 0], p[:, 1], cols.y_true, cols.y_obs))
        table[name] = row

    contrasts: dict[str, Any] = {}
    correct_by_s: dict[str, dict[str, np.ndarray]] = {}
    for s in structures:
        m = cols.structure == s
        ok = [k for k in models if table[k]["per_structure"][s].get("status") == "ok"]
        valid = None
        correct: dict[str, np.ndarray] = {}
        for k in ok:
            v_mask, c = sign_correct(preds[k][m, 0], cols.y_true[m], cols.y_obs[m])
            valid = v_mask if valid is None else valid
            correct[k] = c[valid]
        correct_by_s[s] = correct
        if valid is None or not valid.any():
            contrasts[s] = {"n_valid": 0, "reason": "no episode has |effect| >= 0.1"}
            continue
        naive = [k for k in est.NAIVE if k in correct]
        id_aware = [
            k
            for k in est.IDENTIFICATION_AWARE
            if k in correct and table[k]["per_structure"][s]["uses_do_value"]
        ]
        others = [k for k in correct if k not in est.NAIVE and k != "Oracle"]
        router = "GraphRouter" if "GraphRouter" in correct else None
        routed = est.route(s).estimator
        peers = [k for k in VALID.get(s, ()) if k in correct and k != routed]
        contrasts[s] = structure_contrasts(correct, naive, id_aware, others, router, peers)
        contrasts[s]["router_route"] = routed
    cross = {
        f"{a} - {b}": cross_structure_contrast(correct_by_s[a], correct_by_s[b])
        for a, b in CROSS_STRUCTURE
        if correct_by_s.get(a)
        and correct_by_s.get(b)
        and len(next(iter(correct_by_s[a].values()))) > 0
        and len(next(iter(correct_by_s[b].values()))) > 0
    }
    return {
        "structures": info,
        "estimators": table,
        "contrasts": contrasts,
        "cross_structure": cross,
    }


def cross_structure_contrast(
    correct_a: dict[str, np.ndarray], correct_b: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Effect-sign accuracy on structure a minus structure b, per estimator.

    The two structures hold different episodes, so the bootstrap resamples
    each structure's valid episodes independently (one generator seeded with
    :data:`BOOT_SEED`, structure a drawn first).

    Args:
        correct_a: Per-episode correctness on the valid episodes of a.
        correct_b: The same for b.

    Returns:
        ``{estimator: {"diff", "ci95"}}`` for the estimators scored on both.
    """
    names = [k for k in correct_a if k in correct_b]
    n_a, n_b = len(correct_a[names[0]]), len(correct_b[names[0]])
    rng = np.random.default_rng(BOOT_SEED)
    idx_a = rng.integers(0, n_a, size=(N_BOOT, n_a))
    idx_b = rng.integers(0, n_b, size=(N_BOOT, n_b))
    out = {}
    for k in names:
        boot = correct_a[k][idx_a].mean(axis=1) - correct_b[k][idx_b].mean(axis=1)
        out[k] = {
            "diff": float(correct_a[k].mean() - correct_b[k].mean()),
            "ci95": _ci(boot),
            "n_valid": [n_a, n_b],
        }
    return out


# --------------------------------------------------------------------------- #
# Pre-registered predictions
# --------------------------------------------------------------------------- #


def _cell(result: dict[str, Any], name: str, structure: str) -> dict[str, Any]:
    """One scored cell, or ``{}`` when the estimator or structure is absent.

    Args:
        result: The analysis JSON.
        name: Estimator.
        structure: Structure.

    Returns:
        The cell dict.
    """
    return result["estimators"].get(name, {}).get("per_structure", {}).get(structure, {})


def _verdict(parts: list[dict[str, Any]]) -> str:
    """Combine sub-checks into one outcome.

    Args:
        parts: Sub-checks with ``holds`` set to ``True``, ``False`` or ``None``
            (not evaluable, e.g. pending).

    Returns:
        ``not confirmed`` if any evaluable part fails, ``confirmed`` if every
        evaluable part holds, ``not evaluable`` if none is evaluable.
    """
    held = [p["holds"] for p in parts if p["holds"] is not None]
    if not held:
        return "not evaluable"
    return "confirmed" if all(held) else "not confirmed"


def evaluate_predictions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Grade the pre-registered predictions (README, "Pre-registration") mechanically.

    The criteria below are the operational form of the README statements and
    were committed with them, before any run on 1.2.0.

    Args:
        result: The analysis JSON of one run.

    Returns:
        One record per prediction with its sub-checks and outcome.
    """
    structures = result["structures"]
    contrasts = result["contrasts"]

    def n_valid(s: str) -> int:
        """Episodes of ``s`` with a scorable effect sign.

        Args:
            s: Structure.

        Returns:
            Its ``n_valid``, 0 if absent.
        """
        return int(contrasts.get(s, {}).get("n_valid", 0))

    def evaluable(s: str) -> bool:
        """Whether ``s`` has enough valid episodes to grade a prediction.

        Args:
            s: Structure.

        Returns:
            ``True`` if present with at least :data:`MIN_N_VALID` valid episodes.
        """
        return s in structures and n_valid(s) >= MIN_N_VALID

    out: list[dict[str, Any]] = []

    parts = []
    for k in est.NAIVE:
        for s in structures:
            c = _cell(result, k, s)
            if c.get("status") == "ok" and c["effect_n_valid"] >= MIN_N_VALID:
                acc = c["effect_sign_acc"]
                parts.append(
                    {
                        "estimator": k,
                        "structure": s,
                        "value": acc,
                        "holds": NAIVE_BAND[0] <= acc <= NAIVE_BAND[1],
                    }
                )
    out.append(
        {
            "id": "P1",
            "statement": f"every naive effect-sign accuracy lies in [{NAIVE_BAND[0]}, {NAIVE_BAND[1]}]",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    for s in ("back_door", "confounder_mediator", "instrumental_variable", "front_door"):
        sep = contrasts.get(s, {}).get("separation") or {}
        if not evaluable(s) or sep.get("diff") is None:
            parts.append({"structure": s, "holds": None})
            continue
        parts.append(
            {
                "structure": s,
                "best_identification_aware": sep["best_identification_aware"],
                "diff": sep["diff"],
                "ci95": sep["ci95"],
                "holds": sep["ci95"][0] > 0,
            }
        )
    for s in ("bi_variate", "unobserved_confounder", "bow_graph"):
        if s in structures:
            uses = [k for k in est.IDENTIFICATION_AWARE if _cell(result, k, s).get("uses_do_value")]
            parts.append(
                {"structure": s, "applicable_identification_aware": uses, "holds": not uses}
            )
    out.append(
        {
            "id": "P2",
            "statement": "the best identification-aware estimator beats the best naive one (95% CI above 0) "
            "on back_door, confounder_mediator, instrumental_variable and front_door, and none "
            "applies on bi_variate, unobserved_confounder or bow_graph",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    for s in structures:
        peers = contrasts.get(s, {}).get("router_vs_valid", {})
        for k, d in peers.items():
            parts.append(
                {
                    "structure": s,
                    "peer": k,
                    "diff": d["diff"],
                    "ci95": d["ci95"],
                    "anticipated_exception": (s, k)
                    in (("mediator", "do-SVAR"), ("confounder_mediator", "FrontDoorOLS")),
                    "holds": d["ci95"][1] >= 0 if evaluable(s) else None,
                }
            )
    out.append(
        {
            "id": "P3",
            "statement": "the GraphRouter is not significantly worse (95% CI upper bound >= 0) than any other "
            "valid estimator on any structure",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    s = "unobserved_confounder"
    if s in structures:
        c = _cell(result, "GraphRouter", s)
        parts.append(
            {
                "estimator": "GraphRouter",
                "value": c.get("effect_call_rate"),
                "label": c.get("false_effect_label"),
                "holds": c.get("effect_call_rate") == 0.0 if c.get("status") == "ok" else None,
            }
        )
        for k in ("NaiveOLS", "do-SVAR"):
            c = _cell(result, k, s)
            ok = c.get("status") == "ok"
            parts.append(
                {
                    "estimator": k,
                    "value": c.get("effect_call_rate"),
                    "holds": c["effect_call_rate"] >= MIN_FALSE_EFFECT if ok else None,
                }
            )
    out.append(
        {
            "id": "P4",
            "statement": "on unobserved_confounder the router's tau = 0 has a false-effect rate of exactly 0, "
            f"while NaiveOLS and do-SVAR exceed {MIN_FALSE_EFFECT}",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    s = "bow_graph"
    if s in structures:
        sep = contrasts.get(s, {}).get("separation") or {}
        parts.append(
            {"check": "no identification-aware estimator applies", "holds": sep.get("diff") is None}
        )
        parts.append(
            {"check": "route is not identified", "holds": not structures[s]["route"]["identified"]}
        )
        cross = result.get("cross_structure", {}).get("bi_variate - bow_graph", {})
        for k in ("NaiveOLS", "do-SVAR"):
            d = cross.get(k)
            parts.append(
                {
                    "check": f"{k} loses accuracy from bi_variate to bow_graph",
                    "diff": None if d is None else d["diff"],
                    "ci95": None if d is None else d["ci95"],
                    "holds": None if d is None else d["ci95"][0] > 0,
                }
            )
    out.append(
        {
            "id": "P5",
            "statement": "on bow_graph no identification-aware estimator applies, the router reports no "
            "identification, and NaiveOLS and do-SVAR lose effect-sign accuracy relative to bi_variate "
            "(95% CI of the difference above 0)",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    s = "mediator"
    d = contrasts.get(s, {}).get("router_vs_valid", {}).get("do-SVAR")
    if d is not None and evaluable(s):
        parts.append({"diff": d["diff"], "ci95": d["ci95"], "holds": d["ci95"][1] < 0})
    out.append(
        {
            "id": "E1",
            "statement": "anticipated exception to P3: on mediator queried after the onset, the router "
            "(contemporaneous NaiveOLS) is significantly worse than do-SVAR, which models the lag",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    s = "confounder_mediator"
    if evaluable(s):
        fd, bd = _cell(result, "FrontDoorOLS", s), _cell(result, "BackDoorOLS", s)
        if fd.get("status") == "ok" and bd.get("status") == "ok":
            parts.append(
                {
                    "FrontDoorOLS": fd["effect_sign_acc"],
                    "BackDoorOLS": bd["effect_sign_acc"],
                    "holds": fd["effect_sign_acc"] >= bd["effect_sign_acc"],
                }
            )
    out.append(
        {
            "id": "E2",
            "statement": "anticipated exception to P3: on confounder_mediator FrontDoorOLS scores at least "
            "as high as the routed BackDoorOLS (point estimates)",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    s = "observed_confounder"
    if s in structures:
        rates = {
            k: (
                _cell(result, k, s).get("effect_call_rate")
                if _cell(result, k, s).get("status") == "ok"
                else None
            )
            for k in ("do-SVAR", "BackDoorOLS", "NaiveOLS")
        }
        parts.append(
            {
                "check": f"BackDoorOLS false-effect rate >= {MIN_FALSE_EFFECT}",
                "value": rates["BackDoorOLS"],
                "holds": None
                if rates["BackDoorOLS"] is None
                else rates["BackDoorOLS"] >= MIN_FALSE_EFFECT,
            }
        )
        parts.append(
            {
                "check": "BackDoorOLS below NaiveOLS",
                "values": [rates["BackDoorOLS"], rates["NaiveOLS"]],
                "holds": None
                if None in (rates["BackDoorOLS"], rates["NaiveOLS"])
                else rates["BackDoorOLS"] < rates["NaiveOLS"],
            }
        )
        parts.append(
            {
                "check": "do-SVAR below BackDoorOLS",
                "values": [rates["do-SVAR"], rates["BackDoorOLS"]],
                "holds": None
                if None in (rates["do-SVAR"], rates["BackDoorOLS"])
                else rates["do-SVAR"] < rates["BackDoorOLS"],
            }
        )
    out.append(
        {
            "id": "E3",
            "statement": "on observed_confounder the contemporaneous back-door set leaves A(t-1) paths open: "
            f"BackDoorOLS (the router) makes false effects at a rate >= {MIN_FALSE_EFFECT}, below NaiveOLS, "
            "and do-SVAR, which conditions on the lags, makes fewer than BackDoorOLS",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    for s in ("bi_variate", "mediator"):
        d = contrasts.get(s, {}).get("vs_best_naive", {}).get("do-SVAR")
        if d is not None and evaluable(s):
            parts.append(
                {"structure": s, "diff": d["diff"], "ci95": d["ci95"], "holds": d["ci95"][0] > 0}
            )
    out.append(
        {
            "id": "S1",
            "statement": "do-SVAR beats the best naive estimator (95% CI above 0) on bi_variate and, when it is "
            "queried after the onset, on mediator",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )

    parts = []
    for s in structures:
        c = _cell(result, "Oracle", s)
        if c.get("status") != "ok":
            continue
        if c["effect_n_valid"] > 0:
            parts.append(
                {
                    "structure": s,
                    "effect_sign_acc": c["effect_sign_acc"],
                    "holds": c["effect_sign_acc"] == 1.0,
                }
            )
        if structures[s]["null_effect"]:
            parts.append(
                {
                    "structure": s,
                    "effect_call_rate": c["effect_call_rate"],
                    "holds": c["effect_call_rate"] == 0.0,
                }
            )
    out.append(
        {
            "id": "O1",
            "statement": "the oracle scores 1 wherever the effect sign is defined and makes no false effect",
            "parts": parts,
            "outcome": _verdict(parts),
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def gate_reproduction(table: dict[str, Any], ref_effect: Path, ref_level: Path) -> dict[str, Any]:
    """Gate 1: the package estimators reproduce the released 1.1.0 reference rows.

    Args:
        table: Output of :func:`score` on 1.1.0.
        ref_effect: Released effect-scored reference JSON.
        ref_level: Released level-scored reference JSON.

    Returns:
        The comparison, with ``passed``.
    """
    ref_e = json.loads(ref_effect.read_text())
    ref_l = json.loads(ref_level.read_text())
    rows_e = {r["baseline"]: r for r in ref_e["rows"] if "error" not in r}
    rows_l = {r["baseline"]: r for r in ref_l["rows"] if "error" not in r}
    checks = []
    for name, re_ in rows_e.items():
        ours = table["estimators"].get(name)
        if ours is None or ours.get("pooled") is None:
            checks.append({"estimator": name, "compared": False})
            continue
        pooled = ours["pooled"]
        rl = rows_l[name]
        pairs = {
            "effect_dir_acc": (pooled["effect_sign_acc"], re_["dir_acc"]),
            "effect_n_valid": (pooled["effect_n_valid"], re_["dir_n_valid"]),
            "level_dir_acc": (pooled["level_sign_acc"], rl["dir_acc"]),
            "level_n_valid": (pooled["level_n_valid"], rl["dir_n_valid"]),
            "pooled_rmse": (pooled["level_rmse"], re_["pooled_rmse"]),
        }
        checks.append(
            {
                "estimator": name,
                "compared": True,
                **{k: {"ours": a, "released": b, "equal": a == b} for k, (a, b) in pairs.items()},
            }
        )
    compared = [c for c in checks if c["compared"]]
    passed = bool(compared) and all(
        v["equal"] for c in compared for k, v in c.items() if isinstance(v, dict)
    )
    return {
        "gate": 1,
        "what": "pooled effect/level direction accuracy, n_valid and RMSE of the package "
        "estimators equal the released 1.1.0 reference rows exactly",
        "references": [str(p.relative_to(REPO_ROOT)) for p in (ref_effect, ref_level)],
        "reference_suite_version": [ref_e["suite_version"], ref_l["suite_version"]],
        "checks": checks,
        "passed": passed and ref_e["suite_version"] == ref_l["suite_version"] == GATE1_VERSION,
    }


def gate_stability(table: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Gate 3: structures other than mediator score as on 1.1.0, and the oracle is exact.

    Args:
        table: Output of :func:`score` on the new version.
        reference: The 1.1.0 analysis JSON.

    Returns:
        The comparison, with ``passed``.
    """
    mismatches: list[str] = []
    skipped: list[str] = []
    compared = 0
    common = [
        s for s in reference["structures"] if s in table["structures"] and s not in GATE3_CHANGED
    ]
    for s in common:
        if table["structures"][s]["n_episodes"] != reference["structures"][s]["n_episodes"]:
            mismatches.append(f"{s}: n_episodes")
        for name, row in table["estimators"].items():
            ref_row = reference["estimators"].get(name)
            cell = row["per_structure"].get(s, {})
            ref_cell = (ref_row or {}).get("per_structure", {}).get(s, {})
            if cell.get("status") != "ok" or ref_cell.get("status") != "ok":
                skipped.append(f"{name}/{s}")
                continue
            compared += 1
            diff = [k for k in set(cell) | set(ref_cell) if cell.get(k) != ref_cell.get(k)]
            if diff:
                mismatches.append(f"{name}/{s}: {sorted(diff)}")
        a, b = table["contrasts"].get(s), reference["contrasts"].get(s)
        if a is not None and b is not None and a.get("separation") != b.get("separation"):
            mismatches.append(f"separation/{s}")
    oracle_bad = []
    oracle = table["estimators"].get("Oracle", {})
    for s, cell in oracle.get("per_structure", {}).items():
        if cell.get("status") != "ok":
            oracle_bad.append(f"{s}: not scored")
        elif cell["effect_n_valid"] > 0 and cell["effect_sign_acc"] != 1.0:
            oracle_bad.append(f"{s}: effect sign {cell['effect_sign_acc']}")
        elif table["structures"][s]["null_effect"] and cell["effect_call_rate"] != 0.0:
            oracle_bad.append(f"{s}: effect call rate {cell['effect_call_rate']}")
    return {
        "gate": 3,
        "what": "every structure other than mediator scores exactly as on 1.1.0, and the "
        "oracle scores 1 wherever the effect sign is defined",
        "structures_compared": common,
        "cells_compared": compared,
        "cells_skipped_pending": skipped,
        "mismatches": mismatches,
        "oracle_failures": oracle_bad,
        "passed": compared > 0 and not mismatches and not oracle_bad,
    }


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


def write_predictions(
    path: Path,
    cols: EpisodeColumns,
    preds: dict[str, np.ndarray],
    models: dict[str, est.Estimator | None],
    meta: dict[str, str],
) -> None:
    """Write the per-episode predictions, one row per scored (episode, estimator).

    Args:
        path: Destination parquet file.
        cols: The episode columns.
        preds: Predictions from :func:`predict_all`.
        models: The estimators, ``None`` where pending.
        meta: File-level key/value metadata (suite, version, commit, protocol).
    """
    frames = []
    for name, model in models.items():
        if model is None:
            continue
        p = preds[name]
        keep = ~np.isnan(p[:, 0])
        uses_v = np.array([model.uses_do_value(s) for s in cols.structure[keep]], dtype=bool)
        n = int(keep.sum())
        frames.append(
            {
                "scm_id": cols.scm_id[keep].astype(np.int32),
                "structure": cols.structure[keep].tolist(),
                "estimator": [name] * n,
                "estimator_class": [est.CLASS_OF[name]] * n,
                "uses_do_value": uses_v,
                "onset": cols.onset[keep].astype(np.int16),
                "query_idx": cols.query_idx[keep].astype(np.int16),
                "do_value": cols.v[keep].astype(np.float32),
                "ref_do_value": cols.a_ref[keep].astype(np.float32),
                "y_true": cols.y_true[keep],
                "y_obs": cols.y_obs[keep],
                "pred": p[keep, 0],
                "pred_ref": p[keep, 1],
            }
        )
    arrays = {}
    for k in frames[0]:
        if k in ("structure", "estimator", "estimator_class"):
            # Dictionary encoding stores each label once, not once per row.
            arrays[k] = pa.array([x for f in frames for x in f[k]]).dictionary_encode()
        else:
            arrays[k] = pa.array(np.concatenate([f[k] for f in frames]))
    tbl = pa.table(arrays).replace_schema_metadata({k: str(v) for k, v in meta.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd")
    print(f"wrote {path} ({tbl.num_rows} rows)", flush=True)


def _fmt(x: float | None, digits: int = 3) -> str:
    """Format a number for the Markdown tables.

    Args:
        x: The number or ``None``.
        digits: Decimals.

    Returns:
        The formatted string, ``n/a`` for ``None``.
    """
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    """Markdown tables of a run: effect sign, false effects, contrasts, level metrics.

    Args:
        result: The analysis JSON.

    Returns:
        The Markdown document.
    """
    structures = list(result["structures"])
    ests = result["estimators"]
    head = "| Estimator | Class | " + " | ".join(f"`{s}`" for s in structures) + " | Pooled |"
    rule = "|---|---|" + "---|" * (len(structures) + 1)
    lines = [
        f"# Detection power on {result['suite']} {result['version']}",
        "",
        f"Generated by `scripts/detection_power.py` at commit `{result['code']['git_commit']}` "
        f"from {result['n_episodes']} episodes. Cells give the estimate ± its binomial SE. "
        "† marks an estimator that declines the structure and predicts the pre-onset outcome "
        "mean, so its predicted effect is zero by construction. `pending` marks an estimator "
        "the package does not register yet. n/a marks an undefined value.",
        "",
        "## Pre-registered predictions",
        "",
        "Graded mechanically by `evaluate_predictions`; see README.md for the statements.",
        "",
        "| Id | Outcome | Statement |",
        "|---|---|---|",
        *[
            f"| {p['id']} | {p['outcome']} | {p['statement']} |"
            for p in result.get("preregistered", [])
        ],
        "",
        "## Effect-sign accuracy",
        "",
        "Official protocol: `direction_accuracy(pred - y_obs, y_true - y_obs)`, scored on the "
        "episodes with `|y_true - y_obs| >= 0.1`.",
        "",
        head,
        rule,
    ]
    n_valid = [
        str(result["structures"][s]["n_valid_effect"])
        if "n_valid_effect" in result["structures"][s]
        else "n/a"
        for s in structures
    ]
    pooled_valid = result.get("pooled_n_valid_effect")
    lines.append(
        f"| n_valid | | {' | '.join(n_valid)} | {pooled_valid if pooled_valid is not None else 'n/a'} |"
    )

    def effect_cell(cell: dict[str, Any] | None) -> str:
        """Format one effect-sign cell.

        Args:
            cell: The scored cell, or ``None``.

        Returns:
            ``acc ± se`` with a dagger for declined structures, or a status word.
        """
        if cell is None or cell.get("status") == "pending":
            return "pending"
        if cell["effect_sign_acc"] is None:
            return "n/a"
        mark = "" if cell.get("uses_do_value", True) else " †"
        return f"{cell['effect_sign_acc']:.3f} ± {cell['effect_sign_se']:.3f}{mark}"

    for name, row in ests.items():
        cells = [effect_cell(row["per_structure"].get(s)) for s in structures]
        pooled = row.get("pooled")
        pooled_s = (
            "pending"
            if pooled is None
            else (
                "n/a"
                if pooled["effect_sign_acc"] is None
                else f"{pooled['effect_sign_acc']:.3f} ± {pooled['effect_sign_se']:.3f}"
            )
        )
        lines.append(f"| {name} | {row['class']} | {' | '.join(cells)} | {pooled_s} |")

    nulls = [s for s in structures if result["structures"][s]["null_effect"]]
    if nulls:
        lines += [
            "",
            "## False-effect rate on null-effect structures",
            "",
            "Fraction of episodes with `|pred(do v) - pred(do a_ref)| >= 0.1`, where `a_ref` is the "
            "factual treatment at the intervened row. `trivial`: the estimator ignores the "
            "do-value on this structure. `rule 3`: the router sets tau = 0 from the DAG.",
            "",
            "| Estimator | "
            + " | ".join(f"`{s}` ({result['structures'][s]['null_reason']})" for s in nulls)
            + " |",
            "|---|" + "---|" * len(nulls),
        ]
        for name, row in ests.items():
            cells = []
            for s in nulls:
                cell = row["per_structure"].get(s)
                if cell is None or cell.get("status") == "pending":
                    cells.append("pending")
                elif cell["false_effect_label"] == "estimated":
                    cells.append(f"{cell['effect_call_rate']:.3f} ± {cell['effect_call_se']:.3f}")
                else:
                    cells.append(f"{cell['effect_call_rate']:.3f} ({cell['false_effect_label']})")
            lines.append(f"| {name} | {' | '.join(cells)} |")

    lines += [
        "",
        "## Separation: best identification-aware minus best naive",
        "",
        f"Effect-sign accuracy difference with a paired episode bootstrap 95% interval "
        f"({N_BOOT} resamples of the valid episodes, `default_rng({BOOT_SEED})` per structure, "
        "maxima taken inside each resample).",
        "",
        "| Structure | n_valid | Best identification-aware | Best naive | Difference | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for s in structures:
        c = result["contrasts"].get(s, {})
        sep = c.get("separation")
        if not c.get("n_valid"):
            lines.append(f"| `{s}` | 0 | n/a | n/a | n/a | n/a |")
        elif sep is None or sep.get("diff") is None:
            lines.append(f"| `{s}` | {c['n_valid']} | none applies | n/a | n/a | n/a |")
        else:
            lo, hi = sep["ci95"]
            lines.append(
                f"| `{s}` | {c['n_valid']} | {sep['best_identification_aware']} | "
                f"{sep['best_naive']} | {sep['diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
            )

    others = [k for k in ests if k not in est.NAIVE and k != "Oracle"]
    lines += [
        "",
        "## Every estimator against the best naive one",
        "",
        "Effect-sign accuracy difference, paired bootstrap 95% interval.",
        "",
        "| Estimator | " + " | ".join(f"`{s}`" for s in structures) + " |",
        "|---|" + "---|" * len(structures),
    ]
    for k in others:
        cells = []
        for s in structures:
            d = result["contrasts"].get(s, {}).get("vs_best_naive", {}).get(k)
            cells.append(
                "n/a"
                if d is None
                else f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
            )
        lines.append(f"| {k} | {' | '.join(cells)} |")

    lines += [
        "",
        "## GraphRouter against the other valid estimators",
        "",
        "Route from the structure's DAG. Difference = router minus the named valid estimator, "
        "paired bootstrap 95% interval.",
        "",
        "| Structure | Route | Rule | Compared with | Difference | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for s in structures:
        r = result["structures"][s]["route"]
        peers = result["contrasts"].get(s, {}).get("router_vs_valid", {})
        routed = r["estimator"] or "tau = 0"
        if not peers:
            lines.append(f"| `{s}` | {routed} | {r['rule']} | n/a | n/a | n/a |")
        for k, d in peers.items():
            lo, hi = d["ci95"]
            lines.append(
                f"| `{s}` | {routed} | {r['rule']} | {k} | {d['diff']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
            )

    for pair, rows in result.get("cross_structure", {}).items():
        lines += [
            "",
            f"## Cross-structure contrast: `{pair}`",
            "",
            "Effect-sign accuracy on the first structure minus the second, independent bootstrap 95% interval.",
            "",
            "| Estimator | Difference | 95% CI |",
            "|---|---|---|",
        ]
        for k, d in rows.items():
            lines.append(f"| {k} | {d['diff']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")

    lines += [
        "",
        "## Exploratory: sign of the estimator's own effect",
        "",
        "Not pre-registered; added after the 1.1.0 results. Sign of `pred(do v) - pred(do a_ref)` "
        "against the true effect on the same valid episodes, which drops the level-forecast "
        "term of the official score. First over all valid episodes (a zero contrast, an "
        "abstention, counts as wrong), then over the episodes where the estimator calls a "
        "direction, with that count in brackets.",
        "",
        "| Estimator | " + " | ".join(f"`{s}`" for s in structures) + " |",
        "|---|" + "---|" * len(structures),
    ]
    for name, row in ests.items():
        cells = []
        for s in structures:
            cell = row["per_structure"].get(s)
            if cell is None or cell.get("status") == "pending":
                cells.append("pending")
            elif cell["contrast_sign_acc"] is None:
                cells.append("n/a")
            else:
                calls = cell["contrast_sign_acc_calls"]
                cells.append(
                    f"{cell['contrast_sign_acc']:.3f}; {_fmt(calls)} [{cell['contrast_n_calls']}]"
                )
        lines.append(f"| {name} | {' | '.join(cells)} |")

    for title, key, digits in (
        (
            "Exploratory: effect RMSE, `pred(do v) - pred(do a_ref)` against `y_true - y_obs` "
            "(the naive rows are the no-effect baseline)",
            "effect_rmse",
            3,
        ),
        ("Level RMSE", "level_rmse", 3),
        ("Level-sign accuracy", "level_sign_acc", 3),
    ):
        lines += ["", f"## {title}", "", head, rule]
        for name, row in ests.items():
            cells = []
            for s in structures:
                cell = row["per_structure"].get(s)
                cells.append(
                    "pending"
                    if cell is None or cell.get("status") == "pending"
                    else _fmt(cell[key], digits)
                )
            pooled = row.get("pooled")
            lines.append(
                f"| {name} | {row['class']} | {' | '.join(cells)} | "
                f"{'pending' if pooled is None else _fmt(pooled[key], digits)} |"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Gate 2: synthetic self-test
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A linear Gaussian SVAR laid out in one structure's canonical columns.

    Attributes:
        name: Scenario label.
        structure: Structure whose canonical layout and DAG the episode claims.
        columns: Canonical column order (treatment first, outcome last).
        hidden: Columns zeroed in the released arms.
        inst: ``{child: {parent: coef}}`` within a step.
        lag: ``{child: {parent: coef}}`` from the previous step.
        offset: Query row minus onset row.
        expect: ``{estimator: "recovers" | "biased" | "declines" | "zero" | "report"}``.
    """

    name: str
    structure: str
    columns: tuple[str, ...]
    hidden: tuple[str, ...]
    inst: dict[str, dict[str, float]]
    lag: dict[str, dict[str, float]]
    offset: int
    expect: dict[str, str]


def _self_loops(names: Sequence[str], phi: float = 0.5) -> dict[str, dict[str, float]]:
    """Autoregression ``phi`` on each named variable.

    Args:
        names: Variables.
        phi: Coefficient.

    Returns:
        ``{v: {v: phi}}``.
    """
    return {v: {v: phi} for v in names}


DECLINE_ALL = {"Zero": "declines", "Mean": "declines", "AR1": "declines", "VAR-OLS": "declines"}


def self_test_scenarios() -> list[Scenario]:
    """Synthetic scenarios where each estimator's assumptions hold, plus confounded ones.

    Each scenario switches off the temporal paths that the estimators'
    contemporaneous adjustment sets leave open (see README, Limitations): no
    lagged confounder-to-outcome edge next to an autoregressive treatment for
    the back-door sets, no mediator autoregression for the front-door and
    confounder-mediator regressions, and an i.i.d. instrument for IV2SLS.

    Returns:
        The scenarios, each with its expectations per estimator.
    """
    declines3 = {"BackDoorOLS": "declines", "IV2SLS": "declines", "FrontDoorOLS": "declines"}
    return [
        Scenario(
            "bi_variate",
            "bi_variate",
            ("A", "Y"),
            (),
            {"Y": {"A": 1.0}},
            _self_loops("AY"),
            0,
            {
                **DECLINE_ALL,
                **declines3,
                "NaiveOLS": "recovers",
                "do-SVAR": "recovers",
                "GraphRouter": "recovers",
            },
        ),
        Scenario(
            "back_door",
            "back_door",
            ("A", "X", "Y"),
            (),
            {"A": {"X": 1.0}, "Y": {"A": 1.0, "X": 1.0}},
            _self_loops("AXY"),
            0,
            {
                **DECLINE_ALL,
                "BackDoorOLS": "recovers",
                "GraphRouter": "recovers",
                "NaiveOLS": "biased",
                "do-SVAR": "biased",
                "IV2SLS": "declines",
                "FrontDoorOLS": "declines",
            },
        ),
        Scenario(
            "observed_confounder",
            "observed_confounder",
            ("A", "X", "Y"),
            (),
            {"A": {"X": 1.0}},
            {"X": {"X": 0.9}, "Y": {"Y": 0.5, "X": 1.0}},
            0,
            {
                **DECLINE_ALL,
                "BackDoorOLS": "recovers",
                "GraphRouter": "recovers",
                "NaiveOLS": "biased",
                "IV2SLS": "declines",
                "FrontDoorOLS": "declines",
            },
        ),
        Scenario(
            "mediator, query at offset 1",
            "mediator",
            ("A", "M", "Y"),
            (),
            {"Y": {"M": 1.0}},
            {"A": {"A": 0.5}, "M": {"A": 1.0, "M": 0.5}, "Y": {"Y": 0.5}},
            1,
            {
                **DECLINE_ALL,
                "do-SVAR": "recovers",
                "NaiveOLS": "report",
                "FrontDoorOLS": "report",
                "GraphRouter": "report",
                "BackDoorOLS": "declines",
                "IV2SLS": "declines",
            },
        ),
        Scenario(
            "mediator, query at offset 0",
            "mediator",
            ("A", "M", "Y"),
            (),
            {"Y": {"M": 1.0}},
            {"A": {"A": 0.5}, "M": {"A": 1.0, "M": 0.5}, "Y": {"Y": 0.5}},
            0,
            {"do-SVAR": "recovers"},
        ),
        Scenario(
            "front_door",
            "front_door",
            ("A", "U", "M", "Y"),
            ("U",),
            {"A": {"U": 1.0}, "M": {"A": 1.0}, "Y": {"M": 1.0, "U": 1.0}},
            _self_loops("AY"),
            0,
            {
                **DECLINE_ALL,
                "FrontDoorOLS": "recovers",
                "GraphRouter": "recovers",
                "NaiveOLS": "biased",
                "do-SVAR": "biased",
                "BackDoorOLS": "declines",
                "IV2SLS": "declines",
            },
        ),
        Scenario(
            "confounder_mediator",
            "confounder_mediator",
            ("A", "X", "M", "Y"),
            (),
            {"A": {"X": 1.0}, "M": {"A": 1.0}, "Y": {"M": 1.0, "X": 1.0}},
            _self_loops("AXY"),
            0,
            {
                **DECLINE_ALL,
                "BackDoorOLS": "recovers",
                "FrontDoorOLS": "recovers",
                "GraphRouter": "recovers",
                "NaiveOLS": "biased",
                "do-SVAR": "biased",
                "IV2SLS": "declines",
            },
        ),
        Scenario(
            "instrumental_variable",
            "instrumental_variable",
            ("A", "U", "X", "Y"),
            ("U",),
            {"A": {"X": 1.0, "U": 1.0}, "Y": {"A": 1.0, "U": 1.0}},
            _self_loops("AUY"),
            0,
            {
                **DECLINE_ALL,
                "IV2SLS": "recovers",
                "GraphRouter": "recovers",
                "NaiveOLS": "biased",
                "do-SVAR": "biased",
                "BackDoorOLS": "declines",
                "FrontDoorOLS": "declines",
            },
        ),
        Scenario(
            "unobserved_confounder",
            "unobserved_confounder",
            ("A", "U", "Y"),
            ("U",),
            {"A": {"U": 1.0}, "Y": {"U": 1.0}},
            _self_loops("AUY"),
            0,
            {
                **DECLINE_ALL,
                **declines3,
                "GraphRouter": "zero",
                "NaiveOLS": "biased",
                "do-SVAR": "biased",
            },
        ),
        Scenario(
            "bow_graph",
            "bow_graph",
            ("A", "U", "Y"),
            ("U",),
            {"A": {"U": 1.0}, "Y": {"U": 1.0, "A": 1.0}},
            _self_loops("AUY"),
            0,
            {
                **DECLINE_ALL,
                **declines3,
                "GraphRouter": "biased",
                "NaiveOLS": "biased",
                "do-SVAR": "biased",
            },
        ),
        *benchmark_like_scenarios(),
    ]


def benchmark_like_scenarios() -> list[Scenario]:
    """Report-only scenarios with every lagged edge and self-loop of the benchmark DAGs.

    The released structures give every variable an autoregressive term and some
    lagged cross edges (X(t-1) -> Y(t), M(t-1) -> Y(t)). Under linearity these
    show how far the contemporaneous estimators drift once the temporal paths
    they do not condition on are open. Nothing is asserted.

    Returns:
        One scenario per structure whose benchmark DAG differs from the
        asserted scenario above.
    """
    report = dict.fromkeys(
        ("NaiveOLS", "do-SVAR", "BackDoorOLS", "IV2SLS", "FrontDoorOLS", "GraphRouter"), "report"
    )
    return [
        Scenario(
            "back_door (benchmark-like lags)",
            "back_door",
            ("A", "X", "Y"),
            (),
            {"A": {"X": 1.0}, "Y": {"A": 1.0, "X": 1.0}},
            {"X": {"X": 0.5}, "A": {"A": 0.5}, "Y": {"Y": 0.5, "X": 1.0}},
            0,
            report,
        ),
        Scenario(
            "observed_confounder (benchmark-like lags)",
            "observed_confounder",
            ("A", "X", "Y"),
            (),
            {"A": {"X": 1.0}},
            {"X": {"X": 0.9}, "A": {"A": 0.5}, "Y": {"Y": 0.5, "X": 1.0}},
            0,
            report,
        ),
        Scenario(
            "confounder_mediator (benchmark-like lags)",
            "confounder_mediator",
            ("A", "X", "M", "Y"),
            (),
            {"A": {"X": 1.0}, "M": {"A": 1.0}, "Y": {"M": 1.0}},
            {"X": {"X": 0.5}, "A": {"A": 0.5}, "M": {"M": 0.5}, "Y": {"Y": 0.5, "X": 1.0}},
            0,
            report,
        ),
        Scenario(
            "front_door (benchmark-like lags)",
            "front_door",
            ("A", "U", "M", "Y"),
            ("U",),
            {"A": {"U": 1.0}, "M": {"A": 1.0}, "Y": {"M": 1.0, "U": 1.0}},
            {"U": {"U": 0.5}, "A": {"A": 0.5}, "M": {"M": 0.5}, "Y": {"Y": 0.5, "M": 1.0}},
            0,
            report,
        ),
        Scenario(
            "instrumental_variable (benchmark-like lags)",
            "instrumental_variable",
            ("A", "U", "X", "Y"),
            ("U",),
            {"A": {"X": 1.0, "U": 1.0}, "Y": {"A": 1.0, "U": 1.0}},
            _self_loops("AUXY"),
            0,
            report,
        ),
    ]


def simulate(
    sc: Scenario,
    noise: np.ndarray,
    clamp: tuple[int, float] | None = None,
    start: np.ndarray | None = None,
) -> np.ndarray:
    """Roll a scenario's SVAR forward with the given noise.

    Args:
        sc: The scenario.
        noise: ``(T, n)`` standard normal draws, shared across arms.
        clamp: ``(row, value)`` of a one-shot hard intervention on ``A``.
        start: A trajectory whose rows before the clamp row are reused (the
            arms agree exactly there), or ``None`` to simulate from row 0.

    Returns:
        ``(T, n)`` trajectory in the scenario's column order.
    """
    col = {v: i for i, v in enumerate(sc.columns)}
    dag = nx.DiGraph()
    dag.add_nodes_from(sc.columns)
    dag.add_edges_from((p, c) for c, parents in sc.inst.items() for p in parents)
    order = list(nx.topological_sort(dag))
    x = np.zeros_like(noise)
    first = 0
    if start is not None and clamp is not None:
        first = clamp[0]
        x[:first] = start[:first]
    for t in range(first, len(noise)):
        for v in order:
            if clamp is not None and t == clamp[0] and v == "A":
                x[t, col[v]] = clamp[1]
                continue
            val = noise[t, col[v]]
            for p, c in sc.inst.get(v, {}).items():
                val += c * x[t, col[p]]
            if t > 0:
                for p, c in sc.lag.get(v, {}).items():
                    val += c * x[t - 1, col[p]]
            x[t, col[v]] = val
    return x


def synthetic_episode(sc: Scenario, t_len: int) -> tuple[Episode, float]:
    """A shared-noise counterfactual episode of a scenario and its true effect.

    The do-value is ``a_ref + 1``, so an estimator's slope is its prediction at
    ``v`` minus its prediction at ``a_ref``.

    Args:
        sc: The scenario.
        t_len: Trajectory length.

    Returns:
        ``(episode, tau)``, tau being the exact effect of a unit change of the
        treatment at the onset row on ``Y`` at the query row.
    """
    rng = np.random.default_rng([SELF_TEST_SEED, zlib.crc32(sc.name.encode())])
    noise = rng.standard_normal((t_len, len(sc.columns)))
    onset = t_len - 5
    q = onset + sc.offset
    x = simulate(sc, noise)
    hidden = [sc.columns.index(h) for h in sc.hidden]
    x_obs = torch.as_tensor(x, dtype=torch.float32)
    x_obs[:, hidden] = 0.0
    a_ref = float(x_obs[onset, 0])
    v = a_ref + 1.0
    x_ref = simulate(sc, noise, clamp=(onset, a_ref), start=x)
    x_int = simulate(sc, noise, clamp=(onset, v), start=x)
    tau = float(x_int[q, -1] - x_ref[q, -1])
    x_int_t = torch.as_tensor(x_int, dtype=torch.float32)
    x_int_t[:, hidden] = 0.0
    ep = Episode(
        x_obs=x_obs,
        x_int=x_int_t,
        intervention=InterventionSpec(
            targets=[0], times=[onset], intervention_type=InterventionType.HARD, values=v
        ),
        y_true=x_int_t[q, -1:].clone(),
        query_target=torch.tensor([len(sc.columns) - 1]),
        query_time=torch.tensor([q / t_len], dtype=torch.float32),
        structure=sc.structure,
        scm_id=0,
        metadata={"query_time_idx": [q], "pair_mode": "counterfactual"},
    )
    return ep, tau


EXPECTED_ROUTES = {
    "bi_variate": ("NaiveOLS", True),
    "back_door": ("BackDoorOLS", True),
    "observed_confounder": ("BackDoorOLS", True),
    "mediator": ("NaiveOLS", True),
    "front_door": ("FrontDoorOLS", True),
    "confounder_mediator": ("BackDoorOLS", True),
    "instrumental_variable": ("IV2SLS", True),
    "unobserved_confounder": (None, True),
    "bow_graph": ("NaiveOLS", False),
}


def run_self_test(out_dir: Path) -> dict[str, Any]:
    """Gate 2: every estimator recovers the known effect where its assumptions hold.

    Args:
        out_dir: Where ``self_test.json`` is written.

    Returns:
        The self-test record, with ``passed``.

    Raises:
        AssertionError: If a check fails (after writing the record).
    """
    t0 = time.time()
    models = est.build_estimators()
    failures: list[str] = []
    routes = {s: est.route(s) for s in EXPECTED_ROUTES}
    for s, (name, identified) in EXPECTED_ROUTES.items():
        if (routes[s].estimator, routes[s].identified) != (name, identified):
            failures.append(f"route {s}: {routes[s]} != {(name, identified)}")
    scenarios = []
    for sc in self_test_scenarios():
        ep, tau = synthetic_episode(sc, SELF_TEST_T)
        v, a_ref = est.do_value(ep), est.reference_do_value(ep)
        rows = {}
        for name, model in models.items():
            expect = "oracle" if name == "Oracle" else sc.expect.get(name)
            if expect is None:
                continue
            if model is None or (
                isinstance(model, est.GraphRouter) and model.is_pending(sc.structure)
            ):
                rows[name] = {"expect": expect, "status": "pending"}
                continue
            pv, pr = model.predict(ep, v), model.predict(ep, a_ref)
            slope = pv - pr
            if expect == "oracle":
                ok = pv == float(ep.y_true[0]) and pr == float(query_obs_levels(ep)[0])
            elif expect == "recovers":
                ok = abs(slope - tau) <= RECOVER_TOL
            elif expect == "biased":
                ok = abs(slope - tau) >= BIAS_MIN
            elif expect in ("declines", "zero"):
                ok = slope == 0.0 and (expect == "zero" or not model.uses_do_value(sc.structure))
            else:
                ok = True
            rows[name] = {"expect": expect, "slope": slope, "error": slope - tau, "ok": ok}
            if not ok:
                failures.append(
                    f"{sc.name}/{name}: expected {expect}, slope {slope:.4f}, tau {tau:.4f}"
                )
        scenarios.append(
            {
                "scenario": sc.name,
                "structure": sc.structure,
                "offset": sc.offset,
                "tau": tau,
                "estimators": rows,
            }
        )
        print(
            f"[self-test] {sc.name:55s} tau={tau:+.3f} "
            + " ".join(
                f"{k}={r['slope']:+.3f}{'' if r['ok'] else '!'}" if "slope" in r else f"{k}=pending"
                for k, r in rows.items()
                if k != "Oracle"
            ),
            flush=True,
        )
    record = {
        "gate": 2,
        "what": "linear Gaussian SVARs with T = 20,000: each estimator recovers the known "
        f"effect within {RECOVER_TOL} where its assumptions hold, confounded estimators miss "
        f"it by at least {BIAS_MIN}, declining estimators predict no effect, the oracle "
        "returns y_true at v and y_obs at a_ref, and the router follows the DAG",
        "code": code_provenance(),
        "T": SELF_TEST_T,
        "seed": SELF_TEST_SEED,
        "routes": {
            s: {"estimator": r.estimator, "rule": r.rule, "identified": r.identified}
            for s, r in routes.items()
        },
        "scenarios": scenarios,
        "pending": [n for n, m in models.items() if m is None],
        "failures": failures,
        "passed": not failures,
        "runtime_s": round(time.time() - t0, 1),
    }
    _write_json(out_dir / "self_test.json", record)
    if failures:
        raise AssertionError(f"self-test failed: {failures}")
    return record


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    """Run the self-test, or score a suite and assert its gate.

    Args:
        argv: Command-line arguments (``None`` reads ``sys.argv``).

    Raises:
        SystemExit: On invalid arguments or a missing prerequisite gate.
        AssertionError: If a gate fails (after its record is written).
        RuntimeError: If the targets fail QA or ``dotime`` is not this checkout's.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--self-test", action="store_true", help="run gate 2 and exit")
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--version", default=GATE1_VERSION)
    ap.add_argument(
        "--out", type=Path, default=None, help=f"output directory (default {ANALYSIS_DIR.name}/)"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--cache-dir", type=Path, default=None, help="suite cache root for load_benchmark"
    )
    ap.add_argument(
        "--suite-dir", type=Path, default=None, help="local build of an unregistered version"
    )
    ap.add_argument("--estimators", nargs="+", default=list(est.ESTIMATOR_ORDER))
    ap.add_argument("--compare-to", type=Path, default=None, help="1.1.0 table for gate 3")
    ap.add_argument(
        "--limit-per-structure",
        type=int,
        default=None,
        help="smoke run on the first episodes of each structure (no gate; needs --out)",
    )
    args = ap.parse_args(argv)
    unknown = sorted(set(args.estimators) - set(est.ESTIMATOR_ORDER))
    if unknown:
        ap.error(f"unknown estimators {unknown}; choose from {list(est.ESTIMATOR_ORDER)}")
    if args.limit_per_structure is not None and args.out is None:
        ap.error("--limit-per-structure writes partial outputs, so it needs an explicit --out")
    out_dir = args.out or ANALYSIS_DIR
    _check_dotime_source()
    torch.set_num_threads(1)

    if args.self_test:
        run_self_test(out_dir)
        return

    tag = output_tag(args.suite, args.version)
    is_gate1 = (
        args.suite == SUITE and args.version == GATE1_VERSION and args.limit_per_structure is None
    )
    is_gate3 = (
        args.suite == SUITE and args.version != GATE1_VERSION and args.limit_per_structure is None
    )
    ref_table_path = args.compare_to or out_dir / f"{output_tag(SUITE, GATE1_VERSION)}.json"
    if is_gate3:
        # Gates run in order: the reference table must have passed gate 1 and
        # the estimators gate 2 before a new version is compared with it.
        for prereq in (
            out_dir / f"{output_tag(SUITE, GATE1_VERSION)}_gate.json",
            out_dir / "self_test.json",
        ):
            if not prereq.exists() or not json.loads(prereq.read_text()).get("passed"):
                raise SystemExit(f"{prereq} is missing or did not pass; run the earlier gate first")
        if not ref_table_path.exists():
            raise SystemExit(
                f"gate 3 compares with {ref_table_path}; run --version {GATE1_VERSION} first"
            )

    t0 = time.time()
    episodes = load_episodes(
        args.suite, args.version, args.cache_dir, args.suite_dir, args.limit_per_structure
    )
    print(
        f"[{args.suite} {args.version}] {len(episodes)} episodes loaded in {time.time() - t0:.1f}s",
        flush=True,
    )
    qa = target_qa(episodes, None, "effect")
    cols = EpisodeColumns.from_episodes(episodes)
    per_structure_qa = structure_qa(cols)

    names = [n for n in est.ESTIMATOR_ORDER if n in args.estimators]
    models = est.build_estimators(names)
    t1 = time.time()
    preds = predict_all(episodes, names, args.workers)
    print(
        f"predicted {len(names)} estimators in {time.time() - t1:.1f}s with {args.workers} workers",
        flush=True,
    )

    result = score(cols, preds, models)
    for s, info in result["structures"].items():
        info["n_valid_effect"] = int(
            np.sum(np.abs(cols.y_true - cols.y_obs)[cols.structure == s] >= DIR_ACC_EPS)
        )
        info["target_qa"] = per_structure_qa[s]
    pending = [n for n, m in models.items() if m is None]
    result = {
        "analysis": "detection power of dot-Identifiability-v1 by estimator class and structure",
        "suite": args.suite,
        "version": args.version,
        "source": args.suite_dir.name if args.suite_dir is not None else "load_benchmark",
        "n_episodes": len(episodes),
        "limit_per_structure": args.limit_per_structure,
        "code": code_provenance(),
        "protocol": {
            "effect_sign": "direction_accuracy(pred - y_obs, y_true - y_obs), |y_true - y_obs| >= 0.1",
            "level_sign": "direction_accuracy(pred, y_true), |y_true| >= 0.1",
            "false_effect": f"|pred(do v) - pred(do a_ref)| >= {FALSE_EFFECT_THRESHOLD}, a_ref = x_obs[onset, A]",
            "bootstrap": {
                "seed": BOOT_SEED,
                "n_boot": N_BOOT,
                "unit": "valid episode, paired across estimators",
            },
            "arithmetic": "float32, as dotime.reference.reference_table.run_baseline",
        },
        "pending_estimators": pending,
        "target_qa": qa,
        "pooled_n_valid_effect": int(np.sum(np.abs(cols.y_true - cols.y_obs) >= DIR_ACC_EPS)),
        **result,
    }

    result["preregistered"] = evaluate_predictions(result)
    for pr in result["preregistered"]:
        print(f"[pre-registered] {pr['id']}: {pr['outcome']}", flush=True)

    # Gate before writing the table: a table that fails its gate is not an output.
    gate = None
    if is_gate1:
        gate = gate_reproduction(result, REF_EFFECT, REF_LEVEL)
    elif is_gate3:
        gate = gate_stability(result, json.loads(ref_table_path.read_text()))
        gate["reference_table"] = ref_table_path.name
    if gate is not None:
        gate["code"] = result["code"]
        _write_json(out_dir / f"{tag}_gate.json", gate)
        if not gate["passed"]:
            raise AssertionError(f"gate {gate['gate']} failed; see {out_dir / f'{tag}_gate.json'}")
        print(f"[gate {gate['gate']}] passed", flush=True)

    result["runtime_s"] = round(time.time() - t0, 1)
    _write_json(out_dir / f"{tag}.json", result)
    (out_dir / f"{tag}.md").write_text(render_markdown(result))
    print(f"wrote {out_dir / f'{tag}.md'}", flush=True)
    write_predictions(
        out_dir / f"{tag}_predictions.parquet",
        cols,
        preds,
        models,
        {
            "suite": args.suite,
            "version": args.version,
            "git_commit": str(result["code"]["git_commit"]),
            "pred": "predicted level at do(A = do_value) on the onset row, scored at query_idx",
            "pred_ref": "predicted level at do(A = ref_do_value), ref_do_value = x_obs[onset, A]",
        },
    )
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
