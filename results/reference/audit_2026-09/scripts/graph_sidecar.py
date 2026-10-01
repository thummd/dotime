"""Ground-truth graph sidecars for the frozen dot-Generic-100k and dot-RegimeSwitch-v1.

The v1.0.0 files store no graph, so nothing in them says how far an intervention
sits from its query. This script regenerates EVERY released episode with
``make_episode({**spec, "record_graph": True})`` over
``episode_specs(cfg, suite_seed, 1.0)`` (``cfg`` from scripts/release_config.yaml,
the suite seed from the cached manifest, cross-checked against the build rule
``base + 1000 * (position + 1)``) and compares ``x_obs``, ``x_int``, ``y_true``,
``query_target``, ``query_time``, ``intervention_json`` and ``n_vars`` bit for bit
with the md5-verified cached parquet row. A row counts as verified only if all
seven match. A mismatch is reported, never hidden.

One JSON line per episode, in release order::

    {"idx", "family", "scm_class", "n_regimes", "graph", "treatment", "query",
     "query_row", "onset", "window_end", "steps_after_window", "intervention_type",
     "reachable", "min_lag", "min_hops", "direct_lags", "zeroed_obs", "zeroed_int",
     "verified", "mismatch"}

``graph`` is ``dotime.graph_meta.LaggedGraph.to_dict()`` and the path fields are
``path_lag`` from the treatment columns to the query column, exactly as
``metadata["graph"]`` of a ``record_graph`` build stores them. ``family`` is
diverse, chain or regime, read from the first draw of the episode's prior
generator the way ``DoTime.sample_scm`` decides it. Read a sidecar with
``dotime.graph_meta.load_graph_sidecar``.

Episodes run on a fork pool created per shard after the shard's arrays are
loaded, so workers share them copy-on-write. Every module the workers need is
imported before forking, so they all run the code that was on disk at start.
Per-shard outputs in ``--work-dir`` let an interrupted run resume. The gzip
stream has no name and mtime 0, so equal content gives equal bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import time
import warnings
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

import dotime
import dotime.benchmarks
import dotime.continuous
import dotime.extended
import dotime.graph_meta
import dotime.prior
from dotime import _release_io
from dotime._build import episode_specs, make_episode

_REPO = Path(__file__).resolve().parents[4]


def _portable(path: Path) -> str:
    """A path for a published record: relative to the checkout when inside it.

    Args:
        path: An absolute path.

    Returns:
        The path relative to the repository root, or its last two components
        (e.g. ``dotime/__init__.py``) for an installed package.
    """
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(Path(*path.parts[-2:]))


_SUITES = {
    "dot-Generic-100k": "dot-Generic-100k-v1.0.0_graph.jsonl.gz",
    "dot-RegimeSwitch-v1": "dot-RegimeSwitch-v1.0.0_graph.jsonl.gz",
}
_FLOAT_FIELDS = ("x_obs", "x_int", "query_time", "y_true")
_CHAIN_PROB, _REGIME_PROB = 0.15, 0.15  # DoTime defaults used by the build

# Filled by the parent before each shard's pool forks; read by the workers.
_SHARD: dict = {}


def _row_slice(name: str, i: int) -> np.ndarray:
    """One row of a list column of the loaded shard.

    Args:
        name: Column name.
        i: Row index within the shard.

    Returns:
        The row's values (a view into the shard's flat buffer).
    """
    values, offsets = _SHARD["lists"][name]
    return values[offsets[i] : offsets[i + 1]]


def _family(seed: int) -> str:
    """SCM family of a generic episode, from its prior generator's first draw.

    ``DoTime.sample_scm`` draws one uniform from the prior's generator, seeded
    with the episode seed, and picks a chain below 0.15, a regime-switching SCM
    below 0.30 and a diverse SCM otherwise.

    Args:
        seed: The episode seed (attempt 0; the v1.0.0 build never retried).

    Returns:
        ``"chain"``, ``"regime"`` or ``"diverse"``.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    u = torch.rand(1, generator=g).item()
    return "chain" if u < _CHAIN_PROB else "regime" if u < _CHAIN_PROB + _REGIME_PROB else "diverse"


def _work(i: int) -> dict:
    """Regenerate, verify and describe one episode of the loaded shard.

    Args:
        i: Row index within the shard.

    Returns:
        Dict with the sidecar ``record``, the generation time and the
        metadata keys the regenerated episode has beyond the file's.
    """
    spec = _SHARD["specs"][i]
    t0 = time.perf_counter()
    ep = make_episode({**spec, "record_graph": True})
    gen_s = time.perf_counter() - t0

    regen = {
        "x_obs": ep.x_obs.numpy().astype(np.float64).reshape(-1),
        "x_int": ep.x_int.numpy().astype(np.float64).reshape(-1),
        "query_time": ep.query_time.numpy().astype(np.float64).reshape(-1),
        "y_true": ep.y_true.numpy().astype(np.float64).reshape(-1),
    }
    mismatch = []
    if ep.n_vars != int(_SHARD["n_vars"][i]):
        mismatch.append("n_vars")
    # float32 -> float64 is exact and keeps the sign of zero, so comparing the
    # float64 bytes is a bit-for-bit comparison of the released float32 values.
    mismatch += [f for f in _FLOAT_FIELDS if regen[f].tobytes() != _row_slice(f, i).tobytes()]
    query = ep.query_target.numpy().astype(np.int64).reshape(-1)
    if query.tobytes() != _row_slice("query_target", i).astype(np.int64).tobytes():
        mismatch.append("query_target")
    if json.dumps(ep.intervention.to_dict()) != _SHARD["intervention_json"][i]:
        mismatch.append("intervention_json")

    stored = ep.metadata["graph"]
    graph = {k: v for k, v in stored.items() if k != "path"}
    (path,) = stored["path"]  # one query per generic or regime episode
    x_obs_file, x_int_file = _row_slice("x_obs", i), _row_slice("x_int", i)
    family = "regime" if spec["kind"] == "regime" else _family(spec["seed"])
    times = ep.intervention.times
    query_row = int(ep.metadata["query_time_idx"][0])
    record = {
        "idx": int(spec["idx"]),
        "family": family,
        "scm_class": (
            "RegimeSwitchingTemporalSCM" if graph["regime_edges"] is not None else "TemporalSCM"
        ),
        "n_regimes": None if graph["regime_edges"] is None else len(graph["regime_edges"]),
        "graph": graph,
        "treatment": [int(t) for t in ep.intervention.targets],
        "query": int(path["target"]),
        "query_row": query_row,
        "onset": int(min(times)),
        "window_end": int(max(times)),
        "steps_after_window": query_row - int(max(times)),
        "intervention_type": ep.intervention.intervention_type.value,
        "reachable": path["reachable"],
        "min_lag": path["min_lag"],
        "min_hops": path["min_hops"],
        "direct_lags": path["direct_lags"],
        "zeroed_obs": not bool(np.any(x_obs_file)),
        "zeroed_int": not bool(np.any(x_int_file)),
        "verified": not mismatch,
        "mismatch": mismatch,
    }
    file_meta = json.loads(_SHARD["metadata_json"][i] or "{}")
    extra = sorted(
        k
        for k, v in ep.metadata.items()
        if k not in ("y_oracle", "graph") and file_meta.get(k, object()) != v
    )
    # The family is re-derived independently of the graph, so they must agree.
    columns = graph["columns"]
    consistent = (family == "regime") == (graph["regime_edges"] is not None) and (
        family != "chain" or columns == [f"X{j}" for j in range(len(columns))]
    )
    return {"record": record, "gen_s": gen_s, "extra_meta": extra, "consistent": consistent}


def _pool_init() -> None:
    """Pin each worker to one torch thread and silence divergence warnings."""
    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)


def _load_shard(path: Path) -> dict:
    """Load one release shard into flat numpy buffers.

    Args:
        path: Parquet shard.

    Returns:
        Dict with ``lists`` (column -> (values, offsets)), ``n_vars``,
        ``scm_id``, ``intervention_json`` and ``metadata_json``.
    """
    table = pq.read_table(path)
    lists = {}
    for name in (*_FLOAT_FIELDS, "query_target"):
        arr = table.column(name).combine_chunks()
        # .values ignores any slice offset and .offsets are absolute into it.
        lists[name] = (arr.values.to_numpy(), arr.offsets.to_numpy())
    return {
        "lists": lists,
        "n_vars": table.column("n_vars").to_numpy(),
        "scm_id": table.column("scm_id").to_numpy(),
        "intervention_json": table.column("intervention_json").to_pylist(),
        "metadata_json": table.column("metadata_json").to_pylist(),
    }


def _git_state() -> dict:
    """Commit and cleanliness of the package source that produced the sidecar.

    Returns:
        Dict with ``commit``, ``src_dirty`` and sha256 of the two modules that
        define the recorded graph.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    src = Path(dotime.__file__).resolve().parent
    return {
        "commit": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src")),
        # Relative to the checkout, so the record proves which copy of the
        # package ran without publishing the machine's directory layout.
        "dotime_file": _portable(Path(dotime.__file__).resolve()),
        "graph_meta_sha256": hashlib.sha256((src / "graph_meta.py").read_bytes()).hexdigest(),
        "build_sha256": hashlib.sha256((src / "_build.py").read_bytes()).hexdigest(),
    }


def run_suite(name: str, args: argparse.Namespace, config: dict) -> tuple[list[str], dict]:
    """Regenerate, verify and describe every episode of one suite.

    Args:
        name: Suite name, a key of ``_SUITES``.
        args: Parsed CLI arguments.
        config: The parsed release config.

    Returns:
        ``(lines, summary)``: the sidecar's JSON lines in release order and the
        suite's verification summary.

    Raises:
        ValueError: If a shard fails its md5, the manifest seed disagrees with
            the build rule, or the specs do not line up with the file.
    """
    suite_dir = args.cache_dir / f"{name}-1.0.0"
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    position = list(config["suites"]).index(name)
    rule_seed = int(config["seed"]) + 1000 * (position + 1)
    if manifest["seed"] != rule_seed:
        raise ValueError(f"{name}: manifest seed {manifest['seed']} != build rule {rule_seed}")
    cfg = config["suites"][name]
    specs = episode_specs(cfg, manifest["seed"], 1.0)
    if len(specs) != manifest["n_episodes"]:
        raise ValueError(f"{name}: {len(specs)} specs for {manifest['n_episodes']} episodes")

    work = None if args.work_dir is None else args.work_dir / name
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    shards, mismatches, extra_meta = [], [], Counter()
    inconsistent, start, t_suite = [], 0, time.time()
    for si, shard in enumerate(manifest["shards"]):
        rows = shard["n_episodes"]
        shard_specs = specs[start : start + rows]
        if args.limit is not None:
            shard_specs = shard_specs[: max(0, args.limit - start)]
        start += rows
        if not shard_specs:
            continue
        path = suite_dir / shard["file"]
        md5 = _release_io._md5(path)
        if md5 != shard["md5"]:
            raise ValueError(f"md5 mismatch for {path}")
        done = None if work is None else work / f"shard-{si:04d}.json"
        if done is not None and done.exists():
            cached = json.loads(done.read_text())
            if cached["md5"] == md5 and cached["rows"] == len(shard_specs):
                lines += cached["lines"]
                shards.append(cached["summary"])
                mismatches += cached["mismatches"]
                extra_meta.update(cached["extra_meta"])
                inconsistent += cached["inconsistent"]
                print(f"[{name}] shard {si} reused from {done}", flush=True)
                continue

        _SHARD.clear()
        _SHARD.update(_load_shard(path))
        _SHARD["specs"] = shard_specs
        ids = _SHARD["scm_id"][: len(shard_specs)].tolist()
        if ids != [s["idx"] for s in shard_specs]:
            raise ValueError(f"{name} shard {si}: scm_id does not follow the spec order")
        t0 = time.time()
        with get_context("fork").Pool(args.workers, initializer=_pool_init) as pool:
            results = list(pool.imap(_work, range(len(shard_specs)), chunksize=8))
        shard_lines = [json.dumps(r["record"], separators=(",", ":")) for r in results]
        shard_mism = [
            {"idx": r["record"]["idx"], "fields": r["record"]["mismatch"]}
            for r in results
            if r["record"]["mismatch"]
        ]
        shard_extra = Counter(json.dumps(r["extra_meta"]) for r in results)
        shard_incons = [r["record"]["idx"] for r in results if not r["consistent"]]
        summary = {
            "file": shard["file"],
            "md5": md5,
            "rows": len(shard_specs),
            "verified": sum(r["record"]["verified"] for r in results),
            "gen_seconds_mean": float(np.mean([r["gen_s"] for r in results])),
            "wall_seconds": time.time() - t0,
        }
        if done is not None:
            done.write_text(
                json.dumps(
                    {
                        "md5": md5,
                        "rows": len(shard_specs),
                        "lines": shard_lines,
                        "summary": summary,
                        "mismatches": shard_mism,
                        "extra_meta": dict(shard_extra),
                        "inconsistent": shard_incons,
                    }
                )
            )
        lines += shard_lines
        shards.append(summary)
        mismatches += shard_mism
        extra_meta.update(shard_extra)
        inconsistent += shard_incons
        print(
            f"[{name}] shard {si}: {summary['verified']}/{summary['rows']} verified, "
            f"{summary['gen_seconds_mean']:.3f} s/episode, {summary['wall_seconds']:.0f} s wall",
            flush=True,
        )
    return lines, _summarize(
        name,
        cfg,
        manifest,
        lines,
        shards,
        mismatches,
        extra_meta,
        inconsistent,
        time.time() - t_suite,
        args,
    )


def _summarize(
    name, cfg, manifest, lines, shards, mismatches, extra_meta, inconsistent, seconds, args
) -> dict:
    """Verification summary and headline counts of one suite's sidecar.

    Args:
        name: Suite name.
        cfg: The suite's release config entry.
        manifest: The cached manifest.
        lines: The sidecar lines.
        shards: Per-shard summaries.
        mismatches: ``{"idx", "fields"}`` per unverified episode.
        extra_meta: Counter of regenerated metadata keys absent from or
            different in the file, keyed by the sorted key list.
        inconsistent: Episodes whose re-derived family disagrees with the graph.
        seconds: Wall time for the suite.
        args: Parsed CLI arguments.

    Returns:
        The summary dict.
    """
    recs = [json.loads(line) for line in lines]
    field_counts = Counter(f for m in mismatches for f in m["fields"])

    def hist(key, rs=recs):
        return dict(sorted(Counter(str(r[key]) for r in rs).items()))

    zeroed = Counter(
        "both"
        if r["zeroed_obs"] and r["zeroed_int"]
        else "obs_only"
        if r["zeroed_obs"]
        else "int_only"
        if r["zeroed_int"]
        else "neither"
        for r in recs
    )
    by_family = {}
    for fam in sorted({r["family"] for r in recs}):
        rs = [r for r in recs if r["family"] == fam]
        by_family[fam] = {
            "n": len(rs),
            "reads_parents_false": sum(not r["graph"]["reads_parents"] for r in rs),
            "empty_effective_graph": sum(not r["graph"]["edges"] for r in rs),
            "reachable": sum(r["reachable"] for r in rs),
            "min_lag": hist("min_lag", rs),
            "k_sampled": dict(sorted(Counter(str(r["graph"]["k_sampled"]) for r in rs).items())),
            "k_eff": dict(sorted(Counter(str(r["graph"]["k_eff"]) for r in rs).items())),
        }
    return {
        "suite": name,
        "version": manifest["version"],
        "seed": manifest["seed"],
        "config": cfg,
        "n_episodes_manifest": manifest["n_episodes"],
        "n_rows_checked": len(recs),
        "limit": args.limit,
        "fields_compared": [
            "x_obs",
            "x_int",
            "y_true",
            "query_target",
            "query_time",
            "intervention_json",
            "n_vars",
        ],
        "n_verified": sum(r["verified"] for r in recs),
        "n_mismatched": len(mismatches),
        "mismatch_field_counts": dict(field_counts),
        "mismatches": mismatches,
        "family_graph_inconsistent": inconsistent,
        "regenerated_metadata_keys_not_in_file": {k: v for k, v in sorted(extra_meta.items())},
        "zeroed_arms": dict(sorted(zeroed.items())),
        "by_family": by_family,
        "reachable": sum(r["reachable"] for r in recs),
        "min_lag": hist("min_lag"),
        "steps_after_window": {
            "min": min(r["steps_after_window"] for r in recs),
            "median": float(np.median([r["steps_after_window"] for r in recs])),
            "max": max(r["steps_after_window"] for r in recs),
            "zero": sum(r["steps_after_window"] == 0 for r in recs),
        },
        "workers": args.workers,
        "wall_seconds": seconds,
        "shards": shards,
    }


def _write_gz(path: Path, lines: list[str]) -> dict:
    """Write JSON lines as a reproducible gzip stream (no name, mtime 0).

    Args:
        path: Destination.
        lines: One JSON document per line.

    Returns:
        Dict with the path, byte size, sha256 and line count.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write("".join(line + "\n" for line in lines).encode("utf-8"))
    data = buf.getvalue()
    path.write_bytes(data)
    return {
        "path": str(path.relative_to(_REPO) if path.is_relative_to(_REPO) else path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "lines": len(lines),
    }


def main() -> None:
    """Build the graph sidecars and their verification record."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", choices=list(_SUITES), action="append")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "dotime")
    ap.add_argument("--config", type=Path, default=_REPO / "scripts" / "release_config.yaml")
    ap.add_argument("--out-dir", type=Path, default=_REPO / "results" / "reference")
    ap.add_argument(
        "--verification",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "graph_sidecar_verification.json",
    )
    ap.add_argument("--work-dir", type=Path, default=None, help="per-shard outputs, for resuming")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N episodes per suite (smoke run)"
    )
    args = ap.parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = {
        "script": str(Path(__file__).resolve().relative_to(_REPO)),
        "code": _git_state(),
        "suites": {},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.suite or list(_SUITES):
        lines, summary = run_suite(name, args, config)
        summary["sidecar"] = _write_gz(args.out_dir / _SUITES[name], lines)
        out["suites"][name] = summary
        print(
            f"[{name}] {summary['n_verified']}/{summary['n_rows_checked']} verified, "
            f"sidecar {summary['sidecar']['bytes'] / 1e6:.1f} MB",
            flush=True,
        )
    args.verification.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
