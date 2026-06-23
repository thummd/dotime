"""On-disk schema for frozen benchmark suites (the single source of truth).

A released suite is a directory::

    <suite_dir>/
        manifest.json          # provenance + shard checksums
        shard-0000.parquet     # one row per Episode
        shard-0001.parquet
        ...

Each parquet row is one :class:`~causaltimeprior.benchmarks.Episode`. Trajectories
are stored row-major-flattened with their ``(length, n_vars)`` shape so they
reconstruct exactly; the intervention is stored as a JSON string via
:meth:`InterventionSpec.to_dict`. The manifest records the package version, seed,
schema version, per-suite tier, and an md5 per shard so a cached copy can be
validated against a Zenodo download.

This module is imported lazily (it needs ``pyarrow`` from the ``evaluation``
extra). :func:`write_suite` is used by ``scripts/build_release.py`` and the
``ctp-generate`` CLI; :func:`read_suite` backs ``benchmarks._parse_suite_dir``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from causaltimeprior.interventions import InterventionSpec

if TYPE_CHECKING:
    from causaltimeprior.benchmarks import BenchmarkSuite, Episode, SuiteMetadata

SCHEMA_VERSION = "1"

_COLUMNS = (
    "scm_id",
    "structure",
    "tier",
    "n_vars",
    "length",
    "x_obs",
    "x_int",
    "intervention_json",
    "query_target",
    "query_time",
    "y_true",
    "metadata_json",
)


def _require_pyarrow():
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "reading/writing frozen suites needs the 'evaluation' extra:\n"
            "    pip install 'causaltimeprior[evaluation]'"
        ) from exc
    return pa, pq


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324  (integrity check, not security)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _episode_to_row(ep: Episode) -> dict:
    length, n_vars = int(ep.x_obs.shape[0]), int(ep.x_obs.shape[1])
    meta = {k: v for k, v in ep.metadata.items() if k != "y_oracle"}
    return {
        "scm_id": int(ep.scm_id if ep.scm_id is not None else -1),
        "structure": ep.structure or "",
        "tier": int(ep.metadata.get("tier", 0)),
        "n_vars": n_vars,
        "length": length,
        "x_obs": ep.x_obs.detach().cpu().reshape(-1).tolist(),
        "x_int": ep.x_int.detach().cpu().reshape(-1).tolist(),
        "intervention_json": json.dumps(ep.intervention.to_dict()),
        "query_target": ep.query_target.detach().cpu().reshape(-1).tolist(),
        "query_time": ep.query_time.detach().cpu().reshape(-1).tolist(),
        "y_true": ep.y_true.detach().cpu().reshape(-1).tolist(),
        "metadata_json": json.dumps(meta),
    }


def _row_to_episode(row: dict) -> Episode:
    from causaltimeprior.benchmarks import Episode

    length, n_vars = int(row["length"]), int(row["n_vars"])
    x_obs = torch.tensor(row["x_obs"], dtype=torch.float32).reshape(length, n_vars)
    x_int = torch.tensor(row["x_int"], dtype=torch.float32).reshape(length, n_vars)
    y_true = torch.tensor(row["y_true"], dtype=torch.float32)
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    return Episode(
        x_obs=x_obs,
        x_int=x_int,
        intervention=InterventionSpec.from_dict(json.loads(row["intervention_json"])),
        y_true=y_true,
        query_target=torch.tensor(row["query_target"], dtype=torch.long),
        query_time=torch.tensor(row["query_time"], dtype=torch.float32),
        structure=row["structure"] or None,
        scm_id=int(row["scm_id"]) if int(row["scm_id"]) >= 0 else None,
        metadata={**metadata, "y_oracle": y_true},
    )


def write_suite(
    meta: SuiteMetadata,
    episodes: list[Episode],
    dest: str | Path,
    *,
    package_version: str,
    seed: int,
    shard_size: int = 5000,
    extra_manifest: dict | None = None,
) -> Path:
    """Write episodes to a versioned suite directory; return its path.

    Episodes are split into parquet shards of at most ``shard_size`` rows. A
    ``manifest.json`` records provenance and an md5 per shard.
    """
    pa, pq = _require_pyarrow()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    shards: list[dict] = []
    for shard_idx, start in enumerate(range(0, len(episodes), shard_size)):
        chunk = episodes[start : start + shard_size]
        rows = [_episode_to_row(ep) for ep in chunk]
        table = pa.table({col: [r[col] for r in rows] for col in _COLUMNS})
        fname = f"shard-{shard_idx:04d}.parquet"
        pq.write_table(table, dest / fname)
        shards.append({"file": fname, "n_episodes": len(chunk), "md5": _md5(dest / fname)})

    manifest = {
        "name": meta.name,
        "version": meta.version,
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version,
        "seed": seed,
        "n_episodes": len(episodes),
        "structures": list(meta.structures),
        "license": meta.license,
        "shards": shards,
        **(extra_manifest or {}),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


def read_suite(meta: SuiteMetadata, suite_dir: str | Path) -> BenchmarkSuite:
    """Read a suite directory written by :func:`write_suite` into a BenchmarkSuite."""
    pa, pq = _require_pyarrow()
    from causaltimeprior.benchmarks import BenchmarkSuite

    suite_dir = Path(suite_dir)
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"cached suite at {suite_dir} is missing manifest.json; "
            "delete it and reload with force_download=True"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"suite {suite_dir} has schema_version {manifest.get('schema_version')!r}, "
            f"this package reads {SCHEMA_VERSION!r}"
        )

    episodes: list[Episode] = []
    for shard in manifest["shards"]:
        path = suite_dir / shard["file"]
        if "md5" in shard and _md5(path) != shard["md5"]:
            raise ValueError(f"checksum mismatch for {path}; re-download with force_download=True")
        table = pq.read_table(path)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        for i in range(table.num_rows):
            episodes.append(_row_to_episode({name: cols[name][i] for name in cols}))

    return BenchmarkSuite(meta, episodes)
