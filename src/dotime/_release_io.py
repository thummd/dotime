"""On-disk schema for frozen benchmark suites (the single source of truth).

A released suite is a directory::

    <suite_dir>/
        manifest.json          # provenance + shard checksums
        shard-0000.parquet     # one row per Episode
        shard-0001.parquet
        ...

Each parquet row is one :class:`~dotime.benchmarks.Episode`. Trajectories
are stored row-major-flattened with their ``(length, n_vars)`` shape so they
reconstruct exactly; the intervention is stored as a JSON string via
:meth:`InterventionSpec.to_dict`. The manifest records the package version, seed,
schema version, per-suite tier, and an md5 per shard so a cached copy can be
validated against a Zenodo download.

Schema 1 has the twelve columns of ``_COLUMNS``. Schema 2 adds two optional
columns: ``obs_times`` (the ``T`` observation times) and ``obs_mask`` (``T * N``
booleans in the row-major order of ``x_obs``, ``True`` = observed). A row stores
null when its episode records neither. :func:`write_suite` writes schema 2 only
when an episode needs it, so a suite without observation times, masks or
non-finite values is written byte for byte as in schema 1.

This module is imported lazily (it needs ``pyarrow`` from the ``evaluation``
extra). :func:`write_suite` is used by ``scripts/build_release.py`` and the
``dotime-generate`` CLI; :func:`read_suite` backs ``benchmarks._parse_suite_dir``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from dotime.interventions import InterventionSpec

if TYPE_CHECKING:
    from dotime.benchmarks import BenchmarkSuite, Episode, SuiteMetadata

SCHEMA_VERSION = "1"
#: Manifest schema versions :func:`read_suite` accepts. Readers match versions
#: exactly, so a version is added here rather than bumped in place: the frozen
#: suites are all schema 1 and must stay loadable.
SUPPORTED_SCHEMA_VERSIONS = ("1", "2")
_OBSERVATION_SCHEMA_VERSION = "2"

# Structure labels renamed after the v1 suites were frozen; map old labels from
# released shards to their current names at read time.
_STRUCTURE_ALIASES = {"rct_no_confounding": "bi_variate"}

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
# Schema-2 columns, appended after the schema-1 ones.
_OBSERVATION_COLUMNS = ("obs_times", "obs_mask")


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "reading/writing frozen suites needs the 'evaluation' extra:\n"
            "    pip install 'dotime[evaluation]'"
        ) from exc
    return pa, pq


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _observation_arrow_schema(pa):
    """Explicit Arrow schema of a schema-2 shard.

    Type inference gives a column that is null in every row of a shard the
    ``null`` type, so the shards of one suite would disagree on their schema and
    typed readers (``datasets``, Croissant) would see no list column at all. The
    twelve schema-1 columns keep the types inference gives them in schema 1.

    Args:
        pa: The imported ``pyarrow`` module.

    Returns:
        The ``pyarrow.Schema`` of ``_COLUMNS + _OBSERVATION_COLUMNS``.
    """
    floats, ints = pa.list_(pa.float64()), pa.list_(pa.int64())
    return pa.schema(
        [
            ("scm_id", pa.int64()),
            ("structure", pa.string()),
            ("tier", pa.int64()),
            ("n_vars", pa.int64()),
            ("length", pa.int64()),
            ("x_obs", floats),
            ("x_int", floats),
            ("intervention_json", pa.string()),
            ("query_target", ints),
            ("query_time", floats),
            ("y_true", floats),
            ("metadata_json", pa.string()),
            ("obs_times", floats),
            ("obs_mask", pa.list_(pa.bool_())),
        ]
    )


def _jsonable(value):
    """Coerce a metadata value to something JSON-serializable.

    Args:
        value: A metadata value.

    Returns:
        Tensors, numpy arrays and numpy scalars as (nested) Python lists or
        scalars, anything else unchanged.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def _needs_observation_columns(ep: Episode) -> bool:
    """Whether an episode can only be stored with the schema-2 columns.

    A non-finite value counts even without a mask: schema-1 files promise fully
    observed, finite trajectories, and a reader built for them must refuse the
    file rather than average NaN into its scores.

    Args:
        ep: The episode to store.

    Returns:
        ``True`` if the episode records ``obs_times`` or ``obs_mask``, or has a
        non-finite ``x_obs`` or ``x_int`` value.
    """
    return (
        ep.obs_times is not None
        or ep.obs_mask is not None
        or not bool(torch.isfinite(ep.x_obs).all())
        or not bool(torch.isfinite(ep.x_int).all())
    )


def _episode_to_row(ep: Episode) -> dict:
    """Flatten one :class:`~dotime.benchmarks.Episode` into a parquet row.

    Args:
        ep: The episode to store.

    Returns:
        The row keyed by column name. It has exactly the twelve ``_COLUMNS``
        keys, in that order, unless the episode needs schema 2; then it also
        has ``obs_times`` and ``obs_mask``, each ``None`` when there is nothing
        to store. Without an explicit mask, an ``x_obs`` with non-finite values
        stores ``isfinite(x_obs)``.

    Raises:
        ValueError: If ``obs_times`` does not hold one time per row or
            ``obs_mask`` does not have the shape of ``x_obs``.
    """
    length, n_vars = int(ep.x_obs.shape[0]), int(ep.x_obs.shape[1])
    meta = {k: _jsonable(v) for k, v in ep.metadata.items() if k != "y_oracle"}
    row = {
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
    if not _needs_observation_columns(ep):
        return row
    times, mask = ep.obs_times, ep.obs_mask
    if times is not None and times.numel() != length:
        raise ValueError(f"episode {ep.scm_id} has {times.numel()} obs_times for {length} rows")
    if mask is None and not bool(torch.isfinite(ep.x_obs).all()):
        mask = torch.isfinite(ep.x_obs)
    if mask is not None and tuple(mask.shape) != (length, n_vars):
        raise ValueError(
            f"episode {ep.scm_id} has obs_mask of shape {tuple(mask.shape)}, "
            f"x_obs has shape {(length, n_vars)}"
        )
    row["obs_times"] = (
        None if times is None else times.detach().cpu().to(torch.float64).reshape(-1).tolist()
    )
    row["obs_mask"] = (
        None if mask is None else mask.detach().cpu().to(torch.bool).reshape(-1).tolist()
    )
    return row


def _row_to_episode(row: dict, query_time_encoding: str | None = None) -> Episode:
    """Rebuild one :class:`~dotime.benchmarks.Episode` from a parquet row.

    Args:
        row: One row of a suite shard, keyed by column name. Schema-1 rows have
            no ``obs_times`` / ``obs_mask`` keys, and schema-2 rows hold
            ``None`` in them when the episode records neither.
        query_time_encoding: The suite's declared ``query_time`` encoding. When
            given and the row's metadata records no ``query_time_idx`` (true of
            every frozen v1 file), the exact query rows are resolved from it.

    Returns:
        The reconstructed episode, with ``obs_times`` as float64 of shape
        ``(length,)`` and ``obs_mask`` as bool of shape ``(length, n_vars)``,
        or ``None`` where the row stores null.

    Raises:
        ValueError: If the declared encoding does not match the row's query
            times, or ``obs_times`` / ``obs_mask`` do not match the row's shape.
    """
    from dotime.benchmarks import Episode, query_time_to_index

    length, n_vars = int(row["length"]), int(row["n_vars"])
    x_obs = torch.tensor(row["x_obs"], dtype=torch.float32).reshape(length, n_vars)
    x_int = torch.tensor(row["x_int"], dtype=torch.float32).reshape(length, n_vars)
    y_true = torch.tensor(row["y_true"], dtype=torch.float32)
    metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    # Null cells arrive as None, which torch.tensor rejects, so they stay None.
    obs_times = row.get("obs_times")
    if obs_times is not None:
        if len(obs_times) != length:
            raise ValueError(
                f"row {row['scm_id']} has {len(obs_times)} obs_times for {length} rows"
            )
        obs_times = torch.tensor(obs_times, dtype=torch.float64)
    obs_mask = row.get("obs_mask")
    if obs_mask is not None:
        if len(obs_mask) != length * n_vars:
            raise ValueError(
                f"row {row['scm_id']} has {len(obs_mask)} obs_mask entries for "
                f"{length} x {n_vars} values"
            )
        obs_mask = torch.tensor(obs_mask, dtype=torch.bool).reshape(length, n_vars)
    if query_time_encoding is not None and "query_time_idx" not in metadata:
        # Resolved once, in memory, so every consumer reads the same row; the
        # frozen files themselves are never rewritten.
        metadata["query_time_idx"] = query_time_to_index(
            row["query_time"], length, query_time_encoding
        )
    return Episode(
        x_obs=x_obs,
        x_int=x_int,
        intervention=InterventionSpec.from_dict(json.loads(row["intervention_json"])),
        y_true=y_true,
        query_target=torch.tensor(row["query_target"], dtype=torch.long),
        query_time=torch.tensor(row["query_time"], dtype=torch.float32),
        structure=_STRUCTURE_ALIASES.get(row["structure"], row["structure"]) or None,
        scm_id=int(row["scm_id"]) if int(row["scm_id"]) >= 0 else None,
        metadata={**metadata, "y_oracle": y_true},
        obs_times=obs_times,
        obs_mask=obs_mask,
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
    ``manifest.json`` records provenance and an md5 per shard. The suite is
    schema 2, with ``obs_times`` and ``obs_mask`` columns in every shard, when
    any episode records observation times or a mask or holds a non-finite
    ``x_obs`` / ``x_int`` value. Otherwise it is schema 1 and its shards and
    manifest are byte-identical to those of earlier releases.

    Args:
        meta: Suite metadata (name, version, structures, license).
        episodes: The episodes, in release order.
        dest: Suite directory to create or overwrite.
        package_version: ``dotime`` version recorded in the manifest.
        seed: Suite seed recorded in the manifest.
        shard_size: Maximum number of rows per parquet shard.
        extra_manifest: Extra manifest keys, appended after the standard ones.

    Returns:
        The suite directory.

    Raises:
        ValueError: If an episode's ``obs_times`` or ``obs_mask`` does not
            match the shape of its trajectories.
    """
    pa, pq = _require_pyarrow()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # Decided for the whole suite before any shard is written, so every shard
    # of a schema-2 suite has the same columns whatever its episodes hold.
    observed = any(_needs_observation_columns(ep) for ep in episodes)

    shards: list[dict] = []
    for shard_idx, start in enumerate(range(0, len(episodes), shard_size)):
        chunk = episodes[start : start + shard_size]
        rows = [_episode_to_row(ep) for ep in chunk]
        if observed:
            columns = _COLUMNS + _OBSERVATION_COLUMNS
            table = pa.table(
                {col: [r.get(col) for r in rows] for col in columns},
                schema=_observation_arrow_schema(pa),
            )
        else:
            table = pa.table({col: [r[col] for r in rows] for col in _COLUMNS})
        fname = f"shard-{shard_idx:04d}.parquet"
        pq.write_table(table, dest / fname)
        shards.append({"file": fname, "n_episodes": len(chunk), "md5": _md5(dest / fname)})

    manifest = {
        "name": meta.name,
        "version": meta.version,
        "schema_version": _OBSERVATION_SCHEMA_VERSION if observed else SCHEMA_VERSION,
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
    """Read a suite directory written by :func:`write_suite` into a BenchmarkSuite.

    Args:
        meta: Metadata of the suite; its ``query_time_encoding`` resolves the
            query rows of files that do not record them.
        suite_dir: Directory holding ``manifest.json`` and the shards.

    Returns:
        The suite, one episode per row in shard order.

    Raises:
        FileNotFoundError: If the directory has no ``manifest.json``.
        ValueError: If the manifest declares a schema version outside
            ``SUPPORTED_SCHEMA_VERSIONS``, a shard fails its md5 check, or a
            row does not match the declared ``query_time`` encoding.
    """
    _pa, pq = _require_pyarrow()
    from dotime.benchmarks import BenchmarkSuite

    suite_dir = Path(suite_dir)
    manifest_path = suite_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"cached suite at {suite_dir} is missing manifest.json; "
            "delete it and reload with force_download=True"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"suite {suite_dir} has schema_version {manifest.get('schema_version')!r}, "
            f"this package reads schema versions {', '.join(map(repr, SUPPORTED_SCHEMA_VERSIONS))}"
        )

    episodes: list[Episode] = []
    for shard in manifest["shards"]:
        path = suite_dir / shard["file"]
        if "md5" in shard and _md5(path) != shard["md5"]:
            raise ValueError(f"checksum mismatch for {path}; re-download with force_download=True")
        table = pq.read_table(path)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        for i in range(table.num_rows):
            episodes.append(
                _row_to_episode({name: cols[name][i] for name in cols}, meta.query_time_encoding)
            )

    return BenchmarkSuite(meta, episodes)
