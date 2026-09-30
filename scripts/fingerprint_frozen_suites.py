#!/usr/bin/env python
"""Fingerprint stratified rows of every released DoTime suite version.

Regenerates about 25 rows per released suite version with the current
package, compares every column with the cached, md5-verified release file and
records the exact column hashes, a portable summary and the per-column release
match in ``tests/data/frozen_fingerprints.json``. ``tests/test_frozen_fingerprints.py``
regenerates those rows on every test run, so a change to a default code path
that alters a released suite fails there.

    python scripts/fingerprint_frozen_suites.py
    python scripts/fingerprint_frozen_suites.py --full

``--full`` instead regenerates every row of dot-Continuous-v1 1.0.0 and
dot-Identifiability-v1 1.1.0 and writes the per-column comparison to
``results/reference/audit_2026-09/frozen_regeneration.json``.

The cached suites are read from ``$DOTIME_CACHE`` or ``~/.cache/dotime``.
Columns that differ from the release for a documented reason (see
``KNOWN_DIFFERENCES``) are expected and verified row by row. Any other
difference is a regeneration finding: the script prints every finding and exits
with status 1 without writing anything. Rerunning the script on an unchanged
tree rewrites a byte-identical file, so ``git diff`` shows exactly what a code
change did to the pinned rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from dotime import __version__, _release_io
from dotime._build import episode_specs, make_episode
from dotime._fingerprint import portable_summary, reference_env, row_hashes
from dotime.reference._realignment import load_realignment

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_AUDIT = _ROOT / "results" / "reference" / "audit_2026-09"
_DEFAULT_OUT = _ROOT / "tests" / "data" / "frozen_fingerprints.json"
_FULL_OUT = _AUDIT / "frozen_regeneration.json"
_SCAN = _AUDIT / "half_diverged_released_scan.json"
_SIDECAR = _ROOT / "results" / "reference" / "dot-Identifiability-v1.0.0_realignment.jsonl"

# (suite, version, config file). The seed of each suite is read from its
# cached manifest and must equal build_release.py's derivation, base seed +
# 1000 * (position in the config + 1).
VERSIONS = (
    ("dot-Identifiability-v1", "1.0.0", "release_config.yaml"),
    ("dot-Identifiability-v1", "1.1.0", "release_config_v1_1.yaml"),
    ("dot-RegimeSwitch-v1", "1.0.0", "release_config.yaml"),
    ("dot-Continuous-v1", "1.0.0", "release_config.yaml"),
    ("dot-Generic-100k", "1.0.0", "release_config.yaml"),
    ("dot-Identifiability-v1", "1.2.0", "release_config_v1_2.yaml"),
    ("dot-SeasonalTrend-v1", "1.0.0", "release_config_seasonal_trend.yaml"),
    ("dot-Wide-v1", "1.0.0", "release_config_wide.yaml"),
    ("dot-Observed-v1", "1.0.0", "release_config_observed_v1.yaml"),
    ("dot-ContinuousIrregular-v1", "1.0.0", "release_config_continuous_irregular.yaml"),
)
FULL_VERSIONS = ("dot-Continuous-v1-1.0.0", "dot-Identifiability-v1-1.1.0")
IDENT_V1_0 = "dot-Identifiability-v1-1.0.0"

KNOWN_DIFFERENCES = {
    "metadata_json": (
        "Every version. The released metadata predates query_time_idx, diverged, "
        "pair_mode, query_offset_range, self_query, query_in_window and window_end_idx, "
        "so the current make_episode metadata_json is pinned instead. Checked: every "
        "released key is present with an equal value, except y_causal_effect of "
        "dot-Identifiability-v1 1.0.0, which must equal the released y_true (v1 "
        "erratum: the effect was taken against the causally masked, all-zero "
        "observational arm)."
    ),
    "x_obs": (
        "dot-Identifiability-v1 1.0.0 only (v1 erratum): the released x_obs is in "
        "topological column order with hidden variables unmasked. Checked: realigning "
        "the released x_obs with its row of the realignment sidecar reproduces the "
        "current x_obs bit for bit."
    ),
    "structure": (
        "dot-Identifiability-v1 1.0.0 only: the bi_variate structure was released "
        "under its old label rct_no_confounding, which _release_io._STRUCTURE_ALIASES "
        "maps to bi_variate at read time. Checked: the alias maps the released label "
        "to the current one."
    ),
}


def _cache_root() -> Path:
    """Root of the cached suites, resolved as :func:`dotime.benchmarks.load_benchmark` does.

    Returns:
        ``$DOTIME_CACHE`` when set, else ``~/.cache/dotime``.
    """
    env = os.environ.get("DOTIME_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "dotime"


def _jsonable(value: Any) -> Any:
    """Round-trip a value through JSON, turning tuples into lists.

    Args:
        value: A JSON-able value.

    Returns:
        What ``json.loads(json.dumps(value))`` returns, which is what a stored
        spec compares equal to.
    """
    return json.loads(json.dumps(value))


def load_version(name: str, version: str, config: str, cache: Path) -> dict[str, Any]:
    """Resolve one released suite version: config, seed, specs and cached files.

    Args:
        name: Suite name.
        version: Released version.
        config: Release config file name under ``scripts/``.
        cache: Root of the cached suites.

    Returns:
        Dict with the version ``key``, suite ``cfg``, ``suite_seed``, ``specs``,
        ``suite_dir``, ``manifest``, ``shards`` (``(file, first index, rows)``
        per shard) and the ``config`` file name.

    Raises:
        SystemExit: If the cached copy is missing, or if the config, the
            manifest and the seed derivation disagree. Each of these is a
            finding about the release, not about the regeneration.
    """
    key = f"{name}-{version}"
    cfg_all = yaml.safe_load((_SCRIPTS / config).read_text())
    names = list(cfg_all["suites"])
    cfg = cfg_all["suites"][name]
    derived = int(cfg_all["seed"]) + 1000 * (names.index(name) + 1)
    suite_dir = cache / key
    if not (suite_dir / "manifest.json").exists():
        raise SystemExit(f"no cached copy of {key} under {cache}")
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    problems = []
    if cfg["version"] != version or manifest["version"] != version:
        problems.append(f"versions: config {cfg['version']}, manifest {manifest['version']}")
    if int(manifest["seed"]) != derived:
        problems.append(f"manifest seed {manifest['seed']} != derived {derived}")
    specs = episode_specs(cfg, int(manifest["seed"]), 1.0)
    if len(specs) != int(manifest["n_episodes"]):
        problems.append(f"{len(specs)} specs for {manifest['n_episodes']} released episodes")
    if problems:
        raise SystemExit(f"{key}: {'; '.join(problems)}")
    shards, start = [], 0
    for shard in manifest["shards"]:
        shards.append((shard["file"], start, int(shard["n_episodes"])))
        start += int(shard["n_episodes"])
    return {
        "key": key,
        "cfg": cfg,
        "suite_seed": int(manifest["seed"]),
        "specs": specs,
        "suite_dir": suite_dir,
        "manifest": manifest,
        "shards": shards,
        "config": config,
    }


def read_shard(info: dict[str, Any], file: str, columns: list[str] | None = None):
    """Read one md5-verified shard of a cached suite.

    Args:
        info: Output of :func:`load_version`.
        file: Shard file name.
        columns: Columns to read, all by default.

    Returns:
        The pyarrow table.

    Raises:
        SystemExit: If the shard's md5 differs from the manifest, because every
            comparison must be tied to the exact released bytes.
    """
    path = Path(info["suite_dir"]) / file
    expected = {s["file"]: s["md5"] for s in info["manifest"]["shards"]}[file]
    got = _release_io._md5(path)
    if got != expected:
        raise SystemExit(f"md5 mismatch for {path}: {got} != {expected}")
    return pq.read_table(path, columns=columns)


def read_column(info: dict[str, Any], column: str, shards=None) -> list:
    """Read one column of a cached suite, in suite order.

    Args:
        info: Output of :func:`load_version`.
        column: Column name.
        shards: Entries of ``info["shards"]`` to read, all by default.

    Returns:
        The column's values as Python objects.
    """
    out: list = []
    for file, _start, _n in shards or info["shards"]:
        out += read_shard(info, file, [column]).column(column).to_pylist()
    return out


def released_rows(info: dict[str, Any], indices) -> dict[int, dict[str, Any]]:
    """Read released rows by suite index.

    Args:
        info: Output of :func:`load_version`.
        indices: Suite row indices.

    Returns:
        ``{idx: row}`` with every released column, as pyarrow returns them.
    """
    wanted = sorted({int(i) for i in indices})
    out = {}
    for file, start, n in info["shards"]:
        here = [i for i in wanted if start <= i < start + n]
        if here:
            table = read_shard(info, file)
            for i in here:
                out[i] = {c: table.column(c)[i - start].as_py() for c in table.column_names}
    return out


def compare_row(key: str, idx: int, current: dict, released: dict, sidecar) -> tuple[dict, list]:
    """Compare a regenerated row with the released row, column by column.

    Args:
        key: Suite version key, e.g. ``"dot-Identifiability-v1-1.0.0"``.
        idx: Suite row index.
        current: The row :func:`dotime._release_io._episode_to_row` writes now.
        released: The released parquet row.
        sidecar: ``{scm_id: row}`` realignment sidecar for Identifiability 1.0.0.

    Returns:
        ``(release_match, findings)``: whether each column's hash matches the
        release, and one message per difference that no entry of
        :data:`KNOWN_DIFFERENCES` explains.
    """
    cur_h, rel_h = row_hashes(current), row_hashes(released)
    match = {col: cur_h[col] == rel_h.get(col) for col in cur_h}
    findings = []
    if list(released) != list(current):
        findings.append(f"{key} row {idx}: columns {list(released)} != {list(current)}")
    for col, ok in match.items():
        if ok or col not in released:
            continue
        if col == "metadata_json":
            findings += _check_metadata(key, idx, current, released)
        elif col == "x_obs" and key == IDENT_V1_0:
            findings += _check_realigned_x_obs(key, idx, current, released, sidecar)
        elif col == "structure" and key == IDENT_V1_0:
            alias = _release_io._STRUCTURE_ALIASES.get(released["structure"])
            if alias != current["structure"]:
                findings.append(
                    f"{key} row {idx}: structure {released['structure']!r} -> "
                    f"{current['structure']!r} is not a documented alias"
                )
        else:
            findings.append(f"{key} row {idx}: column {col} differs from the release")
    return match, findings


def _check_metadata(key: str, idx: int, current: dict, released: dict) -> list[str]:
    """Check that the released metadata is contained in the current metadata.

    Args:
        key: Suite version key.
        idx: Suite row index.
        current: The regenerated row.
        released: The released row.

    Returns:
        One message per released key that is missing or differs now, except
        the documented 1.0.0 effect, which must equal the released ``y_true``.
    """
    cur = json.loads(current["metadata_json"])
    rel = json.loads(released["metadata_json"]) if released["metadata_json"] else {}
    out = []
    for k, v in rel.items():
        if key == IDENT_V1_0 and k == "y_causal_effect":
            if v != released["y_true"]:
                out.append(f"{key} row {idx}: released y_causal_effect {v} != y_true")
        elif k not in cur or cur[k] != v:
            out.append(f"{key} row {idx}: released metadata {k}={v!r}, now {cur.get(k)!r}")
    return out


def _check_realigned_x_obs(key: str, idx: int, current: dict, released: dict, sidecar) -> list:
    """Check that the 1.0.0 x_obs erratum explains an x_obs difference exactly.

    Args:
        key: Suite version key.
        idx: Suite row index.
        current: The regenerated row.
        released: The released row.
        sidecar: ``{scm_id: row}`` realignment sidecar.

    Returns:
        A message unless the released ``x_obs``, permuted to canonical order
        and with hidden columns zeroed, equals the current ``x_obs`` bit for bit.
    """
    row = sidecar.get(int(released["scm_id"]))
    if row is None:
        return [f"{key} row {idx}: x_obs differs and the sidecar has no row"]
    shape = (int(released["length"]), int(released["n_vars"]))
    rel = np.asarray(released["x_obs"], dtype=np.float32).reshape(shape)
    fixed = rel[:, [int(c) for c in row["canonical_perm"]]]
    fixed[:, [int(h) for h in row["hidden_canonical"]]] = 0.0
    cur = np.asarray(current["x_obs"], dtype=np.float32).reshape(shape)
    if fixed.tobytes() != cur.tobytes():
        return [f"{key} row {idx}: x_obs differs beyond the documented realignment"]
    return []


# --------------------------------------------------------------------------- #
# Row selection
# --------------------------------------------------------------------------- #


def _zeroed(info: dict[str, Any], shards=None) -> tuple[np.ndarray, np.ndarray]:
    """Which rows of a cached suite have an all-zero observational or interventional arm.

    Args:
        info: Output of :func:`load_version`.
        shards: Entries of ``info["shards"]`` to scan, all by default.

    Returns:
        ``(obs_zero, int_zero)`` boolean arrays over the scanned rows, in suite
        order.
    """
    obs, intv = [], []
    for file, _start, _n in shards or info["shards"]:
        table = read_shard(info, file, ["x_obs", "x_int"])
        for col, out in (("x_obs", obs), ("x_int", intv)):
            arr = table.column(col).combine_chunks()
            offsets = arr.offsets.to_numpy()
            values = np.abs(arr.values.to_numpy(zero_copy_only=False))
            out.append(np.maximum.reduceat(values, offsets[:-1]) == 0.0)
    return np.concatenate(obs), np.concatenate(intv)


def _intervention_kind(intervention_json: str) -> str:
    """Label an intervention by type, and by profile when time-varying.

    Args:
        intervention_json: A released ``intervention_json`` value.

    Returns:
        ``"hard"``, ``"soft"`` or ``"time_varying:<profile>"``.
    """
    spec = json.loads(intervention_json)
    kind = spec["intervention_type"]
    if spec["values"]["kind"] == "profile":
        kind += ":" + spec["values"]["name"]
    return kind


def _blocks(info: dict[str, Any], field: str) -> dict[Any, list[int]]:
    """Group suite indices by a spec field, keeping suite order.

    Args:
        info: Output of :func:`load_version`.
        field: Spec key, e.g. ``"structure"`` or ``"num_regimes"``.

    Returns:
        ``{value: [idx, ...]}`` in first-seen order.
    """
    groups: dict[Any, list[int]] = {}
    for spec in info["specs"]:
        groups.setdefault(spec[field], []).append(int(spec["idx"]))
    return groups


def _add(selected: dict[int, list[str]], idx: int, reason: str) -> None:
    """Record why a row is selected.

    Args:
        selected: ``{idx: reasons}`` being built.
        idx: Suite row index.
        reason: Why the row is fingerprinted.
    """
    selected.setdefault(int(idx), []).append(reason)


def _anchors(info: dict[str, Any], field: str) -> dict[int, list[str]]:
    """Select the first, middle and last row of every group of a suite.

    Args:
        info: Output of :func:`load_version`.
        field: Spec key that defines the groups (structure or regime density).

    Returns:
        ``{idx: reasons}`` with three rows per group.
    """
    selected: dict[int, list[str]] = {}
    for value, idxs in _blocks(info, field).items():
        for where, idx in (
            ("first", idxs[0]),
            ("middle", idxs[len(idxs) // 2]),
            ("last", idxs[-1]),
        ):
            _add(selected, idx, f"{field} {value} ({where})")
    return selected


def _cover(selected, candidates, labels, per_group: int, prefix: str) -> None:
    """Add the first candidate of each label not seen yet, up to ``per_group`` rows.

    Args:
        selected: ``{idx: reasons}`` being built.
        candidates: Suite indices in scan order.
        labels: ``{idx: label}`` covering every candidate.
        per_group: Maximum number of rows to add.
        prefix: Reason prefix, e.g. the structure name.
    """
    seen: set = set()
    for idx in candidates:
        if len(seen) == per_group:
            return
        if labels[idx] not in seen:
            seen.add(labels[idx])
            _add(selected, idx, f"{prefix}: {labels[idx]}")


def select_identifiability(info: dict[str, Any], scan: dict) -> dict[int, list[str]]:
    """Pick Identifiability rows: three per structure plus zeroed and retried rows.

    Args:
        info: Output of :func:`load_version`.
        scan: The audit scan of zeroed arms in the released v1.0.0 files.

    Returns:
        ``{idx: reasons}``.

    Raises:
        SystemExit: If the zeroed rows found here disagree with the audit scan.
    """
    selected = _anchors(info, "structure")
    obs0, int0 = _zeroed(info)
    one_arm = {int(i): ("obs_only" if obs0[i] else "int_only") for i in np.flatnonzero(obs0 ^ int0)}
    if info["key"] in scan:
        audited = {r["row"]: r["category"] for r in scan[info["key"]]["one_arm_rows"]}
        if audited != one_arm:
            raise SystemExit(f"{info['key']}: one-arm rows {one_arm} != audit scan {audited}")
    for idx, category in one_arm.items():
        _add(selected, idx, f"one arm zeroed ({category})")
    for idx in np.flatnonzero(obs0 & int0)[:2]:
        _add(selected, int(idx), "both arms zeroed")
    if int(info["cfg"].get("stability_retries", 0)) > 0:
        # Rows whose first attempt diverged ship a retried seed, the only path
        # that exercises identifiability_retry_seed.
        found = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for spec in info["specs"][::37]:
                if make_episode({**spec, "stability_retries": 0}).metadata["diverged"]:
                    _add(selected, spec["idx"], "first attempt diverged (retried seed)")
                    found += 1
                    if found == 3:
                        break
    return selected


def select_regime(info: dict[str, Any]) -> dict[int, list[str]]:
    """Pick RegimeSwitch rows: per density, three anchors and each intervention kind.

    Args:
        info: Output of :func:`load_version`.

    Returns:
        ``{idx: reasons}``.
    """
    kinds = dict(enumerate(_intervention_kind(s) for s in read_column(info, "intervention_json")))
    selected = _anchors(info, "num_regimes")
    for density, idxs in _blocks(info, "num_regimes").items():
        _cover(selected, idxs, kinds, 6, f"num_regimes {density}")
    return selected


def select_continuous(info: dict[str, Any]) -> dict[int, list[str]]:
    """Pick Continuous rows: per structure, three anchors, each (tier, self-query) cell and edges.

    The edges are the latest intervention onset, which is where the query
    window's upper bound was fixed without changing the draws, and a query at
    the onset itself.

    Args:
        info: Output of :func:`load_version`.

    Returns:
        ``{idx: reasons}``.
    """
    from dotime.benchmarks import _SUITE_REGISTRY, query_time_to_index

    encoding = _SUITE_REGISTRY[info["manifest"]["name"]].query_time_encoding
    tiers = read_column(info, "tier")
    queries = read_column(info, "query_target")
    ivs = [json.loads(s) for s in read_column(info, "intervention_json")]
    rows = [
        query_time_to_index(q, n, encoding)[0]
        for q, n in zip(read_column(info, "query_time"), read_column(info, "length"), strict=True)
    ]
    labels = {
        i: f"tier {tiers[i]}, self_query {queries[i][0] in ivs[i]['targets']}"
        for i in range(len(tiers))
    }
    selected = _anchors(info, "structure")
    for structure, idxs in _blocks(info, "structure").items():
        _cover(selected, idxs, labels, 6, f"structure {structure}")
        late = max(idxs, key=lambda i: (min(ivs[i]["times"]), -i))
        _add(selected, late, f"structure {structure}: latest onset {min(ivs[late]['times'])}")
        at_onset = next(i for i in idxs if rows[i] == min(ivs[i]["times"]))
        _add(selected, at_onset, f"structure {structure}: query at the onset")
    schedules = _metadata_labels(info, "schedule")
    if schedules:
        # Irregular grids: one row per schedule kind, since the grid draws and
        # the sub-stepped integration only exist on the non-regular ones.
        _cover(selected, sorted(schedules), schedules, 8, "schedule")
    return selected


def _metadata_labels(info: dict[str, Any], key: str) -> dict[int, str]:
    """Read one metadata key of every row as a label.

    Args:
        info: Output of :func:`load_version`.
        key: Metadata key, e.g. ``"schedule"`` or ``"obs_cell"``.

    Returns:
        ``{idx: str(value)}`` for the rows whose metadata carries the key.
    """
    out = {}
    for i, raw in enumerate(read_column(info, "metadata_json")):
        meta = json.loads(raw) if raw else {}
        if key in meta:
            out[i] = str(meta[key])
    return out


def select_observed(info: dict[str, Any]) -> dict[int, list[str]]:
    """Pick Observed rows: per structure three anchors, every observation cell, two cells per structure.

    Args:
        info: Output of :func:`load_version`.

    Returns:
        ``{idx: reasons}``.
    """
    cells = _metadata_labels(info, "obs_cell")
    selected = _anchors(info, "structure")
    _cover(selected, sorted(cells), cells, 16, "obs_cell")
    for structure, idxs in _blocks(info, "structure").items():
        # The cells are cell-major, so the reversed block reaches the cells the
        # suite-wide cover above did not take from this structure.
        _cover(selected, list(reversed(idxs)), cells, 2, f"structure {structure}")
    return selected


def select_wide(info: dict[str, Any]) -> dict[int, list[str]]:
    """Pick Wide rows: each (intervention kind, graph-size bucket), the extremes and the anchors.

    Args:
        info: Output of :func:`load_version`.

    Returns:
        ``{idx: reasons}``.
    """
    kinds = [_intervention_kind(s) for s in read_column(info, "intervention_json")]
    n_vars = read_column(info, "n_vars")
    labels = {
        i: f"{kinds[i]}, released n_vars {'<= 15' if n <= 15 else '<= 25' if n <= 25 else '> 25'}"
        for i, n in enumerate(n_vars)
    }
    selected: dict[int, list[str]] = {}
    for where, idx in (("first", 0), ("middle", len(n_vars) // 2), ("last", len(n_vars) - 1)):
        _add(selected, idx, f"suite ({where})")
    _cover(selected, range(len(n_vars)), labels, 9, "wide")
    _add(selected, int(np.argmin(n_vars)), f"fewest released variables ({min(n_vars)})")
    _add(selected, int(np.argmax(n_vars)), f"most released variables ({max(n_vars)})")
    return selected


def select_generic(info: dict[str, Any], scan: dict) -> dict[int, list[str]]:
    """Pick Generic rows: each SCM class x intervention kind, plus zeroed rows.

    ``DoTime.sample_scm`` decides the SCM class with the first draw of the
    prior's own generator, so the class of attempt 0 is read off the episode
    seed without simulating anything.

    Args:
        info: Output of :func:`load_version`.
        scan: The audit scan of zeroed arms in the released v1.0.0 files.

    Returns:
        ``{idx: reasons}``.

    Raises:
        SystemExit: If constructing the prior draws from its generator, if a
            row labelled regime samples another class, or if the zeroed rows
            of shard 0 disagree with the audit scan.
    """
    from dotime import DoTime
    from dotime.regime_switching import RegimeSwitchingTemporalSCM

    prior = DoTime(seed=12345)
    if not torch.equal(
        prior.generator.get_state(), torch.Generator().manual_seed(12345).get_state()
    ):
        raise SystemExit("DoTime() draws before sample_scm; the class shortcut is invalid")
    chain, regime = prior.chain_prob, prior.chain_prob + prior.regime_switching_prob
    shard0 = info["shards"][:1]
    obs0, int0 = _zeroed(info, shard0)
    kinds = [_intervention_kind(s) for s in read_column(info, "intervention_json", shard0)]
    labels = {}
    for i in range(2000):
        if not (obs0[i] or int0[i]):
            r = torch.rand(1, generator=torch.Generator().manual_seed(info["specs"][i]["seed"]))
            cls = "chain" if r.item() < chain else ("regime" if r.item() < regime else "diverse")
            labels[i] = f"{cls} SCM, {kinds[i]}"
    selected: dict[int, list[str]] = {}
    for cls in ("diverse", "chain", "regime"):
        cands = [i for i, lab in labels.items() if lab.startswith(cls)]
        _cover(selected, cands, labels, 5, "generic")
    for idx in np.flatnonzero(obs0 & int0)[:3]:
        _add(selected, int(idx), "both arms zeroed")
    audited = scan[info["key"]]["one_arm_rows"]
    found = {int(i) for i in np.flatnonzero(obs0 ^ int0)}
    if found != {r["row"] for r in audited if r["row"] < len(obs0)}:
        raise SystemExit(f"{info['key']}: one-arm rows of shard 0 disagree with the audit scan")
    for category in ("obs_only", "int_only"):
        for row in [r for r in audited if r["category"] == category][:2]:
            _add(selected, row["row"], f"one arm zeroed ({category})")
    _add(selected, len(info["specs"]) - 1, "last row")
    for idx, reasons in selected.items():
        if any("regime SCM" in r for r in reasons):
            seed = info["specs"][idx]["seed"]
            torch.manual_seed(seed)
            if not isinstance(DoTime(seed=seed).sample_scm(), RegimeSwitchingTemporalSCM):
                raise SystemExit(f"{info['key']} row {idx}: labelled regime, sampled another")
    return selected


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


def fingerprint(cache: Path, out: Path) -> int:
    """Fingerprint stratified rows of every released version.

    Args:
        cache: Root of the cached suites.
        out: Output JSON path.

    Returns:
        0 when every difference from the releases is documented, 1 otherwise
        (nothing is written then).
    """
    scan = json.loads(_SCAN.read_text())
    sidecar = load_realignment(_SIDECAR)
    versions: dict[str, Any] = {}
    findings: list[str] = []
    for name, version, config in VERSIONS:
        info = load_version(name, version, config, cache)
        key = info["key"]
        if name in ("dot-Identifiability-v1", "dot-SeasonalTrend-v1"):
            selected = select_identifiability(info, scan)
        elif name == "dot-Observed-v1":
            selected = select_observed(info)
        elif name == "dot-RegimeSwitch-v1":
            selected = select_regime(info)
        elif name in ("dot-Continuous-v1", "dot-ContinuousIrregular-v1"):
            selected = select_continuous(info)
        elif name == "dot-Wide-v1":
            selected = select_wide(info)
        else:
            selected = select_generic(info, scan)
        released = released_rows(info, selected)
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for idx in sorted(selected):
                spec = info["specs"][idx]
                ep = make_episode(spec)
                current = _release_io._episode_to_row(ep)
                match, found = compare_row(key, idx, current, released[idx], sidecar)
                findings += found
                rows.append(
                    {
                        "idx": idx,
                        "reasons": selected[idx],
                        "spec": _jsonable(spec),
                        "hashes": row_hashes(current),
                        "portable": _jsonable(portable_summary(ep)),
                        "release_match": match,
                    }
                )
        differing = Counter(c for r in rows for c, ok in r["release_match"].items() if not ok)
        print(
            f"[fingerprint] {key}: {len(rows)} rows; rows differing from the release by "
            f"column: {dict(differing) or 'none'}",
            flush=True,
        )
        versions[key] = {
            "suite": name,
            "version": version,
            "config": config,
            "suite_seed": info["suite_seed"],
            "n_episodes": len(info["specs"]),
            "shard_md5": {s["file"]: s["md5"] for s in info["manifest"]["shards"]},
            "rows": rows,
        }
    if findings:
        print("[fingerprint] FINDINGS (nothing written):", *findings, sep="\n  ")
        return 1
    payload = {
        "schema": 1,
        "generated_by": "scripts/fingerprint_frozen_suites.py",
        "package_version": __version__,
        "reference_env": reference_env(),
        "columns": list(_release_io._COLUMNS),
        "known_differences": KNOWN_DIFFERENCES,
        "versions": versions,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[fingerprint] wrote {out}")
    return 0


def _full_chunk(task: dict[str, Any]) -> dict[str, Any]:
    """Regenerate one chunk of rows and compare them with the release (worker).

    Args:
        task: ``info`` (output of :func:`load_version` without its specs),
            ``file`` and ``start`` of the shard and the ``specs`` of the chunk.

    Returns:
        The number of rows, per-column counts of rows matching the release and
        any findings.
    """
    torch.set_num_threads(1)
    warnings.simplefilter("ignore", RuntimeWarning)
    info, start = task["info"], task["start"]
    table = read_shard(info, task["file"])
    counts: Counter = Counter()
    findings: list[str] = []
    for spec in task["specs"]:
        idx = int(spec["idx"])
        released = {c: table.column(c)[idx - start].as_py() for c in table.column_names}
        current = _release_io._episode_to_row(make_episode(spec))
        match, found = compare_row(info["key"], idx, current, released, {})
        counts.update(c for c, ok in match.items() if ok)
        findings += found
    return {"n": len(task["specs"]), "matches": dict(counts), "findings": findings}


def full(cache: Path, out: Path, workers: int) -> int:
    """Regenerate every row of the versions in :data:`FULL_VERSIONS`.

    Args:
        cache: Root of the cached suites.
        out: Output JSON path.
        workers: Worker processes.

    Returns:
        0 when every difference is documented, 1 otherwise (nothing is written).
    """
    results: dict[str, Any] = {}
    findings: list[str] = []
    columns = list(_release_io._COLUMNS)
    for name, version, config in VERSIONS:
        if f"{name}-{version}" not in FULL_VERSIONS:
            continue
        info = load_version(name, version, config, cache)
        light = {k: v for k, v in info.items() if k != "specs"}
        tasks = [
            {"info": light, "file": file, "start": start, "specs": info["specs"][lo : lo + 250]}
            for file, start, n in info["shards"]
            for lo in range(start, start + n, 250)
        ]
        t0 = time.time()
        counts: Counter = Counter()
        n_rows = 0
        # spawn, not fork: torch may already run threads in this process.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as ex:
            for res in ex.map(_full_chunk, tasks):
                n_rows += res["n"]
                counts.update(res["matches"])
                findings += res["findings"]
        key = info["key"]
        results[key] = {
            "suite": name,
            "version": version,
            "config": config,
            "suite_seed": info["suite_seed"],
            "n_rows": n_rows,
            "shard_md5": {s["file"]: s["md5"] for s in info["manifest"]["shards"]},
            "rows_matching_release": {c: counts.get(c, 0) for c in columns},
            "seconds": round(time.time() - t0, 1),
        }
        differing = {c: n_rows - counts.get(c, 0) for c in columns if counts.get(c, 0) < n_rows}
        print(
            f"[fingerprint --full] {key}: {n_rows} rows in {results[key]['seconds']}s; "
            f"rows differing from the release by column: {differing or 'none'}",
            flush=True,
        )
    if findings:
        print("[fingerprint --full] FINDINGS (nothing written):", *findings[:50], sep="\n  ")
        return 1
    payload = {
        "generated_by": "scripts/fingerprint_frozen_suites.py --full",
        "package_version": __version__,
        "env": reference_env(),
        "known_differences": KNOWN_DIFFERENCES,
        "unexplained_differences": 0,
        "versions": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"[fingerprint --full] wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the stratified fingerprint, or the full regeneration with ``--full``.

    Args:
        argv: Command-line arguments, ``sys.argv`` by default.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--full", action="store_true", help=f"Regenerate every row of {', '.join(FULL_VERSIONS)}."
    )
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    ap.add_argument("--cache-dir", type=Path, default=None, help="Root of the cached suites.")
    ap.add_argument("--workers", type=int, default=4, help="Worker processes for --full.")
    args = ap.parse_args(argv)
    cache = args.cache_dir or _cache_root()
    if args.full:
        return full(cache, args.out or _FULL_OUT, args.workers)
    return fingerprint(cache, args.out or _DEFAULT_OUT)


if __name__ == "__main__":
    sys.exit(main())
