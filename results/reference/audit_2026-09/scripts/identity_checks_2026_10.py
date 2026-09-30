#!/usr/bin/env python
"""Post-build identity checks for the 2026-10 suites.

Compares tensors of freshly built suites with the released rows they must
reproduce bit for bit, reading the parquet shards directly so the check does
not depend on the loader or the registry:

1. dot-Identifiability-v1 1.2.0 rows 0..10799 equal the cached 1.1.0 rows,
   except that mediator episodes move their query one step after the onset.
2. dot-Observed-v1 cells ``none+none`` equal the 1.2.0 row of the same
   structure and ``latent_row``.
3. dot-ContinuousIrregular-v1 episodes with ``schedule == "regular"`` equal the
   cached dot-Continuous-v1 1.0.0 rows with the same ``scm_id``.

Usage: python identity_checks.py --v12 <dir> --observed <dir> --irregular <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CACHE = Path.home() / ".cache" / "dotime"
TENSORS = ("x_obs", "x_int", "y_true", "query_target", "query_time")


def load(suite_dir: Path) -> dict[int, dict]:
    """Read every shard of a suite directory into a dict keyed by scm_id.

    Args:
        suite_dir: Directory holding ``*.parquet`` shards.

    Returns:
        ``{scm_id: row}`` with tensors as numpy arrays and metadata parsed.
    """
    rows: dict[int, dict] = {}
    for f in sorted(glob.glob(str(suite_dir / "*.parquet"))):
        t = pq.read_table(f)
        cols = {name: t.column(name).to_pylist() for name in t.column_names}
        for i in range(t.num_rows):
            row = {k: cols[k][i] for k in cols}
            row["metadata"] = json.loads(row.pop("metadata_json"))
            for k in TENSORS:
                row[k] = np.asarray(row[k], dtype=np.float64)
            rows[int(row["scm_id"])] = row
    return rows


def same(a: dict, b: dict, keys=TENSORS) -> list[str]:
    """Names of the tensors that differ between two rows (bit for bit)."""
    bad = []
    for k in keys:
        x, y = a[k], b[k]
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
            bad.append(k)
    return bad


def _qidx(row: dict) -> int:
    """Query row index from the index/T encoded query_time (released 1.1.0 metadata lacks query_time_idx)."""
    return round(float(row["query_time"][0]) * row["length"])


def check_v12(v12_dir: Path) -> dict:
    v11 = load(CACHE / "dot-Identifiability-v1-1.1.0")
    v12 = load(v12_dir)
    report = {
        "n_v11": len(v11),
        "n_v12": len(v12),
        "mismatch": [],
        "mediator_moved": 0,
        "mediator_same_x": 0,
    }
    for sid, old in v11.items():
        new = v12[sid]
        if old["structure"] != new["structure"]:
            report["mismatch"].append((sid, "structure"))
            continue
        if old["structure"] == "mediator":
            bad = same(old, new, ("x_obs", "x_int"))
            if bad:
                report["mismatch"].append((sid, bad))
            else:
                report["mediator_same_x"] += 1
            if _qidx(new) == _qidx(old) + 1:
                report["mediator_moved"] += 1
        else:
            bad = same(old, new)
            if bad or old["intervention_json"] != new["intervention_json"]:
                report["mismatch"].append((sid, bad or ["intervention_json"]))
    report["bow_graph_rows"] = sum(r["structure"] == "bow_graph" for r in v12.values())
    report["diverged"] = sum(bool(r["metadata"].get("diverged")) for r in v12.values())
    report["n_mismatch"] = len(report["mismatch"])
    report["mismatch"] = report["mismatch"][:10]
    return report


def check_observed(obs_dir: Path, v12_dir: Path) -> dict:
    v12 = load(v12_dir)
    obs = load(obs_dir)
    report = {"n_observed": len(obs), "cells": {}, "n_none_none": 0, "mismatch": []}
    for r in obs.values():
        cell = r["metadata"]["obs_cell"]
        report["cells"][cell] = report["cells"].get(cell, 0) + 1
        if cell != "none+none":
            continue
        report["n_none_none"] += 1
        # latent_row is the row of the base suite (Identifiability 1.2.0 shares
        # the suite seed, structures and offsets, so its rows are the base).
        base = v12[int(r["metadata"]["latent_row"])]
        bad = same(base, r)
        if (
            bad
            or base["intervention_json"] != r["intervention_json"]
            or base["structure"] != r["structure"]
        ):
            report["mismatch"].append((int(r["scm_id"]), bad or ["intervention_json"]))
    report["n_mismatch"] = len(report["mismatch"])
    report["mismatch"] = report["mismatch"][:10]
    return report


def check_irregular(irr_dir: Path) -> dict:
    cont = load(CACHE / "dot-Continuous-v1-1.0.0")
    irr = load(irr_dir)
    report = {
        "n_continuous": len(cont),
        "n_irregular": len(irr),
        "schedules": {},
        "mismatch": [],
        "n_regular_checked": 0,
    }
    for sid, r in irr.items():
        sch = r["metadata"].get("schedule")
        report["schedules"][sch] = report["schedules"].get(sch, 0) + 1
        if sch != "regular":
            continue
        base = cont.get(sid)
        if base is None:
            report["mismatch"].append((sid, "missing in continuous"))
            continue
        report["n_regular_checked"] += 1
        bad = same(base, r)
        if bad or base["intervention_json"] != r["intervention_json"]:
            report["mismatch"].append((sid, bad or ["intervention_json"]))
    report["n_mismatch"] = len(report["mismatch"])
    report["mismatch"] = report["mismatch"][:10]
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v12", type=Path)
    ap.add_argument("--observed", type=Path)
    ap.add_argument("--irregular", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = {}
    if args.v12:
        out["identifiability_1_2_0_vs_1_1_0"] = check_v12(args.v12)
        print(json.dumps(out["identifiability_1_2_0_vs_1_1_0"], default=str))
    if args.observed and args.v12:
        out["observed_none_none_vs_1_2_0"] = check_observed(args.observed, args.v12)
        print(json.dumps(out["observed_none_none_vs_1_2_0"], default=str))
    if args.irregular:
        out["irregular_regular_vs_continuous_1_0_0"] = check_irregular(args.irregular)
        print(json.dumps(out["irregular_regular_vs_continuous_1_0_0"], default=str))
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
