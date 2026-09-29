"""Count per-arm zeroed episodes in cached, md5-verified released suite files.

Reads parquet shards with pyarrow directly (no Episode objects) so the full
100k Generic suite fits in memory. Every shard's md5 is checked against the
suite manifest first, so the counts are tied to the exact released bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DIR_ACC_EPS = 0.1  # dotime.evaluation.DIR_ACC_EPS (near-zero rule)


def md5(path: Path) -> str:
    """Return the hex md5 of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.
    """
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_absmax(list_col) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row max |value|, NaN presence and length of a pyarrow list column.

    Args:
        list_col: A pyarrow ChunkedArray of list<double>.

    Returns:
        (absmax, has_nonfinite, lengths) arrays, one entry per row.

    Raises:
        AssertionError: If a row is empty (reduceat would silently misreport it).
    """
    arr = list_col.combine_chunks()
    offsets = arr.offsets.to_numpy()
    values = arr.values.to_numpy(zero_copy_only=False)
    lengths = np.diff(offsets)
    assert (lengths > 0).all(), "empty trajectory row"
    a = np.abs(values)
    nonfinite = ~np.isfinite(values)
    absmax = np.maximum.reduceat(np.where(np.isfinite(a), a, 0.0), offsets[:-1])
    has_nf = np.add.reduceat(nonfinite.astype(np.int64), offsets[:-1]) > 0
    return absmax, has_nf, lengths


def scan(suite_dir: Path) -> dict:
    """Scan one suite directory and classify every episode by zeroed arms.

    Args:
        suite_dir: Directory holding manifest.json and parquet shards.

    Returns:
        Summary dict with exact counts and per-episode details of one-arm rows.

    Raises:
        ValueError: On an md5 mismatch against the manifest.
    """
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    counts = {"both": 0, "obs_only": 0, "int_only": 0, "neither": 0}
    ynz = {k: 0 for k in counts}  # |y_true| >= eps (enters direction accuracy)
    yzero = {k: 0 for k in counts}  # y_true exactly 0
    by_group: dict[str, dict[str, int]] = {}
    one_arm: list[dict] = []
    n_total = 0
    n_nonfinite = 0
    n_partial_zero_rows = 0
    for shard in manifest["shards"]:
        path = suite_dir / shard["file"]
        got = md5(path)
        if got != shard["md5"]:
            raise ValueError(f"md5 mismatch {path}: {got} != {shard['md5']}")
        t = pq.read_table(path)
        om, onf, _ = row_absmax(t.column("x_obs"))
        im, inf_, _ = row_absmax(t.column("x_int"))
        n_nonfinite += int((onf | inf_).sum())
        y = np.array([v[0] for v in t.column("y_true").to_pylist()], dtype=np.float64)
        ylen = np.array([len(v) for v in t.column("y_true").to_pylist()])
        assert (ylen == 1).all(), "expected one query per episode"
        meta = [json.loads(m) if m else {} for m in t.column("metadata_json").to_pylist()]
        struct = t.column("structure").to_pylist()
        scm = t.column("scm_id").to_pylist()
        nv = t.column("n_vars").to_pylist()
        ln = t.column("length").to_pylist()
        tier = t.column("tier").to_pylist()
        qt = [v[0] for v in t.column("query_target").to_pylist()]
        iv = [json.loads(s) for s in t.column("intervention_json").to_pylist()]
        o0, i0 = om == 0.0, im == 0.0
        cat = np.where(
            o0 & i0, "both", np.where(o0, "obs_only", np.where(i0, "int_only", "neither"))
        )
        # Partial zeroing check: a non-zeroed arm with an exactly-zero time row
        # would mean divergence handling other than whole-arm zeroing, which
        # would make the whole-arm test below undercount.
        for col, zeroed in (("x_obs", om == 0.0), ("x_int", im == 0.0)):
            arr = t.column(col).combine_chunks()
            off = arr.offsets.to_numpy()
            vals = arr.values.to_numpy(zero_copy_only=False)
            for r in np.flatnonzero(~zeroed):
                block = vals[off[r] : off[r + 1]].reshape(ln[r], nv[r])
                if (np.abs(block).max(axis=1) == 0.0).any():
                    n_partial_zero_rows += 1
        for r in range(t.num_rows):
            c = str(cat[r])
            counts[c] += 1
            if abs(y[r]) >= DIR_ACC_EPS:
                ynz[c] += 1
            if y[r] == 0.0:
                yzero[c] += 1
            g = struct[r] or (
                f"regime_{meta[r].get('n_regimes')}" if "n_regimes" in meta[r] else "_all"
            )
            by_group.setdefault(g, {k: 0 for k in counts})[c] += 1
            if c in ("obs_only", "int_only"):
                one_arm.append(
                    {
                        "row": n_total + r,
                        "scm_id": scm[r],
                        "category": c,
                        "structure": struct[r],
                        "tier": tier[r],
                        "n_vars": nv[r],
                        "length": ln[r],
                        "y_true": float(y[r]),
                        "query_target": int(qt[r]),
                        "int_targets": iv[r].get("targets"),
                        "int_type": iv[r].get("intervention_type"),
                        "onset": min(iv[r]["times"]) if iv[r].get("times") else None,
                        "n_int_times": len(iv[r].get("times") or []),
                        "obs_absmax": float(om[r]),
                        "int_absmax": float(im[r]),
                        "metadata": meta[r],
                    }
                )
        n_total += t.num_rows
    assert n_total == manifest["n_episodes"], (n_total, manifest["n_episodes"])
    assert sum(counts.values()) == n_total
    either = counts["both"] + counts["obs_only"] + counts["int_only"]
    return {
        "suite": manifest["name"],
        "version": manifest["version"],
        "seed": manifest["seed"],
        "package_version": manifest.get("package_version"),
        "n_episodes": n_total,
        "md5_verified_shards": len(manifest["shards"]),
        "counts": counts,
        "either_arm_zeroed": either,
        "pct": {k: 100.0 * v / n_total for k, v in counts.items()},
        "pct_either": 100.0 * either / n_total,
        "n_abs_y_ge_eps_by_category": ynz,
        "n_y_exactly_zero_by_category": yzero,
        "n_rows_with_nonfinite": n_nonfinite,
        "n_nonzeroed_arms_with_an_all_zero_time_step": n_partial_zero_rows,
        "by_group": by_group,
        "one_arm_rows": one_arm,
    }


def main() -> None:
    """CLI entry: scan suite dirs and write a JSON summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {}
    for d in args.suite_dirs:
        s = scan(d)
        res[s["suite"] + "-" + s["version"]] = s
        print(
            f"{s['suite']} {s['version']} n={s['n_episodes']} counts={s['counts']} "
            f"either={s['either_arm_zeroed']} ({s['pct_either']:.3f}%) "
            f"|y|>=0.1 by cat={s['n_abs_y_ge_eps_by_category']} y==0 by cat={s['n_y_exactly_zero_by_category']} "
            f"nonfinite={s['n_rows_with_nonfinite']} "
            f"partial_zero_arms={s['n_nonzeroed_arms_with_an_all_zero_time_step']}",
            flush=True,
        )
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
