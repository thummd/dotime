"""Per-arm target QA (nonzero fraction, mean, variance) by zeroed-arm category.

For each released episode, y_int is the stored target (x_int at the query) and
y_obs is the observational level at the same query index and variable. The
repository rule is that these stats are logged and asserted before any build or
benchmark number is trusted; here they also show what each zeroed category
does to the targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def stats(x: np.ndarray) -> dict:
    """Nonzero fraction, mean and variance of one arm's targets.

    Args:
        x: 1-D float array.

    Returns:
        Dict with n, nonzero_frac, mean, var (NaN-free by construction).
    """
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "nonzero_frac": float(np.mean(x != 0)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
    }


def main() -> None:
    """Compute and assert per-arm target stats for one suite directory.

    Raises:
        AssertionError: If the clean (neither-arm-zeroed) category fails the
            nonzero-fraction floor of 0.5 on either arm.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    y_int, y_obs, cat = [], [], []
    for shard in manifest["shards"]:
        t = pq.read_table(args.suite_dir / shard["file"])
        cols = {
            c: t.column(c).to_pylist()
            for c in ("x_obs", "x_int", "n_vars", "length", "y_true", "query_target", "query_time")
        }
        for r in range(t.num_rows):
            n, ln = cols["n_vars"][r], cols["length"][r]
            xo = np.asarray(cols["x_obs"][r], dtype=np.float32).reshape(ln, n)
            xi = np.asarray(cols["x_int"][r], dtype=np.float32).reshape(ln, n)
            q = int(cols["query_target"][r][0])
            v = float(cols["query_time"][r][0])
            # Discrete suites store the integer step (generic: T-1).
            idx = int(v) if v > 1.0 else round(v * (ln - 1))
            y_int.append(float(cols["y_true"][r][0]))
            y_obs.append(float(xo[idx, q]))
            o0, i0 = not np.abs(xo).max() > 0, not np.abs(xi).max() > 0
            cat.append(
                "both" if o0 and i0 else "obs_only" if o0 else "int_only" if i0 else "neither"
            )
    y_int, y_obs, cat = np.array(y_int), np.array(y_obs), np.array(cat)
    out = {"suite": manifest["name"], "version": manifest["version"], "n": int(cat.size)}
    for c in ("all", "neither", "both", "obs_only", "int_only", "either_zeroed"):
        m = (
            np.ones(cat.size, bool)
            if c == "all"
            else cat != "neither"
            if c == "either_zeroed"
            else cat == c
        )
        out[c] = {
            "y_int": stats(y_int[m]),
            "y_obs": stats(y_obs[m]),
            "effect": stats(y_int[m] - y_obs[m]),
        }
        print(c, json.dumps(out[c]), flush=True)
    clean = out["neither"]
    assert clean["y_int"]["nonzero_frac"] >= 0.5, clean
    assert clean["y_obs"]["nonzero_frac"] >= 0.5, clean
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
