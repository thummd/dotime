"""Re-score the published dot-Generic-100k CPU rows under zeroed-episode filters.

Gate: with no filter, every row must reproduce results/reference/generic.json
(pooled RMSE and dir_acc/dir_n_valid) exactly, using the same float32 arithmetic
as dotime.reference.reference_table.run_baseline. Only then are the filtered
numbers (drop both-arm zeroed, drop either-arm zeroed) meaningful.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from dotime import _release_io, baselines
from dotime.evaluation import direction_accuracy
from dotime.reference.reference_table import CPU_BASELINES


def score(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """Pooled RMSE and level direction accuracy, as reference_table computes them.

    Args:
        pred: float32 predictions.
        tgt: float32 targets.

    Returns:
        Dict with n, pooled_rmse, dir_acc, dir_n_valid.
    """
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    da = direction_accuracy(torch.from_numpy(pred), torch.from_numpy(tgt))
    return {
        "n": int(pred.size),
        "pooled_rmse": rmse,
        "dir_acc": da["accuracy"],
        "dir_n_valid": da["n_valid"],
    }


def main() -> None:
    """Score every baseline over the md5-verified release, then filter."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    manifest = json.loads((args.suite_dir / "manifest.json").read_text())
    models = {b: baselines.get(b) for b in CPU_BASELINES}
    preds = {b: [] for b in CPU_BASELINES}
    tgts, cats = [], []
    t0 = time.time()
    for si, shard in enumerate(manifest["shards"]):
        path = args.suite_dir / shard["file"]
        if _release_io._md5(path) != shard["md5"]:
            raise ValueError(f"md5 mismatch {path}")
        t = pq.read_table(path)
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        for r in range(t.num_rows):
            ep = _release_io._row_to_episode({c: cols[c][r] for c in cols})
            o0 = float(ep.x_obs.abs().max()) == 0.0
            i0 = float(ep.x_int.abs().max()) == 0.0
            cats.append(
                "both" if o0 and i0 else "obs_only" if o0 else "int_only" if i0 else "neither"
            )
            tgts.append(torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).numpy())
            for b, m in models.items():
                preds[b].append(
                    torch.as_tensor(m.predict(ep), dtype=torch.float32).reshape(-1).cpu().numpy()
                )
        print(f"shard {si} done ({time.time() - t0:.0f}s)", flush=True)
    tgt = np.concatenate(tgts)
    cat = np.array(cats)
    published = {r["baseline"]: r for r in json.loads(args.published.read_text())["rows"]}
    subsets = {
        "all (published protocol)": np.ones(tgt.size, bool),
        "drop both-arm zeroed": cat != "both",
        "drop either-arm zeroed": cat == "neither",
        "only obs_only": cat == "obs_only",
        "only int_only": cat == "int_only",
    }
    out = {
        "n": int(tgt.size),
        "category_counts": {c: int((cat == c).sum()) for c in np.unique(cat)},
        "rows": {},
    }
    for b in CPU_BASELINES:
        p = np.concatenate(preds[b])
        res = {name: score(p[mask], tgt[mask]) for name, mask in subsets.items()}
        pub = published[b]
        res["gate_reproduces_published"] = (
            res["all (published protocol)"]["pooled_rmse"] == pub["pooled_rmse"]
            and res["all (published protocol)"]["dir_acc"] == pub["dir_acc"]
            and res["all (published protocol)"]["dir_n_valid"] == pub["dir_n_valid"]
        )
        # Every history baseline predicts from x_obs alone, so an all-zero
        # observational arm yields a zero prediction; record how often.
        oo = subsets["only obs_only"]
        res["obs_only_pred_exactly_zero"] = int((p[oo] == 0.0).sum())
        out["rows"][b] = res
        print(b, json.dumps(res), flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
