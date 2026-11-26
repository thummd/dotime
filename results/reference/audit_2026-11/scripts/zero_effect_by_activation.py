#!/usr/bin/env python
"""Where do the exactly-zero effects of the identification structures come from?

Under shared noise the effect of an intervention on the queried outcome is
exactly zero in 15% to 36% of the episodes of the structures that do have a
path from the treatment to the outcome (``detection_power_2026-10/ident_v1_2.json``).
This script tests one explanation on freshly sampled pairs of the same
generator: two of the four activations of the named-structure sampler, ReLU and
tanh(ReLU), return zero on every negative input, so a mechanism that carries one
of them does not pass a change of its parents on while it is inactive.

For every structure it samples shared-noise pairs with
``TSCMPrior(pair_mode="counterfactual")``, reads the effect on the outcome at the
query step of the released suite (the onset, and one step later for
``mediator``), and counts the zero effects by whether a rectified activation
lies on the directed path from the treatment to the outcome.

    python results/reference/audit_2026-11/scripts/zero_effect_by_activation.py \\
        --out results/reference/audit_2026-11/zero_effect_by_activation.json

The pairs are new draws, not the episodes of a released suite. The script logs
and asserts the per-arm target statistics of every structure before it counts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import torch

from dotime.extended import TSCMPrior
from dotime.tscm_sampler import DEFAULT_ACTIVATIONS, TSCMStructure

ROOT = Path(__file__).resolve().parents[4]
# Structure -> (nodes on the directed path after the treatment, query offset of version 1.2.0).
PATHS = {
    "bi_variate": (("Y",), 0),
    "back_door": (("Y",), 0),
    "bow_graph": (("Y",), 0),
    "instrumental_variable": (("Y",), 0),
    "front_door": (("M", "Y"), 0),
    "confounder_mediator": (("M", "Y"), 0),
    "mediator": (("M", "Y"), 1),
}
RECTIFIED = ("ReLU", "TanhReLU")


def count_structure(name: str, n_pairs: int, seed0: int, t_len: int) -> dict:
    """Sample pairs of one structure and count the zero effects.

    Args:
        name: Structure name of ``TSCMStructure``.
        n_pairs: Number of pairs to sample.
        seed0: Seed of the first pair. Pair ``i`` uses ``seed0 + i``.
        t_len: Trajectory length.

    Returns:
        Counts of pairs and of zero effects, split by whether a rectified
        activation lies on the path, plus the target statistics of both arms.

    Raises:
        AssertionError: If an arm has a degenerate target, or if the path names
            do not match the sampled SCM.
    """
    path, offset = PATHS[name]
    counts = {"rectified_on_path": [0, 0], "no_rectified_on_path": [0, 0]}
    y_obs, y_int, diverged, outside = [], [], 0, 0
    for i in range(n_pairs):
        prior = TSCMPrior(structure=TSCMStructure(name), seed=seed0 + i, pair_mode="counterfactual")
        x_obs, x_int, intervention, scm = prior.generate_pair(T=t_len)
        if float(x_obs.abs().max()) == 0.0 or float(x_int.abs().max()) == 0.0:
            # A diverged arm is stored as zeros and says nothing about the mechanisms.
            diverged += 1
            continue
        assert all(node in scm.mechanisms for node in path), (name, list(scm.mechanisms))
        # generate_pair returns the columns in topological order, so the outcome is found by name.
        col = scm._topo.index("Y")
        step = intervention.times[0] + offset
        if step > t_len - 1:
            # An onset at the last step leaves no later step to query. Reading the onset
            # itself would count the lagged structure's zero by design as a finding.
            outside += 1
            continue
        obs, itv = float(x_obs[step, col]), float(x_int[step, col])
        y_obs.append(obs)
        y_int.append(itv)
        rectified = any(
            type(scm.mechanisms[node].activation).__name__ in RECTIFIED for node in path
        )
        cell = counts["rectified_on_path" if rectified else "no_rectified_on_path"]
        cell[0] += 1
        cell[1] += itv - obs == 0.0
    stats = {}
    for arm, values in (("y_obs", y_obs), ("y_int", y_int)):
        t = torch.tensor(values, dtype=torch.float64)
        stats[arm] = {
            "n": len(values),
            "nonzero_frac": float((t != 0).double().mean()),
            "mean": float(t.mean()),
            "var": float(t.var()),
        }
        # A corrupted target would make every count below meaningless, so it stops the run.
        assert stats[arm]["nonzero_frac"] > 0.9, (name, arm, stats[arm])
        assert stats[arm]["var"] > 0, (name, arm, stats[arm])
    print(f"[target] {name}: {json.dumps(stats)}", flush=True)
    n_zero = counts["rectified_on_path"][1] + counts["no_rectified_on_path"][1]
    n = counts["rectified_on_path"][0] + counts["no_rectified_on_path"][0]
    return {
        "path_after_treatment": list(path),
        "query_offset": offset,
        "n_pairs": n,
        "n_diverged_dropped": diverged,
        "n_query_outside_dropped": outside,
        "n_zero_effect": n_zero,
        "zero_effect_share": n_zero / n,
        "rectified_on_path": {
            "n_pairs": counts["rectified_on_path"][0],
            "n_zero_effect": counts["rectified_on_path"][1],
        },
        "no_rectified_on_path": {
            "n_pairs": counts["no_rectified_on_path"][0],
            "n_zero_effect": counts["no_rectified_on_path"][1],
        },
        "target_stats": stats,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the count for every structure with a path and write the result file.

    Args:
        argv: Command line, ``sys.argv[1:]`` by default.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=1500)
    ap.add_argument("--seed0", type=int, default=20261126)
    ap.add_argument("--t-len", type=int, default=200)
    args = ap.parse_args(argv)

    warnings.simplefilter("ignore", RuntimeWarning)
    torch.set_num_threads(1)

    def git(*args: str) -> str:
        """Output of one git command in the repository, empty if git fails."""
        done = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=False)
        return done.stdout.strip()

    result = {
        "question": "does an exactly-zero effect under shared noise need a rectified activation on the path from the treatment to the outcome?",
        "generator": "dotime.extended.TSCMPrior(pair_mode='counterfactual'), new draws, not released episodes",
        "git_head": git("rev-parse", "HEAD"),
        "src_dirty": bool(git("status", "--porcelain", "--", "src")),
        "activations": sorted(type(a).__name__ for a in DEFAULT_ACTIVATIONS),
        "rectified": list(RECTIFIED),
        "n_pairs_requested": args.n_pairs,
        "seed0": args.seed0,
        "t_len": args.t_len,
        "structures": {},
    }
    for k, name in enumerate(PATHS):
        # Disjoint seed ranges, so no two structures share a draw.
        result["structures"][name] = count_structure(
            name, args.n_pairs, args.seed0 + 100_000 * k, args.t_len
        )
        row = result["structures"][name]
        print(
            f"[count] {name}: zero in {row['n_zero_effect']}/{row['n_pairs']}, without a rectified activation on the path {row['no_rectified_on_path']}",
            flush=True,
        )
    result["n_zero_effect_without_rectified"] = sum(
        s["no_rectified_on_path"]["n_zero_effect"] for s in result["structures"].values()
    )
    result["n_pairs_without_rectified"] = sum(
        s["no_rectified_on_path"]["n_pairs"] for s in result["structures"].values()
    )
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
