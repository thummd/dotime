#!/usr/bin/env python
"""Validity invariants of the 2026-10 suites, as pass rates.

The same invariants that ``ident_analysis.py`` scored on the Identifiability
files, generalised to suites with missing cells, latent columns dropped from
the release and continuous-time grids:

- **pre-onset agreement**: ``x_obs`` and ``x_int`` are equal before the first
  intervened row (``NaN`` equal to ``NaN``, since an observed suite masks the
  same cells in both arms),
- **do-value placement**: for a hard intervention the treatment column of
  ``x_int`` holds the do-value at the onset row,
- **effect identity**: the stored ``y_causal_effect`` equals ``y_true`` minus
  the factual level at the query (the latent level on an observed suite),
- **zeroed arms**: neither arm is all zero.

Usage::

    python suite_invariants_2026_10.py --suite dot-Wide-v1 --version 1.0.0 \
        --out ../suite_invariants_2026_10.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dotime.benchmarks import load_benchmark
from dotime.evaluation import query_obs_levels


def _nan_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Element-wise equality that treats two NaN cells as equal."""
    return bool(torch.equal(torch.nan_to_num(a, nan=-1e30), torch.nan_to_num(b, nan=-1e30)))


def invariants(episodes) -> dict:
    """Pass rates of the four invariants over a list of episodes.

    Args:
        episodes: Episodes of one suite.

    Returns:
        Counts and rates, pooled and per structure for the pre-onset check.
    """
    n = len(episodes)
    pre_eq = zeroed = 0
    hard_n = hard_ok = 0
    eff_n = eff_ok = 0
    per_struct = defaultdict(lambda: [0, 0])
    for ep in episodes:
        t0 = int(min(ep.intervention.times))
        a = int(ep.intervention.targets[0])
        if not torch.any(torch.nan_to_num(ep.x_obs) != 0) or not torch.any(
            torch.nan_to_num(ep.x_int) != 0
        ):
            zeroed += 1
        same = _nan_equal(ep.x_obs[:t0], ep.x_int[:t0])
        pre_eq += same
        per_struct[ep.structure][0] += same
        per_struct[ep.structure][1] += 1
        if str(getattr(ep.intervention, "intervention_type", "hard")) in (
            "hard",
            "InterventionType.HARD",
        ):
            val = ep.intervention.values
            val = (
                float(val.reshape(-1)[0])
                if torch.is_tensor(val)
                else float(np.asarray(val).reshape(-1)[0])
            )
            hard_n += 1
            hard_ok += bool(abs(float(ep.x_int[t0, a]) - val) < 1e-5)
        ce = ep.metadata.get("y_causal_effect")
        if ce is not None:
            y_obs = float(np.asarray(query_obs_levels(ep)).reshape(-1)[0])
            y_int = float(np.asarray(ep.y_true).reshape(-1)[0])
            eff_n += 1
            eff_ok += bool(abs(float(np.asarray(ce).reshape(-1)[0]) - (y_int - y_obs)) < 1e-5)
    return {
        "n_episodes": n,
        "zeroed_frac": zeroed / n,
        "pre_onset_equal_frac": pre_eq / n,
        "pre_onset_equal_by_structure": {
            s: round(v[0] / v[1], 4) for s, v in sorted(per_struct.items())
        },
        "hard_interventions": hard_n,
        "do_value_in_treatment_col_frac_of_hard": (hard_ok / hard_n) if hard_n else None,
        "episodes_with_effect_field": eff_n,
        "effect_identity_frac_of_those": (eff_ok / eff_n) if eff_n else None,
    }


def main() -> None:
    """Score one suite version and merge the result into the output JSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    suite = load_benchmark(args.suite, version=args.version)
    result = invariants(list(suite))
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    out[f"{args.suite}-{args.version}"] = result
    args.out.write_text(json.dumps(out, indent=1))
    print(f"{args.suite} {args.version}: {json.dumps(result)}")


if __name__ == "__main__":
    main()
