#!/usr/bin/env python
"""Validity metrics for dot-Continuous-v1 1.0.0.

The continuous generator encodes ``query_time`` as ``idx / (T - 1)`` (verified:
``y_true == x_int[round(v * (T - 1)), q]`` on every non-zero target), so the
observational level at the query is read at that index here rather than through
``evaluation.query_obs_levels``, which rounds ``v * T``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dotime.benchmarks import load_benchmark


def arm_stats(x: np.ndarray) -> dict:
    """Nonzero fraction, mean and variance of one target arm.

    Args:
        x: 1-D array of targets.

    Returns:
        Summary dict.
    """
    return {
        "n": int(x.size),
        "nonzero_frac": float(np.mean(x != 0)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
    }


def main() -> None:
    """Compute invariant pass rates and per-arm stats, then write JSON.

    Raises:
        AssertionError: If the query index convention or target QA fails.
    """
    eps = list(load_benchmark("dot-Continuous-v1"))
    n = len(eps)
    pre_eq = do_ok = zeroed = self_q = idx_ok = 0
    y_int, y_obs = [], []
    for ep in eps:
        t_len = ep.x_int.shape[0]
        t0 = int(min(ep.intervention.times))
        a = int(ep.intervention.targets[0])
        q = int(ep.query_target[0])
        qi = round(float(ep.query_time[0]) * (t_len - 1))
        yt = float(ep.y_true.reshape(-1)[0])
        idx_ok += abs(float(ep.x_int[qi, q]) - yt) < 1e-6
        zeroed += not bool(torch.any(ep.x_obs != 0))
        pre_eq += bool(torch.equal(ep.x_obs[:t0], ep.x_int[:t0]))
        do_ok += abs(float(ep.x_int[t0, a]) - float(ep.intervention.values)) < 1e-5
        self_q += bool(ep.is_self_query)
        y_int.append(yt)
        y_obs.append(float(ep.x_obs[qi, q]))
    y_int, y_obs = np.array(y_int), np.array(y_obs)
    assert idx_ok == n, f"query index convention failed on {n - idx_ok} episodes"
    qa = {
        "y_obs_level": arm_stats(y_obs),
        "y_int_level": arm_stats(y_int),
        "effect": arm_stats(y_int - y_obs),
    }
    print("[continuous target QA]", qa)
    assert qa["y_obs_level"]["nonzero_frac"] >= 0.5, qa
    assert qa["y_int_level"]["nonzero_frac"] >= 0.5, qa
    out = {
        "suite": "dot-Continuous-v1",
        "version": "1.0.0",
        "n_episodes": n,
        "query_index_convention": "idx = round(query_time * (T - 1)); y_true matches x_int there on all episodes",
        "zeroed_frac": zeroed / n,
        "pre_onset_equal_frac": pre_eq / n,
        "do_value_at_onset_frac": do_ok / n,
        "self_query_frac": self_q / n,
        "arms": qa,
    }
    Path("results/reference/audit_2026-09/continuous_validity.json").write_text(
        json.dumps(out, indent=2)
    )
    print({k: v for k, v in out.items() if k != "arms"})


if __name__ == "__main__":
    main()
