"""Figure: the mean intervention effect on the outcome builds then decays.

Replaces the paper's fig:decay. The previous PDF plotted mean |Y_int - Y_obs|
over the *discrete* generator, whose two arms carry independent noise; that
difference is the causal effect plus a constant-variance noise floor, so the
curve was flat noise (~0.4) with no visible decay -- exactly the reviewer's
objection. Here the pair is drawn from the continuous-time generator, which
shares one noise realisation across arms, so |Y_int - Y_obs| is the causal
effect alone. Aligned at onset and averaged over episodes it rises during the
intervention window and decays afterwards as the mean-reverting dynamics forget
the clamp.

    python scripts/fig_effect_decay.py --out fig_effect_decay.pdf
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dotime.continuous.extended_prior import ContinuousExtendedPrior

COL = "#DD8452"


def effect_curve(n_episodes: int, max_offset: int, seed0: int):
    """Mean absolute effect |Y_int - Y_obs| by offset from onset, over episodes."""
    per_offset: list[list[float]] = [[] for _ in range(max_offset)]
    used = 0
    for i in range(n_episodes):
        seed = seed0 + i
        torch.manual_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = ContinuousExtendedPrior(tscm_structure="back_door", seed=seed).generate_sample()
        xo, xi = s["X_obs_full"], s["X_int"]
        onset = int(s["int_onset_idx"])
        t_len = xo.shape[0]
        y, a = int(s["query_target"]), int(s["intervention_target"])
        if y == a or not torch.equal(xo[:onset], xi[:onset]):
            continue  # shared-noise sanity: pre-onset arms must coincide
        scale = float(xo[:, y].std())
        if scale < 1e-3:
            continue
        # Effect in units of the outcome's own scale, so effects from SCMs of
        # very different magnitudes are comparable; without this the mean is
        # dominated by a heavy tail of large-magnitude SCMs.
        eff = ((xi[:, y] - xo[:, y]).abs() / scale).numpy()
        for k in range(max_offset):
            t = onset + k
            if t < t_len:
                per_offset[k].append(float(eff[t]))
        used += 1

    offsets = np.arange(max_offset)
    mean = np.array([np.mean(v) if v else np.nan for v in per_offset])
    se = np.array([np.std(v) / np.sqrt(len(v)) if len(v) > 1 else np.nan for v in per_offset])
    return offsets, mean, se, used


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-episodes", type=int, default=800)
    ap.add_argument("--max-offset", type=int, default=36)
    ap.add_argument("--seed0", type=int, default=20260719)
    ap.add_argument("--out", type=Path, default=Path("fig_effect_decay.pdf"))
    args = ap.parse_args(argv)

    offsets, mean, se, used = effect_curve(args.n_episodes, args.max_offset, args.seed0)

    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    ax.fill_between(offsets, mean - se, mean + se, color=COL, alpha=0.25, lw=0)
    ax.plot(offsets, mean, color=COL, lw=1.6, marker="o", ms=2.5)
    ax.axhline(0, color="0.7", lw=0.8, ls=":")
    peak = int(np.nanargmax(mean))
    ax.axvline(peak, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("query offset (steps after onset)", fontsize=8)
    ax.set_ylabel(r"mean $|Y^{\mathrm{int}}-Y^{\mathrm{obs}}|\,/\,\sigma_Y$", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}  ({used} episodes, peak at offset {peak}={mean[peak]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
