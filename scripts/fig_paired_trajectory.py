"""Figure: one paired observational/interventional trajectory with shared noise.

Replaces the paper's fig:traj. The previous PDF was an orphan artifact whose two
panels differed *before* the intervention onset while the caption claimed shared
noise. That mismatch was real: the discrete generators (`DoTime.generate_pair`,
`TSCMPrior`) draw independent noise for the two arms, so their pairs are
interventional twins, not counterfactuals. The continuous-time generator is the
one that shares the noise realisation across arms (``noise=shared_noise`` in
``continuous/extended_prior.py``), so this figure draws its pair from there and
ASSERTS the pre-onset arms are bit-identical before plotting.

Style follows do-over-time-pfn's presentation figures (obs blue, int red,
onset marked, window shaded).

    python scripts/fig_paired_trajectory.py --out fig_trajectory.pdf
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dotime.continuous.extended_prior import ContinuousExtendedPrior

COL_OBS = "#1f77b4"
COL_INT = "#d62728"


def pick_episode(seed0: int, tries: int = 100):
    """First seed giving a mid-trajectory window with a clearly visible effect.

    Deterministic given seed0: seeds are scanned in order and the first
    acceptable one wins, so the figure regenerates identically.
    """
    for i in range(tries):
        seed = seed0 + i
        torch.manual_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = ContinuousExtendedPrior(tscm_structure="back_door", seed=seed).generate_sample()
        xo, xi = s["X_obs_full"], s["X_int"]
        onset = int(s["int_onset_idx"])
        t_len = xo.shape[0]
        n = int(s["num_vars"])
        # t_int_end is the absolute time; the regular schedule has dt=1.
        end = min(round(float(s["t_int_end"])), t_len - 1)
        if onset < 45 or end > t_len - 30 or end - onset < 12:
            continue
        y, a = int(s["query_target"]), int(s["intervention_target"])
        if y == a:
            continue  # the "effect" would just be the clamp itself
        eff = float((xo[onset:, y] - xi[onset:, y]).abs().max())
        if eff < 1.2 * float(xo[:onset, y].std()):
            continue
        # the property this figure exists to show -- and the reason it is drawn
        # from the continuous generator: the arms share one noise realisation.
        assert torch.equal(xo[:onset], xi[:onset]), "arms differ pre-onset"
        assert n > max(y, a)
        return seed, xo, xi, onset, end, a, y, float(s["intervention_value"])
    raise SystemExit(f"no suitable episode in {tries} seeds from {seed0}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed0", type=int, default=20260719)
    ap.add_argument("--out", type=Path, default=Path("fig_trajectory.pdf"))
    args = ap.parse_args(argv)

    seed, xo, xi, onset, end, a, y, val = pick_episode(args.seed0)
    lo = max(0, onset - 45)
    hi = min(xo.shape[0], end + 40)
    ts = range(lo, hi)

    fig, (ax_a, ax_y) = plt.subplots(2, 1, figsize=(3.4, 2.9), sharex=True, height_ratios=[1, 1.55])

    ax_a.plot(ts, xo[lo:hi, a], color=COL_OBS, lw=1.3, alpha=0.85)
    ax_a.plot(ts, xi[lo:hi, a], color=COL_INT, lw=1.5, ls=(0, (3, 1.2)))
    ax_a.set_ylabel("treatment $A$", fontsize=8)

    ax_y.plot(ts, xo[lo:hi, y], color=COL_OBS, lw=1.3, alpha=0.85, label="observational")
    ax_y.plot(
        ts,
        xi[lo:hi, y],
        color=COL_INT,
        lw=1.5,
        ls=(0, (3, 1.2)),
        label=r"counterfactual do$(A{=}v)$, shared noise",
    )
    ax_y.set_ylabel("outcome $Y$", fontsize=8)
    ax_y.set_xlabel("time $t$", fontsize=8)

    for ax in (ax_a, ax_y):
        ax.axvspan(onset, end, color=COL_INT, alpha=0.06, lw=0)
        ax.axvline(onset, color="0.4", ls="--", lw=0.9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    ax_a.text(onset, ax_a.get_ylim()[1], " do$(A{=}v)$", fontsize=7, color="0.3", va="top")
    handles, labels = ax_y.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.align_ylabels()
    fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(
        f"wrote {args.out}  (seed {seed}, onset {onset}, window {onset}-{end}, A=var {a}, Y=var {y}, v={val:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
