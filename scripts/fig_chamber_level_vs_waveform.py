"""Figure: level recovery vs. waveform tracking on the Causal Chambers probe.

Makes the two-part claim in the paper's real-world transfer section visible:
the synthetic-trained PFN relocates the *operating level* after an intervention
(which is what the large RMSE lift over a last-value baseline measures) but does
not track the within-window *waveform* (which is what the near-zero mean Pearson
correlation measures). Both panels are drawn from the released per-episode
metrics in ``results/reference/transfer/`` -- no model is run here.

    python scripts/fig_chamber_level_vs_waveform.py --out fig_level_waveform.pdf
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Aggregate naive last-value RMSE for rpm_in, from transfer_multiseed_aggregate.json.
NAIVE_RMSE = 1264.775


def load(results_dir: Path):
    rows = []
    pattern = str(results_dir / "chamber_*_linear_seed*_rpm_in.json")
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            episodes = json.load(fh)["per_episode"]
        for e in episodes:
            rows.append((e["mean_pred"], e["mean_gt"], e["pearson_r"], e["rmse"]))
    if not rows:
        raise SystemExit(f"no per-episode files matched {pattern}")
    pred, gt, r, rmse = (np.array(c) for c in zip(*rows, strict=True))
    return pred, gt, r, rmse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results/reference/transfer"))
    ap.add_argument("--out", type=Path, default=Path("fig_level_waveform.pdf"))
    args = ap.parse_args(argv)

    pred, gt, r, _rmse = load(args.results_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    # --- (a) level: where the two distributions sit, on the scale of the error
    # a last-value baseline makes. The gap the model closes dwarfs the gap it
    # leaves, which is the honest form of "it recovers the level".
    lo = min(pred.min(), gt.min()) - 60
    bins = np.linspace(lo, gt.max() + 60, 45)
    ax1.hist(pred, bins=bins, color="#4C72B0", alpha=0.85, label="PFN prediction")
    ax1.hist(gt, bins=bins, color="#DD8452", alpha=0.85, label="ground truth")
    ax1.annotate(
        "",
        xy=(pred.mean(), -0.13),
        xytext=(gt.mean(), -0.13),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        annotation_clip=False,
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.0),
    )
    ax1.text(
        (pred.mean() + gt.mean()) / 2,
        -0.30,
        f"residual gap {gt.mean() - pred.mean():.0f} "
        f"({100 * (gt.mean() - pred.mean()) / gt.mean():.0f}% undershoot)",
        ha="center",
        va="top",
        fontsize=7,
        color="0.25",
        transform=ax1.get_xaxis_transform(),
    )
    ax1.set_title("(a) level recovery", fontsize=9)
    ax1.set_xlabel("post-intervention mean level (rpm_in)", fontsize=8)
    ax1.set_ylabel("episodes", fontsize=8)
    ax1.legend(fontsize=7, frameon=False, loc="upper left")
    ax1.text(
        0.98,
        0.95,
        f"last-value baseline errs by {NAIVE_RMSE:.0f}\n(off scale, to the left)",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="0.35",
    )

    # --- (b) waveform: |r| is large but its SIGN is a coin flip, so the model
    # produces a strongly-shaped curve that is as often inverted as aligned.
    ax2.hist(r, bins=np.linspace(-1, 1, 41), color="#55A868", alpha=0.9)
    ax2.axvline(0.0, color="0.4", lw=0.9, ls=":")
    ax2.axvline(r.mean(), color="#C44E52", lw=1.4, label=f"mean $r={r.mean():.2f}$")
    ax2.set_title("(b) waveform tracking", fontsize=9)
    ax2.set_xlabel("per-episode Pearson $r$", fontsize=8)
    ax2.set_ylabel("episodes", fontsize=8)
    ax2.legend(fontsize=7, frameon=False, loc="upper center")
    ax2.text(
        0.5,
        0.62,
        f"|r| > 0.9 in {100 * np.mean(np.abs(r) > 0.9):.0f}% of episodes,\n"
        f"but only {100 * np.mean(r > 0):.0f}% are positive",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="0.3",
    )

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(
        f"  n={len(pred)}  pred level {pred.mean():.0f}+-{pred.std():.0f}  "
        f"true {gt.mean():.0f}+-{gt.std():.0f}  "
        f"corr(pred,true)={np.corrcoef(pred, gt)[0, 1]:+.3f}  "
        f"mean r={r.mean():.3f}  median r={np.median(r):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
