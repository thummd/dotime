"""Paper figure: 1x3 row of CID panels (bd | fd | iv) for Appendix fig:cid.

NOT runnable from the dotime repo alone: it reuses the DoT-PFN repo's
``scripts/plot_cid_trajectory.py`` (seed selection, rollouts, model inference)
and its checkpoints. To run, place this file next to that script -- or in a
shim directory containing symlinks ``checkpoints/`` and ``configs/`` into the
DoT-PFN repo -- on a machine with the ``s9ho_{bd,fd,iv}_{causal,obs}``
checkpoints (HF ``thummd/do-over-time-pfn``) and the DoT-PFN environment:

    python fig_cid_row.py --prior osc --device cuda:0 --out cid_row_osc.pdf

Committed here for provenance of the paper's Appendix CID figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from plot_cid_trajectory import (
    COL_CAUSAL,
    COL_OBSM,
    PRIOR_CFG,
    PRIOR_TAG,
    STRUCT,
    build_model_batch,
    load_model,
    make_prior,
    plot_cid,
    predict_quantiles,
    sample_scm,
    select_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["osc", "btm"], default="osc")
    ap.add_argument("--structures", nargs="+", default=["bd", "fd", "iv"])
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--n-mc", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-effect", type=float, default=2.5)
    ap.add_argument("--out", default="out/cid_row_osc.pdf")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    with open(repo / PRIOR_CFG[args.prior]) as f:
        cfg = yaml.safe_load(f)
    hardening = cfg["prior"]["hardening"]
    dyn_burn = cfg["prior"].get("dynamics_burn_in", 300)
    tagp = PRIOR_TAG[args.prior]
    int_time = args.T // 2

    n = len(args.structures)
    fig, axes = plt.subplots(1, n, figsize=(2.65 * n, 2.3))

    for ax, key in zip(np.atleast_1d(axes), args.structures, strict=True):
        s = STRUCT[key]
        ck_c = repo / "checkpoints" / f"{tagp}_{key}_causal" / "do_over_time_pfn_best.pt"
        ck_o = repo / "checkpoints" / f"{tagp}_{key}_obs" / "do_over_time_pfn_best.pt"
        assert ck_c.exists(), f"missing causal checkpoint for {key}"
        assert ck_o.exists(), f"missing obs checkpoint for {key}"

        ecp = make_prior(s["name"], args.T, hardening, dyn_burn, s["offset"])
        sim = ecp.batched_sim
        seed = select_seed(
            ecp,
            sim,
            args.T,
            int_time,
            s["offset"],
            candidates=list(range(1, 17)),
            M_probe=min(250, args.n_mc),
            device=args.device,
            target_effect=args.target_effect,
        )
        scm = sample_scm(ecp, sim, seed, args.T, int_time, args.n_mc, args.device)
        assert scm is not None, f"degenerate SCM for {key}"
        qt_idx = min(int_time + s["offset"], args.T - 1)

        pre_A = scm["X_obs"][:, :int_time, scm["a_idx"]].mean(1)
        rep = int(np.argmin(np.abs(pre_A - scm["mu_A"])))
        Xrep = scm["X_obs"][rep]

        pred_c = predict_quantiles(
            load_model(ck_c, args.device),
            *build_model_batch(ecp, Xrep, scm, qt_idx, args.T, False, args.device),
        )
        pred_o = predict_quantiles(
            load_model(ck_o, args.device),
            *build_model_batch(ecp, Xrep, scm, qt_idx, args.T, True, args.device),
        )

        plot_cid(
            ax,
            scm,
            qt_idx,
            [
                ("DoT-PFN$_{int}$", pred_c, COL_CAUSAL, "-"),
                ("DoT-PFN$_{obs}$", pred_o, COL_OBSM, (0, (2, 1))),
            ],
        )
        ax.set_title(s["title"], fontsize=9)
        ax.set_xlabel("$Y$ at query time")  # drop plot_cid's two-line label
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        print(f"[{key}] seed={seed} done")

    # one legend for the row, from the last axis
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    for ax in np.atleast_1d(axes):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel("")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.13),
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
