#!/usr/bin/env python
"""Re-derive the §7.1 transfer-probe facts from the released per-episode JSONs.

Chamber: split episodes into the two alternating toggle classes (changepoints
alternate 20 and 30 samples apart) and report the per-class correlation sign,
predicted level and true level per checkpoint. Then bound the within-window
standard deviation of the prediction from the error variance, using
var(err) = s_p^2 + s_g^2 - 2 r s_p s_g across checkpoints on the same episode.

Warfarin: per-seed sign counts over subjects and the cross-seed correlation of
per-subject r values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REL = Path("results/reference/transfer")


def load_chamber(kind: str, seed: int, extra_dir: Path | None) -> list[dict]:
    """Load one checkpoint's per-episode chamber metrics.

    Args:
        kind: ``linear`` or ``mixed`` mechanism prior.
        seed: Training seed.
        extra_dir: Folder with the JSONs that ``results/reference/transfer/``
            does not carry (the mixed-mechanism checkpoints), or None.

    Returns:
        List of per-episode dicts.

    Raises:
        FileNotFoundError: If the JSON is in neither folder.
    """
    p = REL / f"chamber_p13b_pnc000_{kind}_seed{seed}_rpm_in.json"
    if not p.exists() and extra_dir is not None:
        p = extra_dir / p.name
    if not p.exists():
        raise FileNotFoundError(f"{p.name}: pass --extra-dir with the training runs' JSONs")
    return json.loads(p.read_text())["per_episode"]


def main() -> None:
    """Compute and write the transfer facts.

    Raises:
        AssertionError: If checkpoints disagree on the episode ordering.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    # The mixed-mechanism chamber JSONs come from the training repository
    # (results/phase14c_multiseed/), which is not part of this one.
    ap.add_argument("--extra-dir", type=Path, default=None)
    args = ap.parse_args()
    out: dict = {"chamber": {}, "warfarin": {}}
    ckpts = [(k, s) for k in ("linear", "mixed") for s in range(5)]
    data = {c: load_chamber(*c, args.extra_dir) for c in ckpts}
    ref = [e["changepoint"] for e in data[ckpts[0]]]
    for c in ckpts:
        assert [e["changepoint"] for e in data[c]] == ref, c
    cls = np.arange(len(ref)) % 2
    per = {}
    for c in ckpts:
        eps = data[c]
        row = {}
        for k in (0, 1):
            sub = [e for e, z in zip(eps, cls, strict=True) if z == k]
            row[f"class{k}"] = {
                "mean_r": float(np.mean([e["pearson_r"] for e in sub])),
                "frac_positive": float(np.mean([e["pearson_r"] > 0 for e in sub])),
                "mean_pred_level": float(np.mean([e["mean_pred"] for e in sub])),
                "mean_true_level": float(np.mean([e["mean_gt"] for e in sub])),
            }
        row["mean_r_all"] = float(np.mean([e["pearson_r"] for e in eps]))
        per[f"{c[0]}_seed{c[1]}"] = row
    out["chamber"]["per_checkpoint"] = per

    # Error variance per checkpoint and episode: V = rmse^2 - (level gap)^2.
    n_ep = len(ref)
    V = np.array(
        [
            [
                data[c][i]["rmse"] ** 2 - (data[c][i]["mean_pred"] - data[c][i]["mean_gt"]) ** 2
                for i in range(n_ep)
            ]
            for c in ckpts
        ]
    )
    R = np.array([[data[c][i]["pearson_r"] for i in range(n_ep)] for c in ckpts])
    sd = np.sqrt(np.clip(V, 0, None))
    spread = sd.max(0) - sd.min(0)
    # Where one checkpoint has r > 0.9 and another r < -0.9, the SD difference
    # bounds the sum of their within-window prediction SDs (s_p+ + s_p-).
    bounds, sg = [], []
    for i in range(n_ep):
        pos, neg = np.where(R[:, i] > 0.9)[0], np.where(R[:, i] < -0.9)[0]
        if len(pos) and len(neg):
            bounds.append(float(sd[neg, i].max() - sd[pos, i].min()))
            sg.append(float(np.median(sd[:, i])))
    out["chamber"]["error_sd_spread_across_checkpoints"] = {
        "median": float(np.median(spread)),
        "max": float(np.max(spread)),
        "n_episodes": n_ep,
    }
    out["chamber"]["opposite_sign_episodes"] = len(bounds)
    if bounds:
        out["chamber"]["implied_sum_sp_bound_rpm"] = {
            "median": float(np.median(bounds)),
            "max": float(np.max(bounds)),
        }
        out["chamber"]["error_sd_rpm_median"] = float(np.median(sg))
    print(json.dumps(out["chamber"], indent=1)[:3000])

    # Warfarin.
    rs = {}
    for s in range(5):
        d = json.loads((REL / f"warfarin_p13b_pnc000_linear_seed{s}.json").read_text())
        rs[s] = np.array([p["pearson_r_cp"] for p in d["per_subject"]])
        out["warfarin"][f"seed{s}"] = {
            "n_subjects": int(rs[s].size),
            "n_positive": int((rs[s] > 0).sum()),
            "mean_r": float(rs[s].mean()),
            "rmse": d["aggregate"]["pooled_rmse_cp"],
            "subject_mean_reference_rmse": d["aggregate"]["naive_pooled_rmse_cp"],
        }
    out["warfarin"]["corr_of_subject_r_seed3_vs_others"] = {
        f"seed{s}": float(np.corrcoef(rs[3], rs[s])[0, 1]) for s in (0, 1, 2, 4)
    }
    seed_means = np.array([rs[s].mean() for s in range(5)])
    t = seed_means.mean() / (seed_means.std(ddof=1) / np.sqrt(5))
    out["warfarin"]["seed_level_t"] = float(t)
    print(json.dumps(out["warfarin"], indent=1))
    Path("results/reference/audit_2026-09/transfer_facts.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
