#!/usr/bin/env python
"""Grade the s13 predictions under the pre-registered decision rule.

Reads the per-episode predictions written by ``score_prereg.py`` and computes
the primary endpoint of ``PREREG.md`` (Section 7), its label, and the secondary
endpoints (Section 8). Nothing here depends on which way the result comes out:
the labels, intervals and summary sentences are the registered ones.

Usage::

    python results/reference/prereg_2026-09/scripts/analyze_prereg.py \
        --pred results/reference/prereg_2026-09/s13_predictions.parquet \
        --out-dir results/reference/prereg_2026-09
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRIMARY_STRUCTURES = (
    "bi_variate",
    "back_door",
    "confounder_mediator",
    "front_door",
    "instrumental_variable",
    "mediator",
)
TRAINING_STRUCTURES = ("back_door", "front_door", "instrumental_variable")
CONTROL = "bow_graph"
SEEDS = (42, 43, 44)
EPS = 0.1
BOOT_N = 10_000
BOOT_SEED = 20261001
EQUIV_MARGIN = 0.02
# Two-sided Student t quantiles for the seed interval (degrees of freedom = pairs - 1).
T_QUANTILES = {1: (12.706, 6.314), 2: (4.303, 2.920), 3: (3.182, 2.353)}
SUMMARY = {
    "SUPPORTED": (
        "Interventional training changes effect-sign accuracy by {d:+.3f}, with a 95% confidence "
        "interval from {lo:+.3f} to {hi:+.3f}. This supports a narrower claim than the withdrawn "
        "one: joint models on one prior without lagged edges."
    ),
    "REVERSED": (
        "Interventional training changes effect-sign accuracy by {d:+.3f}, with a 95% confidence "
        "interval from {lo:+.3f} to {hi:+.3f}. The ablation is better, so the withdrawal stands."
    ),
    "EQUIVALENT": (
        "Interventional training changes effect-sign accuracy by {d:+.3f}, with a 95% confidence "
        "interval from {lo:+.3f} to {hi:+.3f}, so the withdrawal stands."
    ),
    "INCONCLUSIVE": (
        "Interventional training changes effect-sign accuracy by {d:+.3f}, with a 95% confidence "
        "interval from {lo:+.3f} to {hi:+.3f}, so the withdrawal stands."
    ),
    "NOT_COMPLETED": "The re-test did not complete by the registered deadline.",
}


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    """Load the scoring parquet into column arrays.

    Args:
        path: The ``s13_predictions.parquet`` file.

    Returns:
        Dict of column name to array.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


def correct(pred: np.ndarray, y_true: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """Registered correctness: sign(pred - y_obs) equals sign(y_true - y_obs).

    A prediction equal to ``y_obs`` or a non-finite one counts as incorrect.

    Args:
        pred: Predicted levels.
        y_true: Counterfactual levels.
        y_obs: Factual levels at the query.

    Returns:
        Boolean array.
    """
    ok = np.isfinite(pred) & (pred != y_obs)
    return ok & (np.sign(pred - y_obs) == np.sign(y_true - y_obs))


def arm_table(cols: dict[str, np.ndarray], mask: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    """Per (arm, seed) correctness over the masked episodes, aligned by scm_id.

    Args:
        cols: The prediction columns.
        mask: Episode mask over the per-episode reference order.

    Returns:
        ``{(arm, seed): correct}`` with rows in reference episode order.
    """
    ref_ids = np.unique(cols["scm_id"])
    out: dict[tuple[str, int], np.ndarray] = {}
    for arm in ("int", "B", "A"):
        for seed in SEEDS:
            sel = (cols["arm"] == arm) & (cols["seed"] == seed)
            if not sel.any():
                continue
            order = np.argsort(cols["scm_id"][sel])
            ids = cols["scm_id"][sel][order]
            if not np.array_equal(ids, ref_ids):
                raise ValueError(f"{arm} seed {seed}: episode set differs from the reference")
            c = correct(
                cols["pred"][sel][order], cols["y_true"][sel][order], cols["y_obs"][sel][order]
            )
            out[(arm, seed)] = c[mask]
    return out


def contrast(table, a: str, b: str, seeds) -> dict:
    """Seed-level and episode-bootstrap intervals of acc(a) - acc(b).

    Args:
        table: Output of :func:`arm_table`.
        a: First arm.
        b: Second arm.
        seeds: Seeds with both arms present.

    Returns:
        Dict with the estimate, per-seed differences and both interval families.
    """
    d_s = np.array([table[(a, s)].mean() - table[(b, s)].mean() for s in seeds])
    d = float(d_s.mean())
    n = len(seeds)
    res: dict = {
        "estimate": d,
        "per_seed": {str(s): float(x) for s, x in zip(seeds, d_s, strict=True)},
    }
    if n >= 2:
        t95, t90 = T_QUANTILES[n - 1]
        sd = float(d_s.std(ddof=1))
        res["seed_ci95"] = [d - t95 * sd / np.sqrt(n), d + t95 * sd / np.sqrt(n)]
        res["seed_ci90"] = [d - t90 * sd / np.sqrt(n), d + t90 * sd / np.sqrt(n)]
    rng = np.random.default_rng(BOOT_SEED)
    m = len(table[(a, seeds[0])])
    stack_a = np.stack([table[(a, s)] for s in seeds]).astype(float)
    stack_b = np.stack([table[(b, s)] for s in seeds]).astype(float)
    boots = np.empty(BOOT_N)
    for i in range(BOOT_N):
        idx = rng.integers(0, m, m)
        boots[i] = (stack_a[:, idx].mean(axis=1) - stack_b[:, idx].mean(axis=1)).mean()
    res["boot_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    res["boot_ci90"] = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
    # Two-sided bootstrap p-value for the per-structure Holm correction.
    res["boot_p_two_sided"] = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return res


def label(res: dict) -> str:
    """Apply the registered decision rule (PREREG.md Section 7).

    Args:
        res: Output of :func:`contrast` with both interval families.

    Returns:
        One of SUPPORTED, REVERSED, EQUIVALENT or INCONCLUSIVE.
    """
    if "seed_ci95" not in res:
        return "INCONCLUSIVE"
    lo = (res["seed_ci95"][0], res["boot_ci95"][0])
    hi = (res["seed_ci95"][1], res["boot_ci95"][1])
    if min(lo) > 0:
        return "SUPPORTED"
    if max(hi) < 0:
        return "REVERSED"
    inside = all(-EQUIV_MARGIN < x < EQUIV_MARGIN for x in (*res["seed_ci90"], *res["boot_ci90"]))
    return "EQUIVALENT" if inside else "INCONCLUSIVE"


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down correction.

    Args:
        pvals: p-value per hypothesis.
        alpha: Family-wise level.

    Returns:
        Which hypotheses are rejected.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    rejected, m = {}, len(items)
    stop = False
    for rank, (name, p) in enumerate(items):
        if stop or p > alpha / (m - rank):
            stop = True
            rejected[name] = False
        else:
            rejected[name] = True
    return rejected


def main() -> None:
    """Compute the primary and secondary endpoints and write results.json.

    Raises:
        ValueError: If the arms do not share the same episode set.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--detection", type=Path, default=None, help="H's per-episode parquet (S5).")
    ap.add_argument("--training-prior", type=Path, default=None, help="analyze_s13.py output (S7).")
    ap.add_argument("--tag", default="primary", help="Name of this analysis (e.g. best_pt).")
    args = ap.parse_args()

    cols = load_predictions(args.pred)
    ref_ids, first = np.unique(cols["scm_id"], return_index=True)
    structure = cols["structure"][first]
    y_true, y_obs = cols["y_true"][first], cols["y_obs"][first]
    effect = y_true - y_obs
    seeds_present = {
        arm: sorted({int(s) for s in np.unique(cols["seed"][cols["arm"] == arm])})
        for arm in ("int", "B", "A")
    }
    pairs = [s for s in seeds_present["int"] if s in seeds_present["B"]]
    out: dict = {
        "analysis": args.tag,
        "eps": args.eps,
        "n_episodes": len(ref_ids),
        "seeds_present": seeds_present,
        "complete_pairs": pairs,
    }
    primary_mask = np.isin(structure, PRIMARY_STRUCTURES) & (np.abs(effect) >= args.eps)
    out["primary_set_size"] = int(primary_mask.sum())
    if len(pairs) < 2:
        out["label"] = "NOT_COMPLETED"
        out["summary"] = SUMMARY["NOT_COMPLETED"]
    else:
        table = arm_table(cols, primary_mask)
        prim = contrast(table, "int", "B", pairs)
        out["primary"] = prim
        lab = label(prim)
        out["accuracy"] = {
            f"{arm}_{s}": float(table[(arm, s)].mean()) for (arm, s) in table if (arm, s) in table
        }
        # S1: per structure, Holm over the primary structures; MIXED qualifier.
        per = {}
        for s in PRIMARY_STRUCTURES:
            m = primary_mask & (structure == s)
            if m.sum() < 2:
                continue
            per[s] = {"n": int(m.sum()), **contrast(arm_table(cols, m), "int", "B", pairs)}
        rej = holm({s: v["boot_p_two_sided"] for s, v in per.items()})
        for s in per:
            per[s]["holm_significant"] = rej[s]
        out["S1_per_structure"] = per
        training = primary_mask & np.isin(structure, TRAINING_STRUCTURES)
        if training.sum() >= 2:
            out["S1_pooled_training_structures"] = {
                "n": int(training.sum()),
                **contrast(arm_table(cols, training), "int", "B", pairs),
            }
        opposite = [
            s
            for s, v in per.items()
            if rej[s]
            and np.sign(v["estimate"]) == -np.sign(prim["estimate"])
            and prim["estimate"] != 0
        ]
        out["mixed"] = bool(opposite)
        out["label"] = lab + (" MIXED" if opposite else "")
        out["summary"] = SUMMARY[lab].format(
            d=prim["estimate"],
            lo=min(prim["seed_ci95"][0], prim["boot_ci95"][0]),
            hi=max(prim["seed_ci95"][1], prim["boot_ci95"][1]),
        )
        if lab == "SUPPORTED":
            out["supported_at_least_0.02"] = bool(prim["estimate"] >= EQUIV_MARGIN)
        # S2 and S3: the published contrast and the target-only contrast.
        a_pairs = [s for s in pairs if s in seeds_present["A"]]
        if len(a_pairs) >= 2:
            out["S2_int_minus_A"] = contrast(table, "int", "A", a_pairs)
            out["S3_B_minus_A"] = contrast(table, "B", "A", a_pairs)
        # S4: level RMSE (episode bootstrap) and level-sign accuracy per arm.
        level_mask = np.abs(y_true) >= args.eps
        s4 = {}
        rng = np.random.default_rng(BOOT_SEED)
        for arm in ("int", "B", "A"):
            for seed in seeds_present[arm]:
                sel = (cols["arm"] == arm) & (cols["seed"] == seed)
                order = np.argsort(cols["scm_id"][sel])
                p = cols["pred"][sel][order]
                err = p - y_true
                boots = [
                    float(np.sqrt(np.mean(err[idx] ** 2)))
                    for idx in (rng.integers(0, len(err), len(err)) for _ in range(1000))
                ]
                s4[f"{arm}_{seed}"] = {
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "rmse_ci95": [
                        float(np.percentile(boots, 2.5)),
                        float(np.percentile(boots, 97.5)),
                    ],
                    "level_sign_acc": float(
                        np.mean(np.sign(p[level_mask]) == np.sign(y_true[level_mask]))
                    ),
                }
        out["S4_level"] = s4
        # S6: the non-identified control.
        ctrl = np.isin(structure, [CONTROL]) & (np.abs(effect) >= args.eps)
        if ctrl.sum() >= 2:
            ctable = arm_table(cols, ctrl)
            out["S6_bow_graph"] = {
                "n": int(ctrl.sum()),
                "accuracy": {f"{arm}_{s}": float(v.mean()) for (arm, s), v in ctable.items()},
                "int_minus_B": contrast(ctable, "int", "B", pairs),
            }
        # S5: the CPU estimators on the same primary episodes.
        if args.detection is not None and args.detection.exists():
            det = load_predictions(args.detection)
            s5 = {}
            for est in np.unique(det["estimator"]):
                sel = det["estimator"] == est
                ids = det["scm_id"][sel]
                keep = np.isin(ids, ref_ids[primary_mask])
                idx = np.searchsorted(ref_ids, ids[keep])
                c = correct(det["pred"][sel][keep], y_true[idx], y_obs[idx])
                s5[str(est)] = {"n": int(keep.sum()), "effect_sign_acc": float(c.mean())}
            out["S5_cpu_estimators"] = s5
    if args.training_prior is not None and args.training_prior.exists():
        out["S7_training_prior"] = json.loads(args.training_prior.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "results.json" if args.tag == "primary" else f"results_{args.tag}.json"
    (args.out_dir / name).write_text(json.dumps(out, indent=1, default=float))
    print(out["label"])
    print(out["summary"])
    print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
