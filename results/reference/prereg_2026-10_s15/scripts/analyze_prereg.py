#!/usr/bin/env python
"""Registered analysis of the s15 predictions (PREREG.md Sections 7 to 8 of this folder).

Reads ``s15_predictions.parquet`` from ``score_prereg.py`` and writes
``results.json`` with the primary endpoint P1 (interventional arm against the
do-information ablation, as in s13), the co-primary bar P2 (interventional arm
against the unadjusted in-context regression NaiveOLS on the same episodes),
the secondary endpoints S1 to S12, the registered labels and the registered
summary sentence of the (P1, P2) cell. Nothing in this file changes after
the registration; a later bug fix is a deviation and both versions are
reported.

The file is the s13 analysis (``../../prereg_2026-09/scripts/analyze_prereg.py``)
plus P2, S9 (structure-wise bar), S10 (mechanism probes), S11 (in-support
subset) and S12 (the published Chronos-2 covariate result as context).
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
BAR_MARGIN = 0.02
BAR_ESTIMATOR = "NaiveOLS"
CONTEXT_ESTIMATORS = ("do-SVAR", "GraphRouter", "BackDoorOLS", "FrontDoorOLS", "IV2SLS", "Mean")
# Two-sided Student t quantiles for the seed interval (degrees of freedom = pairs - 1).
T_QUANTILES = {1: (12.706, 6.314), 2: (4.303, 2.920), 3: (3.182, 2.353)}
P1_SENTENCE = {
    "SUPPORTED": (
        "A redesigned Do-Over-Time-PFN with the intervention visible predicts the effect sign more "
        "often than the same model with it withheld, by {d:+.3f} with a 95% confidence interval "
        "from {lo:+.3f} to {hi:+.3f}."
    ),
    "REVERSED": (
        "With the intervention visible the redesigned model predicts the effect sign less often "
        "than the same model with it withheld, by {d:+.3f} with a 95% confidence interval from "
        "{lo:+.3f} to {hi:+.3f}."
    ),
    "EQUIVALENT": (
        "Interventional training changes the redesigned model's effect-sign accuracy by {d:+.3f}, "
        "with a 95% confidence interval from {lo:+.3f} to {hi:+.3f}, inside the registered "
        "equivalence margin."
    ),
    "INCONCLUSIVE": (
        "Interventional training changes the redesigned model's effect-sign accuracy by {d:+.3f}, "
        "with a 95% confidence interval from {lo:+.3f} to {hi:+.3f}, which is inconclusive."
    ),
}
P2_SENTENCE = {
    "ABOVE_BAR": (
        "It is {e:+.3f} above an unadjusted in-context regression on the same episodes, with a "
        "95% confidence interval from {lo:+.3f} to {hi:+.3f}."
    ),
    "AT_BAR": (
        "It is within the registered margin of an unadjusted in-context regression on the same "
        "episodes ({e:+.3f}, 95% confidence interval from {lo:+.3f} to {hi:+.3f})."
    ),
    "BELOW_BAR": (
        "It remains {e:+.3f} below an unadjusted in-context regression on the same episodes, "
        "with a 95% confidence interval from {lo:+.3f} to {hi:+.3f}."
    ),
    "NOT_COMPUTED": "The bar against the unadjusted regression was not computed.",
}
CLAIM_SENTENCE = {
    "SUPPORTED": "This supports the narrower claim that the model uses the do-value.",
    "REVERSED": "This does not support the narrower claim that the model uses the do-value.",
    "EQUIVALENT": "This does not support the narrower claim that the model uses the do-value.",
    "INCONCLUSIVE": "This neither supports nor rejects the narrower claim that the model uses the do-value.",
}
CLOSING = "It does not restore the withdrawn claim."
NOT_COMPLETED = "The re-test did not complete by the registered deadline. " + CLOSING


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    """Load a predictions parquet into column arrays.

    Args:
        path: The parquet file.

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


def arm_rows(cols: dict[str, np.ndarray], arm: str, seed: int) -> np.ndarray:
    """Row indices of one (arm, seed) in reference episode order.

    Args:
        cols: The prediction columns.
        arm: The arm.
        seed: The seed.

    Returns:
        Index array, or an empty array if the run is absent.

    Raises:
        ValueError: If the run's episode set differs from the reference set.
    """
    sel = np.flatnonzero((cols["arm"] == arm) & (cols["seed"] == seed))
    if sel.size == 0:
        return sel
    order = np.argsort(cols["scm_id"][sel])
    rows = sel[order]
    if not np.array_equal(cols["scm_id"][rows], np.unique(cols["scm_id"])):
        raise ValueError(f"{arm} seed {seed}: episode set differs from the reference")
    return rows


def arm_table(cols: dict[str, np.ndarray], mask: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    """Per (arm, seed) correctness over the masked episodes, aligned by scm_id.

    Args:
        cols: The prediction columns.
        mask: Episode mask over the per-episode reference order.

    Returns:
        ``{(arm, seed): correct}`` with rows in reference episode order.
    """
    out: dict[tuple[str, int], np.ndarray] = {}
    for arm in ("int", "B", "A"):
        for seed in SEEDS:
            rows = arm_rows(cols, arm, seed)
            if rows.size == 0:
                continue
            c = correct(cols["pred"][rows], cols["y_true"][rows], cols["y_obs"][rows])
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


def bar_contrast(table, seeds, reference: np.ndarray) -> dict:
    """Intervals of acc(int) - acc(reference) for a seedless reference estimator.

    The reference has no seed variance, so the seed interval uses the spread
    of the int arm alone, and the episode bootstrap resamples both sides.

    Args:
        table: Output of :func:`arm_table`.
        seeds: Seeds with the int arm present.
        reference: Correctness of the reference estimator on the same masked episodes.

    Returns:
        Dict as :func:`contrast`.
    """
    ref_table = {("ref", s): reference for s in seeds}
    merged = {**table, **ref_table}
    return contrast(merged, "int", "ref", seeds)


def label(res: dict) -> str:
    """Apply the registered decision rule for P1 (PREREG.md Section 7).

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


def bar_label(res: dict) -> str:
    """Apply the registered decision rule for P2 (PREREG.md Section 7b).

    Args:
        res: Output of :func:`bar_contrast`.

    Returns:
        One of ABOVE_BAR, AT_BAR or BELOW_BAR.
    """
    if "seed_ci95" not in res:
        return "BELOW_BAR"
    lo = min(res["seed_ci95"][0], res["boot_ci95"][0])
    if lo > 0:
        return "ABOVE_BAR"
    if lo >= -BAR_MARGIN:
        return "AT_BAR"
    return "BELOW_BAR"


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


def estimator_correct(
    det: dict[str, np.ndarray],
    estimator: str,
    ref_ids: np.ndarray,
    y_true: np.ndarray,
    y_obs: np.ndarray,
) -> np.ndarray | None:
    """Correctness of one detection-power estimator aligned to the reference episodes.

    Args:
        det: Columns of the detection-power parquet.
        estimator: Estimator name in that file.
        ref_ids: Sorted reference episode ids.
        y_true: Counterfactual levels in reference order.
        y_obs: Factual levels in reference order.

    Returns:
        Boolean array in reference order, or ``None`` if the estimator is absent
        or does not cover every reference episode.
    """
    sel = det["estimator"] == estimator
    if not sel.any():
        return None
    ids = det["scm_id"][sel]
    keep = np.isin(ids, ref_ids)
    idx = np.searchsorted(ref_ids, ids[keep])
    if len(np.unique(idx)) != len(ref_ids):
        return None
    out = np.zeros(len(ref_ids), dtype=bool)
    out[idx] = correct(det["pred"][sel][keep], y_true[idx], y_obs[idx])
    return out


def mechanism_probes(cols: dict[str, np.ndarray], rows: np.ndarray, mask: np.ndarray) -> dict:
    """S10: mirror flip rate, anchor error ratio and slope-sign agreement of one int run.

    Args:
        cols: The prediction columns (with ``pred_mirror`` and ``pred_anchor``).
        rows: Row indices of the run in reference order.
        mask: Primary-set mask in reference order.

    Returns:
        The three probe values and the number of episodes they use.
    """
    p = cols["pred"][rows][mask]
    pm = cols["pred_mirror"][rows][mask]
    pa = cols["pred_anchor"][rows][mask]
    y_obs = cols["y_obs"][rows][mask]
    y_true = cols["y_true"][rows][mask]
    mu_y = cols["mu_Y"][rows][mask]
    dv = cols["do_value"][rows][mask] - cols["a_ref"][rows][mask]
    ok = np.isfinite(pm) & np.isfinite(pa)
    eff_v, eff_m, eff_a = p - y_obs, pm - y_obs, pa - y_obs
    moved = ok & (np.abs(eff_v) > 1e-6)
    nz = ok & (np.abs(dv) > 1e-6)
    true_slope = np.where(nz, (y_true - y_obs) / np.where(nz, dv, 1.0), np.nan)
    model_slope = np.where(nz, (p - pa) / np.where(nz, dv, 1.0), np.nan)
    return {
        "n": int(ok.sum()),
        "mirror_flip_rate": float(np.mean((np.sign(eff_m) != np.sign(eff_v))[moved]))
        if moved.any()
        else None,
        "anchor_error_ratio_median": float(
            np.median((np.abs(eff_a) / np.maximum(np.abs(mu_y - y_obs), 1e-9))[ok])
        )
        if ok.any()
        else None,
        "slope_sign_agreement": float(np.mean((np.sign(model_slope) == np.sign(true_slope))[nz]))
        if nz.any()
        else None,
    }


def main() -> None:
    """Compute the primary and secondary endpoints and write results.json.

    Raises:
        ValueError: If the arms do not share the same episode set.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument(
        "--detection",
        type=Path,
        default=None,
        help="The detection-power per-episode parquet (P2, S5, S9).",
    )
    ap.add_argument("--training-prior", type=Path, default=None, help="S7 JSON, if computed.")
    ap.add_argument(
        "--chronos", type=Path, default=None, help="Published Chronos-2 JSON (S12 context)."
    )
    ap.add_argument("--tag", default="primary", help="Name of this analysis (e.g. best_pt).")
    args = ap.parse_args()

    cols = load_predictions(args.pred)
    ref_ids, first = np.unique(cols["scm_id"], return_index=True)
    structure = cols["structure"][first]
    y_true, y_obs = cols["y_true"][first], cols["y_obs"][first]
    in_support = cols["in_support"][first].astype(bool)
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
    out["in_support_primary_size"] = int((primary_mask & in_support).sum())
    det = (
        load_predictions(args.detection)
        if args.detection is not None and args.detection.exists()
        else None
    )
    if len(pairs) < 2:
        out["label"] = "NOT_COMPLETED"
        out["summary"] = NOT_COMPLETED
    else:
        table = arm_table(cols, primary_mask)
        prim = contrast(table, "int", "B", pairs)
        out["primary"] = prim
        lab = label(prim)
        out["accuracy"] = {f"{arm}_{s}": float(table[(arm, s)].mean()) for (arm, s) in table}
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
        p1_sentence = P1_SENTENCE[lab].format(
            d=prim["estimate"],
            lo=min(prim["seed_ci95"][0], prim["boot_ci95"][0]),
            hi=max(prim["seed_ci95"][1], prim["boot_ci95"][1]),
        )
        if lab == "SUPPORTED":
            out["supported_at_least_0.02"] = bool(prim["estimate"] >= EQUIV_MARGIN)
        # P2: the co-primary bar against the unadjusted in-context regression.
        bar_lab = "NOT_COMPUTED"
        p2_sentence = P2_SENTENCE["NOT_COMPUTED"]
        bar_ref = (
            estimator_correct(det, BAR_ESTIMATOR, ref_ids, y_true, y_obs)
            if det is not None
            else None
        )
        if bar_ref is not None:
            int_seeds = seeds_present["int"]
            bar = bar_contrast(table, int_seeds, bar_ref[primary_mask])
            bar["reference_accuracy"] = float(bar_ref[primary_mask].mean())
            bar_lab = bar_label(bar)
            out["P2_bar"] = {"estimator": BAR_ESTIMATOR, **bar}
            out["P2_label"] = bar_lab
            p2_sentence = P2_SENTENCE[bar_lab].format(
                e=bar["estimate"],
                lo=min(bar["seed_ci95"][0], bar["boot_ci95"][0]),
                hi=max(bar["seed_ci95"][1], bar["boot_ci95"][1]),
            )
            # Unlabelled context: the same contrast against the other in-context estimators.
            ctx = {}
            for est in CONTEXT_ESTIMATORS:
                ref = estimator_correct(det, est, ref_ids, y_true, y_obs)
                if ref is not None:
                    c = bar_contrast(table, int_seeds, ref[primary_mask])
                    c["reference_accuracy"] = float(ref[primary_mask].mean())
                    ctx[est] = c
            out["P2_context"] = ctx
            # S9: the bar per structure, Holm over the primary structures.
            s9 = {}
            for s in PRIMARY_STRUCTURES:
                m = primary_mask & (structure == s)
                if m.sum() < 2:
                    continue
                c = bar_contrast(arm_table(cols, m), int_seeds, bar_ref[m])
                c["reference_accuracy"] = float(bar_ref[m].mean())
                s9[s] = {"n": int(m.sum()), **c}
            rej9 = holm({s: v["boot_p_two_sided"] for s, v in s9.items()})
            for s in s9:
                s9[s]["holm_significant"] = rej9[s]
            out["S9_bar_per_structure"] = s9
        out["summary"] = " ".join([p1_sentence, p2_sentence, CLAIM_SENTENCE[lab], CLOSING])
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
                rows = arm_rows(cols, arm, seed)
                err = cols["pred"][rows] - y_true
                boots = [
                    float(np.sqrt(np.mean(err[idx] ** 2)))
                    for idx in (rng.integers(0, len(err), len(err)) for _ in range(1000))
                ]
                p = cols["pred"][rows]
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
        # S5: every CPU estimator on the same primary episodes.
        if det is not None:
            s5 = {}
            for est in np.unique(det["estimator"]):
                ref = estimator_correct(det, str(est), ref_ids, y_true, y_obs)
                if ref is not None:
                    s5[str(est)] = {
                        "n": int(primary_mask.sum()),
                        "effect_sign_acc": float(ref[primary_mask].mean()),
                    }
            out["S5_cpu_estimators"] = s5
        # S6: the non-identified control.
        ctrl = np.isin(structure, [CONTROL]) & (np.abs(effect) >= args.eps)
        if ctrl.sum() >= 2:
            ctable = arm_table(cols, ctrl)
            out["S6_bow_graph"] = {
                "n": int(ctrl.sum()),
                "accuracy": {f"{arm}_{s}": float(v.mean()) for (arm, s), v in ctable.items()},
                "int_minus_B": contrast(ctable, "int", "B", pairs),
            }
        # S10: the mechanism probes of every int run on the primary set.
        out["S10_mechanism"] = {
            f"int_{s}": mechanism_probes(cols, arm_rows(cols, "int", s), primary_mask)
            for s in seeds_present["int"]
        }
        # S11: P1 and P2 on the in-support subset.
        sub = primary_mask & in_support
        if sub.sum() >= 2:
            stable = arm_table(cols, sub)
            s11 = {"n": int(sub.sum()), "int_minus_B": contrast(stable, "int", "B", pairs)}
            s11["int_minus_B_label"] = label(s11["int_minus_B"])
            if bar_ref is not None:
                b = bar_contrast(stable, seeds_present["int"], bar_ref[sub])
                b["reference_accuracy"] = float(bar_ref[sub].mean())
                s11["bar"] = b
                s11["bar_label"] = bar_label(b)
            out["S11_in_support"] = s11
    if args.training_prior is not None and args.training_prior.exists():
        out["S7_training_prior"] = json.loads(args.training_prior.read_text())
    if args.chronos is not None and args.chronos.exists():
        ch = json.loads(args.chronos.read_text())
        out["S12_chronos_context"] = {
            k: ch[k]
            for k in ("suite_version", "n", "dir_target", "model_id", "Chronos_int", "Chronos_obs")
            if k in ch
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = "results.json" if args.tag == "primary" else f"results_{args.tag}.json"
    (args.out_dir / name).write_text(json.dumps(out, indent=1, default=float))
    print(out["label"], out.get("P2_label", ""))
    print(out["summary"])
    print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
