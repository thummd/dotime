#!/usr/bin/env python
"""Grade the registered predictions of the lagged back-door set (PREREG.md, Section 5).

The detection-power script writes one row per scored (episode, estimator) when
it is run with the extension estimators named (``run.sh``). This script pairs
every registered contrast on the same episodes, bootstraps the paired
difference with the study's seed and replicate count, and grades L1 to L3
mechanically. The exploratory rows X1 to X3 use the same machinery and are
graded against no threshold. It writes ``grade.json`` and ``grade.md``.

Usage, from the repository root with ``PYTHONPATH=src``::

    python results/reference/lagged_backdoor_2026-10/scripts/grade_lagged.py \\
        --predictions results/reference/lagged_backdoor_2026-10/ident_v1_2_predictions.parquet \\
        --out results/reference/lagged_backdoor_2026-10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
STUDY_SCRIPTS = HERE.parent.parent / "detection_power_2026-10" / "scripts"
# The study's conventions are imported, not copied, so that a validity mask or
# a bootstrap seed can never drift between the two gradings.
sys.path.insert(0, str(STUDY_SCRIPTS))
from detection_power import (  # noqa: E402
    BOOT_SEED,
    FALSE_EFFECT_THRESHOLD,
    MIN_N_VALID,
    N_BOOT,
    _ci,
    sign_correct,
)

LAGGED = "LaggedBackDoorOLS"
LAGGED3 = "LaggedBackDoorOLS-L3"
# Non-inferiority margin of L2 and L3: about two binomial standard errors of
# BackDoorOLS's registered accuracy on back_door (0.014 at n_valid = 789).
MARGIN = 0.03

# (id, structure, estimator a, estimator b, statistic, rule, statement)
REGISTERED = [
    (
        "L1",
        "observed_confounder",
        LAGGED,
        "BackDoorOLS",
        "false_effect",
        "upper_below_zero",
        "LaggedBackDoorOLS makes false effects on observed_confounder at a lower rate than "
        "BackDoorOLS (paired 95% CI of the difference below 0)",
    ),
    (
        "L1b",
        "observed_confounder",
        LAGGED,
        "do-SVAR",
        "false_effect",
        "upper_below_zero",
        "LaggedBackDoorOLS makes false effects on observed_confounder at a lower rate than "
        "do-SVAR (paired 95% CI of the difference below 0)",
    ),
    (
        "L2",
        "back_door",
        LAGGED,
        "BackDoorOLS",
        "effect_sign",
        "lower_above_minus_margin",
        f"on back_door the effect-sign accuracy of LaggedBackDoorOLS is not below BackDoorOLS's "
        f"by more than {MARGIN} (lower bound of the paired 95% CI above -{MARGIN})",
    ),
    (
        "L3",
        "confounder_mediator",
        LAGGED,
        "BackDoorOLS",
        "effect_sign",
        "lower_above_minus_margin",
        f"on confounder_mediator the effect-sign accuracy of LaggedBackDoorOLS is not below "
        f"BackDoorOLS's by more than {MARGIN} (lower bound of the paired 95% CI above -{MARGIN})",
    ),
]

# (id, structure, estimator a, estimator b or None, statistic, statement)
EXPLORATORY = [
    ("X1a", "observed_confounder", LAGGED3, "BackDoorOLS", "false_effect", "lag 3 against BackDoorOLS"),
    ("X1b", "observed_confounder", LAGGED3, "do-SVAR", "false_effect", "lag 3 against do-SVAR"),
    ("X1c", "back_door", LAGGED3, "BackDoorOLS", "effect_sign", "lag 3 against BackDoorOLS"),
    ("X1d", "confounder_mediator", LAGGED3, "BackDoorOLS", "effect_sign", "lag 3 against BackDoorOLS"),
    ("X1e", "back_door", LAGGED3, LAGGED, "effect_sign", "lag 3 against lag 1"),
    ("X2a", "back_door", LAGGED, "do-SVAR", "effect_sign", "lag 1 against do-SVAR"),
    ("X2b", "confounder_mediator", LAGGED, "do-SVAR", "effect_sign", "lag 1 against do-SVAR"),
    ("X2c", "confounder_mediator", LAGGED, "FrontDoorOLS", "effect_sign", "lag 1 against FrontDoorOLS"),
    ("X3a", "observed_confounder", LAGGED, None, "false_effect", "lag 1 rate against the study's 0.10 bar"),
    ("X3b", "observed_confounder", LAGGED3, None, "false_effect", "lag 3 rate against the study's 0.10 bar"),
]


def load(path: Path) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Read the per-episode predictions, grouped by estimator and structure.

    Args:
        path: The ``*_predictions.parquet`` the detection-power script wrote.

    Returns:
        ``{(estimator, structure): columns}`` with every column sorted by
        ``scm_id``, so that two estimators' arrays of one structure are paired
        row by row.
    """
    d = pq.read_table(path).to_pydict()
    est = np.array(d["estimator"])
    struct = np.array(d["structure"])
    scm = np.array(d["scm_id"])
    cols = {k: np.asarray(d[k]) for k in ("pred", "pred_ref", "y_true", "y_obs")}
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for e in np.unique(est):
        for s in np.unique(struct):
            m = (est == e) & (struct == s)
            if not m.any():
                continue
            order = np.argsort(scm[m], kind="stable")
            out[(str(e), str(s))] = {"scm_id": scm[m][order], **{k: v[m][order] for k, v in cols.items()}}
    return out


def per_episode(rows: dict[str, np.ndarray], statistic: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-episode indicator of a statistic and the episodes it is defined on.

    Args:
        rows: One estimator's columns on one structure.
        statistic: ``"false_effect"`` (a predicted effect of at least the
            study's threshold, defined on every episode) or ``"effect_sign"``
            (a correct effect sign, defined on the episodes with a scorable
            effect).

    Returns:
        ``(valid, indicator)`` boolean arrays over the episodes.

    Raises:
        ValueError: If ``statistic`` is not one of the two.
    """
    if statistic == "false_effect":
        calls = np.abs(rows["pred"] - rows["pred_ref"]) >= FALSE_EFFECT_THRESHOLD
        return np.ones_like(calls, dtype=bool), calls
    if statistic == "effect_sign":
        return sign_correct(rows["pred"], rows["y_true"], rows["y_obs"])
    raise ValueError(f"unknown statistic {statistic!r}")


def contrast(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray] | None, statistic: str
) -> dict[str, Any]:
    """A statistic of estimator ``a``, and its paired difference to ``b`` if given.

    Args:
        a: Columns of the first estimator on one structure.
        b: Columns of the second estimator on the same structure, or ``None``
            for a rate alone.
        statistic: See :func:`per_episode`.

    Returns:
        The number of defined episodes, the two statistics, their difference
        and its percentile bootstrap 95% interval, paired over episodes with
        the study's seed and replicate count.

    Raises:
        ValueError: If the two estimators are not scored on the same episodes.
    """
    va, xa = per_episode(a, statistic)
    if b is None:
        return {"n": int(va.sum()), "a": float(xa[va].mean())}
    if not np.array_equal(a["scm_id"], b["scm_id"]):
        raise ValueError("the two estimators are not scored on the same episodes")
    vb, xb = per_episode(b, statistic)
    if not np.array_equal(va, vb):
        raise ValueError("validity differs between the estimators, which depends on the targets only")
    xa, xb = xa[va].astype(float), xb[vb].astype(float)
    n = xa.size
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    diff = xa[idx].mean(axis=1) - xb[idx].mean(axis=1)
    return {
        "n": int(n),
        "a": float(xa.mean()),
        "b": float(xb.mean()),
        "diff": float(xa.mean() - xb.mean()),
        "ci95": _ci(diff),
    }


def holds(c: dict[str, Any], rule: str) -> bool:
    """Apply a registered decision rule to a contrast.

    Args:
        c: Output of :func:`contrast` with a difference.
        rule: ``"upper_below_zero"`` or ``"lower_above_minus_margin"``.

    Returns:
        Whether the rule is met.

    Raises:
        ValueError: If ``rule`` is unknown.
    """
    lo, hi = c["ci95"]
    if rule == "upper_below_zero":
        return hi < 0.0
    if rule == "lower_above_minus_margin":
        return lo > -MARGIN
    raise ValueError(f"unknown rule {rule!r}")


def grade(rows: dict[tuple[str, str], dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Grade every registered and exploratory row.

    Args:
        rows: Output of :func:`load`.

    Returns:
        One record per row with its numbers and outcome: ``confirmed`` or
        ``not confirmed`` for a registered prediction, ``reported`` for an
        exploratory one, ``not gradable`` when an estimator is missing or the
        structure has fewer than the study's minimum of defined episodes.
    """
    out: list[dict[str, Any]] = []
    for pid, s, ea, eb, statistic, rule, statement in REGISTERED:
        rec: dict[str, Any] = {"id": pid, "structure": s, "a": ea, "b": eb, "statistic": statistic, "rule": rule, "statement": statement}
        if (ea, s) not in rows or (eb, s) not in rows:
            rec.update(outcome="not gradable", reason="estimator missing from the predictions")
        else:
            c = contrast(rows[(ea, s)], rows[(eb, s)], statistic)
            rec.update(c)
            if c["n"] < MIN_N_VALID:
                rec.update(outcome="not gradable", reason=f"fewer than {MIN_N_VALID} defined episodes")
            else:
                rec["outcome"] = "confirmed" if holds(c, rule) else "not confirmed"
        out.append(rec)
    for pid, s, ea, eb, statistic, statement in EXPLORATORY:
        rec = {"id": pid, "structure": s, "a": ea, "b": eb, "statistic": statistic, "statement": statement}
        if (ea, s) not in rows or (eb is not None and (eb, s) not in rows):
            rec.update(outcome="not gradable", reason="estimator missing from the predictions")
        else:
            rec.update(contrast(rows[(ea, s)], None if eb is None else rows[(eb, s)], statistic))
            rec["outcome"] = "reported"
        out.append(rec)
    return out


def render(records: list[dict[str, Any]], source: str) -> str:
    """Markdown table of the graded rows.

    Args:
        records: Output of :func:`grade`.
        source: Name of the predictions file, for the header.

    Returns:
        The Markdown text.
    """

    def f(x: Any) -> str:
        return "" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))

    lines = [
        "# Lagged back-door set: graded predictions",
        "",
        f"Graded by `scripts/grade_lagged.py` from `{source}`. Differences are a minus b, paired over "
        f"episodes, percentile bootstrap with {N_BOOT} replicates and seed {BOOT_SEED}. "
        f"`false_effect` is the rate of `|pred(do v) - pred(do a_ref)| >= {FALSE_EFFECT_THRESHOLD}`, "
        "`effect_sign` the effect-sign accuracy on the episodes with a scorable effect.",
        "",
        "| Id | Structure | Statistic | a | b | n | a | b | a minus b | 95% CI | Outcome |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        ci = r.get("ci95")
        lines.append(
            f"| {r['id']} | `{r['structure']}` | {r['statistic']} | {r['a']} | {r.get('b') or ''} | "
            f"{r.get('n', '')} | {f(r.get('a')) if isinstance(r.get('a'), float) else ''} | "
            f"{f(r.get('b')) if isinstance(r.get('b'), float) else ''} | {f(r.get('diff'))} | "
            f"{'' if ci is None else f'[{ci[0]:+.3f}, {ci[1]:+.3f}]'} | {r['outcome']}"
            + (f" ({r['reason']})" if "reason" in r else "")
            + " |"
        )
    lines += ["", "Statements:", ""]
    lines += [f"- {r['id']}: {r['statement']}" for r in records]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Grade a predictions file and write ``grade.json`` and ``grade.md``.

    Args:
        argv: Command-line arguments (``None`` reads ``sys.argv``).
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, default=HERE.parent / "ident_v1_2_predictions.parquet")
    ap.add_argument("--out", type=Path, default=HERE.parent)
    args = ap.parse_args(argv)
    records = grade(load(args.predictions))
    # The numbers in the record keep the estimator names under their own keys
    # and the statistics under ``stat_a``/``stat_b``, so the JSON is unambiguous.
    for r in records:
        r["estimator_a"], r["estimator_b"] = r.pop("a"), r.pop("b")
    md_records = [dict(r, a=r["estimator_a"], b=r["estimator_b"]) for r in records]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "grade.json").write_text(json.dumps({"source": args.predictions.name, "margin": MARGIN, "rows": records}, indent=1))
    (args.out / "grade.md").write_text(render(md_records, args.predictions.name))
    for r in records:
        print(f"[{r['id']}] {r['outcome']}: diff {r.get('diff')} ci {r.get('ci95')}", flush=True)


if __name__ == "__main__":
    main()
