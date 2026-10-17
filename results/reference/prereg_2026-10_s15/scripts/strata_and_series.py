"""Registered endpoints 7c, S13 and S14 of the s15 pre-registration.

Section 7c: the series contrast F = acc(int_s15) - acc(int_s14), seed by seed, on the
primary set, graded from the registered s14 predictions parquet, with the seed interval and
the paired episode bootstrap of Section 7 and the labels of Section 7. S13: D (Section 7),
E (Section 7b) and F in four dose strata of the primary set, split by the dose magnitude
``|v - mu_A| / sd_A`` as the scorer records it. S14: F per structure with the Holm correction.

Every statistic reuses the registered functions of ``analyze_prereg.py`` in this folder, so
the intervals, the bootstrap seed and the labels are those of the primary analysis.

Usage::

    python results/reference/prereg_2026-10_s15/scripts/strata_and_series.py \\
        --pred results/reference/prereg_2026-10_s15/s15_predictions.parquet \\
        --s14-pred results/reference/prereg_2026-11/s14_predictions.parquet \\
        --detection results/reference/detection_power_2026-10/ident_v1_2_predictions.parquet \\
        --out-dir results/reference/prereg_2026-10_s15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# The registered analysis lives beside this file and is run as a script, so it is imported
# by path rather than as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_prereg import (  # noqa: E402
    BAR_ESTIMATOR,
    EPS,
    PRIMARY_STRUCTURES,
    SEEDS,
    bar_contrast,
    bar_label,
    contrast,
    correct,
    estimator_correct,
    holm,
    label,
    load_predictions,
)

STRATA = ((0.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, float("inf")))
S14_PRED_SHA256 = "66714e01a78aff64668e50e1bfc98bb43b1fd677e267020a3dc1bc22d50d7848"


def sha256(path: Path) -> str:
    """SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_frame(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-episode arrays of a predictions parquet in sorted scm_id order.

    Args:
        cols: Output of :func:`analyze_prereg.load_predictions`.

    Returns:
        Dict with ``ids``, ``structure``, ``y_true``, ``y_obs`` and ``dose``, the dose
        magnitude in pre-onset standard deviations of the treatment.
    """
    ids, first = np.unique(cols["scm_id"], return_index=True)
    sd = np.maximum(cols["sd_A"][first].astype(float), 1e-4)
    return {
        "ids": ids,
        "structure": cols["structure"][first],
        "y_true": cols["y_true"][first].astype(float),
        "y_obs": cols["y_obs"][first].astype(float),
        "dose": np.abs(cols["do_value"][first].astype(float) - cols["mu_A"][first].astype(float)) / sd,
    }


def correctness_by_id(cols: dict[str, np.ndarray], arm: str, seed: int, ids: np.ndarray) -> np.ndarray | None:
    """Registered correctness of one (arm, seed) aligned to a sorted id vector.

    Alignment goes through ``scm_id`` explicitly, so two parquets scored in different runs
    line up episode by episode.

    Args:
        cols: Prediction columns.
        arm: The arm.
        seed: The seed.
        ids: Sorted reference episode ids.

    Returns:
        Boolean array in the order of ``ids``, or ``None`` if the arm and seed are absent or
        do not cover every id.
    """
    sel = (cols["arm"] == arm) & (cols["seed"] == seed)
    if not sel.any():
        return None
    row_ids = cols["scm_id"][sel]
    keep = np.isin(row_ids, ids)
    idx = np.searchsorted(ids, row_ids[keep])
    if len(np.unique(idx)) != len(ids):
        return None
    out = np.zeros(len(ids), dtype=bool)
    out[idx] = correct(cols["pred"][sel][keep], cols["y_true"][sel][keep], cols["y_obs"][sel][keep])
    return out


def build_tables(frame: dict, s15: dict, s14: dict, mask: np.ndarray) -> tuple[dict, list[int], list[int]]:
    """Correctness tables of the s15 arms and the s14 int arm on a masked episode set.

    Args:
        frame: Output of :func:`episode_frame` for the s15 parquet.
        s15: The s15 prediction columns.
        s14: The s14 prediction columns.
        mask: Episode mask in the order of ``frame["ids"]``.

    Returns:
        The table ``{(arm, seed): correct}`` with the s14 int arm under the arm name
        ``"s14"``, the seeds with a complete int and B pair, and the seeds with a complete
        s15 and s14 int pair.
    """
    table: dict[tuple[str, int], np.ndarray] = {}
    for arm in ("int", "B", "A"):
        for seed in SEEDS:
            c = correctness_by_id(s15, arm, seed, frame["ids"])
            if c is not None:
                table[(arm, seed)] = c[mask]
    for seed in SEEDS:
        c = correctness_by_id(s14, "int", seed, frame["ids"])
        if c is not None:
            table[("s14", seed)] = c[mask]
    pairs = [s for s in SEEDS if ("int", s) in table and ("B", s) in table]
    series = [s for s in SEEDS if ("int", s) in table and ("s14", s) in table]
    return table, pairs, series


def main() -> None:
    """Compute 7c, S13 and S14 and write ``strata_and_series.json``.

    Raises:
        SystemExit: If the s14 predictions file does not match its registered digest.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True, help="The s15 predictions parquet.")
    ap.add_argument("--s14-pred", type=Path, required=True, help="The registered s14 predictions parquet.")
    ap.add_argument("--detection", type=Path, default=None, help="The detection-power parquet (E).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--allow-digest-mismatch", action="store_true",
                    help="Smoke tests only: accept an s14 file whose digest is not the registered one.")
    args = ap.parse_args()

    digest = sha256(args.s14_pred)
    if digest != S14_PRED_SHA256 and not args.allow_digest_mismatch:
        raise SystemExit(f"{args.s14_pred}: digest {digest} is not the registered s14 predictions file")

    s15 = load_predictions(args.pred)
    s14 = load_predictions(args.s14_pred)
    frame = episode_frame(s15)
    effect = frame["y_true"] - frame["y_obs"]
    primary = np.isin(frame["structure"], PRIMARY_STRUCTURES) & (np.abs(effect) >= args.eps)
    det = load_predictions(args.detection) if args.detection is not None and args.detection.exists() else None
    bar_ref = (
        estimator_correct(det, BAR_ESTIMATOR, frame["ids"], frame["y_true"], frame["y_obs"])
        if det is not None else None
    )

    out: dict = {
        "analysis": args.tag,
        "eps": args.eps,
        "s14_pred_sha256": digest,
        "primary_set_size": int(primary.sum()),
    }

    # Section 7c: the series contrast on the primary set.
    table, pairs, series = build_tables(frame, s15, s14, primary)
    out["series_seeds"] = series
    if len(series) >= 2:
        f = contrast(table, "int", "s14", series)
        f["label"] = label(f)
        f["ge_margin"] = bool(f["estimate"] >= 0.02)
        out["series_contrast"] = f
    else:
        out["series_contrast"] = {"label": "NOT_COMPLETED"}

    # S13: D, E and F per dose stratum.
    strata = {}
    for lo, hi in STRATA:
        name = f"({lo:g}, {hi:g}]" if np.isfinite(hi) else f"({lo:g}, inf)"
        m = primary & (frame["dose"] > lo) & (frame["dose"] <= hi)
        entry: dict = {"n": int(m.sum())}
        if m.sum() >= 2:
            t, p, s = build_tables(frame, s15, s14, m)
            if len(p) >= 2:
                d = contrast(t, "int", "B", p)
                d["label"] = label(d)
                entry["D"] = d
            if bar_ref is not None and len([x for x in SEEDS if ("int", x) in t]) >= 2:
                e = bar_contrast(t, [x for x in SEEDS if ("int", x) in t], bar_ref[m])
                e["label"] = bar_label(e)
                entry["E"] = e
            if len(s) >= 2:
                fs = contrast(t, "int", "s14", s)
                fs["label"] = label(fs)
                entry["F"] = fs
        strata[name] = entry
    out["S13_dose_strata"] = strata

    # S14: F per structure with the Holm correction.
    per = {}
    for st in PRIMARY_STRUCTURES:
        m = primary & (frame["structure"] == st)
        t, _, s = build_tables(frame, s15, s14, m)
        if m.sum() >= 2 and len(s) >= 2:
            per[st] = contrast(t, "int", "s14", s)
    if per:
        rej = holm({st: v["boot_p_two_sided"] for st, v in per.items()})
        for st in per:
            per[st]["holm_significant"] = bool(rej[st])
    out["S14_series_per_structure"] = per

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / ("strata_and_series.json" if args.tag == "primary" else f"strata_and_series_{args.tag}.json")
    path.write_text(json.dumps(out, indent=2))
    sc = out["series_contrast"]
    print(f"series contrast F: {sc.get('estimate', float('nan')):+.4f} label {sc['label']} seeds {series}")
    for name, entry in strata.items():
        parts = [f"n={entry['n']}"]
        for k in ("D", "E", "F"):
            if k in entry:
                parts.append(f"{k} {entry[k]['estimate']:+.3f} {entry[k]['label']}")
        print(f"  stratum {name:12s} " + ", ".join(parts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
