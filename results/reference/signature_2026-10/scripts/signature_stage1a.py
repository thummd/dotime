"""Stage I-a of the signature-space diagnostic: DoTime 1.2.0 primary set only.

Question: which coordinates of a truncated path signature of the PRE-ONSET
history carry the counterfactual effect of a hard intervention, and how far
does a sparse linear model on them get against the in-context estimators on
the same episodes (NaiveOLS 0.764, do-SVAR 0.762 effect-sign accuracy)?

Design choices, so that this is a predictive diagnostic and not a leak:

* Signatures are computed on the pre-onset rows only. The (X, I) cross term
  over the whole window equals the do-value times the outcome's post-onset
  displacement, which is the target.
* The do-value enters as the normalised shift ``dv = (v - a_ref) / sd_A`` and
  as the interaction ``dv * S_w`` with every signature coordinate, because the
  effect in this prior is ``slope(history) * (v - a_ref)``: the sparse model
  then selects which geometry coordinates modulate the dose response.
* The end state ``(x_last - mean) / sd`` is added as an explicit feature, since
  signatures describe increments and carry no absolute location.
* Accuracies are 5-fold cross-validated predictions of a LASSO at the alpha
  that LassoCV picks on the full set (a mild optimism, stated); selected
  coordinates come from stability selection over 40 half-sample refits.

Variants: state only; normalised signature of (t, A, Y) at levels 2 and 3;
the same without the time coordinate; the raw (unnormalised) path; the
log-signature; and per structure the normalised signature of the full
observed path (all non-zero columns).

Usage (public dotime package, from the repository root)::

    python results/reference/signature_2026-10/scripts/signature_stage1a.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import dotime
from dotime.benchmarks import load_benchmark
from dotime.evaluation import query_obs_levels

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = REPO / "results" / "reference" / "signature_2026-10"
DP = REPO / "results" / "reference" / "detection_power_2026-10" / "ident_v1_2_predictions.parquet"
PRIMARY_STRUCTURES = (
    "bi_variate",
    "back_door",
    "confounder_mediator",
    "front_door",
    "instrumental_variable",
    "mediator",
)
EPS = 0.1
SEED = 20261009


# --------------------------------------------------------------------------- signatures
def truncated_signature(path: np.ndarray, level: int) -> list[np.ndarray]:
    """Signature of a piecewise-linear path, truncated at ``level``, by Chen's relation.

    Args:
        path: ``(n + 1, d)`` points.
        level: 1, 2 or 3.

    Returns:
        ``[S1, S2, S3][:level]`` with shapes ``(d,)``, ``(d, d)``, ``(d, d, d)``.

    Raises:
        ValueError: For a level outside 1 to 3.
    """
    if level not in (1, 2, 3):
        raise ValueError("level must be 1, 2 or 3")
    d = path.shape[1]
    s1 = np.zeros(d)
    s2 = np.zeros((d, d))
    s3 = np.zeros((d, d, d))
    for inc in np.diff(path, axis=0):
        e2 = np.outer(inc, inc) / 2.0
        if level >= 3:
            # S <- S (x) exp(inc): level 3 collects every split of the word.
            s3 = (
                s3
                + np.einsum("ij,k->ijk", s2, inc)
                + np.einsum("i,jk->ijk", s1, e2)
                + np.einsum("i,j,k->ijk", inc, inc, inc) / 6.0
            )
        if level >= 2:
            s2 = s2 + np.outer(s1, inc) + e2
        s1 = s1 + inc
    return [s1, s2, s3][:level]


def log_signature(sig: list[np.ndarray]) -> list[np.ndarray]:
    """Truncated tensor logarithm of a signature, ``log(1 + X) = X - X^2/2 + X^3/3``.

    Args:
        sig: ``[S1, S2, S3]`` from :func:`truncated_signature` at level 3.

    Returns:
        ``[L1, L2, L3]`` of the same shapes.

    Raises:
        ValueError: If fewer than three levels are given.
    """
    if len(sig) < 3:
        raise ValueError("the log-signature needs level 3")
    s1, s2, s3 = sig
    l2 = s2 - 0.5 * np.outer(s1, s1)
    l3 = (
        s3
        - 0.5 * (np.einsum("i,jk->ijk", s1, s2) + np.einsum("ij,k->ijk", s2, s1))
        + np.einsum("i,j,k->ijk", s1, s1, s1) / 3.0
    )
    return [s1, l2, l3]


def flatten(sig: list[np.ndarray], names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Flatten signature levels into one vector with word labels like ``AY`` or ``tAY``.

    Args:
        sig: Signature levels.
        names: One letter per path dimension.

    Returns:
        The vector and its labels.
    """
    vals, labels = [], []
    for k, s in enumerate(sig, start=1):
        for idx in itertools.product(range(len(names)), repeat=k):
            vals.append(float(s[idx]))
            labels.append("".join(names[i] for i in idx))
    return np.array(vals), labels


# --------------------------------------------------------------------------- data
def primary_episodes(suite):
    """The registered primary set with per-episode arrays.

    Args:
        suite: The loaded 1.2.0 suite.

    Returns:
        The episodes and a dict of arrays: scm_id, structure, onset, A, Y, v,
        a_ref, y_true, y_obs, mu (per column), sd (per column).
    """
    eps, rec = (
        [],
        {
            k: []
            for k in ("scm_id", "structure", "onset", "A", "Y", "v", "a_ref", "y_true", "y_obs")
        },
    )
    for ep in suite:
        if ep.structure not in PRIMARY_STRUCTURES:
            continue
        y_obs = float(query_obs_levels(ep).reshape(-1)[0])
        y_true = float(ep.y_true[0])
        if abs(y_true - y_obs) < EPS:
            continue
        o = min(ep.intervention.times)
        a = ep.intervention.targets[0]
        eps.append(ep)
        rec["scm_id"].append(int(ep.scm_id))
        rec["structure"].append(ep.structure)
        rec["onset"].append(int(o))
        rec["A"].append(int(a))
        rec["Y"].append(int(ep.query_target[0]))
        rec["v"].append(float(ep.intervention.values))
        rec["a_ref"].append(float(ep.x_obs[o, a]))
        rec["y_true"].append(y_true)
        rec["y_obs"].append(y_obs)
    return eps, {k: np.array(v) for k, v in rec.items()}


def episode_paths(ep, onset: int, cols: list[int], normalise: bool, with_time: bool) -> np.ndarray:
    """The pre-onset path of selected columns, optionally normalised and time-augmented.

    Args:
        ep: The episode.
        onset: Intervention onset row.
        cols: Column indices in path order.
        normalise: Divide each column by its pre-onset mean and standard deviation.
        with_time: Prepend normalised time ``i / (onset - 1)``.

    Returns:
        ``(onset, len(cols) [+ 1])`` array.
    """
    x = ep.x_obs[:onset, cols].numpy().astype(np.float64)
    if normalise:
        mu = x.mean(axis=0)
        sd = np.maximum(x.std(axis=0, ddof=1), 1e-4)
        x = (x - mu) / sd
    if with_time:
        t = (np.arange(onset) / max(onset - 1, 1))[:, None]
        x = np.concatenate([t, x], axis=1)
    return x


def observed_columns(ep) -> list[int]:
    """Columns that carry data (hidden variables are released as zero columns)."""
    x = ep.x_obs.numpy()
    return [j for j in range(x.shape[1]) if np.abs(x[:, j]).max() > 0]


# --------------------------------------------------------------------------- features
def build_features(eps, rec, variant: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Feature matrix for one variant.

    Args:
        eps: Primary episodes.
        rec: Their arrays.
        variant: ``path`` ("AY" or "full"), ``level`` (0 for state only), ``normalise``,
            ``with_time``, ``log``.

    Returns:
        ``(Phi, labels, dv)``: features, their labels and the normalised do-value shift.
    """
    rows, labels, dvs = [], None, []
    for i, ep in enumerate(eps):
        o, a, y = int(rec["onset"][i]), int(rec["A"][i]), int(rec["Y"][i])
        if variant["path"] == "AY":
            cols, names = [a, y], ["A", "Y"]
        else:
            cols = observed_columns(ep)
            names = []
            k = 1
            for c in cols:
                if c == a:
                    names.append("A")
                elif c == y:
                    names.append("Y")
                else:
                    names.append(f"X{k}")
                    k += 1
        x_raw = ep.x_obs[:o, cols].numpy().astype(np.float64)
        mu = x_raw.mean(axis=0)
        sd = np.maximum(x_raw.std(axis=0, ddof=1), 1e-4)
        state = (x_raw[-1] - mu) / sd
        dv = (rec["v"][i] - rec["a_ref"][i]) / sd[cols.index(a)]
        dvs.append(dv)
        feats = list(state)
        feat_labels = [f"state_{n}" for n in names]
        if variant["level"] > 0:
            path = episode_paths(ep, o, cols, variant["normalise"], variant["with_time"])
            pnames = (["t"] if variant["with_time"] else []) + names
            sig = truncated_signature(path, variant["level"])
            if variant.get("log"):
                sig = log_signature(sig)
            vec, lab = flatten(sig, pnames)
            feats += list(vec)
            feat_labels += [("log_" if variant.get("log") else "sig_") + w for w in lab]
        base = np.array(feats)
        # The dose response: every feature also enters multiplied by the do-value shift.
        row = np.concatenate([[dv], base, dv * base])
        rows.append(row)
        if labels is None:
            labels = ["dv"] + feat_labels + [f"dv*{lab}" for lab in feat_labels]
    return np.array(rows), labels, np.array(dvs)


def fit_and_score(phi: np.ndarray, target: np.ndarray, labels: list[str], seed: int) -> dict:
    """Cross-validated LASSO with stability selection.

    Args:
        phi: Features.
        target: Normalised effect.
        labels: Feature labels.
        seed: Random seed for the folds and the half-sample refits.

    Returns:
        Effect-sign accuracy, RMSE of the normalised effect, the chosen alpha,
        the number of non-zero coefficients, and the stable coordinates with
        their selection frequency and coefficient sign.
    """
    scaler = StandardScaler().fit(phi)
    z = scaler.transform(phi)
    z = np.nan_to_num(z)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    lcv = LassoCV(cv=cv, n_alphas=40, max_iter=20000, random_state=seed).fit(z, target)
    alpha = float(lcv.alpha_)
    pred = cross_val_predict(Lasso(alpha=alpha, max_iter=20000), z, target, cv=cv)
    acc = float(np.mean(np.sign(pred) == np.sign(target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    rng = np.random.default_rng(seed)
    n = len(target)
    counts = np.zeros(z.shape[1])
    signs = np.zeros(z.shape[1])
    n_boot = 40
    for _ in range(n_boot):
        idx = rng.choice(n, size=n // 2, replace=False)
        m = Lasso(alpha=alpha, max_iter=20000).fit(z[idx], target[idx])
        nz = m.coef_ != 0
        counts += nz
        signs += np.sign(m.coef_)
    freq = counts / n_boot
    full = Lasso(alpha=alpha, max_iter=20000).fit(z, target)
    stable = sorted(
        [
            (labels[j], float(freq[j]), float(full.coef_[j]))
            for j in range(len(labels))
            if freq[j] >= 0.75
        ],
        key=lambda t: -abs(t[2]),
    )
    # (labels of the stable coordinates, their frequency, their standardised coefficient)
    return {
        "effect_sign_acc_cv": acc,
        "se": float(np.sqrt(acc * (1 - acc) / n)),
        "rmse_norm_effect_cv": rmse,
        "alpha": alpha,
        "n_features": int(z.shape[1]),
        "n_nonzero_full_fit": int(np.sum(full.coef_ != 0)),
        "stable_coordinates": [
            {"feature": f, "selection_freq": fr, "coef_std": c} for f, fr, c in stable[:25]
        ],
    }


def git_commit(repo: Path) -> str:
    """Short commit hash of a repository, ``-dirty`` when tracked files changed."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


VARIANTS = {
    "state_only": {"path": "AY", "level": 0, "normalise": True, "with_time": False},
    "norm_sig2_tAY": {"path": "AY", "level": 2, "normalise": True, "with_time": True},
    "norm_sig3_tAY": {"path": "AY", "level": 3, "normalise": True, "with_time": True},
    "norm_sig3_AY_notime": {"path": "AY", "level": 3, "normalise": True, "with_time": False},
    "raw_sig3_tAY": {"path": "AY", "level": 3, "normalise": False, "with_time": True},
    "norm_logsig3_tAY": {
        "path": "AY",
        "level": 3,
        "normalise": True,
        "with_time": True,
        "log": True,
    },
    "norm_sig2_full": {"path": "full", "level": 2, "normalise": True, "with_time": True},
    "norm_sig3_full": {"path": "full", "level": 3, "normalise": True, "with_time": True},
}


def main() -> int:
    """Run every variant pooled and per structure, compare with the estimators, write JSON.

    Returns:
        0.

    Raises:
        SystemExit: If the primary set is not the registered 3,528 episodes.
    """
    suite = load_benchmark("dot-Identifiability-v1", version="1.2.0")
    eps, rec = primary_episodes(suite)
    if len(eps) != 3528:
        raise SystemExit(f"primary set has {len(eps)} episodes, expected 3,528")
    sd_y = np.array(
        [
            max(float(ep.x_obs[: int(o), int(y)].std(unbiased=True)), 1e-4)
            for ep, o, y in zip(eps, rec["onset"], rec["Y"], strict=True)
        ]
    )
    effect = rec["y_true"] - rec["y_obs"]
    target = effect / sd_y
    digest = hashlib.sha256(
        np.array([rec["scm_id"], rec["y_true"], rec["y_obs"], rec["v"]], dtype=np.float64).tobytes()
    ).hexdigest()[:16]

    # The estimators on the same episodes, from the detection-power table.
    dp = pd.read_parquet(DP)
    dp = dp[dp["scm_id"].isin(rec["scm_id"])]
    ref = {}
    for est, g in dp.groupby("estimator"):
        g = g.set_index("scm_id").loc[rec["scm_id"]]
        ok = np.isfinite(g["pred"].to_numpy())
        if ok.sum() < len(g) // 2:
            continue
        ref[est] = {
            "effect_sign_acc": float(
                np.mean(
                    (np.sign(g["pred"].to_numpy() - g["y_obs"].to_numpy()) == np.sign(effect))[ok]
                )
            ),
            "n": int(ok.sum()),
        }

    out = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dotime_commit": git_commit(REPO),
        "dotime_version": dotime.__version__,
        "n_primary": len(eps),
        "primary_digest": digest,
        "target": "counterfactual effect (y_true - y_obs) over the outcome's pre-onset standard deviation",
        "slope_blind_ceiling": float(
            max(
                np.mean(effect / (rec["v"] - rec["a_ref"]) > 0),
                1 - np.mean(effect / (rec["v"] - rec["a_ref"]) > 0),
            )
        ),
        "estimators_same_episodes": ref,
        "variants": {k: dict(v) for k, v in VARIANTS.items()},
        "pooled": {},
        "per_structure": {s: {} for s in PRIMARY_STRUCTURES},
    }
    for name, variant in VARIANTS.items():
        if variant["path"] == "full":
            # Column layouts differ across structures, so the full path is fit per structure only.
            for s in PRIMARY_STRUCTURES:
                m = rec["structure"] == s
                sub_eps = [e for e, keep in zip(eps, m, strict=True) if keep]
                sub_rec = {k: v[m] for k, v in rec.items()}
                phi, labels, _ = build_features(sub_eps, sub_rec, variant)
                out["per_structure"][s][name] = fit_and_score(phi, target[m], labels, SEED)
                r = out["per_structure"][s][name]
                print(
                    f"{name:22s} {s:22s} acc {r['effect_sign_acc_cv']:.3f} (n {int(m.sum())}) nz {r['n_nonzero_full_fit']} top {[c['feature'] for c in r['stable_coordinates'][:4]]}",
                    flush=True,
                )
            continue
        phi, labels, _ = build_features(eps, rec, variant)
        out["pooled"][name] = fit_and_score(phi, target, labels, SEED)
        r = out["pooled"][name]
        print(
            f"{name:22s} pooled                 acc {r['effect_sign_acc_cv']:.3f} rmse {r['rmse_norm_effect_cv']:.3f} nz {r['n_nonzero_full_fit']}/{r['n_features']} top {[c['feature'] for c in r['stable_coordinates'][:5]]}",
            flush=True,
        )
        for s in PRIMARY_STRUCTURES:
            m = rec["structure"] == s
            out["per_structure"][s][name] = fit_and_score(phi[m], target[m], labels, SEED)
            rs = out["per_structure"][s][name]
            print(
                f"    {s:22s} acc {rs['effect_sign_acc_cv']:.3f} (n {int(m.sum())}) top {[c['feature'] for c in rs['stable_coordinates'][:4]]}",
                flush=True,
            )
    print(
        "estimators on the same episodes:",
        {k: round(v["effect_sign_acc"], 3) for k, v in ref.items()},
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage1a_dotime.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_DIR / 'stage1a_dotime.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
