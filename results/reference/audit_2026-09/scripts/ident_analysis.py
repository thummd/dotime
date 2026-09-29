#!/usr/bin/env python
"""Per-structure scores and validity metrics on dot-Identifiability-v1.

Scores: per-structure effect-sign accuracy of the CPU reference baselines on
the 1.1.0 shared-noise counterfactual targets, plus level/effect sign agreement.
Validity: invariant pass rates and per-arm target statistics for 1.0.0 (after
sidecar realignment) and 1.1.0.

The pooled numbers must reproduce ``results/reference/v1_1/ident_cpu_*.json``
exactly before any per-structure number is trusted (gate asserted below).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dotime import baselines
from dotime.benchmarks import load_benchmark
from dotime.evaluation import direction_accuracy, query_obs_levels, realign_episode

CPU = ["Zero", "Mean", "AR1", "VAR-OLS", "BackDoorOLS", "IV2SLS"]
EPS = 0.1


def arm_stats(x: np.ndarray) -> dict:
    """Summarise one target arm.

    Args:
        x: 1-D array of per-episode target values.

    Returns:
        Dict with n, nonzero fraction, mean and variance.
    """
    return {
        "n": int(x.size),
        "nonzero_frac": float(np.mean(x != 0.0)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x)),
    }


def onset_of(ep) -> int:
    """Return the first intervened time index of an episode.

    Args:
        ep: A dotime Episode.

    Returns:
        Integer onset index.
    """
    return int(min(ep.intervention.times))


def validity(episodes, y_obs: np.ndarray, hidden_by_structure: dict | None = None) -> dict:
    """Invariant pass rates and per-arm target stats for one suite version.

    Args:
        episodes: Episodes whose ``x_obs`` is canonically aligned.
        y_obs: Factual observational level at each episode's query.
        hidden_by_structure: Optional map from structure name to the canonical
            indices of its hidden variables, used for the hidden-leak check.

    Returns:
        Dict of pass rates and per-arm statistics.
    """
    n = len(episodes)
    pre_eq = do_ok = eff_ok = eff_is_level = zeroed = hidden_leak = hidden_n = 0
    per_struct_pre = defaultdict(lambda: [0, 0])
    y_int = np.array([float(ep.y_true.reshape(-1)[0]) for ep in episodes])
    for i, ep in enumerate(episodes):
        t0 = onset_of(ep)
        a = int(ep.intervention.targets[0])
        if not torch.any(ep.x_obs != 0) or not torch.any(ep.x_int != 0):
            zeroed += 1
        same = bool(torch.equal(ep.x_obs[:t0], ep.x_int[:t0]))
        pre_eq += same
        per_struct_pre[ep.structure][0] += same
        per_struct_pre[ep.structure][1] += 1
        val = ep.intervention.values
        val = float(val) if not torch.is_tensor(val) else float(val.reshape(-1)[0])
        do_ok += bool(abs(float(ep.x_int[t0, a]) - val) < 1e-5)
        ce = ep.metadata.get("y_causal_effect")
        if ce is not None:
            ce = float(np.asarray(ce).reshape(-1)[0])
            eff_ok += bool(abs(ce - (y_int[i] - y_obs[i])) < 1e-5)
            # The v1 erratum says the effect field stored the level itself.
            eff_is_level += bool(abs(ce - y_int[i]) < 1e-6)
        hid = (hidden_by_structure or {}).get(ep.structure)
        if hid:
            hidden_n += 1
            hidden_leak += bool(torch.any(ep.x_obs[:, list(hid)] != 0))
    eff = y_int - y_obs
    return {
        "n_episodes": n,
        "zeroed_frac": zeroed / n,
        "pre_onset_equal_frac": pre_eq / n,
        "pre_onset_equal_by_structure": {
            s: round(v[0] / v[1], 4) for s, v in sorted(per_struct_pre.items())
        },
        "do_value_in_treatment_col_frac": do_ok / n,
        "effect_identity_frac": eff_ok / n,
        "effect_field_equals_level_frac": eff_is_level / n,
        "hidden_meta_episodes": hidden_n,
        "hidden_leak_frac_of_those": (hidden_leak / hidden_n) if hidden_n else None,
        "arms": {
            "y_obs_level": arm_stats(y_obs),
            "y_int_level": arm_stats(y_int),
            "effect": arm_stats(eff),
        },
    }


def main() -> None:
    """Run both measurements and write JSON outputs.

    Raises:
        AssertionError: If the pooled numbers do not reproduce the released
            v1.1 reference JSONs, or a target arm fails the QA floor.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--v11-cache", default="output/v1_1")
    ap.add_argument(
        "--sidecar", default="results/reference/dot-Identifiability-v1.0.0_realignment.jsonl"
    )
    ap.add_argument("--out-dir", type=Path, default=Path("results/reference/audit_2026-09"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    side = {}
    with open(args.sidecar) as fh:
        for line in fh:
            r = json.loads(line)
            side[int(r["idx"])] = r
    # Hidden roles are fixed per structure; derive the map from the sidecar and
    # check it really is constant before using it on 1.1.0.
    alias = {"rct_no_confounding": "bi_variate"}
    hidden_by_structure: dict = {}
    for r in side.values():
        s = alias.get(r["structure"], r["structure"])
        h = tuple(r["hidden_canonical"])
        assert hidden_by_structure.setdefault(s, h) == h, (s, h, hidden_by_structure[s])
    hidden_by_structure = {s: list(h) for s, h in hidden_by_structure.items() if h}
    print("[hidden roles]", hidden_by_structure)

    # ---------------- v1.1.0 ----------------
    eps11 = list(load_benchmark("dot-Identifiability-v1", cache_dir=args.v11_cache))
    assert len(eps11) == 10800, len(eps11)
    yobs11 = np.array([float(query_obs_levels(ep)[0]) for ep in eps11])
    val11 = validity(eps11, yobs11, hidden_by_structure)
    for arm in ("y_obs_level", "y_int_level"):
        st = val11["arms"][arm]
        print(f"[v1.1 target QA] {arm}: {st}")
        assert st["nonzero_frac"] >= 0.5, f"v1.1 {arm} nonzero_frac {st['nonzero_frac']}"
    print(f"[v1.1 target QA] effect: {val11['arms']['effect']}")

    ref_eff = {
        r["baseline"]: r
        for r in json.loads(Path("results/reference/v1_1/ident_cpu_effect.json").read_text())[
            "rows"
        ]
    }
    ref_lvl = {
        r["baseline"]: r
        for r in json.loads(Path("results/reference/v1_1/ident_cpu_level.json").read_text())["rows"]
    }
    struct = np.array([ep.structure for ep in eps11])
    ytrue = np.array([float(ep.y_true.reshape(-1)[0]) for ep in eps11])
    out_rows = {}
    for name in CPU:
        model = baselines.get(name)
        pred = np.array(
            [
                float(torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1)[0])
                for ep in eps11
            ]
        )
        da_e = direction_accuracy(torch.from_numpy(pred - yobs11), torch.from_numpy(ytrue - yobs11))
        da_l = direction_accuracy(torch.from_numpy(pred), torch.from_numpy(ytrue))
        # Gate: the pooled numbers must match the released JSONs exactly.
        assert abs(da_e["accuracy"] - ref_eff[name]["dir_acc"]) < 1e-9, (
            name,
            da_e,
            ref_eff[name]["dir_acc"],
        )
        assert da_e["n_valid"] == ref_eff[name]["dir_n_valid"], (name, da_e["n_valid"])
        assert abs(da_l["accuracy"] - ref_lvl[name]["dir_acc"]) < 1e-9, (
            name,
            da_l,
            ref_lvl[name]["dir_acc"],
        )
        per = {}
        for s in sorted(set(struct)):
            m = struct == s
            e = direction_accuracy(
                torch.from_numpy(pred[m] - yobs11[m]), torch.from_numpy(ytrue[m] - yobs11[m])
            )
            lv = direction_accuracy(torch.from_numpy(pred[m]), torch.from_numpy(ytrue[m]))
            per[s] = {
                "dir_acc_effect": e["accuracy"],
                "n_valid_effect": e["n_valid"],
                "dir_acc_level": lv["accuracy"],
                "n_valid_level": lv["n_valid"],
                "rmse": float(np.sqrt(np.mean((pred[m] - ytrue[m]) ** 2))),
            }
        out_rows[name] = {
            "pooled_dir_acc_effect": da_e["accuracy"],
            "pooled_n_valid_effect": da_e["n_valid"],
            "pooled_dir_acc_level": da_l["accuracy"],
            "per_structure": per,
        }
        print(
            f"{name:12s} eff={da_e['accuracy']:.3f} (n={da_e['n_valid']}) lvl={da_l['accuracy']:.3f}  "
            + " ".join(
                f"{s[:6]}={v['dir_acc_effect']:.3f}/{v['n_valid_effect']}" for s, v in per.items()
            )
        )

    # Level/effect sign agreement on the true counterfactual effect.
    eff11 = ytrue - yobs11
    agree = {}
    for label, mask in {
        "both_abs_ge_0.1": (np.abs(ytrue) >= EPS) & (np.abs(eff11) >= EPS),
        "effect_abs_ge_0.1": np.abs(eff11) >= EPS,
    }.items():
        agree[label] = {
            "n": int(mask.sum()),
            "agreement": float(np.mean(np.sign(ytrue[mask]) == np.sign(eff11[mask]))),
            "by_structure": {
                s: {
                    "n": int((mask & (struct == s)).sum()),
                    "agreement": (
                        float(
                            np.mean(
                                np.sign(ytrue[mask & (struct == s)])
                                == np.sign(eff11[mask & (struct == s)])
                            )
                        )
                        if (mask & (struct == s)).sum()
                        else None
                    ),
                }
                for s in sorted(set(struct))
            },
        }
    print(
        "[v1.1] level/effect sign agreement:",
        {k: (v["n"], round(v["agreement"], 4)) for k, v in agree.items()},
    )
    exact_zero_effect = {s: float(np.mean(eff11[struct == s] == 0.0)) for s in sorted(set(struct))}
    print(
        "[v1.1] exactly-zero effect fraction by structure:",
        {k: round(v, 3) for k, v in exact_zero_effect.items()},
    )

    (args.out_dir / "ident_v1_1_per_structure.json").write_text(
        json.dumps(
            {
                "suite": "dot-Identifiability-v1",
                "version": "1.1.0",
                "source": args.v11_cache,
                "dir_eps": EPS,
                "gate": "pooled effect/level dir_acc and n_valid reproduce results/reference/v1_1/ident_cpu_{effect,level}.json exactly",
                "baselines": out_rows,
                "level_effect_sign_agreement": agree,
                "exactly_zero_effect_frac_by_structure": exact_zero_effect,
            },
            indent=2,
        )
    )

    # ---------------- v1.0.0 (realigned with the sidecar) ----------------
    raw10 = list(load_benchmark("dot-Identifiability-v1", version="1.0.0"))
    assert len(raw10) == 10800, len(raw10)
    n_misaligned = sum(
        1
        for ep in raw10
        if list(side[ep.scm_id]["canonical_perm"]) != sorted(side[ep.scm_id]["canonical_perm"])
    )
    hid_leak = sum(
        1
        for ep in raw10
        if side[ep.scm_id]["hidden_canonical"]
        and bool(
            torch.any(
                ep.x_obs[
                    :,
                    [
                        side[ep.scm_id]["canonical_perm"][h]
                        for h in side[ep.scm_id]["hidden_canonical"]
                    ],
                ]
                != 0
            )
        )
    )
    n_hidden = sum(1 for ep in raw10 if side[ep.scm_id]["hidden_canonical"])
    eps10 = [
        realign_episode(ep, side[ep.scm_id]["canonical_perm"], side[ep.scm_id]["hidden_canonical"])
        for ep in raw10
    ]
    yobs10 = np.array([float(side[ep.scm_id]["y_obs_corrected"]) for ep in eps10])
    val10 = validity(eps10, yobs10)
    val10["x_obs_misaligned_frac"] = n_misaligned / len(raw10)
    val10["hidden_leak_frac_of_hidden_episodes"] = hid_leak / n_hidden if n_hidden else None
    val10["n_hidden_episodes"] = n_hidden
    for arm in ("y_obs_level", "y_int_level"):
        print(f"[v1.0 target QA] {arm}: {val10['arms'][arm]}")
    val11["x_obs_misaligned_frac"] = (
        0.0  # 1.1.0 is written canonically (checked via pre-onset equality)
    )

    (args.out_dir / "ident_validity.json").write_text(
        json.dumps(
            {
                "suite": "dot-Identifiability-v1",
                "note": "1.0.0 invariants are computed after sidecar realignment; misalignment/leak are measured on the raw files",
                "v1_0_0": val10,
                "v1_1_0": val11,
            },
            indent=2,
        )
    )
    for k in (
        "zeroed_frac",
        "pre_onset_equal_frac",
        "do_value_in_treatment_col_frac",
        "effect_identity_frac",
    ):
        print(f"{k:32s} v1.0.0={val10[k]:.4f}  v1.1.0={val11[k]:.4f}")
    print(
        f"x_obs misaligned (raw v1.0.0) = {val10['x_obs_misaligned_frac']:.4f}; hidden leak = {val10['hidden_leak_frac_of_hidden_episodes']}"
    )


if __name__ == "__main__":
    main()
