"""Protocol-corrected Do-Over-Time-PFN eval on the released dot-* suites.

Reconstructs the training/eval protocol of the s9 checkpoints
(do-over-time-pfn ``scripts/run_s9_all_structures.sh`` +
``scripts/analyze_s7.evaluate_checkpoint``):

- causal masking = "interpolation": zero X_obs at/after onset BUT restore the
  treatment variable's observational value at the onset step (the vendored
  baseline zeroes everything, which is off-distribution for these models);
- observational mode = the separately trained ``*_obs`` checkpoint with ALL
  intervention features zeroed at eval time;
- predictions denormalized with the query variable's stats and compared
  against raw episode ``y_true`` (level space), as in analyze_s7.

Usage:
    dotime-eval-pfn --suite dot-Identifiability-v1 \
        --ckpt-int .../s9ho_all_causal/do_over_time_pfn_best.pt \
        --ckpt-obs .../s9ho_all_obs/do_over_time_pfn_best.pt \
        --device cuda:0 --out pfn_ident.json

The archived 1.0.0 Identifiability files store ``x_obs`` in topological order,
so pin that version together with the realignment sidecar. Its rows also
supply the observational level behind ``dir_acc_effect``:

    dotime-eval-pfn --suite dot-Identifiability-v1 --version 1.0.0 \
        --realignment results/reference/dot-Identifiability-v1.0.0_realignment.jsonl \
        --ckpt-int ... --ckpt-obs ... --device cuda:0 --out pfn_ident_realigned.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dotime.baselines import _INT_TYPE_CODE  # protocol base
from dotime.benchmarks import load_benchmark
from dotime.evaluation import (
    DEFAULT_DIR_TARGET,
    add_dir_target_argument,
    check_shared_noise,
    describe_dir_target,
    direction_accuracy,
    query_obs_levels,
    resolve_dir_target,
)
from dotime.qa import target_qa
from dotime.reference._realignment import (
    load_realignment,
    realign_episodes,
    sidecar_obs_levels,
)


def episode_to_batch_interp(episode, n_max, device, observational=False):
    """_episode_to_batch with interpolation masking + optional obs-mode zeroing."""
    from dotime.normalization import normalize_batch
    from dotime.observation import require_finite_history

    # A NaN cell would reach the normalization statistics of every query.
    require_finite_history(episode, "dotime-eval-pfn")
    x_obs = episode.x_obs
    t_len, n = x_obs.shape
    onset = min(episode.intervention.times) if episode.intervention.times else t_len
    int_target = episode.intervention.targets[0] if episode.intervention.targets else 0

    masked = x_obs.clone()
    masked[onset:] = 0.0
    # interpolation mask: keep the treatment's observational value at onset
    if onset < t_len:
        masked[onset, int_target] = x_obs[onset, int_target]

    x_padded = torch.zeros(t_len, n_max)
    x_padded[:, :n] = masked
    var_mask = torch.zeros(n_max)
    var_mask[:n] = 1.0

    raw_value = episode.intervention.values
    raw_value = float(raw_value) if isinstance(raw_value, (int, float)) else 0.0
    pre = x_obs[:onset, int_target] if onset > 0 else x_obs[:, int_target]
    int_value_norm = raw_value / max(float(pre.std().item()) if pre.numel() > 1 else 1.0, 1e-4)

    def _norm_time(v):
        return v if v <= 1.0 else v / t_len

    q_time = float(episode.query_time[0]) if episode.query_time.numel() else float(t_len - 1)
    batch = {
        "X_obs": x_padded.unsqueeze(0).to(device),
        "variable_mask": var_mask.unsqueeze(0).to(device),
        "int_onset_idx": torch.tensor([onset], device=device),
        "intervention_target": torch.tensor([int_target], device=device),
        "intervention_type": torch.tensor(
            [_INT_TYPE_CODE.get(episode.intervention.intervention_type.value, 0)], device=device
        ),
        "intervention_value": torch.tensor([int_value_norm], dtype=torch.float32, device=device),
        "intervention_time_start": torch.tensor(
            [
                _norm_time(
                    float(min(episode.intervention.times) if episode.intervention.times else 0)
                )
            ],
            device=device,
            dtype=torch.float32,
        ),
        "intervention_time_end": torch.tensor(
            [
                _norm_time(
                    float(max(episode.intervention.times) if episode.intervention.times else 0)
                )
            ],
            device=device,
            dtype=torch.float32,
        ),
        "query_target": torch.tensor([int(episode.query_target[0])], device=device),
        "query_time": torch.tensor([_norm_time(q_time)], dtype=torch.float32, device=device),
        "Y_true": episode.y_true[:1].to(device),
    }
    if observational:
        # analyze_s7 obs protocol: zero every intervention feature
        for k in (
            "intervention_target",
            "intervention_type",
            "intervention_value",
            "intervention_time_start",
            "intervention_time_end",
        ):
            batch[k] = torch.zeros_like(batch[k])
    normalize_batch(batch)
    return batch


class PFNRef:
    def __init__(self, checkpoint, device="cpu", observational=False):
        self.device = device
        self.observational = observational
        # The PFN architecture needs the `models` extra (pfns); import on use
        # so the console script's --help works without it.
        try:
            from dotime.models.loader import load_dotpfn
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise SystemExit(
                "The PFN architecture is required for this evaluator: pip install 'dotime[models]'"
            ) from exc

        self.model = load_dotpfn(checkpoint, device=device)
        self.n_max = int(getattr(self.model, "n_max", 41))

    @torch.no_grad()
    def predict(self, episode):
        batch = episode_to_batch_interp(episode, self.n_max, self.device, self.observational)
        out = self.model(batch)
        head = getattr(self.model, "quantile_head", None) or getattr(self.model, "bar_head", None)
        if head is None:
            raise AttributeError(
                f"{type(self.model).__name__} has neither a quantile_head nor a bar_head"
            )
        pred_norm = head.predict_mean(out).reshape(-1)
        q = int(episode.query_target[0])
        mean = batch["_norm_means"][0, q]
        std = batch["_norm_stds"][0, q]
        return (pred_norm * std + mean).cpu()


def run(model, episodes, dir_target=DEFAULT_DIR_TARGET, realignment=None):
    """Score one model on the episodes under both direction-accuracy targets.

    Args:
        model: Object whose ``predict(episode)`` returns the level prediction
            at the query, such as :class:`PFNRef`.
        episodes: The episodes to score.
        dir_target: ``"level"``, ``"effect"`` or ``"auto"``, the target that
            the headline ``dir_acc`` reports (see
            :func:`~dotime.evaluation.resolve_dir_target`). Both are always
            computed. Defaults to :data:`~dotime.evaluation.DEFAULT_DIR_TARGET`.
        realignment: Optional ``{scm_id: row}`` map from
            :func:`~dotime.reference._realignment.load_realignment`. When given,
            each episode's observational level is its row's ``y_obs_corrected``
            instead of the level read from ``x_obs``.

    Returns:
        Dict with the pooled RMSE and its episode-cluster bootstrap CI, the
        headline, level and effect direction accuracies, the scored target,
        whether the pairs share their noise, and per-structure metrics.

    Raises:
        ValueError: If ``dir_target`` is not a mode.
        KeyError: If ``realignment`` is given but has no row for an episode.
    """
    episodes = list(episodes)
    noise = check_shared_noise(episodes)
    target = resolve_dir_target(dir_target, noise, warn=False)
    ep_pred, ep_tgt, ep_obs, structs = [], [], [], []
    for ep in episodes:
        p = torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1).numpy()
        t = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).numpy()
        ep_pred.append(p)
        ep_tgt.append(t)
        structs.append(ep.structure)
        # Always collect the observational level so BOTH scorings come from
        # one prediction pass (predictions do not depend on the target). With
        # a sidecar there is no fallback to x_obs, whose column may be another
        # variable in a 1.0.0 episode.
        if realignment is not None:
            ep_obs.append(np.asarray([realignment[ep.scm_id]["y_obs_corrected"]], dtype=np.float32))
        else:
            ep_obs.append(query_obs_levels(ep).cpu().numpy())
    pred = np.concatenate(ep_pred)
    tgt = np.concatenate(ep_tgt)
    obs = np.concatenate(ep_obs)
    # RMSE stays level-space; the y_obs offset cancels in pred - tgt anyway.
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    da_level = direction_accuracy(torch.from_numpy(pred), torch.from_numpy(tgt))
    da_effect = direction_accuracy(torch.from_numpy(pred - obs), torch.from_numpy(tgt - obs))
    da = da_effect if target == "effect" else da_level
    # episode-cluster bootstrap for pooled RMSE
    rng = np.random.default_rng(0)
    sse = np.array([float(np.sum((p - t) ** 2)) for p, t in zip(ep_pred, ep_tgt, strict=True)])
    cnt = np.array([len(t) for t in ep_tgt], dtype=np.float64)
    m = len(sse)
    boot = np.array(
        [np.sqrt(sse[i].sum() / cnt[i].sum()) for i in (rng.integers(0, m, size=(1000, m)))]
    )
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    # per-structure direction accuracy (skip unstructured suites, e.g. Generic)
    per_struct = {}
    uniq = sorted(s for s in set(structs) if s is not None)
    for st in uniq:
        idx = [i for i, s in enumerate(structs) if s == st]
        p = np.concatenate([ep_pred[i] for i in idx])
        t = np.concatenate([ep_tgt[i] for i in idx])
        o = np.concatenate([ep_obs[i] for i in idx])
        d_l = direction_accuracy(torch.from_numpy(p), torch.from_numpy(t))
        d_e = direction_accuracy(torch.from_numpy(p - o), torch.from_numpy(t - o))
        d = d_e if target == "effect" else d_l
        per_struct[st] = {
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "dir_acc": d["accuracy"],
            "dir_acc_level": d_l["accuracy"],
            "dir_acc_effect": d_e["accuracy"],
        }
    import math as _m

    _se = (
        _m.sqrt(da["accuracy"] * (1 - da["accuracy"]) / da["n_valid"])
        if da["n_valid"]
        else float("nan")
    )
    return {
        "pooled_rmse": rmse,
        "rmse_ci95": ci,
        "dir_acc": da["accuracy"],
        "dir_n_valid": da["n_valid"],
        "dir_acc_se": _se,
        "dir_acc_level": da_level["accuracy"],
        "dir_acc_effect": da_effect["accuracy"],
        "dir_target": target,
        "pairs_share_noise": noise.shared,
        "n_episodes": len(ep_pred),
        "per_structure": per_struct,
    }


def main(argv: list[str] | None = None) -> None:
    """Score the Do-Over-Time-PFN int/obs checkpoint pair on a suite.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``, which is how
            the ``dotime-eval-pfn`` console script calls it.

    Raises:
        SystemExit: On invalid arguments, or if the PFN architecture
            (``dotime[models]``) is not installed.
        OSError: If the ``--realignment`` sidecar cannot be read.
        ValueError: If the sidecar is malformed or does not describe the
            evaluated episodes, e.g. a 1.0.0 sidecar against suite 1.1.0.
        dotime.qa.TargetQAError: If the evaluated targets fail target QA and
            ``--target-qa`` is ``enforce``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument(
        "--version",
        default="latest",
        help="Suite version to load, e.g. 1.0.0 (default: the registry's current version).",
    )
    ap.add_argument("--ckpt-int", required=True)
    ap.add_argument("--ckpt-obs", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-structure", type=int, default=0, help="0 = full suite")
    ap.add_argument("--out", type=Path, default=None)
    add_dir_target_argument(ap)
    ap.add_argument(
        "--exclude-self-queries",
        action="store_true",
        help="Drop episodes whose query targets the intervened variable (continuous "
        "suite: ~1/3 of episodes; in-window hard self-queries equal the do-value).",
    )
    ap.add_argument(
        "--realignment",
        type=Path,
        default=None,
        help="JSONL realignment sidecar for dot-Identifiability-v1 1.0.0: permutes "
        "x_obs to canonical order, zeroes hidden variables and supplies the "
        "corrected y_obs for dir_acc_effect. Every evaluated episode must match "
        "its row, so pair it with --version 1.0.0.",
    )
    ap.add_argument(
        "--target-qa",
        choices=["enforce", "warn"],
        default="enforce",
        help="Log and assert per-arm target statistics of the evaluated episodes "
        "before any model runs (dotime.qa). 'warn' reports a failure and scores anyway.",
    )
    args = ap.parse_args(argv)
    # Read the sidecar first: a bad path should fail before a suite download.
    realignment = load_realignment(args.realignment) if args.realignment is not None else None

    suite = load_benchmark(args.suite, version=args.version)
    episodes = list(suite)
    if args.exclude_self_queries:
        n0 = len(episodes)
        episodes = [ep for ep in episodes if not ep.is_self_query]
        print(f"[{args.suite}] excluded {n0 - len(episodes)} self-query episodes")
    if realignment is not None:
        # Every episode must match its row, so a sidecar from another suite
        # version stops the run before a checkpoint is loaded.
        episodes = realign_episodes(episodes, realignment)
        print(
            f"[{args.suite}] realigned x_obs of {len(episodes)} episodes "
            f"with {args.realignment.name}"
        )
    if args.per_structure:
        from collections import defaultdict

        byst = defaultdict(list)
        for ep in episodes:
            byst[ep.structure].append(ep)
        episodes = [e for eps in byst.values() for e in eps[: args.per_structure]]
    print(f"[{args.suite} v{suite.meta.version}] evaluating {len(episodes)} episodes")
    # On exactly the episodes scored below, and before a checkpoint is loaded.
    noise = check_shared_noise(episodes)
    dir_target = resolve_dir_target(args.dir_target, noise, warn=False)
    print(f"[{args.suite}] {describe_dir_target(args.dir_target, dir_target, noise)}")
    qa_report = target_qa(
        episodes,
        obs_levels=sidecar_obs_levels(episodes, realignment),
        dir_target=dir_target,
        raise_on_failure=args.target_qa == "enforce",
    )

    out = {
        "suite": args.suite,
        "suite_version": suite.meta.version,
        "realigned": realignment is not None,
        # File name only: an absolute path would leak the machine's layout
        # into a released result JSON.
        "realignment_sidecar": args.realignment.name if realignment is not None else None,
        "exclude_self_queries": args.exclude_self_queries,
        "dir_target": dir_target,
        "dir_target_mode": args.dir_target,
        "pairs_share_noise": noise.shared,
        "target_qa": qa_report.to_dict(),
    }
    for tag, ck, obs in [("PFN_int", args.ckpt_int, False), ("PFN_obs", args.ckpt_obs, True)]:
        t0 = time.time()
        model = PFNRef(ck, device=args.device, observational=obs)
        r = run(model, episodes, dir_target=dir_target, realignment=realignment)
        out[tag] = {"checkpoint": ck, **r}
        _da = r["dir_acc"] if r["dir_acc"] is not None else float("nan")
        print(
            f"{tag}: RMSE={r['pooled_rmse']:.3f} CI[{r['rmse_ci95'][0]:.3f},{r['rmse_ci95'][1]:.3f}] "
            f"dir_acc={_da:.3f}  ({time.time() - t0:.0f}s)"
        )
        for st, v in r["per_structure"].items():
            vd = v["dir_acc"] if v["dir_acc"] is not None else float("nan")
            print(f"    {st:24s} rmse={v['rmse']:.3f} dir={vd:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
