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
    python scripts/eval_pfn_reference.py --suite dot-Identifiability-v1 \
        --ckpt-int .../s9ho_all_causal/do_over_time_pfn_best.pt \
        --ckpt-obs .../s9ho_all_obs/do_over_time_pfn_best.pt \
        --device cuda:0 --out results/reference/pfn_ident.json
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dotime.benchmarks import load_benchmark
from dotime.baselines import _episode_to_batch, _INT_TYPE_CODE  # protocol base
from dotime.evaluation import direction_accuracy
from dotime.models.loader import load_dotpfn


def episode_to_batch_interp(episode, n_max, device, observational=False):
    """_episode_to_batch with interpolation masking + optional obs-mode zeroing."""
    from dotime.normalization import normalize_batch

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
            [_norm_time(float(min(episode.intervention.times) if episode.intervention.times else 0))],
            device=device, dtype=torch.float32,
        ),
        "intervention_time_end": torch.tensor(
            [_norm_time(float(max(episode.intervention.times) if episode.intervention.times else 0))],
            device=device, dtype=torch.float32,
        ),
        "query_target": torch.tensor([int(episode.query_target[0])], device=device),
        "query_time": torch.tensor([_norm_time(q_time)], dtype=torch.float32, device=device),
        "Y_true": episode.y_true[:1].to(device),
    }
    if observational:
        # analyze_s7 obs protocol: zero every intervention feature
        for k in ("intervention_target", "intervention_type", "intervention_value",
                  "intervention_time_start", "intervention_time_end"):
            batch[k] = torch.zeros_like(batch[k])
    normalize_batch(batch)
    return batch


class PFNRef:
    def __init__(self, checkpoint, device="cpu", observational=False):
        self.device = device
        self.observational = observational
        self.model = load_dotpfn(checkpoint, device=device)
        self.n_max = int(getattr(self.model, "n_max", 41))

    @torch.no_grad()
    def predict(self, episode):
        batch = episode_to_batch_interp(episode, self.n_max, self.device, self.observational)
        out = self.model(batch)
        head = getattr(self.model, "quantile_head", None) or getattr(self.model, "bar_head", None)
        pred_norm = head.predict_mean(out).reshape(-1)
        q = int(episode.query_target[0])
        mean = batch["_norm_means"][0, q]
        std = batch["_norm_stds"][0, q]
        return (pred_norm * std + mean).cpu()


def run(model, episodes):
    ep_pred, ep_tgt, structs = [], [], []
    for ep in episodes:
        p = torch.as_tensor(model.predict(ep), dtype=torch.float32).reshape(-1).numpy()
        t = torch.as_tensor(ep.y_true, dtype=torch.float32).reshape(-1).numpy()
        ep_pred.append(p); ep_tgt.append(t); structs.append(ep.structure)
    pred = np.concatenate(ep_pred); tgt = np.concatenate(ep_tgt)
    rmse = float(np.sqrt(np.mean((pred - tgt) ** 2)))
    da = direction_accuracy(torch.from_numpy(pred), torch.from_numpy(tgt))
    # episode-cluster bootstrap for pooled RMSE
    rng = np.random.default_rng(0)
    sse = np.array([float(np.sum((p - t) ** 2)) for p, t in zip(ep_pred, ep_tgt)])
    cnt = np.array([len(t) for t in ep_tgt], dtype=np.float64)
    m = len(sse)
    boot = np.array([np.sqrt(sse[i].sum() / cnt[i].sum())
                     for i in (rng.integers(0, m, size=(1000, m)))])
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    # per-structure direction accuracy
    per_struct = {}
    for st in sorted(set(structs)):
        idx = [i for i, s in enumerate(structs) if s == st]
        p = np.concatenate([ep_pred[i] for i in idx]); t = np.concatenate([ep_tgt[i] for i in idx])
        d = direction_accuracy(torch.from_numpy(p), torch.from_numpy(t))
        per_struct[st] = {"rmse": float(np.sqrt(np.mean((p - t) ** 2))), "dir_acc": d["accuracy"]}
    return {"pooled_rmse": rmse, "rmse_ci95": ci, "dir_acc": da["accuracy"],
            "n_episodes": len(ep_pred), "per_structure": per_struct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ckpt-int", required=True)
    ap.add_argument("--ckpt-obs", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-structure", type=int, default=0, help="0 = full suite")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    episodes = list(load_benchmark(args.suite))
    if args.per_structure:
        from collections import defaultdict
        byst = defaultdict(list)
        for ep in episodes:
            byst[ep.structure].append(ep)
        episodes = [e for eps in byst.values() for e in eps[:args.per_structure]]
    print(f"[{args.suite}] evaluating {len(episodes)} episodes")

    out = {"suite": args.suite}
    for tag, ck, obs in [("PFN_int", args.ckpt_int, False), ("PFN_obs", args.ckpt_obs, True)]:
        t0 = time.time()
        model = PFNRef(ck, device=args.device, observational=obs)
        r = run(model, episodes)
        out[tag] = {"checkpoint": ck, **r}
        print(f"{tag}: RMSE={r['pooled_rmse']:.3f} CI[{r['rmse_ci95'][0]:.3f},{r['rmse_ci95'][1]:.3f}] "
              f"dir_acc={r['dir_acc']:.3f}  ({time.time()-t0:.0f}s)")
        for st, v in r["per_structure"].items():
            print(f"    {st:24s} rmse={v['rmse']:.3f} dir={v['dir_acc']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
