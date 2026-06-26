"""Parallel-safe suite generation (per-episode deterministic seeding).

This lives in the package (not the ``scripts/build_release.py`` driver) so the
multiprocessing worker :func:`make_episode` is importable as
``dotime._build.make_episode`` from worker processes under both the ``fork`` and
``spawn`` start methods — a function defined in a script's ``__main__`` cannot be
pickled by reference and breaks ``spawn``.

The scheme derives an independent seed per *episode index* (including the global
torch RNG, which parts of the prior use), so the generated suite is identical
regardless of the worker count and parallelises cleanly across processes. This is
the canonical generation scheme for the released suites.
"""

from __future__ import annotations


def scaled(n: int, scale: float) -> int:
    """Scale an episode count by ``scale`` (floored at 1)."""
    return max(1, round(n * scale))


def episode_seed(suite_seed: int, idx: int) -> int:
    """Deterministic per-episode seed (pure function of suite seed + index)."""
    return (suite_seed * 1_000_003 + idx) & 0x7FFFFFFF


def make_episode(spec: dict):
    """Build a single Episode from a spec dict (picklable; runs in a worker)."""
    import warnings as _w

    import torch as _torch

    _torch.set_num_threads(1)  # one core per worker; the pool provides parallelism
    _w.simplefilter("ignore", RuntimeWarning)
    from dotime.benchmarks import episode_from_pair, episode_from_sample

    kind, seed, idx, t_len = spec["kind"], spec["seed"], spec["idx"], spec["T"]
    # Seed the GLOBAL torch RNG per episode too: parts of the prior (e.g. the
    # Beta edge-probability draw) use the global generator rather than the
    # instance one, so this is what makes the v2 output independent of worker
    # count / processing order.
    _torch.manual_seed(seed)
    if kind == "generic":
        from dotime import DoTime

        x_obs, x_int, iv, _ = DoTime(seed=seed).generate_pair(T=t_len)
        return episode_from_pair(x_obs, x_int, iv, scm_id=idx, metadata={"tier": 1})
    if kind == "regime":
        from dotime import DoTime

        d = spec["num_regimes"]
        x_obs, x_int, iv, _ = DoTime(seed=seed).generate_regime_pair(T=t_len, num_regimes=d)
        return episode_from_pair(
            x_obs,
            x_int,
            iv,
            structure=f"regime_{d}",
            scm_id=idx,
            metadata={"tier": spec["tier"], "n_regimes": d},
        )
    if kind == "identifiability":
        from dotime.extended import ExtendedDoTime

        s = ExtendedDoTime(tscm_structure=spec["structure"], n_max=41, seed=seed).generate_sample(
            T=t_len
        )
        return episode_from_sample(
            s, structure=spec["structure"], scm_id=idx, metadata={"tier": spec["tier"]}
        )
    if kind == "continuous":
        from dotime.continuous import ContinuousExtendedPrior

        s = ContinuousExtendedPrior(tscm_structure=spec["structure"], seed=seed).generate_sample(
            T=t_len
        )
        tier = 1
        if "intervention_time_start" in s and "intervention_time_end" in s:
            frac = float(s["intervention_time_end"] - s["intervention_time_start"])
            tier = 1 if frac < 0.15 else (2 if frac < 0.3 else 3)
        return episode_from_sample(
            s, structure=spec["structure"], scm_id=idx, metadata={"tier": tier}
        )
    raise ValueError(f"unknown spec kind {kind!r}")


def episode_specs(cfg: dict, suite_seed: int, scale: float) -> list[dict]:
    """Build the per-episode spec list (deterministic seeds) for a suite config."""
    gen, t_len = cfg["generator"], cfg.get("T", 200)
    specs: list[dict] = []
    if gen == "generic":
        for i in range(scaled(cfg["n_episodes"], scale)):
            specs.append(
                {"kind": "generic", "idx": i, "seed": episode_seed(suite_seed, i), "T": t_len}
            )
    elif gen == "regime":
        densities = {int(k): int(v) for k, v in cfg["densities"].items()}
        per = max(1, scaled(cfg["n_episodes"], scale) // len(densities))
        for density, tier in densities.items():
            for _ in range(per):
                i = len(specs)
                specs.append(
                    {
                        "kind": "regime",
                        "idx": i,
                        "seed": episode_seed(suite_seed, i),
                        "T": t_len,
                        "num_regimes": density,
                        "tier": tier,
                    }
                )
    elif gen == "identifiability":
        per = scaled(cfg["episodes_per_structure"], scale)
        for structure, tier in cfg["structures"].items():
            for _ in range(per):
                i = len(specs)
                specs.append(
                    {
                        "kind": "identifiability",
                        "idx": i,
                        "seed": episode_seed(suite_seed, i),
                        "T": t_len,
                        "structure": structure,
                        "tier": tier,
                    }
                )
    elif gen == "continuous":
        structures = cfg["structures"]
        per = max(1, scaled(cfg["n_episodes"], scale) // len(structures))
        for structure in structures:
            for _ in range(per):
                i = len(specs)
                specs.append(
                    {
                        "kind": "continuous",
                        "idx": i,
                        "seed": episode_seed(suite_seed, i),
                        "T": t_len,
                        "structure": structure,
                    }
                )
    else:
        raise ValueError(f"unknown generator {gen!r}")
    return specs


def build_suite(cfg: dict, seed: int, scale: float, workers: int) -> list:
    """Build a suite via per-episode deterministic seeding across ``workers`` procs.

    Output is independent of ``workers`` (reproducible). ``workers<=1`` runs
    sequentially through the same per-episode path.
    """
    specs = episode_specs(cfg, seed, scale)
    if workers <= 1:
        return [make_episode(s) for s in specs]
    from concurrent.futures import ProcessPoolExecutor

    chunk = max(1, len(specs) // (workers * 8) or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(make_episode, specs, chunksize=chunk))
