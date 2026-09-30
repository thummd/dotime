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

# Build-config keys that episode_specs copies into every spec of a suite that
# sets them. Each one switches on a behaviour the frozen configs never set
# (pair_mode excepted: their identifiability specs already carry it with the
# same value), so their specs stay unchanged.
_OPT_IN_SPEC_KEYS = (
    "record_graph",
    "prior_config",
    "chain_prob",
    "regime_switching_prob",
    "pair_mode",
    "latent",
    "tier_n_edges",
)


def scaled(n: int, scale: float) -> int:
    """Scale an episode count by ``scale`` (floored at 1)."""
    return max(1, round(n * scale))


def episode_seed(suite_seed: int, idx: int) -> int:
    """Deterministic per-episode seed (pure function of suite seed + index)."""
    return (suite_seed * 1_000_003 + idx) & 0x7FFFFFFF


def identifiability_retry_seed(seed: int, attempt: int) -> int:
    """Deterministic seed for the ``attempt``-th resample of an identifiability episode.

    ``ExtendedDoTime`` seeds a ``numpy.random.RandomState``, which only accepts
    seeds below ``2**32``; the unmasked ``seed * 100003 + attempt`` perturbation
    used by the torch-only generic/regime branches overflows it.

    Args:
        seed: The episode's base seed (attempt 0 uses it unchanged).
        attempt: Resample index, 0 for the first try.

    Returns:
        ``seed`` for ``attempt == 0``, otherwise a 31-bit perturbation of it.
    """
    return seed if attempt == 0 else (seed * 100003 + attempt) & 0x7FFFFFFF


def arms_zeroed(*arms) -> bool:
    """Whether any arm of a generated pair came back all-zero (diverged).

    The generators replace a diverged simulation with zeros, and the
    observational and interventional arms are separate simulation calls, so one
    arm can diverge while the other survives. Such a half-diverged pair has no
    valid target: a zeroed interventional arm stores ``y_true == 0``, and a
    zeroed observational arm gives every history-based model an all-zero input
    against a nonzero target. Every branch of :func:`make_episode` (and the
    ``dotime-generate`` parquet writer) flags and resamples on this one rule.

    Args:
        *arms: The episode's trajectory tensors, e.g. ``(x_obs, x_int)``.

    Returns:
        ``True`` if at least one arm has no nonzero entry.
    """
    return any(float(arm.abs().max()) == 0.0 for arm in arms)


def _with_graph(spec: dict, ep, source, columns=None):
    """Attach the episode's ground-truth lagged graph when the spec asks for it.

    With ``spec["record_graph"]`` set, ``ep.metadata["graph"]`` receives
    :meth:`dotime.graph_meta.LaggedGraph.to_dict` plus a ``"path"`` list with one
    :func:`dotime.graph_meta.path_lag` summary per query, from the intervention
    targets to that query's column. Extraction only reads the SCM, so the
    tensors and every random stream are the same as without the flag.

    Args:
        spec: The episode spec passed to :func:`make_episode`.
        ep: The built episode.
        source: The sampled SCM, or ``("identifiability", structure)`` /
            ``("continuous", structure)`` for the named-structure generators,
            whose graph is fixed by the structure.
        columns: SCM node name of each released column, for a builder that
            releases only some nodes. ``None`` means every node, in the SCM's
            topological order.

    Returns:
        ``ep``, with the graph added to its metadata if requested.

    Raises:
        TypeError: If ``spec["record_graph"]`` is not a bool.
        ValueError: If ``source`` names an unknown generator kind.
    """
    flag = spec.get("record_graph", False)
    # A strict check, because a truthy string such as "false" would otherwise
    # switch the recording on.
    if not isinstance(flag, bool):
        raise TypeError(f"spec['record_graph'] must be a bool, got {type(flag).__name__}")
    if not flag:
        return ep
    from dotime.graph_meta import LaggedGraph, path_lag

    if isinstance(source, tuple):
        kind, structure = source
        if kind == "identifiability":
            graph = LaggedGraph.from_structure(structure)
        elif kind == "continuous":
            graph = LaggedGraph.from_continuous_structure(structure)
        else:
            raise ValueError(f"no graph source for generator kind {kind!r}")
    else:
        graph = LaggedGraph.from_scm(source, columns=columns)
    sources = [int(t) for t in ep.intervention.targets]
    paths = [
        {"target": int(q), **path_lag(graph, sources, int(q)).to_dict()}
        for q in ep.query_target.reshape(-1).tolist()
    ]
    ep.metadata["graph"] = {**graph.to_dict(), "path": paths}
    return ep


def make_episode(spec: dict):
    """Build a single Episode from a spec dict (picklable; runs in a worker)."""
    import warnings as _w

    import torch as _torch

    _torch.set_num_threads(1)  # one core per worker; the pool provides parallelism
    _w.simplefilter("ignore", RuntimeWarning)
    from dotime.benchmarks import episode_from_pair, episode_from_sample

    kind, seed, idx, t_len = spec["kind"], spec["seed"], spec["idx"], spec["T"]
    # ``stability_retries``: when either arm of a generic, regime or
    # identifiability episode comes back zeroed (diverged), resample with a
    # deterministic seed perturbation up to this many times (the continuous
    # branch does not retry). Default 0 makes a single attempt whatever the
    # rule, so it preserves the exact v1.0.0 tensors. Hardened builds set it >0:
    # otherwise the generic prior ships ~30% zeroed episodes and identifiability
    # ~5%, because ExtendedDoTime's internal retry only rejects NaN or
    # |x| >= 10, which a zeroed arm passes.
    retries = int(spec.get("stability_retries", 0))
    # Seed the GLOBAL torch RNG per episode too: parts of the prior (e.g. the
    # Beta edge-probability draw) use the global generator rather than the
    # instance one, so this is what makes the v2 output independent of worker
    # count / processing order.
    _torch.manual_seed(seed)

    if kind == "generic":
        from dotime import DoTime

        for attempt in range(retries + 1):
            s = seed if attempt == 0 else seed * 100003 + attempt
            _torch.manual_seed(s)
            x_obs, x_int, iv, scm = DoTime(seed=s).generate_pair(T=t_len)
            if attempt == retries or not arms_zeroed(x_obs, x_int):
                break
        # Flag zeroed (diverged) episodes explicitly: v1.0.0 shipped them
        # unflagged, which the datasheet erratum documents. A pair with only
        # one arm zeroed counts too (about 1.4% of v1.0.0 Generic episodes).
        # Tensors and RNG streams are unchanged; only metadata_json gains the
        # key (v1.1+).
        ep = episode_from_pair(
            x_obs,
            x_int,
            iv,
            scm_id=idx,
            metadata={"tier": 1, "diverged": arms_zeroed(x_obs, x_int)},
        )
        return _with_graph(spec, ep, scm)
    if kind == "regime":
        from dotime import DoTime

        d = spec["num_regimes"]
        for attempt in range(retries + 1):
            s = seed if attempt == 0 else seed * 100003 + attempt
            _torch.manual_seed(s)
            # Not named ``scm``: the generic branch above already binds that
            # name as a TemporalSCM for the type checker.
            x_obs, x_int, iv, regime_scm = DoTime(seed=s).generate_regime_pair(
                T=t_len, num_regimes=d
            )
            if attempt == retries or not arms_zeroed(x_obs, x_int):
                break
        ep = episode_from_pair(
            x_obs,
            x_int,
            iv,
            structure=f"regime_{d}",
            scm_id=idx,
            metadata={"tier": spec["tier"], "n_regimes": d, "diverged": arms_zeroed(x_obs, x_int)},
        )
        return _with_graph(spec, ep, regime_scm)
    if kind == "identifiability":
        from dotime.extended import ExtendedDoTime

        # Same deterministic resampling and either-arm rule as the
        # generic/regime branches: with shared noise the intervention clamp can
        # keep the interventional arm finite while the observational arm
        # diverges, and such a pair has no valid target.
        for attempt in range(retries + 1):
            s_seed = identifiability_retry_seed(seed, attempt)
            _torch.manual_seed(s_seed)
            s = ExtendedDoTime(
                tscm_structure=spec["structure"],
                n_max=41,
                seed=s_seed,
                pair_mode=spec.get("pair_mode", "interventional"),
                # Per-structure query offsets (v1.2 protocol). Default (0, 0)
                # reproduces the v1.0.0 / v1.1.0 builds, which query at onset.
                query_offset_range=tuple(spec.get("query_offset_range", (0, 0))),
            ).generate_sample(T=t_len)
            zeroed = arms_zeroed(s["X_int"], s["X_obs_full"])
            if attempt == retries or not zeroed:
                break
        ep = episode_from_sample(
            s,
            structure=spec["structure"],
            scm_id=idx,
            metadata={
                "tier": spec["tier"],
                "diverged": zeroed,
                "pair_mode": spec.get("pair_mode", "interventional"),
                "query_offset_range": list(spec.get("query_offset_range", (0, 0))),
            },
        )
        return _with_graph(spec, ep, ("identifiability", spec["structure"]))
    if kind == "continuous":
        from dotime.continuous import ContinuousExtendedPrior

        s = ContinuousExtendedPrior(tscm_structure=spec["structure"], seed=seed).generate_sample(
            T=t_len
        )
        tier = 1
        if "intervention_time_start" in s and "intervention_time_end" in s:
            frac = float(s["intervention_time_end"] - s["intervention_time_start"])
            tier = 1 if frac < 0.15 else (2 if frac < 0.3 else 3)
        # Tag rather than drop self-queries (query on the treated variable): they
        # are ~1/3 of continuous queries by construction (uniform target draw
        # over three observable variables). Inside the window of a hard
        # intervention the target equals the do-value, so evaluators need the
        # flag to report with/without them. The window end is recorded because
        # the released InterventionSpec carries only the onset.
        q_abs = float(_torch.as_tensor(s["t_query"]).reshape(-1)[0])
        meta = {
            "tier": tier,
            "self_query": int(s["query_target"].reshape(-1)[0]) == int(s["intervention_target"]),
            "query_in_window": bool(float(s["t_int_start"]) <= q_abs <= float(s["t_int_end"])),
            "window_end_idx": int((s["times"] <= float(s["t_int_end"])).sum().item()) - 1,
        }
        ep = episode_from_sample(s, structure=spec["structure"], scm_id=idx, metadata=meta)
        return _with_graph(spec, ep, ("continuous", spec["structure"]))
    if kind == "generic_configured":
        return _make_configured_generic_episode(spec)
    raise ValueError(f"unknown spec kind {kind!r}")


def _forward_opt_in(cfg: dict, specs: list[dict]) -> list[dict]:
    """Copy the opt-in build-config keys a suite config sets into every spec.

    Args:
        cfg: The suite config.
        specs: The per-episode specs built from it.

    Returns:
        ``specs`` itself when ``cfg`` sets none of :data:`_OPT_IN_SPEC_KEYS`,
        so frozen configs keep their exact specs. Otherwise new spec dicts that
        also carry those keys.
    """
    extra = {k: cfg[k] for k in _OPT_IN_SPEC_KEYS if k in cfg}
    if not extra:
        return specs
    return [{**s, **extra} for s in specs]


def episode_specs(cfg: dict, suite_seed: int, scale: float) -> list[dict]:
    """Build the per-episode spec list (deterministic seeds) for a suite config."""
    gen, t_len = cfg["generator"], cfg.get("T", 200)
    retries = int(cfg.get("stability_retries", 0))
    specs: list[dict] = []
    if gen == "generic":
        kind = _generic_spec_kind(cfg)
        for i in range(scaled(cfg["n_episodes"], scale)):
            specs.append(
                {
                    "kind": kind,
                    "idx": i,
                    "seed": episode_seed(suite_seed, i),
                    "T": t_len,
                    "stability_retries": retries,
                }
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
                        "stability_retries": retries,
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
                        "pair_mode": cfg.get("pair_mode", "interventional"),
                        "stability_retries": retries,
                        "query_offset_range": tuple(
                            cfg.get("query_offsets", {}).get(structure, (0, 0))
                        ),
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
    return _forward_opt_in(cfg, specs)


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


# The opt-in keys that reconfigure the generic prior. Setting any of them moves
# a generic suite to the "generic_configured" spec kind.
_GENERIC_PRIOR_KEYS = (
    "prior_config",
    "chain_prob",
    "regime_switching_prob",
    "pair_mode",
    "latent",
    "tier_n_edges",
)
# Options DoTime reads from its config on top of the DEFAULT_CONFIG keys.
_EXTRA_PRIOR_CONFIG_KEYS = ("N_min", "hardening", "regime_canonical_weights")


def _generic_spec_kind(cfg: dict) -> str:
    """Validate the opt-in prior options of a generic suite and pick its spec kind.

    The checks run while the specs are built, so a bad release config fails
    before any worker starts.

    Args:
        cfg: The config of a suite with ``generator: generic``.

    Returns:
        ``"generic_configured"`` when ``cfg`` sets any of the options, which
        :func:`make_episode` builds with :func:`_make_configured_generic_episode`.
        Otherwise ``"generic"``, the legacy path of the released suites.

    Raises:
        ValueError: If ``prior_config`` is not a mapping or sets a key the prior
            does not read, if ``chain_prob`` or ``regime_switching_prob`` is not
            a probability, if ``pair_mode`` or ``latent`` has an unknown value,
            if ``pair_mode`` is ``"counterfactual"`` without
            ``regime_switching_prob: 0``, or if ``tier_n_edges`` is not a
            strictly ascending list of ints.
    """
    from itertools import pairwise

    from dotime.utils import DEFAULT_CONFIG

    if not any(k in cfg for k in _GENERIC_PRIOR_KEYS):
        return "generic"
    prior_config = cfg.get("prior_config", {})
    if not isinstance(prior_config, dict):
        raise ValueError(f"prior_config must be a mapping, got {type(prior_config).__name__}")
    # A misspelt key would otherwise be merged into the config and ignored.
    unknown = set(prior_config) - set(DEFAULT_CONFIG) - set(_EXTRA_PRIOR_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"unknown prior_config keys {sorted(map(str, unknown))}; valid keys are "
            f"{sorted([*DEFAULT_CONFIG, *_EXTRA_PRIOR_CONFIG_KEYS])}"
        )
    for key in ("chain_prob", "regime_switching_prob"):
        p = cfg.get(key, 0.0)
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise ValueError(f"{key} must be a probability in [0, 1], got {p!r}")
    pair_mode = cfg.get("pair_mode", "interventional")
    if pair_mode not in ("interventional", "counterfactual"):
        raise ValueError(f"pair_mode must be interventional or counterfactual, got {pair_mode!r}")
    # DoTime's own default draws regime-switching SCMs, whose arms cannot
    # share noise, so a counterfactual suite has to switch them off itself.
    if pair_mode == "counterfactual" and cfg.get("regime_switching_prob") != 0:
        raise ValueError("pair_mode: counterfactual needs regime_switching_prob: 0")
    if cfg.get("latent", "drop") != "drop":
        raise ValueError(f"latent must be 'drop', got {cfg['latent']!r}")
    edges = cfg.get("tier_n_edges", [])
    if (
        not isinstance(edges, (list, tuple))
        or any(isinstance(e, bool) or not isinstance(e, int) for e in edges)
        or any(a >= b for a, b in pairwise(edges))
    ):
        raise ValueError(f"tier_n_edges must be a strictly ascending list of ints, got {edges!r}")
    return "generic_configured"


def _make_configured_generic_episode(spec: dict):
    """Build one episode of a generic suite that configures the prior.

    The retry rule is the legacy generic branch's: attempt ``a > 0`` reseeds
    both the global torch RNG and the prior with ``seed * 100003 + a``, and an
    episode is resampled while either arm is zeroed. Options the spec does not
    set keep the defaults of :class:`~dotime.prior.DoTime`.

    Args:
        spec: A ``"generic_configured"`` spec from :func:`episode_specs`, with
            the opt-in keys the suite sets.

    Returns:
        The :class:`~dotime.benchmarks.Episode`. Its metadata holds ``tier``
        (1 plus the number of ``tier_n_edges`` below the SCM's full variable
        count), ``diverged``, ``pair_mode`` and, with ``latent: drop``, the
        ``latent`` record of :func:`_drop_latent_columns`.

    Raises:
        TypeError: If ``prior_config`` holds a value of the wrong type (see
            :class:`~dotime.prior.DoTime`).
        ValueError: If ``prior_config`` holds an invalid value, or if
            ``pair_mode`` is ``"counterfactual"`` while the prior may draw
            regime-switching SCMs.
    """
    import torch as _torch

    from dotime import DoTime
    from dotime.benchmarks import episode_from_pair

    seed, t_len = spec["seed"], spec["T"]
    retries = int(spec.get("stability_retries", 0))
    pair_mode = spec.get("pair_mode", "interventional")
    prior_kwargs = {k: spec[k] for k in ("chain_prob", "regime_switching_prob") if k in spec}
    for attempt in range(retries + 1):
        s = seed if attempt == 0 else seed * 100003 + attempt
        _torch.manual_seed(s)
        prior = DoTime(config=spec.get("prior_config"), seed=s, **prior_kwargs)
        x_obs, x_int, iv, scm = prior.generate_pair(T=t_len, pair_mode=pair_mode)
        if attempt == retries or not arms_zeroed(x_obs, x_int):
            break
    n_vars_full = int(x_obs.shape[1])
    metadata: dict = {
        # Tiers follow the size of the simulated graph, hidden variables
        # included, so dropping latents does not move an episode between tiers.
        "tier": 1 + sum(edge < n_vars_full for edge in spec.get("tier_n_edges", ())),
        "diverged": arms_zeroed(x_obs, x_int),
        "pair_mode": pair_mode,
    }
    if spec.get("latent") == "drop":
        # Before episode_from_pair, so the query is chosen among the columns
        # that are released.
        x_obs, x_int, iv, metadata["latent"] = _drop_latent_columns(
            x_obs, x_int, iv, list(scm._topo)
        )
    return episode_from_pair(x_obs, x_int, iv, scm_id=spec["idx"], metadata=metadata)


def _drop_latent_columns(x_obs, x_int, intervention, names: list[str]) -> tuple:
    """Remove the hidden variables of a sampled SCM from both arms of a pair.

    The generic graph builder names the nodes it hides ``u{i}``. Released as
    ordinary columns they would make every variable observed, so they are
    dropped. A hidden node that the intervention targets stays, because the
    experimenter sets and therefore observes it. Chain and regime-switching SCMs
    name their nodes ``X{i}``, so nothing is dropped from them.

    Args:
        x_obs: Observational arm, shape ``(T, N)`` with columns in SCM order.
        x_int: Interventional arm, same shape.
        intervention: The pair's :class:`~dotime.interventions.InterventionSpec`.
            Its targets index the columns.
        names: The SCM's node names in column order (``scm._topo``).

    Returns:
        ``(x_obs, x_int, intervention, latent)``: both arms restricted to the
        released columns, a copy of the intervention with its targets
        re-indexed to them, and the record ``{"mode": "drop", "columns",
        "hidden", "n_vars_full"}``, where ``columns`` and ``hidden`` list the
        released and the dropped node names in SCM order.

    Raises:
        KeyError: If an intervention target is not a column index.
    """
    import dataclasses

    targets = set(intervention.targets)
    keep = [i for i, name in enumerate(names) if not name.startswith("u") or i in targets]
    position = {old: new for new, old in enumerate(keep)}
    latent = {
        "mode": "drop",
        "columns": [names[i] for i in keep],
        "hidden": [name for i, name in enumerate(names) if i not in position],
        "n_vars_full": len(names),
    }
    remapped = dataclasses.replace(
        intervention, targets=[position[t] for t in intervention.targets]
    )
    return x_obs[:, keep], x_int[:, keep], remapped, latent
