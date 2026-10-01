"""Foundational smoke tests: imports, prior invariants, registry, round-trip, eval."""

from __future__ import annotations

import ast
import re

import pytest
import torch

import dotime as ctp

# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


def test_version_and_eager_core():
    # Exact value is pinned by test_version_strings_agree, not duplicated here.
    assert re.fullmatch(r"\d+\.\d+\.\d+", ctp.__version__)
    for name in [
        "DoTime",
        "TemporalSCM",
        "TemporalDAG",
        "TemporalGraphBuilder",
        "TemporalMechanism",
        "TemporalSCMBuilder",
        "InterventionSpec",
        "InterventionType",
        "InterventionSampler",
        "RegimeSwitchingTemporalSCM",
        "RegimeSwitchingSCMBuilder",
        "DEFAULT_CONFIG",
    ]:
        assert hasattr(ctp, name), name


def test_lazy_submodules_resolve():
    for name in ["extended", "continuous", "data", "benchmarks", "baselines", "evaluation"]:
        assert getattr(ctp, name) is not None


# --------------------------------------------------------------------------- #
# Prior invariants
# --------------------------------------------------------------------------- #


def test_generate_pair_shapes_and_finiteness():
    prior = ctp.DoTime(seed=0)
    x_obs, x_int, intervention, _scm = prior.generate_pair(T=64)
    assert x_obs.shape == x_int.shape
    assert x_obs.shape[0] == 64
    assert x_obs.ndim == 2
    # Non-diverged trajectories must be finite (diverged ones are zeroed, also finite).
    assert torch.isfinite(x_obs).all()
    assert torch.isfinite(x_int).all()
    assert isinstance(intervention, ctp.InterventionSpec)


def test_intervention_targets_within_range():
    prior = ctp.DoTime(seed=1)
    x_obs, _x_int, intervention, _scm = prior.generate_pair(T=32)
    n = x_obs.shape[-1]
    for t in intervention.targets:
        assert 0 <= t < n


# --------------------------------------------------------------------------- #
# Baseline registry
# --------------------------------------------------------------------------- #


def test_baseline_registry_lists_expected():
    names = set(ctp.baselines.available())
    assert {"Zero", "Mean", "VAR-OLS", "Oracle"} <= names


def test_dependency_free_baselines_instantiate():
    for name in ["Zero", "Mean", "VAR-OLS", "Oracle"]:
        assert ctp.baselines.get(name) is not None


def test_unknown_baseline_raises():
    with pytest.raises(KeyError):
        ctp.baselines.get("does-not-exist")


# --------------------------------------------------------------------------- #
# Suite round-trip + evaluation
# --------------------------------------------------------------------------- #


def _seed_local_suite(cache_dir, name, n=8):
    """Write a tiny suite into the loader's cache so load_benchmark reads it
    locally (the hosted suites are GBs; we don't download them in unit tests)."""
    pytest.importorskip("pyarrow")
    from dotime import _release_io
    from dotime.benchmarks import _SUITE_REGISTRY, episode_from_pair

    meta = _SUITE_REGISTRY[name]
    prior = ctp.DoTime(seed=0)
    structs = meta.structures or (None,)
    eps = [
        episode_from_pair(
            *prior.generate_pair(T=60)[:3], structure=structs[i % len(structs)], scm_id=i
        )
        for i in range(n)
    ]
    _release_io.write_suite(
        meta, eps, cache_dir / f"{name}-{meta.version}", package_version="test", seed=0
    )


def test_suite_roundtrip_shapes(tmp_path):
    # Seed a local cache copy and load it (no network / no GB download).
    _seed_local_suite(tmp_path, "dot-Identifiability-v1")
    suite = ctp.benchmarks.load_benchmark("dot-Identifiability-v1", cache_dir=tmp_path)
    assert len(suite) > 0
    seen_structures = set()
    for ep in suite:
        assert ep.x_obs.shape == ep.x_int.shape
        assert ep.y_true.numel() == ep.query_target.numel() == ep.query_time.numel()
        if ep.structure is not None:
            seen_structures.add(ep.structure)
    assert len(seen_structures) >= 2


def test_released_episodes_store_full_unmasked_xobs():
    # Identifiability episodes must store the FULL observational trajectory
    # (causal masking is a model-input transform, not part of the released data).
    from dotime.benchmarks import episode_from_sample
    from dotime.extended import ExtendedDoTime

    prior = ExtendedDoTime(tscm_structure="back_door", n_max=41, seed=0)
    ep = episode_from_sample(prior.generate_sample(T=80), structure="back_door")
    onset = min(ep.intervention.times)
    # At least some post-onset observational values are non-zero (i.e. not masked).
    assert bool((ep.x_obs[onset:] != 0).any())


def test_oracle_is_exact_on_loaded_suite(tmp_path):
    _seed_local_suite(tmp_path, "dot-Generic-100k")
    suite = ctp.benchmarks.load_benchmark("dot-Generic-100k", cache_dir=tmp_path)
    results = ctp.evaluation.evaluate(ctp.baselines.get("Oracle"), suite)
    assert results.pooled["rmse"] == pytest.approx(0.0, abs=1e-5)
    assert results.pooled["mae"] == pytest.approx(0.0, abs=1e-5)
    # summary() and to_dict() must work (the CLI calls both).
    assert "Oracle" in results.summary()
    assert results.to_dict()["baseline"] == "Oracle"


def test_results_report_direction_accuracy_uncertainty(tmp_path):
    """Direction accuracy ships with its (exact binomial) standard error.

    The suites score one query per episode, so there is no clustering to
    correct for and sqrt(p(1-p)/n_valid) is the right SE. Leaderboard
    submissions carry it, so a point estimate is never reported bare.
    """
    _seed_local_suite(tmp_path, "dot-Generic-100k")
    suite = ctp.benchmarks.load_benchmark("dot-Generic-100k", cache_dir=tmp_path)
    results = ctp.evaluation.evaluate(ctp.baselines.get("Mean"), suite)

    for group in [results.pooled, *results.per_structure.values()]:
        n_valid, acc, se = group["dir_n_valid"], group["dir_acc"], group["dir_acc_se"]
        assert 0 <= n_valid <= results.n_queries
        if n_valid > 0:
            assert se == pytest.approx((acc * (1 - acc) / n_valid) ** 0.5)

    assert "dir_acc_se" in results.summary()
    assert "dir_acc_se" in results.to_dict()["pooled"]


def _seed_local_suite_version(cache_dir, name, version, n=8):
    """Like :func:`_seed_local_suite`, for a pinned earlier release."""
    pytest.importorskip("pyarrow")
    from dotime import _release_io
    from dotime.benchmarks import _SUITE_REGISTRY, episode_from_pair

    meta = _SUITE_REGISTRY[name].for_version(version)
    prior = ctp.DoTime(seed=0)
    eps = [episode_from_pair(*prior.generate_pair(T=60)[:3], scm_id=i) for i in range(n)]
    _release_io.write_suite(
        meta, eps, cache_dir / f"{name}-{meta.version}", package_version="test", seed=0
    )


def test_effect_scored_direction_accuracy(tmp_path):
    """dir_target="effect" scores sign(pred - y_obs) against sign(y - y_obs).

    Subtracting the same observational level from both sides changes only what
    the sign test sees: every level metric must be identical to the level run.
    """
    from dotime.evaluation import direction_accuracy, query_obs_levels

    _seed_local_suite(tmp_path, "dot-Generic-100k", n=16)
    suite = ctp.benchmarks.load_benchmark("dot-Generic-100k", cache_dir=tmp_path)
    model = ctp.baselines.get("Mean")
    level = ctp.evaluation.evaluate(model, suite, dir_target="level")
    effect = ctp.evaluation.evaluate(model, suite, dir_target="effect")

    for key in ("rmse", "mae", "nmse", "r2"):
        assert effect.pooled[key] == pytest.approx(level.pooled[key])
    assert (level.dir_target, effect.dir_target) == ("level", "effect")
    assert effect.to_dict()["dir_target"] == "effect"
    assert "sign of the effect" in effect.summary()

    preds = torch.cat([torch.as_tensor(model.predict(ep)).float().reshape(-1) for ep in suite])
    tgts = torch.cat([ep.y_true.float().reshape(-1) for ep in suite])
    obs = torch.cat([query_obs_levels(ep).reshape(-1) for ep in suite])
    expected = direction_accuracy(preds - obs, tgts - obs)
    assert effect.pooled["dir_n_valid"] == expected["n_valid"]
    if expected["n_valid"] > 0:
        assert effect.pooled["dir_acc"] == pytest.approx(expected["accuracy"])

    oracle = ctp.evaluation.evaluate(ctp.baselines.get("Oracle"), suite, dir_target="effect")
    if oracle.pooled["dir_n_valid"] > 0:
        assert oracle.pooled["dir_acc"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="dir_target"):
        ctp.evaluation.evaluate(model, suite, dir_target="sign")


def test_effect_scoring_refuses_misaligned_identifiability_v1_0(tmp_path):
    # The archived 1.0.0 x_obs is in topological order, so reading y_obs from it
    # would score the wrong variable on 6 of 8 structures.
    _seed_local_suite_version(tmp_path, "dot-Identifiability-v1", "1.0.0")
    suite = ctp.benchmarks.load_benchmark(
        "dot-Identifiability-v1", version="1.0.0", cache_dir=tmp_path
    )
    ctp.evaluation.evaluate(ctp.baselines.get("Mean"), suite, dir_target="level")
    with pytest.raises(ValueError, match="realignment"):
        ctp.evaluation.evaluate(ctp.baselines.get("Mean"), suite, dir_target="effect")


def test_dir_target_default_has_a_single_source():
    """Every direction-accuracy default reads evaluation.DEFAULT_DIR_TARGET.

    Flipping the default must be a one-line change, so no function signature,
    dataclass field or --dir-target option may hard-code "level" or "effect".
    """
    import inspect
    import pathlib

    from dotime import cli, evaluation

    default = evaluation.DEFAULT_DIR_TARGET
    assert default in evaluation.DIR_TARGETS
    assert inspect.signature(evaluation.evaluate).parameters["dir_target"].default == default
    assert evaluation.Results("s", "b", 0, 0, {}).dir_target == default
    assert cli._build_benchmark_parser().parse_args([]).dir_target == default

    src = pathlib.Path(evaluation.__file__).parent
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for literal in _dir_target_default_literals(ast.parse(text)):
            raise AssertionError(
                f"{path.name}:{literal.lineno} hard-codes dir_target={literal.value!r}"
            )
        if path.name != "evaluation.py":
            assert '"--dir-target"' not in text, f"{path.name} defines --dir-target itself"


def _dir_target_default_literals(tree):
    """String constants used as the default of a ``dir_target`` parameter or field.

    Keyword arguments at call sites (``f(dir_target="effect")``) are choices,
    not defaults, and are not flagged.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            pairs = list(
                zip(positional[len(positional) - len(args.defaults) :], args.defaults, strict=True)
            )
            pairs += [
                (a, d)
                for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True)
                if d is not None
            ]
            for arg, default in pairs:
                if arg.arg == "dir_target" and isinstance(default, ast.Constant):
                    yield default
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "dir_target"
            and isinstance(node.value, ast.Constant)
        ):
            yield node.value


def test_reference_harness_imports_without_optional_extras():
    """`dotime.reference` must stay importable without tabpfn/chronos installed.

    The console scripts are declared unconditionally in the wheel metadata, so
    an eager third-party import here would break `pip install dotime` users.
    """
    import importlib

    for mod in ("dotime.reference", "dotime.reference.reference_table"):
        importlib.import_module(mod)
    # The dependency-gated ones import too; only *calling* them needs the extra.
    for mod in ("dotime.reference.tabpfn", "dotime.reference.chronos"):
        importlib.import_module(mod)


def test_reference_harness_target_qa_rejects_an_all_zero_arm():
    """`dotime-eval-reference` checks per-arm target stats before scoring.

    The v1 observational training arm was all zeros and passed every seed
    check, so the harness logs nonzero fraction, mean and variance per arm and
    refuses to score a level arm that is mostly zero.
    """
    import dataclasses

    from dotime.reference.reference_table import target_qa

    def toy(seed):
        """One-query episode whose counterfactual effect is exactly +1.

        Args:
            seed: Seed of the random trajectory.

        Returns:
            An episode querying column 2 at the onset, step 20 of 30.
        """
        x = torch.randn(30, 3, generator=torch.Generator().manual_seed(seed))
        spec = ctp.InterventionSpec([0], [20], ctp.InterventionType.HARD, 1.0)
        return ctp.benchmarks.Episode(
            x_obs=x,
            x_int=x + 1.0,
            intervention=spec,
            y_true=x[20:21, 2] + 1.0,
            query_target=torch.tensor([2]),
            query_time=torch.tensor([20 / 30]),
        )

    eps = [toy(s) for s in range(6)]
    stats = target_qa(eps, dir_target="effect")
    assert stats["effect"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["y_int_level"]["nonzero_frac"] == 1.0
    zeroed = [dataclasses.replace(ep, y_true=torch.zeros_like(ep.y_true)) for ep in eps]
    with pytest.raises(RuntimeError, match="y_int_level"):
        target_qa(zeroed)


def test_scale_beyond_default_bounds():
    """N_max/K_max are config bounds, not architectural limits.

    Backs the Limitations claim that the generator scales past the frozen
    suites' DEFAULT_CONFIG (N<=10, K<=3) via a config override. Shapes alone do
    not back it: at N_max=60/K_max=8 most pairs diverge and are zeroed (both
    arms in 65% of 1,000 episodes at T=200, in
    results/reference/audit_2026-09/scaling_lag.json), so the test also requires
    finite output and at least one non-diverged pair wider than the default cap.
    """
    import warnings

    from dotime import DoTime
    from dotime.utils import DEFAULT_CONFIG

    cfg = {**DEFAULT_CONFIG, "N_max": 60, "K_max": 8}
    sizes, wide_ok = [], []
    # A fixed window: with the global RNG seeded, seeds 0-7 hold no non-diverged
    # pair with N > 10, while 40-47 hold two (N=14 plain, N=41 regime-switching)
    # next to four zeroed wide pairs.
    for seed in range(40, 48):
        # The Beta edge-probability draw and the noise use the global torch RNG,
        # so seed it per pair as dotime._build.make_episode does. Otherwise the
        # outcome depends on whichever tests ran earlier in the session.
        torch.manual_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_obs, x_int, _, _ = DoTime(config=cfg, seed=seed).generate_pair(T=40)
        assert x_obs.shape[0] == 40
        assert x_obs.shape == x_int.shape
        # Divergence is handled by zeroing, so no pair may carry NaN or inf.
        assert torch.isfinite(x_obs).all()
        assert torch.isfinite(x_int).all()
        sizes.append(x_obs.shape[1])
        if x_obs.shape[1] > 10 and x_obs.abs().max() > 0 and x_int.abs().max() > 0:
            wide_ok.append(seed)
    # the override actually widens the sampled graph beyond the default cap ...
    assert max(sizes) > 10
    # ... and yields usable wide pairs, not only zeroed (diverged) ones
    assert wide_ok, f"every pair with N > 10 diverged (sizes {sizes})"


def test_version_strings_agree():
    """`__version__`, pyproject, and CITATION.cff must be bumped together.

    They are three hand-edited copies of one fact; a mismatch silently stamps
    the wrong `package_version` into every leaderboard submission JSON.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    citation = re.search(r"^version: (\S+)", (root / "CITATION.cff").read_text(), re.M)
    assert pyproject is not None
    assert citation is not None
    assert ctp.__version__ == pyproject.group(1) == citation.group(1)


def test_continuous_query_index_never_overruns_trajectory():
    """Regression: the continuous prior's query sampler bounded the query index
    with ``max(onset + 1, T - 1)``, which equals ``T`` when the intervention
    onset lands on the final observation — an index one past the trajectory.
    Query indices must always lie in ``[onset, T - 1]``.
    """
    from dotime.continuous import ContinuousExtendedPrior

    prior = ContinuousExtendedPrior(tscm_structure="back_door", seed=0)
    ctx = prior._sample_scm_context()
    for t_len in (2, 3, 10, 50):
        for onset in (t_len - 1, t_len - 2, 0):
            _, t_idx = prior._sample_queries(
                T=t_len,
                n_queries=64,
                query_mode="single",
                int_onset_idx=onset,
                intervention_target_canon=0,
                ctx=ctx,
            )
            assert int(t_idx.max()) <= t_len - 1
            assert int(t_idx.min()) >= min(onset, t_len - 1)

    # End-to-end reproducer (raised IndexError before the fix): a 3-point
    # exponential schedule where the onset can hit the last observation.
    gen = ContinuousExtendedPrior(
        tscm_structure="confounder_mediator", schedule="exponential", seed=0, t_range=(2, 30)
    )
    for _ in range(20):
        out = gen.generate_sample(T=3)
        assert out["X_obs"].shape[0] == 3


def test_intervention_sampler_rejects_too_short_series():
    """Regression: T below 2*min_intervention_length used to die inside
    torch.randint with an opaque message; it must fail early and clearly."""
    from dotime.interventions import InterventionSampler

    with pytest.raises(ValueError, match="need T >= 20"):
        InterventionSampler(N=3, T=19)
    InterventionSampler(N=3, T=20).sample()  # boundary is allowed
    with pytest.raises(ValueError, match="need T >= 4"):
        InterventionSampler(N=3, T=3, min_intervention_length=2)


def test_identifiability_release_tensors_are_canonically_aligned():
    """Regression (v1 erratum): the released unmasked X_obs_full must share the
    canonical column order of X_int/query_target, hide hidden variables, and
    yield Y_causal_effect == Y_true - X_obs_full[query]."""
    import torch as _torch

    from dotime.extended import ExtendedDoTime

    for struct in ("front_door", "back_door", "unobserved_confounder"):
        _torch.manual_seed(5)
        gen = ExtendedDoTime(tscm_structure=struct, n_max=41, seed=5)
        s = gen.generate_sample(T=80)
        n = int(s["num_vars"])
        onset = int(s["int_onset_idx"])
        # pre-onset the masked and unmasked tensors must agree -> same column order
        assert _torch.allclose(s["X_obs_full"][:onset, :n], s["X_obs"][:onset, :n])
        # hidden columns (variable_mask == 0 among real vars) are zero in the release tensor
        hidden = s["variable_mask"][:n] == 0
        if bool(hidden.any()):
            assert float(s["X_obs_full"][:, :n][:, hidden].abs().max()) == 0.0
        # effect identity against the unmasked obs at the query
        qt = int(s["query_target"])
        qti = min(round(float(s["query_time"]) * 80), 79)
        expected = float(s["Y_true"]) - float(s["X_obs_full"][qti, qt])
        assert float(s["Y_causal_effect"]) == pytest.approx(expected, abs=1e-5)


def test_query_obs_levels_matches_manual_lookup():
    from dotime.benchmarks import Episode
    from dotime.evaluation import query_obs_levels
    from dotime.interventions import InterventionSpec, InterventionType

    x_obs = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    ep = Episode(
        x_obs=x_obs,
        x_int=x_obs.clone(),
        intervention=InterventionSpec(
            targets=[0], times=[5], intervention_type=InterventionType.HARD, values=1.0
        ),
        y_true=torch.tensor([1.0]),
        query_target=torch.tensor([2]),
        query_time=torch.tensor([0.7]),  # normalized -> index 7
    )
    assert float(query_obs_levels(ep)[0]) == float(x_obs[7, 2])
    ep.query_time = torch.tensor([8.0])  # absolute index
    assert float(query_obs_levels(ep)[0]) == float(x_obs[8, 2])


def test_load_benchmark_pins_prior_versions(tmp_path, monkeypatch):
    """A registry entry that advances to a new version must keep earlier
    releases loadable by pinning ``version=`` (published numbers stay
    reproducible); unknown versions fail clearly."""
    from dataclasses import replace

    import dotime.benchmarks as B

    base = B._SUITE_REGISTRY["dot-Identifiability-v1"]
    advanced = replace(
        base, version="9.9.9", prior_versions=((base.version, base.zenodo_record_id),)
    )
    monkeypatch.setitem(B._SUITE_REGISTRY, "dot-Identifiability-v1", advanced)
    assert advanced.for_version("latest").version == "9.9.9"
    pinned = advanced.for_version(base.version)
    assert pinned.version == base.version
    assert pinned.zenodo_record_id == base.zenodo_record_id
    with pytest.raises(ValueError, match="versions"):
        advanced.for_version("0.0.1")
    # End to end through the loader against a seeded local cache for the pinned version.
    _seed_local_suite(tmp_path, "dot-Identifiability-v1")  # writes <name>-<registered version>
    suite = B.load_benchmark("dot-Identifiability-v1", version=base.version, cache_dir=tmp_path)
    assert suite.meta.version == base.version


def test_episode_is_self_query_flag():
    from dotime.benchmarks import Episode
    from dotime.interventions import InterventionSpec, InterventionType

    x = torch.zeros(10, 3)
    iv = InterventionSpec(
        targets=[1], times=[4], intervention_type=InterventionType.HARD, values=0.5
    )
    on = Episode(
        x_obs=x,
        x_int=x,
        intervention=iv,
        y_true=torch.tensor([0.5]),
        query_target=torch.tensor([1]),
        query_time=torch.tensor([0.6]),
    )
    off = Episode(
        x_obs=x,
        x_int=x,
        intervention=iv,
        y_true=torch.tensor([0.5]),
        query_target=torch.tensor([2]),
        query_time=torch.tensor([0.6]),
    )
    assert on.is_self_query
    assert not off.is_self_query


def test_generate_cli_rejects_unapplied_intervention_source(tmp_path):
    """Regression: ``dotime-generate --intervention-source`` was listed in --help
    but never applied, so every file used the prior's own intervention values.
    The flag is now hidden, ``prior`` (what the command does) is accepted, and
    any other value fails before anything is sampled or written."""
    from dotime.cli import _build_generate_parser, generate_main

    out = tmp_path / "gen.pt"
    base = ["-n", "1", "-T", "30", "-o", str(out), "--intervention-source"]
    for source in ("observed_normal", "positivity_aware", "bogus"):
        with pytest.raises(SystemExit, match="ExtendedDoTime"):
            generate_main([*base, source])
    assert not out.exists()
    assert "--intervention-source" not in _build_generate_parser().format_help()
    torch.manual_seed(0)
    assert generate_main([*base, "prior"]) == 0
    assert out.exists()


def test_extended_prior_rejects_unknown_intervention_source():
    """Regression: a mode string that generate_sample does not dispatch on (e.g.
    a typo) fell through its if/elif chain and behaved exactly like "prior"."""
    from dotime.extended import INTERVENTION_SOURCES, ExtendedDoTime

    with pytest.raises(ValueError, match="observed_normla") as exc:
        ExtendedDoTime(tscm_structure="back_door", intervention_source="observed_normla")
    for mode in INTERVENTION_SOURCES:
        assert repr(mode) in str(exc.value)
    with pytest.raises(ValueError, match="intervention_source"):
        ExtendedDoTime(intervention_source="observed_normla")  # generic prior too
    for mode in (*INTERVENTION_SOURCES, "observed"):
        ExtendedDoTime(tscm_structure="back_door", intervention_source=mode)


def test_extended_prior_refuses_to_truncate_wide_scms():
    """Regression: pad_to_max_nodes truncated an SCM wider than n_max to its
    first n_max columns, so the extra variables vanished silently (and a query
    on one of them later died with an opaque IndexError)."""
    from dotime.extended import ExtendedDoTime, pad_to_max_nodes

    x = torch.randn(5, 4)
    padded = pad_to_max_nodes(x, 6)
    assert padded.shape == (5, 6)
    assert torch.equal(padded[:, :4], x)
    assert float(padded[:, 4:].abs().max()) == 0.0
    assert torch.equal(pad_to_max_nodes(x, 4), x)
    with pytest.raises(ValueError, match="4 variables but the padded width is 3"):
        pad_to_max_nodes(x, 3)
    # The generic prior's width bound is known up front, so it fails at construction.
    with pytest.raises(ValueError, match="n_max_prior=30 exceeds n_max=12"):
        ExtendedDoTime(seed=0, n_max=12, n_max_prior=30)
    ExtendedDoTime(seed=0, n_max=12, n_max_prior=12)
    # Any other overflow surfaces at generation time (back_door has 3 variables).
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="padded width is 2"):
        ExtendedDoTime(tscm_structure="back_door", n_max=2, seed=0).generate_sample(T=40)


def test_reference_jsons_record_hub_checkpoints():
    """Every checkpoint recorded under results/reference/ is a Hugging Face Hub path.

    Regression: the erratum and v1.1 PFN results were committed with
    machine-local checkpoint paths after 0.1.3 had moved the older files to
    ``hf://thummd/do-over-time-pfn/<tag>/...``.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "results" / "reference"
    if not root.is_dir():
        pytest.skip("results/reference/ ships with the repository, not the sdist")
    bad: list[str] = []

    def walk(obj, where):
        if isinstance(obj, dict):
            for key, val in obj.items():
                # A bare file name says which file of every run was scored, not
                # where it lives, so it cannot be a machine-local path. The
                # pre-registered s13 results record it that way and list each
                # run's file and digest in s13_scoring_provenance.json.
                is_location = "/" in str(val) or "\\" in str(val)
                if (
                    key == "checkpoint"
                    and is_location
                    and not str(val).startswith("hf://thummd/do-over-time-pfn/")
                ):
                    bad.append(f"{where}: {val}")
                walk(val, where)
        elif isinstance(obj, list):
            for val in obj:
                walk(val, where)

    files = sorted(root.rglob("*.json"))
    assert files
    for path in files:
        walk(json.loads(path.read_text()), path.relative_to(root))
    assert not bad, bad
