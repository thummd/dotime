# Changelog

All notable changes to `dotime` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `dot-Identifiability-v1` 1.1.0: regenerated with shared-noise counterfactual
  pairing, aligned `x_obs`, hidden variables zeroed, correct `y_causal_effect`,
  resampled divergences (0 zeroed episodes) and a `diverged` flag. Same base seeds,
  new trajectories and targets. Zenodo version record 22673322 (concept DOI
  10.5281/zenodo.20846063), Hugging Face tag `v1.1.0`. 1.0.0 stays loadable with
  `version="1.0.0"`.
- Shared-noise (counterfactual) pairing for the discrete generator:
  `TemporalSCM.freeze_noise` / `clear_noise` and `pair_mode="counterfactual"` on
  `ExtendedDoTime` / `TSCMPrior`. One exogenous-noise realisation is drawn per
  episode and shared by both arms, so they agree exactly before the intervention
  onset and `Y_causal_effect` is a per-episode counterfactual effect. The default
  `pair_mode="interventional"` keeps the v1.0.0 independent-draw path
  byte-identical. `scripts/release_config_v1_1.yaml` builds
  `dot-Identifiability-v1` 1.1.0 with it (plus `stability_retries`).

### Changed
- Documented that 34.0% of `dot-Continuous-v1` queries are self-queries (query on
  the intervened variable; 33.7% of in-window queries), that the published
  Continuous rows include them, and that in-window ones equal the do-value.
  `Episode.is_self_query`, `--exclude-self-queries`, a per-episode query
  sidecar for the frozen files, and `self_query`/`query_in_window` metadata in
  new builds make the split explicit.
- Documented that the released Identifiability suite (1.0.0 and 1.1.0) queries
  every episode at offset 0, so under shared-noise targets lagged and mediated
  effects are exactly zero at the query for three structures (erratum item 8).
- Documentation and docstrings no longer describe the v1.0.0 discrete suites as
  carrying exact counterfactuals: their two arms are independent noise draws
  (interventional twins). Only the continuous suite, and discrete suites from
  1.1.0 on, share noise across arms.
- The `dev` extra pins `ruff==0.16.7` and `mypy==2.3.1` (was `ruff>=0.4`,
  `mypy>=1.8`) so local runs and CI resolve the same versions. Unpinned, ruff
  0.16 began formatting Python code blocks in Markdown and failed the CI format
  check, which also kept mypy from running.

### Fixed
- Docs build under `-W` (and Read the Docs `fail_on_warning`):
  `TemporalSCM.freeze_noise` used Google-style sections, which the numpydoc-only
  napoleon configuration parses as a malformed definition list. It is now
  numpydoc, as are `SuiteMetadata.for_version`, `Episode.is_self_query` and
  `InterventionSpec`, which rendered their sections as literal text.
- Python examples in `docs/custom_data.md`, `docs/quickstart.md` and
  `docs/troubleshoot.md` are formatted for ruff 0.16.
- mypy errors in `dotime.reference` (chronos, pfn, reference_table,
  stationarity, tabpfn): type-only fixes, verified to leave every computed value
  unchanged. `DEFAULT_CONFIG` is annotated `dict[str, Any]`. `PFNRef.predict`
  now names the problem when a checkpoint has neither a `quantile_head` nor a
  `bar_head`, instead of failing on `NoneType`.
- `unobserved_confounder` is documented as a null-effect control, not as a
  non-identifiable structure. It has only U→A and U→Y, with no A→Y edge at any
  lag, so its effect is identified and equals zero (`TSCMStructure` docstrings,
  `docs/benchmarks.md`, and the `s9ho_extra` note in
  `results/reference/structure_matched/README.md`, whose numbers are unchanged).
  A test now pins the exactly-zero effect under shared-noise pairing.
- The erratum and v1.1 PFN reference results (`results/reference/erratum/*.json`,
  `results/reference/v1_1/pfn_ident_dual.json`) record their checkpoints as
  `hf://thummd/do-over-time-pfn/<tag>/do_over_time_pfn_best.pt` instead of
  machine-local paths. Only the ten path strings changed. The local and hosted
  checkpoint files have identical SHA-256 hashes. A test now rejects any
  non-Hub checkpoint path under `results/reference/`.
- `build_release.py --stability-retries` help and the `dotime._build` comment
  now say that identifiability episodes are resampled too (when either arm is
  zeroed) and that continuous episodes are not. The comment no longer claims
  that `ExtendedDoTime` retries zeroed episodes internally: its retry only
  rejects NaN or `|x| >= 10`.
- `test_scale_beyond_default_bounds` asserted only shapes, although most
  N_max=60/K_max=8 pairs diverge to all-zero. It now seeds the global torch RNG
  per pair as the release build does, asserts finite output, and requires at
  least one non-diverged pair with N > 10.

### Fixed
- `dotime-generate --intervention-source` was listed in `--help` in every
  release but never applied, so each generated file used the prior's own
  intervention values whatever mode was chosen. The flag is no longer listed.
  `prior` is still accepted, and any other value now exits with an error that
  points to `ExtendedDoTime(intervention_source=...)` and
  `TemporalInterventionDataLoader(intervention_source=...)`. Generated files are
  unchanged.
- `ExtendedDoTime` raises a `ValueError` listing the valid modes when
  `intervention_source` is not recognized. An unknown string (a typo such as
  `"observed_normla"`) used to fall through `generate_sample` and behave
  exactly like `"prior"`.
- `ExtendedDoTime` no longer drops variables beyond `n_max` (default 41).
  `pad_to_max_nodes` used to truncate a wider SCM to its first `n_max` columns
  without warning. It now raises a `ValueError`, and a generic prior with
  `n_max_prior > n_max` is rejected at construction. Configurations that fit
  within `n_max` produce bit-identical output.
- `ExtendedDoTime.generate_sample`/`generate_batch`: the released (unmasked)
  observational tensor now receives the same canonical column permutation and
  hidden-variable zeroing as `X_int` — v1.0.0 `dot-Identifiability-v1` shipped
  `x_obs` in topological order (misaligned for 6/8 structures) and leaked
  hidden-confounder values.
- `Y_causal_effect`/`Y_obs` are computed from the unmasked observational
  trajectory; the causally-masked tensor is zero at every post-onset query, so
  v1 metadata stored the interventional level instead of the effect. RNG
  streams and all other fixed-seed outputs are bit-identical.
- API reference: Google-style `Args:`/`Returns:`/`Raises:` sections render as
  parameter, return and exception fields. `docs/conf.py` now enables napoleon's
  Google parser next to the NumPy one. Before, these sections fell through as raw
  definition lists, and the multi-line entry in `TemporalSCM.freeze_noise` failed
  the `-W` docs build.

### Added
- `dotime-eval-pfn` reports both `dir_acc_level` and `dir_acc_effect` (pooled and
  per structure) from a single prediction pass, whichever `--dir-target` is
  selected for the headline `dir_acc`.
- `evaluation.query_obs_levels` and a `--dir-target {level,effect}` /
  `--realignment` option on `dotime-eval-reference` and `dotime-eval-pfn`:
  score direction accuracy on the causal effect instead of the interventional
  level (the v1 paper protocol scored levels).
- `dotime._build` flags diverged (all-zero) episodes with a `diverged`
  metadata key (v1.0.0 shipped them unflagged: 28.7% of Generic-100k, 4.6% of
  Identifiability).
- Datasheet erratum section in `docs/benchmarks.md` documenting v1.0.0 field
  semantics, column alignment, and the realignment sidecar.

### Changed
- `dot-Continuous-v1` registry description states the actual query protocol
  (uniform over [onset, T-1]); the dead `query_offsets` key was removed from
  `release_config.yaml`.

### Fixed
- `InterventionSampler` (generic discrete prior) now raises a clear `ValueError`
  when `T < 2 * min_intervention_length` (default: `T < 20`) instead of failing
  inside `torch.randint` with an opaque range error. Only previously-crashing
  calls are affected; the RNG stream of every valid configuration is unchanged.

## [0.1.3] - 2026-08-15

### Fixed
- `ContinuousExtendedPrior.generate_sample`: the query-time sampler bounded the
  query index with `max(onset + 1, T - 1)`, which equals `T` when the intervention
  onset lands on the final observation, so a query index could overrun the
  trajectory (`IndexError`). Bound is now pinned to `T - 1`; draws for every
  onset `< T - 1` are unchanged, so fixed-seed outputs (and the released suites)
  are bit-identical.

### Changed
- Package metadata: `authors` lists the code author only; `CITATION.cff` now
  points its preferred citation at the DoTime paper (arXiv:2607.27263).
- Reference-result JSONs record model checkpoints as Hugging Face Hub paths
  (`hf://thummd/do-over-time-pfn/...`) instead of machine-local paths.

### Added
- `dotime-diagnose-stationarity` (`dotime.reference.stationarity`): measures the
  reduced-form companion spectral radius of sampled generic SCMs directly from
  their weight matrices (no simulation) and compares burn-in moments on
  non-diverged episodes. Backs the scope paragraph of the paper's convergence
  appendix; released output in `results/reference/stationarity_diagnostic.json`.
  `--activations identity` restricts every mechanism to a linear activation --
  the condition under which the companion radius is the exact stability
  criterion rather than an upper bound -- released as
  `results/reference/stationarity_diagnostic_identity.json`. On that condition
  divergence follows rho >= 1 in 96% of cases and never occurs for rho < 1.

### Fixed
- `docs/troubleshoot.md` claimed the SCM divergence rate was "< 1% on the
  released suites". The actual zeroed fraction is 28.7% on `dot-Generic-100k`
  and 4.6% on `dot-Identifiability-v1`, as disclosed in the paper and asserted
  by `tests/test_build_release.py`. The page now states the real rates and
  points at `--stability-retries` for a divergence-free rebuild.
- `stability_retries` is now honoured by the `regime` generator, not only
  `generic`. `scripts/build_release.py --stability-retries` documented both,
  but the regime branch had no retry loop and its episode specs did not carry
  the setting, so a hardened rebuild would have silently left
  `dot-RegimeSwitch-v1` unhardened. (No observable change at present: the
  regime generator's measured divergence rate is 0%.)

## [0.1.2] - 2026-07-20

### Added
- Reference evaluation harness now ships with the package as
  `dotime.reference`, exposed as four console scripts so the published
  reference tables reproduce from a plain `pip install` (they previously lived
  in `scripts/`, which is not part of the wheel):
  `dotime-eval-reference`, `dotime-eval-pfn`, `dotime-eval-tabpfn`,
  `dotime-eval-chronos`. The leaderboard submission path moved the same way:
  `scripts/eval_submission.py` is now `dotime-eval-submission`. The TabPFN and Chronos evaluators need the
  `baselines` extra; their imports are deferred so the package stays
  importable without it.
- `Results` now reports the uncertainty on direction accuracy: `dir_acc_se`
  (binomial standard error, exact because the suites score one query per
  episode) and `dir_n_valid` (queries with a scoreable sign) appear in
  `pooled`, in `per_structure`, in `to_dict()`, and in `summary()`.
- `scripts/build_release.py --stability-retries R` overrides the per-suite
  `stability_retries`, deterministically resampling numerically diverged
  episodes instead of releasing them as all-zero. The override is folded into
  the recorded `config_hash`, so a hardened rebuild is never mistaken for the
  frozen v1 suites.

### Changed
- Renamed the `rct_no_confounding` identification structure to `bi_variate`.
  The old string still loads: `TSCMStructure("rct_no_confounding")` resolves to
  `BI_VARIATE`, and episodes from the released v1 suites are relabeled at read
  time.
- Removed the review-policy sentence from the leaderboard submission docs.

### Added
- Docs: "Evaluating on Your Own Data" page covering the Do-Over-Time-PFN
  inference path (`Episode` construction, the `DoOverTimePFN` baseline, and
  `evaluate` over a custom suite).

## [0.1.1] - 2026-06-26

### Changed
- Removed forward-looking references to an unpublished paper from the package
  description, documentation, and dataset metadata.

## [0.1.0] - 2026-06-26

### Added
- Initial `src/` package layout consolidating the DoTime base prior
  (from the TSALM workshop code), the Do-Over-Time-PFN extended prior and
  dataloaders, and the continuous-time / fine-grid generation.
- Reimplemented the small `Do-PFN-prior` sampling/graph/mechanism surface as
  first-class, attributed modules (no git submodule required).
- First public release.
