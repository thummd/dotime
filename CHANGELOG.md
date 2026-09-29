# Changelog

All notable changes to `dotime` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `dotime-eval-reference` logs and asserts per-arm target statistics (nonzero
  fraction, mean and variance of the observational level, the interventional
  level and their difference) before scoring. It refuses to score a level arm
  that is less than 50% nonzero. The statistics and the suite version are
  recorded in the output JSON.
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
- `--version` and `--realignment` options on `dotime-eval-tabpfn` and
  `dotime-eval-chronos`. Both evaluators index `x_obs` by canonical column, so on
  the archived `dot-Identifiability-v1` 1.0.0 files they need `--version 1.0.0
  --realignment results/reference/dot-Identifiability-v1.0.0_realignment.jsonl`.
  The released `results/reference/server/tabpfn_intobs_ident.json` and
  `chronos_ident.json` predate these options and read unrealigned 1.0.0 columns.
  Every realigned episode must match its sidecar row (variable count, query
  target and `y_true`), so the 1.0.0 sidecar is refused on 1.1.0, whose `x_obs`
  is already canonical. The result JSON records `suite_version`, `realigned` and
  `realignment_sidecar`.

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
- `dotime._build` counts a generic or regime pair as diverged when **either** arm
  is all-zero, both for the `diverged` metadata flag and for the
  `stability_retries` resampling gate, as the identifiability branch already did.
  The arms are separate simulations, and pairs with only one arm zeroed were
  shipped with `diverged=False` and never resampled, even in hardened builds. In
  the v1.0.0 files they are 1,377 of 100,000 Generic episodes (682 with only the
  observational arm zeroed, whose nonzero target sits behind an all-zero history,
  and 695 with only the interventional arm zeroed, whose `y_true` is 0) and 7 of
  10,800 Identifiability episodes. The either-arm zeroed fractions are therefore
  30.1% and 4.7%. The 28.7% and 4.6% stated before count only pairs with both arms
  zeroed. RegimeSwitch and Continuous have no zeroed arm. With
  `stability_retries=0` the tensors are unchanged and only the flag value of
  half-diverged episodes changes. `dotime-generate` parquet output now carries the
  same flag.
- `evaluation.query_obs_levels` decoded every fractional `query_time` as
  `index / T`, the Identifiability encoding. `dot-Continuous-v1` stores
  `index / (T - 1)`, its normalized time on the regular grid, so on that suite
  the observational level behind effect-scored direction accuracy was read one
  step after the query in 9,251 of 9,999 episodes (every query at index 100 to
  198). `dotime-eval-chronos` ended its forecast horizon one step late in the
  same episodes. Each registered suite now declares its encoding
  (`SuiteMetadata.query_time_encoding`), the loader resolves the rows of the
  frozen files from it, `episode_from_sample` and `episode_from_pair` record
  `metadata["query_time_idx"]` in new builds (exact on irregular schedules
  too), and every lookup goes through `Episode.query_time_idx`. A declared
  encoding that does not land on whole rows raises. Identifiability,
  RegimeSwitch and Generic lookups are bit-identical, and no generator, RNG
  stream or frozen file changed. Level-scored rows are unaffected, except
  Chronos on Continuous. The `dir_acc_effect` fields of
  `results/reference/erratum/pfn_cont_noself.json` are recomputed:
  `PFN_int` 0.735 to 0.577 and `PFN_obs` 0.738 to 0.564.
- `BackDoorOLS` adjusted for every variable other than the treatment and the
  outcome. On `confounder_mediator` (columns A, X, M, Y) that set included the
  mediator M, a descendant of A. Adjusting for M blocks the causal path
  A -> M -> Y and violates the back-door criterion. The adjustment set is now
  derived from each structure's DAG: the observed variables other than A and Y
  that are not descendants of A. That set is X for `back_door`,
  `observed_confounder` and `confounder_mediator`. Predictions are bit-identical
  on every other structure and on all of `dot-Continuous-v1`, whose queries on
  the treatment or the confounder keep the previous adjustment. On
  `dot-Identifiability-v1` 1.1.0, BackDoorOLS effect-sign accuracy on
  `confounder_mediator` rises from 0.499 to 0.678 (n = 351). Pooled effect sign
  moves from 0.601 to 0.621, level sign from 0.693 to 0.695 and RMSE from 0.582
  to 0.575. The recomputed rows are in
  `results/reference/v1_1/ident_cpu_{level,effect}_backdoor_fix.json`. The
  released v1.1 rows, the v1.0.0 erratum rows and the v1.0.0 Table 3 row
  (`results/reference/ident.json`) were computed with the old adjustment set.
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
- `dotime-eval-tabpfn` chose its adjustment columns by position. The back-door
  branch adjusted for every column other than the treatment and the outcome.
  On `confounder_mediator` (columns A, X, M, Y) that set includes the mediator
  M, a descendant of A. The front-door branch used the first such column as the
  mediator, which on `front_door` (columns A, U, M, Y) is the hidden confounder
  U, all zeros in 1.1.0 `x_obs`. Both roles are now derived from each
  structure's DAG: the adjustment set is X for the back-door family and the
  mediator is M for `mediator` and `front_door`. TabPFN inputs are unchanged on
  `back_door`, `observed_confounder` and `mediator`. An episode whose column
  count or treatment column contradicts its structure raises `ValueError`, and
  a query of a variable other than the structure's outcome (only
  `dot-Continuous-v1` has them) takes the mean fallback. The released TabPFN
  results were not recomputed: `results/reference/server/tabpfn_ident.json` and
  `tabpfn_intobs_ident.json` (the int-vs-obs comparison) used the old columns on
  120 of their 480 episodes, every `front_door` and `confounder_mediator` one,
  and were computed on the 1.0.0 files, whose `x_obs` is in topological order.
  `tabpfn_generic.json` is unaffected, since no Generic episode reaches an
  adjustment branch.
- `TemporalInterventionDataLoader` no longer hangs when batch generation fails
  on the default prefetch path (`prefetch > 0`). The exception used to kill the
  background thread before it queued the end-of-stream sentinel, leaving the
  consumer blocked in `queue.get()` forever with only a thread traceback on
  stderr. It is now re-raised in the consumer as the same exception object,
  after the batches generated before it, exactly as with `prefetch=0`. A
  consumer that stops early (`break`, `close()`) also releases the producer
  thread instead of leaving it blocked on a full queue. Batches are
  bit-identical for fixed seeds.

### Added
- `dotime-eval-pfn` reports both `dir_acc_level` and `dir_acc_effect` (pooled and
  per structure) from a single prediction pass, whichever `--dir-target` is
  selected for the headline `dir_acc`.
- `evaluation.query_obs_levels` and a `--dir-target {level,effect}` /
  `--realignment` option on `dotime-eval-reference` and `dotime-eval-pfn`:
  score direction accuracy on the causal effect instead of the interventional
  level (the v1 paper protocol scored levels).
- `dotime._build` flags diverged (zeroed) episodes with a `diverged`
  metadata key (v1.0.0 shipped them unflagged: both arms zeroed in 28.7% of
  Generic-100k and 4.6% of Identifiability, either arm in 30.1% and 4.7%).
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
