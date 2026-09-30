# Changelog

All notable changes to `dotime` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `TimeOLS` baseline: NaiveOLS's regression `Y_t ~ 1 + A_t + Y_(t-1)` plus a
  linear trend and, when the Bayesian information criterion prefers it, one
  seasonal harmonic whose period is searched on the pre-onset rows. It predicts
  at the query's own time and reads no post-onset data. A seasonal or trend
  driver lies in the span of its time columns, so it blocks `A <- D -> Y` even
  when D is hidden, which no packaged estimator did. On the full
  dot-SeasonalTrend-v1 it scores 0.78 to 0.83 effect-sign accuracy on the hidden
  labels, against 0.60 to 0.72 for NaiveOLS, and matches NaiveOLS without a
  driver. `results/reference/seasonal_trend/time_ols.py` reproduces the released
  rows exactly before scoring it.
- `dotime-eval-tabpfn` and `dotime-eval-chronos` score direction accuracy on
  the level and on the causal effect for every arm and take `--dir-target
  {level,effect}` to choose which fills `dir_acc` (`dotime.reference._scoring`).
- Frozen fingerprints now cover ten versions: the five released ones plus
  dot-Identifiability-v1 1.2.0, dot-SeasonalTrend-v1, dot-Wide-v1, dot-Observed-v1
  and dot-ContinuousIrregular-v1 1.0.0 (`scripts/fingerprint_frozen_suites.py`,
  `tests/data/frozen_fingerprints.json`). Portable summaries compare floats with
  a relative tolerance of 1e-4, because the float32 trajectories of time-varying
  interventions differ between macOS arm64 and x86 Linux by about 1e-5.
- Reference rows (effect-scored, with NaiveOLS) for Identifiability 1.2.0
  (`results/reference/v1_2/`) and for the four new suites, with per-cell,
  per-schedule and per-label splits (`results/reference/{observed,
  continuous_irregular, seasonal_trend, wide}/`,
  `results/reference/audit_2026-09/scripts/per_group_baselines.py`), and the
  detection-power analysis on 1.2.0 (gate 3 passed,
  `results/reference/detection_power_2026-10/ident_v1_2.*`).
- Audit scripts for the 2026-10 builds: `identity_checks_2026_10.py` (rows a new
  version inherits, compared bit for bit with their release) and
  `suite_invariants_2026_10.py` (pre-onset agreement, do-value placement,
  effect identity and zeroed arms as pass rates), with their results in
  `results/reference/audit_2026-09/`.
- Release tooling: `scripts/release_2026_10.sh` (offline digest check, Hugging
  Face mirror, Zenodo new version for Identifiability, new Zenodo records for
  the new suites), `upload_zenodo.py --publish`, and `zenodo_update.py` hanging
  a LOCAL version off the newest published record.
- Ground-truth graph metadata, `dotime.graph_meta`. `LaggedGraph` holds an
  episode's lagged causal graph in released column order, with edges
  `(src, dst, lag)` where lag 0 is a same-step edge. It is extracted from a
  sampled `TemporalSCM` or `RegimeSwitchingTemporalSCM` (`from_scm`), from a
  named identifiability structure (`from_structure`) or from its continuous-time
  version (`from_continuous_structure`). An edge counts when the child's
  mechanism reads the parent, so the regime-switching SCMs of the v1.0.0 suites
  get an empty graph with `reads_parents=False`, and their sampled per-regime
  graphs are kept in `regime_edges`. `path_lag` reports whether the intervened
  columns reach a queried column, the smallest summed lag, the fewest hops and
  the direct lags. `load_graph_sidecar` reads the sidecars below.
- `record_graph: true` in a build config adds `metadata["graph"]` (the graph plus
  one `path` entry per query) to every episode that `dotime._build` generates.
  Extraction draws no random numbers, so tensors and all other fields are
  bit-identical with and without it. No frozen config sets it.
- Graph sidecars for the frozen `dot-Generic-100k` and `dot-RegimeSwitch-v1`
  1.0.0: `results/reference/dot-Generic-100k-v1.0.0_graph.jsonl.gz` and
  `results/reference/dot-RegimeSwitch-v1.0.0_graph.jsonl.gz`, one line per
  episode, made by regenerating every episode with `record_graph` and verified
  bit for bit against the released files. A per-lag breakdown of the CPU
  baselines on Generic-100k is in
  `results/reference/audit_2026-09/generic_lag_breakdown.json`.
- Suite schema 2: optional `obs_times` (observation time of each row) and
  `obs_mask` (`True` where `x_obs` is observed) parquet columns, loaded as
  `Episode.obs_times` and `Episode.obs_mask`. `write_suite` writes schema 2 only
  when an episode records times or a mask or holds a non-finite `x_obs` or
  `x_int` value, and then stores `isfinite(x_obs)` as the mask of an episode
  with non-finite `x_obs` values and no mask of its own. Every other suite is
  written as schema 1, byte for byte as before. `read_suite` accepts both
  (`SUPPORTED_SCHEMA_VERSIONS`), the Croissant descriptor of a schema-2 suite
  lists the two fields, `evaluation.realign_episode` permutes `obs_mask` with
  `x_obs`, and episode metadata may hold numpy scalars and arrays.
- Irregular observation grids for continuous suites, opt-in through the
  suite-config keys `schedules` and `record_obs_times`. Episode `idx` uses
  `schedules[idx % len(schedules)]`: `regular` (the unchanged
  `dot-Continuous-v1` call), `jittered` (gaps `dt * (1 + jitter * U)`) or
  `poisson` (`Exp(rate)` gaps truncated to `[0.001, max_gap]`). The irregular
  grids come from a generator of their own (`dotime._observation_grids`) and
  are replayed by the new `ContinuousExtendedPrior(schedule="fixed",
  fixed_times=...)`, so an irregular episode keeps the SCM and intervention of
  the regular episode with the same seed. A config whose Euler sub-steps
  (`num_substeps`) can exceed 1.0 is refused. `metadata["schedule"]` names the
  schedule, and `record_obs_times` stores the grid as `Episode.obs_times`
  (`episode_from_sample(..., record_obs_times=True)`). The query-time encoding
  `"time/span"` resolves rows from `obs_times`
  (`query_time_to_index(..., times=...)`). Suite-config keys reach the episode
  specs through `dotime._build._OPT_IN_SPEC_KEYS`. Released configs set none,
  so their specs and episodes are unchanged.
- `scripts/release_config_continuous_irregular.yaml` prepares
  `dot-ContinuousIrregular-v1` 1.0.0 (not built): the structures, `T`, episode
  count and suite seed of `dot-Continuous-v1`, one third each on the `regular`,
  `jittered` (`jitter` 0.5, 2 sub-steps) and `poisson` (`rate` 1, `max_gap` 4,
  4 sub-steps) schedules, and `record_obs_times`. Its regular third is the
  released `dot-Continuous-v1` rows with the same index, bit for bit. On 300
  episodes per structure and schedule, no irregular episode exceeds |x| = 10,
  against 3.7% of the regular third. The regular third keeps the single Euler
  step of `dot-Continuous-v1`, so its amplitudes and effects differ from the
  sub-stepped thirds (`results/reference/continuous_irregular/`).
- `dotime.observation`, an observation layer applied after simulation.
  `ObservationModel` pairs a measurement model with a missingness model. The
  measurement adds Gaussian noise at a signal-to-noise ratio relative to each
  column's latent pre-onset variance, censors at pre-onset quantiles and
  quantizes in units of that standard deviation. Missingness is `none`, `mcar`,
  `block` (one contiguous gap per column) or `mnar` (high values only).
  `apply_observation(episode, model, seed)` reads every `x_obs` row and the
  `x_int` rows before the onset with the same draws and mask, so shared-noise
  arms still agree before the onset, while `y_true` and `x_int` from the onset
  on stay latent. Query cells, all-zero (hidden or diverged) columns and one
  pre-onset cell per column are never missing. The draws come from
  `SeedSequence([salt, episode_seed])` and touch no global RNG, so all cells of
  one latent episode share them. Episodes gain `observation`, `obs_cell`,
  `y_obs_latent` and `obs_missing_frac` metadata. `impute_history` and
  `impute_episode` forward-fill missing cells without reading the future.
- Build configs accept an `observation:` section (`latent_per_structure` and
  named `measurement` and `missingness` levels). It expands a suite into the
  cells of the factorial design, cell-major, each observing the same latent
  episodes of the base suite with their base seeds, and adds `obs_cell` and
  `latent_row`. `dotime._build.make_episode` simulates and then observes when a
  spec carries an observation model. Without one it returns the simulation
  unchanged, so the frozen suites regenerate bit-identically.
  `scripts/release_config_observed_v1.yaml` prepares `dot-Observed-v1` 1.0.0
  (not built): measurement {none, snr10, snr3} × missingness {none, mcar10,
  block, mnar} on 100 latent episodes of each `dot-Identifiability-v1` v1.2
  structure, 10,800 rows.
- `per_variable_normalize(..., obs_mask=...)` computes the statistics over
  observed cells only and zeroes missing ones, the hook for mask-aware models.
  The default `None` returns the same output as before.
- `TSCMStructure.BOW_GRAPH` (`"bow_graph"`), a structure whose effect is not
  identifiable: hidden U -> A, U -> Y and a causal edge A -> Y. It is
  `unobserved_confounder` plus A -> Y, so it takes over the role the paper gave
  `unobserved_confounder` (non-identifiable by design), which has no A -> Y
  edge and is a null-effect control. U is zeroed in both released arms, and the
  DAG-derived back-door adjustment set is empty. The generator builds it; no
  frozen suite contains it yet.
- `NaiveOLS`, the unadjusted regression baseline: `BackDoorOLS` with an empty
  adjustment set, that is OLS of `Y_t` on `[1, A_t, Y_{t-1}]` over the
  pre-intervention window, evaluated at the do-value and averaged over the
  history. It applies to every structure and to the generic prior, and
  `dotime-eval-reference` runs it right after `BackDoorOLS`. Its slope carries
  the omitted-confounder bias that `BackDoorOLS` removes on the back-door
  structures and that no observed adjustment set removes on `bow_graph`.
  `scripts/release_config_v1_2.yaml` (prepared, not yet minted) appends
  `bow_graph` at tier 3 as episodes 10800 to 12149. That makes nine structures
  and 12,150 episodes, with every 1.1.0 episode index and seed unchanged. In the
  first 100 `bow_graph` episodes of that config, 61% of the effects reach
  `|effect| >= 0.1` and none is zeroed.
- `evaluate(..., dir_target="effect")` and `--dir-target {level,effect}` on
  `dotime-benchmark` and `dotime-eval-submission`: score direction accuracy on
  the sign of the causal effect `y - y_obs` at the query instead of the sign of
  the interventional level. The level metrics are unchanged. `Results` records
  `dir_target`, and `summary()` states it. The default stays `"level"`, so
  existing calls return the same numbers. It is set in one place,
  `dotime.evaluation.DEFAULT_DIR_TARGET`, which `evaluate()`, `Results`,
  `dotime.qa.target_qa` and the `--dir-target` option of all six evaluators
  (through `add_dir_target_argument`) read, and a test rejects a default
  hard-coded anywhere else. Effect scoring refuses the archived
  `dot-Identifiability-v1` 1.0.0 files, whose `x_obs` is misaligned.
- `dotime-eval-reference` logs and asserts per-arm target statistics (nonzero
  fraction, mean and variance of the observational level, the interventional
  level and their difference) before scoring. It refuses to score a level arm
  that is less than 50% nonzero. The statistics and the suite version are
  recorded in the output JSON.
- Opt-in stability hardening for the generic prior,
  `DoTime(config={"hardening": ...})` (`dotime.hardening`).
  - **`unit_norm_rows`** normalizes each variable's incoming weights to unit L2
    norm.
  - **`spectral_rho`** caps the reduced-form companion spectral radius by scaling
    the lagged weights.
  - **`bounded_square`** swaps the unbounded `x^2` activation for `tanh(x)^2`.

  `dotime.hardening.RECOMMENDED_HARDENING` enables all three. On 300 episodes at
  `N_max=60, K_max=8`, divergence (either arm) falls from 67.3% to 0%, and the
  surviving graphs keep the prior's size distribution: median 27 variables,
  against 7 unhardened. At the default `N_max=10, K_max=3` it falls from 29.6% to
  0.2%. See `results/reference/hardening/`. The option is off by default, and it
  rescales sampled weights without drawing random numbers. The released suites
  are therefore unchanged, and a hardened prior draws the same graphs,
  interventions and noise as an unhardened one with the same seed.
- Release config for the wide generic suite `dot-Wide-v1` 1.0.0,
  `scripts/release_config_wide.yaml`: 10 000 episodes with 12 to 40 variables
  and up to 8 lags, `RECOMMENDED_HARDENING`, shared-noise counterfactual pairs,
  and the prior's hidden variables removed from the released columns. It needs
  new opt-in plumbing, and every default stays byte-identical.
  - **`DoTime(config={"N_min": n})`** sets the smallest number of variables of
    diverse and regime-switching SCMs. The default of 3 draws the same numbers
    as before, and chain SCMs keep 3 to 7 variables.
  - **`DoTime.generate_pair(pair_mode="counterfactual")`** draws one noise
    realisation from a generator derived from the seed and shares it across
    both arms. It draws the same SCM and intervention as the default
    `"interventional"` mode and needs `regime_switching_prob=0`.
  - **Generic suite keys** `prior_config`, `chain_prob`,
    `regime_switching_prob`, `pair_mode`, `latent: drop` and `tier_n_edges`
    configure the prior of a `generic` suite in a release config.
    `latent: drop` removes each hidden `u*` variable that is not intervened on
    before the query is chosen, and records the released and dropped names in
    `metadata["latent"]`. The suite manifest records the keys a suite sets.
  - A suite may set its own `seed`, and `build_manifest.json` records each
    suite's seed.

  - **`query_row: window_end`** queries the intervention window's last step
    instead of the last step of the trajectory (`episode_from_pair(query_row=)`,
    recorded as `metadata["query_row"]`). In these hardened graphs the window
    closes a median 36 steps before the end and the effect decays, so at the
    last step only 13.5% of the queries carry an effect of at least 0.1.

  On the first 200 episodes none diverged, 14.3% of the variables were hidden
  and dropped, and the median counterfactual effect at the query is 0.48, with
  88.5% of the queries at or above 0.1 (see `results/reference/wide/`). A full
  build projects to about 0.4 h on 15 workers.
- `dot-Identifiability-v1` 1.1.0: regenerated with shared-noise counterfactual
  pairing, aligned `x_obs`, hidden variables zeroed, correct `y_causal_effect`,
  resampled divergences (0 zeroed episodes) and a `diverged` flag. Same base seeds,
  new trajectories and targets. Zenodo version record 22673322 (concept DOI
  10.5281/zenodo.20846063), Hugging Face tag `v1.1.0`. 1.0.0 stays loadable with
  `version="1.0.0"`.
- Opt-in seasonal and trend confounding drivers for the named structures
  (`dotime.drivers`). A driven label `"<base>+<kind>_<visibility>"`, such as
  `"back_door+seasonal_hidden"`, adds an exogenous driver D, a sinusoid with
  period `U[12, 48)` or a ramp between -1 and 1 over burn-in plus `T`. D is a
  root with instantaneous edges D -> A and D -> Y and enters both equations
  additively with loadings `±U[0.5, 1)`. It is drawn once per episode and
  shared by both arms, so drivers need `pair_mode="counterfactual"`, and
  `generate_batch` refuses them. An observed D is released at column `N-2`
  with `known_future: true`. A hidden D is zeroed in both arms like U. Each
  episode records `metadata["driver"]`, and `released_driver_series` rebuilds D
  from it. Driver draws come from their own seed stream, so strength 0
  reproduces the base structure bit for bit, and plain labels and the released
  suites are unchanged. `BackDoorOLS` adjusts for `{X, D}` or `{D}` on
  observed-driver labels and predicts the pre-onset mean on hidden ones.
  `scripts/release_config_seasonal_trend.yaml` defines `dot-SeasonalTrend-v1`
  1.0.0, 10 labels of 1,000 episodes with suite seed 20262001. On its first
  100 episodes per label (`results/reference/seasonal_trend/`), effect-sign
  accuracy on the seasonal labels is 0.60 to 0.67 without adjusting for an
  observed D and 0.91 and 0.97 with it. Estimators without D score 0.61 and
  0.67 on the hidden seasonal labels, against 0.92 to 0.96 on the plain
  structures.
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
- `divergence_fallback` on `ExtendedDoTime` and `TemporalInterventionDataLoader`
  selects how `generate_batch` replaces a diverged sample of a named
  `tscm_structure`. `"sequential"`, the default for interventional pairs, keeps
  the per-sample `generate_sample` replacement that released checkpoints were
  trained with, bit-identical. That replacement comes from a different simulator
  (noise added after the activation instead of inside it), ignores `hardening`,
  draws its own intervention time, and 3 to 7 % of replacements are all-zero
  diverged episodes. `"batched"` redraws diverged samples with the batch's own
  `BatchedTSCMSimulator` at the batch's intervention time, and also counts a
  non-finite value or |x| > 10 anywhere in the recorded window as diverged. The
  simulator's 50-step check misses the tail after the last multiple of 50 and
  passed 0.07 to 0.9 % of such samples. For front_door under the OSC hardening
  they inflate `var(Y_true)` by a third. In a fixed-seed scan with batches of 16,
  3.5 to 6.2 % of samples diverge without hardening (45 to 65 % of batches need a
  replacement), and 0.6 to 1.8 % under the OSC hardening. Slots that stay valid,
  and every later batch, are bit-identical to `"sequential"`.
  `BatchedTSCMSimulator.generate_pairs` gains `int_time`, `shared_noise` and
  `check_recorded_window`. All are off by default, which keeps its output
  unchanged.
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
- Frozen-suite fingerprint test. `tests/test_frozen_fingerprints.py` regenerates
  131 stratified rows of the five released suite versions (`dot-Identifiability-v1`
  1.0.0 and 1.1.0, `dot-RegimeSwitch-v1`, `dot-Continuous-v1` and
  `dot-Generic-100k` 1.0.0) pinned in `tests/data/frozen_fingerprints.json`. They
  cover every structure, regime density, SCM class and intervention kind, the
  zeroed rows of the audit scan and retried 1.1.0 seeds. The test checks that the
  specs rebuilt from the release configs equal the pinned ones and that rows keep
  the 12 v1 columns. A portable summary of each row (shapes, intervention, query,
  RNG-only metadata) must match on every platform. SHA-256 hashes of all 12
  columns must match in the recorded environment, and `DOTIME_FINGERPRINT_EXACT=1`
  or `0` overrides that. `scripts/fingerprint_frozen_suites.py` records them from
  the md5-verified release files (`dotime._fingerprint`) and stops on any
  difference from a release that is not documented: the 1.0.0 `x_obs` erratum
  (verified by realignment), the old `rct_no_confounding` label and metadata keys
  added since the releases. Its `--full` mode regenerated all 9,999
  `dot-Continuous-v1` 1.0.0 rows and all 10,800 `dot-Identifiability-v1` 1.1.0
  rows. Every data column matches the release bit for bit
  (`results/reference/audit_2026-09/frozen_regeneration.json`).
- `dotime.qa`, per-arm target QA. `target_qa(episodes, ...)` and
  `batch_target_qa(batches, ...)` log and assert the nonzero fraction, mean,
  variance, non-finite count and largest magnitude of the observational level, the
  interventional level and the effect at each query, pooled and per structure, and
  return a JSON-able `QAReport`. A failure raises `TargetQAError`, a
  `RuntimeError`. Level arms must be finite, varied and at least 50% nonzero. An
  effect that is scored or trained on must be at least 5% nonzero on the queries
  that can carry one. `is_null_effect` reads the shortest A -> Y lag off each named
  structure's temporal DAG, so `observed_confounder` and `unobserved_confounder` are
  always exempt and `mediator` is exempt at offset 0. Groups under 10 queries are
  reported but not asserted. All five released suite versions pass the defaults
  (`results/reference/audit_2026-09/frozen_target_qa.json`, Identifiability 1.0.0
  with the sidecar's observational level).

### Changed
- `evaluate(..., impute=True, nonfinite="raise")` imputes episodes with missing
  cells through `impute_episode` unless the model sets `mask_aware = True`. A
  finite episode passes through unchanged, so every latent suite scores exactly
  as before. A non-finite prediction now raises an error that names the
  baseline and the episode, where it used to turn every pooled metric into NaN
  silently. `nonfinite="exclude"` leaves such predictions out of the level
  metrics, scores them as wrong directions and reports `n_nonfinite`.
  `dotime-eval-reference` imputes the same way, and the PFN, TabPFN and Chronos
  evaluators refuse episodes with missing cells.
- `query_obs_levels` returns `metadata["y_obs_latent"]` when an episode records
  it, so effect-scored direction accuracy on an observed suite subtracts the
  latent observational level, not a noisy measurement. The
  `direction_accuracy` docstring states its tie rules: a prediction with sign 0,
  or a non-finite one, counts as wrong, and targets with `|target| < eps` are
  excluded.
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
- `dot-Continuous-v1` registry description states the actual query protocol
  (uniform over [onset, T-1]); the dead `query_offsets` key was removed from
  `release_config.yaml`.
- Target QA now runs before anything is written, scored or trained, with an
  opt-out. `scripts/build_release.py` checks every arm of each suite, per structure
  and with the effect, after generation and records the report in `manifest.json`
  and `build_manifest.json`. With `--target-qa enforce` (the default) a failing
  suite is not written and the build exits with status 1, `warn` writes it anyway
  and `off` skips the check. `dotime-benchmark`, `dotime-eval-submission`,
  `dotime-eval-pfn`, `dotime-eval-tabpfn` and `dotime-eval-chronos` check the
  evaluated episodes and store `target_qa` in their output JSON
  (`--target-qa {enforce,warn}`). The `target_qa` of `dotime-eval-reference` is now
  a wrapper over `dotime.qa` with the same signature, keys and `RuntimeError`, and
  it also asserts each structure. `TemporalInterventionDataLoader(target_qa=True)`
  checks the raw targets of the first 64 queries of each structure, the effect too
  when `target_key="Y_causal_effect"`, logs through `logging` and raises in the
  consumer, also with prefetch. The checks draw no random numbers, so shards,
  batches and random streams are unchanged.

### Fixed
- `scripts/build_release.py --suite X` seeded a suite by its position in the
  list being built, which for a single suite is always the first. Every suite
  built alone therefore got the first suite's seed (base seed + 1000) instead of
  the seed a full build gives it. The seed now comes from the suite's position
  in the config, or from the suite's own `seed` key when it sets one. Full
  builds keep their seeds, and `build_manifest.json` records each suite's seed.
- `dotime-eval-reference` and `dotime-eval-pfn` always loaded the registry's
  current version, `dot-Identifiability-v1` 1.1.0 since 2026-09-09, and applied
  `--realignment` rows by episode id alone. Passing the 1.0.0 sidecar as
  documented therefore re-permuted the already canonical 1.1.0 `x_obs` of 8,100
  of the 10,800 episodes and scored the effect sign against 1.0.0
  `y_obs_corrected`, without an error. Episodes without a row were scored
  unrealigned, and `dotime-eval-reference` logged the sidecar's size as the
  realigned count. Both evaluators now take `--version` (default `latest`) and
  realign through the same checks as `dotime-eval-tabpfn` and
  `dotime-eval-chronos`: every evaluated episode needs a row whose variable
  count, query target and `y_true` match it, which no 1.1.0 episode passes. The
  result JSON records `suite_version`, `realigned` and `realignment_sidecar`.
  The `results/reference/v1_1/` CPU rows were computed without a sidecar and are
  unaffected.
- Docs build under `-W` (and Read the Docs `fail_on_warning`) and the API
  reference: `docs/conf.py` parsed NumPy sections only, so Google-style
  `Args:`/`Returns:`/`Raises:` sections fell through as raw definition lists,
  and the multi-line entry in `TemporalSCM.freeze_noise` failed the `-W` build.
  napoleon's Google parser is now enabled next to the NumPy one. `freeze_noise`,
  `SuiteMetadata.for_version`, `Episode.is_self_query` and `InterventionSpec`,
  which rendered their sections as literal text, are numpydoc like the rest of
  their modules.
- Python examples in `docs/custom_data.md`, `docs/quickstart.md` and
  `docs/troubleshoot.md` are formatted for ruff 0.16.
- mypy errors in `dotime.reference` (chronos, pfn, reference_table,
  stationarity, tabpfn): type-only fixes, verified to leave every computed value
  unchanged. `DEFAULT_CONFIG` is annotated `dict[str, Any]`. `PFNRef.predict`
  now names the problem when a checkpoint has neither a `quantile_head` nor a
  `bar_head`, instead of failing on `NoneType`.
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
- `ExtendedDoTime.generate_batch` honours `pair_mode="counterfactual"` for named
  structures. The batched simulator now reuses the observational noise draw for
  the interventional arm, so every slot agrees with its twin before the onset. It
  used to return independent-noise twins, apart from replaced diverged samples,
  which were shared-noise counterfactuals from the per-sample simulator.
  Counterfactual batches redraw diverged samples with the batched simulator by
  default, and `divergence_fallback="sequential"` is rejected for them. For a
  fixed seed, observational arms and intervention values are unchanged.
  `TemporalInterventionDataLoader` accepts `pair_mode`. Interventional batches
  are bit-identical.
- Regime-switching SCMs never read their parents. `RegimeSwitchingSCMBuilder`
  renames each regime's nodes to `X0..X{N-1}` but kept the mechanism weights
  under the old names (`x3`, `u1`, `y`), so `TemporalMechanism.forward` matched
  no parent and every variable was its own noise term. There was no lagged or
  cross-variable dependence, and an intervention never reached another variable.
  This covers all of `dot-RegimeSwitch-v1` and the 15,041 regime-switching
  episodes (15.0%) of `dot-Generic-100k` (datasheet erratum). The fix is opt-in,
  `DoTime(config={"regime_canonical_weights": True})`. It re-keys the sampled
  weights (`TemporalMechanism.rename_nodes`) and zeroes any arm whose values
  exceed 500 with a `RuntimeWarning`, as `TemporalSCM` does
  (`RegimeSwitchingTemporalSCM(divergence_threshold=...)`). It draws the same
  random numbers, and the default path stays byte-identical, so the v1.0.0 suites
  still regenerate. With the flag, 64.5% of RegimeSwitch episodes diverge at the
  default prior and 94% at `N_max=60, K_max=8`, so a rebuilt suite needs
  `stability_retries` or hardening (`results/reference/regime_weights/`).
- `ExtendedDoTime.generate_sample`/`generate_batch`: the released (unmasked)
  observational tensor now receives the same canonical column permutation and
  hidden-variable zeroing as `X_int` — v1.0.0 `dot-Identifiability-v1` shipped
  `x_obs` in topological order (misaligned for 6/8 structures) and leaked
  hidden-confounder values.
- `Y_causal_effect`/`Y_obs` are computed from the unmasked observational
  trajectory; the causally-masked tensor is zero at every post-onset query, so
  v1 metadata stored the interventional level instead of the effect. RNG
  streams and all other fixed-seed outputs are bit-identical.
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
- `ExtendedDoTime.generate_batch` raises `NotImplementedError` when a named
  `tscm_structure` is combined with `intervention_source` `observed_discrete`,
  `observed_normal`, `observed_uniform` or the alias `observed`. Batches for a
  named structure come from the vectorized simulator, which implements only
  `prior` and `positivity_aware`, so these modes silently produced prior-sampled
  intervention values. Each diverged sample, regenerated by `generate_sample`, did
  apply the mode, so one batch could mix both semantics.
  `TemporalInterventionDataLoader` runs the same check at construction, because an
  error raised in its prefetch thread would hang the training loop.
  `generate_sample` and the generic prior (`tscm_structure=None`) still support
  every mode. Outputs for `prior` and `positivity_aware` are bit-identical.
- `ExtendedDoTime.generate_batch` with a named `tscm_structure` no longer raises
  `KeyError: 'X_obs_full'` when the first sample of a batch diverges (5 to 10 % of
  batches of 16 in a fixed-seed scan over all eight structures). The replacement
  sample from `generate_sample` carried an `X_obs_full` key that the vectorized
  samples lack, and the collate step takes its keys from the first sample. With
  the default prefetching, `TemporalInterventionDataLoader` turned the error into
  a hang. Batches that did not crash, and all RNG streams, are bit-identical.
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
