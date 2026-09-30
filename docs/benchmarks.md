# Frozen Benchmark Suites

DoTime ships four versioned, immutable suites for reproducible evaluation. Each has a Zenodo DOI and Croissant metadata.

## Suites

- **`dot-Identifiability-v1`** — ~10.8k trajectories across **eight** named structures: `back_door`, `observed_confounder`, `confounder_mediator` (back-door family); `front_door`, `mediator` (front-door family); `instrumental_variable` (IV); `bi_variate` (trivially identified); `unobserved_confounder` (null-effect control: a hidden U drives A and Y, and there is no A→Y edge at any lag, so the effect is identified and equals zero). Targets are exact interventional outcomes. In v1.0.0 the two arms are independent noise draws from the same SCM (interventional twins, so `y_int - y_obs` is not a per-episode counterfactual effect). From v1.1.0 one pre-drawn noise stream is shared across arms and the pair is a true counterfactual.
- **`dot-RegimeSwitch-v1`** — regime-switching trajectories with controllable break density. In v1.0.0 no regime mechanism reads its parents, so every variable is independent noise. See the erratum below.
- **`dot-Continuous-v1`** — continuous-time intervention windows, multiple query offsets.
- **`dot-Generic-100k`** — 100 000 trajectories from the full diverse prior. Training-scale.

**`dot-Identifiability-v1` 1.1.0 (2026-09)** regenerates the suite with one exogenous-noise realisation per episode shared across both arms (`pair_mode="counterfactual"`): the arms agree exactly before the intervention onset, `y_true - y_obs` is a per-episode counterfactual effect, the released `x_obs` is canonically aligned with hidden variables zeroed, `y_causal_effect` is correct, and diverged episodes are resampled (0 zeroed episodes, verified on all 10 800 episodes). It is a new artifact with new trajectories and targets. Pin `version="1.0.0"` to load the frozen original.

The generator also builds a ninth structure that no frozen suite contains yet,
`bow_graph`: a hidden U drives A and Y, and A drives Y. It is
`unobserved_confounder` plus the causal edge A→Y, and it is the structure that
is **not identifiable**. Nothing observed blocks the back-door path A←U→Y and
there is no mediator, so two SCMs can agree on every observational
distribution and still differ in the effect of `do(A)`.

```python
ExtendedDoTime(tscm_structure="bow_graph", pair_mode="counterfactual")
```

**`dot-Identifiability-v1` 1.2.0 (prepared, not yet minted)** is configured by
`scripts/release_config_v1_2.yaml`. It keeps the 1.1.0 generator, seeds and
episode indices and makes two changes. `mediator` is queried one step after the
onset, because its A→M edge is lagged and its effect at the onset is exactly
zero. `bow_graph` is appended at tier 3 as episodes 10 800 to 12 149, which
brings the suite to nine structures and 12 150 episodes. Every other episode is
identical to 1.1.0, and so are the `mediator` trajectories. Only the `mediator`
query and target move.

## Loader

```python
from dotime.benchmarks import load_benchmark

suite = load_benchmark("dot-Identifiability-v1", version="1.0.0")
```

On first access the suite is fetched into `~/.cache/dotime/` — from the Hugging Face
mirror ([`thummd/dot-*`](https://huggingface.co/thummd)) by default, falling back to the
Zenodo archive of record (concept DOIs `10.5281/zenodo.20846063`, `.20846073`, `.20845980`,
`.20845982`, each resolving to the latest archived version) — and md5-verified
against the manifest. Pass `force_download=True` to
re-fetch. Override the cache with `$DOTIME_CACHE` or `cache_dir=`.

## Wide graphs with latent variables (dot-Wide-v1)

The released suites have at most 10 variables and 3 lags. `dot-Wide-v1` 1.0.0
draws 10 000 episodes from the generic prior with 12 to 40 variables and 1 to 8
lags. PFN checkpoints pad to 41 variables, which caps the size at 40. Unhardened,
about two thirds of such episodes diverge, so every SCM gets
`RECOMMENDED_HARDENING` (see the troubleshooting guide on large graphs). The two
arms share one exogenous-noise realisation, so they agree exactly before the
onset and `y_true - y_obs` is a per-episode counterfactual effect. The prior
names the variables it hides `u{i}`. `dot-Generic-100k` releases them as
ordinary columns, while `dot-Wide-v1` removes them from both arms unless the
intervention targets them, so they are unobserved and never queried. Its
release config is `scripts/release_config_wide.yaml`:

```bash
python scripts/build_release.py --config scripts/release_config_wide.yaml
```

A `generic` suite accepts these opt-in keys. A suite that sets none of them
builds exactly as before.

| Key | Effect |
|---|---|
| `prior_config` | Passed to `DoTime(config=...)`. `dot-Wide-v1` sets `N_min: 12`, `N_max: 40`, `K_max: 8` and the recommended `hardening`. |
| `pair_mode` | `counterfactual` shares the noise across arms (`DoTime.generate_pair(pair_mode=...)`). It needs `regime_switching_prob: 0`, since regime-switching SCMs draw their noise step by step. |
| `chain_prob`, `regime_switching_prob` | Shares of chain and regime-switching SCMs. `dot-Wide-v1` sets both to 0. Chain SCMs have 3 to 7 variables whatever `N_min` is. |
| `latent` | `drop` removes the hidden variables before the query is chosen. |
| `tier_n_edges` | The tier is 1 plus the number of these edges below the simulated variable count, counted before latents are dropped. `[20, 30]` gives tier 1 for 12 to 20 variables, 2 for 21 to 30 and 3 for 31 to 40. |

Each episode's metadata records `tier`, `diverged`, `pair_mode` and `latent`,
which is `{"mode": "drop", "columns": [...], "hidden": [...], "n_vars_full": N}`.
`columns` names the released columns in order and `hidden` the dropped
variables.

On the first 200 episodes (`results/reference/wide/`), none diverged, 12 to 40
variables were simulated and 8 to 40 released, and 14.3% of all variables were
hidden and dropped. The query sits at the last step, as in `dot-Generic-100k`,
and the intervention window closes before it in 95.5% of those episodes, a
median 36 steps earlier. The contractive hardened dynamics let the effect decay
in that time, so the median |`y_true - y_obs`| at the query is 3.3e-6 and 13.5%
of the queries have an effect of at least 0.1. Score the level targets, or read
the effect from `x_int - x_obs` inside the window, where 88.5% of the episodes
have an effect of at least 0.1 at the window's last step.

## v1.0.0 field semantics and known issues (erratum)

The archived v1.0.0 files are frozen; the issues below are **documented, not
silently patched**. They are fixed in the generator for any v1.1+ build.

| Suite | Issue | Consequence | Workaround on v1 data |
|---|---|---|---|
| Identifiability | `metadata.y_causal_effect` stores the **interventional level** (`y_true`), not the effect | any effect-based analysis using the field is wrong | use the released realignment sidecar's `y_effect_corrected` |
| Identifiability | released `x_obs` columns are in **topological** order while `x_int`/`query_target`/`intervention_target` are canonical (identity only for `bi_variate`) | baselines reading `x_obs[:, query_target]` touch the wrong variable on 6/8 structures | realignment sidecar maps each episode's canonical→topo permutation |
| Identifiability | hidden variables (`front_door`, `instrumental_variable`, `unobserved_confounder`) are **not zeroed** in the released `x_obs` | the "unobserved" confounder is readable from the data | zero the sidecar's `hidden_canonical` columns after realigning |
| Generic-100k / Identifiability | diverged arms are stored as all-zero trajectories with **no flag**. Both arms are zeroed in 28.7% / 4.6% of episodes (28,734 / 497); **only one arm** is zeroed in a further 1,377 / 7 (Generic: 682 observational, 695 interventional), so 30.1% / 4.7% have a zeroed arm | both-arm and interventional-arm zeroing store `y_true == 0`: included in RMSE, excluded from direction accuracy by the near-zero filter. An observational-arm zeroing leaves a nonzero target behind an all-zero history: 657 of the 682 Generic cases are scored for direction, and every history-based baseline predicts exactly 0 there | drop an episode when **either** arm is all-zero (`x_obs.abs().max() == 0` or `x_int.abs().max() == 0`); builds from v1.1 on set a `diverged` metadata flag on the same either-arm rule |
| Continuous | documentation said query offsets `{1,2,3,5,10}`; actual query times are **uniform over [onset, T-1]** (observed offsets 0–138) | protocol description only — data and scoring are self-consistent | none needed |
| all | reported direction accuracy scores the sign of the **interventional level**, not the causal effect | see the paper erratum; `--dir-target effect` re-scores | `dotime-eval-reference --dir-target effect [--version 1.0.0 --realignment <sidecar>]` |
| Continuous (harness, fixed) | `query_time` is normalized time, `index / (T - 1)`, while Identifiability stores `index / T`; `evaluation.query_obs_levels` and `dotime-eval-chronos` decoded every fraction as `index / T` | the effect-scored `y_obs` (and the Chronos forecast horizon) sat one step past the query on **92.5%** of episodes, every query at index 100 to 198; level-scored rows of the other baselines are unaffected | each suite declares its encoding (`SuiteMetadata.query_time_encoding`) and consumers read `Episode.query_time_idx` |
| Continuous (1.0.0) | the query target is drawn uniformly over the three observable variables, treatment included, so **34.0%** of episodes are **self-queries** (query on the intervened variable); **16.6%** of all episodes are in-window self-queries whose target equals the do-value (the suite is hard-intervention only) and 17.4% are post-window self-queries (relaxation after release); 33.7% of in-window queries are self-queries | the published Continuous rows **include** them, which favours any model that receives the intervention value as input on the in-window subset | `Episode.is_self_query`; `--exclude-self-queries` on the eval harness; per-episode `self_query` / `in_window` / `window_end_idx` in `results/reference/dot-Continuous-v1.0.0_query_sidecar.jsonl`; 1.1.0+ builds tag them in metadata |
| Identifiability (1.0.0 and 1.1.0) | every episode is queried at **offset 0** (the onset step); the paper's per-structure offset protocol was used by the model's loaders, not by the release build | with true counterfactuals (1.1.0) the effect at the query is exactly 0 for `mediator`, `observed_confounder`, `unobserved_confounder` and ~1/3 of `front_door`/`confounder_mediator` episodes, so the effect metric is defined only for structures with an instantaneous A→Y edge | filter `|effect| >= 0.1` (done by `direction_accuracy`); a 1.2.0 protocol with path-length offsets is prepared |
| Identifiability / RegimeSwitch / Generic (1.0.0) | the two arms are **independent noise draws** from the same SCM (interventional twins), so `y_int - y_obs` is not a per-episode counterfactual effect | effect-based analyses on these files score a noisy twin difference | Identifiability 1.1.0 shares the noise across arms; the continuous suite always did |
| RegimeSwitch, Generic-100k (1.0.0) | every regime-switching SCM is **per-variable independent noise**. The builder renamed each regime's nodes to `X0..X{N-1}` but kept the mechanism weights under the old names, so no mechanism reads its parents. This is all 9,999 RegimeSwitch episodes and the 15,041 regime-switching episodes (15.0%) of Generic-100k. In the released RegimeSwitch `x_obs` the median absolute lag-1 autocorrelation is 0.048, 1 of 64,830 variables exceeds 0.3, and the median cross-correlation is 0.048 | these episodes have no temporal or causal structure. An intervention changes only the treated variable, so every other variable, the query target included, has the same distribution in both arms and a true effect of 0 | nothing to recover from v1 data. Drop the regime share of Generic-100k with the recipe below. New builds can opt into the fix |

`x_int`, `y_true`, `query_target`, and `intervention_*` are correct and mutually
consistent in all four v1 suites, apart from the zeroed (diverged) episodes in the
table above; `dot-Continuous-v1` and `dot-RegimeSwitch-v1` carry none of the
column/field issues above and have no zeroed arm.
The RegimeSwitch issue lies in the simulated dynamics, not in how the fields
are stored.

**Regime-switching episodes (1.0.0).** A Generic-100k episode comes from a
regime-switching SCM exactly when the first draw of a generator seeded with its
per-episode seed lies in `[0.15, 0.30)`:

```python
import torch


def in_regime_share(scm_id: int) -> bool:
    """True for the 15,041 dot-Generic-100k 1.0.0 episodes that are independent noise."""
    seed = (20264719 * 1_000_003 + scm_id) & 0x7FFFFFFF  # per-episode seed of the build
    g = torch.Generator()
    g.manual_seed(seed)
    return 0.15 <= torch.rand(1, generator=g).item() < 0.30
```

The unreleased opt-in `DoTime(config={"regime_canonical_weights": True})` re-keys
the weights and zeroes any arm whose values exceed 500, as for the other SCMs. It
draws the same random numbers as the default, and the default stays byte-identical,
so the v1.0.0 files still regenerate. With live weights most regime SCMs of the
current prior are unstable. At the default prior 64.5% of RegimeSwitch episodes and
62.7% of the Generic regime share diverge, and at `N_max=60, K_max=8` 94% and 97%
do (`results/reference/regime_weights/`). No suite has been rebuilt with the flag.

## Ground-truth graphs

The frozen v1 files store trajectories, the intervention and the query, but not
the graph that generated them. `dotime.graph_meta` records that graph in the
episode's own column order, so that results can be split by lag.

**Lag convention.** An edge `[src, dst, lag]` means that column `src` at step
`t - lag` enters the mechanism of column `dst` at step `t`. Lag 0 is a same-step
edge, and `[i, i, lag]` is an autoregressive term. An edge counts only when the
child's mechanism reads the parent, that is, holds a weight under the parent's
name. `tests/test_graph_meta.py` checks the convention by simulation: a parent
perturbed at step `t` moves its lag-2 child at `t + 2` and not before.

**New builds.** Set `record_graph: true` on a suite of a build config, for
example a copy of `scripts/release_config.yaml`, and every episode gains
`metadata["graph"]`:

| Key | Meaning |
|---|---|
| `n`, `columns` | Number of released columns and the SCM node name of each |
| `latent` | SCM nodes a build does not release. They are graph nodes `n`, `n + 1`, and so on, so paths through them still count |
| `edges` | Effective edges `[src, dst, lag]`, sorted. For a regime-switching SCM, the union over regimes |
| `hidden` | Released columns that hold no data, such as the zeroed confounder `U` |
| `k_sampled` | The maximum lag `K` that the prior drew |
| `k_eff` | The largest lag in `edges`, self-edges included, or 0 |
| `reads_parents` | Whether every sampled parent is read by its child's mechanism |
| `regime_edges` | Regime-switching SCMs only: the edges each regime sampled, read or not. Otherwise `null` |
| `time` | `"discrete"`, or `"continuous"` for the continuous prior, whose edges all have lag 1 in observation steps |
| `path` | One entry per query, from the intervention targets to the query column: `target`, `reachable`, `min_lag` (smallest summed lag), `min_hops` (fewest edges) and `direct_lags` |

Recording draws no random numbers, so tensors, targets and all other metadata
are bit-identical with and without the flag. No frozen config sets it.

```python
from dotime.graph_meta import LaggedGraph, path_lag

graph = LaggedGraph.from_dict(episode.metadata["graph"])
path = path_lag(graph, episode.intervention.targets, int(episode.query_target[0]))
print(graph.k_sampled, path.min_lag)
```

**Frozen v1.0.0 suites.** `results/reference/dot-Generic-100k-v1.0.0_graph.jsonl.gz`
and `results/reference/dot-RegimeSwitch-v1.0.0_graph.jsonl.gz` hold one JSON line
per released episode, keyed by `idx`, the episode's `scm_id`. They were made by
regenerating every episode from the release seeds with `record_graph` and
comparing `x_obs`, `x_int`, `y_true`, `query_target`, `query_time`, the
intervention and `n_vars` bit for bit with the released files. All 100,000 and
9,999 episodes matched
(`results/reference/audit_2026-09/graph_sidecar_verification.json`). Besides
`graph` and the path fields, each line records the SCM `family` (`diverse`,
`chain` or `regime`), `treatment`, `query`, the intervention `onset` and
`window_end`, the `steps_after_window` until the query, the `intervention_type`,
which arms are zeroed, and `verified`.

```python
from dotime.graph_meta import load_graph_sidecar

sidecar = load_graph_sidecar("results/reference/dot-Generic-100k-v1.0.0_graph.jsonl.gz")
record = sidecar[episode.scm_id]
print(record["family"], record["graph"].k_sampled, record["min_lag"])
```

The named-structure suites need no sidecar, because the structure fixes the
graph. `LaggedGraph.from_structure(episode.structure)` for
`dot-Identifiability-v1` and `LaggedGraph.from_continuous_structure(episode.structure)`
for `dot-Continuous-v1` give it in the released column order, with the
treatment first and the outcome last. The archived Identifiability 1.0.0 `x_obs`
is in topological order, so realign it first (see the erratum above).

**Regime caveat.** No regime-switching SCM of the v1.0.0 suites reads its
parents (see the erratum above). Their graphs therefore have `reads_parents`
false and no `edges`, and every query is unreachable. This covers all 9,999
RegimeSwitch episodes and the 15,041 regime-family episodes of
Generic-100k. `regime_edges` still lists the graphs the regimes sampled. A build
with `regime_canonical_weights` draws the same graphs and makes them effective.

**Unreachable queries.** A Generic-100k or RegimeSwitch query is the variable
whose two arms differ most at the last step, among those not intervened on. The
arms of the 1.0.0 files are independent noise draws, so that variable need not
be a descendant of the treatment. In Generic-100k 38,038 of the 100,000 queries
(38.0%) have no path from any intervened column: all 15,041 regime-family
episodes, 6,890 of the 14,986 chains (46.0%) and 16,107 of the 69,973 diverse
SCMs (23.0%). For them the true effect at the query is zero and the difference
between the arms is noise. The sidecars flag them with `reachable` false.

**Per-lag scores.** `results/reference/audit_2026-09/generic_lag_breakdown.json`
splits the published Generic-100k CPU baseline rows by SCM family, sampled `K`,
min lag and steps from the end of the intervention window to the query, with
and without episodes whose arms are zeroed. Without zeroed arms, the Mean
baseline's level-scored direction accuracy falls with the min lag: 0.806 at lag
0, 0.699 at lag 1, 0.638 at lag 4 or more and 0.558 on unreachable queries.
Effect-scored accuracy does not fall. It stays between 0.83 and 0.85 in every min-lag bin, and a
constant prediction of 0 scores 0.810 on the unreachable queries and 0.828 on
the regime-family episodes, where no effect exists. The arms are independent
draws, so the sign of `y - y_obs` tends to oppose the deviation of `y_obs` from
its mean. For two independent zero-mean symmetric draws it does so with
probability 3/4, and choosing the query where the arms differ most raises this
further. On these files the effect-scored sign test therefore rewards
regression to the mean, not knowledge of the causal path.
## Seasonal and trend confounders (dot-SeasonalTrend-v1)

A driven structure label `"<base>+<kind>_<visibility>"` adds one exogenous
driver D to a named structure (`dotime.drivers`):

```python
ExtendedDoTime(tscm_structure="back_door+seasonal_hidden", pair_mode="counterfactual")
```

- `kind` is `seasonal`, `sin(2πt/P + φ)` with period `P ~ U[12, 48)` and phase
  `φ ~ U[0, 2π)`, or `trend`, a ramp from -1 to 1 or from 1 to -1 over the
  burn-in and the `T` released steps.
- D is a root of the DAG with instantaneous edges D→A and D→Y. It enters both
  structural equations additively after the activation, with loadings
  `±U[0.5, 1)`, so D confounds A and Y. The sign of the product of the two
  loadings is the episode's confounding sign.
- D is drawn once per episode and shared by both arms, and `do(A)` cannot move
  it. Drivers therefore need `pair_mode="counterfactual"`, and `generate_batch`
  refuses them. Driver draws come from their own seed stream, so every
  simulation draws the base structure's mechanisms, noise and intervention
  exactly as it would without the driver. Plain labels are unchanged, and the
  released suites still regenerate bit for bit.
- The columns are A, the base structure's middle columns, D at `N-2` and Y at
  `N-1`.

`scripts/release_config_seasonal_trend.yaml` defines `dot-SeasonalTrend-v1`
1.0.0, 10,000 counterfactual episodes with `T = 200`, 1,000 per label. The
labels are the plain `bi_variate` and `back_door` structures (tier 1) and each
of them with an observed (tier 2) or hidden (tier 3) seasonal or trend driver.
The label is the episode's `structure`, so per-structure evaluation reports
every stratum separately.

**Observed or hidden.** An observed D is a released column whose metadata sets
`known_future: true`. D is exogenous and a deterministic function of time, like
a calendar feature, so an evaluator may leave it unmasked after the onset and a
model may read its future values. A hidden D is zeroed in both released arms
and in the variable mask, exactly like the hidden confounder U. A hidden driver
confounds A and Y through time. Because D is a deterministic function of time,
the effect stays identifiable in principle by modelling time, for example with
trend or seasonal terms as in an interrupted time series. What a hidden driver
defeats are estimators that neither adjust for D nor model time, and estimators
that assume a stationary series. The structure whose effect is not identifiable
is `bow_graph`.

**Metadata.** Every driven episode records `metadata["driver"]` with `kind`,
`observed`, `column`, `strength`, `params` (`period` and `phase`, or
`direction`), `loadings` (`A` and `Y`), `confounding_sign`, `known_future`,
`burn_in` and `generation_seed`. Calling
`dotime.drivers.released_driver_series(metadata["driver"], T)` rebuilds D on the
released rows bit for bit, a hidden D included.

**Baselines.** `BackDoorOLS` adjusts for `{X, D}` or `{D}` on observed-driver
labels whose base is in the back-door family or is `bi_variate`, and predicts
the pre-onset mean on hidden-driver labels. On the first 100 episodes of each
label (`results/reference/seasonal_trend/`), effect-sign accuracy on the
seasonal labels is 0.60 to 0.67 without adjusting for an observed D and 0.91
and 0.97 with it. Estimators that neither see D nor model time score 0.61 and
0.67 on the hidden seasonal labels, against 0.92 to 0.96 on the plain
structures.

**Stationarity.** Trend episodes are not stationary. They fall outside the
stationarity-after-burn-in assumption of the paper's convergence result, which
therefore does not cover them. A seasonal driver has a uniformly random phase,
so over episodes it is a stationary process, although within one episode D is
periodic.

## Evaluation protocol

The default evaluation reports RMSE, NMSE, MAE, direction accuracy, lift-over-naive, and effect-error correlation, computed per-structure and pooled.

```python
from dotime.evaluation import evaluate

results = evaluate(model, suite)  # sign of the interventional level (v1)
results = evaluate(model, suite, dir_target="effect")  # sign of the causal effect y - y_obs
```

Direction accuracy asks whether a model gets the direction of the
intervention right. With `dir_target="level"` (the v1 protocol and the
default) it compares the sign of the predicted and true interventional
level, which a positive baseline can make positive whatever the intervention
did. With `dir_target="effect"` it compares the sign of `y_pred - y_obs` and
`y_true - y_obs` at the query, which is the direction of the intervention's
effect. On `dot-Identifiability-v1` 1.1.0 the two signs disagree on 22% of the
episodes where both are scoreable. RMSE, MAE, NMSE and R² are level metrics
and are the same either way. `dotime-benchmark` and `dotime-eval-submission`
take the same choice as `--dir-target`. Effect scoring refuses the archived
`dot-Identifiability-v1` 1.0.0 files, whose `x_obs` is misaligned; score them
with `dotime-eval-reference --dir-target effect --realignment <sidecar>`.

See the {doc}`api` reference for the full `benchmarks`, `baselines`, and
`evaluation` module documentation.

## Target QA

Seed protocols guard against variance. They do not guard against a systematically
corrupted target: the v1 observational training arm was all zeros and passed every
seed check. Every suite build, benchmark run and training loader therefore logs and
asserts per-arm target statistics with `dotime.qa` before its numbers are trusted.
The arms of a query are

- `y_obs_level`, the observational level of the queried variable at the query row,
- `y_int_level`, the interventional or counterfactual level `y_true`,
- `effect`, their difference `y_int_level - y_obs_level`.

For each arm the report records the query count, the non-finite count, the nonzero
fraction, the mean, the variance and the largest magnitude, pooled and per structure
(per regime density for `dot-RegimeSwitch-v1`). Both level arms must be finite, have
a positive variance and be nonzero on at least half of the queries. When a run
scores or trains on the effect, the effect must be nonzero on at least 5% of the
queries that can carry one. A query cannot carry an effect when its structure's DAG
has no directed path from `A` to `Y` (`observed_confounder`, `unobserved_confounder`),
or when it comes fewer steps after the onset than the shortest such path (`mediator`
queried at the onset). `dotime.qa.is_null_effect` reads this off the structure's
temporal DAG, and structures it does not know are never exempt. Groups with fewer
than 10 queries are reported but not asserted.

```python
from dotime.benchmarks import load_benchmark
from dotime.qa import target_qa

suite = load_benchmark("dot-Identifiability-v1")
report = target_qa(list(suite), dir_target="effect")  # raises TargetQAError on failure
report.to_dict()  # the JSON stored as "target_qa" in every output
```

| Entry point | What it checks | Opt-out |
|---|---|---|
| `scripts/build_release.py` | Every arm of each suite before it is written. The report goes into `manifest.json` and `build_manifest.json` | `--target-qa warn` or `off` |
| `dotime-benchmark`, `dotime-eval-submission`, `dotime-eval-pfn` | The evaluated episodes, with the run's `--dir-target` | `--target-qa warn` |
| `dotime-eval-tabpfn`, `dotime-eval-chronos` | The evaluated subsample, level arms | `--target-qa warn` |
| `dotime-eval-reference` | The evaluated episodes, pooled and per structure | None |
| `TemporalInterventionDataLoader` | The raw targets of the first 64 queries of each structure, and the effect when `target_key="Y_causal_effect"` | `target_qa=False` |

The loader reports through `logging` at warning level, so the statistics appear in a
training log without any logging setup, and it draws no random numbers, so its
batches are bit-identical with the check on or off. All five released suite versions
pass the defaults, with the observational level of `dot-Identifiability-v1` 1.0.0 read
from the realignment sidecar (`results/reference/audit_2026-09/frozen_target_qa.json`).
