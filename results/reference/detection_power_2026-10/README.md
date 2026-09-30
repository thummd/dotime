# Detection power of dot-Identifiability-v1 (October 2026)

The v1 paper claimed that a model trained on interventions beats an observational twin.
That claim is withdrawn: corrected, no such model gap exists. The benchmark can still
detect causal reasoning in *estimators*. On dot-Identifiability-v1 1.1.0 (shared-noise
counterfactual targets), BackDoorOLS scores 0.810 effect-sign accuracy on `back_door`
(n = 789) and IV2SLS 0.613 on `instrumental_variable`, against 0.52 to 0.53 for the naive
baselines (`../v1_1/ident_cpu_effect_backdoor_fix.json`,
`../audit_2026-09/ident_v1_1_per_structure_backdoor_fix.json`).

This folder asks the general question: structure by structure, which estimator classes
does the benchmark separate? It covers the non-identified structure `bow_graph` and the
null-effect controls. Every estimator and every structure is reported, including the rows
that disappoint.

## Protocol

Every estimator predicts the interventional *level* of the outcome at the query row. The
analysis evaluates it twice per episode:

- at the episode's do-value `v`, which gives the official effect-sign score
  `direction_accuracy(pred - y_obs, y_true - y_obs)`. Here `y_obs` comes from
  `dotime.evaluation.query_obs_levels`. The score counts episodes with
  `|y_true - y_obs| >= 0.1` (`n_valid`), exactly as `evaluate(..., dir_target="effect")`
  does.
- at `a_ref = x_obs[onset, A]`, the factual treatment of the intervened row. With shared
  noise, `do(A = a_ref)` reproduces the factual trajectory. So `pred(v) - pred(a_ref)` is
  the estimator's estimated effect. For every structure queried at the onset, `a_ref` is
  `a_obs(t_q)`.

All arithmetic is float32, as in `dotime.reference.reference_table.run_baseline`.

| Metric | Definition |
|---|---|
| Effect-sign accuracy | The official protocol above, with `n_valid` and the binomial SE `sqrt(p (1 - p) / n_valid)` |
| Level RMSE | `sqrt(mean((pred - y_true)^2))` |
| Level-sign accuracy | `direction_accuracy(pred, y_true)`, the v1 paper protocol |
| False-effect rate | On null-effect structures, the fraction of episodes with `|pred(v) - pred(a_ref)| >= 0.1`. It is labelled `trivial` where the estimator ignores the do-value on that structure and `rule 3` for the router's `tau = 0` |
| Separation | Best identification-aware minus best naive effect-sign accuracy, with a paired episode bootstrap 95% interval. The bootstrap uses 2,000 resamples of the valid episodes and a fresh `numpy.random.default_rng(20261001)` per structure, and takes the maxima inside each resample |
| Against the best naive | The same paired interval for every non-naive estimator |
| Router against valid peers | Router minus each other valid estimator of the structure, paired |
| Cross-structure | Accuracy on `bi_variate` minus accuracy on `bow_graph`, with an independent bootstrap. `bow_graph` is `bi_variate` plus the hidden confounder U |

A structure is *null-effect* when its DAG has no path from A to Y (`observed_confounder`,
`unobserved_confounder`). It is also null-effect when every episode's effect is exactly
zero at its query. That is the case for `mediator` in 1.1.0, whose only path is lagged and
which is queried at the onset.

## Estimators

`scripts/estimators.py` defines them. The package baselines are wrapped, not changed.

| Estimator | Class | Definition | Reads the do-value on |
|---|---|---|---|
| Zero | naive | 0 | never |
| Mean | naive | Pre-onset mean of Y (TrajMean) | never |
| AR1 | naive | Last pre-onset Y | never |
| VAR-OLS | naive | Package VAR(3). It fits the whole observational trajectory and forecasts one step past its end | never |
| NaiveOLS | association | Package baseline from session A: OLS `Y_t ~ 1 + A_t + Y_{t-1}` before the onset, at `do(A = v)`, averaged over the history. It is `pending` until the package registers it, and it is never reimplemented here | every structure |
| do-SVAR | do-SVAR | Recursive structural VAR ported from the private do-over-time-pfn `svar_forecast`: p = 3, ridge 1e-2, pre-onset columns standardised with the loader statistics, all-zero hidden columns dropped, canonical order (A first, Y last), A clamped to `v` at the onset row and rolled to the query row. Histories shorter than 3 times the parameters per equation fall back to the pre-onset mean | every structure |
| BackDoorOLS | identification-aware | Package baseline: `Y_t ~ 1 + A_t + X_t + Y_{t-1}` with X the DAG's back-door set | `back_door`, `observed_confounder`, `confounder_mediator` |
| IV2SLS | identification-aware | Package baseline: two-stage least squares with every column other than A and Y as instruments (the zeroed hidden U contributes nothing). It falls back to the mean when the first-stage R^2 < 0.1 | `instrumental_variable` |
| FrontDoorOLS | identification-aware | New. `alpha` from `M_s ~ 1 + A_s + M_{s-1}`, `gamma` from `Y_s ~ 1 + M_s + A_s + Y_{s-1}` (conditioning on A blocks `M <- A <- U -> Y`), `tau = alpha * gamma`, prediction `mean(Y_pre) + tau * (v - mean(A_pre))`. The mediator column comes from `dotime.baselines._front_door_columns` | `mediator`, `front_door`, `confounder_mediator` |
| GraphRouter | router | The estimator the structure's DAG licenses (below) | per route |
| Oracle | oracle | `y_true` at `v`, `y_obs` at `a_ref` | every structure |

On a structure it does not read the do-value on, an estimator predicts the pre-onset
outcome mean (or 0 for Zero). Its predicted effect is then zero by construction. The
tables mark these cells.

### GraphRouter routes

`estimators.route` derives the route from the structure's summary graph. The rules apply
in order:

1. With no hidden confounder of A and Y, the back-door criterion holds. The router uses
   the DAG's adjustment set, or no adjustment when that set is empty.
2. Otherwise, if A is not an ancestor of Y, do-calculus rule 3 gives `tau = 0`.
3. Otherwise a front-door mediator gives FrontDoorOLS.
4. Otherwise an instrument gives IV2SLS.
5. If none of these applies, the effect is not identified. The router then reports the
   unadjusted estimate, labelled as such.

The self-test asserts that the rules give this table:

| Structure | Route | Rule |
|---|---|---|
| `bi_variate` | NaiveOLS | Unadjusted (no confounding) |
| `mediator` | NaiveOLS | Unadjusted (no confounding) |
| `back_door`, `observed_confounder`, `confounder_mediator` | BackDoorOLS | Back-door adjustment |
| `front_door` | FrontDoorOLS | Front-door adjustment |
| `instrumental_variable` | IV2SLS | Instrumental variable |
| `unobserved_confounder` | `tau = 0`, predicting the pre-onset mean | Rule 3 |
| `bow_graph` | NaiveOLS | No identification (unadjusted estimate) |

`observed_confounder` has no A to Y path either. It stays with BackDoorOLS, the route its
back-door label gives. That keeps it a test of whether estimated adjustment removes
observed confounding, while `unobserved_confounder` tests the graph rule.

### Valid estimators per structure

This table is fixed before any run, and the router comparison uses it. An estimator is
valid where its identification assumption holds on the structure's summary graph. For
do-SVAR that means the canonical order is causal and nothing hidden confounds A and Y.

| Structure | Valid estimators |
|---|---|
| `bi_variate` | NaiveOLS, do-SVAR |
| `back_door` | BackDoorOLS |
| `observed_confounder` | BackDoorOLS |
| `mediator` | NaiveOLS, do-SVAR, FrontDoorOLS |
| `front_door` | FrontDoorOLS |
| `confounder_mediator` | BackDoorOLS, FrontDoorOLS |
| `instrumental_variable` | IV2SLS |
| `unobserved_confounder` | none (the router's `tau = 0` only) |
| `bow_graph` | none |

## Gates

The script asserts the gates in this order. A table that fails its gate is not written.

1. **Reproduction (1.1.0).** On `load_benchmark("dot-Identifiability-v1", version="1.1.0")`,
   the package estimators reproduce the pooled effect and level `dir_acc`, `n_valid` and
   RMSE of `../v1_1/ident_cpu_{effect,level}_backdoor_fix.json` exactly. The record is
   written to `ident_v1_1_gate.json`.
2. **Self-test (`--self-test`).** Synthetic linear Gaussian SVARs with T = 20,000, laid out
   in each structure's canonical columns, give each estimator the known effect within 0.05
   where its assumptions hold. Confounded estimators miss it by at least 0.2. NaiveOLS is
   among the confounded ones on `back_door`, `observed_confounder`, `front_door`,
   `confounder_mediator`, `instrumental_variable`, `unobserved_confounder` and `bow_graph`.
   Declining estimators predict no effect, the oracle returns `y_true` at `v` and `y_obs`
   at `a_ref`, and the routes match the table above. Report-only scenarios add every
   lagged edge and autoregression of the benchmark DAGs (see Limitations). The record is
   written to `self_test.json`.
3. **Stability (1.2.0 and later).** This gate needs gates 1 and 2 on record in the output
   folder. Every structure other than `mediator` then scores exactly as in
   `ident_v1_1.json`, cell for cell and separation for separation. The oracle scores 1
   wherever the effect sign is defined, with no false effect. The record is written to
   `ident_v1_2_gate.json`.

## Pre-registration

This section was committed together with the scripts. That was before any run on 1.2.0
and before any benchmark run of NaiveOLS, do-SVAR, FrontDoorOLS or the GraphRouter. There
were two earlier runs on 1.1.0, and only their mechanics were inspected. A smoke run on 40
episodes (5 per structure) checked that the pipeline runs, and its metrics were not read.
A gate-1 run of the six published package estimators and the oracle was read only for its
gate record. Later commits leave this section
unchanged, and `git log -p README.md` shows that. `evaluate_predictions` in
`scripts/detection_power.py` grades the predictions mechanically on every run. The
outcomes are listed at the top of each `ident_v*.md`.

What was known when it was written:

- the 1.1.0 per-structure effect-sign accuracies of Zero, Mean, AR1, VAR-OLS, BackDoorOLS
  and IV2SLS (`../audit_2026-09/`). P1 on 1.1.0 and the point estimates behind P2 on
  `back_door`, `confounder_mediator` and `instrumental_variable` were therefore expected
  to hold on 1.1.0. The informative tests are the new estimators, the 1.2.0 `mediator`
  query at offset 1 and `bow_graph`.
- the synthetic self-test, including the report-only benchmark-like scenarios. Their
  linear drifts ground E1 to E3.

A structure is gradable when it has at least 100 valid episodes. A statement with several
parts is confirmed only if every gradable part holds.

**Primary predictions** (from the analysis brief):

| Id | Prediction | Criterion | Confidence |
|---|---|---|---|
| P1 | Naive estimators sit near 0.5 everywhere | Every naive effect-sign accuracy in [0.40, 0.60] | High |
| P2 | Identification-aware estimators are ahead only where their assumptions hold | Separation 95% CI above 0 on `back_door`, `confounder_mediator`, `instrumental_variable` and `front_door`. No identification-aware estimator applies on `bi_variate`, `unobserved_confounder` or `bow_graph`. No prediction for `mediator` | High for the first three, moderate for `front_door` (see E4) |
| P3 | The GraphRouter is at least as good as the best single valid estimator | For every structure and every other valid estimator, the paired 95% CI of router minus that estimator has an upper bound >= 0. E1 and E2 are anticipated exceptions | Moderate |
| P4 | The router makes no false effect on `unobserved_confounder`, while association does | Router false-effect rate exactly 0 there. NaiveOLS and do-SVAR rates >= 0.10 | High |
| P5 | On `bow_graph` no observational estimator identifies the effect | No identification-aware estimator applies, and the route is marked not identified. NaiveOLS and do-SVAR lose effect-sign accuracy from `bi_variate` to `bow_graph` (95% CI of the difference above 0) | High for the structural parts, moderate for the loss |

**Anticipated exceptions and secondary predictions** (grounded in the self-test):

| Id | Prediction | Criterion | Confidence |
|---|---|---|---|
| E1 | On `mediator` queried after the onset (1.2.0), the router's contemporaneous NaiveOLS is worse than do-SVAR, which models the lag. With a lagged effect of 1, NaiveOLS recovers 0.53 in the linear self-test (A's autocorrelation times the effect) and do-SVAR 1.02 | Router minus do-SVAR, 95% CI entirely below 0 | Moderate |
| E2 | On `confounder_mediator`, FrontDoorOLS scores at least as high as the routed BackDoorOLS. In the benchmark-like linear scenario (A and M autoregressive, lagged X to Y) BackDoorOLS drifts to 1.50 for an effect of 1, FrontDoorOLS to 1.12 | Point estimates | Low |
| E3 | On `observed_confounder`, the contemporaneous back-door set leaves `A_t <- A_{t-1} <- X_{t-1} -> Y_t` open. BackDoorOLS (the router) makes false effects at a rate >= 0.10, below NaiveOLS. do-SVAR, which conditions on the lags, makes fewer than BackDoorOLS. Linear benchmark-like slopes for a null effect are 0.25, 0.40 and -0.01 | Point estimates of the false-effect rates | Moderate. do-SVAR's short-history variance may offset its lack of bias |
| E4 | FrontDoorOLS on `front_door` is the least certain part of P2. Its gamma regression does not condition on `M_{t-1}`, which drives `Y_t`, and conditioning on `A_s` opens a collider path through the autoregressive A. The benchmark-like linear slope is 1.64 for an effect of 1 | Graded within P2 | Stated only |
| S1 | do-SVAR beats the best naive estimator on `bi_variate`, and on `mediator` when the query follows the onset | 95% CI above 0 | Moderate |
| O1 | The oracle scores 1 wherever the effect sign is defined and makes no false effect | Exact | High |

## Running

Run from the repository root with the dev environment. In a worktree whose editable
install points elsewhere, prefix every command with `PYTHONPATH=<checkout>/src`, because
the script refuses to score with another checkout's `dotime`. The script uses 4 worker
processes with one thread each, and a full run takes about 2 minutes.

```bash
python results/reference/detection_power_2026-10/scripts/detection_power.py --self-test
```

```bash
python results/reference/detection_power_2026-10/scripts/detection_power.py --version 1.1.0
```

```bash
python results/reference/detection_power_2026-10/scripts/detection_power.py --version 1.2.0
```

A 1.2.0 build that is not registered in `dotime.benchmarks` yet loads from its build
directory:

```bash
python results/reference/detection_power_2026-10/scripts/detection_power.py --version 1.2.0 --suite-dir <build>/dot-Identifiability-v1-1.2.0
```

`--estimators` restricts the rows. `--limit-per-structure N --out <dir>` makes a smoke run
on the first N episodes of each structure, and it skips the gates.

## Outputs

| File | Content |
|---|---|
| `self_test.json` | Gate 2: routes, per-scenario slopes and expectations |
| `ident_v1_1_gate.json` | Gate 1: every compared number, ours against released |
| `ident_v1_1.json` | Full 1.1.0 table: target QA per structure, per estimator and structure metrics, contrasts, graded predictions, code provenance |
| `ident_v1_1.md` | The same as Markdown tables |
| `ident_v1_1_predictions.parquet` | One row per scored (episode, estimator): `scm_id`, `structure`, `estimator`, `estimator_class`, `uses_do_value`, `onset`, `query_idx`, `do_value`, `ref_do_value`, `y_true`, `y_obs`, `pred` (level at `do(v)`), `pred_ref` (level at `do(a_ref)`). The pre-registered PFN re-test reads it |
| `ident_v1_2*` | The same for 1.2.0, with gate 3 in `ident_v1_2_gate.json` |

Every run logs and asserts the per-arm target statistics (nonzero fraction, mean,
variance) pooled (`reference_table.target_qa`) and per structure before scoring. Rows of
an estimator the package does not register yet are `pending`. The router is `pending` on
the structures it routes to such an estimator.

## Limitations

- **Temporal adjustment.** The package BackDoorOLS and the new FrontDoorOLS condition on
  contemporaneous variables and the outcome's own lag, which is a valid set only in the
  summary graph. In the unrolled time graph, the benchmark's autoregressive treatment and
  lagged edges (`X(t-1) -> Y(t)`, `M(t-1) -> Y(t)`) leave paths such as
  `A_t <- A_{t-1} <- X_{t-1} -> Y_t` open. The self-test switches these paths off where
  it asserts recovery. Its benchmark-like scenarios measure the linear drift that remains
  when they are on: BackDoorOLS 1.13 on `back_door`, 1.50 on `confounder_mediator` and
  0.25 on `observed_confounder` (true 1, 1, 0). FrontDoorOLS reaches 1.64 on `front_door`
  and IV2SLS 1.31 on `instrumental_variable` (true 1). "Valid" in the tables above means
  valid on the summary graph.
- **Effect sign mixes in level noise.** The official protocol compares `pred - y_obs` with
  the true effect. An estimator that knows the effect exactly still misses when
  `|mean(Y_pre) - y_obs|` exceeds the effect, so no observational estimator reaches the
  oracle's 1.
- **Short histories.** Onsets are spread over 10 to 189. do-SVAR needs 3 times the
  parameters per equation (27, 39 or 51 rows for 2, 3 or 4 kept columns) and otherwise
  predicts the pre-onset mean. On 1.1.0 that happens in 8% of the episodes of the
  two-column structures, 15% to 18% of the three-column ones and 23% of
  `confounder_mediator`. Every estimator is noisier on short histories.
- **Multiplicity.** The intervals are per structure and per contrast, with no correction.
- **VAR-OLS** is the package baseline, unchanged. It fits the whole observational
  trajectory, post-onset rows included, and forecasts one step past its end, so it never
  targets the query row.
