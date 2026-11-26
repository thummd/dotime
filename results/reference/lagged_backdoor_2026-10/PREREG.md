# Pre-registration: the lagged back-door adjustment set on dot-Identifiability-v1 1.2.0

Registered on 2026-10-28 by the commit that adds this file to the repository, on the branch
`phase2-pfnref-mirror`, under the local tag `lagged-backdoor-prereg`. The project makes no public
push before the KDD notification, so the public push event will be added to README.md when it
happens. Until then the record is the commit hash, which README.md states, and
`code_manifest.sha256`, which lists the digest of every file this registration depends on. No run
of the detection-power script with the new estimators on the suite precedes the commit, except the
smoke run disclosed in Section 2, item 7. Author: Dennis Thumm.

## 1. Question

The detection-power study of October 2026 (`../detection_power_2026-10/`) registered its
prediction E3: on `observed_confounder`, where A has no effect on Y, the contemporaneous back-door
set of BackDoorOLS leaves the path `A_t <- A_{t-1} <- X_{t-1} -> Y_t` open, so BackDoorOLS makes
false effects at a rate of at least 0.10 while do-SVAR, which conditions on the lags, makes fewer.
The study graded E3 "not confirmed", because the rate of BackDoorOLS, 0.479, was not below the
rate of NaiveOLS, 0.450, as the statement also required. E3 named the open lagged path as the
mechanism but did not test the remedy, because the package had no lag-aware back-door estimator.
The thesis proposal (Chapter 2) and its presentation state that the lagged adjustment set is not
implemented. This registration implements it and asks two questions.

Q1. Does conditioning on one lag of the treatment, the confounder and the outcome remove the false
effects that the contemporaneous set makes on `observed_confounder`?

Q2. Does the lagged set cost effect-sign accuracy where the contemporaneous set already identifies
the sign, on `back_door` and `confounder_mediator`?

## 2. Prior knowledge

1. The structure of the released suites. `dotime.tscm_sampler` builds every identification
   structure with `max_lag = 1`, an autoregressive self-edge on every variable (`G_k[i, i] = 1`)
   and, in the back-door family, the lagged edge `X(t-1) -> Y(t)`. The default construction of
   `TSCMPrior` for these structures keeps `max_lag = 1`, and the graph record stored with every
   released episode of the three structures has order one (Section 4). The
   canonical column order is `[A, X, Y]` for `back_door` and `observed_confounder` and
   `[A, X, M, Y]` for `confounder_mediator` (`baselines._back_door_columns`). In the unrolled graph
   of order one, the parents of `A_t` are `X_t` and `A_{t-1}`, neither a descendant of `A_t`, and
   `X_{t-1}` and `Y_{t-1}` precede `A_t`. The set `{X_t, A_{t-1}, X_{t-1}, Y_{t-1}}` therefore
   contains every parent of `A_t` and no descendant of it, which makes it a valid back-door set by
   the back-door criterion in all three structures. The mediator `M_t` of `confounder_mediator`
   stays excluded as before.
2. The registered numbers of the study on 1.2.0 (`../detection_power_2026-10/ident_v1_2.md`,
   12,150 episodes, 1,350 per structure). Effect-sign accuracy: BackDoorOLS 0.810 ± 0.014 on
   `back_door` (789 valid episodes) and 0.678 ± 0.025 on `confounder_mediator` (351); do-SVAR
   0.776 and 0.658; NaiveOLS 0.788 and 0.672; FrontDoorOLS 0.712 on `confounder_mediator`.
   False-effect rate on `observed_confounder` (1,350 episodes): BackDoorOLS 0.479 ± 0.014, do-SVAR
   0.364 ± 0.013, NaiveOLS 0.450 ± 0.014. Separation of the best identification-aware estimator
   from the best naive one: +0.275 [+0.238, +0.302] on `back_door`, +0.154 [+0.100, +0.194] on
   `confounder_mediator`. E3 not confirmed.
3. The false-effect criterion is absolute: a call is `|pred(do v) - pred(do a_ref)| >= 0.1` in the
   suite's units. For an estimator that is affine in the do-value the contrast equals the fitted
   coefficient of `A_t` times `v - a_ref`. The primary do-values of the suite have a median of 5.26
   pre-onset standard deviations of the treatment (`../prereg_2026-10_s15/PREREG.md`, item D10), so
   the sampling noise of the coefficient alone produces calls on a share of the episodes. A correctly
   specified estimator is therefore not expected to reach a rate near zero, and this registration
   does not predict that the study's 0.10 bar is met (Section 5, X3).
4. The mechanisms of the suite are nonlinear (sampled activations), so a linear adjustment is
   misspecified. The lagged set can remove the bias of the open path but not misspecification bias.
5. Linear checks before this commit, not on the suite (`tests/test_baselines.py`,
   `_order_one_scm`). On a linear order-one SCM with the null structure, 4,000 rows, seeds 0 to 2,
   the unit-step contrast of BackDoorOLS is 0.16 to 0.17 and that of LaggedBackDoorOLS is -0.014 to
   +0.004; at 200 rows, 0.17 to 0.23 against -0.010 to +0.006. With a unit effect added,
   BackDoorOLS gives 1.07 to 1.09 and LaggedBackDoorOLS 0.99 to 1.00. The three-lag variant equals
   the one-lag variant to three decimals on these systems.
6. Expectation stated in advance: L1 confirmed, L2 and L3 confirmed. L1b is uncertain, because
   do-SVAR conditions on three lags of every variable with a ridge penalty while the lagged OLS
   conditions on the exact set, and the mechanisms are nonlinear.
7. Smoke test before this commit: `detection_power.py --version 1.2.0 --limit-per-structure 5`
   with the thirteen estimators of `scripts/run.sh`, 45 episodes, written to a scratch directory
   that is not kept. It checked that the pipeline runs and writes its three files, and that
   `scripts/grade_lagged.py` runs on its output. Its log was read for the exit status, the episode
   count and the file list only, and no metric of it was read.

## 3. The estimator

`LaggedBackDoorOLS` (`dotime.baselines.LaggedBackDoorOLSBaseline`, a subclass of
`BackDoorOLSBaseline`) fits by ordinary least squares, on the pre-intervention rows `t` in
`[lag, onset)`,

    Y_t ~ 1 + A_t + X_t + sum_{k=1..lag} (A_{t-k} + X_{t-k} + Y_{t-k}),

with `X` the structure's back-door set as the parent reads it off the DAG, and predicts
`E[Y | do(A = v)]` by plugging `v` into `A_t` and averaging over the observed rows of every other
regressor. The parent's rule decides applicability (the back-door family, else the pre-onset
outcome mean), and an episode with fewer fitted rows than coefficients plus one takes the mean as
well. `lag = 1` is the registered variant. `LaggedBackDoorOLS-L3` is the same estimator with
`lag = 3`, the order do-SVAR uses, and is exploratory. The parent is unchanged, and so are the
routes, the valid sets and the graded predictions of the study: the two names are listed in
`estimators.EXTENSIONS` and scored only when named on the command line.

## 4. Data and scoring

Suite: `dot-Identifiability-v1` version 1.2.0 from the local cache of `dotime.benchmarks`,
12,150 episodes, manifest md5 `3994e20d23af5aecb322e8503f1b47ae`, `e3f5a3354f471d730f1537fe6b44dae5`
and `121ce5c241664325ddb30979485ccc80` for its three shards. The graph record stored with each
episode (`metadata_json`, key `graph`) has `k_sampled = 1`, `k_eff = 1` and no edge beyond lag one
for all 4,050 episodes of `back_door`, `observed_confounder` and `confounder_mediator`, read
before the registering commit and after the smoke run; no prediction or target was read with it.
Scoring script:
`../detection_power_2026-10/scripts/detection_power.py` at the registering commit, invoked by
`scripts/run.sh` with every registered estimator plus the two extensions, so that gate 3 compares
the registered rows with the 1.1.0 table exactly as the study did; the gate record, the table, its
Markdown rendering and the per-episode predictions are written to this folder. Should gate 3 fail
on a registered row for a reason unrelated to the extensions, such as an arithmetic difference of
the environment, the run is repeated with `--limit-per-structure 1350`, which scores every episode
and skips the gates, the failed gate record is kept, and DEVIATIONS.md says so. Grading:
`scripts/grade_lagged.py` reads the predictions file, pairs each contrast on the same episodes,
and bootstraps the paired difference with the study's 2,000 replicates and seed 20261001. Its
validity mask is the study's `sign_correct`, its threshold the study's `FALSE_EFFECT_THRESHOLD`,
both imported. A row is gradable when its structure has at least 100 defined episodes.

## 5. Predictions

Primary, graded `confirmed` or `not confirmed`:

| Id | Prediction | Criterion | Confidence |
|---|---|---|---|
| L1 | On `observed_confounder` the lagged set makes fewer false effects than the contemporaneous set | Paired 95% CI of rate(LaggedBackDoorOLS) minus rate(BackDoorOLS) below 0 | High |
| L1b | On `observed_confounder` the lagged set makes fewer false effects than do-SVAR | Paired 95% CI of rate(LaggedBackDoorOLS) minus rate(do-SVAR) below 0 | Low |
| L2 | On `back_door` the lagged set does not cost effect-sign accuracy | Lower bound of the paired 95% CI of acc(LaggedBackDoorOLS) minus acc(BackDoorOLS) above -0.03 | Moderate |
| L3 | On `confounder_mediator` the lagged set does not cost effect-sign accuracy | Lower bound of the paired 95% CI of acc(LaggedBackDoorOLS) minus acc(BackDoorOLS) above -0.03 | Moderate |

The margin of 0.03 is about two binomial standard errors of the registered accuracy on `back_door`
(0.014 at 789 valid episodes).

Exploratory, reported with the same intervals and graded against nothing: X1, the three-lag
variant in the four contrasts above and against the one-lag variant on `back_door`; X2, the one-lag
variant against do-SVAR on `back_door` and `confounder_mediator` and against FrontDoorOLS on
`confounder_mediator`; X3, the false-effect rates of both variants on `observed_confounder`
against the study's 0.10 bar.

## 6. Interpretation registered in advance

- L1, L2 and L3 confirmed: the open lagged path is the mechanism behind the failure of E3, and the
  lag-one set is the right back-door set for the released suites. Chapter 2 of the thesis proposal
  and its presentation are updated from "not implemented" to the result. The router keeps
  BackDoorOLS; replacing it would be a change of the registered study and would need its own
  registration.
- L1 not confirmed: the false effects of BackDoorOLS on `observed_confounder` are not, or not only,
  the open path. The remaining candidates are the absolute threshold under sampling noise (X3) and
  the nonlinear mechanisms. No follow-up is registered here.
- L2 or L3 not confirmed: at windows of a few hundred rows the lagged set costs more variance than
  it removes bias, and the trade-off is reported as such.
- L1b either way: reported. do-SVAR is not what this registration is about.

## 7. Provenance

Package: `dotime` at the registering commit, Python 3 with the versions `run.log` records. The
files this registration depends on and their SHA-256 digests at the registering commit are listed
in `code_manifest.sha256`: `src/dotime/baselines.py`, `tests/test_baselines.py`, the two study
scripts and the two scripts of this folder. README.md, written after the run, states the commit,
the tag, the runtime and the outcome of every row.
