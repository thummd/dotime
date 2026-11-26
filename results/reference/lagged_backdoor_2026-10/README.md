# The lagged back-door adjustment set on dot-Identifiability-v1 1.2.0: results

Registered in PREREG.md by commit `e1e0b6c3551f660b0d55c9b161b7ebf1f0300b8c` (tag
`lagged-backdoor-prereg`, 2026-10-28 16:32 SGT) and scored the same day between 16:35 and 16:35 SGT
at that commit with a clean tree. The project makes no public push before the KDD notification;
the push event will be added here when it happens. The question was whether the lag-one back-door
set, which blocks the path `A_t <- A_{t-1} <- X_{t-1} -> Y_t` that the detection-power study
named in its prediction E3, removes the false effects of BackDoorOLS on `observed_confounder`
(L1) without costing effect-sign accuracy where the contemporaneous set already identifies the
sign (L2, L3).

**Public push, added 2026-11-27 SGT.** The branch `phase2-pfnref-mirror` and the tags
`lagged-backdoor-prereg` and `lagged-backdoor-scored` were first pushed on 2026-11-26 17:06:07 UTC
with their original hashes (`push_event.json`), after the KDD notification, and merged into
`main` by `f751f00`. Until then the registration existed only in the local repository. The push
therefore dates the publication, 29 days after the run, and does not show that the registration
preceded the run. That order rests on the commit chain (`e1e0b6c` before `44c785e`), on the two
tag objects and on `code_manifest.sha256`.

## Graded predictions (`grade.md`, written by `scripts/grade_lagged.py`)

Differences are a minus b, paired over the same episodes, percentile bootstrap with 2,000
replicates and seed 20261001, the study's settings.

| Id | Structure | Statistic | a | b | a | b | a minus b | 95% CI | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| L1 | `observed_confounder` | false-effect rate | LaggedBackDoorOLS | BackDoorOLS | 0.464 | 0.479 | -0.014 | [-0.034, +0.007] | not confirmed |
| L1b | `observed_confounder` | false-effect rate | LaggedBackDoorOLS | do-SVAR | 0.464 | 0.364 | +0.101 | [+0.078, +0.124] | not confirmed |
| L2 | `back_door` | effect-sign accuracy | LaggedBackDoorOLS | BackDoorOLS | 0.806 | 0.810 | -0.004 | [-0.019, +0.011] | confirmed |
| L3 | `confounder_mediator` | effect-sign accuracy | LaggedBackDoorOLS | BackDoorOLS | 0.672 | 0.678 | -0.006 | [-0.031, +0.020] | not confirmed |

L3 fails its criterion by 0.0013: the lower bound is -0.0313 against the registered margin of
-0.03, with a point difference of -0.006 on 351 valid episodes. It is reported as graded.

Exploratory rows (reported, no threshold): the three-lag variant makes false effects at 0.456
(X1a, -0.022 [-0.047, +0.002] against BackDoorOLS) and scores 0.795 on `back_door` (X1c, -0.015
[-0.033, +0.004]) and 0.667 on `confounder_mediator` (X1d, -0.011 [-0.049, +0.028]); it is 0.011
[-0.027, +0.004] below the one-lag variant on `back_door` (X1e). The one-lag variant beats do-SVAR
on `back_door`, +0.030 [+0.003, +0.058] (X2a), is level with it on `confounder_mediator`, +0.014
[-0.026, +0.054] (X2b), and lies below FrontDoorOLS there, -0.040 [-0.083, +0.003] (X2c). Neither
variant approaches the study's 0.10 bar for the false-effect rate (X3a 0.464, X3b 0.456).

The full table with every registered estimator is `ident_v1_2.md`. With LaggedBackDoorOLS in
the place of BackDoorOLS the separation of the best identification-aware estimator from the best
naive one would be +0.271 [+0.236, +0.299] on `back_door` against the study's +0.275 [+0.238,
+0.302], and +0.114 [+0.054, +0.157] on `confounder_mediator` against +0.120 [+0.057, +0.165].

## Reading, as registered in PREREG.md Section 6

- L1 not confirmed. Blocking the lagged path moves the false-effect rate from 0.479 to 0.464, a
  change inside its interval. The open path is therefore not what produces the false effects of
  BackDoorOLS on `observed_confounder`. The registered alternative reading applies: the calls come
  from the sampling noise of the treatment coefficient against the absolute threshold of 0.1, under
  nonlinear mechanisms, on fit windows of about 140 to 155 rows. The call rates on `back_door`
  point the same way: do-SVAR, which fits with a ridge penalty on standardised columns, calls an
  effect on 0.570 of the episodes against 0.699 for BackDoorOLS and 0.707 for LaggedBackDoorOLS,
  so its lower false-effect rate of 0.364 goes with fewer calls of every kind, a shrinkage effect
  rather than a better adjustment set.
- L2 confirmed and L3 not confirmed by 0.0013 of its lower bound: the lagged set costs no accuracy
  that the intervals can resolve, on `back_door` it is the better estimator against do-SVAR (X2a),
  and the three-lag variant pays for its extra coefficients with about 0.015 of accuracy (X1c, X1e).
- Consequence for the study and the thesis. The sentence that explains the 0.479 by the open lagged
  path (study README, prediction E3, and Chapter 2 of the thesis proposal) is corrected to this
  result; the registered sections of the study README stay as written and this folder is the
  correction. The router keeps BackDoorOLS, as PREREG.md Section 6 says.

## Execution record

- First invocation (`run_gate3_failed.log`, `ident_v1_2_gate.json`): `scripts/run.sh` as
  registered. Gate 3 failed on 8 of the 77 compared cells, all of them `level_rmse` or
  `effect_rmse` of VAR-OLS, NaiveOLS and the router on NaiveOLS. Against the study's own 1.2.0
  table, 10 of 1,881 compared cell values differ, by at most 3.6e-7 (`level_rmse`, relative 4.2e-7)
  and 6.0e-8 (`effect_rmse`, relative 1.1e-7); every effect-sign accuracy, call rate, interval and
  separation and all ten pre-registered outcomes of the study are identical. The 14 skipped cells
  are the two extensions, which the 1.1.0 table does not hold. This is the environment case that
  PREREG.md Section 4 anticipated (torch 2.10.0 here against the study's environment).
- Second invocation (`run.log`): `scripts/run.sh --limit-per-structure 1350`, the registered
  fallback, which scores all 12,150 episodes and skips the gates; 38 seconds with 8 workers on 16
  cores. Its outputs are `ident_v1_2.json`, `ident_v1_2.md`, `ident_v1_2_predictions.parquet`
  (157,950 rows) and, from the grader, `grade.json`, `grade.md` and `grade.log`.
- DEVIATIONS.md records the fallback.
- `ident_v1_2_predictions.parquet` (SHA-256 `6d0dc69d7cb2ed7f54d07a45d5e4593e4046050a23bccdcbab38728dfd391ed3`)
  was held back from the scoring commit by the repository's `*.parquet` ignore rule and was added on
  2026-11-27, as the other registrations keep theirs. Regrading from it with
  `scripts/grade_lagged.py` reproduces `grade.json` and `grade.md` byte for byte.
