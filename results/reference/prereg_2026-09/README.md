# Pre-registered re-test s13: results

Registered in `PREREG.md` (commit 05ae974, pushed 2026-09-30 03:55:25 UTC, `push_event.json`)
before any run started. This file reports every endpoint. Nothing here was chosen after
seeing the numbers: the scripts in `scripts/` are byte-identical to the registered ones.

## Outcome

**Label: REVERSED.** Interventional training changes effect-sign accuracy by -0.007, with a 95% confidence interval from -0.012 to -0.001. The ablation is better, so the withdrawal stands.

| Quantity | Value |
|---|---|
| Primary set | 3,528 episodes of dot-Identifiability-v1 1.2.0 with an effect of at least 0.1 on the six primary structures |
| Effect-sign accuracy, interventional arm | 0.5378 (seeds: 0.5357, 0.5377, 0.5400) |
| Effect-sign accuracy, Arm B (same target, intervention withheld) | 0.5443 (seeds: 0.5434, 0.5439, 0.5456) |
| D, mean over seeds of int minus B | -0.0065 (seeds: -0.0077, -0.0062, -0.0057) |
| Seed interval, 95% and 90% | [-0.0091, -0.0040] and [-0.0082, -0.0048] |
| Paired episode bootstrap, 95% and 90% | [-0.0119, -0.0011] and [-0.0111, -0.0020] |

Both 95% intervals lie below zero, which is the registered condition for REVERSED. The
difference is small: every interval also lies inside the registered equivalence margin of
0.02, and the rule assigns REVERSED because it is checked before EQUIVALENT. No structure
differs significantly in the opposite direction after the Holm correction, so there is no
MIXED qualifier.

All three arms score 0.53 to 0.55. On the same 3,528 episodes the naive
baselines score 0.522 to 0.534 and an unadjusted regression that reads the do-value
scores 0.764 (S5). Under the published recipe the model, with or without the intervention
in its input, stays close to naive forecasting on the sign of the effect.

## Secondary endpoints

**S1, per structure** (int minus B, paired episode bootstrap, Holm over six structures):

| Structure | n | D | 95% interval | p | Holm significant |
|---|---|---|---|---|---|
| `bi_variate` | 846 | -0.0032 | [-0.0110, +0.0043] | 0.458 | no |
| `back_door` | 789 | -0.0076 | [-0.0253, +0.0101] | 0.406 | no |
| `confounder_mediator` | 351 | -0.0019 | [-0.0076, +0.0038] | 0.596 | no |
| `front_door` | 371 | -0.0126 | [-0.0216, -0.0054] | 0.000 | yes |
| `instrumental_variable` | 837 | +0.0032 | [-0.0012, +0.0076] | 0.154 | no |
| `mediator` | 334 | -0.0349 | [-0.0639, -0.0060] | 0.021 | no |

Pooled over the three training structures (n = 1,997): -0.0040 [-0.0114, +0.0035].

**S2, int minus A** (the published contrast, which also changes the target): +0.0033,
seed interval [-0.0042, +0.0108], bootstrap [-0.0045, +0.0110].

**S3, B minus A** (the effect of the training target alone): +0.0098, seed interval
[+0.0040, +0.0157], bootstrap [+0.0009, +0.0187].

**S4, level error and level sign** on all 12,150 episodes (seeds 42, 43, 44):

| Arm | RMSE | | | Level-sign accuracy | | |
|---|---|---|---|---|---|---|
| int | 0.631 | 0.627 | 0.635 | 0.690 | 0.692 | 0.689 |
| B | 0.626 | 0.627 | 0.632 | 0.692 | 0.693 | 0.689 |
| A | 0.613 | 0.613 | 0.613 | 0.697 | 0.695 | 0.694 |

**S5, CPU estimators on the same primary episodes** (from
`../detection_power_2026-10/ident_v1_2_predictions.parquet`):

| Estimator | Effect-sign accuracy |
|---|---|
| Zero | 0.525 |
| Mean | 0.534 |
| AR1 | 0.526 |
| VAR-OLS | 0.522 |
| NaiveOLS | 0.764 |
| do-SVAR | 0.762 |
| BackDoorOLS | 0.612 |
| IV2SLS | 0.552 |
| FrontDoorOLS | 0.573 |
| GraphRouter | 0.724 |
| Oracle | 1.000 |

**S6, the non-identified control `bow_graph`** (n = 828): int 0.556, B 0.560, A 0.542.
Int minus B: -0.0040, bootstrap [-0.0189, +0.0105].

**S7, training-prior check** (`s7_training_prior.json`: 40 held-out batches per structure,
loader seed 12345, T = 200, shared-noise pairs, effect sign, mean over three seeds):

| Structure | int | B | A | int minus B |
|---|---|---|---|---|
| `back_door` | 0.623 | 0.617 | 0.623 | +0.006 |
| `front_door` | 0.617 | 0.619 | 0.628 | -0.002 |
| `instrumental_variable` | 0.597 | 0.599 | 0.612 | -0.001 |

**S8, sensitivity.** Effect thresholds 0.05 and 0.2 (`results_eps_0.05.json`,
`results_eps_0.2.json`): REVERSED with D = -0.0051 and REVERSED with D = -0.0091. The
best-validation checkpoints (`results_best_pt.json`, predictions in `best_pt/`) give EQUIVALENT
with D = -0.0048, seed interval [-0.0079, -0.0017] and bootstrap
[-0.0096, +0.0000].

## How the runs ended

All nine runs wrote their final checkpoint. Five ran the full 5,000 steps. Four stopped
early under the recipe's patience of 5 (`int` seed 43, `B` seed 43 and `A` seed 43 at
step 3,500, `A` seed 42 at step 3,000), printed "Training complete", saved
`do_over_time_pfn_last.pt`, and then the Python process aborted while shutting down
("terminate called without an active exception", a prefetch thread still alive at exit).
The abort follows the final save, so these runs are valid under Section 9, and the
registration scores the last checkpoint with early stopping included.

`A` seed 43 was relaunched once. Its first attempt died of a CUDA out-of-memory error
caused by other users' jobs at step 0 and left no checkpoint. The relaunch
`s13ho_all_A_seed43_r2` used identical flags (its `cmd.txt` records `relaunch_of`).
No run was relaunched or excluded because of its scores.

## Provenance

Scoring ran on 2026-10-01 on the training host's GPU with the dotime package at commit
`e8b9577`, on the 1.2.0 build whose shard digests are recorded in
`s13_scoring_provenance.json` and equal the build that is released. Every checkpoint was
loaded with the strict key check (117 tensors).

| Run | Checkpoint | SHA-256 (first 16) |
|---|---|---|
| `s13ho_all_int_seed42` | `checkpoints/s13ho_all_int_seed42/do_over_time_pfn_last.pt` | `8f15c8316a3c7c0d` |
| `s13ho_all_int_seed43` | `checkpoints/s13ho_all_int_seed43/do_over_time_pfn_last.pt` | `9daf293aa18fc630` |
| `s13ho_all_int_seed44` | `checkpoints/s13ho_all_int_seed44/do_over_time_pfn_last.pt` | `8db7e98683c47914` |
| `s13ho_all_B_seed42` | `checkpoints/s13ho_all_B_seed42/do_over_time_pfn_last.pt` | `59cb19ffbe1a082b` |
| `s13ho_all_B_seed43` | `checkpoints/s13ho_all_B_seed43/do_over_time_pfn_last.pt` | `c80c166b9dbd6373` |
| `s13ho_all_B_seed44` | `checkpoints/s13ho_all_B_seed44/do_over_time_pfn_last.pt` | `47fbc9b4420c0e60` |
| `s13ho_all_A_seed42` | `checkpoints/s13ho_all_A_seed42/do_over_time_pfn_last.pt` | `d2bd90a394af62b6` |
| `s13ho_all_A_seed43` | `checkpoints/s13ho_all_A_seed43_r2/do_over_time_pfn_last.pt` | `83632fc62638133e` |
| `s13ho_all_A_seed44` | `checkpoints/s13ho_all_A_seed44/do_over_time_pfn_last.pt` | `ee3547851d9849ed` |

`s13_predictions.parquet` holds one row per (episode, arm, seed) with the prediction, the
target and the factual level, so every number above can be recomputed with
`scripts/analyze_prereg.py`.
