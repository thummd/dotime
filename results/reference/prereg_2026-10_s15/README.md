# Pre-registered test s15: results

Registered in `PREREG.md` (commit 3106898, pushed 2026-10-17 04:51:24 UTC, `push_event.json`) before any run started. The nine runs trained from 2026-10-17 12:52 to 2026-10-19 03:41 SGT, and the registered scorer ran on 2026-10-19 from 13:36 to 17:06 SGT at the registering commit. This file reports every endpoint. Nothing here was chosen after seeing the numbers: the scripts in `scripts/` are byte-identical to the registered ones, and the one procedural deviation is in `DEVIATIONS.md`.

## Outcome

**Labels: SUPPORTED (P1), BELOW_BAR (P2), INCONCLUSIVE (P3).** Under the standardized training dose, a Do-Over-Time-PFN with the intervention visible predicts the effect sign more often than the same model with it withheld, by +0.135 with a 95% confidence interval from +0.120 to +0.151. It remains −0.102 below an unadjusted in-context regression on the same episodes, with a 95% confidence interval from −0.115 to −0.089. Covering the benchmark's dose range in training changes the effect-sign accuracy against s14 by +0.009, with a 95% confidence interval from −0.021 to +0.039, which is inconclusive. This supports the narrower claim that the model uses the do-value. It does not restore the withdrawn claim.

| Quantity | Value |
|---|---|
| Primary set | 3,528 episodes of dot-Identifiability-v1 1.2.0 with an effect of at least 0.1 on the six primary structures |
| Effect-sign accuracy, interventional arm | 0.6617 (seeds: 0.6638, 0.6559, 0.6655) |
| Effect-sign accuracy, Arm B (same target, intervention withheld) | 0.5266 (seeds: 0.5278, 0.5249, 0.5272) |
| Effect-sign accuracy, Arm A (natural continuation) | 0.5266 (seeds: 0.5255, 0.5281, 0.5264) |
| D, mean over seeds of int minus B | +0.1351 (seeds: +0.1361, +0.1310, +0.1383) |
| Seed interval of D, 95% and 90% | [+0.1257, +0.1445] and [+0.1287, +0.1415] |
| Paired episode bootstrap of D, 95% and 90% | [+0.1195, +0.1505] and [+0.1223, +0.1478] |
| NaiveOLS on the same episodes | 0.7636 |
| E, mean over seeds of int minus NaiveOLS | −0.1019 (seeds: −0.0998, −0.1077, −0.0981) |
| Seed interval of E, 95% | [−0.1146, −0.0891] |
| Paired episode bootstrap of E, 95% | [−0.1151, −0.0886] |
| s14 interventional arm on the same episodes | 0.6529 (seeds: 0.6635, 0.6522, 0.6429) |
| F, mean over seeds of int minus s14 int | +0.0089 (seeds: +0.0003, +0.0037, +0.0227) |
| Seed interval of F, 95% and 90% | [−0.0211, +0.0389] and [−0.0115, +0.0292] |
| Paired episode bootstrap of F, 95% and 90% | [+0.0006, +0.0174] and [+0.0018, +0.0160] |

Both 95% intervals of D lie above zero, the registered condition for SUPPORTED, and D exceeds the 0.02 margin (yes). No structure differs significantly in the opposite direction after the Holm correction, so there is no MIXED qualifier. Both 95% intervals of E lie below −0.02, the registered condition for BELOW_BAR. For F the episode interval lies above zero but the seed interval does not, so F is not SUPPORTED, and the 90% seed interval is not inside the equivalence margin, so F is INCONCLUSIVE; F is below 0.02.

**Against the expectation stated in advance** (PREREG.md Section 2, item 6). P1 held. P2 moved towards the bar by 0.009, from s14's −0.111 to −0.102, and no structure reaches AT_BAR under the registered rule (S9): back_door sits at −0.015 with both lower interval ends just outside the 0.02 margin, instrumental_variable at −0.034. P3 was expected SUPPORTED and is INCONCLUSIVE. The strata (S13) place the gain where the expectation put it, in the stratum at or below 3 pre-onset standard deviations (+0.028, SUPPORTED) and directionally above 12 (+0.052, INCONCLUSIVE), with nothing between 3 and 12, but its size is a quarter of what closing the pooled gap would need.

**What the test establishes.** The point mass of s14's training dose cost about 0.03 of accuracy on the quarter of the primary set whose doses lie at or below 3 standard deviations, and nothing elsewhere. Covering the benchmark's dose range does not close the gap to the unadjusted regression. That gap grows with the dose magnitude even when training covers it: by stratum, E is −0.041, −0.084, −0.146 and −0.168, the interventional arm's accuracy is 0.668, 0.674, 0.655 and 0.617, and NaiveOLS's is 0.709, 0.758, 0.801 and 0.785. The model reads large doses less well than the regression does, under a prior that now draws them. Training support for the dose is therefore not the cause of the residual gap, and the mediated structures (S1, S9) stay where s14 left them.

## Secondary endpoints

**S1, per structure** (int minus B, paired episode bootstrap, Holm over six structures), with the mean accuracy of each arm over the three seeds, NaiveOLS and the s14 interventional arm on the episodes of the primary set of that structure:

| Structure | n | int | B | A | NaiveOLS | s14 int | D | 95% bootstrap | Holm |
|---|---|---|---|---|---|---|---|---|---|
| bi_variate | 846 | 0.551 | 0.534 | 0.533 | 0.849 | 0.563 | +0.0169 | [−0.0189, +0.0516] | no |
| back_door | 789 | 0.774 | 0.526 | 0.511 | 0.788 | 0.752 | +0.2480 | [+0.2167, +0.2797] | yes |
| confounder_mediator | 351 | 0.553 | 0.515 | 0.517 | 0.672 | 0.563 | +0.0380 | [+0.0009, +0.0750] | no |
| front_door | 371 | 0.618 | 0.535 | 0.538 | 0.698 | 0.608 | +0.0827 | [+0.0449, +0.1213] | yes |
| instrumental_variable | 837 | 0.778 | 0.528 | 0.538 | 0.812 | 0.764 | +0.2497 | [+0.2206, +0.2796] | yes |
| mediator | 334 | 0.549 | 0.508 | 0.515 | 0.536 | 0.511 | +0.0409 | [−0.0060, +0.0868] | no |

Pooled over the three training structures: D = +0.2180, [+0.1990, +0.2372].

**S2, int minus A** (the published contrast): +0.1351, 95% bootstrap [+0.1197, +0.1507]. **S3, B minus A** (the target alone): +0.0000, [−0.0089, +0.0089].

**S4, level error** (RMSE against the counterfactual level with an episode bootstrap, and the sign of the level):

| Arm, seed | RMSE | 95% bootstrap | Level-sign accuracy |
|---|---|---|---|
| int 42, 43, 44 | 0.599, 0.665, 0.700 | [0.582, 0.617], [0.646, 0.684], [0.678, 0.722] | 0.691, 0.686, 0.683 |
| B 42, 43, 44 | 0.617, 0.617, 0.622 | | 0.676, 0.680, 0.675 |
| A 42, 43, 44 | 0.607, 0.607, 0.608 | | 0.685, 0.690, 0.687 |

**S5, estimators of the detection-power analysis on the same primary episodes** (effect-sign accuracy), with the int arm's contrast against each, reported without a label: NaiveOLS 0.764 (E = −0.102, the registered bar), do-SVAR 0.762 (−0.100), GraphRouter 0.724 (−0.062), BackDoorOLS 0.612 (+0.050), FrontDoorOLS 0.573 (+0.089), IV2SLS 0.552 (+0.109), Mean 0.534 (+0.128), AR1 0.526, Zero 0.525, VAR-OLS 0.522, Oracle 1.000.

**S6, negative control** (`bow_graph`, non-identified, n = 828): int 0.798, 0.803, 0.768; B 0.551, 0.556, 0.559; A 0.533, 0.527, 0.540; D = +0.2347, seed interval [+0.1793, +0.2901], bootstrap [+0.2041, +0.2657]. As in s14, the model reads the association of a non-identified structure as the unadjusted regression does.

**S7, training prior** (the pilot's gate script on its fixed training-prior set, run on the int checkpoints on 2026-10-19 from 18:25 to 18:43 SGT, after the registered scoring had started; `s7_training_prior.json`, records in `training_prior_reads/`): pooled effect-sign accuracy 0.796, 0.794, 0.794 (s14: 0.789, 0.783, 0.789; NaiveOLS 0.775 and do-SVAR 0.802 on the same rows), back_door 0.925, 0.930, 0.925, front_door 0.633, 0.622, 0.628, instrumental_variable 0.709, 0.700, 0.704; flip rate 0.274, 0.271, 0.279; anchor ratio 1.008, 1.006, 1.001; slope-sign agreement 0.783, 0.763, 0.758. The gate loader keeps the pilot's 3 sd clip, so these read the checkpoints at a dose the s15 training never pinned; the s15 checkpoints sit where the s14 checkpoints did.

**S8, sensitivity:** on `do_over_time_pfn_best.pt` (`results_best_pt.json`, `s15_best_predictions.parquet`, scored 2026-10-19 13:36 to 18:25 SGT on cuda:3): int 0.6644, 0.6604, 0.6596; D = +0.1353, seed interval [+0.1293, +0.1413], bootstrap [+0.1199, +0.1506], SUPPORTED; E = −0.1021, [−0.1085, −0.0957] and [−0.1153, −0.0888], BELOW_BAR. With the effect threshold at 0.05 (`results_eps_0.05.json`): D = +0.117, [+0.103, +0.130], SUPPORTED; E = −0.089, [−0.101, −0.077], BELOW_BAR. At 0.2 (`results_eps_0.2.json`): D = +0.157, [+0.138, +0.175], SUPPORTED; E = −0.122, [−0.137, −0.107], BELOW_BAR.

**S9, the bar per structure** (int minus NaiveOLS, Holm over six structures; the label applies the rule of Section 7b to the two intervals):

| Structure | E | Seed interval 95% | Bootstrap 95% | Label | Holm |
|---|---|---|---|---|---|
| bi_variate | −0.2975 | [−0.3651, −0.2299] | [−0.3298, −0.2644] | BELOW_BAR | yes |
| back_door | −0.0148 | [−0.0369, +0.0073] | [−0.0393, +0.0093] | BELOW_BAR | no |
| confounder_mediator | −0.1197 | [−0.2268, −0.0126] | [−0.1614, −0.0788] | BELOW_BAR | yes |
| front_door | −0.0800 | [−0.1130, −0.0469] | [−0.1168, −0.0422] | BELOW_BAR | yes |
| instrumental_variable | −0.0342 | [−0.0610, −0.0075] | [−0.0542, −0.0151] | BELOW_BAR | yes |
| mediator | +0.0130 | [+0.0087, +0.0173] | [−0.0289, +0.0569] | BELOW_BAR | no |

**S10, mechanism probes on the primary set** (the int checkpoints; s14's were 0.474, 0.572 and 0.540 for the flip rate): mirror flip rate 0.462, 0.565, 0.546; median anchor error ratio 1.011, 1.015, 1.039; slope-sign agreement 0.711, 0.695, 0.688.

**S11, in-support subset** (845 primary episodes whose do-value lies within 3 pre-onset standard deviations of the treatment): D = +0.1195, [+0.0923, +0.1471], SUPPORTED; E = −0.0406, [−0.0631, −0.0189], BELOW_BAR (NaiveOLS 0.709 on the subset). s14 had D = +0.086 and E = −0.069 on the same subset.

**S12, context** (not the same episodes): the published Chronos-2 result on this suite (`results/reference/v1_2/chronos_ident_v1_2.json`, 540 episodes, effect-sign on 188 scorable ones): with the do-value as a covariate 0.702, without it 0.617.

**S13, dose strata** of the primary set by |v − mu_A| / sd_A, with the mean accuracy of the int arm, Arm B, NaiveOLS and the s14 int arm, and the three contrasts with the labels of their sections (no multiplicity correction, no claim):

| Stratum (sd) | n | int | B | NaiveOLS | s14 int | D | E | F |
|---|---|---|---|---|---|---|---|---|
| at most 3 | 845 | 0.668 | 0.549 | 0.709 | 0.640 | +0.1195 [+0.0923, +0.1471] SUPPORTED | −0.0406 [−0.0631, −0.0189] BELOW_BAR | +0.0284 [+0.0049, +0.0519] SUPPORTED |
| 3 to 6 | 1,184 | 0.674 | 0.515 | 0.758 | 0.679 | +0.1591 [+0.1320, +0.1858] SUPPORTED | −0.0845 [−0.1067, −0.0622] BELOW_BAR | −0.0053 [−0.0232, +0.0125] EQUIVALENT |
| 6 to 12 | 1,243 | 0.655 | 0.529 | 0.801 | 0.655 | +0.1260 [+0.0992, +0.1523] SUPPORTED | −0.1464 [−0.1706, −0.1220] BELOW_BAR | +0.0003 [−0.0479, +0.0484] INCONCLUSIVE |
| above 12 | 256 | 0.617 | 0.497 | 0.785 | 0.565 | +0.1198 [+0.0651, +0.1745] SUPPORTED | −0.1680 [−0.2253, −0.1107] BELOW_BAR | +0.0521 [−0.0031, +0.1073] INCONCLUSIVE |

The intervals shown are the outer ends of the seed interval and the bootstrap interval of each contrast (`strata_and_series.json` holds both).

**S14, the series contrast per structure** (int minus s14 int, Holm over six structures): bi_variate −0.0114 [−0.0256, +0.0028]; back_door +0.0211 [+0.0051, +0.0368]; confounder_mediator −0.0104 [−0.0380, +0.0161]; front_door +0.0099 [−0.0198, +0.0404]; instrumental_variable +0.0139 [−0.0044, +0.0323]; mediator +0.0379 [+0.0040, +0.0729]. None survives the Holm correction.

## How the runs ended

Nine runs of 20,000 steps, no relaunch, no non-finite loss, every step-zero target check passed. Launched from `/data/dennis/s15/dopfn` (git archive of the training commit) by `scripts/run_s15_train.sh`, primary pairs first, on aidf-svr-gpu04 under shared load (13 to 15 hours per run). SHA-256 of `do_over_time_pfn_last.pt` with the start time of each run (SGT):

| Run | Start | SHA-256 |
|---|---|---|
| s15ho_all_int_seed42 | 2026-10-17 12:52:03 | 155eecc8b4d6c757b1bc9c12b4e310d195048a68aa48561359c018133488ed3f |
| s15ho_all_B_seed42 | 2026-10-17 12:52:23 | 071138d7f53e0003dbabf8a2dfc08ddeace401bd86e9eebb78885b82dfbb8b7b |
| s15ho_all_int_seed43 | 2026-10-17 12:52:43 | 856696cad4c9e22bb1cfe2c84595c60f152a384c0050b640e7053eb300818e54 |
| s15ho_all_B_seed43 | 2026-10-18 01:41:09 | 8445f3e84595b13331a20754592f3ceaacebad10bc495c53903e63f35293ee46 |
| s15ho_all_int_seed44 | 2026-10-18 03:17:30 | 3984f3ad380e49da744dea69b833157f85b4d92777d532f6833d050de5130f42 |
| s15ho_all_B_seed44 | 2026-10-18 03:34:51 | 7700d6a39e7ff8d3bb24e421b29adeae42e5b823a3178da40b33c47d6136a945 |
| s15ho_all_A_seed42 | 2026-10-18 14:41:16 | a4f17a2afdeb79af692a2c7ab91dc81d71452dd584c49c1d00380a92e96117e1 |
| s15ho_all_A_seed43 | 2026-10-18 16:12:37 | 1bb047db1115ac6f9210e0565696dc5d8b51d6681e41faa6045b42449716b562 |
| s15ho_all_A_seed44 | 2026-10-18 17:06:58 | c7a52676845ffcbba739ce4a62fb7b5835667915e074383037a2086bf3f97fe6 |

## Provenance

Training code: do-over-time-pfn commit 877a88d2e61f2fe7dc4ea677551e85f7cb82674c (tag `s15-train`), `code_manifest.sha256` (196 tracked files without the historical checkpoints), `ctp_manifest.sha256` re-verified on the training host on 2026-10-17 (132 files unchanged). Every run's `cmd.txt` records that commit. Scoring: `scripts/score_prereg.py` of this folder on the public package at the registering commit 3106898, deployed by git archive to `/data/dennis/s15/dotime_public`, on aidf-svr-gpu04 cuda:1 in the isolated venv of s14, 2026-10-19 13:36:29 to 17:06:18 SGT (`s15_score.log`, `s15_scoring_provenance.json`, which records the package commit, torch 2.10.0+cu128 and the suite's shard digests). Analysis: `scripts/analyze_prereg.py` and `scripts/strata_and_series.py` of this folder, unchanged since registration, with the frozen NaiveOLS predictions of `results/reference/detection_power_2026-10/` and the registered s14 predictions. The `checkpoint` field of each gate read under `training_prior_reads/` carries the Hugging Face Hub location of the identical file, as the repository's test suite requires. The nine checkpoints are to be uploaded to the public Hugging Face repository thummd/do-over-time-pfn under `s15`.
