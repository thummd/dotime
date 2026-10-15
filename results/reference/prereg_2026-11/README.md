# Pre-registered re-test s14: results

Registered in `PREREG.md` (commit 74769ff, pushed 2026-10-15 06:44:54 UTC, `push_event.json`) after the nine runs had been trained and before any of them touched the benchmark; the registration pins the nine checkpoints by digest. This file reports every endpoint. Nothing here was chosen after seeing the numbers: the scripts in `scripts/` are byte-identical to the registered ones, and the one procedural deviation is in `DEVIATIONS.md`.

## Outcome

**Labels: SUPPORTED (P1), BELOW_BAR (P2).** A redesigned Do-Over-Time-PFN with the intervention visible predicts the effect sign more often than the same model with it withheld, by +0.126 with a 95% confidence interval from +0.103 to +0.150. It remains -0.111 below an unadjusted in-context regression on the same episodes, with a 95% confidence interval from -0.136 to -0.085. This supports the narrower claim that the model uses the do-value. It does not restore the withdrawn claim.

| Quantity | Value |
|---|---|
| Primary set | 3,528 episodes of dot-Identifiability-v1 1.2.0 with an effect of at least 0.1 on the six primary structures |
| Effect-sign accuracy, interventional arm | 0.6529 (seeds: 0.6635, 0.6522, 0.6429) |
| Effect-sign accuracy, Arm B (same target, intervention withheld) | 0.5265 (seeds: 0.5261, 0.5320, 0.5213) |
| Effect-sign accuracy, Arm A (natural continuation) | 0.5261 (seeds: 0.5275, 0.5224, 0.5283) |
| D, mean over seeds of int minus B | +0.1264 (seeds: +0.1375, +0.1202, +0.1216) |
| Seed interval, 95% and 90% | [+0.1026, +0.1503] and [+0.1102, +0.1426] |
| Paired episode bootstrap, 95% and 90% | [+0.1108, +0.1417] and [+0.1135, +0.1392] |
| NaiveOLS on the same episodes | 0.7636 |
| E, mean over seeds of int minus NaiveOLS | -0.1107 (seeds: -0.1001, -0.1114, -0.1207) |
| Seed interval of E, 95% | [-0.1365, -0.0850] |
| Paired episode bootstrap of E, 95% | [-0.1245, -0.0973] |

Both 95% intervals of D lie above zero, the registered condition for SUPPORTED, and D exceeds the 0.02 margin (yes). Both 95% intervals of E lie below −0.02, the registered condition for BELOW_BAR. No structure differs significantly in the opposite direction after the Holm correction, so there is no MIXED qualifier.

Compared with s13 (int 0.538, B 0.544 under the published recipe), the redesigned model reads the do-value: the int arm gains 0.13 over its matched ablation on every seed, while the B and A arms stay at the naive level. It does not reach the simplest estimator that reads the do-value: the unadjusted in-context regression scores 0.764 on the same episodes, 0.11 above the int arm.

## Secondary endpoints

**S1, per structure** (int minus B, paired episode bootstrap, Holm over six structures), with the mean accuracy of each arm over the three seeds on the episodes of the primary set of that structure:

| Structure | n | int | B | A | D | 95% interval | p | Holm significant |
|---|---|---|---|---|---|---|---|---|
| `bi_variate` | 846 | 0.563 | 0.541 | 0.532 | +0.0217 | [-0.0134, +0.0575] | 0.243 | no |
| `back_door` | 789 | 0.752 | 0.518 | 0.511 | +0.2341 | [+0.2015, +0.2662] | 0.000 | yes |
| `confounder_mediator` | 351 | 0.563 | 0.500 | 0.520 | +0.0627 | [+0.0190, +0.1064] | 0.006 | yes |
| `front_door` | 371 | 0.608 | 0.531 | 0.534 | +0.0773 | [+0.0395, +0.1150] | 0.000 | yes |
| `instrumental_variable` | 837 | 0.764 | 0.527 | 0.540 | +0.2370 | [+0.2087, +0.2660] | 0.000 | yes |
| `mediator` | 334 | 0.511 | 0.529 | 0.509 | -0.0180 | [-0.0629, +0.0259] | 0.430 | no |
| pooled training structures | 1997 | | | | +0.2061 | [+0.1871, +0.2253] | 0.000 | |

The gain is largest on the two training structures with a direct effect of the treatment on the outcome, back_door and instrumental_variable, present but smaller on the mediated structures front_door and confounder_mediator, and absent on bi_variate, which has no third variable and was not among the training structures, and on mediator, whose effect is lagged by one step.

**S2, int minus A** (the published contrast): +0.1268, 95% bootstrap [+0.1110, +0.1426]. **S3, B minus A** (the target alone): +0.0004, [-0.0089, +0.0096].

**S4, level error** (RMSE against the counterfactual level with an episode bootstrap, and the sign of the level):

| Run | RMSE | 95% interval | Level-sign accuracy |
|---|---|---|---|
| int_42 | 0.5882 | [0.5694, 0.6083] | 0.6963 |
| int_43 | 0.6443 | [0.6245, 0.6636] | 0.6618 |
| int_44 | 0.6245 | [0.6068, 0.6428] | 0.6761 |
| B_42 | 0.6241 | [0.6016, 0.6471] | 0.6685 |
| B_43 | 0.6156 | [0.5929, 0.6397] | 0.6896 |
| B_44 | 0.6268 | [0.6058, 0.6484] | 0.6589 |
| A_42 | 0.6063 | [0.5838, 0.6291] | 0.6854 |
| A_43 | 0.6082 | [0.5854, 0.6319] | 0.6898 |
| A_44 | 0.6090 | [0.5873, 0.6318] | 0.6854 |

**S5, estimators of the detection-power analysis on the same primary episodes** (effect-sign accuracy):

| Estimator | Accuracy |
|---|---|
| Oracle | 1.0000 |
| NaiveOLS | 0.7636 |
| do-SVAR | 0.7616 |
| GraphRouter | 0.7239 |
| BackDoorOLS | 0.6120 |
| FrontDoorOLS | 0.5726 |
| IV2SLS | 0.5524 |
| Mean | 0.5340 |
| AR1 | 0.5258 |
| Zero | 0.5247 |
| VAR-OLS | 0.5224 |

**S6, negative control** (`bow_graph`, non-identified, n = 828): int 0.806, 0.779, 0.810; B 0.550, 0.539, 0.547; A 0.521, 0.525, 0.531; D = +0.2532, [+0.2242, +0.2818]. The effect of this structure is not identified, and the model reads its association as the unadjusted regression does (0.81 in the detection-power analysis).

**S7, training prior** (the pilot's gate script on its fixed training-prior set, run on the int checkpoints before the registration push; `s7_training_prior.json`):

| Run | Pooled effect sign | back_door | front_door | instrumental_variable | Flip rate | Anchor ratio | Slope agreement |
|---|---|---|---|---|---|---|---|
| s14ho_all_int_seed42 | 0.789 | 0.915 | 0.633 | 0.700 | 0.277 | 1.019 | 0.778 |
| s14ho_all_int_seed43 | 0.783 | 0.920 | 0.628 | 0.674 | 0.255 | 1.023 | 0.768 |
| s14ho_all_int_seed44 | 0.789 | 0.930 | 0.622 | 0.683 | 0.269 | 1.010 | 0.769 |

On that set NaiveOLS scores 0.775 and do-SVAR 0.802, so the gap to the unadjusted regression opens on the benchmark (S11 below).

**S8, sensitivity:**

| Analysis | n | D | 95% bootstrap | P1 label | E | 95% bootstrap | P2 label |
|---|---|---|---|---|---|---|---|
| effect threshold 0.05 | 4,347 | +0.1103 | [+0.0968, +0.1238] | SUPPORTED | -0.0952 | [-0.1074, -0.0829] | BELOW_BAR |
| effect threshold 0.2 | 2,600 | +0.1514 | [+0.1328, +0.1697] | SUPPORTED | -0.1290 | [-0.1450, -0.1129] | BELOW_BAR |
| best-validation checkpoints | 3,528 | +0.1389 | [+0.1241, +0.1535] | SUPPORTED | -0.1007 | [-0.1136, -0.0882] | BELOW_BAR |

**S9, the bar per structure** (int minus NaiveOLS, Holm over six structures):

| Structure | NaiveOLS | E | 95% interval | Holm significant |
|---|---|---|---|---|
| `bi_variate` | 0.849 | -0.2861 | [-0.3203, -0.2506] | yes |
| `back_door` | 0.788 | -0.0359 | [-0.0613, -0.0106] | yes |
| `confounder_mediator` | 0.672 | -0.1092 | [-0.1529, -0.0674] | yes |
| `front_door` | 0.698 | -0.0898 | [-0.1249, -0.0548] | yes |
| `instrumental_variable` | 0.812 | -0.0482 | [-0.0693, -0.0275] | yes |
| `mediator` | 0.536 | -0.0250 | [-0.0689, +0.0200] | no |

**S10, mechanism probes on the primary set** (the int checkpoints; s13 checkpoints had flip rates of 0.02 to 0.03 and slope agreement near 0.47):

| Run | Mirror flip rate | Anchor error ratio (median) | Slope-sign agreement |
|---|---|---|---|
| int_42 | 0.474 | 1.210 | 0.688 |
| int_43 | 0.572 | 1.088 | 0.680 |
| int_44 | 0.540 | 1.009 | 0.681 |

**S11, in-support subset** (845 primary episodes whose do-value lies within 3 pre-onset standard deviations of the treatment): D = +0.0860, [+0.0588, +0.1136], SUPPORTED; E = -0.0690, [-0.0931, -0.0454], BELOW_BAR (NaiveOLS 0.709 on the subset).

**S12, context** (not the same episodes): the published Chronos-2 result on this suite (`results/reference/v1_2/chronos_ident_v1_2.json`, 540 episodes, effect-sign on 188 scorable ones): with the do-value as a covariate 0.702, without it 0.617.

## How the runs ended

All nine runs completed 20,000 steps without a relaunch (launcher log in the private repository, start times in `PREREG.md` Section 10). Every run passed its step-zero target check. The scored checkpoints are the ones pinned at registration; their digests in `s14_scoring_provenance.json` equal the registered table.

## Provenance

Scoring ran on aidf-svr-gpu04 (cuda:1, torch 2.10.0+cu128) with the public package at the registering commit (dotime 0.2.0; the commit is recorded in `s14_score.log` because the provenance field is empty, see `DEVIATIONS.md`), on dot-Identifiability-v1 1.2.0 (12,150 episodes; shard MD5 digests in the provenance file). `s14_predictions.parquet` holds every prediction, including the mirrored and anchored predictions of the int arms. `results.json`, `results_eps_0.05.json`, `results_eps_0.2.json` and `results_best_pt.json` are the outputs of `scripts/analyze_prereg.py`. The training code is do-over-time-pfn commit 2a37d3f (tag `s14-train`, `code_manifest.sha256`), and the checkpoints go to the Hugging Face repository thummd/do-over-time-pfn under `s14/`.

