# dot-Identifiability-v1 1.2.0: reference rows (September 2026)

Rows computed on the built 1.2.0 suite (12,150 episodes, nine structures, shared-noise
counterfactual targets). Every run asserts the per-arm target statistics before scoring.

| File | Contents | Command |
|---|---|---|
| `ident_cpu_effect.json` | The eight CPU baselines, effect-scored (`sign(pred - y_obs)` against `sign(y_true - y_obs)`) | `dotime-eval-reference --suite dot-Identifiability-v1 --version 1.2.0 --dir-target effect` |
| `ident_cpu_level.json` | The same, level-scored (the v1 protocol) | `... --dir-target level` |
| `tabpfn_ident_v1_2.json` | TabPFN adjustment, interventional and observational arms, 60 episodes per structure | `dotime-eval-tabpfn --suite dot-Identifiability-v1 --version 1.2.0 --per-structure 60 --max-total 540 --dir-target effect` (server GPU, tabpfn 7.1.1) |
| `chronos_ident_v1_2.json` | Chronos-2 with the do-value as a known future covariate against the univariate forecast, same subsample | `dotime-eval-chronos ... --per-structure 60 --max-total 540 --dir-target effect` (CPU, chronos-forecasting 2.2.2, transformers 4.57.6) |

## The two observational foundation models on 1.2.0

The submission's Table 4 compared these two models on a 480-episode subsample of the
misaligned 1.0.0 files with the level sign. On 1.2.0, the first 60 episodes of each of the
nine structures (540), both scores side by side:

| Arm | RMSE | Direction accuracy, level (n) | Direction accuracy, effect ± SE (n) |
|---|---|---|---|
| TabPFN, do-value plugged in | 0.543 | 0.726 (446) | 0.660 ± 0.035 (188) |
| TabPFN, observational | 0.561 | 0.720 (446) | 0.596 ± 0.036 (188) |
| Chronos-2, do-value as covariate | 0.487 | 0.717 (446) | 0.702 ± 0.033 (188) |
| Chronos-2, univariate | 0.547 | 0.720 (446) | 0.617 ± 0.035 (188) |

Gaps (do-value arm minus observational arm): TabPFN +0.007 on the
level sign and +0.064 on the effect sign, Chronos-2
-0.002 and +0.085. On the level sign the
submission's finding holds: giving an observational model the do-value buys nothing. On the
effect sign it does buy something, by a margin of one to two standard errors on 188 scored
episodes, in line with the detection-power analysis where every estimator that reads the
do-value beats the naive forecasters. Any interventional-training claim therefore has to be
read against this bar, not against a model that never sees the do-value. The larger samples below tighten these numbers.

### Larger samples

`tabpfn_ident_v1_2_ps150.json` scores the first 150 episodes of each structure (1,350,
server GPU, 17 minutes per arm) and `chronos_ident_v1_2_full.json` the whole suite (12,150,
CPU, 48 minutes for the covariate arm):

| Arm | Episodes | RMSE | Direction accuracy, level (n) | Direction accuracy, effect ± SE (n) |
|---|---|---|---|---|
| TabPFN, do-value plugged in | 1,350 | 0.581 | 0.714 (1,127) | 0.590 ± 0.022 (488) |
| TabPFN, observational | 1,350 | 0.596 | 0.705 (1,127) | 0.531 ± 0.023 (488) |
| Chronos-2, do-value as covariate | 12,150 | 0.486 | 0.727 (10,235) | 0.658 ± 0.007 (4,356) |
| Chronos-2, univariate | 12,150 | 0.598 | 0.696 (10,235) | 0.526 ± 0.008 (4,356) |

Effect-sign gaps with the unpaired standard error of the difference: TabPFN
+0.059 ± 0.032, Chronos-2 +0.132 ± 0.010. Level-sign gaps:
+0.009 and +0.031. An observational forecaster that is handed the do-value as a
known future covariate gains 13 points of effect-sign accuracy over its univariate arm on
the full suite, and its level error drops from 0.598 to 0.486. That is the bar any claim
about interventional training has to clear.

## A dependency pin that matters

`chronos-forecasting` 2.2.2 pins `transformers<5` and `huggingface_hub<1.0`. With
transformers 5.3 the pipeline loads without an error but with randomly initialised weights,
different on every load, and the first run of this evaluator wrote NaN errors. Both
evaluators now stop when an arm has no finite prediction and record `n_nonfinite`.
