# Pre-registration s13: interventional training against the do-information ablation

Registered on 2026-09-30 by the commit that adds this file to the public dotime repository. The
timestamp is that commit and its GitHub push event, saved as push_event.json. No s13 training run
starts before the push. Authors: the DoTime authors.

## 1. Question
The submitted paper claimed in its abstract and in §6.3 to §6.5 that interventional training buys a
measurable direction-accuracy advantage over an observational model of identical capacity. We
withdrew the claim after an audit showed that every observational arm trained on an all-zero target
and that direction accuracy scored the sign of the predicted level instead of the effect. This
document fixes the design, statistics and decision rule of one re-test before any model is trained.
We will report the outcome whichever way it comes out.

The question: under the published training recipe, with correct targets, shared-noise
counterfactual pairs and effect-sign scoring, does a model trained with the intervention visible
predict the sign of the counterfactual effect more often than the same model trained on the same
target with the intervention withheld?

## 2. Prior knowledge
1. On dot-Identifiability-v1 1.1.0 the published interventional checkpoint s9ho_all_causal and the
   retrained observational checkpoint s10ho_all_obs score 0.537 and 0.535 effect-sign accuracy
   (n = 3,194). Both trained on independent-noise pairs.
2. Input-perturbation probes showed that encoders trained with this recipe ignore the pre-onset
   history. The published gap came through the intervention value alone. We therefore expect a
   null result. This re-test concerns the published recipe only. The s12 series is not part of it.
3. Before this commit we ran one smoke test of 50 steps for the int and B arms at seed 42
   (2026-09-30, 11:36 to 11:41 SGT) to check the launcher, the step-zero target check, the
   checkpoint save, the strict loading path and the scoring script on 30 episodes of 1.1.0. The
   two arms logged identical first-batch target statistics. Those predictions were not graded.

## 3. Hypotheses
H1. The interventional arm has a higher pooled effect-sign accuracy than the do-information
ablation (Arm B) on the primary scoring set.
H0. It does not.
We make no directional prediction for the non-identified structure.

## 4. Arms
All arms share recipe, seed, initialisation and training data stream. They differ only in the
training target and in whether the intervention is visible.

| Arm | Extra flags | Target | Intervention in the input |
|---|---|---|---|
| int | none | Y_true | visible |
| B, do-information ablation | --observational-only | Y_true | withheld |
| A, natural continuation | --observational-only --obs-only-target Y_obs | Y_obs | withheld |

Withheld means that intervention_target, intervention_type, intervention_value,
intervention_time_start and intervention_time_end are zeroed in training, in the trainer's
evaluation and at scoring time. With the interpolation mask every arm sees the treatment's factual
value at the onset row. Arm B is the primary control because it shares the target of the int arm.
The int minus A contrast is the published one, and it also changes the target.

## 5. Recipe
The s10 recipe (do-over-time-pfn scripts/run_s10_obs_retrain.sh, lines 42 to 57) with shared-noise
pairs and explicit seeds:
--config configs/server_s9_hardened_oscillatory.yaml --sim-device cpu --total-steps 5000
--batch-size 16 --head-type quantile --target-key Y_true --n-queries 10 --query-mode all_pairs
--causal-mask interpolation --intervention-source positivity_aware --early-stop-patience 5
--num-workers 4 --prefetch 4 --no-tscm-lag
--tscm-structures back_door front_door instrumental_variable --pair-mode counterfactual --seed S

The joint loader queries back_door at offset 0, front_door at offsets 1 to 5 and
instrumental_variable at offsets 0 to 5. Learning rate and warmup come from the configuration file.
The model has 24.9 million parameters. Seeds are 42, 43 and 44. The nine runs are
s13ho_all_{int,B,A}_seed{42,43,44}. We score do_over_time_pfn_last.pt, the weights at the last
step, early stopping included.

Out of scope: structure-matched models, the break-trajectory-mean prior, lagged edges and the
trajectory-length sweep. A positive result would not restore those parts of the withdrawn claim.

## 6. Scoring set
Suite: dot-Identifiability-v1 version 1.2.0. The manifest MD5 digests are recorded in the results.
Scoring may run on a local build whose digests equal the released ones. If 1.2.0 is not released
by 2026-10-04 12:00 Singapore time (SGT), we score on 1.1.0 and drop mediator and the non-identified
control. That fallback is reported as a deviation.

Primary set: every episode whose structure is bi_variate, back_door, confounder_mediator,
front_door, instrumental_variable or mediator, and whose effect satisfies |y_true − y_obs| ≥ 0.1.
Mediator is queried one step after onset in 1.2.0. The factual level y_obs comes from
dotime.evaluation.query_obs_levels, and 0.1 is DIR_ACC_EPS. The set depends only on the suite.

Prediction: dotime.reference.pfn.PFNRef(checkpoint, observational = arm is not int).
An episode is correct when sign(pred − y_obs) equals sign(y_true − y_obs). A prediction equal to
y_obs or not finite counts as incorrect.

## 7. Primary endpoint and decision rule
acc(arm, s) is the fraction of correct episodes of the primary set, pooled over episodes, for seed
s. D_s = acc(int, s) − acc(B, s). The estimate D is the mean of D_42, D_43 and D_44.

Seed interval: D ± t·SD(D_s)/√3, with t = 4.303 for 95% and 2.920 for 90% (two degrees of freedom).
Episode interval: 10,000 paired bootstrap resamples of the primary set (numpy default_rng, seed
20261001). Each resample is applied to both arms and all seeds, and D is recomputed. Percentiles
2.5 and 97.5 give the 95% interval, and 5 and 95 give the 90% interval.

Exactly one label applies:
- SUPPORTED: the lower ends of both 95% intervals are above 0.
- REVERSED: the upper ends of both 95% intervals are below 0.
- EQUIVALENT: neither of the above, and both 90% intervals lie inside (−0.02, +0.02). This is two
  one-sided tests at level 0.05 with 0.02 as the smallest effect of interest.
- INCONCLUSIVE: anything else.
If SUPPORTED, we also state whether D ≥ 0.02. If some structure in S1 differs significantly in the
opposite direction after Holm correction, we add the qualifier MIXED.

## 8. Secondary endpoints
These are reported with intervals and carry no confirmatory claims.
S1. D per structure of the primary set, and pooled over the three training structures. Paired
    bootstrap p-values with Holm correction over six structures.
S2. int minus A, the published contrast.
S3. B minus A, the effect of the training target alone.
S4. Level root mean squared error (RMSE) with an episode-cluster bootstrap, and level-sign accuracy.
S5. Each arm against the CPU estimators of the detection-power analysis
    (results/reference/detection_power_2026-10) on the same episodes: Mean, NaiveOLS, BackDoorOLS,
    IV2SLS, FrontDoorOLS and the clamped structural vector autoregression (do-SVAR), each on the
    structures where it is valid.
S6. Negative control: the non-identified structure of 1.2.0 (U→A, U→Y, A→Y, U hidden). Accuracy per
    arm and D.
S7. Training-prior check, the analogue of §6.5: 40 held-out batches of the training prior per
    structure (loader seed 12345, T = 200, pair_mode counterfactual), effect-sign scored.
S8. Sensitivity: do_over_time_pfn_best.pt, and effect thresholds of 0.05 and 0.2.

## 9. Failures, exclusions and missing data
No episode is excluded beyond Section 6. A run is invalid only if its step-zero target check aborts,
if it crashes or produces non-finite losses before its final save, or if its code digest differs
from Section 10. An invalid run is relaunched once from scratch with identical flags under the
suffix _r2, and every relaunch is reported. No run is relaunched or excluded because of its scores.
If a relaunch fails again, that seed has no pair. The primary analysis then uses the complete int
and B pairs, with degrees of freedom equal to the number of pairs minus one. With fewer than two
complete pairs the primary endpoint is reported as not completed.

## 10. Provenance
Training code: do-over-time-pfn commit 7d7bbe6a19aae99daeb359f914e7868a81a3f8a5, tag prereg-s13, a private repository.
code_manifest.sha256 lists the SHA-256 digest of every tracked code file at that commit.
ctp_manifest.sha256 pins the causal_time_prior copy. Each run writes its command, host, start time
and code commit to results/<tag>/cmd.txt. Scoring uses scripts/score_prereg.py and
scripts/analyze_prereg.py in this folder, together with the dotime package at the commit recorded
in s13_scoring_provenance.json. Neither script changes after registration. A bug fix after
unblinding is a deviation, and both versions are reported. code_manifest.sha256 covers every file
tracked at the training commit. ctp_manifest.sha256 covers the code and configuration files of the
causal_time_prior copy (.py, .yaml, .yml, .toml, .cfg, .sh, .txt and .json, without its bundled
virtual environment, caches, results, outputs and logs). The checkpoints go to the public Hugging Face repository thummd/do-over-time-pfn under s13.

## 11. Freeze and reporting
The primary analysis runs once all six int and B checkpoints exist, and no later than 2026-10-04
12:00 SGT on whatever is complete. results.json and README.md report every endpoint, including the
unfavourable ones. In the summary sentence, L and U are the outer ends of the seed interval and
the episode interval. Summary sentence by label:
- EQUIVALENT or INCONCLUSIVE: "Interventional training changes effect-sign accuracy by D, with a 95%
  confidence interval from L to U, so the withdrawal stands."
- SUPPORTED: the same numbers, then "This supports a narrower claim than the withdrawn one: joint
  models on one prior without lagged edges."
- REVERSED: the same numbers, then "The ablation is better, so the withdrawal stands."
- Not completed: "The re-test did not complete by the registered deadline."

## 12. Design sensitivity
Each seed scores about 3,500 episodes. If the arms disagree on a fraction d of episodes, the
within-seed standard error of D_s is about √(d/3500), which is 0.004 at d = 0.05 and 0.008 at
d = 0.2. With a seed standard deviation of 0.008, the 95% seed interval has a half-width of 0.02.
A gap of the published size (0.08 to 0.10) would be SUPPORTED. EQUIVALENT needs the three seeds to
agree within about 0.005.

## 13. Deviations
DEVIATIONS.md records every departure from this document with a timestamp and a reason.
