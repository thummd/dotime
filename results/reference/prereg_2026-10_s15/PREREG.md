# Pre-registration s15: the standardized training dose against the do-information ablation, the unadjusted in-context regression and s14

Registered on 2026-10-17 by the commit that adds this file to the public dotime repository. The
timestamp is that commit and its GitHub push event, saved as push_event.json. No s15 training run
starts before the push. The one 50-step smoke run made before it is disclosed in Section 2 and is
not scored. Authors: the DoTime authors.

## 1. Question
The confirmatory test s14 (results/reference/prereg_2026-11) found that a redesigned
Do-Over-Time-PFN uses the do-value (P1 SUPPORTED, D = +0.126, 95% intervals from +0.103 to +0.150)
and that it stays below an unadjusted in-context regression that reads the do-value (P2
BELOW_BAR, E = −0.111, 95% intervals from −0.137 to −0.085), on every primary structure except
mediator, including the two direct-effect structures. The withdrawn claim of the submitted paper
stays withdrawn whatever this document finds.

A diagnostic after s14 (Section 2) found that the training dose of s13 and s14 was a point mass:
the raw do-value draw is in raw units while the treatment's pre-onset standard deviation is of
order 0.01 to 0.2, so the positivity clip binds on 92 to 94 percent of draws and pins the dose at
exactly 3 pre-onset standard deviations (s13 and the pilot's gate loader) or exactly 6 (the pilot
and s14), whereas the benchmark's primary do-values have a median of 5.26 and a 5th to 95th
percentile range of 0.96 to 12.89. s15 changes exactly one thing in the s14 recipe: the dose is
drawn in pre-onset standard deviations of the treatment, signed log-uniform on 0.5 to 20, so that
every dose magnitude the benchmark exhibits has non-zero training density. Three questions, with
one registered label each:

P1. Does the model trained with the intervention visible predict the sign of the counterfactual
effect more often than the same model trained on the same target with the intervention withheld?

P2. Where does the interventional model stand against an unadjusted in-context regression that
reads the do-value (NaiveOLS) on the same episodes?

P3. Does the interventional model predict the effect sign more often than the registered s14
interventional model, seed by seed, on the same episodes?

P1 repeats s14's narrower claim. P2 says whether the dose change closes the gap to the simplest
estimator that reads the do-value. P3 attributes any change to the dose draw, because the s14
runs share everything else with the s15 runs, including seeds, initialisation and the training
data stream up to the dose. None of the three restores the withdrawn claim.

## 2. Prior knowledge
1. s14 (registered in results/reference/prereg_2026-11, 3,528 primary episodes of
   dot-Identifiability-v1 1.2.0, three seeds): int 0.6529, B 0.5265, A 0.5261, NaiveOLS 0.7636.
   Per structure, int: bi_variate 0.563, back_door 0.752, confounder_mediator 0.563, front_door
   0.608, instrumental_variable 0.764, mediator 0.511. On the in-support subset (845 episodes
   whose do-value lies within 3 pre-onset standard deviations) E = −0.069 against −0.111 pooled.
2. Diagnostic D9 (do-over-time-pfn results/diag_2026-10, 2026-10-16). Drawing the same 640
   training-prior episodes per structure under a clip too wide to bind, the registered 3 sd clip
   and the 6 sd clip, with the do-value standardized by the sampler's own pre-onset mean and
   standard deviation: the raw dose has a median of 20 to 34 and a 95th percentile of 350 to 700
   pre-onset standard deviations; the 3 sd clip changes 0.923 to 0.944 of the draws and the 6 sd
   clip 0.833 to 0.901; after either clip the median and the 95th percentile of the dose equal the
   clip width to within 2e-5. Diagnostic D1 of the same folder had reported a clipped fraction of
   zero because it standardized by the loader's normalization statistics, under which 3 pre-onset
   standard deviations read as 2.63 to 2.75. The s14 registration's statement that the 3 sd clip
   "never binds on the training prior" (its Section 2, item 3, via the pilot) rests on D1 and is
   therefore wrong; s14's "in-support" subset lies below its own 6 sd training dose.
3. Diagnostic D10 (2026-10-17, public package). The dose law of the 3,528 primary episodes in
   pre-onset standard deviations of the treatment: quantiles 1, 5, 10, 25, 50, 75, 90, 95 and 99
   percent equal 0.17, 0.96, 1.59, 3.09, 5.26, 8.15, 11.11, 12.89 and 17.73; shares beyond 3, 5,
   10 and 20 standard deviations 0.760, 0.531, 0.148 and 0.004; positive doses 0.479. Per
   structure the median runs from 4.52 (instrumental_variable) to 5.96 (front_door).
4. Diagnostic D11 (2026-10-17). The realized dose of the standardized source on the training
   prior, same 640 episodes per structure, range 0.5 to 20: median 2.95 to 3.02, 5th percentile
   0.58 to 0.62, 95th percentile 16.5 to 17.0, share beyond 3 sd 0.492 to 0.503, positive 0.500 to
   0.508; no point mass. The range 1 to 20 gives a median of 4.23 to 4.31 and a share beyond 3 sd
   of 0.619 to 0.623. We register 0.5 to 20 because it covers the suite's 1st to 99th percentile;
   its centre sits about 2 standard deviations below the suite's, and both ranges put more mass
   above 15 standard deviations than the suite (2.7 percent). The choice was made before any s15
   run existed.
5. Precedent against the direction of our expectation: on the real chemical reactor, widening
   the training value distribution made transfer worse (do-over-time-pfn RESUBMISSION.md, round
   R16 to R17), and the market study found that wider impact-scale training compressed the
   readout further. Neither concerns this benchmark, and both are recorded here so that a
   negative result is read against them.
6. Expectation stated in advance. P1 SUPPORTED, as in s14. P2 moves towards the bar: pooled E
   above s14's −0.111, with S9 giving AT_BAR or ABOVE_BAR on back_door and instrumental_variable
   and BELOW_BAR on the mediated structures (front_door, confounder_mediator, mediator), whose
   failure the pilot traced to the readout and not to the dose. P3 SUPPORTED. S13: the gain over
   s14 is largest in the strata farthest from s14's training dose of 6 standard deviations, the
   strata at or below 3 and above 12, and smallest in the stratum from 3 to 6. If P3 is REVERSED,
   the dose law and not its support is implicated, and the 1 to 20 range of item 4 is the next
   candidate; no such follow-up is registered here.
7. Smoke test before this commit: one run of 50 steps of the int arm at seed 42 on the training
   host (2026-10-17, 12:45 to 12:46 SGT, cuda:3, 41 seconds of training) under the separate run
   root /data/dennis/s15/smoke, to check the launcher, the standardized source on the host, the
   step-zero target check (passed: Y_true non-zero fraction 0.707, Y_obs 0.746, effect identity
   exact) and the checkpoint files. Its weights were deleted, its log and command record were
   kept, and nothing was scored.
8. Read of the public s14 predictions before this commit. The smoke test of
   scripts/strata_and_series.py (Section 10) ran on the registered s14 predictions against
   themselves and, as a by-product, reported s14's D and E by dose stratum: in the strata at most
   3, 3 to 6, 6 to 12 and above 12 pre-onset standard deviations (845, 1,184, 1,243 and 256
   episodes), D is +0.086, +0.166, +0.129 and +0.062, all SUPPORTED, and E is −0.069, −0.079,
   −0.147 and −0.220, all BELOW_BAR. s14's gap to NaiveOLS therefore widens with the distance
   from its training dose of 6. These numbers involve no s15 run; they were seen after item 6 was
   written and are consistent with it, and item 6 was not changed.

## 3. Hypotheses
H1. The interventional arm has a higher pooled effect-sign accuracy than the do-information
ablation (Arm B) on the primary scoring set.
H0. It does not.
H1'. The interventional arm has a higher pooled effect-sign accuracy than the registered s14
interventional arm on the primary scoring set, paired by seed.
H0'. It does not.
P2 carries no hypothesis; it receives one of three labels (Section 7b).

## 4. Arms
As in s14. All arms share recipe, seed, initialisation and training data stream. They differ only
in the training target and in whether the intervention is visible.

| Arm | Extra flags | Target | Intervention in the input |
|---|---|---|---|
| int | none | Y_true | visible |
| B, do-information ablation | --observational-only | Y_true | withheld |
| A, natural continuation | --observational-only --obs-only-target Y_obs | Y_obs | withheld |

Withheld means that intervention_target, intervention_type, intervention_value,
intervention_time_start and intervention_time_end are zeroed in training, in the trainer's
evaluation and at scoring time. Every arm receives the treatment's factual onset value as a scalar
on the query token (a_ref_norm); it is not an intervention field and is not withheld. The learned
role vector of the treated variable marks column 0 in the withheld arms, an artefact of the
withheld protocol shared by B and A, stated here rather than changed. Arm B is the primary control
because it shares the target of the int arm. The withheld arms are drawn under the same
standardized dose; the dose reaches them only through the interventional target.

## 5. Recipe
s14 (pilot row 9 of do-over-time-pfn PILOT_S14.md at 20,000 steps) with the dose source as the
only change (do-over-time-pfn scripts/run_s15_train.sh at the training commit of Section 10):
--config configs/server_s9_hardened_oscillatory.yaml --sim-device cpu --total-steps 20000
--batch-size 16 --head-type quantile --target-key Y_true --n-queries 10 --query-mode all_pairs
--causal-mask interpolation --intervention-source standardized --dose-range 0.5 20
--early-stop-patience 0 --num-workers 4 --prefetch 4 --no-tscm-lag
--tscm-structures back_door front_door instrumental_variable --pair-mode counterfactual
--lr 2e-5 --token-lags 4 --readout mean_last --pos-init-std 1.0 --horizon-embed scalar
--center-intervention-value --query-a-ref
--stat-tokens ols_path --role-embed --value-aux-weight 0.5 --value-aux-anchor --seed S

Against s14 the flags --intervention-source positivity_aware --positivity-clip-sd 6
--eval-positivity-clip-sd 3 are replaced by --intervention-source standardized --dose-range 0.5 20.
The trainer's evaluation set mirrors the training source, so no clip applies to it either; the
evaluation losses in the run logs are therefore not comparable with s14's. The standardized source
draws, per episode, v = mu + s m sigma with (mu, sigma) the mean and the unbiased standard
deviation of the treatment's pre-onset window (the same statistics the clip used), m log-uniform on
[0.5, 20] and s a fair sign; the two extra draws come after the raw draw, so the mechanisms, the
noise and the onsets of every episode are those s14 drew from the same seed. The joint loader
queries back_door at offset 0, front_door at offsets 1 to 5 and instrumental_variable at offsets 0
to 5, one query per trajectory. Warmup and weight decay come from the configuration file. The model
has 26.25 million parameters (embedding width 512, six encoder layers). The auxiliary loss applies
to the int arm only. Seeds are 42, 43 and 44. The nine runs are s15ho_all_{int,B,A}_seed{42,43,44},
launched after the push with the primary pairs first. We score do_over_time_pfn_last.pt, the
weights at step 20,000.

Out of scope: the published recipe and model, the bi_variate structure (absent from the training
sampler), structure-matched models, lagged edges, the trajectory-length sweep, other dose ranges,
and transfer to real data. A positive result supports joint models of this redesign trained under
the standardized dose on the hardened oscillatory prior, scored by the effect sign at the
registered offsets on dot-Identifiability-v1 1.2.0, and nothing wider.

## 6. Scoring set
As in s14. Suite: dot-Identifiability-v1 version 1.2.0 as released (Zenodo record 23095083); the
manifest MD5 digests are recorded in the results. Primary set: every episode whose structure is
bi_variate, back_door, confounder_mediator, front_door, instrumental_variable or mediator and whose
effect satisfies |y_true − y_obs| ≥ 0.1, the same 3,528 episodes as s13 and s14. The factual level
y_obs comes from dotime.evaluation.query_obs_levels, and 0.1 is DIR_ACC_EPS.

Prediction: dotime.reference.pfn.PFNRef(checkpoint, observational = arm is not int) of the public
dotime package at the commit of Section 10, which builds the batch under the input convention
recorded in the checkpoint (the centred do-value and the onset value; the checkpoint's data block
also records the dose source and range, which the scorer does not read) and refuses a checkpoint
whose weights do not match its configuration. An episode is correct when sign(pred − y_obs) equals
sign(y_true − y_obs). A prediction equal to y_obs or not finite counts as incorrect. For the int
arms the scorer also records the prediction at the mirrored do-value 2 a_ref − v and at the anchor
a_ref, and for every episode the treatment's pre-onset mean and standard deviation; these feed
S10, S11 and S13.

## 7. Primary endpoint and decision rule
As in s14. acc(arm, s) is the fraction of correct episodes of the primary set, pooled over
episodes, for seed s. D_s = acc(int, s) − acc(B, s). The estimate D is the mean of D_42, D_43 and
D_44.

Seed interval: D ± t·SD(D_s)/√3, with t = 4.303 for 95% and 2.920 for 90% (two degrees of freedom).
Episode interval: 10,000 paired bootstrap resamples of the primary set (numpy default_rng, seed
20261001). Each resample is applied to both arms and all seeds, and D is recomputed. Percentiles
2.5 and 97.5 give the 95% interval, and 5 and 95 give the 90% interval.

Exactly one label applies:
- SUPPORTED: the lower ends of both 95% intervals are above 0.
- REVERSED: the upper ends of both 95% intervals are below 0.
- EQUIVALENT: neither of the above, and both 90% intervals lie inside (−0.02, +0.02).
- INCONCLUSIVE: anything else.
If SUPPORTED, we also state whether D ≥ 0.02. If some structure in S1 differs significantly in the
opposite direction after Holm correction, we add the qualifier MIXED.

## 7b. Co-primary bar
As in s14. E_s = acc(int, s) − acc(NaiveOLS) on the same primary episodes, where the NaiveOLS
predictions are the rows of the frozen detection-power file
results/reference/detection_power_2026-10/ident_v1_2_predictions.parquet (SHA-256
a83cb4d81b05aaac78c315016c6ac1d15d0102e65e02a25af1706c175722f65a, estimator NaiveOLS), graded by
the same rule. E is the mean over the three seeds. The seed interval uses the spread of E_s over the
int seeds; the episode interval is the same paired bootstrap, resampling the estimator's and the
model's correctness together. Exactly one label applies:
- ABOVE_BAR: the lower ends of both 95% intervals are above 0.
- AT_BAR: not ABOVE_BAR, and the lower ends of both 95% intervals are at or above −0.02.
- BELOW_BAR: anything else.
The same contrast against do-SVAR, GraphRouter, BackDoorOLS, FrontDoorOLS, IV2SLS and the pre-onset
mean is reported without a label.

## 7c. Co-primary series contrast
F_s = acc(int, s) − acc(int_s14, s) on the same primary episodes, where acc(int_s14, s) is graded
from the registered s14 predictions results/reference/prereg_2026-11/s14_predictions.parquet
(SHA-256 66714e01a78aff64668e50e1bfc98bb43b1fd677e267020a3dc1bc22d50d7848), arm int, seed s. F is
the mean over the three seeds, with the seed interval and the paired episode bootstrap of Section 7
(the resample is applied to both series and all seeds). The labels of Section 7 apply to F with the
same thresholds: SUPPORTED, REVERSED, EQUIVALENT, INCONCLUSIVE. If SUPPORTED, we also state whether
F ≥ 0.02.

## 8. Secondary endpoints
These are reported with intervals and carry no confirmatory claims.
S1. D per structure of the primary set, and pooled over the three training structures. Paired
    bootstrap p-values with Holm correction over six structures.
S2. int minus A, the published contrast.
S3. B minus A, the effect of the training target alone.
S4. Level root mean squared error (RMSE) with an episode-cluster bootstrap, and level-sign accuracy.
S5. Each CPU estimator of the detection-power analysis on the same primary episodes.
S6. Negative control: the non-identified structure bow_graph. Accuracy per arm and D.
S7. Training-prior check: the pilot's gate script on each int checkpoint (the fixed training-prior
    set of PILOT_S14.md), reported for comparability with the pilot and s14. Its loader keeps the
    pilot's 3 sd clip, so it reads the checkpoints at a dose the s15 training never pinned.
S8. Sensitivity: do_over_time_pfn_best.pt, and effect thresholds of 0.05 and 0.2.
S9. E per structure of the primary set, with Holm correction over six structures.
S10. Mechanism probes of each int checkpoint on the primary set: the mirror flip rate, the median
     anchor error ratio and the slope-sign agreement, as in s14.
S11. P1 and P2 on the in-support subset of s14's definition: primary episodes whose do-value lies
     within 3 pre-onset standard deviations of the treatment's pre-onset mean. Kept for
     comparability with s14; under the standardized dose no subset is "in support" by training.
S12. Context: the published Chronos-2 result on this suite with and without the do-value as a
     covariate (results/reference/v1_2/chronos_ident_v1_2.json), pooled numbers only.
S13. Dose strata. The primary set is split by the dose magnitude |v − mu_A| / sd_A, with mu_A and
     sd_A the treatment's pre-onset mean and standard deviation as the scorer records them, into
     four strata: at most 3, above 3 and at most 6, above 6 and at most 12, and above 12. In each
     stratum D (Section 7), E (Section 7b) and F (Section 7c) are reported with both interval
     families and the labels of their sections; the strata are reported without multiplicity
     correction and carry no claim. The boundaries follow the two training doses of s13 and s14 (3
     and 6) and the suite's 90th percentile (11.1, rounded to 12).
S14. F per structure of the primary set, with Holm correction over six structures.

## 9. Failures, exclusions and missing data
No episode is excluded beyond Section 6. A run is invalid only if its step-zero target check aborts,
if it crashes or produces non-finite losses before its final save, or if its code digest differs
from Section 10. An invalid run is relaunched once from scratch with identical flags under the
suffix _r2, and every relaunch is reported. No run is relaunched or excluded because of its scores.
If a relaunch fails again, that seed has no pair. The primary analysis then uses the complete int
and B pairs, with degrees of freedom equal to the number of pairs minus one. With fewer than two
complete pairs the primary endpoint is reported as not completed. P3 uses the complete int seeds
of s15 paired with the same seeds of s14; with fewer than two it is reported as not completed.

## 10. Provenance
Training code: do-over-time-pfn commit 877a88d2e61f2fe7dc4ea677551e85f7cb82674c (tag s15-train), a
private repository. It differs
from the s14 training commit 2a37d3f in the standardized dose source (sampler, extended prior,
loader, trainer and command line), diagnostics D9 to D11 with their records, the s15 launcher and
one test file; the model, the loss, the pilot's representations and every other flag of the recipe
are unchanged. code_manifest.sha256 lists the SHA-256 digest of every tracked file at that commit
except the directory of historical checkpoints the repository tracks, which plays no part in
training. ctp_manifest.sha256 is the s13 manifest of the causal_time_prior copy; every one of its
132 files was re-hashed on the training host on 2026-10-17 and none had changed. Each run writes
its command, host, start time and code commit to results/<tag>/cmd.txt; the deployed tree carries
the commit in a COMMIT file.

Scoring uses scripts/score_prereg.py and scripts/analyze_prereg.py in this folder, together with
the public dotime package at the commit of this registration, and scripts/strata_and_series.py
for S13, S14 and the series contrast of Section 7c. The two former scripts are the registered s14
scripts with four changes: the series prefix and the output prefix default to s15, the docstrings
name s15, and the scorer's fallback that reads the package commit from the COMMIT file of a git
archive deployment looks four directory levels above the script instead of three, which repairs
deviation 1 of s14. The third script is new and was smoke-tested on the s14 predictions, where the
series contrast of a file against itself is zero by construction. Neither script changes after
registration. A bug fix after unblinding is a deviation, and both versions are reported. The
checkpoints go to the public Hugging Face repository thummd/do-over-time-pfn under s15, and the
results README pins the nine do_over_time_pfn_last.pt files by SHA-256 with their start times.

## 11. Freeze and reporting
The primary analysis runs once all six int and B checkpoints exist, and no later than 2026-10-24
12:00 SGT on whatever is complete. results.json and README.md report every endpoint, including the
unfavourable ones. The summary sentence is composed of five parts in this order: the P1 sentence
of the label, the P2 sentence of the label, the P3 sentence of the label, the claim sentence, and
the closing sentence. L and U are the outer ends of the seed interval and the episode interval of
the contrast in question.

P1 sentences:
- SUPPORTED: "Under the standardized training dose, a Do-Over-Time-PFN with the intervention
  visible predicts the effect sign more often than the same model with it withheld, by D with a 95%
  confidence interval from L to U."
- REVERSED: "Under the standardized training dose, with the intervention visible the model predicts
  the effect sign less often than the same model with it withheld, by D with a 95% confidence
  interval from L to U."
- EQUIVALENT: "Under the standardized training dose, interventional training changes the model's
  effect-sign accuracy by D, with a 95% confidence interval from L to U, inside the registered
  equivalence margin."
- INCONCLUSIVE: "Under the standardized training dose, interventional training changes the model's
  effect-sign accuracy by D, with a 95% confidence interval from L to U, which is inconclusive."

P2 sentences:
- ABOVE_BAR: "It is E above an unadjusted in-context regression on the same episodes, with a 95%
  confidence interval from L to U."
- AT_BAR: "It is within the registered margin of an unadjusted in-context regression on the same
  episodes (E, 95% confidence interval from L to U)."
- BELOW_BAR: "It remains E below an unadjusted in-context regression on the same episodes, with a
  95% confidence interval from L to U."

P3 sentences:
- SUPPORTED: "Covering the benchmark's dose range in training raises the effect-sign accuracy over
  s14 by F, with a 95% confidence interval from L to U."
- REVERSED: "Covering the benchmark's dose range in training lowers the effect-sign accuracy
  against s14 by F, with a 95% confidence interval from L to U."
- EQUIVALENT: "Covering the benchmark's dose range in training changes the effect-sign accuracy
  against s14 by F, with a 95% confidence interval from L to U, inside the registered equivalence
  margin."
- INCONCLUSIVE: "Covering the benchmark's dose range in training changes the effect-sign accuracy
  against s14 by F, with a 95% confidence interval from L to U, which is inconclusive."

Claim sentence: P1 SUPPORTED, "This supports the narrower claim that the model uses the
do-value."; P1 REVERSED or EQUIVALENT, "This does not support the narrower claim that the model
uses the do-value."; P1 INCONCLUSIVE, "This neither supports nor rejects the narrower claim that
the model uses the do-value." Closing sentence, in every cell: "It does not restore the withdrawn
claim." Not completed: "The test did not complete by the registered deadline. It does not restore
the withdrawn claim."

## 12. Design sensitivity
As in s14 for D and E: each seed scores about 3,500 episodes, the within-seed standard error of
D_s is about √(d/3500) for a disagreement fraction d, and a seed standard deviation of 0.008 gives
a 95% seed half-width of 0.02. For F the s14 and s15 int arms share seeds and initialisation, so
the per-seed differences are expected to be less variable than D_s; a gain of 0.02 is at the edge
of what three seeds can tell from zero, and the training-prior gain that would close the pooled gap
to NaiveOLS (0.11) would be SUPPORTED by a wide margin.

## 13. Deviations
DEVIATIONS.md records every departure from this document with a timestamp and a reason.
