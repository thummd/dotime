# dot-RegimeSwitch-v1 version 1.1.0

Released on 2026-11-26 (UTC). The Zenodo record is `24570950` (DOI `10.5281/zenodo.24570950`),
a new version under the concept DOI `10.5281/zenodo.20846073` of the suite. The Hugging Face
mirror is `thummd/dot-RegimeSwitch-v1` at the tag `v1.1.0`. From package 0.3.0 on the registry
serves this version, and `load_benchmark("dot-RegimeSwitch-v1", version="1.0.0")` loads the
first release.

Version 1.1.0 repairs the two defects of version 1.0.0. Its regime-switching mechanisms read no
parents, so every variable of the released suite is independent noise, and its two arms are
independent draws. Version 1.1.0 has mechanisms that read their parents and arms that share
their noise and their regime path, so `y_true - y_obs` is a per-episode counterfactual effect.
The recipe is `scripts/release_config_regime_v1_1.yaml`: the suite seed, the episode seeds, the
regime densities and the episode count of 1.0.0, with six changes.

| Setting | Why |
|---|---|
| `regime_canonical_weights: true` | The mechanisms read their parents |
| `hardening` (recommended values) | With the repair alone 387 of 600 measured episodes diverge, with the hardening 1 of 600 (`../regime_weights/`) |
| `stability_retries: 10` | A diverged episode is resampled, so no episode has a zeroed arm |
| `amplitude_limit: 50` | Each regime is stabilised on its own, which does not bound a system that switches between regimes. Without the limit 8 of 9,999 episodes exceed 50 in magnitude, up to 322.8 (`amplitude_tail.json`) |
| `pair_mode: counterfactual` | With independent arms no reference estimator beat the pre-onset mean (`independent_arms/`) |
| `query_row: window_end` | The stabilised dynamics are contractive, so an effect has decayed by the last step (design pilot below) |

The build goes through the spec kind `regime_configured` of `dotime._build`. The shared noise
and the shared regime path come from the prior's counterfactual generator
(`RegimeSwitchingTemporalSCM.freeze_noise`), which is seeded from the episode seed. With
`record_graph` every episode also records the regime of each step as
`metadata["graph"]["regime_path"]`.

## Files

| File | Content |
|---|---|
| `design_pilot.json` | `scripts/effect_persistence.py` on the first 200 episodes of each regime density, from a clean checkout of `e3b9375`. The header of the release config quotes its numbers, and `tests/test_regime_configured.py` asserts that it does |
| `effect_persistence_full.json` | The same script on all 9,999 episodes, from a clean checkout of `deba4d8` |
| `amplitude_tail.json` | `scripts/amplitude_tail.py` on a build of the same recipe without the amplitude limit (config of `bbccbf2`) |
| `build_validation.json` | `scripts/validate_build.py` on the release build |
| `regime_v1_1_cpu.json` | Reference rows of the release build, from `scripts/reference_rows.py` |
| `regime_v1_1_per_lag.json` | The same rows split by the shortest lag from the treatment to the query |
| `independent_arms/` | The first candidate, which repaired the mechanisms and kept independent arms. Superseded, never released |
| `scripts/` | The four scripts |

The release build is not in the repository (158 MB). It was built from a clean checkout of
`deba4d8` with package version 0.3.0 and 13 workers, has the config hash `e39f9a5a3d61643a` and
the shard digests `28249c5b9e4c27e2cc7a5a0dfc29eebd` and `4c09485d68c27c158d6a26f184ea19d9`.
An earlier build of the same recipe with package 0.2.0 (`7c55f59`) has the same two digests.
The three result files below were written from the release build and are equal, number for
number, to the ones of that earlier build. The public files are this build, byte for byte
(release record below).

```bash
python scripts/build_release.py --config scripts/release_config_regime_v1_1.yaml \
    --output-dir output/regime_v1_1 --workers 12
python results/reference/regime_v1_1/scripts/validate_build.py \
    --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \
    --out results/reference/regime_v1_1/build_validation.json
python results/reference/regime_v1_1/scripts/reference_rows.py \
    --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \
    --out results/reference/regime_v1_1/regime_v1_1_cpu.json \
    --per-lag-out results/reference/regime_v1_1/regime_v1_1_per_lag.json
python results/reference/regime_v1_1/scripts/effect_persistence.py --per-block 200 \
    --workers 10 --out results/reference/regime_v1_1/design_pilot.json
```

`reference_rows.py` scores a local build without touching the cache. It did so before the
registry named the version, and it does so for a rebuild now. On the released files the
console script of the package prints the same rows:

```bash
dotime-eval-reference --suite dot-RegimeSwitch-v1 --version 1.1.0 --out rows.json
```

## What holds for the release build

All of this is asserted by `validate_build.py` on the 9,999 stored episodes.

- **Targets.** The per-arm target statistics pass `dotime.qa.target_qa`, the effect included, in
  the build and again on the stored files.
- **No divergence, no outlier.** No episode has a zeroed arm, and the largest value of any
  trajectory is 48.6. The largest target is 25.8, and the variance of the level target is 2.28
  with its ten largest values and 2.04 without them.
- **Repair.** Every episode records `reads_parents: true`. 31.1% of the variables have a lag-one
  autocorrelation above 0.3 in the observational arm. Without the repair no variable of 600
  measured episodes does (`../regime_weights/`).
- **Shared noise.** The two arms of every episode agree exactly before the onset.
- **Query.** Every query sits at the last step of its intervention window, on the untreated
  variable that the intervention moves most, and `y_true` is the interventional value there. The
  window closes a median 33 steps before the last step.
- **Effect.** The absolute effect on the query is at least 0.1 in 78.8% of the episodes, with a
  median of 0.46. It is exactly zero in 8.3%. By intervention type the share is 82.4% for hard,
  70.8% for soft and 81.7% for time-varying interventions.
- **Graph.** The recorded graph reaches the query from the treatment in 97.8% of the episodes,
  within the same step in 6,002 and at lag one in 3,395. In the 215 episodes where it does not,
  the effect is exactly zero, as it must be under shared noise.
- **Reproducibility.** 300 randomly sampled episodes regenerate exactly from the config and the
  seeds, in one process, although the build used 13 workers. The recorded regime path is part of
  the comparison. 299 of them are first attempts and have the regime graphs and the intervention
  of the version 1.0.0 episode with the same seed. One was resampled.

## What the design pilot shows

`effect_persistence.py` reads the effect off the two arms of 600 episodes (one resampled, none
diverged, arms equal before the onset in all of them) and compares the two query rules.

| | Query at the last step | Query at the window's end |
|---|---|---|
| Share of episodes with an effect of at least 0.1 on the query | 9.2% | 77.8% |
| Median absolute effect | 0.00 | 0.41 |
| Share with an effect of exactly zero | 58.5% | 7.2% |

The window closes a median 30.5 steps before the last step in the pilot, and the contractive
dynamics let the effect decay in that time. The config therefore queries the window's end, as
dot-Wide-v1 does.

On all 9,999 episodes (`effect_persistence_full.json`) the shares are 8.5% at the last step and
78.8% at the window's end. 23 episodes are resampled, for a zeroed arm or a value above the
amplitude limit, and none stays diverged.

## Reference rows

`regime_v1_1_cpu.json` scores the reference estimators of the package on the 9,999 episodes. The
arms share their noise, so direction accuracy scores the sign of the effect, on the 7,876
queries whose absolute effect is at least 0.1. RMSE is on the level, with 95% cluster-bootstrap
intervals.

| Estimator | Effect-sign accuracy | Standard error | Level RMSE | 95% interval |
|---|---|---|---|---|
| Zero | 0.562 | 0.006 | 1.574 | 1.506 to 1.650 |
| Mean | 0.620 | 0.005 | 1.494 | 1.420 to 1.576 |
| AR1 | 0.580 | 0.006 | 1.823 | 1.747 to 1.910 |
| VAR-OLS | 0.606 | 0.006 | 1.560 | 1.490 to 1.638 |
| NaiveOLS | 0.628 | 0.005 | 2.047 | 1.862 to 2.266 |
| BackDoorOLS, IV2SLS | 0.620 | 0.005 | 1.494 | 1.420 to 1.576 |
| Oracle | 1.000 | 0.000 | 0.000 | |

BackDoorOLS and IV2SLS do not apply to these structures and return the mean. By the shortest
lag from the treatment to the query (`regime_v1_1_per_lag.json`):

| Lag | Episodes | Scored | Mean | VAR-OLS | NaiveOLS |
|---|---|---|---|---|---|
| 0 | 6,002 | 5,196 | 0.647 | 0.625 | 0.660 |
| 1 | 3,395 | 2,501 | 0.564 | 0.565 | 0.561 |
| 2 | 339 | 163 | 0.626 | 0.626 | 0.656 |

Two things follow, and both limit what the suite shows today.

- **The naive level is not 0.5.** The pre-onset mean predicts the sign of the effect in 62.0% of
  the scored queries. An intervention replaces the treated variable's own path by an unrelated
  value, which pulls the queried variable back towards its mean more often than away from it.
  dot-Wide-v1 shows a similar level: 0.629 for its best naive forecaster at lag zero
  (`../audit_2026-11/wide_per_lag.json`).
- **The reference estimators do not separate.** NaiveOLS, which reads the intervention value,
  leads the mean by 0.9 points pooled and by 1.3 points at lag zero. Each row has a standard
  error of about 0.6 points, so the lead is not established, and its level RMSE is the worst
  of the table. The suite has an exact per-episode
  effect and an oracle at 1.0, so the room above 0.63 is real, but no estimator of the package
  fills it. A contemporaneous linear regression is a weak model of an effect that builds up over
  a window of at least ten steps in a nonlinear system that switches regimes.

## Release record

1. Package version 0.3.0 (`deba4d8`), then the release build from a clean checkout, with
   `validate_build.py` and `reference_rows.py` on it.
2. The registry named version 1.1.0 with the sentinel `LOCAL` before the upload (`385d9f7`),
   because its description is the text that `scripts/upload_zenodo.py` sends to Zenodo.
   `scripts/release_2026_10.sh` refuses a registry that does not name the version it uploads.
3. Upload on 2026-11-26 (UTC) with `scripts/release_2026_10.sh`, first with `--check`.
   `scripts/zenodo_update.py` added the version to the concept of record `20919599`, which is
   version 1.0.0. `scripts/upload_huggingface.py` moved `main` of the mirror to the new files
   and tagged them `v1.1.0`. It also wrote the dataset card `README.md` into the suite
   directory, so both hosts hold five files. Version 1.0.0 stays at record `20919599` and at
   the tag `v1.0.0`.
4. The registry then got the record id and the mirror. Checks on the public copies with that
   registry, each from an empty cache and from a clean checkout:
   - The five files of record `24570950` and of the tag `v1.1.0` are equal, byte for byte, to
     the release build (`manifest.json` has the md5 `066a821ee81642e5cc65e0de1540c41a`).
   - `load_benchmark("dot-RegimeSwitch-v1")` downloads the tag and returns the 9,999 episodes
     of version 1.1.0. The download of the Zenodo record parses to the same 9,999 episodes.
     `version="1.0.0"` returns the 9,999 episodes of the first release, from files that are
     equal to the copy downloaded before the upload.
   - `dotime-eval-reference` on the downloaded tag gives `regime_v1_1_cpu.json` without its
     `build` block, number for number. `reference_rows.py` on the release build writes both
     result files again, apart from the file name that the per-lag file records for its
     reference.
5. Package 0.3.0 is the first package version whose registry serves 1.1.0.
