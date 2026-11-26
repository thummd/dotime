# dot-RegimeSwitch-v1 version 1.1.0 (prepared, not released)

Status on 2026-11-27: a candidate is built and validated. Nothing is minted, and the registry
still serves version 1.0.0. The release needs a decision, which the last two sections lay out.

Version 1.1.0 repairs version 1.0.0, whose regime-switching mechanisms read no parents, so that
every variable of the released suite is independent noise. The recipe is
`scripts/release_config_regime_v1_1.yaml`: the suite seed, the episode seeds, the regime
densities and the episode count of 1.0.0, with four changes.

| Setting | Why |
|---|---|
| `regime_canonical_weights: true` | The mechanisms read their parents |
| `hardening` (recommended values) | With the repair alone 387 of 600 measured episodes diverge, with the hardening 1 of 600 (`../regime_weights/`) |
| `stability_retries: 10` | A diverged episode is resampled, so no episode has a zeroed arm |
| `query_row: window_end` | The stabilised dynamics are contractive, so an effect has decayed by the last step (design pilot below) |

The build goes through the spec kind `regime_configured` of `dotime._build`, which also seeds
numpy's global generator for every episode. That generator draws the regime path of each arm.
Version 1.0.0 never seeded it, which was harmless only because its mechanisms ignored the path.

## Files

| File | Content |
|---|---|
| `design_pilot.json` | `scripts/effect_persistence.py` on the first 200 episodes of each regime density, from a clean checkout of `d349ed2`. The header of the release config quotes its numbers, and `tests/test_regime_configured.py` asserts that it does |
| `build_validation.json` | `scripts/validate_build.py` on the candidate build |
| `regime_v1_1_cpu.json` | Reference rows of the candidate, from `scripts/reference_rows.py` |
| `scripts/` | The three scripts |

The candidate is not in the repository (157 MB). It was built from a clean checkout of `917081b`
with 13 workers, has the config hash `338d087ab99e29ef` and the shard digests
`27102dd4dfc6427f3a007f05281bb51b` and `7dfb513c3f6d54ec164fcbefa3e1c326`. An earlier build
from `7af7082` with 12 workers has the same two digests. Its manifest names package version
0.2.0, which does not contain the build path, so a release build has to follow a version bump.

```bash
python scripts/build_release.py --config scripts/release_config_regime_v1_1.yaml \
    --output-dir output/regime_v1_1 --workers 12
python results/reference/regime_v1_1/scripts/validate_build.py \
    --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \
    --out results/reference/regime_v1_1/build_validation.json
python results/reference/regime_v1_1/scripts/reference_rows.py \
    --suite-dir output/regime_v1_1/<run>/dot-RegimeSwitch-v1-1.1.0 \
    --out results/reference/regime_v1_1/regime_v1_1_cpu.json
python results/reference/regime_v1_1/scripts/effect_persistence.py --per-block 200 \
    --workers 10 --out results/reference/regime_v1_1/design_pilot.json
```

## What holds for the candidate

All of this is asserted by `validate_build.py` on the 9,999 stored episodes.

- **Targets.** The per-arm target statistics pass `dotime.qa.target_qa`, in the build and again
  on the stored files: no non-finite and no zero target in either arm, pooled and per density.
- **No divergence.** No episode has a zeroed arm and none is flagged as diverged.
- **Repair.** Every episode records `reads_parents: true`. 31.2% of the variables have a lag-one
  autocorrelation above 0.3 in the observational arm. Without the repair no variable of 600
  measured episodes does (`../regime_weights/`).
- **Query.** Every query sits at the last step of its intervention window, on an untreated
  variable, and `y_true` is the interventional value there. The window closes a median 33 steps
  before the last step. The recorded graph reaches the query from the treatment in 96.7% of the
  episodes, within the same step in 4,037 and at lag one in 4,899.
- **Reproducibility.** 300 randomly sampled episodes regenerate exactly from the config and the
  seeds, in one process, although the build used 13 workers. 299 of them are first attempts and
  have the regime graphs and the intervention of the version 1.0.0 episode with the same seed.
  One was resampled.

## What the design pilot shows

The two arms of an episode are independent draws, so their gap mixes the effect of the
intervention with noise. `effect_persistence.py` measures the effect itself: it rebuilds the SCM
and the intervention and simulates the interventional arm twice under common random numbers, so
that both runs see the same noise and the same regime path. A soft intervention is contrasted
with the observational run. A hard or time-varying one is contrasted with a hard intervention at
the treated variable's pre-onset mean, because both runs then skip the same noise draws.

On 600 episodes (one resampled, none diverged), with the effect threshold 0.1 of the benchmark:

| Share of episodes in which the intervention moves ... by at least 0.1 | At the last step | At the window's end |
|---|---|---|
| the queried variable | 6.0% | 47.7% |
| some untreated variable | 9.5% | 78.3% |

The median contrast of the queried variable at the window's end is 0.085, and that of the most
affected variable is 0.43. The queried variable is the one with the largest gap between the two
independent arms, and it is the most affected variable in 29.0% of the episodes.

## What limits the suite

No reference estimator beats the pre-onset mean on the candidate (`regime_v1_1_cpu.json`, level
scoring, 9,999 episodes, 95% cluster-bootstrap intervals).

| Estimator | RMSE | 95% interval | Sign of the level |
|---|---|---|---|
| Zero | 2.350 | 2.094 to 2.761 | |
| Mean | 2.277 | 2.037 to 2.663 | 0.605 |
| AR1 | 2.851 | 2.611 to 3.215 | 0.567 |
| VAR-OLS | 2.456 | 2.219 to 2.832 | 0.583 |
| NaiveOLS | 3.067 | 2.764 to 3.437 | 0.589 |
| BackDoorOLS, IV2SLS | 2.277 | 2.037 to 2.663 | 0.605 |

BackDoorOLS and IV2SLS do not apply to these structures and return the mean. The estimator that
reads the intervention value, NaiveOLS, is the worst. On version 1.0.0 the same table has the
mean at chance on the sign of the level (0.498, `../regime.json`). The repair therefore gives the
levels structure, but it does not make the suite separate estimators.

The reason is the pairing, not the repair. With independent arms the observational history tells
a model which SCM it faces and nothing about the noise of the interventional arm. The best
possible prediction is the interventional mean, and the intervention shifts that mean by little
against the spread of the target (variance 5.4). A per-episode effect needs arms that share their
noise, which is what the six counterfactual suites have.

## The decision before a release

1. **Release this candidate as 1.1.0.** It is a faithful repair and every claim above holds. It
   stays a suite of interventional twins on which the effect sign is not scored, and on which
   the pre-onset mean is the best reference.
2. **Build a shared-noise version instead.** The regime simulator would pre-draw the noise and
   the regime path of a pair and reuse them in both arms, as `TemporalSCM.freeze_noise` does for
   the other SCMs. The arms would agree before the onset, `y_true - y_obs` would be a
   per-episode effect and the query could be the most affected variable. This is a different
   suite and needs a simulator option, tests, a new config and a new build. It is not built.
3. **Release neither** and keep the disclosure of the paper.

For either release: bump the package version first, because the manifest records it, and
rebuild from a clean checkout. Then run `validate_build.py` and `reference_rows.py` on the
release build and `effect_persistence.py` on the whole suite. The registry description is the
text that `scripts/upload_zenodo.py` sends to Zenodo, so the registry has to name the new
version before the upload. After the mint come the record id in the registry, the config hash
in `tests/test_release_config_provenance.py`, the fingerprints of the new version and the
paper.
