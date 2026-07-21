# Changelog

All notable changes to `dotime` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `dotime-diagnose-stationarity` (`dotime.reference.stationarity`): measures the
  reduced-form companion spectral radius of sampled generic SCMs directly from
  their weight matrices (no simulation) and compares burn-in moments on
  non-diverged episodes. Backs the scope paragraph of the paper's convergence
  appendix; released output in `results/reference/stationarity_diagnostic.json`.

### Fixed
- `docs/troubleshoot.md` claimed the SCM divergence rate was "< 1% on the
  released suites". The actual zeroed fraction is 28.7% on `dot-Generic-100k`
  and 4.6% on `dot-Identifiability-v1`, as disclosed in the paper and asserted
  by `tests/test_build_release.py`. The page now states the real rates and
  points at `--stability-retries` for a divergence-free rebuild.
- `stability_retries` is now honoured by the `regime` generator, not only
  `generic`. `scripts/build_release.py --stability-retries` documented both,
  but the regime branch had no retry loop and its episode specs did not carry
  the setting, so a hardened rebuild would have silently left
  `dot-RegimeSwitch-v1` unhardened. (No observable change at present: the
  regime generator's measured divergence rate is 0%.)

## [0.1.2] - 2026-07-20

### Added
- Reference evaluation harness now ships with the package as
  `dotime.reference`, exposed as four console scripts so the published
  reference tables reproduce from a plain `pip install` (they previously lived
  in `scripts/`, which is not part of the wheel):
  `dotime-eval-reference`, `dotime-eval-pfn`, `dotime-eval-tabpfn`,
  `dotime-eval-chronos`. The leaderboard submission path moved the same way:
  `scripts/eval_submission.py` is now `dotime-eval-submission`. The TabPFN and Chronos evaluators need the
  `baselines` extra; their imports are deferred so the package stays
  importable without it.
- `Results` now reports the uncertainty on direction accuracy: `dir_acc_se`
  (binomial standard error, exact because the suites score one query per
  episode) and `dir_n_valid` (queries with a scoreable sign) appear in
  `pooled`, in `per_structure`, in `to_dict()`, and in `summary()`.
- `scripts/build_release.py --stability-retries R` overrides the per-suite
  `stability_retries`, deterministically resampling numerically diverged
  episodes instead of releasing them as all-zero. The override is folded into
  the recorded `config_hash`, so a hardened rebuild is never mistaken for the
  frozen v1 suites.

### Changed
- Renamed the `rct_no_confounding` identification structure to `bi_variate`.
  The old string still loads: `TSCMStructure("rct_no_confounding")` resolves to
  `BI_VARIATE`, and episodes from the released v1 suites are relabeled at read
  time.
- Removed the review-policy sentence from the leaderboard submission docs.

### Added
- Docs: "Evaluating on Your Own Data" page covering the Do-Over-Time-PFN
  inference path (`Episode` construction, the `DoOverTimePFN` baseline, and
  `evaluate` over a custom suite).

## [0.1.1] - 2026-06-26

### Changed
- Removed forward-looking references to an unpublished paper from the package
  description, documentation, and dataset metadata.

## [0.1.0] - 2026-06-26

### Added
- Initial `src/` package layout consolidating the DoTime base prior
  (from the TSALM workshop code), the Do-Over-Time-PFN extended prior and
  dataloaders, and the continuous-time / fine-grid generation.
- Reimplemented the small `Do-PFN-prior` sampling/graph/mechanism surface as
  first-class, attributed modules (no git submodule required).
- First public release.
