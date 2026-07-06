# Changelog

All notable changes to `dotime` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
