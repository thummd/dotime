# Deviations from PREREG.md

Every departure from the pre-registration is listed here with a timestamp in Singapore time (SGT)
and a reason, in the order it happened.

1. **2026-10-28 16:35 SGT, registered fallback used.** The first invocation of `scripts/run.sh`
   failed gate 3 on last-digit arithmetic differences of this environment in eight RMSE cells of
   registered estimators (README.md, "Execution record"); nothing of the extensions was involved.
   PREREG.md Section 4 registers the fallback `--limit-per-structure 1350` for exactly this case,
   and it was run next. Listed here so that the record of the run is complete: the failed gate
   record and its log are kept, and the fallback changes no number of the registered analysis.

2. **2026-10-28, after scoring, style-only edit of a registered script.** The registering commit had
   re-sorted `__all__` of `../detection_power_2026-10/scripts/estimators.py` in plain ASCII order
   when it added `DEFAULT_ESTIMATORS` and `EXTENSIONS`; the repository's lint rule (RUF022) wants the
   isort order, and the list was reordered accordingly. No behaviour, number or output changes;
   `code_manifest.sha256` keeps the digests of the registering commit, as PREREG.md says.

3. **2026-11-27 SGT, after publication, edits that change no number.** The registering commit was
   formatted with an older ruff than the one CI pins (0.16.7). (a) The pinned formatter wraps one
   expression of `tests/test_baselines.py`, and the file is reformatted accordingly. (b) One sentence
   in the docstring of `LaggedBackDoorOLSBaseline` (`src/dotime/baselines.py`) now states the outcome
   of L1, so that the API documentation does not suggest the opposite. (c) `scripts/grade_lagged.py`
   fails the same format check and is left as registered: `pyproject.toml` exempts it from ruff by
   path. `code_manifest.sha256` keeps the digests of the registering commit.
