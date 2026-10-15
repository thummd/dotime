# Deviations from PREREG.md

Every departure from the pre-registration is listed here with a timestamp in Singapore time (SGT)
and a reason, in the order it happened.

1. **2026-10-15 15:40 SGT, empty package-commit field in the scoring provenance.** `s14_scoring_provenance.json`
   records `dotime_commit` as an empty string. The registered `score_prereg.py` reads the commit from `git rev-parse`
   or, in a `git archive` deployment, from a `COMMIT` file at the repository root, but it looks for that file three
   directory levels above the script instead of four (`Path(__file__).resolve().parents[3]` where the root is
   `parents[4]`). The deployment did carry the file: `s14_score.log`, written by the launcher before the scorer started,
   prints the commit it read from it, `74769ff5facadfc588a45eb52d0e1ae632e07f16`, which is the registering commit
   whose archive was deployed (Section 10). The script is left as registered; the predictions and every other
   provenance field are unaffected.
