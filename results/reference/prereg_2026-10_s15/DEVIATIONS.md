# Deviations from PREREG.md

Every departure from the pre-registration is listed here with a timestamp in Singapore time (SGT)
and a reason, in the order it happened.

1. **2026-10-19 17:20 SGT, summary sentences composed by hand.** `scripts/analyze_prereg.py` is the
   registered s14 script with its series defaults changed, and its `summary` field composes the s14
   sentence templates ("A redesigned Do-Over-Time-PFN ..."). PREREG.md Section 11 registers
   different P1 sentences ("Under the standardized training dose, ...") and a P3 sentence that the
   script does not compose. The README therefore composes the Section 11 sentences by hand from the
   labels and the numbers in `results.json` and `strata_and_series.json`; the script is left as
   registered and its own sentence stays in `results.json` for the record. No number is affected.
