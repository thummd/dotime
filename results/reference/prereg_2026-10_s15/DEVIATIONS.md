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

2. **2026-10-23, commit messages rewritten.** The three s15 commits on the public main, the registration
   (`3106898b3330f76b662286e29eeb4b2d72614061`), its push record (`0b7833a1d8c1b945a89bc4118b54795ed979c420`) and the
   results (`bbe4745`), carried a `Co-Authored-By` trailer naming an AI system, which this project never records as an
   author. The history was rewritten to remove the trailer and nothing else: the twins `2c7d062`, `d508f4b` and
   `d288b64` have the same trees (`bf7afecb`, `cb390c79`, `893d503f`), authors and timestamps as the originals.
   `push_event.json` is GitHub's record of the original push and names the original hash; the trees being identical,
   it dates exactly the content now on main. `s15_scoring_provenance.json` records the original registering commit
   as the package commit for the same reason and is left as scored. The README states both hashes.
