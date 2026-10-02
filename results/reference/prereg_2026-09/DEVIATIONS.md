# Deviations from PREREG.md

Every departure from the pre-registration is listed here with a timestamp in Singapore time (SGT)
and a reason, in the order it happened. The analysis itself (scoring set, endpoint, decision rule,
scripts) followed the document as written. The two entries below are procedural.

1. **2026-09-30 12:17 SGT, relaunch helper.** The tagged launcher `run_s13_prereg.sh` has no code
   path for the single relaunch that Section 9 allows. The relaunch of `s13ho_all_A_seed43` was
   started by a separate helper on the training host (`relaunch_r2.sh`) that rebuilds the same
   command with the suffix `_r2` and waits for 12 GB of free GPU memory. The flags are identical
   to the original run's, as the two `cmd.txt` files show. The helper is not part of the code
   manifest. Reason: the first attempt died of a CUDA out-of-memory error at step 0.
2. **2026-10-01 15:30 SGT, scoring before the release.** Section 6 allows scoring on a local
   build whose digests equal the released ones. The scoring ran before dot-Identifiability-v1
   1.2.0 was uploaded, on the build that is being released. The digests in
   `s13_scoring_provenance.json` are the ones to compare with the released files.
3. **2026-10-02, scoring commit renamed.** The public history after the registering commit
   was rewritten to remove an AI co-author trailer from one commit message. The commit the
   scoring ran on, `e8b95779862730e1473770136c8c02e3e29d7591` as recorded in
   `s13_scoring_provenance.json`, is now `e640e02` on `main`. Both commits have the same tree,
   `5aac54c604a3d1bb95e7e29b35c49eec8bd2e9f1`, so the scoring code is byte-identical. The registering commit 05ae974 and its
   push record are unchanged.
