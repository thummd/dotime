"""Reference evaluation harness — reproduces the released reference results.

These are the evaluators behind the published reference tables. They ship with
the package so the numbers can be reproduced from a plain ``pip install``:

.. code-block:: bash

    dotime-eval-submission --suite dot-Identifiability-v1 --baseline VAR-OLS \\
        --out submission.json                      # leaderboard submission JSON
    dotime-eval-reference --suite dot-Identifiability-v1 --out ident.json
    dotime-eval-pfn       --suite dot-Identifiability-v1 \\
        --ckpt-int <s9ho_all_causal.pt> --ckpt-obs <s9ho_all_obs.pt>
    dotime-eval-tabpfn    --suite dot-Identifiability-v1   # needs [baselines]
    dotime-eval-chronos   --suite dot-Identifiability-v1   # needs [baselines]

``submission`` is the reproducible leaderboard submission path for your own
model; ``reference_table`` covers the classical baselines and ``Oracle`` on a full
suite; ``pfn`` evaluates the Do-Over-Time-PFN in interventional vs.
observational mode; ``tabpfn`` and ``chronos`` are the foundation-model
comparisons. Released per-cell outputs live under ``results/reference/`` in the
project repository.

Submodules are intentionally *not* imported here: ``tabpfn`` and ``chronos``
need optional dependencies (the ``baselines`` / ``models`` extras), so importing
this package must stay cheap and dependency-free. Import the submodule you need.
"""

__all__ = ["chronos", "pfn", "reference_table", "submission", "tabpfn"]
