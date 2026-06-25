# CausalTimePrior — PyPI Release Handover Plan

This document hands off the `causaltimeprior` packaging effort to Claude Code (or any
local agent) for the work that has to run against the real repository. Everything
referenced here exists as drafted scaffolding; the job is to wire it to the actual
code in `causal_time_prior/`, `dotime/`, and `continuous-time-causal-pfn/`, then ship.

**Source repos.** Three first-party trees plus submodules:
- `causal_time_prior/` — the base prior (TSALM workshop code).
- `dotime/` — Do-Over-Time-PFN: model, training, baselines, discrete-time extensions.
- `continuous-time-causal-pfn/` — the FMSD (ICML 2026) line: **fine-grid / continuous-time
  implementations**. This is the source of truth for the package's continuous-time module
  (intervention windows, fine-grid sampling) — it is *not* in the chat's project knowledge,
  so read it directly in Claude Code.
- submodules `Do-PFN-prior`, `TempoPFN` (handled in Phase 2).

**Canonical paper source (Overleaf).** The KDD paper LaTeX lives in an Overleaf
git-backed repo:
`git clone https://git@git.overleaf.com/6a3a09320428658b019d7f8e`
It uses the ACM template (`\documentclass[sigconf,review]{acmart}`, 8 content pages).
*This could not be cloned in the chat sandbox (no network); clone it in Claude Code* and
treat it as the authoritative `main.tex` — do not start a parallel paper file. Review is
single-blind, so author names stay in `main.tex` (no anonymization).

**Drafted already (in this output bundle):**
`pyproject.toml`, `.readthedocs.yaml`, `CITATION.cff`, the `docs/` Sphinx skeleton,
`.github/workflows/{ci,release}.yml`, and under `src/causaltimeprior/`:
`__init__.py`, `cli.py`, `benchmarks.py`, `baselines.py`.

**Guiding rule:** prefer reading the real source over trusting this plan's
signatures. Every place a real signature might differ from what the templates
assume is marked in-code with `TODO(consolidate)` or `TODO(release)`. Grep for those.

---

## Reference artifact: CausalDynamics (NeurIPS 2025)

`CausalDynamics` (Herdeanu et al., 2025; arXiv:2505.16620; PyPI `causaldynamics`;
docs at kausable.github.io/CausalDynamics) is the closest published reference for
*how to ship this kind of artifact*. It is a large-scale benchmark + extensible
generator for **structural discovery** of dynamical causal models (coupled ODE/SDE
systems, noise/confounding/lags, climate models). Adopt its release patterns; keep
the scope distinct.

**Scope contrast to state in the paper (not to copy):** CausalDynamics is
*observational structural discovery* — recover the causal graph from dynamics.
CausalTimePrior is *interventional / counterfactual effect estimation* via a causal
foundation model. They reconstruct graphs; we answer `do(...)` queries over time.
That is a clean differentiation axis — make it explicit in Related Work, and add
CausalDynamics as a row in the paper's comparison table (it will sit in the
"temporal ✓ / interventional ✗ / counterfactual ✗" cell, complementary to ours).

**Patterns worth borrowing (woven into the phases below):**
- A public, versioned **leaderboard** with a documented submission path (Phase 5).
- **Hugging Face Datasets** as the primary discovery surface, with a one-command
  `process_*.py` download script (Phase 5).
- A **difficulty-tiered / "challenge"** presentation of the suites — simple →
  coupled → harder — rather than a flat list (Phase 4).
- **Config-driven generation** with timestamped, manifested output dirs (Phase 4).
- README **status badges** (arXiv, docs, HF dataset, license, CI) and a docs
  **Troubleshoot** page + an executable notebook hierarchy (Phase 6).

**Possible direct connection:** the CausalDynamics lead author is B. Herdeanu — if
that is your collaborator, there may be an opportunity to align formats, cite
reciprocally, or reuse their HF/leaderboard tooling. Confirm before assuming; do not
state a relationship in the paper that isn't real.

---

## Phase 0 — Orient (do this first)

- [ ] Read the source trees: `causal_time_prior/`, `dotime/`,
      `continuous-time-causal-pfn/`, and the two submodules `Do-PFN-prior/`,
      `TempoPFN/`. Note actual module paths, class names, and the real signatures of
      `CausalTimePrior.generate_pair`, `CausalTimePrior.generate_dataset`, and
      `TemporalInterventionDataLoader`.
- [ ] In `continuous-time-causal-pfn/`, locate the **fine-grid / continuous-time**
      implementations (continuous intervention windows, fine-grid sampling / the
      underlying-SDE discretization). These are the source for
      `causaltimeprior`'s continuous-time module — map their class/function names and
      how they relate to `dotime/prior/extended_prior.py` (which has the discrete-time
      `intervention_source` modes). Decide which repo owns which piece before Phase 3.
- [ ] Clone the Overleaf paper repo
      (`git clone https://git@git.overleaf.com/6a3a09320428658b019d7f8e`) and confirm it
      uses `\documentclass[sigconf,review]{acmart}`. Treat its `main.tex` as canonical.
- [ ] Inventory the existing baseline classes and the `BASELINE_STRING_TO_CLASS`
      table in `scripts/tscm_identifiability.py`. List the exact `predict`/call
      signature they expose — `baselines.py` must adapt to it, not the reverse. Note
      the eight `TSCMStructure` values (`back_door`, `observed_confounder`,
      `confounder_mediator`, `front_door`, `mediator`, `instrumental_variable`,
      `rct_no_confounding`, `unobserved_confounder`) — the benchmark suite list must
      match all eight.
- [ ] Inventory the metric functions in `dotime/eval/metrics.py`
      (`compute_rmse`, `compute_mae`, `compute_nmse`, `compute_r2`,
      `compute_quantile_calibration`, `compute_pinball_metric`) plus the
      `_direction_accuracy` and `_bootstrap_ci` helpers. These become
      `causaltimeprior.evaluation`.
- [ ] Confirm the licenses of `Do-PFN-prior` and `TempoPFN` before vendoring or
      depending on them (see Phase 2).

---

## Phase 0.5 — Preserve paper-result code states (DO THIS BEFORE PHASE 1)

**Why this is non-negotiable.** Phases 1–2 are destructive to reproducibility:
consolidating `causal_time_prior` + `dotime` (+ the continuous-time code), deleting
`.gitmodules`, stripping the `n_max=41` padding, and rewriting imports all change the
exact code that produced the accepted **TSALM (ICLR 2026)** and **FMSD (ICML 2026)**
results. Once those changes land, the camera-ready numbers can no longer be reproduced
from `main`. Snapshot the result-producing states first, as immutable references, so the
consolidation can proceed freely without stranding the published results.

**Note on FMSD:** the FMSD/continuous-time results come from
`continuous-time-causal-pfn/` (fine-grid implementations), which may be a *separate repo*
from `dotime/`. Tag the result state **in whichever repo actually produced each paper's
numbers** — if FMSD lives in its own repo, the `paper/fmsd-icml2026` tag goes there, not
in the consolidation repo. Confirm which repo owns the FMSD results in Phase 0 before
tagging.

The single easiest thing to lose is the **submodule commit SHAs**: after `.gitmodules`
is deleted in Phase 2, the only record of which `Do-PFN-prior` / `TempoPFN` commits the
results used is whatever is written down now. Capture it before touching anything.

- [ ] **Identify the result-producing commits.** For each paper, find the commit(s)
      that generated the camera-ready numbers (cross-check against the dates on
      `results/**` and any checkpoint timestamps), **in the correct repo** (TSALM in the
      base/`dotime` tree; FMSD in `continuous-time-causal-pfn/`). If a paper's results
      span multiple commits, pick the last commit before submission and note later fixes.
- [ ] **Record submodule SHAs first.** At each result commit, capture
      `git submodule status` and the `.gitmodules` URLs into a committed file
      (e.g. `repro/SUBMODULES-tsalm.txt`, `repro/SUBMODULES-fmsd.txt`). Include the
      full SHA for `Do-PFN-prior` and `TempoPFN`. This must happen before Phase 2
      deletes `.gitmodules`.
- [ ] **Tag the states (source of truth).** Create *annotated* tags — immutable,
      can't drift — **in the repo that owns each result**:
      - `paper/tsalm-iclr2026`
      - `paper/fmsd-icml2026` (in `continuous-time-causal-pfn/` if that is where FMSD ran)
      Use `git tag -a paper/tsalm-iclr2026 <sha> -m "TSALM ICLR 2026 camera-ready results"`.
      Push tags explicitly (`git push origin --tags`).
- [ ] **Cut convenience branches.** From each tag, create a long-lived branch for
      anyone who needs to patch the reproduction path without disturbing the tag:
      - `repro/tsalm`
      - `repro/fmsd`
      These keep the pre-consolidation code (including submodule pins) installable
      exactly as-published.
- [ ] **Freeze each environment.** On (or matching) the machine that produced the
      numbers, capture a resolved dependency lock — `pip freeze > repro/requirements-lock-tsalm.txt`
      (and `-fmsd`). "Reproducible" includes dependency versions, not just your code.
      Note Python version and CUDA/torch build.
- [ ] **Archive the heavy artifacts.** The checkpoints and `results/**` JSONs are
      gitignored, so a tag/branch alone does NOT preserve them. Upload the per-paper
      checkpoints and result JSONs to Zenodo (alongside the Phase 5 benchmark suites)
      and record the DOIs in `repro/ARTIFACTS.md`. Without this, the tags reproduce the
      *code* but not the *numbers*.
- [ ] **Record the encoder backend each paper used.** Load each camera-ready
      checkpoint and read `ckpt['config']['encoder_backend']` (and `encoder_config`).
      Note it in `repro/README.md`. If any published run used `gdp`, its reproduction
      additionally requires the pinned `TempoPFN`/`fla` versions and a GPU — flag that
      explicitly, since the Phase 2b `[gdp]` extra is otherwise off the core path.
- [ ] **Write a short repro note per paper.** `repro/README.md`: for each paper, the
      tag name, the exact commands that regenerate the headline tables/figures, the
      submodule SHAs, the lockfile path, and the artifact DOIs.

**Design decision to confirm with the team (see also Open Decisions):**
annotated tags + thin `repro/*` branches is the recommended default. If the two
papers' code has already diverged substantially inside the shared trees — such that a
single working tree can't cleanly check out either result state — prefer a clean
snapshot into a **separate archived repo per paper** (`causaltimeprior-archive-tsalm`,
`...-fmsd`). Decide this by inspecting the real `git log` / divergence in Claude Code
before tagging.

**Acceptance:** both tags exist and are pushed; `git checkout paper/fmsd-icml2026`
yields a tree that installs (with its recorded submodule SHAs and lockfile) and
regenerates at least one headline result within tolerance; submodule SHAs, lockfiles,
and artifact DOIs are committed under `repro/`. Only then proceed to Phase 1.

---

## Phase 1 — Restructure to a `src/` layout

- [ ] Create `src/causaltimeprior/` and move the consolidated modules in:
      `prior.py`, `temporal_scm.py`, `temporal_graph.py`, `temporal_mechanism.py`,
      `temporal_scm_builder.py`, `interventions.py`, `regime_switching.py`,
      `chain_scm.py`, `utils.py`, `visualization.py` from `causal_time_prior/`;
      `extended.py`, `data.py` from the relevant `dotime/` modules
      (`dotime/prior/extended_prior.py`, `dotime/data/temporal_dataloader.py`); and a
      new `continuous.py` carrying the **fine-grid / continuous-time** generation from
      `continuous-time-causal-pfn/` (intervention windows, fine-grid sampling). Settle
      the boundary between `extended.py` (discrete-time `intervention_source` modes) and
      `continuous.py` (continuous-time windows) explicitly so the two don't overlap.
- [ ] Rewrite intra-package imports from `causal_time_prior.X` / `dotime.Y.Z` /
      `continuous-time-causal-pfn` paths to `causaltimeprior.X`. The drafted
      `__init__.py` already assumes the consolidated names — make the files match it
      (and add `continuous` to the lazy-submodule set if it carries heavier deps).
- [ ] Delete the `~/repos/ctp` path hacks: remove every
      `sys.path.insert(0, os.path.expanduser("~/repos/ctp"))` and any
      `export PYTHONPATH` instructions from docs/READMEs.
- [ ] Verify `pip install -e .` succeeds in a clean venv and
      `python -c "import causaltimeprior; print(causaltimeprior.__version__)"` prints `0.1.0`.

**Acceptance:** a fresh `python3.11 -m venv` + `pip install -e ".[dev]"` imports the
package and all eager core symbols without touching `PYTHONPATH`.

---

## Phase 2 — Resolve external dependencies (per-dependency strategy)

The current `.gitmodules` pulls `Do-PFN-prior` and `TempoPFN`. PyPI installs can't
resolve git submodules, so both must go — but they sit on very different code paths,
so they get different treatment.

**Key finding (verify in Phase 0.5):** the `TempoPFN` `GatedDeltaProduct` encoder is
**not on the default code path**. `TemporalEncoder` defaults to `backend="transformer"`
(plain `nn.TransformerEncoder`, no external deps); the `gdp` backend is only built when
`--backend gdp` is passed and additionally requires a GPU + `fla`
(flash-linear-attention). `Do-PFN-prior`, by contrast, is on the used path via a small
surface (`dopfnprior.utils.sampling`, `dopfnprior.configs.default_config`).

### 2a. `Do-PFN-prior` → reimplement (with attribution)

The used surface is small and well-understood, so reimplement it cleanly rather than
vendor — this removes the submodule entirely, gives a coherent single-author codebase,
removes license entanglement, and strengthens the "self-contained artifact" story.

- [ ] Enumerate every symbol used from `dopfnprior` (start with
      `utils.sampling`, `configs.default_config`); confirm the list by grep, not memory.
- [ ] Reimplement into first-class modules (e.g. `causaltimeprior/_sampling.py`),
      with an attribution note in each module docstring crediting Do-PFN and a
      citation entry in `CITATION.cff` + the paper.
- [ ] **Equivalence test (required).** Add `tests/test_dopfn_equivalence.py` that
      pins a seed and asserts the reimplementation matches the original
      (vendored-temporarily or submodule) output to a tolerance. Without this,
      "reimplemented with credit" silently breaks reproducibility of published
      numbers. Keep a temporary copy of the original around *only* until this test
      passes, then delete it.
- [ ] Confirm the reimplementation is genuinely independent (not a line-by-line
      transcription) so it stands on its own regardless of the upstream license.

### 2b. `TempoPFN` / GatedDeltaProduct → optional `[gdp]` extra (do NOT reimplement or vendor)

Reimplementing a linear-attention kernel for an optional, GPU-only backend is high
effort and high numerical risk for no benefit to the benchmark contribution. Keep it
external and optional.

- [ ] Make the `gdp` backend import lazy and guarded (it already is — the import lives
      inside `_init_gdp_layers`). Raise a clear, actionable error if `backend="gdp"`
      is requested without the extra installed.
- [ ] Add a `gdp` extra in `pyproject.toml` depending on `tempopfn` (+ `fla`) — publish
      a `tempopfn` distribution with the authors' consent if none exists, otherwise pin
      a VCS/source dependency *in the extra only* so it never blocks the core install.
- [ ] Fix the fragile `from src.models.blocks import ...` path so it targets the
      installed package name, not the submodule's `src` layout.

### 2c. Common cleanup

- [ ] Delete `.gitmodules` and the submodule entries once nothing on the core path
      imports them. **Precondition:** submodule SHAs are already recorded under `repro/`
      (Phase 0.5) — do not delete `.gitmodules` until that capture is committed.
- [ ] Remove the `n_max=41` padding from the data-generation public API: push it down
      into a model-facing adapter so released tensors are clean `(T, N)` without
      zero-padded columns (reviewers will inspect this).

**Acceptance:**
`grep -rn "Do-PFN-prior\|dopfnprior\|~/repos/ctp" src/` returns nothing (Do-PFN fully
reimplemented); the only remaining `TempoPFN`/`fla` references are inside the lazy
`gdp` path and the `[gdp]` extra; `pip install causaltimeprior` (no extras) imports and
runs the transformer backend on CPU with no GPU/`fla`/`tempopfn` present; the Do-PFN
equivalence test passes.

---

## Phase 3 — Wire the drafted modules to real code

### `benchmarks.py`
- [ ] Fill the `zenodo_record_id` and `doi` fields in `_SUITE_REGISTRY` after
      Phase 5 mints them.
- [ ] Implement `_download_from_zenodo` with stdlib `urllib` (keep it
      dependency-light so it stays in core): fetch the record JSON, stream files,
      verify md5, write `manifest.json`.
- [ ] Implement `_parse_suite_dir` against the schema emitted by
      `scripts/build_release.py` (Phase 4). Reuse `InterventionSpec`
      (de)serialization from `interventions.py`; do not reinvent it.
- [ ] Replace `_generate_fallback`'s placeholder `y_true=zeros` with the real
      counterfactual target from the SCM, OR delete `_generate_fallback` entirely
      once Zenodo hosting is live (it's a dev convenience only).

### `baselines.py`
- [ ] Reconcile the `Baseline.predict(episode) -> Tensor` interface with the
      existing baseline call signature. If they differ, write thin adapters in
      the registered classes rather than changing the existing baselines.
- [ ] Wire `Oracle` to the stored ground-truth target.
- [ ] Wire `PCMCI+`, `BayesianITS`, `Chronos`, `DoOverTimePFN` to their real
      implementations (`tigramite`, `causalpy`, the existing `Chronos2Observational`
      and `BackDoorDoTPFNCausalEffect`). `DoOverTimePFN` should accept a checkpoint.
- [ ] The trivial baselines (`Zero`, `Mean`, `VAR-OLS`) are already correct —
      VAR-OLS was verified to recover a known VAR(1). Leave them.

### `evaluation.py` (not yet drafted — create it)
- [ ] Port the metric functions and `_direction_accuracy` / `_bootstrap_ci` from
      `tscm_identifiability.py` and `dotime/eval/metrics.py`.
- [ ] Implement `evaluate(model, suite, metrics=None) -> Results`, iterating
      episodes, calling `model.predict`, aggregating per-structure and pooled.
- [ ] Give `Results` a `.summary()` (human-readable table) and `.to_dict()`
      (JSON-serializable) — `cli.py` calls both.
- [ ] Keep `DIR_ACC_EPS = 0.1` (near-zero targets excluded from direction accuracy).

**Acceptance:** `ctp-benchmark --suite CTP-Generic-100k --baseline VAR-OLS` runs
end-to-end and prints a results table (using the local fallback suite until Zenodo
is live).

---

## Phase 4 — Reproducible artifact build

- [ ] Write `scripts/build_release.py`: a single committed script that regenerates
      all four frozen suites with fixed seeds and records the package version +
      hardware. This is the reproducibility anchor cited in the paper.
- [ ] **Config-driven generation (CausalDynamics pattern).** Drive `build_release.py`
      and `ctp-generate` from a committed `config.yaml`, and write outputs to a
      timestamped, manifested dir (`output/<timestamp>/` + `manifest.json` recording
      config hash, seed, package version). This makes every regeneration
      self-describing and auditable.
- [ ] **Difficulty-tiered suite design (CausalDynamics "challenge" pattern).** Present
      the suites as an increasing-difficulty hierarchy rather than a flat list — e.g.
      tier the identifiability/continuous/regime suites by #variables, lag depth,
      confounding, and intervention-window complexity, and expose the tier as suite
      metadata. This strengthens the benchmark narrative and gives the leaderboard
      (Phase 5) meaningful per-tier columns. Reflect the tiers in the paper's §5.
- [ ] Define the tidy on-disk schema once (parquet shards + `manifest.json`) and
      implement the parquet writer in `cli.py::_write_parquet` to match it
      (currently `NotImplementedError`).
- [ ] Generate Croissant JSON-LD metadata per suite.
- [ ] Record exact seeds, version, and runtime stats for `CTP-Generic-100k`
      (the energy-footprint figure goes in the paper's responsible-release section).

---

## Phase 5 — Data hosting, DOIs, and community surfaces

- [ ] Mint a Zenodo **concept DOI** per suite with the **real author block** (the D&B
      track is *single-blind* — no anonymization needed; the DOI is cited directly in the
      submission). Pledging artifact availability with a DOI-bearing archive at submission
      is viewed positively by reviewers (ACM **"Artifacts Available"** badge — see Phase 7).
- [ ] Upload parquet/HDF5 + Croissant metadata to Zenodo (DOI + long-term archive).
- [ ] **Hugging Face Datasets as the primary discovery surface (CausalDynamics
      pattern).** Mirror each suite to a HF dataset repo and ship a one-command
      download script (their pattern: `wget .../process_<name>.py && python
      process_<name>.py`) so users get data without reading docs. Keep Zenodo as the
      citable archive of record; HF is for reach. Make `load_benchmark` able to pull
      from either (HF first for speed, Zenodo as fallback).
- [ ] Backfill `zenodo_record_id` / `doi` (and HF repo id) into `_SUITE_REGISTRY`
      (Phase 3).
- [ ] Enable the **Zenodo–GitHub integration** on the repo so every tagged GitHub
      Release is auto-archived under the concept DOI.
- [ ] **Public leaderboard (CausalDynamics pattern — high impact for D&B).** Stand up
      a versioned leaderboard with a documented, reproducible submission path
      (results JSON schema + an eval script that anyone can run against a suite).
      Seed it with the Phase 6 baseline numbers and Do-Over-Time-PFN. Per-tier
      columns (Phase 4) make it informative. A simple static leaderboard page in the
      docs (or a HF Space) is enough for v1 — it materially raises adoption and is a
      concrete artifact reviewers reward. (Single-blind track — the leaderboard and
      submission path can use real names; no anonymization needed.)

---

## Phase 6 — Quality gates

- [ ] Add `tests/`:
      - property-based (Hypothesis) tests for SCM invariants — acyclicity of
        `G_0`, spectral-radius/stability after clipping, no NaNs over a long roll-out;
      - a round-trip test: `load_benchmark(...)` → iterate → shapes/dtypes;
      - a registry test: every name in `baselines.available()` instantiates;
      - mark GPU/slow tests with the `gpu` / `slow` markers (CI skips them).
- [ ] Get `ruff check .`, `ruff format --check .`, and `mypy` clean on `src/`.
      Expect to add type annotations to ported code; `_vendor/` is exempt.
- [ ] Confirm `make -C docs html` builds with zero warnings (CI treats warnings as
      errors). Convert `algorithm_walkthrough.ipynb` into `docs/tutorials/` and
      commit it **with outputs** (nbsphinx is set to `never` execute).
- [ ] **README status badges (CausalDynamics pattern).** Add badges for arXiv, docs
      homepage, Hugging Face dataset, license (Apache-2.0), and CI/tests status. They
      are low-effort, high-signal trust markers for a benchmark artifact.
- [ ] **Docs Troubleshoot page (CausalDynamics pattern).** Add `docs/troubleshoot.md`
      covering the predictable install/runtime snags: torch/CUDA mismatches, the
      `gdp` extra needing a GPU + `fla`, HF/Zenodo download/cache issues, and
      benchmark-cache invalidation.
- [ ] **Executable notebook hierarchy.** Mirror the simple→complex tutorial ladder:
      e.g. quickstart → identifiability suites → continuous-time/regime → training a
      PFN. Commit with outputs; cross-link from the difficulty tiers (Phase 4).

---

## Phase 6.5 — Decide the package name (before Phase 7 reserves it)

The drafted files use **`causaltimeprior`** as a *placeholder* import + distribution name.
The name is genuinely under reconsideration — see the note below — so settle it before
reserving on PyPI, because a later rename touches every drafted file.

**Decision framework:**
- **Three surfaces, can differ.** Project/paper name ("CausalTimePrior"), PyPI
  *distribution* name (`pip install X`), and *import* name (`import x`). PyPI allows the
  distribution and import names to differ (e.g. distribution `scikit-learn`, import
  `sklearn`). The project/paper name need not equal the package name.
- **Constraints.** Lowercase; import name must be a valid identifier (letters/digits/
  underscore, can't start with a digit); avoid the `-py` suffix (PyPI discourages it —
  redundant); avoid collisions with existing PyPI/well-known packages and with
  causal-inference terms-of-art that imply a narrower scope than the tool has
  (e.g. `cate`, `causica`).
- **Availability check FIRST (gating).** Before committing, verify the name is free:
  `pip index versions <name>` and check `pypi.org/project/<name>`. The sandbox had no
  network, so this must be done locally.
- **Fallback rule.** If the chosen import name is taken, use a distinct *distribution*
  name with the clean *import* name where possible; do **not** fall back to a `-py`
  suffix.

**Candidate names (decision pending — `dotime` is the current front-runner):**
A catchier name that fits the research-program brand likely beats `causaltimeprior`, which
over-emphasizes "prior". **Leading choice: `dotime`** — encodes do-calculus (intervention)
+ time series, and unifies the brand with Do-Over-Time-PFN across the TSALM/FMSD/KDD/ICAIF
line. Two conditions before committing:
  1. **Existing-repo collision (critical).** `dotime` is *already one of the three source
     repos* (the model hub). Consolidate **into that repo**, cut the Phase 0.5 tags FIRST
     (so `paper/tsalm-iclr2026` still resolves to the old `dotime` module), then let the
     new unified package supersede the old module. Archive the other two repos
     (`causal_time_prior`, `continuous-time-causal-pfn`) after tagging each for its paper.
  2. **PyPI availability.** "dotime" may be taken (time-tracking vibe) — verify locally
     (`pip index versions dotime`). Brand-preserving fallbacks if taken: `dotime-pfn` or
     `do-time` (distribution) with `dotime` (import).
  Note: `dotime` encodes *interventional* but not *counterfactual* — acceptable, since
  counterfactuals are computed via interventions. CLI would become `dotime-generate` /
  `dotime-benchmark` (replacing the `ctp-` prefix).

**Avoid known collisions:** `causaltime` (competing ICLR observational causal-*discovery*
benchmark, causaltime.cc), `dice` (DiCE counterfactual-explanation lib), anything near
`dowhy`, `alibi` (Seldon), `causica` (Microsoft), `cate` (the estimand — narrow scope).
Other candidates if `dotime` is unavailable/undesired: `interfact` (INTERventional +
counterFACTual — states the novelty), `cactus` (memorable real word), `scena` (scenario
metaphor), or the descriptive `causaltimeprior` (safe fallback). Whatever is chosen, the
**paper title can remain "CausalTimePrior"** even if the package ships under a shorter name.

**Rename mechanics (once decided):** the new name must be applied consistently across
`pyproject.toml` (`name`, `[project.scripts]`, `[tool.*]` paths), the entire `src/<name>/`
tree (directory + all imports), the whole `docs/` tree (every `import`/`from` example +
`conf.py`), `.github/workflows/*` (the `url`/badge references), `CITATION.cff`, and the
CLI command prefix if it changes from `ctp-`. Do this as one mechanical pass and re-run
the full test/build to catch stragglers. Reserve the chosen name (and a sensible alias)
on PyPI in Phase 7 immediately after.

---

## Phase 7 — Reserve the name + dry-run publish

- [ ] **Reserve the chosen name now** (before any arXiv preprint), per Phase 6.5: upload a
      `0.0.0` placeholder or use `twine` to claim it. Reserve a sensible alias too
      (e.g. the descriptive long form) to prevent confusion/squatting.
- [ ] Configure **PyPI Trusted Publishing** on both PyPI and TestPyPI: add a GitHub
      Actions publisher for the repo, workflow `release.yml`,
      environments `pypi` and `testpypi` respectively.
- [ ] Create the `pypi` and `testpypi` **GitHub Environments** (Settings →
      Environments). The release workflow references both.
- [ ] Dry-run the build locally: `python -m build && twine check dist/*`.
- [ ] Tag `v0.1.0`. The release workflow builds → TestPyPI → PyPI → GitHub Release.
      Verify the TestPyPI install in a clean venv first.

---

## Phase 8 — Pre-submission verification (the paper's repro checklist)

- [ ] `pip install causaltimeprior` works on a clean Python 3.10/3.11/3.12 venv on
      Linux **and** macOS.
- [ ] `pip install "causaltimeprior[all]"` pulls baselines/chamber/evaluation cleanly.
- [ ] All four suites round-trip through `load_benchmark` without warnings.
- [ ] Reference baseline numbers reproduce within ~1% of the values in
      `results/do-over-time-pfn/*.json`.
- [ ] `ctp-generate -n 100 -T 100 -o /tmp/smoke.pt` and
      `ctp-benchmark --list` both succeed.
- [ ] Docs build clean; Read the Docs project connected and green.
- [ ] **Single-blind submission (no anonymization):** keep real author names and
      affiliations in `main.tex` and package metadata; link the real GitHub repo (a repo
      link is recommended but cannot be shared after the deadline during review). Pledge
      the **"Artifacts Available"** badge at submission with the Zenodo DOI.
- [ ] **CFP logistics:** every author has a complete OpenReview profile; ≤2 submissions
      per author this cycle; ≥1 qualified author nominated as reviewer; main paper is **8
      self-contained content pages** in `acmart` `sigconf,review`; Cycle-1 dates —
      abstract **Jul 19 2026**, paper **Jul 26 2026** (AoE, strict).

---

## Open decisions to confirm with the team

0. **Result-preservation strategy (Phase 0.5).** Annotated tags + `repro/*`
   branches (recommended) vs. separate archived repos per paper — decided by how
   far the TSALM and FMSD code states have diverged inside the shared trees.
   Resolve before any Phase 1 work.
1. **Single package vs. meta-package — RESOLVED: single package + extras.**
   One `causaltimeprior` distribution. The PFN model and the GPU-only `gdp` encoder
   are gated behind optional extras (`[models]`, `[gdp]`), not separate
   distributions — same benefit (light core install for the benchmark artifact),
   none of the inter-package version-skew cost. Revisit only if an external
   contributor ecosystem with independent release cadences actually materializes.
2. **External dependencies — RESOLVED (per-dependency, see Phase 2).**
   `Do-PFN-prior`: reimplement with attribution + equivalence test (small, on the
   used path). `TempoPFN`/GatedDeltaProduct: keep external behind the optional `[gdp]`
   extra (optional GPU-only backend, not default; reimplementing it is not worth the
   risk). Confirm the FMSD checkpoint's backend in Phase 0.5.
3. **Authorship** on `CITATION.cff` and the software DOI — include Bernhard /
   Ruben on the software artifact, or only on the paper? Update `CITATION.cff`
   accordingly.
4. **Co-submission framing** — if a separate methods paper goes to NeurIPS/ICML,
   the KDD D&B submission must stand alone on the dataset/benchmark contribution.
