# KDD 2027 D&B Paper — Section-by-Section Outline

**Working title.** *CausalTimePrior: A Synthetic Benchmark Generator for Interventional and Counterfactual Time Series.*

**Target venue.** KDD 2027, Datasets and Benchmarks Track (San Jose; Cycle 1).
**Format (per CFP).** ACM double-column, `\documentclass[sigconf,review]{acmart}`.
**Submission length: 8 content pages** + unlimited references *and* unlimited Appendix
(the first 8 pages must be **self-contained** — reviewers need not read past them).
Camera-ready: 12 pages total, up to 9 content pages, references + Appendix combined ≤3.
**Estimated length.** Target ≈6 000–6 500 words across the 8 content pages with figures
and tables; the per-section budgets below slightly exceed 8 pages by design — push all
proofs, per-structure tables, the datasheet, and extended results into the **unlimited
Appendix** so the main body stays self-contained and tight. Reviewing is **single-blind**
(real author names and affiliations listed — no anonymization).
**Headline claim.** *CausalTimePrior is the first open, scalable, and theoretically grounded benchmark generator for multivariate time-series causal inference under interventions, covering hard / soft / time-varying / continuous-time / regime-switching interventions and four counterfactual sampling modes, with reference implementations of seven baselines and four frozen evaluation suites released under Apache-2.0 / CC-BY-4.0 with Zenodo DOIs and Croissant metadata.*

---

## Positioning: which CFP category, and how we satisfy it

The CFP lists distinct submission categories with *different evidentiary bars*. This paper
spans two — **"Benchmarks and Benchmarking Tools"** and **"Data Generators and
Environments"** — and we position deliberately rather than evenly:

- **Primary: Benchmark.** Lead as a benchmark (frozen suites + evaluation protocol +
  baselines + leaderboard). This is where the evidence is strongest (the obs-vs-int gap,
  §6) and matches the successful CausalDynamics framing (titled "a benchmark", generator
  as engine). Under this category, representativeness is a *supporting* argument.
- **Secondary: Data Generator.** The generator is the mechanism producing the benchmark.
  But the CFP attaches a *specific, stricter* clause to this category: synthetic data
  "**must be accompanied with a quantification and a discussion of its representativeness**,
  in addition to proving its utility." We must satisfy this even though it is secondary,
  because a reviewer can invoke it. Utility is covered by §6; **representativeness is the
  load-bearing addition — see §7.1**, grounded in the FMSD workshop transfer tasks
  (CausalChamber + pharmacokinetic).

**Implication for framing:** title and abstract lead "benchmark"; the generator is the
engine; §7.1 explicitly discharges the data-generator representativeness requirement.
Do not make representativeness the *central* claim (the evidence is an honest
proof-of-concept, not a strong real-world guarantee) — make it a satisfied requirement.

---

## Abstract — *~200 words*

State the gap (existing time-series causal benchmarks are predominantly observational, small, or domain-specific). Name the artifact (`causaltimeprior` PyPI package + four frozen suites). List the four extensions vs. the workshop version: continuous-time intervention windows, counterfactual sampling modes, regime-switching SCMs as a strict generalization of ITS, and a Causal Foundation Model reference implementation. State two headline numbers: scale of the generic suite (100 k trajectories) and number of named identification structures covered (8). End with the falsifiable claim that on `CTP-Identifiability-v1` the average direction-accuracy gap between an interventional PFN and an observational-only baseline of equal capacity is positive and significant.

> **Tone.** Empirical, restrained. Avoid "first" without "to our knowledge". Reviewers in D&B punish overclaiming about scope.

---

## 1. Introduction — *~700 words*

**Goal.** Convince a non-causal reader that this is a contribution they would want in their toolkit, and a causal reader that the design is principled.

**Paragraph plan.**

1. *Why intervention data for time series is the bottleneck.* Healthcare, policy evaluation, economics, climate — all need counterfactuals over time. Anchor with one well-known motivating example (RCT-emulation, sepsis treatment, or pandemic NPI evaluation).
2. *Why existing benchmarks fall short.* (a) Observational-only generators (Dream3, NetSim, fMRI sims); (b) Domain-specific interventional datasets (CausalChamber, small RCT panels) that don't scale to foundation-model training; (c) Static-tabular causal benchmarks (CausalFM, ACIC) ignore time. Build a 2 × 2 to be made explicit in §2 (observational/interventional × static/temporal).
3. *Our contribution as artifact.* The `causaltimeprior` PyPI package; four frozen suites with DOIs and Croissant metadata; reference baselines; an evaluation harness.
4. *Our contribution as benchmark scope.* The four axes — intervention type, structural family, counterfactual mode, time grid — with a forward pointer to §3.
5. *Why it's principled.* One sentence on Markov / positivity / acyclicity guarantees with forward pointer to the (appendix) convergence theorem.
6. *Roadmap.*

**Figure 1 (full column).** Schematic showing: prior sampler → temporal SCM → paired (obs, do) trajectories → loaders/baselines → metrics. Same visual logic as the CausalFM toolkit diagram but with time on the x-axis.

**Reviewer hook.** End the introduction with one bullet of "what claims our benchmark supports / what claims it does not". KDD D&B reviewers want this explicit.

---

## 2. Related Work and Positioning — *~500 words*

**Three short paragraphs, no subsections.**

1. *Static causal benchmarks*: ACIC, IHDP, CausalBench, CausalFM. Note what they don't address (time).
2. *Time-series causal evaluation*: Dream3 / NetSim (observational only), CausalChamber (real, small), interventional EHR cohorts (proprietary), and **CausalDynamics** (Herdeanu et al., NeurIPS 2025) — a large-scale benchmark for *structural discovery* over coupled ODE/SDE dynamics. Position CTP against it explicitly: CausalDynamics recovers graphs from observational dynamics; CTP evaluates *interventional/counterfactual effect estimation* via a foundation model. Complementary, not overlapping. Cite Poinsot et al. 2025 on the absence of widely-agreed time-series causal benchmarks.
3. *Synthetic generators for foundation models*: Do-PFN (static), TabPFN priors, TempoPFN (observational time series). Position CTP as their natural temporal-interventional generalization.

**Table 1 (half-column).** Comparison matrix across ~8 benchmarks: rows = benchmarks (include **CausalDynamics** as the temporal-but-observational-discovery row that sits closest to CTP); columns = {temporal, interventional, counterfactual, multivariate, identification structures, configurable scale, OS license, DOI/Croissant}. A green-tick / red-cross table makes the gap visceral — CTP should be the only row ticking interventional + counterfactual + identification structures.

---

## 3. The CausalTimePrior Framework — *~1 000 words*

**Goal.** Give just enough formalism that a reader can implement an alternative sampler and a reviewer can verify the claims.

**3.1 Temporal SCM family** (~350 words). Define $\psi = (\mathcal{G}, f, p_\varepsilon)$ with instantaneous DAG $G_0$, lagged adjacencies $G_1, \ldots, G_K$, mechanism $f_i$ per node, and noise distribution $p_\varepsilon$. State stability + Markov assumptions plainly. One display equation for the forward recursion.

**3.2 Graph prior** (~200 words). Erdős–Rényi for $G_0$ with acyclicity rejection; Bernoulli with geometric decay $\gamma^k$ for $G_{k\geq 1}$. Beta(2, 5) edge probability; uniform $N \in [3, 10]$, $K \in [1, 3]$.

**3.3 Mechanism and noise priors** (~200 words). Mechanism set: linear, $\tanh$, $\sin$, $\cos$, $|\cdot|$, $(\cdot)^2$, ReLU, $\tanh\circ\,(\cdot)^2$, $\tanh\circ\mathrm{ReLU}$. Root vs non-root noise. Stability checks (clipping, divergence rejection, burn-in).

**3.4 Sampling pipeline** (~250 words). Type sampler: 70% diverse / 15% chain / 15% regime-switching. Forward simulation. Burn-in. Trajectory length. Refer to Algorithm 1 (one-page, pseudocode) in an appendix.

**Figure 2 (half-column).** Four panels: (a) a sampled temporal DAG (graphviz); (b) paired obs/int trajectory; (c) intervention-effect decay curve; (d) edge-probability decay across lag $k$.

---

## 4. Scope and Extensions — *~1 500 words* (the main novelty over the workshop paper)

This is the section reviewers will read most carefully. Four short subsections, one per extension axis, each with a concrete API example and a figure or table.

**4.1 Intervention types** (~250 words). Hard, soft, time-varying. Show the three on the *same* TSCM to illustrate qualitative differences. **Figure 3a**.

**4.2 Continuous-time intervention windows** (~350 words). Move from single-step `do(A_t = v)` to window-based `do(A_{[t_0, t_1]} = v)`. Use the `intervention_time_start / end` float encoding from `dotime/prior/extended_prior.py`. Argue that this is the discrete-grid sampling of an underlying continuous-time SDE and forward-references the regular-grid limitation as future work for irregular sampling. **Figure 3b**: window duration vs effect magnitude.

**4.3 Counterfactual sampling modes** (~400 words). The four `intervention_source` modes (`prior`, `observed_discrete`, `observed_normal`, `observed_uniform`) with the $[\mu \pm 3\sigma]$ positivity guard. Define each in one display equation. Argue that this is *the* knob practitioners care about: do you train your model on extrapolative interventions, or counterfactuals inside the observed support? **Table 2**: mode × OOD ratio × mean effect magnitude.

**4.4 Regime-switching as ITS generalization** (~500 words). Define the sticky-Markov regime chain. Show formally that a 2-regime chain with deterministic transition at $t^*$ recovers the piecewise / ABA ITS design. Argue this lets the same benchmark stress-test both modern causal foundation models *and* classical Bayesian piecewise ITS (CausalPy) in a fair head-to-head. **Figure 4**: regime-switching trajectory with CausalPy posterior overlay. *This subsection is the most likely cite-magnet for the KDD audience.*

---

## 5. The Released Benchmark Suites — *~1 200 words*

The artifact paragraph. One short subsection per suite, plus a usage subsection.

**5.1 `CTP-Identifiability-v1`** (~250 words). **Eight** named identification structures
(the full `TSCMStructure` set in `dotime`): `back_door`, `observed_confounder`,
`confounder_mediator` (back-door family); `front_door`, `mediator` (front-door family);
`instrumental_variable` (IV); `rct_no_confounding` (trivially identified); and
`unobserved_confounder` (deliberately non-identifiable, tests robustness). ~1 350
episodes each (≈10.8k total); T = 200; per-structure intervention/query offset protocol
matched to identification theory (back-door offset 0; front-door/mediator 1–5; IV 0–5).
Counterfactuals are exact. State the **identification asymmetry** plainly: back-door and
front-door are nonparametrically point-identifiable; IV identifies only under
linearity/monotonicity; the unobserved-confounder case is non-identifiable by design and
is included to measure graceful degradation, not accuracy. **Table 3**: structure ×
identification family × episodes × sample protocol.

**5.2 `CTP-RegimeSwitch-v1`** (~200 words). 10 k trajectories, regime density $\in \{2, 3, 5\}$. Bayesian piecewise ITS reference posterior shipped alongside.

**5.3 `CTP-Continuous-v1`** (~200 words). 10 k trajectories with continuous-time intervention windows of varying duration; query offsets $\{1, 2, 3, 5, 10\}$.

**5.4 `CTP-Generic-100k`** (~250 words). 100 000 trajectories from the full diverse prior; the training-scale snapshot. Validate sample quality: divergence rate, effect-magnitude distribution, intervention-type ratio.

**5.5 Distribution and metadata** (~300 words). Zenodo DOI per suite, Croissant JSON-LD, Hugging Face Datasets mirror, parquet / HDF5 formats, exact `causaltimeprior` version and seed used to regenerate. Reproducibility script `scripts/build_release.py`.

**Figure 5 (full-width).** Six small multiples: one trajectory + effect-decay curve per suite, plus a histogram of trajectory norms across suites.

---

## 6. Baselines and Reference Results — *~1 000 words*

State the protocol once, then show numbers.

**6.1 Protocol** (~200 words). Predict $Y^{\text{int}}_{t+\text{offset}}$ on its raw (un-normalised) scale. Per-trajectory normalization at evaluation time. Confidence intervals via bootstrap over trajectories.

**6.2 Baselines** (~250 words). Align names with `dotime`. Trajectory heuristics:
`TrajMean` (pre-intervention trajectory mean of the query var) and `AR1` (last observed
value). Classical structural baselines implementing the correct adjustment per graph:
`VAR(p=3)`, back-door OLS adjustment (`BackDoorOLS`), and instrumental-variable two-stage
least squares (`IV2SLS`). Foundation-model gold standard: a **TabPFN regressor** carrying
out the ground-truth adjustment formula per structure (replaces TempoPFN, which is
*univariate* and so unsuitable as a multivariate causal baseline) — this is the
"best-attainable" reference. Observational foundation-model baseline: `Chronos`, applied
per variable (note its univariate framing as a limitation). Bayesian piecewise ITS via
CausalPy (RegimeSwitch suite only). Headline method: Do-Over-Time-PFN in both regimes —
`TrainedPFN_int` (interventional) vs. `TrainedPFN_obs` (observational, identical
capacity) — the obs-vs-int gap is the central contrast. `Oracle` (true SCM) is the
upper bound on synthetic suites.

**6.3 Results** (~400 words). One paragraph per suite. **Table 4 (full-width)**: rows = baselines; columns = (suite, metric); cells = mean ± 95 % CI. Use the existing `results/do-over-time-pfn/*.json` numbers and rerun for repro.

**6.4 What the numbers tell us** (~150 words). Lift over naive is positive and significant for `Do-Over-Time-PFN`; observational baselines collapse on `CTP-Continuous-v1`; Bayesian ITS is competitive on `CTP-RegimeSwitch-v1` for the 2-regime case but degrades with more regimes.

**Figure 6 (half-column).** Direction accuracy vs intervention effect magnitude, faceted by suite. Demonstrates the effect-error-correlation diagnostic visually.

---

## 7. Representativeness and Validation — *~850 words*

Short, technical, defensive. Reviewers will look for problems here — and a
data-generator-category reviewer will look *specifically* for representativeness (§7.1).

**7.1 Representativeness: zero-shot transfer to real systems** (~300 words).
*This subsection discharges the CFP's data-generator requirement that synthetic data come
with a quantification and discussion of representativeness.* The argument is external
validity by transfer: models trained purely on the synthetic prior transfer **zero-shot**
to two distinct real domains, evidence from the FMSD workshop paper ("Towards
Continuous-time Causal Foundation Models"):
  - **Physical — CausalChamber light tunnel.** A PFN trained only on synthetic TSCMs
    generalizes zero-shot to the real optical/electronic system. Report the honest result:
    the trained PFNs are the **only baseline family with positive Pearson correlation with
    ground truth**, with **+19–26% RMSE lift over the naive (last-value) baseline** —
    while being candid that adjustment baselines match that lift. The point is
    *representativeness* (synthetic-trained models behave sensibly on real dynamics), not
    dominance.
  - **Biological — pharmacokinetic data.** Drug-concentration dynamics (compartmental,
    continuous-time, dosing = intervention) are a natural fit for the continuous-time/
    fine-grid generator and a second, disjoint domain. **TODO(Claude Code): pull the exact
    pharmacokinetic transfer numbers (metric, lift, correlation) from the FMSD paper /
    `continuous-time-causal-pfn` repo** — they are not in this outline's source and must be
    quoted accurately, not approximated.
Frame both honestly: two real domains (physical + biological) is *representativeness
evidence*, not a transportability guarantee. Tie back to §8's explicit statement of what
the benchmark does **not** establish. **Figure 7a**: synthetic-trained prediction vs.
real outcome scatter (Pearson r) for each domain.

**7.2 Internal validation and sanity checks** (~550 words).

- **Acyclicity and stability** (~100 words). Empirical divergence rate < 1 % on Generic-100k.
- **Identifiability sanity** (~150 words). Oracle achieves RMSE ≈ 0 on synthetic suites; chance-level direction accuracy on shuffled-treatment ablation.
- **Distributional shift between suites** (~150 words). Show that `CTP-RegimeSwitch-v1` and `CTP-Generic-100k` have non-trivial KL distance via a 2-sample test on summary statistics, justifying separate evaluation.
- **Counterfactual mode comparison** (~200 words). Show that `observed_normal` produces strictly smaller mean prediction errors than `prior` mode for observational baselines, confirming the positivity advantage.

**Figure 7b (half-column).** A "divergence dashboard": four small panels.

---

## 8. Responsible Release and Limitations — *~400 words*

Two paragraphs.

1. *Limitations.* $N \le 10$, $K \le 3$, regular time grid, additive Markovian noise, no measurement model, no missingness. On representativeness: §7.1 shows zero-shot transfer to two real domains (physical + biological) as *evidence*, but the frozen suites themselves are synthetic and the transfer lift is modest — so state plainly that CTP demonstrates representativeness, not a real-world transportability *guarantee*, and that broad real-world covariate shift is out of scope (the CausalChamber/PK bridges are transfer probes, not CTP suites).
2. *Responsible release.* OSI license, no PII (purely synthetic), reproducible from seeds, frozen DOI per suite, Croissant metadata, energy footprint estimate for generating the 100 k suite.

---

## 9. Conclusion — *~200 words*

Restate the artifact, the four scope axes, the four suites, the seven baselines. One forward-looking sentence on each of: irregular-grid extensions, partial observability, learned mechanism priors, and connection to causal world-model literature.

---

## Appendices (not counted against page budget)

- **A. Algorithms.** Pseudocode for the full sampling pipeline (Algorithm 1) and the counterfactual reconstruction (Algorithm 2).
- **B. Convergence theorem.** Restate the convergence result from the workshop paper with the six assumptions; full proof.
- **C. Hyperparameter tables.** Default config, regime-switching config, continuous-time config.
- **D. Per-structure breakdown.** Full tables for `CTP-Identifiability-v1`.
- **E. Datasheet.** Following Gebru et al., one page covering motivation, composition, collection, preprocessing, uses, distribution, maintenance.
- **F. Croissant metadata sample.** One representative JSON-LD block.
- **G. Reproducibility checklist.** KDD-style.

---

## Submission logistics (per KDD 2027 D&B CFP)

- **Dates (Cycle 1, AoE).** Abstract **July 19, 2026**; Paper **July 26, 2026**;
  rebuttal **Sept 29–Oct 13, 2026**; notification **Nov 14, 2026**. Deadlines are strict,
  no extensions; placeholder/dummy abstracts are grounds for desk rejection.
- **Venue/site.** San Jose; submitted via **OpenReview** (not public during review).
  Every author needs a *complete* OpenReview profile (affiliations ≥5 yrs, homepage,
  DBLP, ORCID, advisors) — an incomplete profile is grounds for desk rejection.
- **Per-author cap.** Max **2 submissions per author per cycle**.
- **Reviewer duty.** Must nominate ≥1 qualified author reviewer (≥3 papers at KDD/related);
  submitting commits that person to review. Plan for this.
- **Format.** ACM double-column `\documentclass[sigconf,review]{acmart}`; **8 self-contained
  content pages** + unlimited refs + unlimited Appendix.
- **COI.** Declare conflicts in OpenReview — active collaborators, co-authors within 24 mo,
  advisor/advisee. Relevant to the CausalDynamics/Herdeanu link and the Cambridge / MPI /
  Freiburg / Turing collaborators: a COI is a *matching* constraint, not a bar to citing.

## Writing & process notes

- **Anonymity — single-blind (NOT double-blind).** Per the CFP, the D&B review is
  single-blind: **list real author names and affiliations** in `main.tex` and keep the
  package metadata/GitHub URL real. No anonymized fork, no stripped affiliations. (This
  corrects the earlier double-blind assumption.)
- **Originality / prior workshop versions.** Workshop and non-archival presentations are
  explicitly *allowed* by the CFP. The TSALM (ICLR 2026) and FMSD (ICML 2026) workshop
  papers are non-archival, so building on them here is permitted — but the KDD submission
  must present *new* contribution (the package + frozen suites + benchmark study), not the
  same paper. Cite the workshop versions normally.
- **Co-submission strategy.** If a separate *methods* paper (PFN + theory) targets an
  archival venue, that paper's archival status is what matters — the KDD D&B submission
  must stand alone on the dataset/benchmark contribution and not overlap an archival
  methods paper under review. Workshop priors are fine; an archival methods paper under
  review at another venue is the case to keep disjoint.
- **Artifacts Available badge.** The CFP promotes the ACM **"Artifacts Available"** badge
  for code/data in a DOI-bearing archival repo — which the Zenodo plan already provides.
  **Pledge availability at submission** (revealed to reviewers, viewed positively);
  reneging can retract acceptance. This makes the Zenodo DOIs a submission-time asset.
- **Open access / APC.** From Jan 2026 ACM is fully OA; an APC may apply unless the
  author institution participates in ACM Open. Check NUS's ACM Open status early to know
  whether a fee applies at camera-ready.
- **Camera-ready additions.** Real author block + funding acknowledgement; software
  citation (`CITATION.cff`) and Zenodo DOI badges on the README; expand to the 9-content-page
  camera-ready allowance (refs + Appendix combined ≤3 pages).
- **Pre-submission checks.** (a) Zenodo DOIs minted (real authors — single-blind); (b)
  `pip install causaltimeprior` works on a clean Python 3.11 venv on Linux + macOS; (c)
  baselines reproduce within 1 % of reported numbers; (d) `make html` builds the Sphinx
  docs cleanly; (e) all four suites round-trip through the loader without warnings; (f)
  GitHub repo linked in the submission (recommended) — note no links can be shared after
  the deadline during review.
