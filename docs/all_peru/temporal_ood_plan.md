# Temporal out-of-distribution skill — plan for steps 1–3

> **Status: proposed, not started (2026-08-10).** The successor to
> [`window_plan.md`](window_plan.md), whose three pre-gates all failed
> ([`RESULTS.md`](RESULTS.md) §8). That plan tried to change the *estimand*; this one attacks
> the *cause* — the model is trained on 1996–2009 and asked to predict 2019–2023, and nothing
> in the design ever made it good at that, or measured whether it was.
>
> **Read first:** `RESULTS.md` §8 (what failed and why), `window_plan.md` §9 (the scoreboard),
> `../perennial/RESULTS.md` §4.6 (`frac_l7` — gain ≠ contribution), CLAUDE.md §8.

---

## 0. The problem, stated precisely

Every parcel carries **one** label year — its PETT declaration — and its features are
extracted for that same year. The modelling table is a cross-section in which different rows
are different years, not a time series. `splits.py` assigns folds by 5 km region and
stratifies on **label**; it never read `year` at all until the §4.4 balancing added on
2026-08-10, and even that touches only the *test* draw.

Consequences, measured:

| | value |
|---|---:|
| national labels in 1996–2009 | **99.46 %** |
| national labels ≥ 2010 | 0.54 % (189 in 2010, 35 in 2011, single digits after) |
| Piura labels in 1998–99 | 73.7 % |
| per-fold TV distance on label year (val vs train), national | 0.157 – 0.241 |
| per-fold TV distance, Piura | 0.096 – 0.284 |
| test vs trainval TV on year — national / balanced draw / Piura | 0.167 / **0.092** / 0.232 |

So **train and val always span the same years**: no CV fold is ever scored on an unseen year,
and the deliverable needs 2019–2023 — 10 to 27 years past any supervision that exists. LOYO
(`allperu/loyo.py`) is the closest test available and it only probes *inside* 1996–2009; even
there the worst non-1998 cohort scores 0.406 against CV 0.581, and `PERENNIAL` recall swings
0.370–0.882.

Two mechanisms are measured, not assumed:

1. **Observation-density compression.** Clear Landsat observations fall ~24 → ~13 per
   parcel-year between W04 and W19 (L5 retired, L7 SLC-off since 2003). Within parcel, the
   window probability tracks density: **+0.052 per log-observation on PETT-`PERENNIAL`
   parcels, −0.022 on PETT-`ANNUAL` ones** (p < 1e-6 both). Both classes slide toward the
   base rate as evidence thins, which at the level of a share is indistinguishable from a
   real conversion. It explains 0–29 % of the control-pool drift, depending on arm.
2. **Statics manufacture stability.** Third appearance of the pattern: the compression
   coefficient is monotone in how *few* time-invariant features an arm carries
   (`nometa` +0.001 → `nolat` +0.052 → `ltae` +0.085).

**⚠️ Nothing in steps 1–3 can be verified at the endpoint.** They improve temporal transfer;
only endpoint labels (step 4, out of scope here) can confirm it. Every claim these steps
produce must be phrased as "transfer improved on the cohorts we can test", never "the 2019–23
prediction is now correct".

---

## 1. Decisions that bind this work

| # | decision | rationale |
|---|---|---|
| **T-D1** | **A change is adopted only if it improves out-of-distribution-year skill.** Spatial CV is *not* the criterion and a CV regression is acceptable. | This is the same rule LODO applied to space (`RESULTS.md` §6.2); applying it to time is the whole point |
| **T-D2** | The deciding metrics are **LOYO mean + worst-cohort macro-F1** and **`PERENNIAL` worst-cohort recall**, with **T3's W2 control-pool slope** as the panel-side check | LOYO is the only labelled temporal test; W2 is the only endpoint-era signal, even though it is unlabelled |
| **T-D3** | Report **CV, LODO and LOYO together** for every candidate. A change that buys LOYO by wrecking LODO is not an improvement | space and time are separate failures; fixing one at the other's expense is a wash |
| **T-D4** | Every experiment writes its own tagged run dir and artifacts. **Nothing overwrites an existing run, table or JSON** | four models, three panels and two workspaces already coexist; a silent overwrite destroys a comparison that cost hours |
| **T-D5** | The national **locked test stays unspent**, and T4/T5 of `window_plan.md` stay unrun | gates still failed; `estimate.py` enforces this |
| **T-D6** | Negative results are written up in as much detail as positive ones | the project's three most valuable findings (`frac_l7`, `centroid_lat`, LTAE) are all negative results |

---

## 2. Step 1 — make the endpoint look like the training years

**No GEE.** The panel pixel store (`<CC_FEAT>/panel/pixels_{year}.parquet`) keeps per-date
observations, so everything here is re-assembly plus retraining.

### 1a — feature density-sensitivity audit *(first, it scopes the rest)*

For every feature in the LightGBM store, regress its panel value on `log(n_valid_obs)` with a
**parcel fixed effect** and SEs clustered by parcel — the machinery already exists in
`allperu/windows.py::density_confound_test`; generalise it to loop over features.

Expect order-statistic features (`NDVI_max`, `BSI_max`, percentile tails) to be the worst
offenders: a max over 24 draws is not the same statistic as a max over 13. Output a ranked
CSV. **This alone may justify replacing a handful of features with density-robust
equivalents** (trimmed quantiles, harmonic amplitude/phase) — cheaper than anything else here.

### 1b — degradation augmentation

Build degraded copies of the **training-year** features: subsample each parcel-year's
observations down to the endpoint density, and reproduce the L7 SLC-off gap structure (the
striped ~22 % pixel loss) rather than only the count. Match the *distribution* of
`n_valid_obs` in W19, not its median. Train on the union of original + degraded rows.

Watch for: parcels whose degraded copy falls under the `n_valid_obs >= 4` quality gate (drop
them, don't impute); the normaliser must be fit on the augmented train split only.

### 1c — density-conditional calibration

Temperature is currently one scalar. Fit **T(n_valid_obs)** instead: take held-out CV
parcels, artificially subsample them across a density range, and fit temperature per density
band (or a monotone function of `log n`). If compression is substantially a calibration
phenomenon, this partly undoes it for a few hours' work — and it is testable without touching
the model.

### Acceptance for step 1

Adopt if **all** of:

* LOYO worst-cohort gap shrinks materially against the 0.175 baseline (target: < 0.10, the
  registered tolerance);
* T3's W2 control-pool slope improves toward flat (target |slope| < 0.01/decade) on
  `lightgbm_nometa_nolat`;
* LODO mean does not fall by more than 0.01.

**Report the at-risk series both ways.** If the control flattens *and* the at-risk rise
vanishes with it, that is a real finding — the rise was the artefact — and it must be written
up as such, not buried.

---

## 3. Step 2 — select on temporal transfer, and remove year-leaking features

### 2a — year-leak audit

Fit a model predicting the **label-year cohort** from the features. Rank by gain. Any feature
that identifies which year you are in is a feature that will mislead you in a year you have
never seen — this is the generalised form of the `frac_l7` finding. Drop the top-k and re-run
LOYO for k in a small sweep.

⚠️ **Gain ≠ contribution** — measured twice in this project already (`frac_l7` was the
2nd-highest-gain feature and worth +0.0013 to drop; the 12-class audit found the same). Rank
by gain to *generate* candidates; decide by LOYO.

### 2b — LOYO-based selection

Run the existing candidates (`lightgbm_nometa`, `lightgbm_nometa_nolat`, `ltae`, plus
whatever step 1 produces) through LOYO and write a selection table with **CV / LODO / LOYO**
side by side. Update `runs/all_peru/selected_model.json` only under T-D2/T-D3, and record the
reasoning in the file as the existing one does.

### 2c — optional: per-year distribution alignment

Quantile-map each panel year's feature distribution onto the training-year distribution.
Cheap and powerful — **and dangerous**: it also erases genuine aggregate change. It must be
validated against the control pool (W2) *and* checked for whether it flattens the at-risk
series too. Treat as a sensitivity arm, not a default.

---

## 4. Step 3 — admit OLI, with harmonisation

`PANEL_MISSIONS = {"L5", "L7"}` (`perennial/panel.py`) was chosen so every panel year is
inferred on radiometry the model trained on (training is 52.8 % L5 / 47.2 % L7 / 0 % OLI).
Defensible — but it is also **what creates the endpoint density collapse**: from 2013 two OLI
sensors are flying and W19 could have ~3× the observations. `perennial/harmonization.py`
implements the Roy et al. coefficients and has never been used.

### 3a — the overlap test *(do this before re-extracting anything)*

2013–2023 has L7 and L8 flying together. For the existing panel sample, extract OLI for a
handful of those years and, **for the same parcel-year**, compare predictions from L7-only
features against harmonised-OLI-only features.

Pass if per-parcel probability agreement is inside the noise floor already measured
(adjacent-window disagreement 3–4 pp; suggested thresholds: median |Δp| < 0.05 and class
agreement above the 95 % band), with no systematic shift in the PETT-`PERENNIAL` control
pool. Fail ⇒ the mission policy stands and step 3 stops there, which is itself a result.

### 3b — if it passes

Re-extract the panel (or at minimum W14/W19) with OLI admitted and harmonised, re-assemble,
re-infer, re-run T3. The endpoint then has its observation density back, and step 1's
degradation augmentation becomes a smaller correction rather than the main one.

### Cost and GEE hygiene

This is the only step that spends GEE. Before launching:

* run `timing_probe` and size the job from the measured rate — never extrapolate;
* **verify every year by counting output parcels**, never by "the process ended" or "the file
  exists" (CLAUDE.md §8: 25 `ChunkTimeout` events, and the orphaned-thread bug that made a
  finished job sit at 0 % CPU for 3 h);
* `--years` now accepts `2013-2015,2019-2023`, so a two-window extraction is one command.

---

## 5. Documentation contract

The project has four models, three panels, two workspaces and a long trail of negative
results; documentation is what keeps that legible.

* **`RESULTS.md` §9** — one new section, sub-sectioned per step, with the numbers, the
  verdict against the T-D2 criteria, and what was *not* adopted and why.
* **This file** — a `§6 What happened` section at the end, scoreboard-first, written against
  the plan as registered (the same pattern `window_plan.md` §9 uses).
* **CLAUDE.md §8** — update the status block; it is what a fresh agent reads first.
* **`PIPELINE.md`** — if a module's interface changes (`assemble.py`, `calibration.py`,
  `panel.py`), update its entry there.
* **Memory** — a file per durable lesson, indexed in `MEMORY.md`.
* **Run artifacts** — every experiment gets a tagged run dir under `runs/all_peru/` and
  tagged tables in the workspace. Never overwrite (T-D4).
* **Tests** — new behaviour gets a test that pins the *decision*, not the implementation
  (`tests/test_window_pivot.py` is the model to follow).

---

## 5b. Out of scope here, deliberately (as registered)

* **Endpoint labels by photo-interpretation** (~1,500–3,000 parcels on recent sub-metre
  imagery, stratified on department × predicted class × tenure). This is the highest-value
  step in the whole project and the only one that *verifies* rather than mitigates — it would
  give an endpoint test set, an endpoint error matrix for Olofsson, and a direct check of
  T1's non-differential-error assumption at the endpoint, which is currently assumed. It is
  out of scope only because it needs a human labelling effort, not because it ranks below
  steps 1–3.
* **The 2012 CENAGRO anchor** (Chain B, on disk) and **SIEA/MIDAGRI district statistics**
  (harness built: `allperu/external.py`) — intermediate and external anchors.
* **The two-period tenure DiD** (`tenure.tenure_two_period`, 1.78 M parcels, 8.6 % newly
  registered) — a design change that makes endpoint *accuracy* matter much less than endpoint
  *comparability*.
* T4/T5 of `window_plan.md`, and the locked test (T-D5).

---

## 6. What happened when it was run (2026-08-11)

Written after the fact, against the plan as registered above. Full numbers:
[`RESULTS.md`](RESULTS.md) §9. Code: `allperu/{density,yearleak,oli_overlap}.py`,
`loyo.lodo_by_cohort`; tests `tests/test_density.py`, `tests/test_temporal_ood.py`.

### 6.1 Scoreboard

| step | verdict | the number that decided it |
|---|---|---|
| **1a** feature density audit | ✅ **ran, and found something structural** | order statistics move **0.347 within-SD per e-fold of observation count**, the harmonic/slope fits only **0.080** — a 4.3× gap, with the land held fixed |
| **1b** degradation augmentation | ⚠️ **small real gain, adopted as a training change** | LOYO mean **+0.0054** (paired t 1.95, p 0.073, 10/14 cohorts), W2 **−0.0592 → −0.0512**, LODO +0.0005, CV −0.0010 |
| **1c** density-conditional calibration | ⛔ **NOT adopted — route closed** | T(n) is real and *decreasing* (1.19 at n≈5 → 1.00 at n≈19; ECE 0.076 → 0.008) but applying it moves W2 by **0.0001** |
| **step 1 acceptance** | ⛔ **FAIL** | worst-cohort gap 0.175 → best **0.151** (target < 0.10); W2 −0.059 → best **−0.051** (target < 0.01) |
| **2a** year-leak audit | ✅ **ran; k = 10 is the optimum** | label-year cohort is predictable from the spectral features at **0.508** vs a 0.155 baseline, held out by region; dropping the top-10 gives LOYO mean +0.003 and **sd 0.0568 → 0.0500** |
| **2b** LOYO-based selection | ⛔ **selection unchanged — and LOYO was shown to be unfit as the sole criterion** | `centroid_lat` is worth **+0.0476 on CV, +0.0474 on LOYO, −0.0596 on LODO**; on the new joint test (LODYO) its advantage is **−0.005** |
| **2c** per-year quantile alignment | ⛔ **NOT adopted — worse than doing nothing** | W2 **−0.1023/decade** against the baseline's −0.0592 |
| **3a** OLI overlap | ⛔ **FAIL on the control leg; step 3 stops** | control-pool mean Δp **−0.0423** raw / **−0.1073** harmonised, against a split-half self-noise of −0.0035 |
| **3b** OLI panel re-extraction | **not run** | 3a's stop rule |
| **3c** refit the correction on our own data *(follow-up, not in the plan)* | ⛔ **route closed** | 27,573 same-day L7/OLI parcel pairs cannot identify a *slope* (SD of the difference **exceeds** the SD of either sensor's values); the offset that *is* identified is the best arm on every axis and still fails, because the sensor difference is **cover-type dependent** — 0.025 NDVI between perennial and pasture parcels — and no global linear map can remove a between-class difference |

### 6.2 What the plan got right

* **The mechanism it named is real and it is measurable.** Order statistics *are* the
  density-fragile features, exactly as predicted, and the audit puts a number on it.
* **The cheap experiments really were cheap.** Steps 1 and 2 spent no GEE at all; the panel
  pixel store carried per-date observations and everything was re-assembly plus retraining,
  as the plan claimed.
* **"Probe before extracting" worked.** 0.97–1.17 s/parcel-year measured, and the three-year
  OLI extraction landed where the probe said it would.
* **§2c's own warning was correct**, and stronger than it knew: quantile alignment does not
  merely erase genuine change, it actively *worsens* the artefact it was meant to remove.

### 6.3 What the plan got wrong

* **T-D2 named LOYO as a deciding metric, and LOYO cannot decide alone.** `loyo.py` holds
  region approximately fixed so that year varies — which means a time-invariant lookup table
  is as available to it as to spatial CV. The proof is arithmetic: `centroid_lat` buys
  **+0.0476 on CV and +0.0474 on LOYO**, the same number, and only LODO's sign flips. The fix
  is `allperu lodyo` (`loyo.lodo_by_cohort`), which re-scores existing LODO predictions per
  cohort so place and year are both out of distribution — **free, because the per-department
  fits already exist**. It should have been in §2b from the start. T-D3 ("do not buy LOYO by
  wrecking LODO") happens to reach the right answer anyway, which is why it was written.
* **The step-1 acceptance thresholds were not reachable by step 1.** Asking a worst-cohort gap
  of 0.175 to fall below 0.10 by re-assembling features was optimistic; density explains part
  of the drift (§8.3 said 0–29 %) and the best arm here recovered ~14 %, which is consistent
  with that measurement rather than a surprise. The criteria were right to be strict; the
  expectation of clearing them was not.
* **§3a's 0.95 agreement criterion was never achievable.** The split-half control — same
  model, sensor, year and parcel, two disjoint halves of the acquisitions — gives **0.655**.
  The criterion should have been expressed *relative to the model's own reproducibility*, and
  a control that measures it should have been registered alongside. Read that way, OLI's
  per-parcel disagreement is at the noise floor and the failure is entirely the control shift.
* **§1's "trimmed quantiles" suggestion is a trap.** Acting on the 1a ranking (`norder`) is
  the arm LOYO *rejects* — mean −0.0026, the worst `PERENNIAL` recall of any arm. The plan
  warned about gain ≠ contribution for the 2a ranking and should have applied the same
  scepticism to its own 1a ranking.

### 6.4 New facts on the record

* **`centroid_lat` costs nothing in CV *or* LOYO and everything in LODO.** Any future
  evaluation here must be out-of-distribution along the axis it is being trusted on.
* **The features identify the label year at 0.508 vs a 0.155 baseline**, across unseen
  regions. `frac_l7` was one symptom of a systemic property.
* **The Roy harmonisation, used for the first time, is worse than not correcting at all** on
  every axis measured. An unused correction is an untested correction.
* **OLI alone would give 21.8 clear observations per parcel-year against L7's 13.1** — the
  largest lever anyone has found on the density mechanism, and it is unusable until the
  class-specific offset is explained or refitted on this data.
* **Measured, and now the default in code:** the SLC-off loss a *parcel* sees is ~11 pp, not
  the nominal scene-level 22 %.

### 6.5 What to do next, cheapest first

1. **Report LODYO for every future candidate** and stop treating any single held-out axis as
   sufficient. It is free.
2. **Refit the OLI→ETM+ coefficients on this data** — 12,940 paired parcel-years across three
   years already exist in `oli_overlap_pairs_nolat.parquet`, which is a regression sample for
   exactly this. If a fitted correction removes the −0.042 class-specific offset, step 3
   reopens and the endpoint gets 66 % more observations.
3. **The two-period tenure DiD** — now its own plan,
   [`tenure_did_plan.md`](tenure_did_plan.md), with a **free pilot already run**
   (RESULTS.md §9.6): the drift genuinely does cancel between the two arms, power is far
   better than feared (~1,300 treated parcels for 2 pp), **but the pre-treatment placebo does
   not clear** and that is what the study must resolve.
4. **Endpoint labels by photo-interpretation** — now its own plan,
   [`endpoint_labels_plan.md`](endpoint_labels_plan.md). (⚖️ 2026-08-12: the fifth estimand,
   the tenure DiD, in fact returned a bounded null — RESULTS.md §11.) Estimands have failed and every
   technical mitigation has been measured; it is the only step that *verifies*. The two plans
   are complements: the DiD's outcome is still a prediction, and only endpoint labels can show
   whether the classifier can see the transition at all.
