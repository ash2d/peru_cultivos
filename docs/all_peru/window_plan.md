# The window pivot — plan for the tenure → export-crop deliverable

> **⛔ STATUS: RUN, AND THE GATES FAILED (2026-08-10). Read [§9](#9-what-happened-when-it-was-run-2026-08-10) first.**
> T1, T2 and T3 were all implemented and run. **All three failed.** Per §5's own stop rule,
> the T4 extraction was **not** funded and the T5 estimate was **not** produced; the national
> locked test is still unspent. The §4 split work and the §6 risk work were completed anyway,
> because they are cheap and they inform whatever comes next — and one of them (§6.2) turned
> up a second dated tenure observation that changes what designs are available.
> Numbers live in [`RESULTS.md`](RESULTS.md) §8. Everything below §9 is the plan **as
> registered**, left unedited so the outcome can be read against what was predicted.
>
> **Status: proposed, not started (2026-08-10).** Supersedes the panel/trajectory strand of
> [`../perennial/plan.md`](../perennial/plan.md) §7–§9 and [`plan.md`](plan.md) as the route to the
> project's actual research question. Read
> [`RESULTS.md`](RESULTS.md) and [`../perennial/RESULTS.md`](../perennial/RESULTS.md) first —
> everything here is a response to the gate failures recorded there.
>
> **Nothing in the earlier work is retracted.** The classifier is sound; the *estimand* was wrong.

---

## 0. Why this plan exists

Both panels (Piura 28 years, national 25 years) failed the same Phase-7 gate: flicker 0.43–0.98
against a 0.15 criterion, on every architecture. The conclusion recorded in RESULTS.md §7.4 was
that a ~0.55–0.59-accuracy single-year 3-class classifier cannot support a **per-parcel annual
trajectory**.

That conclusion stands. What was not tested is that the research question never needed one.

The question is:

> *Has land shifted from domestic annual crops to export perennial crops, and does that shift
> differ by land-tenure status?*

That needs (a) a per-parcel state at **two** well-supported points, and (b) a **between-group**
comparison. It does not need 25 consecutive per-parcel calls, and it does not need the annual
trend to be individually defensible.

---

## 1. The pivot, in four moves

### M1 — Aggregate to multi-year windows, using probabilities

Define non-overlapping 5-year windows. Per parcel per window, take the **mean of
`prob_PERENNIAL`** over that window's non-abstained years (require ≥3), then threshold once.

**Not** the mode of the yearly `pred_label` — the mode discards the confidence information that
makes this work, and turns a parcel sitting stably at p≈0.45 into a coin flip.

Measured on the existing Piura panel (`data/processed/perennial/panel_predictions*.parquet`,
7,690 parcels, 5 windows W99/W04/W09/W14/W19 = 1999-2003 / 2004-2008 / 2009-2013 / 2014-2018 /
2019-2023):

| estimand | `lightgbm_nometa` | `..._nolat` | `ltae` |
|---|---:|---:|---:|
| 3-class annual flicker (the S5 gate) | 0.432 | 0.552 | 0.787 |
| perennial-vs-rest annual flicker | 0.350 | 0.460 | 0.762 |
| **≥2 window-state changes over 5 windows** | **0.039** | **0.053** | 0.218 |
| ≥1 window-state change | 0.085 | 0.110 | 0.304 |
| zero changes | **0.915** | 0.890 | 0.696 |

Parcel-level discrimination at the label year is **AUC 0.970** on locked-test parcels. The
perennial signal was never the problem; per-year `argmax` was.

**Adjacent-window disagreement — the noise floor — is 3.0–3.6 % and symmetric in both
directions.** That number is the yardstick every change estimate must clear.

⚠️ **LTAE is unusable for any temporal comparison.** Its window shares wander
0.160 → 0.203 → 0.253 → 0.163 → 0.155. That is drift, not signal. It remains a useful
single-year sensitivity arm and nothing more.

### M2 — Take the baseline from the labels, not from the imagery

Every documented failure — the El Niño confound (§8.1/§8.2), S4 backward transfer, the
1998 label↔year confound, the 1996–98 baseline — is a failure at **predicting the early years**.

The PETT declaration *is* the baseline state, observed on the ground in 1997–2006. Condition on
it instead of predicting it and all four problems disappear at once.

Measured, Piura, conditioning on the observed PETT label:

| PETT baseline (observed) | n | W99 | W04 | W09 | W14 | W19 |
|---|---:|---:|---:|---:|---:|---:|
| **ANNUAL** — the at-risk pool | 4,059 | 0.022 | 0.023 | 0.034 | 0.041 | **0.042** |
| PASTURE_FALLOW | 1,504 | 0.046 | 0.058 | 0.063 | 0.075 | 0.071 |
| PERENNIAL — the drift control | 1,066 | 0.771 | 0.708 | 0.758 | 0.762 | 0.753 |

The at-risk pool roughly doubles, monotonically, while **the PERENNIAL pool stays flat**. That
flatness is the key internal control: generic model drift or an L5→L7 radiometric ramp would move
both pools the same way. It does not. The `nolat` arm (fewer statics) gives a *larger* rise
(0.020 → 0.049), so the trend is not manufactured by static features either.

Size gradient, at-risk pool: <0.5 ha 0.010→0.021; 0.5–1 ha 0.012→0.033; 1–3 ha 0.032→0.052;
≥3 ha 0.197→0.202.

### M3 — Identify the tenure contrast *within* year

The estimator is a difference between two tenure groups measured in the **same** window. Every
artefact that destroyed the trend — year effects, El Niño, sensor era, coverage — is common to
both groups and differences out.

**This is the single most important design point in this document.** The panel does not have to
produce a defensible *level* or *trend* to produce a defensible tenure *contrast*. That is a much
lower bar than Phase 7 set, and it is the bar the research question actually sets.

The assumption it needs — classifier error is **non-differential** with respect to tenure,
conditional on region and parcel size — is directly testable with data already on disk (§5, T1).

### M4 — Go national, and stop being blocked by Piura

First look at the actual question, on the existing Piura panel (tenure read from the raw xlsx and
joined by `CodigoSSET` → `COD_PREDIO`):

| tenure | n (at-risk ANNUAL) | W99 | W19 | Δ |
|---|---:|---:|---:|---:|
| INSCRITO | 825 | 0.033 | 0.047 | +0.015 |
| NO INSCRITO | 3,234 | 0.019 | 0.035 | +0.015 |

Registered parcels sit ~1.7× higher at *every* window (the cross-sectional association
`../perennial/RESULTS.md` §7.5 already reported) but the **change is identical**.

**Do not read that as a null.** 825 parcels × a ~1.5 pp conversion rate is **about 12 events**.
This is an empty cell, not a result. §7.5's "power is not the problem" verdict was about the
*cross-sectional gap* and does not transfer to the *change* estimand.

⚠️ **And Piura is the worst department in Peru for this.** `ESTADO en RRPP` was audited across all
eight `data/raw/BD_SSET/*.xlsx` workbooks on 2026-08-10: **present in every one, on the first
sheet, 100 % non-null, binary INSCRITO / NO INSCRITO.** Record-level INSCRITO share for the 14
linkable departments:

| dept | INSCRITO | dept | INSCRITO | dept | INSCRITO |
|---|---:|---|---:|---|---:|
| AYACUCHO | 83 % | LAMBAYEQUE | 55 % | HUANCAVELICA | 28 % |
| ANCASH | 78 % | AREQUIPA | 55 % | LIMA | ~24 % |
| CAJAMARCA | 70 % | ICA | 53 % | TACNA | 23 % |
| LA LIBERTAD | 64 % | TUMBES | 52 % | **PIURA** | **16 %** |
| MOQUEGUA | 64 % | PASCO | 60 % | | |

(Record-level, pre-bridge; re-derive at parcel level in Phase 1. Note the sheet is named `DATOS1`,
not `DATOS`, in `BD SSET(AREQUIPA-AYACUCHO-CAJAMARCA).xlsx` — read `wb.sheetnames[0]`, never a
hard-coded name. `DEPARTAMENTO` also carries small numbers of rows for other departments, the
trap already documented in [`DATA_AUDIT.md`](DATA_AUDIT.md) §4.4.)

Nationally the tenure mix is near 50/50 and the linked pool is 946,872 polygons / 726,808
3-class-eligible parcels. The underpowered cell is a Piura artefact.

**The deliverable is national.** Piura becomes one department in it.

---

## 2. The estimand — write this down before running anything

> **Population.** Parcels in the 14 linkable departments whose PETT declaration resolves to
> `ANNUAL` (the at-risk pool), with a non-null `ESTADO en RRPP`, passing the area/quality gates.
>
> **Outcome.** `perennial_W19` — window-mean `prob_PERENNIAL` over 2019–2023 (≥3 observed years),
> thresholded at the calibrated operating point. Report the continuous window mean alongside.
>
> **Primary estimate.** Share perennial in W19, by tenure at titling, with department and
> registration-year fixed effects, weighted by `sample_weight`, SEs clustered by `region_id`.
>
> **Secondary.** Δ = share(W19) − share(W99) by tenure — the difference-in-differences version.
> Weaker (it needs the baseline window predicted, not just observed) but it controls for any
> tenure-correlated baseline propensity the fixed effects miss.
>
> **Controls that must be reported alongside, every time.**
> (a) the PETT-`PERENNIAL` pool's window series — must be flat;
> (b) adjacent-window disagreement — the noise floor;
> (c) the tenure-by-error test (T1).

**This is descriptive, not causal.** Reverse causality is live and unresolved: a farmer planning a
20-year orchard has strong reason to register title *first*. State that in every write-up. See §6.

---

## 3. Decisions

| # | decision | rationale |
|---|---|---|
| **W-D1** | Aggregate window-mean **probability**, not modal class | mode discards confidence; §1 M1 |
| **W-D2** | 5-year windows, non-overlapping, ≥3 observed years to qualify | matches the measured stability; shorter windows untested |
| **W-D3** | Baseline = **observed** PETT label; only the endpoint is predicted | removes El Niño, S4, the year↔label confound |
| **W-D4** | Extract **two** windows (W99 1999–2003 + W19 2019–2023), not the endpoint alone | W99 is the validation control, not the estimand — without it there is no drift check |
| **W-D5** | Primary model = `lightgbm_nometa_nolat` (the LODO-selected national model) | `RESULTS.md` §6.2; carries no `centroid_lat` |
| **W-D6** | LTAE is a **single-year sensitivity arm only** — never a temporal comparison | §1 M1, window shares wander |
| **W-D7** | **TM/ETM+ only, both windows** — no OLI, no Sentinel-2 | training radiometry is 0 % OLI (`../perennial/RESULTS.md` §6.2), and with no endpoint labels there is nothing to retrain a new-sensor model on. This bounds W19 at 2023 (L7 acquisitions stop) |
| **W-D8** | Locked test: the **national** one is unspent — keep it that way; spend once, at the end, on the window estimand | Piura's is spent twice |
| **W-D9** | Report the continuous window-mean probability as the primary outcome; thresholded share as secondary | ROC AUC 0.97 vs argmax F1 0.68 — the threshold is what loses information |
| **W-D10** | No per-parcel annual trajectories, no conversion dates, no annual area series | the Phase-7 gate failed for those and this plan does not rescue them |

---

## 4. The split design

The existing design ([`../../src/crop_classifier/splits.py`](../../src/crop_classifier/splits.py))
is well built on one axis and blind on two others. Keep the machinery; add axes.

### 4.1 Keep — spatial blocking

~86 % of adjacent parcels share a crop; 3-class agreement is 0.845 at <100 m and still 0.610 at
4–5 km. A random split leaks neighbouring farms and every number becomes fiction. Contiguous 5 km
regions + buffered dead-zone stay exactly as they are.

**Change:** run the headline number at `buffer_m = 3000` as well
(`config/split_b3000.yaml` exists and appears unused) and report both. The audit already warns
that 1500 m is below the measured decorrelation range, so CV is optimistic by an unknown amount.

### 4.2 Keep — leave-one-department-out

LODO is the project's strongest methodological result and stays the primary generalisation metric.
Nothing here changes it.

### 4.3 **Add — leave-one-year-out (LOYO) cohort validation** ⭐

The gap in the current design: **everything is held out in space, and scored at the label year.**
The failure that actually matters — *does this model work in 2020?* — is untestable by
construction.

The national data makes the fix available for the first time (label years spread 1997–2006):

* Restrict to regions containing **≥2 label-year cohorts**, so region is held fixed while year
  varies. This is the same design as the El Niño confound test
  (`perennial/diagnostics.py::elnino_confound_test`) — reuse its region-intersection logic.
* For each cohort `y`: train on cohorts ≠ `y`, test on cohort `y` at its own label year.
* Report per-cohort macro-F1 and `PERENNIAL` recall, plus the mean and spread.

This is the temporal analogue of LODO and the single most valuable evaluation the project has not
run. It is what licenses (or refuses) a 2019–2023 prediction.

### 4.4 Add — balance the test draw on `year × label`

`splits.py` never reads `year`, and titling swept region by region, so the Piura locked test came
out **48.3 % 1998 against trainval's 32.9 %** — an accidental temporal split nobody designed.

Fix in `pick_test_units`: draw ~200 candidate region sets at the target `test_frac`, keep the one
minimising total-variation distance between test and trainval on the joint `year × label`
distribution. Cheap, and it removes a whole class of confusion. Record the achieved distance in
`splits_meta.json`.

### 4.5 Add — stratify the extraction sample on `tenure × label × department`

The current national sample allocates sqrt-proportionally by department with a region cap, and the
Piura panel sampled proportionally by `label × region`. Tenure balance "came out fine by luck"
(`../perennial/RESULTS.md` §7.5). The whole deliverable now rests on the
`INSCRITO × ANNUAL` cell, so it must be stratified, not lucky.

* Strata: `department × tenure × PETT label`.
* **Oversample `INSCRITO × ANNUAL`** and `NO INSCRITO × ANNUAL` to the target in §4.6.
* Keep a proportional `PERENNIAL` allocation — it is the drift control (§1 M2) and needs ~1,500+
  parcels to be readable.
* Carry `sample_weight` through everything, as now. **Any share reported without it is wrong.**

### 4.6 Sample size — the binding constraint

To detect a 2 pp differential in conversion (e.g. 3 % vs 5 %) at 80 % power, α = 0.05:

```
n per group ≈ 7.85 · [p₁(1−p₁) + p₂(1−p₂)] / δ²
            ≈ 7.85 · [0.0291 + 0.0475] / 0.0004  ≈ 1,500
```

With the measured design effect from the weights (deff ≈ 1.4) that is **~2,100 per tenure group**.
For department fixed effects and any heterogeneity analysis, target **≥5,000 per group**, i.e.
**≥10,000 at-risk `ANNUAL` parcels with tenure**, plus ~1,500 `PERENNIAL` and ~1,500
`PASTURE_FALLOW` for controls. **Total ≈ 13,000–15,000 parcels.**

If the true differential is 1 pp rather than 2, this quadruples. Run the power calculation with the
Phase-2 pilot's observed rates before committing to the full extraction.

### 4.7 Clustering

Group assignment stays by region. ~56 % of farmers hold >1 parcel, so **standard errors must be
clustered by `region_id`** (and ideally by owner where `NOMBRES` resolves). Parcels are not
independent draws.

---

## 5. Task sequence

Each task states its acceptance criterion. **Stop and report if one fails** — several are designed
to falsify the plan cheaply, and that is their purpose.

### T1 — Non-differential error by tenure *(no new extraction; hours)*

**Why.** The estimator is a difference in *predicted* share between tenure groups. That equals the
difference in *true* share only if the classifier errs the same way on both. The mapping is

```
observed_share = true_share · sensitivity + (1 − true_share) · (1 − specificity)
```

Equal sens/spec (**non-differential**) → the contrast is attenuated toward zero but keeps its sign,
and a single error matrix corrects it (the Olofsson code in
[`perennial/area.py`](../../src/crop_classifier/perennial/area.py)). Unequal (**differential**) →
the bias has unknown sign and no single correction exists.

**Do.** Join tenure to held-out label-year predictions (`preds_cv.parquet`). Per tenure group,
compute `PERENNIAL` sensitivity, false-positive rate, and predicted-vs-true share. Then compare
**conditional on true class, area and region** — INSCRITO parcels are genuinely different (Piura:
0.78 vs 0.90 ha mean, 1.7× more perennial). Compact form: logistic regression
`correct ~ tenure + true_class + log(area) + C(region_id)`; the tenure coefficient is the test.

**Pass.** Tenure coefficient not significantly different from 0, and |Δsensitivity| < 0.05 within
area strata. **Fail.** Diagnose the proxy — usually size, road distance, or parcel regularity
driving clean-pixel count. If it survives conditioning, use group-specific error matrices or
restrict to strata where it holds.

⚠️ **Limit, and it does not go away in this plan:** this tests error at the label year
(~1999, TM/ETM+) while the estimand sits in 2019–2023. There is no endpoint-year ground truth
anywhere in the project, so **non-differential error at the endpoint is assumed, not verified.**
State that as a limitation wherever the contrast is reported. The only ways to close it are a
second supervision point (2012 CENAGRO via the weak name link, aggregate-level only) or endpoint
ground truth from outside this plan.

### T2 — LOYO cohort transfer *(no new extraction; hours)*

Implement §4.3. **Pass:** worst-cohort macro-F1 within 0.10 of pooled CV, and `PERENNIAL` recall
within 0.10, on cohorts other than 1998. **Fail:** the endpoint prediction is not licensed and no
further extraction should be funded until it is understood.

Report 1998 separately and expect it to fail — that is the known El Niño cohort, and W-D3 means the
deliverable no longer depends on it.

### T3 — Replicate the window diagnostic on the existing national panel *(no new extraction; hours)*

Re-run §1 M1/M2 on `data/processed/all_peru/panel_predictions*.parquet` (4,565 parcels × 25 years,
three arms). This is the first real test of whether the Piura result generalises.

**Pass (all four):**
* **W1** — ≥2 window-state changes over 5 windows < 0.10, on `lightgbm_nometa_nolat`;
* **W2** — the PETT-`PERENNIAL` pool's window share is flat (|slope| < 1 pp/decade);
* **W3** — adjacent-window disagreement is symmetric (|up − down| < 1 pp) and the W99→W19 net
  change in the at-risk pool exceeds it;
* **W4** — the at-risk pool's window series is monotone or near-monotone.

**Fail:** the pivot does not survive nationally and this plan should be revised before spending GEE
budget. n = 4,565 is small, so read W3/W4 as directional, not conclusive.

*(Scripts for T1–T3 exist only as ad-hoc scratch code from the 2026-08-10 session; write them
properly into `src/crop_classifier/allperu/windows.py` with tests.)*

### T4 — Re-draw and extract the window sample *(the expensive step)*

1. Build the national at-risk table: PETT label, tenure (parcel-level, from the raw workbooks —
   `load_sset()`'s `usecols` omits `ESTADO en RRPP`, so extend it), department, region, area.
2. Stratify per §4.5, size per §4.6 (~13,000–15,000 parcels).
3. Run a `timing_probe` (already in `perennial/panel.py`) before the full run and size the job from
   the measured rate — do not extrapolate from the old panel.
4. Extract **10 years** per parcel: 1999–2003 and 2019–2023. Reuse `panel.py` with an explicit
   `--years` list; note `--years` needs `lo-hi`, so pass two ranges or loop.
5. **Verify every year by counting output parcels**, never by "the process ended" — the lesson from
   the 25 `ChunkTimeout` events and the orphaned-thread bug (`CLAUDE.md` §8).

**Cost.** ~13,000 parcels × 10 years = ~130,000 parcel-years, comparable to the existing national
panel's 114,125 — but buying **~3× the parcels** at the same cost, because it buys 10 years instead
of 25. That trade is the whole point.

### T5 — Produce the estimate

Fit §2 on the T4 sample. Report, together, always:
* the tenure contrast with department + registration-year FE, weighted, clustered SEs;
* the three controls from §2;
* the Olofsson-corrected share with CIs — ⚠️ **the error matrix must come from the pooled
  spatial-CV confusion of the model actually used** (`preds_cv.parquet`), at the label year. The
  matrix on disk (`lightgbm_3c_final/test_confusion.csv`) belongs to a disqualified model, and no
  endpoint-year matrix exists. CV-derived CIs carry the residual spatial leakage of §4.1 and are
  mildly optimistic — **label them CV-derived, never test-derived**;
* the continuous window-mean outcome as well as the thresholded share (W-D9);
* the `buffer_m = 3000` variant.

Spend the national locked test **once**, here (W-D8).

### T6 — External validation *(cheap, do it in parallel)*

MIDAGRI/SIEA publishes district-level hectares by crop for Peru back to the 1990s. Compare
district-level predicted perennial growth against it. This is an **independent** check on the
aggregate — a much stronger claim than anything internal — and it also answers the §6 coverage
risk.

⚠️ With no endpoint ground truth in this plan (T1's limit), this is the **only** external check on
the endpoint.

---

## 6. Risks that no amount of modelling fixes

1. **⚠️ The export boom may be largely outside this cadastre.** Piura's mango/grape/banana
   expansion happened substantially on newly irrigated desert developed by agro-export firms
   (San Lorenzo, Chira-Piura), not on 0.5 ha PETT smallholdings. A true conversion rate of
   2 → 4 % on titled smallholdings may be the *answer*, not a detection failure — but if the growth
   sits outside the cadastre, no classifier improvement recovers it. **T6 is the check**, and it
   should run early.
2. **Reverse causality.** Tenure is observed once, at titling. A farmer planning an orchard has
   strong reason to register first. Cross-sectional data cannot separate this from
   tenure → perennial. The only real fix is **time-varying tenure** — if any later titling or
   registration wave with dates exists (SUNARP / COFOPRI), a staggered difference-in-differences
   becomes available and the design becomes causal. **Worth an explicit hunt before T4.**
3. **Woody non-crop false positives.** `woody_noncrop_policy: exclude` removes algarrobo and
   plantations from *training* but not from the world; at inference they read as `PERENNIAL`. At
   national scale this inflates the endpoint, and **nothing in this plan measures the rate.** Mask
   with a tree-cover / natural-vegetation layer, and state the residual as a limitation.
4. **Perennial ≠ export.** Coffee, plátano and naranja are in the `PERENNIAL` lexicon and are
   largely domestic; asparagus and paprika are export *annuals*. Build an explicit export-crop
   mapping from `config/perennial.yaml` and report what fraction of `PERENNIAL` is actually
   export-oriented in the label data. Cheap, and it bounds the interpretation.

---

## 7. What this plan does **not** do

* No per-parcel annual trajectories, conversion dates, or annual area series. The Phase-7 gate
  failed for those on two panels and three architectures; nothing here rescues them. The
  transition/area code in `perennial/trajectories.py` and `perennial/area.py` stays unrun for
  annual series — `area.py` is used only for the *window* shares in T5.
* **No new labels and no new features.** The classifier is used exactly as selected
  (`lightgbm_nometa_nolat`); this plan changes the estimand, the sample and the evaluation, not the
  model. Feature work and endpoint annotation were considered and are deliberately out of scope.
* No re-use of a spent locked test. Piura's is spent twice (`../perennial/RESULTS.md` §4.4, §4.5).
* No causal claim without time-varying tenure (§6.2).

---

## 8. Reproducing the §1 numbers

The diagnostics quoted above were computed on 2026-08-10 from files already on disk, with ad-hoc
scripts. The inputs are:

* `data/processed/perennial/panel_predictions.parquet` (and `_nolat`, `_ltae`) — per-parcel,
  per-year `prob_PERENNIAL`, `abstained`, `sample_weight`, `pett_label`;
* `data/processed/perennial/modeling_parcels.parquet` — `label`, `year` (= label year),
  `area_ha`, `region_id`, `split`;
* tenure — raw `data/raw/BD_SSET/BD SSET(MOQUEGUA-PASCO-PIURA).xlsx`, column `ESTADO en RRPP`,
  aggregated to `CodigoSSET` then mapped to `COD_PREDIO` via `Grafica_Tabular/Piura.dta`.

Recipe: drop `abstained` rows, group by `(COD_PREDIO, window)` taking `mean(prob_PERENNIAL)` with
`n_years >= 3`, threshold at 0.5, then tabulate by `pett_label` and `tenure`.

**First implementation task: move this into `src/crop_classifier/allperu/windows.py` with tests, so
T1–T3 are reproducible commands rather than scratch code.**

---

## 9. What happened when it was run (2026-08-10)

Written after the fact, against the plan as registered above. Full numbers:
[`RESULTS.md`](RESULTS.md) §8. Code: `allperu/{windows,loyo,tenure,window_sample,estimate,
external,export_crops}.py`; tests `tests/test_window_pivot.py`.

### 9.1 Scoreboard

| task | verdict | the number that decided it |
|---|---|---|
| **T1** non-differential error by tenure | ⛔ **FAIL** | accuracy is fine (tenure coef +0.013, p 0.107) but the **at-risk false-positive rate is 0.111 INSCRITO vs 0.155 NO INSCRITO** — −1.75 pp after conditioning on region and size (p 0.016), i.e. the size of the target effect and pointing the other way |
| **T2** LOYO cohort transfer | ⛔ **FAIL** | worst non-1998 cohort **0.406** vs CV 0.581 (tolerance 0.10); `PERENNIAL` recall spans **0.370–0.882**. 1998 itself is mid-pack (0.523) |
| **T3** window diagnostic, national | ⛔ **FAIL on all three arms** | **W1 PASSES** (≥2 window changes 0.088 vs annual flicker 0.75 — M1 works) but **W2 fails everywhere**: the PETT-`PERENNIAL` control drifts −0.044 / −0.059 / −0.103 per decade while the at-risk pool rises only +0.018 |
| T4 extraction | **not run** | §5's stop rule. The sample is drawn (13,002 parcels) and reviewable; the GEE budget is unspent |
| T5 estimate | **not run** | `estimate.py` refuses while a gate is failed; `--force` exists and stamps `gates_failed` into the output |
| T6 external validation | **harness built, not run** | needs a MIDAGRI/SIEA CSV, which is not redistributed here. Still the only external check on the endpoint |

### 9.2 What was right, and what was wrong

**M1 was right.** Aggregating window-mean *probability* and thresholding once really does
remove the instability that failed Phase 7: annual 3-class flicker 0.75 → 8.8 % of parcels
with ≥2 window-state changes, 82 % with none. The §1 Piura figures reproduce exactly.

**M2 was wrong, and it was wrong in a way §1 could not see.** The internal control is not
flat. It falls 0.511 → 0.386 nationally on the selected arm while the at-risk pool rises
0.017 → 0.035 — the yardstick moves seven times further than the thing being measured, and in
the opposite direction. It is not composition (parcels present in all five windows give the
same series).

**Why §1 looked clean:** the Piura control *is* flat — on `lightgbm_nometa`. On
`lightgbm_nometa_nolat`, the LODO-selected arm this plan chose in W-D5, Piura's control
already drifts −0.017/decade. §1's M2 evidence came from the arm §6.2 of RESULTS.md
disqualified.

**A mechanism, partly.** Landsat observation density falls from ~24 clear observations per
parcel-year in W04 to ~13 in W19 (L5 retired, L7 SLC-off). Within parcel, the window
probability tracks it: +0.052 per log-observation for PETT-`PERENNIAL` parcels and −0.022 for
PETT-`ANNUAL` ones on the selected arm (p < 1e-6 both). That is probability compression toward
the base rate, and it looks exactly like a conversion. It explains 0–29 % of the drift
depending on the arm — a mechanism, not the whole mechanism. **And it is monotone in how few
statics an arm carries**, the third independent appearance of that pattern.

### 9.3 Corrections to the plan's own claims

* **§4.6's design effect was optimistic.** The measured `deff` of the stratified draw is
  **2.36**, not 1.4. ~2,100 per group becomes ~3,550; ≥5,000 still covers a 2 pp differential,
  but 1 pp would need ~12,500 per group.
* **§4.1's leakage worry is not detectable.** Retraining at `buffer_m = 3000` gives CV
  **0.580 ± 0.024** against 0.5805 ± 0.002 at 1,500 m — doubling the dead-zone costs nothing
  in the mean. It bounds one leakage channel; the 5 km decorrelation range still stands.
* **§6.2's "worth an explicit hunt" found something.** The bridge `.dta` carries the
  cadastre's own titling status and a snapshot date (`estado`, `fech_tran` ≈ 2011-12). With BD
  SSET's declaration-time `ESTADO en RRPP` that is **two dated tenure observations for
  1,780,580 parcels**, and **8.6 % move NO INSCRITO → REGISTERED** (Piura 25.5 %). A
  two-period difference-in-differences is therefore available without new data.
* **§6.3 cannot be closed with MapBiomas.** Over Piura it only ever assigns `MOSAIC` and
  `ANNUAL`. The declaration-based bound is 2.79 % of the eligible pool — bigger than the
  effect — and it is a *lower* bound.
* **§6.4 is now a number:** PERENNIAL is 57.3 % export basket / 19.9 % mixed / 22.8 % domestic
  by weighted parcel count, ranging from Tacna 0.92 to Lambayeque 0.13.

### 9.4 What to do next, cheapest first

1. **Equalise observation density across windows** before assembling features — subsample each
   year to a fixed observation count or a fixed DOY skeleton — and re-run T3. This attacks the
   one mechanism that is measured, needs no new labels or extraction, and runs off the panel
   store already on disk. If the control flattens, M2 is recoverable and T4 becomes fundable.
2. **Re-run T2 with that fix**, since the same degradation plausibly drives part of the cohort
   spread.
3. **Switch the design to the two-period tenure event** (§9.3). A within-parcel registration
   change differences out drift common to both tenure groups, which is exactly the artefact
   that broke M2 — and it answers the reverse-causality objection the plan flagged as
   unresolvable.
4. **Only then** revisit T1's differential false-positive rate: it is a property of the
   classifier at the label year, and any of the above changes it.
