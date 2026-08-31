# Status

**Last updated 2026-08-28.** Read this before starting work. Numbers and their evidence live in
[`RESULTS.md`](RESULTS.md) — this page says only what is done, what is closed, and what to do.

---

## The question

Has farmland in Peru shifted from domestic annual crops (rice, maize, beans) to export perennial
crops (mango, lime, coffee, banana)? And does giving a farmer secure legal title make that shift
more likely?

The data allows it because Peru's land-titling programme recorded, for ~1 M parcels, the crop
growing there **and** the parcel boundary — but only **once**, mostly 1997–2006. There is no
second visit. Satellite imagery exists every year from 1996.

---

## What works

* **A single-year 3-class land-state classifier** (`PERENNIAL` / `ANNUAL` / `PASTURE_FALLOW`).
  Piura locked test **0.681 macro-F1**; national CV **0.628**, and 3× steadier across folds.
* ⭐ **The S2 endpoint classifier, verified on 2019+ imagery** — the project's first accuracy
  measured in the period the research question is about. 865 photo-interpreted parcels,
  LightGBM, **0.672 macro-F1 on spatially-blocked CV and 0.539 held out of a whole department**
  (4-class), **0.747 / 0.697** on the 3-class reading. `RESULTS.md` §8.2.
* ⭐ **The declared → observed transition matrix**, with no classifier in it: of parcels
  declared `ANNUAL` in 1996–2006, **2.9 % [0, 5.9] read as perennial in 2019+**; **57.9 %**
  read as farmable ground not currently cropped. ⚠️ National average — **it is not a Piura
  finding** (86.7 % of Piura's declared-annual parcels still read `ANNUAL`). `RESULTS.md` §8.2b.
* ⭐ **The PETT → CENAGRO 2012 paired comparison** — the project's only before/after between
  two *declarations*, no satellite and no classifier: on 8,669 Piura parcels, perennial goes
  **28.2 % → 40.7 % of parcels (+12.5 pp)** and **51.4 % → 66.0 % of cadastral area
  (+14.6 pp)**, with conversion concentrated on the larger parcels. ⚠️ Piura only, farmer-level
  link, non-random subsample. `RESULTS.md` §8.5.
* **The national label build** — 14 departments, 946,872 linked polygons, label years spread
  over 1997–2006.
* **The tenure difference-in-differences**, which returned the project's only actual estimate:
  a **bounded null**, headline **−0.0011 [−0.0126, +0.0104]**.
* **The evaluation protocol** — CV / LODO / LOYO / LODYO. This is what caught the rest.

---

## ⛔ What is closed, and must not be reopened

| route | why closed | where |
|---|---|---|
| 12-class "which crop?" | annual crops are not separable at 30 m — maize 0.23 F1, beans 0.11 | `RESULTS.md` §1 |
| Per-parcel annual trajectories (Piura **and** national) | flicker 0.43–0.98 against a 0.15 criterion; **no thin years nationally**, so not coverage. A ~0.55–0.59 classifier cannot support a 25-year per-parcel series | §3, §4.4 |
| "Try another architecture" **on the Landsat store** | LTAE fails *harder* on every axis (flicker 0.98, LODO 0.442, LOYO 0.478) | §3.1, §4.3 |
| ⚠️ …and on the S2 store the answer is **split**, not reversed | LTAE wins **CV in 8 of 8 arms** and loses **LODO in 6 of 6 targets**. Its extra CV skill does not leave the training departments. Measured on 14 held-out departments; the earlier "LTAE wins LODO" reading had 4 | §8.2, §8.2c |
| ⛔ **Collapsing the label space to 2 classes** (`t2`, perennial vs not) | macro-F1 rises to 0.769 and the **floor rises further** (0.467 vs 0.171 at 4 classes). Normalised, `t2` is the lowest-skill arm in the study — LODO skill **0.180**. It gains **+0.004** `PERENNIAL` F1 over the 3-class model. Take a binary output by summing probabilities instead | §8.2c |
| The 5-year window pivot | the control pool drifts 7× further than the signal, the other way, on **every** arm | §5 |
| Recalibration / quantile alignment / density matching | measured: moved the target metric by 0.0001, −0.1023, and ~⅓ of the artefact respectively | §6.3 |
| **Admitting OLI** (twice) | the sensor difference is **cover-type dependent** (0.025 NDVI between classes) — no global linear map can remove it. **Do not propose refitting the coefficients; it is done.** Reopening needs a cleaner paired sample, not better fitting | §6.4 |
| **A bigger tenure DiD** | the population is exhausted — **6,559 treated parcels is all of Peru**. The binding limit is 7× amplification of a 2.5-year placebo. Reopening needs a *dated registration event*, not two snapshots. Do not relax R4, widen the band, or drop the at-risk restriction | §7 |

---

## ⭐ What to do next

**The labelling is effectively done and the models are trained.** 1,012 of 1,112 labellings
returned (2026-08-27/28), **865 usable**, all 14 departments over the G3 floor. Every arm —
3 models × 4 label targets × 2 training pools, plus leave-one-department-out on all 12 — has
been fitted. Numbers: `RESULTS.md` §8.1–8.2b.

**Only the 100-parcel overlap shard is outstanding.** Full plan and gates:
**[`s2_labelling/plan.md`](s2_labelling/plan.md)**.

1. **▶ THE OVERLAP SHARD — the one remaining blocker, and now the *only* thing worth labelling.**
   G1 (κ_called ≥ 0.75) is **still unmeasured**: one annotator, and the overlap shard was not in
   the return. With one labeller κ becomes **intra-rater — re-label 100 parcels after a gap.**
   Until it exists the campaign has no measure of its own label noise and every figure in §8.2
   inherits that. Related and on the record: the annotator's **median time per parcel is
   2–4 seconds** against a ~2 min budget.
   ⭐ **And it is now the *only* good use of 100 more labellings**: the learning curve has
   flattened — at 354 labels the pilot's +48 % moved LightGBM +0.050 on `t4`; at 865 the same
   pilot's +14 % moves it +0.021 and moves LTAE −0.001.
2. **⚠️ Carry LightGBM forward, not LTAE.** This reverses the previous entry here. LTAE wins CV
   in **8 of 8** arms and loses LODO in **4 of 4** targets, dropping 1.4–3.4× more than LightGBM
   when the department changes. The earlier "LTAE wins LODO" reading was taken on **4**
   departments; there are now **14**. §8.2.
3. **G4 still needs restating before it can be adjudicated**, unchanged: as written it compares
   against "existing Landsat predictions on the same parcels", which cover **22 of 1,112**. The
   substitute (the Landsat booster scored on S2 features) is a **cross-sensor lower bound**, for
   exactly the reason §6.4 closed the OLI route. Either (a) extract genuine Landsat features for
   these parcels, or (b) let G4 become the S2 model's own held-out number against a stated floor.
4. **Decide whether to spend the locked test.** 201 parcels, **161 now labelled**, still
   **UNSPENT** — `allperu s2-train` exposes no `--eval-test` and nothing in §8 reads it. Spending
   it buys the project's only clean held-out endpoint number at roughly ±7 pp; it can be spent
   **once**, on **one** pre-declared arm. Do G1 first, so the number is not inheriting an
   unmeasured noise floor.
5. **⭐ The CENAGRO route is open and has now gone national.** A paired before/after with
   **no classifier in it**, and the only such thing the project has. `allperu cenagro` is the
   Piura-only version (§8.5); `allperu cenagro-extract` → `cenagro-link` → `cenagro-shift` runs
   it over the **25-department extract** on all 14 linkable departments — 63,766 like-for-like
   parcels, **+9.9 pp perennial post-stratified**, split by tenure (§8.6). What it still cannot
   do is reach a **parcel key**: the link is farmer-level, so post-stratification is not
   optional. ⚠️ **Before extending it, run `cenagro_shift.token_audit()`** — an unmapped census
   crop-token tail passed its budget check **twice** while one token in it moved the headline by
   10 pp (Piura) and 0.6 pp (national).
6. **Climate covariates are built and ready to add** (`DATA.md` §7.4): per-parcel mean
   temperature and rainfall for all 726,808 national parcels. Not yet in any model. **Use
   `parcel_rainfall_annual` (year-resolved) for anything across years and the normals only for a
   single-year model** — the normals are time-invariant, which is the family §4.4/§5 measured
   manufacturing stability. Evaluate on LODO as well as CV: a 1 km climate surface is a smooth
   function of location, hence a `centroid_lat` proxy.

---

## Standing constraints on any new work

**The method rules live in two places and are not repeated here.**
[`CLAUDE.md`](../CLAUDE.md) carries the ones an agent must have loaded before touching anything
— never select on CV alone, report CV / LODO / LOYO / LODYO, the majority-class floor beside
every macro-F1, the same lexicon on both sides of a before/after, an OOD mean over 4 units is 4
numbers, and the libomp / `sample_weight` / count-the-rows gotchas.
[`LESSONS.md`](LESSONS.md) carries why each one was learned.

What follows is only what is **true of this project right now**, and would be wrong to infer
from a general rule:

* **The national locked test is UNSPENT.** Keep it that way until an estimand passes its gate.
* **The S2 locked test is UNSPENT** — 201 parcels, 161 labelled. `allperu s2-train` deliberately
  exposes no `--eval-test`. It can be spent **once**, on **one** pre-declared arm.
* ⚠️ **The Piura locked test is SPENT TWICE** (2026-08-05). It is no longer a clean held-out
  estimate for any selection.
* ⚠️ **The 12-class locked test is spent** (macro-F1 0.427).
* **`--drop-features meta,location` is not optional** for anything applied across years, and
  `allperu lodo` must be run with the same ablation as the arm it is judging.
* Every accuracy number from strands 1–7 is measured in ~1997–2006. **Do not quote one as if it
  applied to 2019–2023.** The **only** figures measured on 2019+ imagery are `RESULTS.md` §8.2's,
  and they rest on **865 labels from one annotator with κ unmeasured**.
* **Run a feasibility check before funding any GEE extraction** — `allperu tenure-ceiling` style:
  required n from measured variance against available n from the archive. It takes minutes.

## Where things live

| | |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every strand, its verdict, the numbers |
| [`LESSONS.md`](LESSONS.md) | findings that generalise beyond this project |
| [`DATA.md`](DATA.md) | raw datasets, linkage chains, the four silent traps |
| [`PIPELINE.md`](PIPELINE.md) | module reference, CLI table, cookbook |
| [`cenagro_columns.md`](cenagro_columns.md) | the 2012 census extract, column by column |
| [`s2_labelling/`](s2_labelling/) | the live campaign + frozen codebook |
| [`../reports/`](../reports/REPORT.md) | the written-up narratives — `REPORT.md` is current, the PDF and `.tex` are superseded snapshots |
