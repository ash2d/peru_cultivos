# Status

**Last updated 2026-09-01.** Read this before starting work.
`uv run cc reproduce` re-derives every published headline number from the committed data in
~90 s; run it after any change that could move one. Numbers and their evidence live in
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
* ⭐ **The S2 endpoint classifier, verified on 2019+ imagery, and now on a locked test** — the
  project's first accuracy measured in the period the research question is about. 865
  photo-interpreted parcels, LightGBM with `--climate temp`, 3-class (`t3w`): **0.762 CV,
  0.724 held out of a whole department (14 depts, SD 0.112), and 🔓 0.774 macro-F1 / 0.789
  accuracy / `PERENNIAL` F1 0.780 on the 161-parcel locked test** (`RESULTS.md` §8.2, §8.8b,
  §8.9). The 4-class reading is 0.695 CV / 0.568 LODO. **The test agrees with CV**, so the
  cross-validated estimate was not inflated — but it is an in-department number; 0.724 remains
  the expectation for an unseen department.
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
4. **~~Decide whether to spend the locked test.~~** ✅ **DONE 2026-09-01 (§8.9).** Spent once, on
   the arm §8.8b had already selected — LightGBM / `t3w`+pilot / `--climate temp`:
   **0.789 accuracy, 0.774 macro-F1 [95 % CI 0.697–0.845], `PERENNIAL` F1 0.780** on 161 parcels
   across 14 departments, against a CV estimate of 0.762. The `both` arm was scored alongside it
   (0.764) and is indistinguishable — 8 disagreements in 161, McNemar p = 1.000.
   ⛔ **It is now spent. Score nothing else on it.**
   ⚠️ Read 0.774 as the **in-department** number: every department is on both sides of the split,
   so it tracks CV, not LODO. The out-of-department expectation stays **0.724 ± 0.112**.
   ⚠️ It was spent **before** G1, so it inherits an unmeasured annotator-noise floor.
5. **⭐ The CENAGRO route is open and has now gone national.** A paired before/after with
   **no classifier in it**, and the only such thing the project has. `allperu cenagro` is the
   Piura-only version (§8.5); `allperu cenagro-extract` → `cenagro-link` → `cenagro-shift` runs
   it over the **25-department extract** on all 14 linkable departments — 63,766 like-for-like
   parcels, **+9.9 pp perennial post-stratified**, split by tenure (§8.6). What it still cannot
   do is reach a **parcel key**: the link is farmer-level, so post-stratification is not
   optional. ⚠️ **Before extending it, run `cenagro_shift.token_audit()`** — an unmapped census
   crop-token tail passed its budget check **twice** while one token in it moved the headline by
   10 pp (Piura) and 0.6 pp (national).
6. **✅ A climate covariate is IN — `--climate temp`, and it is the first feature that helps
   LODO** (§8.8, amended by §8.8b). WorldClim normals as extra inputs on the endpoint arm, three
   model classes × four feature sets + a lat/lon control, run on **both** ends of the `t4`/`t3w`
   codebook bracket. **LightGBM `temp`: LODO 0.539 → 0.568 (`t4`) and 0.697 → 0.724 (`t3w`),
   `PERENNIAL` LODO F1 +0.057 / +0.058** — the same gain at both ends, against a raw-coordinate
   control that buys +0.014 / +0.002.
   ⚠️ **`--climate both` was the §8.8 recommendation and is NOT.** Its `t4` case rested on
   12/14 departments at p = 0.004; on `t3w` that is **7/14 at p = 0.345**, and the column that
   costs it the replication is `precip_mm_yr`, worth **−0.001** LODO there. Honest effect size
   for `temp`: **+0.027 LODO at 9/14 departments, p = 0.17.**
   ⚠️ **Adopt for LightGBM only.** LTAE gains the same amount as the coordinate control at
   *both* targets, so for LTAE it is department memorisation by another route (§8.2 again).
   ⛔ **Normals must not enter the panel** — time-invariant, the §4.4/§5 family; use
   `parcel_rainfall_annual` across years. And climate alone recovers the department at 0.676
   accuracy (prior 0.091), so the memorisation channel is open, merely outweighed.
   `allperu s2-train {prep,fit,lodo,report} --target {t4,t3w} --climate {none,temp,rain,both,latlon}`.

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
* 🔓 ⚠️ **The S2 locked test is SPENT** (2026-09-01, §8.9) — 201 parcels, 161 usable, scored
  once on the two arms §8.8b had already selected (`temp` 0.774, `both` 0.764). It is no longer
  a clean held-out set: `allperu s2-train fit --eval-test` is the flag that spends it and it
  **must not be used again**. Any future model change is a CV/LODO decision.
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
| [`howto/06_label_more_parcels.md`](howto/06_label_more_parcels.md) | the label-more-parcels-and-retrain loop, written for a non-programmer |
| `../reports/peru_report.tex` | the written-up narrative, with its PDF; earlier snapshots in [`../reports/archive/`](../reports/archive/REPORT.md) |
