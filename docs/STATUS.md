# Status

**Last updated 2026-09-01.** Read this before starting work. Numbers and their evidence are in
[`RESULTS.md`](RESULTS.md); this page says what is done, what is closed, and what to do next.

`uv run cc reproduce` re-derives every published headline number from the committed data in
about 90 s. Run it after any change that could move one.

---

## The question

Has farmland in Peru shifted from domestic annual crops (rice, maize, beans) to export
perennials (mango, lime, coffee, banana)? And does giving a farmer secure legal title make that
shift more likely?

The data allows the question because Peru's land-titling programme recorded, for ~1 M parcels,
the crop growing there **and** the parcel boundary — but only once, mostly 1997–2006. Satellite
imagery exists every year from 1996.

---

## What works

* **The Sentinel-2 endpoint classifier**, the model of record: 865 photo-interpreted parcels,
  LightGBM, 3 classes (`t3w`), `--climate temp`. **0.762 CV, 0.724 held out of a whole
  department** (14 departments, SD 0.112), and **0.774 macro-F1 / 0.789 accuracy /
  `PERENNIAL` F1 0.780** on the 161-parcel locked test. The test agrees with CV, so the
  cross-validated estimate was not inflated — but it is an in-department number; 0.724 remains
  the expectation for an unseen department. §8.2, §8.8b, §8.9.
* **A single-year 3-class land-state classifier on Landsat.** Piura locked test 0.681
  macro-F1; national CV 0.628, and three times steadier across folds.
* **The declared → observed transition matrix**, with no classifier in it: of parcels declared
  `ANNUAL` in 1996–2006, 2.9 % [0, 5.9] read as perennial in 2019+, and 57.9 % read as farmable
  ground not currently cropped. National average, not a Piura finding (86.7 % of Piura's
  declared-annual parcels still read `ANNUAL`). §8.2b.
* **The PETT → CENAGRO 2012 paired comparison**, the only before/after between two
  *declarations*: on 8,669 Piura parcels, perennial goes 28.2 % → 40.7 % of parcels (+12.5 pp)
  and 51.4 % → 66.0 % of cadastral area (+14.6 pp). Piura only, farmer-level link, non-random
  subsample. §8.5. Nationally, +9.9 pp of parcels across 14 departments. §8.6.
* **The national label build** — 14 departments, 946,872 linked polygons, label years 1997–2006.
* **The tenure difference-in-differences**, the project's only causal estimate: a bounded null,
  −0.0011 [−0.0126, +0.0104].
* **The evaluation protocol** — CV / LODO / LOYO / LODYO. It is what caught everything below.

---

## What is closed, and must not be reopened

| route | why closed | where |
|---|---|---|
| 12-class "which crop?" | annual crops are not separable at 30 m — maize 0.23 F1, beans 0.11 | §1 |
| Per-parcel annual trajectories (Piura **and** national) | flicker 0.43–0.98 against a 0.15 criterion; no thin years nationally, so not a coverage problem. A ~0.55–0.59 classifier cannot support a 25-year series | §3, §4.4 |
| "Try another architecture" **on the Landsat store** | LTAE fails harder on every axis (flicker 0.98, LODO 0.442, LOYO 0.478) | §3.1, §4.3 |
| On the S2 store the answer is **split**, not reversed | LTAE wins CV in 8 of 8 arms and loses LODO in 6 of 6 targets: its extra CV skill does not leave the training departments. Measured on 14 held-out departments; the earlier "LTAE wins LODO" reading had 4 | §8.2, §8.2c |
| Collapsing the label space to 2 classes | macro-F1 rises to 0.769 and the floor rises further (0.467 vs 0.171). Normalised it is the lowest-skill arm in the study, for +0.004 `PERENNIAL` F1. Take a binary output by summing probabilities instead | §8.2c |
| The 5-year window pivot | the control pool drifts 7× further than the signal, the other way, on every arm | §5 |
| Recalibration / quantile alignment / density matching | measured: moved the target metric by 0.0001, −0.1023, and about a third of the artefact | §6.3 |
| **Admitting OLI** (twice) | the sensor difference is cover-type dependent (0.025 NDVI between classes) — no global linear map removes it. Do not propose refitting the coefficients; reopening needs a cleaner paired sample | §6.4 |
| **A bigger tenure DiD** | the population is exhausted — 6,559 treated parcels is all of Peru. The binding limit is 7× amplification of a 2.5-year placebo. Reopening needs a *dated registration event*, not two snapshots | §7 |

---

## What to do next

The labelling is effectively done and the models are trained: 1,012 of 1,112 labellings
returned, **865 usable**, all 14 departments over the G3 floor, every arm fitted. Full plan and
gates: [`s2_labelling/plan.md`](s2_labelling/plan.md).

1. **The overlap shard — the one remaining blocker.** G1 (κ ≥ 0.75) is still unmeasured: one
   annotator, and the overlap shard was not in the return. With one labeller κ becomes
   intra-rater — re-label 100 parcels after a gap. Until it exists the campaign has no measure
   of its own label noise and every figure in §8.2 inherits that. On the record: the
   annotator's median time per parcel was 2–4 seconds against a ~2 min budget.
   This is also the **only** good use of 100 more labellings: the learning curve has flattened
   (at 354 labels the pilot's +48 % moved LightGBM +0.050 on `t4`; at 865 the same pilot's
   +14 % moves it +0.021).
2. **Carry LightGBM forward, not LTAE.** LTAE wins CV in 8 of 8 arms and loses LODO in 4 of 4
   targets, dropping 1.4–3.4× more than LightGBM when the department changes. §8.2.
3. **G4 needs restating before it can be judged.** As written it compares against existing
   Landsat predictions on the same parcels, which cover 22 of 1,112. The substitute — the
   Landsat model scored on S2 features — is a cross-sensor lower bound, for the reason §6.4
   closed the OLI route. Either extract genuine Landsat features for these parcels, or let G4
   become the S2 model's own held-out number against a stated floor.
4. **The CENAGRO route is open and has gone national.** A paired before/after with no
   classifier in it: 63,766 like-for-like parcels, +9.9 pp perennial post-stratified, split by
   tenure (§8.6). It cannot reach a parcel key — the link is farmer-level, so
   post-stratification is not optional. Before extending it, run
   `cenagro_shift.token_audit()`: an unmapped crop-name tail passed its budget check twice
   while one name in it moved the headline by 10 pp (Piura) and 0.6 pp (national).
5. **A climate covariate is adopted: `--climate temp`.** The first feature that helps LODO as
   well as CV — LightGBM LODO 0.539 → 0.568 (`t4`) and 0.697 → 0.724 (`t3w`), against a
   coordinate control worth +0.014 / +0.002. Honest effect size: +0.027 LODO at 9 of 14
   departments, p = 0.17. Three limits: `--climate both` was the earlier recommendation and is
   superseded (rainfall is worth −0.001 on `t3w`); adopt for LightGBM only, since LTAE gains
   the same from the coordinate control; and the static normals must never enter the panel —
   use `parcel_rainfall_annual` across years. §8.8, §8.8b.

---

## Standing constraints on new work

The method rules are in [`CLAUDE.md`](../CLAUDE.md) (what to have loaded before touching
anything) and [`LESSONS.md`](LESSONS.md) (why each was learned). What follows is only what is
true of *this project right now*:

* **The national locked test is unspent.** Keep it that way until an estimand passes its gate.
* **The S2 locked test is spent** (2026-09-01, §8.9), scored once on the two arms §8.8b had
  already selected. `--eval-test` must not be used again; any future model change is a CV/LODO
  decision.
* **The Piura locked test has been spent twice** (2026-08-05), and the 12-class one is spent
  (macro-F1 0.427). Neither is a clean held-out estimate any more.
* **`--drop-features meta,location` is not optional** for anything applied across years, and
  `advanced lodo` must use the same ablation as the arm it is judging.
* Every accuracy number from strands 1–7 was measured in ~1997–2006. **Do not quote one as if
  it applied to 2019–2023.** The only figures measured on 2019+ imagery are §8.2's, and they
  rest on 865 labels from one annotator with κ unmeasured.
* **Run a feasibility check before funding any Earth Engine extraction**: required n from
  measured variance against available n from the archive. It takes minutes.

## Where things live

| | |
|---|---|
| [`RESULTS.md`](RESULTS.md) | every strand, its verdict, the numbers |
| [`LESSONS.md`](LESSONS.md) | findings that generalise beyond this project |
| [`DATA.md`](DATA.md) | raw datasets, linkage chains, the four silent traps |
| [`PIPELINE.md`](PIPELINE.md) | module reference, CLI table, cookbook |
| [`cenagro_columns.md`](cenagro_columns.md) | the 2012 census extract, column by column |
| [`s2_labelling/`](s2_labelling/plan.md) | the live campaign and its codebook |
| [`howto/`](howto/01_setup.md) | task-shaped guides: set up, reproduce, train, predict, label |
| `../reports/peru_report.tex` | the written-up narrative, with its PDF |
